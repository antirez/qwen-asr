#!/usr/bin/env python3
"""
ASR regression harness for qwen_asr.

Usage examples:
  # Generate missing references next to each WAV (samples/**/*.txt)
  ./asr_regression.py --generate-missing

  # Refresh all references
  ./asr_regression.py --refresh-refs

  # Run regression checks against existing references
  ./asr_regression.py

The harness always prints two distances per sample:
  1) exact character-level distance (case/punctuation preserved)
  2) normalized character-level distance
     (punctuation -> spaces, lowercase, collapsed whitespace)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

# ---- ANSI colors (auto-disabled when stdout is not a tty) ----

_USE_COLOR = hasattr(sys.stdout, "isatty") and sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

def _sgr(code: str) -> str:
    return f"\033[{code}m" if _USE_COLOR else ""

C_RESET   = _sgr("0")
C_BOLD    = _sgr("1")
C_DIM     = _sgr("2")
C_RED     = _sgr("31")
C_GREEN   = _sgr("32")
C_YELLOW  = _sgr("33")
C_CYAN    = _sgr("36")
C_BRED    = _sgr("1;31")
C_BGREEN  = _sgr("1;32")
C_BYELLOW = _sgr("1;33")
C_BCYAN   = _sgr("1;36")
C_BWHITE  = _sgr("1;37")

SEGMENTED_SECONDS = "20"
STREAM_CACHE_DEFAULT_MODEL_DIR = "qwen3-asr-0.6b"
STREAM_CACHE_DEFAULT_SAMPLES = (
    "night_of_the_living_dead_1968/10s_back_down_the_road.wav",
    "night_of_the_living_dead_1968/45s_dont_be_afraid_of_me.wav",
)


def levenshtein(seq_a: Sequence[str], seq_b: Sequence[str]) -> int:
    """Memory-efficient Levenshtein distance."""
    if len(seq_a) < len(seq_b):
        seq_a, seq_b = seq_b, seq_a
    if not seq_b:
        return len(seq_a)

    prev = list(range(len(seq_b) + 1))
    for i, a in enumerate(seq_a, 1):
        cur = [i]
        for j, b in enumerate(seq_b, 1):
            cost = 0 if a == b else 1
            cur.append(min(
                prev[j] + 1,       # deletion
                cur[j - 1] + 1,    # insertion
                prev[j - 1] + cost # substitution
            ))
        prev = cur
    return prev[-1]


def normalize_text(text: str) -> str:
    out = []
    for ch in text:
        if ch.isalnum() or ch.isspace():
            out.append(ch.lower())
        else:
            out.append(" ")
    return " ".join("".join(out).split())


def find_wavs(samples_root: Path) -> List[Path]:
    return sorted(samples_root.rglob("*.wav"))


def ref_for_wav(wav: Path) -> Path:
    return wav.with_suffix(".txt")


def run_once(cmd: Sequence[str], timeout_s: int, show_output: bool = False) -> Tuple[int, str, str]:
    if not show_output:
        cp = subprocess.run(
            list(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        return cp.returncode, cp.stdout.strip(), cp.stderr.strip()

    # Stream stdout to stderr in real-time so the user sees tokens as they appear,
    # while still capturing them for comparison.
    proc = subprocess.Popen(
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    chunks: list[bytes] = []
    try:
        while True:
            b = proc.stdout.read(1)
            if not b:
                break
            chunks.append(b)
            sys.stderr.buffer.write(b)
            sys.stderr.buffer.flush()
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    stdout_text = b"".join(chunks).decode("utf-8", errors="replace").strip()
    return proc.returncode, stdout_text, ""


def fmt_time(secs: float) -> str:
    """Format seconds as human-readable string."""
    if secs < 60:
        return f"{secs:.1f}s"
    m, s = divmod(int(secs), 60)
    return f"{m}m{s:02d}s"


def transcribe(
    binary: Path,
    model_dir: Path,
    wav: Path,
    timeout_s: int,
    extra_args: Sequence[str],
    verbose: bool = False,
    show_output: bool = False,
) -> str:
    # Default reference/check profile:
    # 1) try full-context decode (-S 0) for best quality when it works well
    # 2) fallback to explicit segmented decode if output collapses to empty
    #
    # When show_output is True, omit --silent so tokens stream to stdout
    # (which run_once will tee to stderr for the user to see).
    base = [str(binary), "-d", str(model_dir), "-i", str(wav)]
    if not show_output:
        base.append("--silent")

    cmd_full = base + ["-S", "0"] + list(extra_args)
    t0 = time.monotonic()
    rc, out, err = run_once(cmd_full, timeout_s, show_output=show_output)
    elapsed = time.monotonic() - t0
    if rc != 0:
        msg = err or f"exit code {rc}"
        raise RuntimeError(f"transcription failed for {wav}: {msg}")
    if out:
        if verbose:
            print(f"       transcribed in {fmt_time(elapsed)} (full-context)")
        return out

    # If caller explicitly provided -S/--segment-overlap in extra args, keep behavior strict.
    explicit_seg = any(a in ("-S", "--segment-overlap") for a in extra_args)
    if explicit_seg:
        return out

    if verbose:
        print(f"       full-context returned empty after {fmt_time(elapsed)}, trying segmented fallback...")
    cmd_fallback = base + ["-S", SEGMENTED_SECONDS] + list(extra_args)
    t0 = time.monotonic()
    rc2, out2, err2 = run_once(cmd_fallback, timeout_s, show_output=show_output)
    elapsed2 = time.monotonic() - t0
    if rc2 != 0:
        msg = err2 or f"exit code {rc2}"
        raise RuntimeError(f"fallback transcription failed for {wav}: {msg}")
    if verbose:
        print(f"       fallback completed in {fmt_time(elapsed2)}")
    return out2


def transcribe_segmented(
    binary: Path,
    model_dir: Path,
    wav: Path,
    timeout_s: int,
    extra_args: Sequence[str],
    past_text_conditioning: bool,
    show_output: bool = False,
) -> str:
    cmd = [str(binary), "-d", str(model_dir), "-i", str(wav), "-S", SEGMENTED_SECONDS]
    if not show_output:
        cmd.append("--silent")
    if past_text_conditioning:
        cmd += ["--past-text", "yes"]
    else:
        cmd += ["--past-text", "no"]
    cmd += list(extra_args)

    rc, out, err = run_once(cmd, timeout_s, show_output=show_output)
    if rc != 0:
        mode = "with past-text conditioning" if past_text_conditioning else "without past-text conditioning"
        msg = err or f"exit code {rc}"
        raise RuntimeError(f"segmented transcription failed for {wav} ({mode}): {msg}")
    return out


def run_segment_conditioning_regression(
    samples_root: Path,
    binary: Path,
    model_dir: Path,
    timeout_s: int,
    extra_args: Sequence[str],
    min_ratio: float,
    show_output: bool = False,
) -> int:
    target = samples_root / "night_of_the_living_dead_1968" / "89s_ill_come_back_down_as_soon_as.wav"
    if not target.exists():
        print(f"{C_BYELLOW}[SKIP seg-check]{C_RESET} missing sample: {target}")
        return 0

    # Keep this check independent from CLI default by forcing both modes explicitly.
    if any(a in ("-S", "--segment-overlap", "--stream", "--stdin",
                 "--past-text") for a in extra_args):
        print(f"{C_BYELLOW}[SKIP seg-check]{C_RESET} explicit segmentation/stream args provided")
        return 0

    print(f"{C_BCYAN}[....  seg-check]{C_RESET} transcribing with past-text conditioning...", flush=True)
    t0 = time.monotonic()
    with_past = transcribe_segmented(
        binary=binary,
        model_dir=model_dir,
        wav=target,
        timeout_s=timeout_s,
        extra_args=extra_args,
        past_text_conditioning=True,
        show_output=show_output,
    )
    t1 = time.monotonic()
    print(f"       done in {C_DIM}{fmt_time(t1 - t0)}{C_RESET}, transcribing without conditioning...", flush=True)
    no_past = transcribe_segmented(
        binary=binary,
        model_dir=model_dir,
        wav=target,
        timeout_s=timeout_s,
        extra_args=extra_args,
        past_text_conditioning=False,
        show_output=show_output,
    )
    t2 = time.monotonic()
    print(f"       done in {C_DIM}{fmt_time(t2 - t1)}{C_RESET}", flush=True)

    with_past_words = len(normalize_text(with_past).split())
    no_past_words = len(normalize_text(no_past).split())
    baseline = max(1, no_past_words)
    ratio = with_past_words / baseline

    # Ignore trivial edge cases with very short outputs.
    if baseline < 80:
        print(
            f"{C_BYELLOW}[SKIP seg-check]{C_RESET} baseline too short ({no_past_words} words), "
            "cannot evaluate collapse robustly"
        )
        return 0

    if ratio < min_ratio:
        print(
            f"{C_BRED}[FAIL seg-check]{C_RESET} {C_BWHITE}{target.name}{C_RESET} | "
            f"words with={with_past_words}, without={no_past_words}, "
            f"{C_RED}ratio={ratio:.3f} < {min_ratio:.3f}{C_RESET}"
        )
        return 1

    print(
        f"{C_BGREEN}[ OK  seg-check]{C_RESET} {C_BWHITE}{target.name}{C_RESET} | "
        f"words with={with_past_words}, without={no_past_words}, "
        f"{C_GREEN}ratio={ratio:.3f}{C_RESET}"
    )
    return 0


TIMELINE_FORMATS = ("srt", "vtt", "jsonl", "txt")


def _wav_duration_s(wav: Path) -> float:
    with wave.open(str(wav), "rb") as w:
        return w.getnframes() / float(w.getframerate() or 1)


def _parse_timeline(path: Path, fmt: str) -> List[Tuple[float, float, str]]:
    """Parse a timeline file into (start, end, text) tuples."""
    entries: List[Tuple[float, float, str]] = []
    text = path.read_text(encoding="utf-8")
    if fmt == "jsonl":
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            entries.append((float(obj["start"]), float(obj["end"]), str(obj["text"])))
    elif fmt == "txt":
        pat = re.compile(
            r"^\[(\d{2}):(\d{2}):(\d{2}\.\d{3}) - (\d{2}):(\d{2}):(\d{2}\.\d{3})\] (.*)$"
        )
        for line in text.splitlines():
            m = pat.match(line.strip())
            if not m:
                continue
            h1, m1, s1, h2, m2, s2, body = m.groups()
            a = int(h1) * 3600 + int(m1) * 60 + float(s1)
            b = int(h2) * 3600 + int(m2) * 60 + float(s2)
            entries.append((a, b, body))
    else:  # srt / vtt blocks
        pat = re.compile(
            r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
        )
        for block in re.split(r"\n\s*\n", text.strip()):
            lines = [l for l in block.strip().splitlines() if l.strip()]
            idx = next((i for i, l in enumerate(lines) if pat.search(l)), None)
            if idx is None:
                continue
            m = pat.search(lines[idx])
            h1, m1, s1, ms1, h2, m2, s2, ms2 = m.groups()
            a = int(h1) * 3600 + int(m1) * 60 + int(s1) + int(ms1) / 1000.0
            b = int(h2) * 3600 + int(m2) * 60 + int(s2) + int(ms2) / 1000.0
            body = " ".join(lines[idx + 1:]).strip()
            entries.append((a, b, body))
    return entries


def _timeline_sanity_errors(entries: List[Tuple[float, float, str]], dur_s: float) -> List[str]:
    """Basic structural invariants every timeline file must satisfy."""
    errors: List[str] = []
    if not entries:
        return ["no entries parsed"]
    prev_end = -1.0
    for i, (a, b, body) in enumerate(entries):
        if b < a:
            errors.append(f"entry {i + 1}: end {b:.3f} < start {a:.3f}")
        if a < -0.01:
            errors.append(f"entry {i + 1}: start {a:.3f} < 0")
        if b > dur_s + 0.2:
            errors.append(f"entry {i + 1}: end {b:.3f} > duration {dur_s:.3f} + 0.2")
        if a < prev_end - 0.02:
            errors.append(f"entry {i + 1}: start {a:.3f} before previous end {prev_end:.3f}")
        if not body.strip():
            errors.append(f"entry {i + 1}: empty text")
        prev_end = max(prev_end, b)
    if entries[-1][1] < dur_s - 5.0:
        errors.append(
            f"last entry ends at {entries[-1][1]:.3f}, expected >= duration - 5s ({dur_s - 5.0:.3f})"
        )
    return errors


def run_timeline_regression(
    samples_root: Path,
    binary: Path,
    model_dir: Path,
    timeout_s: int,
    extra_args: Sequence[str],
    show_output: bool = False,
) -> int:
    long_target = samples_root / "night_of_the_living_dead_1968" / "45s_dont_be_afraid_of_me.wav"
    short_target = samples_root / "jfk.wav"

    if any(a in ("--timeline-file", "--timeline-format", "--stream", "--stdin",
                 "-S", "--skip-silence") for a in extra_args):
        print(f"{C_BYELLOW}[SKIP tl-check]{C_RESET} explicit segmentation/stream/timeline args provided")
        return 0

    failures = 0
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        # Multi-segment SRT: structural sanity + text consistency with stdout.
        if long_target.exists():
            out_srt = tmp / "multi.srt"
            cmd = [str(binary), "-d", str(model_dir), "-i", str(long_target), "--silent",
                   "-S", "20", "--timeline-file", str(out_srt)] + list(extra_args)
            print(f"{C_BCYAN}[....  tl-check]{C_RESET} multi-segment SRT on {long_target.name}...", flush=True)
            t0 = time.monotonic()
            rc, out, err = run_once(cmd, timeout_s, show_output=show_output)
            if rc != 0:
                print(f"{C_BRED}[FAIL tl-check]{C_RESET} transcription failed: {err or rc}")
                failures += 1
            else:
                entries = _parse_timeline(out_srt, "srt")
                errors = _timeline_sanity_errors(entries, _wav_duration_s(long_target))
                # Cue texts must reassemble (normalized) into the transcript.
                if not errors and out:
                    cue_text = normalize_text(" ".join(e[2] for e in entries))
                    ref_text = normalize_text(out)
                    denom = max(1, len(ref_text))
                    rate = levenshtein(cue_text, ref_text) / denom
                    if rate > 0.05:
                        errors.append(
                            f"cue text mismatch (norm rate {rate:.3f}): cues='{cue_text[:80]}...' stdout='{ref_text[:80]}...'"
                        )
                if errors:
                    failures += 1
                    for e in errors:
                        print(f"{C_BRED}[FAIL tl-check]{C_RESET} {e}")
                else:
                    print(
                        f"{C_BGREEN}[ OK  tl-check]{C_RESET} {C_BWHITE}{long_target.name}{C_RESET} | "
                        f"{len(entries)} cues, monotonic, in-range, text matches stdout | "
                        f"{fmt_time(time.monotonic() - t0)}"
                    )
        else:
            print(f"{C_BYELLOW}[SKIP tl-check]{C_RESET} missing sample: {long_target}")

        # Format sweep on the short sample (format inferred from extension).
        if short_target.exists():
            dur = _wav_duration_s(short_target)
            for fmt in TIMELINE_FORMATS:
                out_f = tmp / f"jfk.{fmt}"
                cmd = [str(binary), "-d", str(model_dir), "-i", str(short_target), "--silent",
                       "--timeline-file", str(out_f)] + list(extra_args)
                rc, out, err = run_once(cmd, timeout_s, show_output=show_output)
                if rc != 0:
                    print(f"{C_BRED}[FAIL tl-check]{C_RESET} {fmt}: transcription failed: {err or rc}")
                    failures += 1
                    continue
                entries = _parse_timeline(out_f, fmt)
                errors = _timeline_sanity_errors(entries, dur)
                if errors:
                    failures += 1
                    for e in errors:
                        print(f"{C_BRED}[FAIL tl-check]{C_RESET} {fmt}: {e}")
                else:
                    print(f"{C_BGREEN}[ OK  tl-check]{C_RESET} {fmt}: {len(entries)} cue(s) parsed and sane")

            # Explicit format override on an extensionless path.
            out_x = tmp / "jfk.out"
            cmd = [str(binary), "-d", str(model_dir), "-i", str(short_target), "--silent",
                   "--timeline-file", str(out_x), "--timeline-format", "jsonl"] + list(extra_args)
            rc, out, err = run_once(cmd, timeout_s, show_output=show_output)
            if rc != 0:
                print(f"{C_BRED}[FAIL tl-check]{C_RESET} format override: transcription failed: {err or rc}")
                failures += 1
            else:
                entries = _parse_timeline(out_x, "jsonl")
                errors = _timeline_sanity_errors(entries, dur)
                if errors:
                    failures += 1
                    for e in errors:
                        print(f"{C_BRED}[FAIL tl-check]{C_RESET} format override: {e}")
                else:
                    print(f"{C_BGREEN}[ OK  tl-check]{C_RESET} format override jsonl: {len(entries)} cue(s) parsed")
        else:
            print(f"{C_BYELLOW}[SKIP tl-check]{C_RESET} missing sample: {short_target}")

    return failures


def transcribe_stream_stdin(
    binary: Path,
    model_dir: Path,
    wav: Path,
    timeout_s: int,
    extra_args: Sequence[str],
    show_output: bool = False,
) -> str:
    cmd = [str(binary), "-d", str(model_dir), "--stdin", "--stream"]
    if not show_output:
        cmd.append("--silent")
    cmd += list(extra_args)

    cp = subprocess.run(
        cmd,
        input=wav.read_bytes(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_s,
        check=False,
    )
    out = cp.stdout.decode("utf-8", errors="replace").strip()
    err = cp.stderr.decode("utf-8", errors="replace").strip()
    if cp.returncode != 0:
        msg = err or f"exit code {cp.returncode}"
        raise RuntimeError(f"stream stdin transcription failed for {wav}: {msg}")
    if show_output and out:
        print(out)
    return out


def run_stream_stdin_regression(
    samples_root: Path,
    binary: Path,
    model_dir: Path,
    timeout_s: int,
    extra_args: Sequence[str],
    max_norm_rate: float,
    max_exact_rate: float,
    show_output: bool = False,
) -> int:
    target = samples_root / "jfk.wav"
    target_ref = target.with_suffix(".txt")
    if not target.exists() or not target_ref.exists():
        print(f"{C_BYELLOW}[SKIP stream-check]{C_RESET} missing sample/reference: {target}")
        return 0

    if any(a in ("-S", "--segment-overlap", "--stream", "--stdin", "--past-text") for a in extra_args):
        print(f"{C_BYELLOW}[SKIP stream-check]{C_RESET} explicit segmentation/stream args provided")
        return 0

    print(f"{C_BCYAN}[.... stream-check]{C_RESET} transcribing via --stdin --stream...", flush=True)
    t0 = time.monotonic()
    pred = transcribe_stream_stdin(
        binary=binary,
        model_dir=model_dir,
        wav=target,
        timeout_s=timeout_s,
        extra_args=extra_args,
        show_output=show_output,
    )
    elapsed = time.monotonic() - t0

    ref = target_ref.read_text(encoding="utf-8").strip()
    exact_dist = levenshtein(pred, ref)
    exact_den = max(1, len(ref))
    exact_rate = exact_dist / exact_den

    norm_ref = normalize_text(ref)
    norm_pred = normalize_text(pred)
    norm_dist = levenshtein(norm_pred, norm_ref)
    norm_den = max(1, len(norm_ref))
    norm_rate = norm_dist / norm_den

    ok = (norm_rate <= max_norm_rate) and (exact_rate <= max_exact_rate)
    if not ok:
        print(
            f"[DONE: {C_RED}FAIL{C_RESET}] stream-check jfk.wav | "
            f"exact {exact_dist}/{exact_den} ({C_RED}{exact_rate:.3f}{C_RESET}) | "
            f"norm {norm_dist}/{norm_den} ({C_RED}{norm_rate:.3f}{C_RESET}) | "
            f"{C_DIM}{fmt_time(elapsed)}{C_RESET}"
        )
        show_text_diff("ref", ref, "got", pred)
        return 1

    print(
        f"[DONE: {C_GREEN}OK{C_RESET}] stream-check jfk.wav | "
        f"exact {exact_dist}/{exact_den} ({C_GREEN}{exact_rate:.3f}{C_RESET}) | "
        f"norm {norm_dist}/{norm_den} ({C_GREEN}{norm_rate:.3f}{C_RESET}) | "
        f"{C_DIM}{fmt_time(elapsed)}{C_RESET}"
    )
    return 0


def run_stream_cache_once(
    binary: Path,
    model_dir: Path,
    wav: Path,
    timeout_s: int,
    enc_window_sec: float,
    threads: int,
    cache_on: bool,
) -> Tuple[int, str, str, float]:
    cmd = [
        str(binary),
        "-t", str(threads),
        "-d", str(model_dir),
        "-i", str(wav),
        "--stream",
        "--enc-window-sec", f"{enc_window_sec:g}",
        "--debug",  # Keep a token callback: --silent bypasses streaming.
    ]
    env = os.environ.copy()
    if cache_on:
        env.pop("QWEN_STREAM_NO_ENC_CACHE", None)
    else:
        env["QWEN_STREAM_NO_ENC_CACHE"] = "1"

    t0 = time.monotonic()
    cp = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_s,
        check=False,
        env=env,
    )
    elapsed = time.monotonic() - t0
    rc = cp.returncode
    err = cp.stderr.strip()
    if rc == 0 and (err.count("  Commit:") < 2 or "direct final refinement" in err):
        rc = 1
        err = "Streaming loop was not exercised\n" + err
    return rc, cp.stdout.strip(), err, elapsed


def run_stream_cache_regression(
    samples_root: Path,
    binary: Path,
    model_dir: Path,
    timeout_s: int,
    enc_window_sec: float,
    threads: int,
    sample_args: Sequence[str],
) -> int:
    if sample_args:
        wavs = [Path(p).resolve() for p in sample_args]
    else:
        wavs = [(samples_root / rel).resolve() for rel in STREAM_CACHE_DEFAULT_SAMPLES]

    missing = [w for w in wavs if not w.exists()]
    if missing:
        print(f"{C_BYELLOW}[SKIP stream-cache-check]{C_RESET} missing sample(s):")
        for w in missing:
            print(f"       - {w}")
        return 0

    total = len(wavs)
    failures = 0
    print(
        f"{C_BCYAN}[.... stream-cache-check]{C_RESET} cache on/off equivalence "
        f"(model={model_dir.name}, window={enc_window_sec:g}s, threads={threads})"
    )
    for idx, wav in enumerate(wavs, 1):
        print(f"[START cache {idx}/{total}] {C_BWHITE}{wav.name}{C_RESET} ...", flush=True)
        rc_on, out_on, err_on, t_on = run_stream_cache_once(
            binary=binary,
            model_dir=model_dir,
            wav=wav,
            timeout_s=timeout_s,
            enc_window_sec=enc_window_sec,
            threads=threads,
            cache_on=True,
        )
        rc_off, out_off, err_off, t_off = run_stream_cache_once(
            binary=binary,
            model_dir=model_dir,
            wav=wav,
            timeout_s=timeout_s,
            enc_window_sec=enc_window_sec,
            threads=threads,
            cache_on=False,
        )

        if rc_on != 0 or rc_off != 0:
            failures += 1
            print(
                f"[DONE: {C_RED}FAIL{C_RESET} cache {idx}/{total}] {C_BWHITE}{wav.name}{C_RESET} | "
                f"rc_on={rc_on} rc_off={rc_off}"
            )
            if rc_on != 0 and err_on:
                print(f"       stderr on: {err_on[:220]}")
            if rc_off != 0 and err_off:
                print(f"       stderr off: {err_off[:220]}")
            continue

        exact_dist = levenshtein(out_on, out_off)
        exact_den = max(1, len(out_on))
        exact_rate = exact_dist / exact_den
        norm_on = normalize_text(out_on)
        norm_off = normalize_text(out_off)
        norm_dist = levenshtein(norm_on, norm_off)
        norm_den = max(1, len(norm_on))
        norm_rate = norm_dist / norm_den
        ok = (exact_dist == 0)
        if not ok:
            failures += 1

        status = f"{C_GREEN}OK{C_RESET}" if ok else f"{C_RED}FAIL{C_RESET}"
        rate_color = C_GREEN if ok else C_RED
        print(
            f"[DONE: {status} cache {idx}/{total}] {C_BWHITE}{wav.name}{C_RESET} | "
            f"exact {exact_dist}/{exact_den} ({rate_color}{exact_rate:.3f}{C_RESET}) | "
            f"norm {norm_dist}/{norm_den} ({rate_color}{norm_rate:.3f}{C_RESET}) | "
            f"time on/off {C_DIM}{fmt_time(t_on)}/{fmt_time(t_off)}{C_RESET}"
        )
        if not ok:
            show_text_diff("cache on", out_on, "cache off", out_off)

    if failures:
        print(f"{C_BRED}[FAIL stream-cache-check]{C_RESET} {failures}/{total} samples")
        return 1
    print(f"{C_BGREEN}[ OK  stream-cache-check]{C_RESET} {total}/{total} samples")
    return 0


def generate_refs(
    wavs: Iterable[Path],
    binary: Path,
    model_dir: Path,
    timeout_s: int,
    extra_args: Sequence[str],
    refresh: bool,
    show_output: bool = False,
) -> int:
    generated = 0
    skipped = 0
    wav_list = list(wavs)
    total = len(wav_list)
    t_start = time.monotonic()
    for idx, wav in enumerate(wav_list, 1):
        ref = ref_for_wav(wav)
        if ref.exists() and not refresh:
            skipped += 1
            continue
        print(f"{C_BCYAN}[gen {idx}/{total}]{C_RESET} {C_BWHITE}{wav.name}{C_RESET} ...", end="" if not show_output else "\n", flush=True)
        t0 = time.monotonic()
        txt = transcribe(binary, model_dir, wav, timeout_s, extra_args, show_output=show_output)
        elapsed = time.monotonic() - t0
        preview = txt[:70] + ("..." if len(txt) > 70 else "")
        print(f" {C_DIM}{fmt_time(elapsed)}{C_RESET}  {C_DIM}\"{preview}\"{C_RESET}")
        ref.write_text(txt + "\n", encoding="utf-8")
        generated += 1
    total_time = time.monotonic() - t_start
    if skipped:
        print(f"{C_DIM}Skipped {skipped} existing references{C_RESET}")
    print(f"{C_BOLD}Generated {generated} references in {fmt_time(total_time)}{C_RESET}")
    return generated


def show_text_diff(label_a: str, text_a: str, label_b: str, text_b: str, max_chars: int = 200) -> None:
    """Print a compact side-by-side of two texts when they differ."""
    def trunc(s: str) -> str:
        return s[:max_chars] + ("..." if len(s) > max_chars else "")
    print(f"       {C_GREEN}{label_a}: \"{trunc(text_a)}\"{C_RESET}")
    print(f"       {C_RED}{label_b}: \"{trunc(text_b)}\"{C_RESET}")


def run_regression(
    wavs: Iterable[Path],
    binary: Path,
    model_dir: Path,
    timeout_s: int,
    extra_args: Sequence[str],
    max_norm_rate: float,
    max_exact_rate: float,
    show_output: bool = False,
) -> int:
    wav_list = list(wavs)
    total = len(wav_list)
    failures = 0
    skipped_missing_ref = 0
    t_start = time.monotonic()

    print(f"{C_BOLD}Running regression on {total} samples{C_RESET}")
    print(f"Thresholds: normalized <= {C_BYELLOW}{max_norm_rate:.3f}{C_RESET}, exact <= {C_BYELLOW}{max_exact_rate:.3f}{C_RESET}")
    print()

    for idx, wav in enumerate(wav_list, 1):
        ref = ref_for_wav(wav)
        if not ref.exists():
            print(f"{C_BYELLOW}[SKIP {idx}/{total}]{C_RESET} {C_BWHITE}{wav.name}{C_RESET} | missing reference")
            skipped_missing_ref += 1
            continue

        print(
            f"{C_BCYAN}[START {idx}/{total}]{C_RESET} {C_BWHITE}{wav.name}{C_RESET} ...",
            end="" if not show_output else "\n",
            flush=True,
        )
        target = ref.read_text(encoding="utf-8").strip()
        t0 = time.monotonic()
        pred = transcribe(binary, model_dir, wav, timeout_s, extra_args, show_output=show_output)
        elapsed = time.monotonic() - t0

        exact_dist = levenshtein(pred, target)
        exact_den = max(1, len(target))
        exact_rate = exact_dist / exact_den

        norm_target = normalize_text(target)
        norm_pred = normalize_text(pred)
        norm_dist = levenshtein(norm_pred, norm_target)
        norm_den = max(1, len(norm_target))
        norm_rate = norm_dist / norm_den

        ok = (norm_rate <= max_norm_rate) and (exact_rate <= max_exact_rate)
        if not ok:
            failures += 1

        if ok:
            done_status = f"{C_GREEN}OK{C_RESET}"
            rate_color = C_GREEN
        else:
            done_status = f"{C_RED}FAIL{C_RESET}"
            rate_color = C_RED

        print(
            f"[DONE: {done_status} {idx}/{total}] {C_BWHITE}{wav.name}{C_RESET} | "
            f"exact {exact_dist}/{exact_den} ({rate_color}{exact_rate:.3f}{C_RESET}) | "
            f"norm {norm_dist}/{norm_den} ({rate_color}{norm_rate:.3f}{C_RESET}) | "
            f"{C_DIM}{fmt_time(elapsed)}{C_RESET}"
        )

        if not ok:
            show_text_diff("ref", target, "got", pred)

    total_time = time.monotonic() - t_start
    print()
    if failures:
        print(f"{C_BRED}Regression FAILED: {failures}/{total} samples out of threshold  ({fmt_time(total_time)} total){C_RESET}")
        if skipped_missing_ref:
            print(f"{C_BYELLOW}Skipped {skipped_missing_ref} sample(s) with missing references{C_RESET}")
        return 1
    passed = total - skipped_missing_ref
    print(f"{C_BGREEN}Regression PASSED: {passed}/{total} samples within threshold  ({fmt_time(total_time)} total){C_RESET}")
    if skipped_missing_ref:
        print(f"{C_BYELLOW}Skipped {skipped_missing_ref} sample(s) with missing references{C_RESET}")
    return 0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="qwen_asr regression suite (reference generation + quality checks)"
    )
    ap.add_argument(
        "--samples-root",
        default="samples",
        help="Root folder to scan recursively for *.wav (default: samples)",
    )
    ap.add_argument(
        "--binary",
        default="./qwen_asr",
        help="Path to qwen_asr binary (default: ./qwen_asr)",
    )
    ap.add_argument(
        "--model-dir",
        default="qwen3-asr-1.7b",
        help="Model directory used for references/checks (default: qwen3-asr-1.7b)",
    )
    ap.add_argument(
        "--timeout-s",
        type=int,
        default=1200,
        help="Per-sample transcription timeout seconds (default: 1200)",
    )
    ap.add_argument(
        "--max-norm-rate",
        type=float,
        default=0.20,
        help="Max normalized distance rate for pass/fail (default: 0.20)",
    )
    ap.add_argument(
        "--max-exact-rate",
        type=float,
        default=1.00,
        help="Max exact distance rate for pass/fail (default: 1.00; mostly informational)",
    )
    ap.add_argument(
        "--arg",
        action="append",
        default=[],
        help="Extra arg forwarded to qwen_asr (can be repeated)",
    )
    ap.add_argument(
        "--generate-missing",
        action="store_true",
        help="Generate missing sibling .txt references for WAV files",
    )
    ap.add_argument(
        "--refresh-refs",
        action="store_true",
        help="Regenerate all sibling .txt references",
    )
    ap.add_argument(
        "--segment-check-only",
        action="store_true",
        help="Run only segmented-conditioning collapse regression check",
    )
    ap.add_argument(
        "--skip-segment-check",
        action="store_true",
        help="Skip segmented-conditioning collapse regression check",
    )
    ap.add_argument(
        "--stream-check-only",
        action="store_true",
        help="Run only streaming stdin regression check",
    )
    ap.add_argument(
        "--skip-stream-check",
        action="store_true",
        help="Skip streaming stdin regression check",
    )
    ap.add_argument(
        "--stream-cache-check-only",
        action="store_true",
        help="Run only stream cache on/off equivalence regression check",
    )
    ap.add_argument(
        "--skip-stream-cache-check",
        action="store_true",
        help="Skip stream cache on/off equivalence regression check",
    )
    ap.add_argument(
        "--timeline-check-only",
        action="store_true",
        help="Run only timeline (cue timestamps) regression check",
    )
    ap.add_argument(
        "--skip-timeline-check",
        action="store_true",
        help="Skip timeline (cue timestamps) regression check",
    )
    ap.add_argument(
        "--stream-cache-model-dir",
        default=STREAM_CACHE_DEFAULT_MODEL_DIR,
        help=f"Model directory used for stream cache check (default: {STREAM_CACHE_DEFAULT_MODEL_DIR})",
    )
    ap.add_argument(
        "--stream-cache-enc-window-sec",
        type=float,
        default=8.0,
        help="Encoder attention window seconds for stream cache check (default: 8.0)",
    )
    ap.add_argument(
        "--stream-cache-threads",
        type=int,
        default=1,
        help="Threads for stream cache check (default: 1, deterministic)",
    )
    ap.add_argument(
        "--stream-cache-sample",
        action="append",
        default=[],
        help="WAV path for stream cache check (repeatable; default uses built-in samples)",
    )
    ap.add_argument(
        "--segment-min-ratio",
        type=float,
        default=0.80,
        help=(
            "Min acceptable ratio between --past-text yes output words and "
            "--past-text no output words on long sample (default: 0.80)"
        ),
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    samples_root = Path(args.samples_root).resolve()
    binary = Path(args.binary).resolve()

    if not binary.exists():
        print(f"missing binary: {binary}", file=sys.stderr)
        return 2
    if not samples_root.exists():
        print(f"missing samples root: {samples_root}", file=sys.stderr)
        return 2
    if args.stream_cache_threads <= 0:
        print("--stream-cache-threads must be > 0", file=sys.stderr)
        return 2
    if args.stream_cache_enc_window_sec < 1.0 or args.stream_cache_enc_window_sec > 8.0:
        print("--stream-cache-enc-window-sec must be in [1, 8]", file=sys.stderr)
        return 2

    all_wavs = find_wavs(samples_root)
    if not all_wavs:
        print(f"no wav files found under: {samples_root}", file=sys.stderr)
        return 2
    wavs_with_refs = [w for w in all_wavs if ref_for_wav(w).exists()]

    print(
        f"{C_BOLD}Discovered {len(all_wavs)} wav files under {samples_root} "
        f"({len(wavs_with_refs)} with references){C_RESET}"
    )

    focused_count = sum(
        1 for f in (args.segment_check_only, args.stream_check_only,
                    args.stream_cache_check_only, args.timeline_check_only) if f
    )
    if focused_count > 1:
        print("--segment-check-only, --stream-check-only, --stream-cache-check-only and "
              "--timeline-check-only are mutually exclusive",
              file=sys.stderr)
        return 2

    if args.segment_check_only and (args.generate_missing or args.refresh_refs):
        print("--segment-check-only cannot be combined with reference generation", file=sys.stderr)
        return 2
    if args.stream_check_only and (args.generate_missing or args.refresh_refs):
        print("--stream-check-only cannot be combined with reference generation", file=sys.stderr)
        return 2
    if args.stream_cache_check_only and (args.generate_missing or args.refresh_refs):
        print("--stream-cache-check-only cannot be combined with reference generation", file=sys.stderr)
        return 2
    if args.timeline_check_only and (args.generate_missing or args.refresh_refs):
        print("--timeline-check-only cannot be combined with reference generation", file=sys.stderr)
        return 2

    show_output = True

    should_generate = args.generate_missing or args.refresh_refs
    any_focused_only = (args.segment_check_only or args.stream_check_only or
                        args.stream_cache_check_only or args.timeline_check_only)

    run_segment = (not args.skip_segment_check and
                   not args.stream_check_only and
                   not args.stream_cache_check_only and
                   not args.timeline_check_only)
    run_stream = (not args.skip_stream_check and
                  not args.segment_check_only and
                  not args.stream_cache_check_only and
                  not args.timeline_check_only)
    run_stream_cache = (not args.skip_stream_cache_check and
                        not args.segment_check_only and
                        not args.stream_check_only and
                        not args.timeline_check_only)
    run_timeline = (not args.skip_timeline_check and
                    not args.segment_check_only and
                    not args.stream_check_only and
                    not args.stream_cache_check_only)

    need_primary_model = (should_generate or run_segment or run_stream or
                          run_timeline or (not any_focused_only))
    model_dir = Path(args.model_dir).resolve()
    if need_primary_model and not model_dir.exists():
        print(f"missing model dir: {model_dir}", file=sys.stderr)
        return 2

    stream_cache_model_dir = Path(args.stream_cache_model_dir).resolve()
    if run_stream_cache and not stream_cache_model_dir.exists():
        print(f"missing stream-cache model dir: {stream_cache_model_dir}", file=sys.stderr)
        return 2

    if should_generate:
        generated = generate_refs(
            all_wavs,
            binary=binary,
            model_dir=model_dir,
            timeout_s=args.timeout_s,
            extra_args=args.arg,
            refresh=args.refresh_refs,
            show_output=show_output,
        )
        print(f"{C_BGREEN}Reference generation completed: wrote {generated} .txt files{C_RESET}")

    failures = 0
    if run_segment:
        failures += run_segment_conditioning_regression(
            samples_root=samples_root,
            binary=binary,
            model_dir=model_dir,
            timeout_s=args.timeout_s,
            extra_args=args.arg,
            min_ratio=args.segment_min_ratio,
            show_output=show_output,
        )

    if run_stream:
        failures += run_stream_stdin_regression(
            samples_root=samples_root,
            binary=binary,
            model_dir=model_dir,
            timeout_s=args.timeout_s,
            extra_args=args.arg,
            max_norm_rate=args.max_norm_rate,
            max_exact_rate=args.max_exact_rate,
            show_output=show_output,
        )

    if run_stream_cache:
        failures += run_stream_cache_regression(
            samples_root=samples_root,
            binary=binary,
            model_dir=stream_cache_model_dir,
            timeout_s=args.timeout_s,
            enc_window_sec=args.stream_cache_enc_window_sec,
            threads=args.stream_cache_threads,
            sample_args=args.stream_cache_sample,
        )

    if run_timeline:
        failures += run_timeline_regression(
            samples_root=samples_root,
            binary=binary,
            model_dir=model_dir,
            timeout_s=args.timeout_s,
            extra_args=args.arg,
            show_output=show_output,
        )

    if any_focused_only:
        if failures:
            print(f"\n{C_BRED}Focused regression checks FAILED{C_RESET}")
            return 1
        print(f"\n{C_BGREEN}Focused regression checks PASSED{C_RESET}")
        return 0

    if not wavs_with_refs:
        print(f"{C_BYELLOW}[SKIP]{C_RESET} no wav files with sibling .txt references")
        rc = 0
    else:
        rc = run_regression(
            wavs_with_refs,
            binary=binary,
            model_dir=model_dir,
            timeout_s=args.timeout_s,
            extra_args=args.arg,
            max_norm_rate=args.max_norm_rate,
            max_exact_rate=args.max_exact_rate,
            show_output=show_output,
        )
    if failures:
        print(f"\n{C_BRED}Overall result FAILED due to focused regression check failure{C_RESET}")
        return 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
