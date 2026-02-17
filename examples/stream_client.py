#!/usr/bin/env python3
"""
Streaming ASR client sample for Qwen3-ASR C implementation.

Connects to the WebSocket streaming endpoint, sends audio (WAV converted to
s16le 16 kHz mono), and prints received transcript chunks to stdout.

Metrics (printed to stderr):
  - TTFB (time to first byte): seconds from end of sending audio until the first
    transcript chunk is received. Lower is better for perceived latency.
  - RTF (real-time factor): (processing wall time) / (audio duration).
    RTF < 1 means faster than realtime; e.g. 0.25 = 4x realtime.

Usage:
  python examples/stream_client.py --url ws://localhost:2020/stream --audio samples/jfk.wav

Requires: pip install websockets
"""

import argparse
import struct
import sys
import time
import wave


def wav_to_s16le_16k_mono(wav_path: str) -> bytes:
    """Read WAV and return raw s16le 16 kHz mono. Resamples if needed (stdlib only)."""
    with wave.open(wav_path, "rb") as w:
        nch = w.getnchannels()
        sampwidth = w.getsampwidth()
        framerate = w.getframerate()
        nframes = w.getnframes()
        if sampwidth != 2:
            raise ValueError(f"WAV must be 16-bit; got {sampwidth} bytes/sample")
        raw = w.readframes(nframes)

    # bytes -> int16
    count = len(raw) // 2
    samples = list(struct.unpack(f"<{count}h", raw))

    # to mono
    if nch == 2:
        samples = [(samples[i] + samples[i + 1]) // 2 for i in range(0, count, 2)]
    elif nch != 1:
        raise ValueError(f"WAV must be mono or stereo; got {nch} channels")

    # resample to 16 kHz if needed
    if framerate != 16000:
        new_len = int(round(len(samples) * 16000 / framerate))
        resampled = []
        for i in range(new_len):
            pos = i * framerate / 16000
            idx = int(pos)
            frac = pos - idx
            if idx >= len(samples) - 1:
                resampled.append(samples[-1])
            else:
                a, b = samples[idx], samples[idx + 1]
                resampled.append(int(a + frac * (b - a)))
        samples = resampled

    return struct.pack(f"<{len(samples)}h", *samples)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stream audio to ASR WebSocket and print transcript")
    parser.add_argument("--url", default="ws://localhost:2020/stream", help="WebSocket URL")
    parser.add_argument("--audio", required=True, help="Path to WAV file (16-bit; will be resampled to 16 kHz mono)")
    parser.add_argument("--chunk", type=int, default=8192, help="Bytes per WebSocket binary frame")
    parser.add_argument("--no-stats", action="store_true", help="Do not print TTFB/RTF to stderr")
    args = parser.parse_args()

    try:
        import asyncio
        import websockets
    except ImportError:
        print("Install: pip install websockets", file=sys.stderr)
        return 1

    try:
        audio_bytes = wav_to_s16le_16k_mono(args.audio)
    except Exception as e:
        print(f"Error reading WAV: {e}", file=sys.stderr)
        return 1

    # Audio duration: 16 kHz, 16-bit mono = 32000 bytes per second
    audio_duration_sec = len(audio_bytes) / 32000.0

    async def run():
        ttfb_sec = None
        t_after_end = None

        async with websockets.connect(args.url) as ws:
            # Send audio in chunks
            for i in range(0, len(audio_bytes), args.chunk):
                chunk = audio_bytes[i : i + args.chunk]
                await ws.send(chunk)
            # Signal end of input so server closes stdin
            await ws.send("end")
            t_after_end = time.perf_counter()

            # Print transcript as it arrives
            while True:
                try:
                    msg = await ws.recv()
                except Exception:
                    break
                if ttfb_sec is None:
                    ttfb_sec = time.perf_counter() - t_after_end
                if isinstance(msg, str):
                    print(msg, end="", flush=True)
                elif isinstance(msg, bytes):
                    print(msg.decode("utf-8", errors="replace"), end="", flush=True)
        print()

        # Stats
        if not args.no_stats and t_after_end is not None:
            total_sec = time.perf_counter() - t_after_end
            rtf = total_sec / audio_duration_sec if audio_duration_sec > 0 else 0
            ttfb_str = f"{ttfb_sec:.3f}s" if ttfb_sec is not None else "N/A"
            print(f"TTFB: {ttfb_str}  RTF: {rtf:.3f}  (audio: {audio_duration_sec:.2f}s)", file=sys.stderr)

    asyncio.run(run())
    return 0


if __name__ == "__main__":
    sys.exit(main())
