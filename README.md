# Qwen3-ASR Pure C Implementation

This is a C implementation of the inference pipeline for [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) speech-to-text models (both 0.6B and 1.7B). The CPU build has zero external dependencies beyond the C standard library and a BLAS implementation (Accelerate on macOS, OpenBLAS on Linux). Optional GPU backends accelerate inference on **AMD GPUs with ROCm** (`make rocm`) and **NVIDIA GPUs with CUDA** (`make cuda`). See [ROCm / AMD GPUs](#rocm--amd-gpus) and [CUDA / NVIDIA GPUs](#cuda--nvidia-gpus). Tokens stream to stdout as they are generated. The implementation runs at speed multiple of the file length even in very modest hardware, like low end Intel or AMD processor.

**Important**: this implementation explicitly **avoids implementing support for MPS**. Transcription systems are very important pieces of infrastructure, and are often run on remote Linux servers. Adding the MPS target would focus the efforts too much on Apple hardware, so for now I'm skipping it. The code runs very well anyway on Apple hardware (NEON optimized). Please, **don't send pull requests** about this feature, fork the code instead, in order to add MPS support. I'll add it much later when the other optimizations are already mature.

## Supported modes and models

Both normal (offline) and streaming (online) modes are supported. The CPU build defaults to full offline decode (`-S 0`), so the whole audio is encoded at once. The ROCm and CUDA builds default to 20-second segments (`-S 20`) to bound GPU attention and memory use; `-S 0` explicitly enables full-audio decode. Streaming mode processes audio in 2-second chunks with prefix rollback (it keeps the last few decoded tokens as context for the decoder/LLM when transcribing the next chunk).

*Important practical note*: in this implementation, interactive `--stream` prioritizes incremental token stability over throughput and can be much slower than normal mode when you process an already-recorded file end-to-end.

Audio can be piped from stdin (`--stdin`), making it easy to transcode and transcribe any format via ffmpeg. Language is usually auto-detected from audio, and can be forced with `--language`. A system prompt can bias the model toward specific terms or spellings.

Both the 0.6B and 1.7B parameters models are supported. While the 1.7B model is generally more powerful, the 0.6B model seems the sweet spot for CPU inference, however the speed difference is not huge, so you may want to try both and decide what to use depending on your use case.

## Quick Start

```bash
# Build
make blas

# Download a model (interactive selector: small=0.6B, large=1.7B)
./download_model.sh

# Transcribe audio (tokens stream to stdout as generated)
./qwen_asr -d qwen3-asr-0.6b -i audio.wav

# Pipe any format via ffmpeg
ffmpeg -i audio.mp3 -f s16le -ar 16000 -ac 1 - 2>/dev/null | \
    ./qwen_asr -d qwen3-asr-0.6b --stdin

# Streaming mode (incremental output for live audio)
./qwen_asr -d qwen3-asr-0.6b -i long_recording.wav --stream
```

## Features

- **Almost zero dependencies**: The CPU implementation only needs BLAS (Accelerate on macOS, OpenBLAS on Linux).
- **NVIDIA GPU acceleration**: Native CUDA kernels, cuBLAS, Q8 decoding, and CUDA graphs. Tested on DGX Spark (GB10, `sm_121`).
- **AMD GPU acceleration**: Optional ROCm backend with HIP kernels, rocBLAS, Q8 decoding, and HIP graphs. Tested on Strix Halo (Radeon 8060S, `gfx1151`).
- **Both models**: Automatically detects Qwen3-ASR-0.6B or 1.7B from the weight files.
- **Streaming output**: Tokens are printed to stdout as they are generated, word by word, even in offline mode (no `--stream`).
- **Streaming mode**: `--stream` processes audio in chunks with prefix rollback. A sliding window bounds encoder and decoder context for indefinite streaming.
- **Monitor mode**: `--monitor` shows inline Unicode symbols on stderr for real-time pipeline diagnostics.
- **Language control**: `--language Italian` forces the target language (otherwise it is usually auto-detected).
- **Prompt biasing**: `--prompt` injects a system prompt to bias the model toward specific terms or spellings. Note that prompt biasing is very soft. The models may or may not care about your instructions. Usually spelling instructions are followed decently.
- **Optional silence skipping**: `--skip-silence` drops long silent spans before inference (off by default). It may use less CPU for the same file.
- **Memory-mapped weights**: CPU inference accesses decoder BF16 weights directly from safetensors files. GPU backends convert and upload weights at load time.
- **WAV input**: Supports 16-bit PCM WAV files at any sample rate (auto-resampled to 16kHz).
- **Stdin input**: Reads from stdin with auto-detection (WAV header or raw s16le 16kHz mono).
- **Optional segment splitting**: use `-S 20` / `-S 30` for large files with segment-cutting silence search (`-W 3`).
- **Timeline output**: `--timeline-file` writes approximate cue timestamps as SRT/VTT/JSONL/plain text (offline modes).

## Usage

### Normal Mode (Default)

```bash
./qwen_asr -d qwen3-asr-0.6b -i recording.wav
```

The CPU binary defaults to `-S 0` (full-audio offline decode); the GPU binaries default to `-S 20`.
The model sees the entire recording in one shot, which is usually best for short/medium files.
For long files, memory/time grow with sequence length, so segmented mode (`-S 20` or similar) is often preferable.

Tokens stream to stdout as they are generated. By default, timing info is printed to stderr (`Inference: ...` and `Audio: ... (Xx realtime)`). Use `--silent` or `--debug` to control verbosity:

```bash
./qwen_asr -d qwen3-asr-0.6b -i audio.wav --silent    # no stderr output
./qwen_asr -d qwen3-asr-0.6b -i audio.wav --debug      # per-layer/per-chunk details
```

Token emission behavior depends on mode:
- With `-S 0`, text is emitted token-by-token as soon as each decode step produces it.
- With segmented mode (`-S > 0`), default behavior is still token-by-token ASAP.
- With segmented mode plus `--past-text yes`, boundary cleanup is enabled automatically and output is emitted once per segment after post-processing.

For very long files, decoder cost still grows with sequence history. Use `--stream` when you need incremental output while audio is arriving.

### Which Mode To Use (By File Length)

- **Up to ~60s**: try `-S 0` (the CPU default) if full-context quality and speed are acceptable. Prefer `-S 20` if full-context decoding produces empty or incomplete output.
- **Large prerecorded files**: use segmented offline mode, e.g. `-S 20` (or `-S 30`, or even more).
- **Long live/continuous audio or low-latency UI needs**: use `--stream`.
- **Batch/offline file transcription**: prefer `-S 20`/`-S 30`; it is usually much faster than interactive `--stream`.
- **If segmented output drops/warps around boundaries**: try a different segment size and keep default `--past-text auto`.
- **If you want stronger continuity across segments/chunks**: try `--past-text yes` (can help continuity, can also cause drift on some files).

Large-file tradeoff summary:
- `-S 20`: offline segmented decode, usually best throughput on long files, stable memory, and token-by-token output.
- `-S 20 --past-text yes`: buffered per-segment output with boundary cleanup and continuity bias.
- `--stream`: incremental output while audio arrives, lower interaction latency, but usually higher total compute for full prerecorded files.

### Streaming Mode (`--stream`)

```bash
./qwen_asr -d qwen3-asr-0.6b -i long_recording.wav --stream
```

Streaming mode processes audio in **2-second chunks** with rollback text conditioning:

1. Audio arrives chunk by chunk.
2. Encoder uses local windows (`--enc-window-sec`, default `8s`); completed windows are cached, only the current partial tail window is re-encoded.
3. Decoder prompt includes previous output minus a rollback suffix (5 tokens by default) to stabilize chunk boundaries.
4. Per chunk decode is bounded by `--stream-max-new-tokens` (default `32`).
5. Only stable text is emitted; final chunk flushes remaining text.

This keeps encoder recomputation under control. For long streams, a **sliding window** automatically bounds both encoder and decoder context so that memory and compute stay flat indefinitely:

- **Encoder window eviction**: only the most recent 4 encoder attention windows (~32 s of audio context) are kept; older windows are freed.
- **Decoder prefix capping**: only the most recent ~150 tokens of previously decoded text are fed as decoder prefix context. Internal token history is still tracked for streaming commit/dedup decisions, while only the capped tail is embedded into the decoder input.

These limits activate automatically when the stream is long enough to exceed them. For short files or live sessions under ~40 s, they have no effect.

`--stream --silent` has a special non-interactive behavior for file input: it skips chunk-by-chunk streaming and runs one direct final refinement pass. (For live stdin streaming, chunked mode is still used.)

Default stream settings:
- `chunk_size`: 2s
- `encoder_window`: 8s (`--enc-window-sec`, range `1..8`)
- `rollback`: 5 tokens
- `unfixed_chunks`: 2
- `max_new_tokens`: 32 (`--stream-max-new-tokens`)
- `past_text`: `auto` by default (effectively `yes` for `--stream`, `no` otherwise)

Streaming tuning:

```bash
# default streaming
./qwen_asr -d qwen3-asr-0.6b -i audio.wav --stream

# lower-latency encoder window (may reduce quality)
./qwen_asr -d qwen3-asr-0.6b -i audio.wav --stream --enc-window-sec 4

# allow more text generation per chunk
./qwen_asr -d qwen3-asr-0.6b -i audio.wav --stream --stream-max-new-tokens 64
```

### Monitor Mode (`--monitor`)

```bash
./qwen_asr -d qwen3-asr-0.6b -i audio.wav --stream --monitor
```

Shows inline Unicode symbols on stderr alongside the transcription, useful for diagnosing streaming pipeline behavior in real time:

| Symbol | Meaning |
|--------|---------|
| `▶` | Encoder chunk processed |
| `·` | Decoder prefill completed |
| `▪` | Decode step (normal speed) |
| `▸` | Decode step (slow, >30 ms/token) |
| `⟳` | Encoder window evicted (sliding window) |

Example output (stderr + stdout interleaved):
```
▶·▪▶·▪▶·▪And so, my fellow Americans,▶·▪ ask not what your country...
```

Monitor output goes to stderr and does not affect the transcription text on stdout. It can be combined with `--debug` for full diagnostics, or used alone for a lightweight visual heartbeat.

### Timeline / Timestamps (`--timeline-file`)

```bash
# SRT subtitles (format inferred from the .srt extension)
./qwen_asr -d qwen3-asr-0.6b -i recording.wav --timeline-file out.srt

# Explicit format override; "-" writes the timeline to stdout
./qwen_asr -d qwen3-asr-0.6b -i recording.wav --timeline-file out.jsonl
./qwen_asr -d qwen3-asr-0.6b -i recording.wav --timeline-file - --timeline-format txt
```

Writes timed text cues for offline transcription (`-i` and `--stdin`; not `--stream`). Four formats are supported, inferred from the output extension (`srt`, `vtt`, `jsonl`, `txt`; default `srt`) and overridable with `--timeline-format`. The transcription itself is unchanged: full text still goes to stdout, the timeline is written at the end.

How the times are produced:

- **Segment boundaries are exact.** With `-S > 0`, cues never cross a segment cut, and segment cuts come from the silence search at known audio offsets.
- **Within a segment, times are estimated.** Text is split at sentence punctuation (then clause punctuation when a cue would exceed `--timeline-max-cue-sec`, default 7), and the pieces are distributed proportionally over the segment's speech-active windows (energy-based voice detection), with boundaries snapped to nearby silences. Expect roughly ±1–2 s accuracy inside a segment — good enough for subtitles, not word-accurate alignment.

Notes:

- If `-S` is not given, `--timeline-file` enables `-S 20` automatically (a notice is printed to stderr) because timeline granularity comes from segment boundaries. `-S 0` is allowed and still produces intra-segment cues.
- Not compatible with `--skip-silence` (silence compaction destroys absolute time) — the CLI rejects the combination.
- `--timeline-max-cue-sec <s>` bounds the estimated cue length (default 7).

Example SRT output:

```text
1
00:00:00,000 --> 00:00:03,450
如何把广东省鞋拍成买不起的样子？

2
00:00:03,450 --> 00:00:09,180
首先，拍一个你这辈子都不会穿拖鞋去的地方。
```

C API:

```c
qwen_set_timeline(ctx, 1);              /* collect cues during transcription */
qwen_transcribe(ctx, "audio.wav");      /* or any offline transcribe call */

const qwen_timeline_entry_t *entries;
int n = qwen_get_timeline(ctx, &entries);
for (int i = 0; i < n; i++) {
    printf("[%.2f - %.2f] %s\n", entries[i].start_sec, entries[i].end_sec, entries[i].text);
}
qwen_timeline_reset(ctx);               /* optional: clear; also freed by qwen_free() */
```

Entries accumulate across transcribe calls until `qwen_timeline_reset()`.

### Segment Splitting (`-S`)

```bash
./qwen_asr -d qwen3-asr-0.6b -i long_recording.wav -S 20
```

Splits audio into segments of ~N seconds, finding segment-cutting silence boundaries within a search window (`-W`, default 3 seconds). Segments are transcribed and concatenated.

Default segmented behavior (`-S > 0`) emits tokens ASAP, like full offline mode.

When `--past-text yes` is used, segmented mode switches to buffered per-segment emission and enables boundary post-processing:
- Split points are chosen near low-energy (silence-like) regions within the `-W` window to avoid cutting in the middle of words.
- If past-text conditioning causes a segment collapse (too short for its duration) or large duplicate span, that segment is retried without conditioning.
- If collapses keep happening, past-text conditioning is disabled for the remainder of the run.
- Boundary whitespace and spacing are normalized when segments are appended.

By default (`--past-text auto`), segmented mode does **not** use past-text conditioning. This is usually more stable on long files.
If you want extra continuity bias across boundaries, enable conditioning explicitly:

```bash
./qwen_asr -d qwen3-asr-0.6b -i lecture.wav -S 20 --past-text yes
```

If repeated conditioned segment collapses are detected, conditioning is disabled automatically for the rest of the run (fail-open behavior).

The same flags apply to `--stream` mode:
- `--past-text auto` (default) enables text-prefix conditioning in streaming mode.
- `--past-text yes` forces conditioning on.
- `--past-text no` forces conditioning off.

```bash
# 20-second segments with default segment-cutting silence search window
./qwen_asr -d qwen3-asr-0.6b -i lecture.wav -S 20

# Same segmentation with past-text conditioning (auto boundary cleanup)
./qwen_asr -d qwen3-asr-0.6b -i lecture.wav -S 20 --past-text yes
```

### Silence Skipping (`--skip-silence`)

```bash
./qwen_asr -d qwen3-asr-0.6b -i recording.wav --skip-silence
```

When enabled, long silent spans are removed before transcription (short pauses are kept). This reduces compute on recordings with long dead-air sections.

Tradeoffs:
- Useful for podcasts, meetings, and captured audio with long pauses.
- Can slightly alter timing-sensitive boundary behavior and punctuation.
- Disabled by default to preserve baseline behavior.

### Language (`--language`)

```bash
./qwen_asr -d qwen3-asr-0.6b -i audio.wav --language Italian
```

Forces the model to transcribe (or translate) into the specified language by adding language tokens into the decoder prompt. If omitted, language is usually auto-detected from the audio. The 0.6b and 1.7b models can behave differently, with the smaller model being more likely to translate when forced into a different language than the source audio.

Example of this behavior:

```
$ ./qwen_asr -d qwen3-asr-0.6b -i samples/jfk.wav --language Italian --silent
E così, miei amici americani, chiedete non ciò che il vostro paese può
fare per voi, chiedete ciò che voi possiate fare per il vostro paese.
```

Supported languages: Chinese, English, Cantonese, Arabic, German, French, Spanish, Portuguese, Indonesian, Italian, Korean, Russian, Thai, Vietnamese, Japanese, Turkish, Hindi, Malay, Dutch, Swedish, Danish, Finnish, Polish, Czech, Filipino, Persian, Greek, Romanian, Hungarian, Macedonian.

### System Prompt (`--prompt`)

```bash
./qwen_asr -d qwen3-asr-0.6b -i audio.wav --prompt "Preserve spelling: PostgreSQL, Redis, CUDA"
```

Injects a system prompt into the model's chat template. This slightly biases the model without changing the fundamental transcription behavior. Useful for:

- **Preserving technical terms**: `--prompt "Preserve spelling: PostgreSQL, CUDA, FFmpeg"`
- **Domain context**: `--prompt "This is a medical consultation about cardiology."`
- **Style hints**: `--prompt "Use formal punctuation and capitalization."`

The prompt is encoded once and prepended to every segment/chunk. Its effect is subtle — it nudges the model's token probabilities rather than forcing specific output.

### Reading Audio from Stdin

The **`--stdin` flag** reads audio from standard input. The format is auto-detected: if the data starts with a RIFF header it is parsed as WAV, otherwise it is treated as **raw signed 16-bit little-endian, 16 kHz, mono** (`s16le`).

```bash
# Transcribe an MP3 file
ffmpeg -i podcast.mp3 -f s16le -ar 16000 -ac 1 - 2>/dev/null | \
    ./qwen_asr -d qwen3-asr-0.6b --stdin

# Pipe a WAV directly
cat recording.wav | ./qwen_asr -d qwen3-asr-0.6b --stdin

# Live transcription of a web radio stream
curl -sL http://stream.live.vc.bbcmedia.co.uk/bbc_world_service | \
    ffmpeg -i pipe:0 -ar 16000 -ac 1 -f s16le pipe:1 2>/dev/null | \
    ./qwen_asr -d qwen3-asr-0.6b --stdin --stream --monitor

# Same flow, but keep WAV framing on stdin
curl -sL http://stream.live.vc.bbcmedia.co.uk/bbc_world_service | \
    ffmpeg -i pipe:0 -ar 16000 -ac 1 -f wav pipe:1 2>/dev/null | \
    ./qwen_asr -d qwen3-asr-0.6b --stdin --stream --monitor
```

To convert files to WAV format, just use ffmpeg:

    ffmpeg -i input.ogg output.wav

There are two example WAV files under the `samples/` directory.

### C API

The library exposes a simple callback-based API:

**Offline transcription:**

```c
#include "qwen_asr.h"

qwen_ctx_t *ctx = qwen_load("qwen3-asr-0.6b");

/* Optional: set a callback to receive tokens as they are decoded */
qwen_set_token_callback(ctx, my_token_handler, userdata);

/* Optional: force language or set system prompt */
qwen_set_force_language(ctx, "Italian");
qwen_set_prompt(ctx, "Preserve spelling: PostgreSQL, Redis");

/* Transcribe — returns malloc'd string */
char *text = qwen_transcribe(ctx, "audio.wav");
printf("%s\n", text);
free(text);

/* Or from raw samples */
char *text2 = qwen_transcribe_audio(ctx, samples, n_samples);

qwen_free(ctx);
```

**Streaming transcription:**

```c
/* Load audio first */
float *samples = qwen_load_wav("long_audio.wav", &n_samples);

/* Stream-transcribe with prefix rollback */
qwen_set_token_callback(ctx, my_token_handler, userdata);
char *text = qwen_transcribe_stream(ctx, samples, n_samples);
free(text);
free(samples);
```

Tokens are emitted via the callback as they become "fixed" (past the rollback window). The returned string contains the full concatenated text.

## Regression Tests

The repository includes `asr_regression.py` (repo root), a stdlib-only regression harness.
It scans `samples/**/*.wav` recursively:
- quality regression runs on WAV files that already have a sibling `.txt` reference
- focused checks (segmented conditioning, streaming, stream-cache equivalence) use fixed targets

Generate references (using the larger model and full-context decode):

```bash
./asr_regression.py --generate-missing \
    --binary ./qwen_asr --model-dir qwen3-asr-1.7b
```

Run regression checks:

```bash
./asr_regression.py \
    --binary ./qwen_asr --model-dir qwen3-asr-1.7b
```

Or run the default regression profile via make:

```bash
make test
```

Streaming cache equivalence regression (cache on vs off):

```bash
./asr_regression.py --stream-cache-check-only \
    --binary ./qwen_asr --stream-cache-model-dir qwen3-asr-0.6b
```

Or via make:

```bash
make test-stream-cache
```

Output format:
- Each sample starts with a progress line: `START i/N`.
- Live model text is shown while that sample is transcribed.
- The sample closes with `DONE: OK i/N` (only `OK` is green) or `DONE: FAIL i/N` (status in red).

Example:

```text
[START 1/22] jfk.wav ...
And so, my fellow Americans, ask not what your country can do for you...
[DONE: OK 1/22] jfk.wav | exact 0/108 (0.000) | norm 0/104 (0.000) | 2.6s
```

Per sample, the tool reports two distances:
- `exact`: character-level Levenshtein distance on raw text.
- `norm`: character-level Levenshtein distance after normalization
  (punctuation -> spaces, lowercase, whitespace collapsed).

## Building

```bash
make blas       # BLAS acceleration (Accelerate on macOS, OpenBLAS on Linux)
make rocm       # AMD GPU acceleration (Linux, HIP + rocBLAS)
make cuda       # NVIDIA GPU acceleration (CUDA + cuBLAS, DGX Spark by default)
make test       # Run regression checks (requires built binary + model files)
make test-stream-cache  # Check stream cache on/off equivalence
make test-rocm  # ROCm kernel checks, no model files needed
make test-rocm-asr  # ROCm transcription checks, needs qwen3-asr-0.6b
make test-cuda  # CUDA kernel checks, no model files needed
make test-cuda-asr  # CUDA transcription checks, needs qwen3-asr-0.6b
make clean      # Clean build artifacts
```

For the Linux CPU build, install OpenBLAS first:
```bash
# Ubuntu/Debian
sudo apt install libopenblas-dev

# Fedora
sudo dnf install openblas-devel
```

### ROCm / AMD GPUs

The ROCm backend uses the same C model loader, tokenizer, audio preprocessing,
segmentation, streaming code, and model files as the CPU backend. Encoder and
decoder computation runs on the GPU through HIP C++ kernels and rocBLAS.
There is no Python inference runtime or OpenBLAS dependency for this build.

It has been tested on Linux with **ROCm 7.1 and Radeon 8060S / Strix Halo
(`gfx1151`)**. The kernels require a wave32 GPU; wave64 devices are rejected.
Other wave32 AMD GPUs need the appropriate `GPU_ARCH` and validation on that
hardware. This is not a claim of support for every ROCm device.

Install a working ROCm environment with `hipcc`, HIP headers/runtime, and
rocBLAS, and ensure your user can access the GPU device (normally through the
`render` group). Then:

```bash
make rocm -j8
./download_model.sh --model large  # skip if the model is already downloaded
./qwen_asr_rocm -d qwen3-asr-1.7b -i samples/jfk.wav

# Same stdin and streaming interface as the CPU binary
ffmpeg -i audio.mp3 -f s16le -ar 16000 -ac 1 - 2>/dev/null | \
    ./qwen_asr_rocm -d qwen3-asr-1.7b --stdin
./qwen_asr_rocm -d qwen3-asr-0.6b -i audio.wav --stream
```

`make rocm` builds **`qwen_asr_rocm`**, leaving the CPU `qwen_asr` binary intact.
ROCm objects live under `build/rocm/<architecture>/`, separate from CPU objects.
Build settings can be overridden, for example:

```bash
make rocm ROCM_PATH=/opt/rocm GPU_ARCH=gfx1151
# HIPCC, HIPFLAGS, and ROCM_CFLAGS can also be overridden.
```

The ROCm default is `-S 20`, which bounds attention and workspace for long
recordings. Use `-S 0` for full-audio decoding. As with the CPU build,
`--stream --silent` on a file runs final refinement instead of incremental
streaming. Normal transcription emits tokens immediately to stdout; diagnostics
go to stderr and `--silent` suppresses them.

**Precision:** `--precision q8` is the default. Token-by-token decoder matrix
operations use weights quantized to signed 8-bit integers, with one FP32 scale
per 32 weights. Encoder and batched decoder prefill operations use FP16 weights;
accumulation, activations, normalization, attention, and KV caches use FP32.
Both FP16 and Q8 weight copies are retained, so Q8 saves decoding bandwidth
rather than total weight memory. Use `--precision fp16` to disable Q8 decoding
for comparison. Neither mode guarantees identical greedy output to the CPU
backend on every recording.

HIP graphs reduce the launch overhead of each generated token. For debugging,
`QWEN_ROCM_GRAPHS=0` uses direct kernel launches, and `QWEN_ROCM_Q8=0` selects
FP16 decoding. An explicit `--precision` overrides the Q8 environment setting.

No background process is started: each CLI invocation loads its model and
releases its CPU/GPU allocations on exit. For 20-second segmentation on Strix
Halo, the 1.7B Q8 implementation used about **6 GiB of GPU allocations**, plus
1.3 GiB of CPU allocations and roughly 4 GiB of reclaimable mapped file pages
after warmup. CPU and GPU share RAM on this host. Longer full-context inputs
need additional workspace.

Validate the backend on a ROCm host:

```bash
make test-rocm
make test-rocm-asr ROCM_MODEL_DIR=qwen3-asr-0.6b
./asr_regression.py --stream-cache-check-only \
    --binary ./qwen_asr_rocm --stream-cache-model-dir qwen3-asr-0.6b
```

Kernel tests compare against independent scalar calculations and exercise
attention masks, cache offsets, and KV growth with graph invalidation.
Transcription checks cover file/stdin input, both precisions, graph replay,
language forcing, segmentation, and incremental streaming.

The wave32 GEMV and normalization/RoPE kernels adapt techniques from
[ds4.c](https://github.com/antirez/ds4); its MIT notice is retained in
[rocm/LICENSE.ds4](rocm/LICENSE.ds4).

### CUDA / NVIDIA GPUs

The native CUDA backend runs the encoder and decoder on NVIDIA GPUs using
CUDA C++ kernels and cuBLAS. It shares the C model loader, tokenizer, audio
preprocessing, segmentation, and streaming implementation with the CPU and
ROCm builds. It uses the same unmodified safetensors models; Python, PyTorch,
cuDNN, and HIP are not needed for inference.

The default target is **DGX Spark / NVIDIA GB10 (`sm_121`)**, tested on Linux
AArch64 with CUDA Toolkit 13.0. Install the CUDA toolkit and a compatible driver,
then build and run:

```bash
make cuda -j8
./download_model.sh --model large  # skip if already downloaded
./qwen_asr_cuda -d qwen3-asr-1.7b -i samples/jfk.wav

# Both models, stdin, and streaming use the existing interface
./qwen_asr_cuda -d qwen3-asr-0.6b -i audio.wav --stream
ffmpeg -i audio.mp3 -f s16le -ar 16000 -ac 1 - 2>/dev/null | \
    ./qwen_asr_cuda -d qwen3-asr-1.7b --stdin
```

`make cuda` produces **`qwen_asr_cuda`**, independently of the CPU and ROCm
binaries. CUDA objects are stored in `build/cuda/<architecture>/`. The toolkit
path, compiler, flags, and GPU architecture can be overridden:

```bash
make cuda CUDA_PATH=/usr/local/cuda CUDA_ARCH=sm_121
# NVCC, NVCCFLAGS, and CUDA_CFLAGS can also be overridden.
```

For another NVIDIA GPU, set `CUDA_ARCH` to its architecture and run the tests
below; other devices have not been validated here. DGX Spark requires a toolkit
that supports `sm_121`, such as CUDA 13.0. NVIDIA's
[Spark porting guide](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/compilation.html)
describes the platform's compilation requirements.

Like ROCm, CUDA defaults to **20-second segments** (`-S 20`) and **Q8 decoding**
(`--precision q8`). Use `-S 0` for full-audio decoding, or `--precision fp16`
for an unquantized decoder comparison. Q8 stores decoder weights as signed
8-bit values with an FP32 scale per 32 weights. Batched matrix operations use
FP16 weights with FP32 accumulation; activations, attention, and KV caches stay
FP32. Both weight formats are retained in Q8 mode, so the benefit is lower
bandwidth during token generation, not a smaller total weight allocation.
GPU weights alone occupy about 2.08 GiB for 0.6B and 5.60 GiB for 1.7B in Q8
mode; working buffers, runtime allocations, CPU weights, and mapped model
files add to that. DGX Spark uses shared CPU/GPU system memory.

CUDA graphs replay the single-token decoder. Set `QWEN_CUDA_GRAPHS=0` to use
direct launches for comparison, or `QWEN_CUDA_Q8=0` for FP16 decoding. An
explicit `--precision` overrides the Q8 environment setting. Use
`CUDA_VISIBLE_DEVICES` to select a GPU. The CLI loads a model for each run and
releases its allocations on exit; it does not install or start a resident
service.

Validate kernels and transcription on the CUDA host:

```bash
make test-cuda
make test-cuda-asr CUDA_MODEL_DIR=qwen3-asr-0.6b
make test-cuda-asr CUDA_MODEL_DIR=qwen3-asr-1.7b
./asr_regression.py --stream-cache-check-only \
    --binary ./qwen_asr_cuda --stream-cache-model-dir qwen3-asr-0.6b

# Optional NVIDIA memory and synchronization checks
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 1 \
    ./build/cuda/sm_121/kernel-test
/usr/local/cuda/bin/compute-sanitizer --tool racecheck --error-exitcode 1 \
    ./build/cuda/sm_121/kernel-test
/usr/local/cuda/bin/compute-sanitizer --tool synccheck --error-exitcode 1 \
    ./build/cuda/sm_121/kernel-test
```

On the tested Spark (driver 580.173.02, CUDA 13.0), Q8 transcription times
were as follows. These are medians of three runs after one warmup, with the
model already loaded, using 20-second segments and eight CPU threads. They
exclude model loading and WAV reading:

| Recording | Duration | 0.6B | 1.7B |
| --- | ---: | ---: | ---: |
| JFK | 11.0 s | 0.264 s | 0.452 s |
| Short test speech | 3.64 s | 0.159 s | 0.278 s |
| Don't be afraid of me | 45.0 s | 0.626 s | 0.986 s |
| Another broadcast | 119.3 s | 2.498 s | 4.402 s |

**Fresh-process latency is higher:** the 11-second JFK clip took a median
2.44 seconds with 0.6B and 6.01 seconds with 1.7B across three CLI invocations,
including model loading and process teardown, with files cached by the OS.
The CLI retains the load-per-invocation behavior.

Both precisions matched the CPU reference's normalized transcripts on 24
FLEURS recordings across eight languages (Arabic, Chinese, German, Spanish,
French, Italian, Japanese, and Russian). This is a small regression set, not
an accuracy benchmark representative of all supported languages.

The CUDA kernels are ported from this project's ROCm backend, with native CUDA
warp synchronization and cuBLAS calls. The original ds4.c attribution also
applies; see [rocm/LICENSE.ds4](rocm/LICENSE.ds4).

## How Fast Is It?

Benchmarks were recomputed on **Apple M3 Max** (128GB RAM) with `make blas` (single run per row).
`Inference`/`Audio` are from program summary. `wall` includes model-load and process overhead.

### Offline Mode (Full + Segmented)

| Setup | Audio | 0.6B (`Inference`, realtime, wall) | 1.7B (`Inference`, realtime, wall) |
|-------|-------|-------------------------------------|-------------------------------------|
| `samples/jfk.wav -S 0` | `11.0s` | `1.4s`, `7.99x`, `1.83s` | `2.6s`, `4.29x`, `3.17s` |
| `45s_dont_be_afraid_of_me.wav -S 30 -W 3` | `45.0s` | `3.4s`, `13.38x`, `3.64s` | `5.9s`, `7.63x`, `6.52s` |
| `89s_ill_come_back_down_as_soon_as.wav -S 30 -W 3` | `88.9s` | `13.1s`, `6.78x`, `13.39s` | `26.6s`, `3.34x`, `27.22s` |

### Streaming Mode (45s clip, interactive `--stream`)

| Setup | 0.6B (`Inference`, realtime) | 1.7B (`Inference`, realtime) |
|-------|--------------------------------|--------------------------------|
| cache ON (default, with prefill KV reuse) | `9.6s`, `4.69x` | `17.7s`, `2.54x` |
| cache OFF (`QWEN_STREAM_NO_ENC_CACHE=1`) | `22.0s`, `2.05x` | `34.3s`, `1.31x` |

Outputs were exact matches between cache ON/OFF in this benchmark.

### Streaming Non-Interactive Path (`--stream --silent`, file input)

For file input, `--stream --silent` does not run interactive chunk commits: it executes one direct final refinement pass and prints only the final transcript. This is useful for quiet batch runs, but its timing is not directly comparable to interactive `--stream`.

### Long-file Example (`/tmp/nirvana.wav`, 135s, 0.6B)

| Mode | Result |
|------|--------|
| `--stream` | `141.3s` inference (`0.96x` realtime) |
| offline segmented mode (`-S 30` in this measurement) | `14.0s` inference (`9.64x` realtime) |

## Model Architecture

Qwen3-ASR is a speech-to-text model available in 0.6B and 1.7B parameter variants:

**Pipeline:**
```
WAV -> 16kHz -> Mel Spectrogram -> Conv2D Stem -> Encoder -> Projection -> Decoder -> Tokens
```

| Component | Architecture |
|-----------|-------------|
| Conv2D Stem | 3 layers (480 channels, 3x3, stride 2), 8x time downsampling |
| Audio Encoder | Transformer with bidirectional windowed attention, sinusoidal PE |
| Projection | Linear -> GELU -> Linear (encoder dim -> decoder dim) |
| LLM Decoder | Qwen3 with GQA, per-head Q/K RMSNorm, NeoX split-half RoPE, SwiGLU |

| Parameter | 0.6B | 1.7B |
|-----------|------|------|
| Encoder layers | 18 | 24 |
| Encoder dim | 896 | 1024 |
| Decoder layers | 28 | 28 |
| Decoder dim | 1024 | 2048 |
| GQA heads | 16 Q / 8 KV | 16 Q / 8 KV |
| Vocab size | 151,936 | 151,936 |
| Weight format | BF16 | BF16 |
| Supported languages | 30 (see `--language`) |

## CPU Memory Requirements

These measurements describe the CPU backend. See [ROCm / AMD GPUs](#rocm--amd-gpus)
and [CUDA / NVIDIA GPUs](#cuda--nvidia-gpus) for GPU allocation behavior.

Memory usage has two parts:
- Static model footprint (allocated once at load time).
- Runtime footprint (depends on input length and decoding mode).

### Static Footprint (Model Load)

These numbers come from the current implementation and model files:
- Safetensors are memory-mapped.
- Encoder BF16 weights are converted to F32 and kept in heap memory.
- Decoder builds a fused gate/up matrix copy for faster decode.

| Component | 0.6B | 1.7B |
|-----------|------|------|
| safetensors mmap files | 1.747 GiB | 4.376 GiB |
| encoder copied F32 weights | 0.694 GiB | 1.183 GiB |
| decoder extra heap (fused + norms) | 0.328 GiB | 1.313 GiB |
| static total (theoretical) | 2.770 GiB | 6.871 GiB |

### Runtime Scaling (Why `-S 0` Grows)

For one segment, dominant runtime allocations scale with sequence length:

- `mel_frames ~= floor(audio_seconds * 100)`
- `enc_tokens ~= 13 * floor(mel_frames / 100) + ceil((mel_frames % 100) / 8)`
- `total_seq = enc_tokens + 15` (plus prompt/language/past-text tokens if used)
- `prefill_len = total_seq - 1`
- `pref_cap = next_pow2(prefill_len)`
- `kv_max = prefill_len + 1024`

Main growing buffers:
- KV cache: `2 * 28 * kv_max * 1024 * 4` bytes
- Prefill buffers:
  - 0.6B: `77,824 * pref_cap` bytes
  - 1.7B: `131,072 * pref_cap` bytes

Implications:
- `-S 0` (full-audio decode) lets `total_seq` grow with audio duration, so peak memory increases with file length.
- `-S 20` (or any segmented mode) bounds per-segment `total_seq`, so memory stays nearly flat as file length increases.
- Enabling `--past-text yes` adds previous text tokens to each segment/chunk prompt and can increase memory again.

### Measured Peak RSS (`--silent`)

Measured on Apple M3 Max using the current codebase:

| Audio length | 0.6B `-S 0` | 0.6B `-S 20` | 1.7B `-S 0` | 1.7B `-S 20` |
|--------------|-------------:|-------------:|-------------:|-------------:|
| 10.000s | 2.695 GiB | 2.688 GiB | 6.573 GiB | 6.573 GiB |
| 45.000s | 2.861 GiB | 2.757 GiB | 6.783 GiB | 6.700 GiB |
| 88.890s | 3.173 GiB | 2.815 GiB | 7.113 GiB | 6.742 GiB |
| 119.262s | 3.254 GiB | 2.789 GiB | 7.288 GiB | 6.706 GiB |

In practice:
- For long files, segmented mode is safer for both speed and memory.
- Default is `-S 0`, so for large files explicitly pick segmented mode (`-S 20` or `-S 30`).
- Use `-S 0` mainly for short files where full-context quality is worth the extra memory/time.

## License

MIT. See also [rocm/LICENSE.ds4](rocm/LICENSE.ds4) for the adapted ROCm kernels.
