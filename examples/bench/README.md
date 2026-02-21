# Qwen ASR Bench Example

Benchmarks the Qwen3-ASR inference engine on your hardware. Works with synthetic silence (no audio file required) or a real WAV file.

## Build

```bash
# From root
make bench
```

The resulting binary is `./qwen_asr_bench` in the project root.

## Usage

```
./qwen_asr_bench -d <model_dir> [options]
```

Options:

```
./qwen_asr_bench -h

Required:
  -d DIR        Model directory (*.safetensors + vocab.json)

Options:
  -t N          Number of threads (default: auto)
  -n N          Number of benchmark runs (default: 5)
  -i FILE       Audio file to benchmark with (default: synthetic silence)
  -s N          Synthetic audio length in seconds (default: 5)
                Ignored when -i is given.
  -w N          What to benchmark:
                  0 (default) — full pipeline (encode + decode)
                  1           — encoder only
```

## Audio source

### Synthetic silence (default)

When `-i` is not given, the bench tool generates `N` seconds of silence
internally via `calloc`. No audio file needed. Useful for a pure
hardware/kernel throughput baseline.

### Real audio (`-i FILE`)

When `-i` is given, the file is loaded once with `qwen_load_wav()` before
the measurement loop. The same audio is fed into every run, so timing
reflects the model's behaviour on that specific content. The mel spectrogram is computed once (for `-w 1`) or recomputed inside each `qwen_transcribe_audio()` call (for `-w 0`), matching normal inference behaviour.

Real audio is more representative for decode-heavy profiles because the model generates actual tokens instead of the near-zero output it produces on silence.

## Modes

### `-w 0` — Full pipeline (default)

Calls `qwen_transcribe_audio()` on the audio source and reports timing statistics across runs. Exercises mel spectrogram computation, the audio encoder, decoder prefill, and autoregressive token generation.

One warm-up run is executed before measurement (not included in the stats). The KV cache is reset to zero between measured runs to ensure each run starts from a clean state.

```
$ ./qwen_asr_bench -d qwen3-asr-1.7b -n 5 -s 5

system_info: n_threads = 8 / 8

Loading model from qwen3-asr-1.7b ...
Mode: full pipeline  |  5 run(s)  |  5.0 s  [synthetic silence]

  warmup ...
  run 1/5:  total=  1842 ms  enc=  1204 ms  dec=   638 ms  tokens=0  rt=0.37x
  run 2/5:  total=  1790 ms  enc=  1183 ms  dec=   607 ms  tokens=0  rt=0.36x
  ...

               min      mean       max
total        1788.0    1812.0    1842.0  ms
encode       1181.0    1194.0    1204.0  ms
decode        603.0     614.0     638.0  ms
rt_factor       0.36      0.36      0.37  x RT
```

With a real file:

```
$ ./qwen_asr_bench -d qwen3-asr-1.7b -n 5 -i samples/jfk.wav

Mode: full pipeline  |  5 run(s)  |  11.0 s  [samples/jfk.wav]

  warmup ...
  run 1/5:  total=  3210 ms  enc=  2190 ms  dec=  1020 ms  tokens=32  rt=0.29x
  ...
```

`rt_factor` is `total_ms / audio_duration_ms`. Values below `1.0` mean the
model runs faster than real-time.

### `-w 1` — Encoder only

Calls `qwen_encoder_forward()` on a mel spectrogram, preceded by one warm-up run. Reports total encoder time and time per transformer layer. Useful for isolating encoder performance independently of decoding.

With synthetic silence, a zeroed mel is constructed directly. With `-i`, the mel spectrogram is computed once from the WAV before the measurement loop starts.

```
$ ./qwen_asr_bench -d qwen3-asr-1.7b -w 1 -n 5 -s 5

Mode: encoder only  |  5 run(s)  |  500 frames (5.0 s)  [synthetic silence]

  run 1/5:  enc=  1198 ms  seq_len=50
  ...

               min      mean       max
encode       1181.0    1192.0    1198.0  ms
per layer      49.2      49.7      49.9  ms/layer
```

## Tips

Run with `-n 10` or more for stable mean values.

Use real audio for a decode-heavy profile (silence generates almost no tokens):

```bash
./qwen_asr_bench -d qwen3-asr-1.7b -n 10 -i samples/jfk.wav
```

Compare 0.6B vs 1.7B to see the size/speed trade-off:

```bash
./qwen_asr_bench -d qwen3-asr-0.6b -n 10
./qwen_asr_bench -d qwen3-asr-1.7b -n 10
```

Vary `-t` to find the thread count sweet spot for your CPU:

```bash
./qwen_asr_bench -d qwen3-asr-1.7b -t 4 -n 5
./qwen_asr_bench -d qwen3-asr-1.7b -t 8 -n 5
```
