# Qwen3-ASR — Model Reference

Models: `Qwen/Qwen3-ASR-1.7B` and `Qwen/Qwen3-ASR-0.6B`

This document describes the model architecture, weight format, tokenizer layout,
and inference algorithm needed to implement Qwen3-ASR from scratch.
The Python reference implementation (`python_simple_implementation.py`) is the
executable version of this document.

---

## Architecture Overview

Qwen3-ASR is a speech-to-text model with two main components:
- **Audio Encoder (AuT)**: Conv2D downsampling + transformer encoder
- **LLM Decoder (Qwen3)**: Standard Qwen3 transformer with Q/K norms and MRoPE

**Pipeline:**
```
WAV → 16kHz → Mel Spectrogram → Conv2D ×3 (8× downsample) → Transformer Encoder → Projector → Qwen3 Decoder → Tokens
```

### Model Variants

| Parameter | 1.7B | 0.6B |
|-----------|------|------|
| **Encoder d_model** | 1024 | 896 |
| **Encoder layers** | 24 | 18 |
| **Encoder heads** | 16 | 14 |
| **Encoder FFN dim** | 4096 | 3584 |
| **Encoder output_dim** | 2048 | 1024 |
| **Decoder hidden_size** | 2048 | 1024 |
| **Decoder layers** | 28 | 28 |
| **Decoder heads** | 16 | 16 |
| **Decoder KV heads** | 8 | 8 |
| **Decoder head_dim** | 128 | 128 |
| **Decoder intermediate** | 6144 | 3072 |
| **Vocab size** | 151,936 | 151,936 |

---

## Audio Preprocessing

| Parameter | Value |
|-----------|-------|
| Sample rate | 16000 Hz |
| Mel bins | 128 |
| Hop length | 160 samples (10ms) |
| Window size (n_fft) | 400 samples (25ms) |
| Frame rate | 100 Hz (before downsampling) |
| Token rate | 12.5 Hz (after 8× conv downsample) |

**Exact mel computation** (WhisperFeatureExtractor):
1. Window: `hann(window_size=400)`
2. STFT: `torch.stft(audio, n_fft=400, hop_length=160, window=window, return_complex=True)`
3. Power: `magnitudes = stft[..., :-1].abs() ** 2` (drops last frame)
4. Mel filter bank: Slaney-style, 128 bins, 0-8000 Hz
5. `mel_spec = mel_filters.T @ magnitudes`
6. `log_spec = log10(clamp(mel_spec, min=1e-10))`
7. Dynamic range: `log_spec = max(log_spec, log_spec.max() - 8.0)`
8. Normalize: `log_spec = (log_spec + 4.0) / 4.0`

Note: Unlike Voxtral which uses a fixed `global_log_mel_max=1.5`, Qwen3-ASR uses
the dynamic maximum of the spectrogram for clamping.

**CRITICAL: Per-chunk convolution.** The encoder does NOT process the entire mel
spectrogram at once. It splits the mel into chunks of `n_window*2 = 100` frames
and applies Conv2D independently per chunk. Each chunk of 100 frames produces
13 output tokens. The mel is NOT padded to 3000 frames.

**CRITICAL: Windowed attention.** The encoder uses windowed attention where tokens
can only attend within windows of `tokens_per_chunk * (n_window_infer / chunk_size)`
= `13 * (800/100)` = 104 tokens. For audio longer than ~8 seconds, this creates
multiple attention windows. Tokens in different windows cannot attend to each other.

Per the paper (arXiv:2601.21337), the encoder uses "dynamic attention windows
ranging from 1s to 8s" during training. At inference, `n_window_infer=800`
(8 seconds) is used as the fixed window size.

**CRITICAL: Per-chunk position embeddings.** Sinusoidal position embeddings are
applied per-chunk (each chunk starts from position 0), not globally.

---

## Audio Encoder

### Conv2D Stem (3 layers, 8× total downsampling)

Three Conv2D layers, each with stride=2 in both frequency and time dimensions:
```
conv2d1: Conv2d(in=1, out=480, kernel=3×3, stride=2, padding=1) → GELU
conv2d2: Conv2d(in=480, out=480, kernel=3×3, stride=2, padding=1) → GELU
conv2d3: Conv2d(in=480, out=480, kernel=3×3, stride=2, padding=1) → GELU
```

Input shape: `[1, 1, 128, T]` (batch, channel, mel_bins, time_frames)

After 3 convolutions, frequency dimension: 128 → 64 → 32 → 16

Output is reshaped: `[1, 480, 16, T/8]` → permute → `[1, T/8, 480×16]` = `[1, T/8, 7680]`

Then projected: `conv_out: Linear(7680 → d_model, no bias)`

### Sinusoidal Position Embeddings

Standard sinusoidal embeddings (not RoPE) added after conv projection:
```python
log_timescale_increment = log(10000) / (d_model/2 - 1)
inv_timescales = exp(-arange(d_model/2) * log_timescale_increment)
pe = concat(sin(pos * inv_timescales), cos(pos * inv_timescales))  # [seq, d_model]
```

### Transformer Encoder Layers

| Parameter | 1.7B | 0.6B |
|-----------|------|------|
| d_model | 1024 | 896 |
| n_layers | 24 | 18 |
| n_heads | 16 | 14 |
| head_dim | 64 | 64 |
| FFN dim | 4096 | 3584 |
| Norm | LayerNorm (with bias) | LayerNorm (with bias) |
| Attention | Full (bidirectional) | Full (bidirectional) |
| Biases | YES (all Q,K,V,Out,FC1,FC2 + norms) | YES |

Per-layer computation:
```
residual = h
h_norm = LayerNorm(h, self_attn_layer_norm)
q = h_norm @ Wq + bq
k = h_norm @ Wk + bk
v = h_norm @ Wv + bv
attn_out = full_attention(q, k, v)  # bidirectional, no mask
h = residual + (attn_out @ Wo + bo)

residual = h
h_norm = LayerNorm(h, final_layer_norm)
ffn_out = GELU(h_norm @ W_fc1 + b_fc1) @ W_fc2 + b_fc2
h = residual + ffn_out
```

### Projection (proj1 + proj2)

After the final encoder LayerNorm (ln_post):
```
h = LayerNorm(h, ln_post)     # with bias
h = GELU(h @ proj1 + b_proj1) # d_model → d_model
h = h @ proj2 + b_proj2       # d_model → output_dim (= decoder hidden_size)
```

For 1.7B: 1024 → 1024 → 2048
For 0.6B: 896 → 896 → 1024

---

## LLM Decoder (Qwen3)

| Parameter | 1.7B | 0.6B |
|-----------|------|------|
| hidden_size | 2048 | 1024 |
| n_layers | 28 | 28 |
| n_heads | 16 | 16 |
| n_kv_heads | 8 (GQA 2:1) | 8 (GQA 2:1) |
| head_dim | 128 | 128 |
| intermediate_size | 6144 | 3072 |
| Norm | RMSNorm (eps=1e-6) | RMSNorm (eps=1e-6) |
| Position | RoPE (theta=1e6, NeoX-style) | RoPE (theta=1e6, NeoX-style) |
| Attention | causal | causal |
| Biases | NO (none in decoder) | NO |
| Vocab size | 151,936 | 151,936 |
| Tied embeddings | yes (embed_tokens == lm_head) | yes |

### Key feature: Q/K RMSNorm

The decoder applies per-head RMSNorm on Q and K **after** linear projection
but **before** RoPE:
```python
q = q_proj(h_norm)                          # [seq, n_heads * head_dim]
q = q.view(seq, n_heads, head_dim)          # [seq, 16, 128]
q = RMSNorm_per_head(q, q_norm_weight)      # normalize each head independently
# Then apply RoPE
```

The `q_norm` and `k_norm` weights have shape `[head_dim]` = `[128]`.

### RoPE (NeoX/split-half style)

The decoder uses standard NeoX-style RoPE (rotate_half):
```python
inv_freq = 1.0 / (theta ** (arange(0, head_dim, 2) / head_dim))  # [64]
angles = positions * inv_freq  # [seq, 64]
emb = cat(angles, angles)     # [seq, 128] (duplicate for full head_dim)
cos, sin = emb.cos(), emb.sin()

# rotate_half: x1 = x[..., :64], x2 = x[..., 64:]
# result = x * cos + cat(-x2, x1) * sin
```

Note: The config mentions MRoPE with `mrope_section=[24,20,20]` and `interleaved=True`.
For ASR (audio-only, no spatial dims), all three position dimensions are identical,
so MRoPE reduces to standard RoPE. The "interleaved" flag refers to how MRoPE
sections are mixed, not the per-pair rotation style.

### Decoder Forward Pass

Per-layer computation for hidden state `h` at positions `pos..pos+seq-1`:

1. **Input RMSNorm**: `x = RMSNorm(h, input_layernorm, eps=1e-6)`
2. **QKV projections (GQA)**:
   - `q = x @ Wq^T` → `[seq, n_heads×128]` → reshape `[seq, 16, 128]`
   - `k = x @ Wk^T` → `[seq, n_kv_heads×128]` → reshape `[seq, 8, 128]`
   - `v = x @ Wv^T` → `[seq, n_kv_heads×128]`
3. **Per-head Q/K RMSNorm** (eps=1e-6, weight shape [128])
4. **RoPE** on Q and K (NeoX style, theta=1e6)
5. **KV cache**: append K, V to per-layer cache
6. **Causal attention**: scale=1/sqrt(128), GQA repeat 2:1
7. **Output projection + residual**: `h = h + attn_out @ Wo^T`
8. **Post-attention RMSNorm**: `h_norm = RMSNorm(h, post_attention_layernorm, eps=1e-6)`
9. **SwiGLU MLP + residual**:
   - `gate = silu(h_norm @ W_gate^T)`
   - `up = h_norm @ W_up^T`
   - `h = h + (gate * up) @ W_down^T`

After last layer: `h = RMSNorm(h, norm.weight)`, then `logits = h @ lm_head^T`.

---

## Tokenizer (Qwen2 BPE)

### Special Token IDs
```
<|endoftext|>   = 151643  (pad token, EOS)
<|im_start|>    = 151644
<|im_end|>      = 151645  (EOS)
<|audio_start|> = 151669
<|audio_end|>   = 151670
<|audio_pad|>   = 151676  (placeholder for audio embeddings)
<asr_text>      = 151704  (marks start of transcription text)
```

EOS token IDs: `{151643, 151645}`

### Token Decoding

Uses GPT-2 style byte-level BPE from `vocab.json`. The vocabulary maps
byte-encoded strings to token IDs. Characters are encoded using the GPT-2
bytes-to-unicode mapping (printable ASCII + extended Latin-1, with remaining
bytes mapped to Unicode chars starting at U+0100).

To decode: look up token string in inverted vocab → convert each character
through reverse byte mapping → decode resulting bytes as UTF-8.

---

## Prompt Format

The prompt template for ASR:
```
<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|>×N<|audio_end|><|im_end|>\n<|im_start|>assistant\n
```

As token IDs:
```
PREFIX: [151644, 8948, 198, 151645, 198, 151644, 872, 198, 151669]
AUDIO:  [151676] × N_audio_tokens
SUFFIX: [151670, 151645, 198, 151644, 77091, 198]
```

Where `N_audio_tokens` equals the number of encoder output tokens (after 8× conv downsampling).

---

## Weight Format

### Files (1.7B)
- `model-00001-of-00002.safetensors` + `model-00002-of-00002.safetensors`: ~4.7 GB total, BF16
- `model.safetensors.index.json`: weight-to-shard mapping
- `vocab.json` + `merges.txt`: BPE tokenizer
- `config.json`: model configuration

### Files (0.6B)
- `model.safetensors`: ~1.9 GB, BF16 (single file)
- Same tokenizer and config files

### Tensor Names

**Audio Encoder** (prefix: `thinker.audio_tower.`):
```
conv2d1.weight              [480, 1, 3, 3] + bias [480]
conv2d2.weight              [480, 480, 3, 3] + bias [480]
conv2d3.weight              [480, 480, 3, 3] + bias [480]
conv_out.weight             [d_model, 7680]  (no bias)
layers.{i}.self_attn.q_proj.weight   [d_model, d_model] + bias
layers.{i}.self_attn.k_proj.weight   [d_model, d_model] + bias
layers.{i}.self_attn.v_proj.weight   [d_model, d_model] + bias
layers.{i}.self_attn.out_proj.weight [d_model, d_model] + bias
layers.{i}.self_attn_layer_norm.weight [d_model] + bias
layers.{i}.fc1.weight       [ffn_dim, d_model] + bias
layers.{i}.fc2.weight       [d_model, ffn_dim] + bias
layers.{i}.final_layer_norm.weight [d_model] + bias
ln_post.weight              [d_model] + bias
proj1.weight                [d_model, d_model] + bias
proj2.weight                [output_dim, d_model] + bias
```

**Token Embeddings**:
```
thinker.model.embed_tokens.weight  [151936, hidden_size]
```

**LM Head** (tied with embeddings):
```
thinker.lm_head.weight             [151936, hidden_size]
```

**LLM Decoder** (prefix: `thinker.model.layers.{i}.`):
```
input_layernorm.weight              [hidden_size]
self_attn.q_proj.weight             [n_heads×128, hidden_size]
self_attn.k_proj.weight             [n_kv_heads×128, hidden_size]
self_attn.v_proj.weight             [n_kv_heads×128, hidden_size]
self_attn.o_proj.weight             [hidden_size, n_heads×128]
self_attn.q_norm.weight             [128]  (per-head RMSNorm)
self_attn.k_norm.weight             [128]  (per-head RMSNorm)
post_attention_layernorm.weight     [hidden_size]
mlp.gate_proj.weight                [intermediate, hidden_size]
mlp.up_proj.weight                  [intermediate, hidden_size]
mlp.down_proj.weight                [hidden_size, intermediate]
```
Plus `thinker.model.norm.weight [hidden_size]` (final norm). NO biases in decoder.

**Transformers-native repos** (`Qwen/Qwen3-ASR-{0.6B,1.7B}-hf`): same tensors,
shapes and dtypes, only renamed. The tokenizer ships as `tokenizer.json` (same
vocab and merges) instead of `vocab.json` + `merges.txt`.

| Original | `-hf` |
|---|---|
| `thinker.audio_tower.*` | `model.audio_tower.*` |
| `thinker.audio_tower.proj1.*` | `model.multi_modal_projector.linear_1.*` |
| `thinker.audio_tower.proj2.*` | `model.multi_modal_projector.linear_2.*` |
| `thinker.model.*` | `model.language_model.*` |
| `thinker.lm_head.weight` | (absent, tied to `embed_tokens`) |

Both layouts are detected automatically at load time.

---

## Decode Schedule

Unlike Voxtral which adds audio+text embeddings at every position, Qwen3-ASR
uses a **replacement** strategy: audio embeddings replace `<|audio_pad|>` token
embeddings at their positions.

### Algorithm

1. **Build prompt**: Construct input_ids with PREFIX + `<|audio_pad|>×N` + SUFFIX
2. **Embed tokens**: Look up all token embeddings via `embed_tokens`
3. **Replace audio positions**: Find positions where `input_ids == 151676` and
   replace those embeddings with the corresponding audio encoder outputs
4. **Prefill**: Feed the combined embedding sequence through the decoder to build
   KV caches. Generate first token from last prefill position.
5. **Autoregressive decode**: For each subsequent step, embed the previous token,
   feed through decoder, greedy argmax. Stop on EOS.

### Output Parsing

The model generates text in the format:
```
language English<asr_text>The actual transcription text.<|im_end|>
```

Parse by splitting on `<asr_text>` and taking the text after it.

---

## Long Audio Handling (Official Pipeline)

The official `qwen-asr` Python package handles long audio via two modes:
**non-streaming** (batch) and **streaming** (incremental).

### Non-Streaming Mode

Found in `qwen_asr/inference/utils.py` and `qwen_asr/inference/qwen3_asr.py`.

**Key constants:**
```python
SAMPLE_RATE = 16000
MAX_ASR_INPUT_SECONDS = 1200       # 20 minutes per chunk (!)
MAX_FORCE_ALIGN_INPUT_SECONDS = 180 # 3 minutes when using forced alignment
MIN_ASR_INPUT_SECONDS = 0.5        # minimum 500ms, zero-padded if shorter
```

**Segmentation function:**
```python
def split_audio_into_chunks(wav, sr, max_chunk_sec,
                            search_expand_sec=5.0,
                            min_window_ms=100.0):
```

**Algorithm:**
1. If audio <= `max_chunk_sec`, process as single chunk.
2. Otherwise, iteratively split at `max_chunk_sec` boundaries:
   - Search within ±5 seconds around the cut point.
   - Compute energy in 100ms sliding windows (using np.convolve).
   - Split at the lowest-energy sample (silence/pause).
3. Any chunk shorter than 0.5 seconds is zero-padded to 500ms.
4. Each chunk is processed independently through the full pipeline
   (mel → encoder → decoder), and text results are concatenated.

**Important:** The default max chunk is **1200 seconds (20 minutes)**, meaning
the official pipeline almost never segments audio. The 30-second `chunk_length`
in `preprocessor_config.json` is a legacy Whisper field and is NOT used.

No VAD (Voice Activity Detection) is used. The model handles silence natively.

### Streaming Mode

Found in `qwen_asr/inference/qwen3_asr.py` (lines 584-830).

**Key parameters (code defaults):**
- `chunk_size_sec`: 2.0 seconds (configurable)
- `unfixed_chunk_num`: 2 (first 2 chunks have no text prefix)
- `unfixed_token_num`: 5 (rollback: drop last 5 tokens from prefix)

**Paper parameters (arXiv:2601.21337):** "2-second chunk size, a 5-token
fallback, and keeping the last four chunks unfixed." The paper says 4 unfixed
chunks vs 2 in the code default — this may vary by evaluation setting.

**Algorithm:**
1. Audio arrives in arbitrary-sized pieces, buffered until `chunk_size_sec`
   samples accumulate.
2. On each trigger, ALL accumulated audio from the start is re-fed through
   the encoder (not just the new chunk).
3. The decoder prompt includes previous transcription as a text prefix:
   - First `unfixed_chunk_num` chunks: No prefix (cold start).
   - Later chunks: Previous decoded text minus last `unfixed_token_num`
     tokens is prepended. This "prefix rollback" reduces boundary jitter
     (the last few tokens may be unstable and get corrected with more
     context).
4. The decoder generates from where the prefix ends, producing new text.
5. Output is the full accumulated transcription.

**Critical detail:** Streaming re-processes the entire audio through the
encoder each time. This is O(n²) in audio length but bounded by the 2-second
chunk interval. The prefix rollback strategy is what makes this produce
coherent output despite incremental processing.

### Our C Implementation

Our C pipeline uses a simplified approach: energy-based silence splitting
(same algorithm as the official non-streaming mode) with configurable segment
size. Each segment is processed independently with a fresh KV cache. Tokens
are streamed to a callback as they are decoded.

This is a practical middle ground: shorter segments (e.g. 10 seconds) give
lower latency output, while longer segments (e.g. 30+ seconds) give the model
more context for better accuracy. The official pipeline's streaming mode with
prefix rollback would require significant additional complexity (re-encoding
all audio, managing text prefixes, token rollback).

---

## Paper Reference (arXiv:2601.21337)

Key facts from the Qwen3-ASR technical report:

**Training:** 4-stage pipeline:
1. AuT encoder pretraining on ~40M hours of pseudo-labeled ASR data
2. Omni pretraining with 3 trillion tokens (multi-modal)
3. ASR supervised fine-tuning with multilingual + streaming + context biasing data
4. Reinforcement learning (GSPO) on ~50k utterances

**Languages:** 30 languages + 22 Chinese dialects (52 total for ASR).

**Encoder:** Dynamic attention windows ranging 1s-8s during training.
At inference: 8s window (`n_window_infer=800`). AuT encoder is pretrained
separately, then integrated with the Qwen3 LLM.

**Performance (0.6B):** 92ms average time-to-first-token, RTF 0.064 at
128 concurrency. Can process ~2000 seconds of speech per second at scale.

**Benchmarks:** LibriSpeech WER 1.63-3.38 (1.7B), competitive with
GPT-4o-Transcribe and Gemini-2.5-Pro. WenetSpeech CER 4.97-5.88,
outperforming commercial APIs.
