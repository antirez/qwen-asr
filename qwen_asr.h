/*
 * qwen_asr.h - Qwen3-ASR Pure C Inference Engine
 *
 * Supports both Qwen3-ASR-1.7B and Qwen3-ASR-0.6B models.
 */

#ifndef QWEN_ASR_H
#define QWEN_ASR_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <pthread.h>

/* ========================================================================
 * Constants
 * ======================================================================== */

#define QWEN_SAMPLE_RATE      16000
#define QWEN_MEL_BINS         128
#define QWEN_HOP_LENGTH       160
#define QWEN_WINDOW_SIZE      400
#define QWEN_VOCAB_SIZE       151936

/* Maximum layer counts (for static array sizing) */
#define QWEN_MAX_ENC_LAYERS   24
#define QWEN_MAX_DEC_LAYERS   28

/* Special token IDs */
#define QWEN_TOKEN_IM_START     151644
#define QWEN_TOKEN_IM_END       151645
#define QWEN_TOKEN_ENDOFTEXT    151643
#define QWEN_TOKEN_AUDIO_START  151669
#define QWEN_TOKEN_AUDIO_END    151670
#define QWEN_TOKEN_AUDIO_PAD    151676
#define QWEN_TOKEN_ASR_TEXT     151704

/* Conv2D stem constants */
#define QWEN_CONV_HIDDEN      480
#define QWEN_CONV_KERNEL      3

/* ========================================================================
 * Model Configuration (populated from config.json)
 * ======================================================================== */

typedef struct {
    /* Audio encoder */
    int enc_d_model;           /* 1024 or 896 */
    int enc_layers;            /* 24 or 18 */
    int enc_heads;             /* 16 or 14 */
    int enc_head_dim;          /* 64 */
    int enc_ffn_dim;           /* 4096 or 3584 */
    int enc_output_dim;        /* 2048 or 1024 */
    int enc_n_window;          /* 50 */
    int enc_n_window_infer;    /* 800 */
    int enc_chunk_size;        /* n_window * 2 = 100 */
    int enc_conv_proj_dim;     /* CONV_HIDDEN * 16 = 7680 */

    /* LLM decoder */
    int dec_hidden;            /* 2048 or 1024 */
    int dec_layers;            /* 28 */
    int dec_heads;             /* 16 */
    int dec_kv_heads;          /* 8 */
    int dec_head_dim;          /* 128 */
    int dec_intermediate;      /* 6144 or 3072 */
    int vocab_size;            /* 151936 */
    float dec_rms_norm_eps;    /* 1e-6 */
    float dec_rope_theta;      /* 1e6 */
} qwen_config_t;

/* ========================================================================
 * Audio Encoder Layer
 * ======================================================================== */

typedef struct {
    /* Self-attention (ALL have biases) - pre-converted to f32 */
    float *wq_weight;          /* [d_model, d_model] */
    float *wq_bias;            /* [d_model] */
    float *wk_weight;          /* [d_model, d_model] */
    float *wk_bias;            /* [d_model] */
    float *wv_weight;          /* [d_model, d_model] */
    float *wv_bias;            /* [d_model] */
    float *wo_weight;          /* [d_model, d_model] */
    float *wo_bias;            /* [d_model] */

    /* Pre-attention LayerNorm (with bias) */
    float *attn_norm_weight;   /* [d_model] */
    float *attn_norm_bias;     /* [d_model] */

    /* FFN: GELU(fc1(x)) -> fc2 (ALL have biases) - pre-converted to f32 */
    float *fc1_weight;         /* [ffn_dim, d_model] */
    float *fc1_bias;           /* [ffn_dim] */
    float *fc2_weight;         /* [d_model, ffn_dim] */
    float *fc2_bias;           /* [d_model] */

    /* Pre-FFN LayerNorm (with bias) */
    float *ffn_norm_weight;    /* [d_model] */
    float *ffn_norm_bias;      /* [d_model] */
} qwen_enc_layer_t;

typedef struct {
    /* Conv2D stem (3 layers, each 3x3, stride 2) */
    float *conv1_weight;       /* [480, 1, 3, 3] */
    float *conv1_bias;         /* [480] */
    float *conv2_weight;       /* [480, 480, 3, 3] */
    float *conv2_bias;         /* [480] */
    float *conv3_weight;       /* [480, 480, 3, 3] */
    float *conv3_bias;         /* [480] */

    /* Conv output projection - pre-converted to f32 */
    float *conv_out_weight;    /* [d_model, 7680] */

    /* Transformer layers */
    qwen_enc_layer_t layers[QWEN_MAX_ENC_LAYERS];

    /* Final LayerNorm */
    float *ln_post_weight;     /* [d_model] */
    float *ln_post_bias;       /* [d_model] */

    /* Projection layers - pre-converted to f32 */
    float *proj1_weight;       /* [d_model, d_model] */
    float *proj1_bias;         /* [d_model] */
    float *proj2_weight;       /* [output_dim, d_model] */
    float *proj2_bias;         /* [output_dim] */
} qwen_encoder_t;

/* ========================================================================
 * LLM Decoder Layer
 * ======================================================================== */

typedef struct {
    /* Self-attention (NO biases in decoder) */
    uint16_t *wq_weight_bf16;  /* [n_heads*head_dim, hidden] */
    uint16_t *wk_weight_bf16;  /* [n_kv_heads*head_dim, hidden] */
    uint16_t *wv_weight_bf16;  /* [n_kv_heads*head_dim, hidden] */
    uint16_t *wo_weight_bf16;  /* [hidden, n_heads*head_dim] */

    /* Per-head Q/K RMSNorm */
    float *q_norm_weight;      /* [head_dim] = [128] */
    float *k_norm_weight;      /* [head_dim] = [128] */

    /* RMSNorm (no bias) */
    float *input_norm;         /* [hidden] */
    float *post_attn_norm;     /* [hidden] */

    /* SwiGLU MLP (NO biases) */
    uint16_t *gate_weight_bf16; /* [intermediate, hidden] */
    uint16_t *up_weight_bf16;   /* [intermediate, hidden] */
    uint16_t *down_weight_bf16; /* [hidden, intermediate] */

    /* Fused gate+up weight for single-token matvec [2*intermediate, hidden] */
    uint16_t *gate_up_fused_bf16;
} qwen_dec_layer_t;

typedef struct {
    /* Token embeddings (tied with lm_head) */
    uint16_t *tok_embeddings_bf16; /* [vocab_size, hidden] */

    /* Transformer layers */
    qwen_dec_layer_t layers[QWEN_MAX_DEC_LAYERS];

    /* Final RMSNorm */
    float *norm;               /* [hidden] */
} qwen_decoder_t;

/* ========================================================================
 * Token Callback (streaming output)
 * ======================================================================== */

/* Called for each decoded text token during autoregressive generation.
 * 'piece' is the decoded token string (UTF-8). */
typedef void (*qwen_token_cb)(const char *piece, void *userdata);

/* ========================================================================
 * Main Context
 * ======================================================================== */

typedef struct {
    qwen_config_t config;
    qwen_encoder_t encoder;
    qwen_decoder_t decoder;

    /* Model files (kept open for mmap) */
    void *safetensors;         /* multi_safetensors_t* */
    char model_dir[512];

    /* KV cache for decoder */
    float *kv_cache_k;         /* [layers, max_seq, kv_heads * head_dim] */
    float *kv_cache_v;
    int kv_cache_len;
    int kv_cache_max;

    /* Persistent decoder buffers (single-token generation) */
    float *dec_x, *dec_x_norm, *dec_q, *dec_k, *dec_v;
    float *dec_attn_out, *dec_proj_out;
    float *dec_gate, *dec_up, *dec_ffn_out;
    float *dec_rope_cos, *dec_rope_sin;

    /* Persistent decoder prefill buffers (multi-token prefill) */
    float *pref_x, *pref_x_norm, *pref_q, *pref_k, *pref_v;
    float *pref_attn_out, *pref_proj_out, *pref_ffn_out;
    float *pref_gate, *pref_gate_up;
    int pref_seq_cap;

    /* Cached RoPE tables for decoder positions */
    float *rope_cache_cos, *rope_cache_sin;   /* [pos, head_dim] */
    float *rope_inv_freq;                     /* [head_dim / 2] */
    int rope_cache_cap;                       /* cached positions */
    int rope_inv_freq_half;                   /* cached half-dim */

    /* Token streaming callback (optional) */
    qwen_token_cb token_cb;
    void *token_cb_userdata;

    /* Segmentation settings */
    float segment_sec;             /* 0 = no splitting, default full-audio decode */
    float search_sec;              /* segment-cutting silence search window ± seconds (default 3) */

    /* Streaming settings */
    float stream_chunk_sec;        /* chunk interval in seconds (default 2.0) */
    int stream_rollback;           /* tokens to roll back per chunk (default 5) */
    int stream_unfixed_chunks;     /* cold-start chunks without prefix (default 2) */
    int stream_max_new_tokens;     /* max generated tokens per streaming step (default 32) */
    int past_text_conditioning;    /* 1=enable past text conditioning in -S/--stream (default: off).
                                    * In segmented mode, this also enables boundary cleanup/post-processing. */
    int skip_silence;              /* 1=drop long silent spans before transcription */

    /* Optional prompt/language controls */
    char *prompt;                  /* system prompt text (UTF-8) */
    char *force_language;          /* normalized language name, or NULL */
    int *prompt_tokens;            /* cached token ids for prompt text */
    int n_prompt_tokens;
    int *force_prompt_tokens;      /* cached token ids for "language X" + <asr_text> */
    int n_force_prompt_tokens;
    int prompt_tokens_ready;       /* cache valid flag */

    /* Per-run performance stats (populated by last transcription call) */
    double perf_total_ms;          /* end-to-end inference time in milliseconds */
    int perf_text_tokens;          /* emitted text tokens (after <asr_text>) */
    double perf_audio_ms;          /* input audio duration in milliseconds */
    double perf_encode_ms;         /* mel + encoder time in milliseconds */
    double perf_decode_ms;         /* decoder prefill + decode time in milliseconds */
} qwen_ctx_t;

/* ========================================================================
 * Live Audio (incremental stdin streaming)
 * ======================================================================== */

typedef struct {
    /* Written by reader thread under mutex */
    float *samples;
    int64_t sample_offset;      /* global index of samples[0] */
    int64_t n_samples;          /* number of valid samples in buffer */
    int64_t capacity;           /* allocated capacity (in samples) */
    int eof;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    pthread_t thread;
} qwen_live_audio_t;

/* ========================================================================
 * API Functions
 * ======================================================================== */

/* Load model from directory */
qwen_ctx_t *qwen_load(const char *model_dir);

/* Free all resources */
void qwen_free(qwen_ctx_t *ctx);

/* Set a callback to receive each decoded token as it's generated.
 * Set cb=NULL to disable. The callback is invoked during transcription. */
void qwen_set_token_callback(qwen_ctx_t *ctx, qwen_token_cb cb, void *userdata);

/* Set optional system prompt text (UTF-8). Pass NULL or "" to clear.
 * Returns 0 on success, -1 on allocation/encoding errors. */
int qwen_set_prompt(qwen_ctx_t *ctx, const char *prompt);

/* Set optional forced language. Pass NULL or "" to clear.
 * Returns 0 on success, -1 if language is unsupported. */
int qwen_set_force_language(qwen_ctx_t *ctx, const char *language);

/* Comma-separated supported language names for --language. */
const char *qwen_supported_languages_csv(void);

/* Transcribe a WAV file, returns allocated string (caller must free) */
char *qwen_transcribe(qwen_ctx_t *ctx, const char *wav_path);

/* Transcribe from raw audio samples (mono float32, 16kHz) */
char *qwen_transcribe_audio(qwen_ctx_t *ctx, const float *samples, int n_samples);

/* Transcribe from stdin (auto-detect WAV or raw s16le) */
char *qwen_transcribe_stdin(qwen_ctx_t *ctx);

/* Streaming transcription: process audio in chunks with prefix rollback.
 * Re-encodes growing audio and uses previous text as decoder context.
 * Tokens are emitted via the token callback as they become "fixed". */
char *qwen_transcribe_stream(qwen_ctx_t *ctx, const float *samples, int n_samples);

/* Live streaming transcription from an incrementally-filled audio source.
 * The streaming loop waits for new data instead of terminating at EOF.
 * Tokens are emitted via the token callback as they become "fixed". */
char *qwen_transcribe_stream_live(qwen_ctx_t *ctx, qwen_live_audio_t *live);

/* ========================================================================
 * Push-Based Incremental Streaming API
 *
 * Feed audio incrementally and pull decoded tokens, without blocking on
 * a complete file or stdin thread.
 *
 * Usage:
 *   qwen_stream_t *s = qwen_stream_init(ctx);
 *   while (have_audio) {
 *       qwen_stream_feed(s, samples, n);
 *       const char *tokens[64];
 *       int n = qwen_stream_get(s, tokens, 64);
 *       for (int i = 0; i < n; i++) printf("%s", tokens[i]);
 *   }
 *   qwen_stream_finish(s);
 *   // drain remaining tokens with qwen_stream_get
 *   qwen_stream_free(s);
 * ======================================================================== */

typedef struct qwen_stream qwen_stream_t;

/* Initialize a push-based streaming session.
 * The ctx must remain valid for the lifetime of the stream.
 * Returns NULL on failure. */
qwen_stream_t *qwen_stream_init(qwen_ctx_t *ctx);

/* Feed audio samples (mono float32, 16 kHz). Internally processes complete
 * chunks as they accumulate. Returns 0 on success, -1 on error. */
int qwen_stream_feed(qwen_stream_t *s, const float *samples, int n_samples);

/* Retrieve decoded token strings. Fills out_tokens with up to max_tokens
 * pointers. Each pointer is valid until qwen_stream_free() is called.
 * Returns the number of tokens written. */
int qwen_stream_get(qwen_stream_t *s, const char **out_tokens, int max_tokens);

/* Signal end of audio. Processes any remaining samples as the final chunk
 * and emits all pending tokens. Returns 0 on success, -1 on error. */
int qwen_stream_finish(qwen_stream_t *s);

/* Free the streaming session and all associated resources. */
void qwen_stream_free(qwen_stream_t *s);

/* Change the chunk interval (default comes from ctx->stream_chunk_sec).
 * Must be called before the first qwen_stream_feed for full effect. */
void qwen_stream_set_interval(qwen_stream_t *s, float seconds);

/* ========================================================================
 * Internal Functions
 * ======================================================================== */

/* Audio encoder forward pass */
float *qwen_encoder_forward(qwen_ctx_t *ctx, const float *mel, int mel_frames,
                             int *out_seq_len);

/* Decoder prefill (multiple tokens) */
void qwen_decoder_prefill(qwen_ctx_t *ctx, const float *input_embeds, int seq_len);

/* Decoder forward (single token, uses KV cache, returns greedy token) */
int qwen_decoder_forward(qwen_ctx_t *ctx, const float *input_embed);

/* Global verbose flag */
extern int qwen_verbose;

/* Monitor mode: show inline Unicode symbols on stderr for streaming diagnostics.
 * Symbols: ▶ encoder  · prefill  ▪ decode  ▸ slow decode  ⟳ window eviction */
extern int qwen_monitor;

#endif /* QWEN_ASR_H */
