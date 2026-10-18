/*
 * qwen_asr_encoder.c - Audio encoder forward pass
 *
 * Architecture:
 *   Per-chunk Conv2D stem: 3 layers of Conv2D(3x3, stride=2, pad=1) -> GELU
 *     128 mel bins -> 64 -> 32 -> 16 frequency, time/8
 *     Reshape [480, 16, T/8] -> [T/8, 7680], project to d_model
 *   Per-chunk sinusoidal position embeddings
 *   Transformer encoder layers (bidirectional windowed attention):
 *     LayerNorm -> MHA (Q,K,V all have biases) -> residual
 *     LayerNorm -> GELU FFN (fc1,fc2 with biases) -> residual
 *   Final LayerNorm
 *   Projection: proj1 (GELU) -> proj2
 */

#include "qwen_asr.h"
#include "qwen_asr_kernels.h"
#include "qwen_asr_safetensors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ========================================================================
 * Weight Loading
 * ======================================================================== */

static float *load_f32(multi_safetensors_t *ms, const char *name) {
    safetensors_file_t *sf = NULL;
    const safetensor_t *t = multi_safetensors_find(ms, name, &sf);
    if (!t) {
        fprintf(stderr, "encoder: weight not found: %s\n", name);
        return NULL;
    }
    return safetensors_get_f32(sf, t);
}

/* Load bf16 weight and convert to f32 at load time.
 * Encoder always processes batches, so pre-converting avoids
 * repeated scratch-buffer conversion during forward pass. */
static float *load_bf16_as_f32(multi_safetensors_t *ms, const char *name) {
    safetensors_file_t *sf = NULL;
    const safetensor_t *t = multi_safetensors_find(ms, name, &sf);
    if (!t) {
        fprintf(stderr, "encoder: weight not found: %s\n", name);
        return NULL;
    }
    uint16_t *bf16 = safetensors_get_bf16_direct(sf, t);
    if (!bf16) return NULL;

    /* Compute number of elements from tensor shape */
    size_t n = 1;
    for (int i = 0; i < t->ndim; i++) n *= t->shape[i];

    float *f32 = (float *)malloc(n * sizeof(float));
    if (!f32) return NULL;

    uint32_t *d = (uint32_t *)(void *)f32;
    for (size_t i = 0; i < n; i++)
        d[i] = ((uint32_t)bf16[i]) << 16;

    return f32;
}

int qwen_encoder_load(qwen_encoder_t *enc, multi_safetensors_t *ms,
                       const qwen_config_t *cfg) {
    char name[512];
    const char *pfx = cfg->hf_layout ? "model.audio_tower." : "thinker.audio_tower.";
    const char *proj1 = cfg->hf_layout ? "model.multi_modal_projector.linear_1"
                                       : "thinker.audio_tower.proj1";
    const char *proj2 = cfg->hf_layout ? "model.multi_modal_projector.linear_2"
                                       : "thinker.audio_tower.proj2";

    /* Conv2D stem (small, f32) */
    snprintf(name, sizeof(name), "%sconv2d1.weight", pfx);
    enc->conv1_weight = load_f32(ms, name);
    snprintf(name, sizeof(name), "%sconv2d1.bias", pfx);
    enc->conv1_bias = load_f32(ms, name);
    snprintf(name, sizeof(name), "%sconv2d2.weight", pfx);
    enc->conv2_weight = load_f32(ms, name);
    snprintf(name, sizeof(name), "%sconv2d2.bias", pfx);
    enc->conv2_bias = load_f32(ms, name);
    snprintf(name, sizeof(name), "%sconv2d3.weight", pfx);
    enc->conv3_weight = load_f32(ms, name);
    snprintf(name, sizeof(name), "%sconv2d3.bias", pfx);
    enc->conv3_bias = load_f32(ms, name);

    if (!enc->conv1_weight || !enc->conv2_weight || !enc->conv3_weight) return -1;

    /* Conv output projection (bf16, no bias) */
    snprintf(name, sizeof(name), "%sconv_out.weight", pfx);
    enc->conv_out_weight = load_bf16_as_f32(ms, name);
    if (!enc->conv_out_weight) return -1;

    /* Transformer layers */
    for (int i = 0; i < cfg->enc_layers; i++) {
        qwen_enc_layer_t *l = &enc->layers[i];
        /* Attention weights (bf16) and biases (f32) */
        snprintf(name, sizeof(name), "%slayers.%d.self_attn.q_proj.weight", pfx, i);
        l->wq_weight = load_bf16_as_f32(ms, name);
        snprintf(name, sizeof(name), "%slayers.%d.self_attn.q_proj.bias", pfx, i);
        l->wq_bias = load_f32(ms, name);
        snprintf(name, sizeof(name), "%slayers.%d.self_attn.k_proj.weight", pfx, i);
        l->wk_weight = load_bf16_as_f32(ms, name);
        snprintf(name, sizeof(name), "%slayers.%d.self_attn.k_proj.bias", pfx, i);
        l->wk_bias = load_f32(ms, name);
        snprintf(name, sizeof(name), "%slayers.%d.self_attn.v_proj.weight", pfx, i);
        l->wv_weight = load_bf16_as_f32(ms, name);
        snprintf(name, sizeof(name), "%slayers.%d.self_attn.v_proj.bias", pfx, i);
        l->wv_bias = load_f32(ms, name);
        snprintf(name, sizeof(name), "%slayers.%d.self_attn.out_proj.weight", pfx, i);
        l->wo_weight = load_bf16_as_f32(ms, name);
        snprintf(name, sizeof(name), "%slayers.%d.self_attn.out_proj.bias", pfx, i);
        l->wo_bias = load_f32(ms, name);

        /* Pre-attention LayerNorm */
        snprintf(name, sizeof(name), "%slayers.%d.self_attn_layer_norm.weight", pfx, i);
        l->attn_norm_weight = load_f32(ms, name);
        snprintf(name, sizeof(name), "%slayers.%d.self_attn_layer_norm.bias", pfx, i);
        l->attn_norm_bias = load_f32(ms, name);

        /* FFN weights (bf16) and biases (f32) */
        snprintf(name, sizeof(name), "%slayers.%d.fc1.weight", pfx, i);
        l->fc1_weight = load_bf16_as_f32(ms, name);
        snprintf(name, sizeof(name), "%slayers.%d.fc1.bias", pfx, i);
        l->fc1_bias = load_f32(ms, name);
        snprintf(name, sizeof(name), "%slayers.%d.fc2.weight", pfx, i);
        l->fc2_weight = load_bf16_as_f32(ms, name);
        snprintf(name, sizeof(name), "%slayers.%d.fc2.bias", pfx, i);
        l->fc2_bias = load_f32(ms, name);

        /* Pre-FFN LayerNorm */
        snprintf(name, sizeof(name), "%slayers.%d.final_layer_norm.weight", pfx, i);
        l->ffn_norm_weight = load_f32(ms, name);
        snprintf(name, sizeof(name), "%slayers.%d.final_layer_norm.bias", pfx, i);
        l->ffn_norm_bias = load_f32(ms, name);

        if (!l->wq_weight || !l->wk_weight ||
            !l->wv_weight || !l->wo_weight) {
            fprintf(stderr, "encoder: failed to load layer %d weights\n", i);
            return -1;
        }

    }

    /* Final LayerNorm */
    snprintf(name, sizeof(name), "%sln_post.weight", pfx);
    enc->ln_post_weight = load_f32(ms, name);
    snprintf(name, sizeof(name), "%sln_post.bias", pfx);
    enc->ln_post_bias = load_f32(ms, name);

    /* Projection layers */
    snprintf(name, sizeof(name), "%s.weight", proj1);
    enc->proj1_weight = load_bf16_as_f32(ms, name);
    snprintf(name, sizeof(name), "%s.bias", proj1);
    enc->proj1_bias = load_f32(ms, name);
    snprintf(name, sizeof(name), "%s.weight", proj2);
    enc->proj2_weight = load_bf16_as_f32(ms, name);
    snprintf(name, sizeof(name), "%s.bias", proj2);
    enc->proj2_bias = load_f32(ms, name);

    if (!enc->ln_post_weight || !enc->proj1_weight || !enc->proj2_weight)
        return -1;

    return 0;
}

/* ========================================================================
 * Forward Pass
 * ======================================================================== */

#if !defined(USE_ROCM) && !defined(USE_CUDA)
float *qwen_encoder_forward(qwen_ctx_t *ctx, const float *mel, int mel_frames,
                             int *out_seq_len) {
    const qwen_config_t *cfg = &ctx->config;
    qwen_encoder_t *enc = &ctx->encoder;

    int d_model = cfg->enc_d_model;
    int n_heads = cfg->enc_heads;
    int head_dim = cfg->enc_head_dim;
    int ffn_dim = cfg->enc_ffn_dim;
    int output_dim = cfg->enc_output_dim;
    int chunk_size = cfg->enc_chunk_size;          /* 100 */
    int n_window_infer = cfg->enc_n_window_infer;  /* 800 */


    /* ---- Per-chunk Conv2D stem ---- */
    /* mel: [128, mel_frames] (already in Conv2D-friendly layout)
     * Process chunks of chunk_size frames, each producing tokens_per_chunk tokens */
    int n_chunks = (mel_frames + chunk_size - 1) / chunk_size;
    int tokens_per_chunk = 0; /* computed from first chunk */

    /* First: determine output tokens per chunk from a full chunk */
    {
        int w = chunk_size;
        int w1 = (w + 2 * 1 - 3) / 2 + 1;
        int w2 = (w1 + 2 * 1 - 3) / 2 + 1;
        int w3 = (w2 + 2 * 1 - 3) / 2 + 1;
        tokens_per_chunk = w3; /* 13 for chunk_size=100 */
    }

    /* Collect all chunks' output tokens */
    int total_tokens = 0;

    /* Pre-calculate total tokens */
    for (int c = 0; c < n_chunks; c++) {
        int start = c * chunk_size;
        int end = start + chunk_size;
        if (end > mel_frames) end = mel_frames;
        int chunk_w = end - start;
        int w1 = (chunk_w + 2 - 3) / 2 + 1;
        int w2 = (w1 + 2 - 3) / 2 + 1;
        int w3 = (w2 + 2 - 3) / 2 + 1;
        total_tokens += w3;
    }


    /* Allocate main sequence buffer: [total_tokens, d_model] */
    float *x = (float *)calloc((size_t)total_tokens * d_model, sizeof(float));
    int token_offset = 0;

    /* Process each chunk through Conv2D + reshape + project + sinusoidal PE */
    for (int c = 0; c < n_chunks; c++) {
        int start = c * chunk_size;
        int end = start + chunk_size;
        if (end > mel_frames) end = mel_frames;
        int chunk_w = end - start;

        /* Extract chunk mel: [128, chunk_w] */
        float *chunk_mel = (float *)malloc(128 * chunk_w * sizeof(float));
        for (int m = 0; m < 128; m++) {
            memcpy(chunk_mel + m * chunk_w, mel + m * mel_frames + start,
                   chunk_w * sizeof(float));
        }

        /* Conv2D layer 1: [1, 128, chunk_w] -> [480, 64, w1] */
        int h1 = (128 + 2 - 3) / 2 + 1; /* 64 */
        int w1 = (chunk_w + 2 - 3) / 2 + 1;
        float *c1 = (float *)malloc(QWEN_CONV_HIDDEN * h1 * w1 * sizeof(float));
        qwen_conv2d(c1, chunk_mel, enc->conv1_weight, enc->conv1_bias,
                     1, QWEN_CONV_HIDDEN, 128, chunk_w, 3, 3, 2, 1);
        qwen_gelu(c1, QWEN_CONV_HIDDEN * h1 * w1);
        free(chunk_mel);

        /* Conv2D layer 2: [480, 64, w1] -> [480, 32, w2] */
        int h2 = (h1 + 2 - 3) / 2 + 1; /* 32 */
        int w2 = (w1 + 2 - 3) / 2 + 1;
        float *c2 = (float *)malloc(QWEN_CONV_HIDDEN * h2 * w2 * sizeof(float));
        qwen_conv2d(c2, c1, enc->conv2_weight, enc->conv2_bias,
                     QWEN_CONV_HIDDEN, QWEN_CONV_HIDDEN, h1, w1, 3, 3, 2, 1);
        qwen_gelu(c2, QWEN_CONV_HIDDEN * h2 * w2);
        free(c1);

        /* Conv2D layer 3: [480, 32, w2] -> [480, 16, w3] */
        int h3 = (h2 + 2 - 3) / 2 + 1; /* 16 */
        int w3 = (w2 + 2 - 3) / 2 + 1;
        float *c3 = (float *)malloc(QWEN_CONV_HIDDEN * h3 * w3 * sizeof(float));
        qwen_conv2d(c3, c2, enc->conv3_weight, enc->conv3_bias,
                     QWEN_CONV_HIDDEN, QWEN_CONV_HIDDEN, h2, w2, 3, 3, 2, 1);
        qwen_gelu(c3, QWEN_CONV_HIDDEN * h3 * w3);
        free(c2);

        /* Reshape [480, 16, w3] -> [w3, 480*16=7680] then project to d_model */
        int conv_proj_dim = QWEN_CONV_HIDDEN * h3; /* 480 * 16 = 7680 */
        float *reshaped = (float *)malloc(w3 * conv_proj_dim * sizeof(float));
        for (int t = 0; t < w3; t++) {
            for (int ch = 0; ch < QWEN_CONV_HIDDEN; ch++) {
                for (int f = 0; f < h3; f++) {
                    reshaped[t * conv_proj_dim + ch * h3 + f] =
                        c3[ch * h3 * w3 + f * w3 + t];
                }
            }
        }
        free(c3);

        /* Project: [w3, 7680] -> [w3, d_model] (no bias) */
        float *projected = x + (size_t)token_offset * d_model;
        qwen_linear_nobias(projected, reshaped, enc->conv_out_weight,
                            w3, conv_proj_dim, d_model);
        free(reshaped);

        /* Add per-chunk sinusoidal position embeddings (starting from pos 0) */
        float *pe = (float *)malloc(w3 * d_model * sizeof(float));
        qwen_sinusoidal_pe(pe, w3, d_model);
        qwen_add_inplace(projected, pe, w3 * d_model);
        free(pe);

        token_offset += w3;
    }

    /* ---- Build attention window boundaries ---- */
    /* Window size = tokens_per_chunk * (n_window_infer / chunk_size) */
    int window_token_size = tokens_per_chunk * (n_window_infer / chunk_size);
    int n_windows = (total_tokens + window_token_size - 1) / window_token_size;
    int *window_starts = (int *)malloc((n_windows + 1) * sizeof(int));
    for (int w = 0; w < n_windows; w++) {
        window_starts[w] = w * window_token_size;
    }
    window_starts[n_windows] = total_tokens;


    /* ---- Transformer layers ---- */
    float *x_norm = (float *)malloc(total_tokens * d_model * sizeof(float));
    float *q = (float *)malloc(total_tokens * d_model * sizeof(float));
    float *k = (float *)malloc(total_tokens * d_model * sizeof(float));
    float *v = (float *)malloc(total_tokens * d_model * sizeof(float));
    float *attn_out = (float *)malloc(total_tokens * d_model * sizeof(float));
    float *proj_out = (float *)malloc(total_tokens * d_model * sizeof(float));
    float *ffn_mid = (float *)malloc(total_tokens * ffn_dim * sizeof(float));
    float *ffn_out = (float *)malloc(total_tokens * d_model * sizeof(float));

    float scale = 1.0f / sqrtf((float)head_dim);

    for (int layer = 0; layer < cfg->enc_layers; layer++) {
        qwen_enc_layer_t *l = &enc->layers[layer];

        /* ---- Self-attention ---- */
        qwen_layer_norm(x_norm, x, l->attn_norm_weight, l->attn_norm_bias,
                        total_tokens, d_model, 1e-5f);

        qwen_linear(q, x_norm, l->wq_weight, l->wq_bias,
                     total_tokens, d_model, d_model);
        qwen_linear(k, x_norm, l->wk_weight, l->wk_bias,
                     total_tokens, d_model, d_model);
        qwen_linear(v, x_norm, l->wv_weight, l->wv_bias,
                     total_tokens, d_model, d_model);

        qwen_bidirectional_attention(attn_out, q, k, v,
                                      total_tokens, n_heads, head_dim, scale,
                                      window_starts, n_windows);

        /* Output projection + residual */
        qwen_linear(proj_out, attn_out, l->wo_weight, l->wo_bias,
                     total_tokens, d_model, d_model);
        qwen_add_inplace(x, proj_out, total_tokens * d_model);

        /* ---- FFN ---- */
        qwen_layer_norm(x_norm, x, l->ffn_norm_weight, l->ffn_norm_bias,
                        total_tokens, d_model, 1e-5f);

        /* GELU FFN: fc1 -> GELU -> fc2 */
        qwen_linear(ffn_mid, x_norm, l->fc1_weight, l->fc1_bias,
                     total_tokens, d_model, ffn_dim);
        qwen_gelu(ffn_mid, total_tokens * ffn_dim);
        qwen_linear(ffn_out, ffn_mid, l->fc2_weight, l->fc2_bias,
                     total_tokens, ffn_dim, d_model);
        qwen_add_inplace(x, ffn_out, total_tokens * d_model);

    }

    /* Final LayerNorm */
    qwen_layer_norm(x, x, enc->ln_post_weight, enc->ln_post_bias,
                    total_tokens, d_model, 1e-5f);

    /* Projection: proj1 (GELU) -> proj2 */
    float *proj_mid = (float *)malloc(total_tokens * d_model * sizeof(float));
    qwen_linear(proj_mid, x, enc->proj1_weight, enc->proj1_bias,
                 total_tokens, d_model, d_model);
    qwen_gelu(proj_mid, total_tokens * d_model);

    float *enc_output = (float *)malloc(total_tokens * output_dim * sizeof(float));
    qwen_linear(enc_output, proj_mid, enc->proj2_weight, enc->proj2_bias,
                 total_tokens, d_model, output_dim);
    free(proj_mid);

    /* Clean up */
    free(x); free(x_norm); free(q); free(k); free(v);
    free(attn_out); free(proj_out);
    free(ffn_mid); free(ffn_out);
    free(window_starts);

    *out_seq_len = total_tokens;
    return enc_output;
}

#endif /* CPU backend */
