/*
 * qwen_asr_decoder.c - Qwen3 LLM decoder
 *
 * Architecture (per layer):
 *   RMSNorm -> QKV (no bias) -> per-head Q/K RMSNorm -> NeoX RoPE
 *   -> Causal GQA attention -> Output proj -> residual
 *   RMSNorm -> SwiGLU MLP (gate/up/down, no bias) -> residual
 *
 * Features: Q/K per-head RMSNorm, NeoX split-half RoPE, GQA 2:1,
 * tied embeddings (tok_embeddings == lm_head).
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
        fprintf(stderr, "decoder: weight not found: %s\n", name);
        return NULL;
    }
    return safetensors_get_f32(sf, t);
}

static uint16_t *load_bf16_direct(multi_safetensors_t *ms, const char *name) {
    safetensors_file_t *sf = NULL;
    const safetensor_t *t = multi_safetensors_find(ms, name, &sf);
    if (!t) {
        fprintf(stderr, "decoder: weight not found: %s\n", name);
        return NULL;
    }
    return safetensors_get_bf16_direct(sf, t);
}

int qwen_decoder_load(qwen_decoder_t *dec, multi_safetensors_t *ms,
                       const qwen_config_t *cfg) {
    char name[512];
    const char *pfx = cfg->hf_layout ? "model.language_model." : "thinker.model.";

    /* Token embeddings (large, bf16 mmap direct) */
    snprintf(name, sizeof(name), "%sembed_tokens.weight", pfx);
    dec->tok_embeddings_bf16 = load_bf16_direct(ms, name);
    if (!dec->tok_embeddings_bf16) return -1;

    /* Transformer layers */
    for (int i = 0; i < cfg->dec_layers; i++) {
        qwen_dec_layer_t *l = &dec->layers[i];

        /* Attention weights (bf16, no bias) */
        snprintf(name, sizeof(name), "%slayers.%d.self_attn.q_proj.weight", pfx, i);
        l->wq_weight_bf16 = load_bf16_direct(ms, name);
        snprintf(name, sizeof(name), "%slayers.%d.self_attn.k_proj.weight", pfx, i);
        l->wk_weight_bf16 = load_bf16_direct(ms, name);
        snprintf(name, sizeof(name), "%slayers.%d.self_attn.v_proj.weight", pfx, i);
        l->wv_weight_bf16 = load_bf16_direct(ms, name);
        snprintf(name, sizeof(name), "%slayers.%d.self_attn.o_proj.weight", pfx, i);
        l->wo_weight_bf16 = load_bf16_direct(ms, name);

        /* Per-head Q/K RMSNorm weights */
        snprintf(name, sizeof(name), "%slayers.%d.self_attn.q_norm.weight", pfx, i);
        l->q_norm_weight = load_f32(ms, name);
        snprintf(name, sizeof(name), "%slayers.%d.self_attn.k_norm.weight", pfx, i);
        l->k_norm_weight = load_f32(ms, name);

        /* RMSNorm weights */
        snprintf(name, sizeof(name), "%slayers.%d.input_layernorm.weight", pfx, i);
        l->input_norm = load_f32(ms, name);
        snprintf(name, sizeof(name), "%slayers.%d.post_attention_layernorm.weight", pfx, i);
        l->post_attn_norm = load_f32(ms, name);

        /* SwiGLU MLP weights (bf16, no bias) */
        snprintf(name, sizeof(name), "%slayers.%d.mlp.gate_proj.weight", pfx, i);
        l->gate_weight_bf16 = load_bf16_direct(ms, name);
        snprintf(name, sizeof(name), "%slayers.%d.mlp.up_proj.weight", pfx, i);
        l->up_weight_bf16 = load_bf16_direct(ms, name);
        snprintf(name, sizeof(name), "%slayers.%d.mlp.down_proj.weight", pfx, i);
        l->down_weight_bf16 = load_bf16_direct(ms, name);

        if (!l->wq_weight_bf16 || !l->wk_weight_bf16 ||
            !l->wv_weight_bf16 || !l->wo_weight_bf16 ||
            !l->gate_weight_bf16 || !l->up_weight_bf16 || !l->down_weight_bf16) {
            fprintf(stderr, "decoder: failed to load layer %d\n", i);
            return -1;
        }

        #if !defined(USE_ROCM) && !defined(USE_CUDA)
        /* Fuse gate+up weights: interleave rows [gate_row0, up_row0, gate_row1, up_row1, ...] */
        {
            int inter = cfg->dec_intermediate;
            int hidden = cfg->dec_hidden;
            size_t row_bytes = (size_t)hidden * sizeof(uint16_t);
            l->gate_up_fused_bf16 = (uint16_t *)malloc(2 * (size_t)inter * row_bytes);
            for (int r = 0; r < inter; r++) {
                memcpy(l->gate_up_fused_bf16 + (size_t)(2 * r) * hidden,
                       l->gate_weight_bf16 + (size_t)r * hidden, row_bytes);
                memcpy(l->gate_up_fused_bf16 + (size_t)(2 * r + 1) * hidden,
                       l->up_weight_bf16 + (size_t)r * hidden, row_bytes);
            }
        }

        #endif
    }

    /* Final RMSNorm */
    snprintf(name, sizeof(name), "%snorm.weight", pfx);
    dec->norm = load_f32(ms, name);
    if (!dec->norm) return -1;

    return 0;
}

/* ========================================================================
 * KV Cache Management
 * ======================================================================== */

#if !defined(USE_ROCM) && !defined(USE_CUDA)
static int kv_cache_init(qwen_ctx_t *ctx, int max_seq) {
    int kv_dim = ctx->config.dec_kv_heads * ctx->config.dec_head_dim;
    size_t cache_size = (size_t)ctx->config.dec_layers * max_seq * kv_dim * sizeof(float);
    ctx->kv_cache_k = (float *)calloc(1, cache_size);
    ctx->kv_cache_v = (float *)calloc(1, cache_size);
    ctx->kv_cache_len = 0;
    ctx->kv_cache_max = max_seq;
    if (!ctx->kv_cache_k || !ctx->kv_cache_v) return -1;
    return 0;
}

static int kv_cache_grow(qwen_ctx_t *ctx, int required) {
    if (required <= ctx->kv_cache_max) return 0;

    int kv_dim = ctx->config.dec_kv_heads * ctx->config.dec_head_dim;
    int new_max = ctx->kv_cache_max;
    while (new_max < required) new_max *= 2;

    size_t new_stride = (size_t)new_max * kv_dim;
    size_t old_stride = (size_t)ctx->kv_cache_max * kv_dim;
    size_t total = (size_t)ctx->config.dec_layers * new_stride * sizeof(float);

    float *new_k = (float *)calloc(1, total);
    float *new_v = (float *)calloc(1, total);
    if (!new_k || !new_v) { free(new_k); free(new_v); return -1; }

    size_t copy = (size_t)ctx->kv_cache_len * kv_dim * sizeof(float);
    for (int l = 0; l < ctx->config.dec_layers; l++) {
        memcpy(new_k + l * new_stride, ctx->kv_cache_k + l * old_stride, copy);
        memcpy(new_v + l * new_stride, ctx->kv_cache_v + l * old_stride, copy);
    }

    free(ctx->kv_cache_k);
    free(ctx->kv_cache_v);
    ctx->kv_cache_k = new_k;
    ctx->kv_cache_v = new_v;
    ctx->kv_cache_max = new_max;
    return 0;
}

static float *kv_cache_k_at(qwen_ctx_t *ctx, int layer, int pos) {
    int kv_dim = ctx->config.dec_kv_heads * ctx->config.dec_head_dim;
    return ctx->kv_cache_k + ((size_t)layer * ctx->kv_cache_max + pos) * kv_dim;
}

static float *kv_cache_v_at(qwen_ctx_t *ctx, int layer, int pos) {
    int kv_dim = ctx->config.dec_kv_heads * ctx->config.dec_head_dim;
    return ctx->kv_cache_v + ((size_t)layer * ctx->kv_cache_max + pos) * kv_dim;
}

static int ensure_prefill_buffers(qwen_ctx_t *ctx, int seq_len) {
    const qwen_config_t *cfg = &ctx->config;
    int dim = cfg->dec_hidden;
    int q_dim = cfg->dec_heads * cfg->dec_head_dim;
    int kv_dim = cfg->dec_kv_heads * cfg->dec_head_dim;
    int intermediate = cfg->dec_intermediate;

    if (seq_len <= ctx->pref_seq_cap) return 0;

    int new_cap = ctx->pref_seq_cap > 0 ? ctx->pref_seq_cap : 64;
    while (new_cap < seq_len) new_cap *= 2;

#define REALLOC_PREF(ptr, count) do {                                          \
    void *tmp__ = realloc((ptr), (size_t)(count) * sizeof(float));             \
    if (!tmp__) return -1;                                                      \
    (ptr) = (float *)tmp__;                                                     \
} while (0)

    REALLOC_PREF(ctx->pref_x, new_cap * dim);
    REALLOC_PREF(ctx->pref_x_norm, new_cap * dim);
    REALLOC_PREF(ctx->pref_q, new_cap * q_dim);
    REALLOC_PREF(ctx->pref_k, new_cap * kv_dim);
    REALLOC_PREF(ctx->pref_v, new_cap * kv_dim);
    REALLOC_PREF(ctx->pref_attn_out, new_cap * q_dim);
    REALLOC_PREF(ctx->pref_proj_out, new_cap * dim);
    REALLOC_PREF(ctx->pref_ffn_out, new_cap * dim);
    REALLOC_PREF(ctx->pref_gate, new_cap * intermediate);
    REALLOC_PREF(ctx->pref_gate_up, new_cap * 2 * intermediate);

#undef REALLOC_PREF

    ctx->pref_seq_cap = new_cap;
    return 0;
}

static int ensure_rope_inv_freq(qwen_ctx_t *ctx, int head_dim, float theta) {
    int half = head_dim / 2;
    if (ctx->rope_inv_freq && ctx->rope_inv_freq_half == half) return 0;

    float *inv = (float *)realloc(ctx->rope_inv_freq, (size_t)half * sizeof(float));
    if (!inv) return -1;
    ctx->rope_inv_freq = inv;

    for (int d = 0; d < half; d++) {
        ctx->rope_inv_freq[d] = 1.0f / powf(theta, (float)(2 * d) / (float)head_dim);
    }
    ctx->rope_inv_freq_half = half;
    return 0;
}

static int ensure_rope_cache(qwen_ctx_t *ctx, int required_pos, int head_dim, float theta) {
    if (required_pos <= ctx->rope_cache_cap) return 0;
    if (ensure_rope_inv_freq(ctx, head_dim, theta) != 0) return -1;

    int new_cap = ctx->rope_cache_cap > 0 ? ctx->rope_cache_cap : 1024;
    while (new_cap < required_pos) new_cap *= 2;

    size_t n = (size_t)new_cap * head_dim;
    float *new_cos = (float *)realloc(ctx->rope_cache_cos, n * sizeof(float));
    if (!new_cos) return -1;
    ctx->rope_cache_cos = new_cos;

    float *new_sin = (float *)realloc(ctx->rope_cache_sin, n * sizeof(float));
    if (!new_sin) return -1;
    ctx->rope_cache_sin = new_sin;

    int half = head_dim / 2;
    for (int pos = ctx->rope_cache_cap; pos < new_cap; pos++) {
        float p = (float)pos;
        float *cos_row = ctx->rope_cache_cos + (size_t)pos * head_dim;
        float *sin_row = ctx->rope_cache_sin + (size_t)pos * head_dim;
        for (int d = 0; d < half; d++) {
            float angle = p * ctx->rope_inv_freq[d];
            float c = cosf(angle);
            float s = sinf(angle);
            cos_row[d] = c;
            cos_row[half + d] = c;
            sin_row[d] = s;
            sin_row[half + d] = s;
        }
    }

    ctx->rope_cache_cap = new_cap;
    return 0;
}

/* ========================================================================
 * Decoder Prefill (Multiple Tokens)
 * ======================================================================== */

void qwen_decoder_prefill(qwen_ctx_t *ctx, const float *input_embeds, int seq_len) {
    qwen_decoder_t *dec = &ctx->decoder;
    const qwen_config_t *cfg = &ctx->config;
    int dim = cfg->dec_hidden;
    int n_heads = cfg->dec_heads;
    int n_kv_heads = cfg->dec_kv_heads;
    int head_dim = cfg->dec_head_dim;
    int intermediate = cfg->dec_intermediate;
    float eps = cfg->dec_rms_norm_eps;
    float theta = cfg->dec_rope_theta;
    int q_dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;

    /* Ensure KV cache */
    if (!ctx->kv_cache_k) {
        if (kv_cache_init(ctx, seq_len + 1024) != 0) return;
    } else if (ctx->kv_cache_len + seq_len > ctx->kv_cache_max) {
        if (kv_cache_grow(ctx, ctx->kv_cache_len + seq_len + 1024) != 0) return;
    }

    if (ensure_prefill_buffers(ctx, seq_len) != 0) return;

    float *x = ctx->pref_x;
    float *x_norm = ctx->pref_x_norm;
    float *q = ctx->pref_q;
    float *k = ctx->pref_k;
    float *v = ctx->pref_v;
    float *attn_out = ctx->pref_attn_out;
    float *proj_out = ctx->pref_proj_out;
    float *ffn_out = ctx->pref_ffn_out;
    float *gate = ctx->pref_gate;
    float *gate_up = ctx->pref_gate_up;

    memcpy(x, input_embeds, (size_t)seq_len * dim * sizeof(float));

    int start_pos = ctx->kv_cache_len;
    if (ensure_rope_cache(ctx, start_pos + seq_len, head_dim, theta) != 0) return;
    const float *rope_cos = ctx->rope_cache_cos + (size_t)start_pos * head_dim;
    const float *rope_sin = ctx->rope_cache_sin + (size_t)start_pos * head_dim;

    float scale = 1.0f / sqrtf((float)head_dim);

    for (int layer = 0; layer < cfg->dec_layers; layer++) {
        qwen_dec_layer_t *l = &dec->layers[layer];

        /* Input RMSNorm */
        qwen_rms_norm(x_norm, x, l->input_norm, seq_len, dim, eps);

        /* QKV projections (no bias) */
        qwen_linear_nobias_bf16(q, x_norm, l->wq_weight_bf16, seq_len, dim, q_dim);
        qwen_linear_nobias_bf16(k, x_norm, l->wk_weight_bf16, seq_len, dim, kv_dim);
        qwen_linear_nobias_bf16(v, x_norm, l->wv_weight_bf16, seq_len, dim, kv_dim);

        /* Per-head Q/K RMSNorm */
        qwen_rms_norm_per_head(q, l->q_norm_weight, seq_len, n_heads, head_dim, eps);
        qwen_rms_norm_per_head(k, l->k_norm_weight, seq_len, n_kv_heads, head_dim, eps);

        /* Apply NeoX RoPE */
        qwen_apply_rope_neox(q, rope_cos, rope_sin, seq_len, n_heads, head_dim);
        qwen_apply_rope_neox(k, rope_cos, rope_sin, seq_len, n_kv_heads, head_dim);

        /* Store K, V in cache */
        for (int s = 0; s < seq_len; s++) {
            memcpy(kv_cache_k_at(ctx, layer, start_pos + s),
                   k + s * kv_dim, kv_dim * sizeof(float));
            memcpy(kv_cache_v_at(ctx, layer, start_pos + s),
                   v + s * kv_dim, kv_dim * sizeof(float));
        }

        /* Causal attention */
        int total_seq = start_pos + seq_len;
        float *full_k = kv_cache_k_at(ctx, layer, 0);
        float *full_v = kv_cache_v_at(ctx, layer, 0);
        qwen_causal_attention(attn_out, q, full_k, full_v,
                               seq_len, total_seq, n_heads, n_kv_heads,
                               head_dim, scale, start_pos);

        /* Output projection + residual */
        qwen_linear_nobias_bf16(proj_out, attn_out, l->wo_weight_bf16,
                                 seq_len, q_dim, dim);
        qwen_add_inplace(x, proj_out, seq_len * dim);

        /* Post-attention RMSNorm */
        qwen_rms_norm(x_norm, x, l->post_attn_norm, seq_len, dim, eps);

        /* SwiGLU MLP */
        qwen_linear_nobias_bf16(gate_up, x_norm, l->gate_up_fused_bf16,
                                 seq_len, dim, 2 * intermediate);
        qwen_swiglu_multiply(gate, gate_up, seq_len, intermediate);
        qwen_linear_nobias_bf16(ffn_out, gate, l->down_weight_bf16,
                                 seq_len, intermediate, dim);

        qwen_add_inplace(x, ffn_out, seq_len * dim);

    }

    ctx->kv_cache_len = start_pos + seq_len;
}

/* ========================================================================
 * Decoder Forward (Single Token Generation)
 * ======================================================================== */

static void ensure_dec_buffers(qwen_ctx_t *ctx) {
    if (ctx->dec_x) return;
    const qwen_config_t *cfg = &ctx->config;
    int dim = cfg->dec_hidden;
    int q_dim = cfg->dec_heads * cfg->dec_head_dim;
    int kv_dim = cfg->dec_kv_heads * cfg->dec_head_dim;
    int intermediate = cfg->dec_intermediate;
    int head_dim = cfg->dec_head_dim;

    ctx->dec_x        = (float *)malloc(dim * sizeof(float));
    ctx->dec_x_norm   = (float *)malloc(dim * sizeof(float));
    ctx->dec_q        = (float *)malloc(q_dim * sizeof(float));
    ctx->dec_k        = (float *)malloc(kv_dim * sizeof(float));
    ctx->dec_v        = (float *)malloc(kv_dim * sizeof(float));
    ctx->dec_attn_out = (float *)malloc(q_dim * sizeof(float));
    ctx->dec_proj_out = (float *)malloc(dim * sizeof(float));
    ctx->dec_gate     = (float *)malloc(2 * intermediate * sizeof(float));
    ctx->dec_up       = NULL; /* unused: gate buffer holds fused gate+up */
    ctx->dec_ffn_out  = (float *)malloc(dim * sizeof(float));
    ctx->dec_rope_cos = (float *)malloc(head_dim * sizeof(float));
    ctx->dec_rope_sin = (float *)malloc(head_dim * sizeof(float));
}

int qwen_decoder_forward(qwen_ctx_t *ctx, const float *input_embed) {
    qwen_decoder_t *dec = &ctx->decoder;
    const qwen_config_t *cfg = &ctx->config;
    int dim = cfg->dec_hidden;
    int n_heads = cfg->dec_heads;
    int n_kv_heads = cfg->dec_kv_heads;
    int head_dim = cfg->dec_head_dim;
    int intermediate = cfg->dec_intermediate;
    float eps = cfg->dec_rms_norm_eps;
    float theta = cfg->dec_rope_theta;
    int q_dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;

    ensure_dec_buffers(ctx);
    float *x = ctx->dec_x;
    float *x_norm = ctx->dec_x_norm;
    float *q = ctx->dec_q;
    float *k = ctx->dec_k;
    float *v = ctx->dec_v;
    float *attn_out = ctx->dec_attn_out;
    float *proj_out = ctx->dec_proj_out;
    float *gate_buf = ctx->dec_gate;
    float *ffn_out = ctx->dec_ffn_out;
    memcpy(x, input_embed, dim * sizeof(float));

    int pos = ctx->kv_cache_len;

    /* Grow KV cache if needed */
    if (pos >= ctx->kv_cache_max) {
        if (kv_cache_grow(ctx, pos + 1024) != 0) return QWEN_TOKEN_IM_END;
    }

    if (ensure_rope_cache(ctx, pos + 1, head_dim, theta) != 0) {
        return QWEN_TOKEN_IM_END;
    }
    const float *rope_cos = ctx->rope_cache_cos + (size_t)pos * head_dim;
    const float *rope_sin = ctx->rope_cache_sin + (size_t)pos * head_dim;

    float scale = 1.0f / sqrtf((float)head_dim);

    for (int layer = 0; layer < cfg->dec_layers; layer++) {
        qwen_dec_layer_t *l = &dec->layers[layer];

        qwen_rms_norm(x_norm, x, l->input_norm, 1, dim, eps);
        qwen_linear_nobias_bf16_qkv(q, k, v, x_norm,
                                    l->wq_weight_bf16,
                                    l->wk_weight_bf16,
                                    l->wv_weight_bf16,
                                    dim, q_dim, kv_dim);

        /* Per-head Q/K RMSNorm */
        qwen_rms_norm_per_head(q, l->q_norm_weight, 1, n_heads, head_dim, eps);
        qwen_rms_norm_per_head(k, l->k_norm_weight, 1, n_kv_heads, head_dim, eps);

        /* Apply NeoX RoPE */
        qwen_apply_rope_neox(q, rope_cos, rope_sin, 1, n_heads, head_dim);
        qwen_apply_rope_neox(k, rope_cos, rope_sin, 1, n_kv_heads, head_dim);

        memcpy(kv_cache_k_at(ctx, layer, pos), k, kv_dim * sizeof(float));
        memcpy(kv_cache_v_at(ctx, layer, pos), v, kv_dim * sizeof(float));

        int total_seq = pos + 1;
        float *full_k = kv_cache_k_at(ctx, layer, 0);
        float *full_v = kv_cache_v_at(ctx, layer, 0);

        qwen_causal_attention(attn_out, q, full_k, full_v,
                               1, total_seq, n_heads, n_kv_heads,
                               head_dim, scale, pos);

        qwen_linear_nobias_bf16(proj_out, attn_out, l->wo_weight_bf16, 1, q_dim, dim);
        qwen_add_inplace(x, proj_out, dim);

        qwen_rms_norm(x_norm, x, l->post_attn_norm, 1, dim, eps);

        /* Fused gate+up matvec: one pass over x_norm, output interleaved [g0,u0,g1,u1,...] */
        qwen_linear_nobias_bf16(gate_buf, x_norm, l->gate_up_fused_bf16,
                                 1, dim, 2 * intermediate);
        /* In-place for seq=1: gate_buf[0:inter] receives SwiGLU output. */
        qwen_swiglu_multiply(gate_buf, gate_buf, 1, intermediate);
        qwen_linear_nobias_bf16(ffn_out, gate_buf, l->down_weight_bf16, 1, intermediate, dim);
        qwen_add_inplace(x, ffn_out, dim);
    }

    ctx->kv_cache_len = pos + 1;

    /* Final norm + streaming argmax (no logits buffer needed) */
    qwen_rms_norm(x, x, dec->norm, 1, dim, eps);
    return qwen_argmax_matvec_bf16(x, dec->tok_embeddings_bf16, dim, cfg->vocab_size);
}

#endif /* CPU backend */
