/*
 * bench.c — Qwen3-ASR inference benchmark
 *
 * Runs the encoder and/or decoder on audio and reports timing statistics
 * across multiple runs.
 *
 * Audio source (choose one):
 *   (default)  synthetic silence generated internally
 *   -i FILE    real WAV file (16-bit PCM, 16 kHz, mono; or any WAV that
 *              qwen_load_wav() can handle)
 *
 * Modes:
 *   0 (default) — full pipeline: encoder + decoder via qwen_transcribe_audio()
 *   1           — encoder only: qwen_encoder_forward() on a mel spectrogram
 */

#include "qwen_asr.h"
#include "qwen_asr_audio.h"
#include "qwen_asr_kernels.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ── Timing ──────────────────────────────────────────────────────────────── */

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1.0e6;
}

/* ── CLI ─────────────────────────────────────────────────────────────────── */

static void usage(const char *prog) {
    fprintf(stderr, "qwen_asr_bench — Qwen3-ASR inference benchmark\n\n");
    fprintf(stderr, "Usage: %s -d <model_dir> [options]\n\n", prog);
    fprintf(stderr, "Required:\n");
    fprintf(stderr, "  -d DIR        Model directory (*.safetensors + vocab.json)\n");
    fprintf(stderr, "\nOptions:\n");
    fprintf(stderr, "  -t N          Number of threads (default: auto)\n");
    fprintf(stderr, "  -n N          Number of benchmark runs (default: 5)\n");
    fprintf(stderr, "  -i FILE       Audio file to use (default: synthetic silence)\n");
    fprintf(stderr, "  -s N          Synthetic audio length in seconds (default: 5)\n");
    fprintf(stderr, "                Ignored when -i is given.\n");
    fprintf(stderr, "  -w N          What to benchmark:\n");
    fprintf(stderr, "                  0 (default) — full pipeline (encode + decode)\n");
    fprintf(stderr, "                  1           — encoder only\n");
    fprintf(stderr, "  -h            Show this help\n");
}

/* ── Stats helper ─────────────────────────────────────────────────────────── */

static void stats(const double *arr, int n,
                  double *out_min, double *out_mean, double *out_max) {
    double mn = arr[0], mx = arr[0], sum = 0.0;
    for (int i = 0; i < n; i++) {
        if (arr[i] < mn) mn = arr[i];
        if (arr[i] > mx) mx = arr[i];
        sum += arr[i];
    }
    *out_min  = mn;
    *out_mean = sum / n;
    *out_max  = mx;
}

/* ── Benchmark: full pipeline ─────────────────────────────────────────────── */

static int bench_full(qwen_ctx_t *ctx, int n_runs,
                      const float *samples, int n_samples,
                      double audio_ms, const char *src_desc) {
    double *total_ms  = (double *)malloc(n_runs * sizeof(double));
    double *encode_ms = (double *)malloc(n_runs * sizeof(double));
    double *decode_ms = (double *)malloc(n_runs * sizeof(double));
    int    *n_tokens  = (int    *)malloc(n_runs * sizeof(int));

    fprintf(stderr, "Mode: full pipeline  |  %d run(s)  |  %.1f s  [%s]\n\n",
            n_runs, audio_ms / 1000.0, src_desc);

    /* Warm-up run (not measured) */
    fprintf(stderr, "  warmup ...\n");
    ctx->kv_cache_len = 0;
    char *warmup = qwen_transcribe_audio(ctx, samples, n_samples);
    if (warmup) free(warmup);

    for (int i = 0; i < n_runs; i++) {
        ctx->kv_cache_len = 0;   /* reset KV cache for a clean run */
        double t0     = now_ms();
        char  *result = qwen_transcribe_audio(ctx, samples, n_samples);
        double t1     = now_ms();
        if (result) free(result);

        total_ms [i] = t1 - t0;
        encode_ms[i] = ctx->perf_encode_ms;
        decode_ms[i] = ctx->perf_decode_ms;
        n_tokens [i] = ctx->perf_text_tokens;

        fprintf(stderr, "  run %d/%d:  total=%6.0f ms  enc=%6.0f ms  dec=%6.0f ms"
                        "  tokens=%d  rt=%.2fx\n",
                i + 1, n_runs,
                total_ms[i], encode_ms[i], decode_ms[i], n_tokens[i],
                total_ms[i] / audio_ms);
    }

    double tot_min,  tot_mean,  tot_max;
    double enc_min,  enc_mean,  enc_max;
    double dec_min,  dec_mean,  dec_max;
    stats(total_ms,  n_runs, &tot_min, &tot_mean, &tot_max);
    stats(encode_ms, n_runs, &enc_min, &enc_mean, &enc_max);
    stats(decode_ms, n_runs, &dec_min, &dec_mean, &dec_max);

    fprintf(stderr, "\n");
    fprintf(stderr, "%-14s  %8s  %8s  %8s\n", "",        "min",    "mean",   "max");
    fprintf(stderr, "%-14s  %8.1f  %8.1f  %8.1f  ms\n",  "total",  tot_min,  tot_mean,  tot_max);
    fprintf(stderr, "%-14s  %8.1f  %8.1f  %8.1f  ms\n",  "encode", enc_min,  enc_mean,  enc_max);
    fprintf(stderr, "%-14s  %8.1f  %8.1f  %8.1f  ms\n",  "decode", dec_min,  dec_mean,  dec_max);
    fprintf(stderr, "%-14s  %8.2f  %8.2f  %8.2f  x RT\n","rt_factor",
            tot_min / audio_ms, tot_mean / audio_ms, tot_max / audio_ms);
    fprintf(stderr, "\n");

    free(total_ms);
    free(encode_ms);
    free(decode_ms);
    free(n_tokens);
    return 0;
}

/* ── Benchmark: encoder only ──────────────────────────────────────────────── */

static int bench_encoder(qwen_ctx_t *ctx, int n_runs,
                          const float *mel, int n_frames,
                          double audio_ms, const char *src_desc) {
    double *elapsed = (double *)malloc(n_runs * sizeof(double));

    fprintf(stderr, "Mode: encoder only  |  %d run(s)  |  %d frames (%.1f s)  [%s]\n\n",
            n_runs, n_frames, audio_ms / 1000.0, src_desc);

    /* Warm-up run (not measured) */
    int out_seq_len = 0;
    float *enc_out = qwen_encoder_forward(ctx, mel, n_frames, &out_seq_len);
    free(enc_out);

    for (int i = 0; i < n_runs; i++) {
        double t0 = now_ms();
        enc_out = qwen_encoder_forward(ctx, mel, n_frames, &out_seq_len);
        double t1 = now_ms();
        free(enc_out);

        elapsed[i] = t1 - t0;
        fprintf(stderr, "  run %d/%d:  enc=%6.0f ms  seq_len=%d\n",
                i + 1, n_runs, elapsed[i], out_seq_len);
    }

    double mn, mean, mx;
    stats(elapsed, n_runs, &mn, &mean, &mx);

    fprintf(stderr, "\n");
    fprintf(stderr, "%-14s  %8s  %8s  %8s\n", "",        "min",  "mean", "max");
    fprintf(stderr, "%-14s  %8.1f  %8.1f  %8.1f  ms\n",  "encode", mn, mean, mx);
    fprintf(stderr, "%-14s  %8.2f  %8.2f  %8.2f  ms/layer\n", "per layer",
            mn   / ctx->config.enc_layers,
            mean / ctx->config.enc_layers,
            mx   / ctx->config.enc_layers);
    fprintf(stderr, "\n");

    free(elapsed);
    return 0;
}

/* ── main ────────────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    const char *model_dir  = NULL;
    const char *audio_path = NULL;
    int n_threads = 0;  /* 0 = auto */
    int n_runs    = 5;
    int audio_sec = 5;
    int what      = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            audio_path = argv[++i];
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            n_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n_runs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            audio_sec = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-w") == 0 && i + 1 < argc) {
            what = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    if (!model_dir) {
        usage(argv[0]);
        return 1;
    }
    if (n_runs < 1) {
        fprintf(stderr, "error: -n must be >= 1\n");
        return 1;
    }
    if (!audio_path && (audio_sec < 1 || audio_sec > 300)) {
        fprintf(stderr, "error: -s must be in [1, 300]\n");
        return 1;
    }
    if (what != 0 && what != 1) {
        fprintf(stderr, "error: -w must be 0 or 1\n");
        return 1;
    }

    /* Thread pool */
    if (n_threads > 0)
        qwen_set_threads(n_threads);
    else
        n_threads = qwen_get_num_cpus();

    fprintf(stderr, "system_info: n_threads = %d / %d\n\n",
            n_threads, qwen_get_num_cpus());

    /* Load model */
    fprintf(stderr, "Loading model from %s ...\n", model_dir);
    qwen_ctx_t *ctx = qwen_load(model_dir);
    if (!ctx) {
        fprintf(stderr, "error: failed to load model from '%s'\n", model_dir);
        return 1;
    }

    /* Silence per-inference status lines */
    qwen_verbose = 0;

    int ret = 0;

    if (what == 0) {
        /* Full pipeline — needs raw samples */
        float      *samples  = NULL;
        int         n_samples;
        double      audio_ms;
        const char *src_desc;

        if (audio_path) {
            samples = qwen_load_wav(audio_path, &n_samples);
            if (!samples) {
                fprintf(stderr, "error: failed to load audio from '%s'\n", audio_path);
                qwen_free(ctx);
                return 1;
            }
            audio_ms = n_samples * 1000.0 / QWEN_SAMPLE_RATE;
            src_desc = audio_path;
        } else {
            n_samples = audio_sec * QWEN_SAMPLE_RATE;
            samples   = (float *)calloc(n_samples, sizeof(float));
            if (!samples) {
                fprintf(stderr, "error: out of memory\n");
                qwen_free(ctx);
                return 1;
            }
            audio_ms = audio_sec * 1000.0;
            src_desc = "synthetic silence";
        }

        ret = bench_full(ctx, n_runs, samples, n_samples, audio_ms, src_desc);
        free(samples);

    } else {
        /* Encoder only — needs mel spectrogram */
        float      *mel = NULL;
        int         n_frames;
        double      audio_ms;
        const char *src_desc;

        if (audio_path) {
            int    n_samples;
            float *wav = qwen_load_wav(audio_path, &n_samples);
            if (!wav) {
                fprintf(stderr, "error: failed to load audio from '%s'\n", audio_path);
                qwen_free(ctx);
                return 1;
            }
            audio_ms = n_samples * 1000.0 / QWEN_SAMPLE_RATE;
            src_desc = audio_path;
            mel = qwen_mel_spectrogram(wav, n_samples, &n_frames);
            free(wav);
            if (!mel) {
                fprintf(stderr, "error: mel spectrogram computation failed\n");
                qwen_free(ctx);
                return 1;
            }
        } else {
            n_frames = (audio_sec * QWEN_SAMPLE_RATE) / QWEN_HOP_LENGTH;
            mel      = (float *)calloc(QWEN_MEL_BINS * n_frames, sizeof(float));
            if (!mel) {
                fprintf(stderr, "error: out of memory\n");
                qwen_free(ctx);
                return 1;
            }
            audio_ms = audio_sec * 1000.0;
            src_desc = "synthetic silence";
        }

        ret = bench_encoder(ctx, n_runs, mel, n_frames, audio_ms, src_desc);
        free(mel);
    }

    qwen_free(ctx);
    return ret;
}
