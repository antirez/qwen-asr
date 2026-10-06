/*
 * main.c - CLI entry point for Qwen3-ASR
 *
 * Usage: qwen_asr -d <model_dir> -i <input.wav> [options]
 */

#include "qwen_asr.h"
#include "qwen_asr_audio.h"
#include "qwen_asr_kernels.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

/* Token streaming callback: print each piece as it's decoded */
static void stream_token(const char *piece, void *userdata) {
    (void)userdata;
    fputs(piece, stdout);
    fflush(stdout);
}

/* ========================================================================
 * Timeline output (SRT / WebVTT / JSONL / plain text)
 * ======================================================================== */

enum {
    TIMELINE_FMT_SRT = 0,
    TIMELINE_FMT_VTT,
    TIMELINE_FMT_JSONL,
    TIMELINE_FMT_TXT
};

static int parse_timeline_format(const char *s) {
    if (strcmp(s, "srt") == 0) return TIMELINE_FMT_SRT;
    if (strcmp(s, "vtt") == 0) return TIMELINE_FMT_VTT;
    if (strcmp(s, "jsonl") == 0) return TIMELINE_FMT_JSONL;
    if (strcmp(s, "txt") == 0) return TIMELINE_FMT_TXT;
    return -1;
}

/* Infer the output format from a filename extension (default SRT). */
static int infer_timeline_format(const char *path) {
    const char *dot = strrchr(path, '.');
    if (!dot || dot == path) return TIMELINE_FMT_SRT;
    if (strcasecmp(dot, ".vtt") == 0) return TIMELINE_FMT_VTT;
    if (strcasecmp(dot, ".jsonl") == 0 || strcasecmp(dot, ".json") == 0)
        return TIMELINE_FMT_JSONL;
    if (strcasecmp(dot, ".txt") == 0) return TIMELINE_FMT_TXT;
    return TIMELINE_FMT_SRT;
}

/* Format seconds as HH:MM:SS,mmm (SRT, comma separator) or HH:MM:SS.mmm. */
static void timeline_fmt_time(char *buf, size_t n, float t, int comma_ms) {
    if (t < 0.0f) t = 0.0f;
    long ms = (long)(t * 1000.0f + 0.5f);
    int h = (int)(ms / 3600000L);
    int m = (int)((ms / 60000L) % 60L);
    int s = (int)((ms / 1000L) % 60L);
    ms %= 1000L;
    snprintf(buf, n, "%02d:%02d:%02d%c%03ld", h, m, s, comma_ms ? ',' : '.', ms);
}

/* Write a JSON string body (no surrounding quotes) with minimal escaping. */
static void timeline_json_escape(FILE *f, const char *s) {
    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;
        switch (c) {
        case '"': fputs("\\\"", f); break;
        case '\\': fputs("\\\\", f); break;
        case '\n': fputs("\\n", f); break;
        case '\r': fputs("\\r", f); break;
        case '\t': fputs("\\t", f); break;
        default: fputc(c, f); break;
        }
    }
}

static int write_timeline_file(const char *path, int fmt,
                               const qwen_timeline_entry_t *entries, int n) {
    FILE *f = (strcmp(path, "-") == 0) ? stdout : fopen(path, "w");
    if (!f) return -1;
    if (fmt == TIMELINE_FMT_SRT || fmt == TIMELINE_FMT_VTT) {
        if (fmt == TIMELINE_FMT_VTT) fprintf(f, "WEBVTT\n\n");
        for (int i = 0; i < n; i++) {
            char a[16], b[16];
            timeline_fmt_time(a, sizeof(a), entries[i].start_sec, fmt == TIMELINE_FMT_SRT);
            timeline_fmt_time(b, sizeof(b), entries[i].end_sec, fmt == TIMELINE_FMT_SRT);
            fprintf(f, "%d\n%s --> %s\n%s\n\n", i + 1, a, b, entries[i].text);
        }
    } else if (fmt == TIMELINE_FMT_JSONL) {
        for (int i = 0; i < n; i++) {
            fprintf(f, "{\"start\": %.3f, \"end\": %.3f, \"text\": \"",
                    entries[i].start_sec, entries[i].end_sec);
            timeline_json_escape(f, entries[i].text);
            fputs("\"}\n", f);
        }
    } else {
        for (int i = 0; i < n; i++) {
            char a[16], b[16];
            timeline_fmt_time(a, sizeof(a), entries[i].start_sec, 0);
            timeline_fmt_time(b, sizeof(b), entries[i].end_sec, 0);
            fprintf(f, "[%s - %s] %s\n", a, b, entries[i].text);
        }
    }
    if (f != stdout) fclose(f);
    return 0;
}

/* Parse --past-text value.
 * out_mode:  1=yes, 0=no, -1=auto */
static int parse_past_text_mode(const char *s, int *out_mode) {
    char buf[16];
    size_t n = strlen(s);
    if (n == 0 || n >= sizeof(buf)) return -1;
    for (size_t i = 0; i < n; i++) {
        buf[i] = (char)tolower((unsigned char)s[i]);
    }
    buf[n] = '\0';

    if (strcmp(buf, "yes") == 0) {
        *out_mode = 1;
        return 0;
    }
    if (strcmp(buf, "no") == 0) {
        *out_mode = 0;
        return 0;
    }
    if (strcmp(buf, "auto") == 0) {
        *out_mode = -1;
        return 0;
    }
    return -1;
}

static void usage(const char *prog) {
#ifdef USE_ROCM
    fprintf(stderr, "qwen_asr — Qwen3-ASR speech-to-text (C + ROCm/HIP)\n\n");
#elif defined(USE_CUDA)
    fprintf(stderr, "qwen_asr — Qwen3-ASR speech-to-text (C + CUDA)\n\n");
#else
    fprintf(stderr, "qwen_asr — Qwen3-ASR speech-to-text (pure C)\n\n");
#endif
    fprintf(stderr, "Usage: %s -d <model_dir> (-i <input.wav> | --stdin) [options]\n\n", prog);
    fprintf(stderr, "Required:\n");
    fprintf(stderr, "  -d <dir>      Model directory (with *.safetensors, vocab.json)\n");
    fprintf(stderr, "  -i <file>     Input WAV file (16-bit PCM, any sample rate)\n");
    fprintf(stderr, "  --stdin       Read audio from stdin (auto-detect WAV or raw s16le 16kHz mono)\n");
    fprintf(stderr, "\nOptions:\n");
    fprintf(stderr, "  --list-languages           Print supported languages and exit\n");
#if defined(USE_ROCM) || defined(USE_CUDA)
    fprintf(stderr, "  --precision <q8|fp16>      Decoder weights (default: q8)\n");
#endif
    fprintf(stderr, "  -t <n>        Number of threads (default: all CPUs)\n");
#if defined(USE_ROCM) || defined(USE_CUDA)
    fprintf(stderr, "  -S <secs>     Segment target seconds (default: 20; 0 = full-audio decode)\n");
#else
    fprintf(stderr, "  -S <secs>     Segment target seconds (default: 0 = full-audio decode)\n");
#endif
    fprintf(stderr, "  -W <secs>     Segment-cutting silence search window ± seconds (default: 3.0)\n");
    fprintf(stderr, "  --stream      Streaming mode: process in chunks with prefix rollback\n");
    fprintf(stderr, "  --stream-max-new-tokens <n>  Max generated tokens per stream step (default: 32)\n");
    fprintf(stderr, "  --enc-window-sec <secs>    Encoder attention window in seconds (1..8, default 8)\n");
    fprintf(stderr, "  --past-text <yes|no|auto>  Reuse previously decoded text as context for the next\n");
    fprintf(stderr, "                             segment/chunk (continuity bias; auto=yes for --stream)\n");
    fprintf(stderr, "  --skip-silence              Drop long silent spans before inference (off by default)\n");
    fprintf(stderr, "  --timeline-file <path>     Write cue timestamps (offline modes; not --stream/--skip-silence)\n");
    fprintf(stderr, "                             Format from extension (.srt/.vtt/.jsonl/.txt, default srt);\n");
    fprintf(stderr, "                             \"-\" = stdout. Uses -S 20 unless -S is set explicitly\n");
    fprintf(stderr, "  --timeline-format <fmt>    Override timeline format: srt|vtt|jsonl|txt\n");
    fprintf(stderr, "  --timeline-max-cue-sec <s> Max cue length in seconds (default: 7)\n");
    fprintf(stderr, "  --prompt <text>            System prompt for biasing (example: \"Preserve spelling: CPU, CUDA, PostgreSQL, Redis\")\n");
    fprintf(stderr, "  --language <lang>          Force output language via token conditioning\n");
    fprintf(stderr, "                             (usually auto-detected if omitted)\n");
    fprintf(stderr, "  --monitor     Show inline Unicode symbols on stderr (streaming diagnostics)\n");
    fprintf(stderr, "  --debug       Debug output (per-layer details)\n");
    fprintf(stderr, "  --silent      No status output (only final transcription on stdout)\n");
    fprintf(stderr, "                 with -i + --stream, uses non-interactive final refinement\n");
    fprintf(stderr, "  -h            Show this help\n");
}

int main(int argc, char **argv) {
    const char *model_dir = NULL;
    const char *input_wav = NULL;
    int verbosity = 1;
    int use_stdin = 0;
    int n_threads = 0; /* 0 = auto-detect */
    float segment_sec = -1; /* -1 = use backend default */
    float search_sec = -1;  /* -1 = use default (3) */
    int stream_mode = 0;
    int stream_max_new_tokens = -1; /* -1 = use default (32) */
    float enc_window_sec = -1;   /* -1 = use default (8s) */
    const char *prompt_text = NULL;
    const char *force_language = NULL;
    int past_text_conditioning_mode = -1; /* -1 auto, 0 off, 1 on */
    int skip_silence = 0;
    int emit_tokens = 1;
    const char *timeline_file = NULL;
    const char *timeline_format = NULL;
    float timeline_max_cue_sec = -1.0f; /* -1 = use default (7) */

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            input_wav = argv[++i];
        } else if (strcmp(argv[i], "--list-languages") == 0) {
            puts(qwen_supported_languages_csv());
            return 0;
        } else if (strcmp(argv[i], "--precision") == 0) {
            if (i + 1 >= argc ||
                (strcmp(argv[i + 1], "fp16") && strcmp(argv[i + 1], "q8"))) {
                fprintf(stderr, "Error: --precision requires q8 or fp16\n");
                return 1;
            }
#if defined(USE_ROCM) || defined(USE_CUDA)
            const char *value = argv[++i];
#ifdef USE_CUDA
            const char *precision_env = "QWEN_CUDA_Q8";
#else
            const char *precision_env = "QWEN_ROCM_Q8";
#endif
            if (setenv(precision_env, strcmp(value, "q8") == 0 ? "1" : "0", 1) != 0) {
                perror("Cannot set GPU precision");
                return 1;
            }
#else
            fprintf(stderr, "Error: --precision requires make rocm or make cuda\n");
            return 1;
#endif
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            n_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-S") == 0 && i + 1 < argc) {
            segment_sec = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "-W") == 0 && i + 1 < argc) {
            search_sec = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--stream") == 0) {
            stream_mode = 1;
        } else if (strcmp(argv[i], "--stream-max-new-tokens") == 0 && i + 1 < argc) {
            stream_max_new_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--enc-window-sec") == 0 && i + 1 < argc) {
            enc_window_sec = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--past-text") == 0 && i + 1 < argc) {
            const char *mode = argv[++i];
            if (parse_past_text_mode(mode, &past_text_conditioning_mode) != 0) {
                fprintf(stderr, "Error: --past-text must be one of yes|no|auto, got '%s'\n", mode);
                return 1;
            }
        } else if (strcmp(argv[i], "--past-text") == 0) {
            fprintf(stderr, "Error: --past-text requires an argument: yes|no|auto\n");
            return 1;
        } else if (strcmp(argv[i], "--skip-silence") == 0) {
            skip_silence = 1;
        } else if (strcmp(argv[i], "--timeline-file") == 0 && i + 1 < argc) {
            timeline_file = argv[++i];
        } else if (strcmp(argv[i], "--timeline-file") == 0) {
            fprintf(stderr, "Error: --timeline-file requires a path (or - for stdout)\n");
            return 1;
        } else if (strcmp(argv[i], "--timeline-format") == 0 && i + 1 < argc) {
            timeline_format = argv[++i];
        } else if (strcmp(argv[i], "--timeline-max-cue-sec") == 0 && i + 1 < argc) {
            timeline_max_cue_sec = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt_text = argv[++i];
        } else if (strcmp(argv[i], "--language") == 0 && i + 1 < argc) {
            force_language = argv[++i];
        } else if (strcmp(argv[i], "--stdin") == 0) {
            use_stdin = 1;
        } else if (strcmp(argv[i], "--monitor") == 0) {
            qwen_monitor = 1;
        } else if (strcmp(argv[i], "--debug") == 0) {
            verbosity = 2;
        } else if (strcmp(argv[i], "--silent") == 0) {
            verbosity = 0;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    if (!model_dir || (!input_wav && !use_stdin)) {
        usage(argv[0]);
        return 1;
    }
    if (enc_window_sec >= 0 && (enc_window_sec < 1.0f || enc_window_sec > 8.0f)) {
        fprintf(stderr, "Error: --enc-window-sec must be in [1, 8], got %.3f\n", enc_window_sec);
        return 1;
    }
    if (stream_max_new_tokens == 0 || stream_max_new_tokens < -1) {
        fprintf(stderr, "Error: --stream-max-new-tokens must be > 0\n");
        return 1;
    }
    if (input_wav && use_stdin) {
        fprintf(stderr, "Error: -i and --stdin are mutually exclusive\n");
        return 1;
    }
    if (timeline_file && stream_mode) {
        fprintf(stderr, "Error: --timeline-file is not supported with --stream\n");
        return 1;
    }
    if (timeline_file && skip_silence) {
        fprintf(stderr, "Error: --timeline-file cannot be combined with --skip-silence\n");
        return 1;
    }
    int timeline_fmt = -1;
    if (timeline_format) {
        timeline_fmt = parse_timeline_format(timeline_format);
        if (timeline_fmt < 0) {
            fprintf(stderr, "Error: --timeline-format must be one of srt|vtt|jsonl|txt, got '%s'\n",
                    timeline_format);
            return 1;
        }
    } else if (timeline_file) {
        timeline_fmt = infer_timeline_format(timeline_file);
    }
    if (timeline_file && segment_sec < 0) {
        /* Timeline granularity comes from segment boundaries. */
        segment_sec = 20.0f;
        if (verbosity >= 1)
            fprintf(stderr, "Timeline: enabling segmented mode (-S 20) for timestamp granularity\n");
    }

    qwen_verbose = verbosity;
    emit_tokens = (verbosity > 0);

    /* Initialize thread pool */
    if (n_threads <= 0) n_threads = qwen_get_num_cpus();
    qwen_set_threads(n_threads);

    /* Load model */
    qwen_ctx_t *ctx = qwen_load(model_dir);
    if (!ctx) {
        fprintf(stderr, "Failed to load model from %s\n", model_dir);
        return 1;
    }

    /* Apply segmentation settings */
    if (segment_sec >= 0) ctx->segment_sec = segment_sec;
    if (search_sec >= 0) ctx->search_sec = search_sec;
    if (enc_window_sec >= 0) {
        int window_frames = (int)(enc_window_sec * 100.0f + 0.5f);
        if (window_frames < 100) window_frames = 100;
        if (window_frames > 800) window_frames = 800;
        ctx->config.enc_n_window_infer = window_frames;
    }
    if (stream_max_new_tokens > 0) ctx->stream_max_new_tokens = stream_max_new_tokens;
    if (past_text_conditioning_mode >= 0)
        ctx->past_text_conditioning = past_text_conditioning_mode;
    else if (stream_mode)
        /* Official streaming path uses prefix rollback by default.
         * Keep segmented mode default unchanged (off). */
        ctx->past_text_conditioning = 1;
    if (skip_silence) ctx->skip_silence = 1;
    if (timeline_file) {
        qwen_set_timeline(ctx, 1);
        if (timeline_max_cue_sec > 0) qwen_set_timeline_max_cue_sec(ctx, timeline_max_cue_sec);
    }
    if (prompt_text && qwen_set_prompt(ctx, prompt_text) != 0) {
        fprintf(stderr, "Failed to set --prompt text\n");
        qwen_free(ctx);
        return 1;
    }
    if (force_language && qwen_set_force_language(ctx, force_language) != 0) {
        fprintf(stderr, "Unsupported language for --language: %s\n",
                force_language);
        fprintf(stderr, "Supported languages: %s\n", qwen_supported_languages_csv());
        qwen_free(ctx);
        return 1;
    }

    /* Stream tokens to stdout only in non-silent mode.
     * In silent mode we print the final string returned by the API. */
    if (emit_tokens) qwen_set_token_callback(ctx, stream_token, NULL);
    else qwen_set_token_callback(ctx, NULL, NULL);

    /* Transcribe */
    char *text = NULL;
    if (stream_mode && use_stdin) {
        /* Live incremental streaming from stdin */
        qwen_live_audio_t *live = qwen_live_audio_start_stdin();
        if (live) {
            text = qwen_transcribe_stream_live(ctx, live);
            qwen_live_audio_free(live);
        }
    } else if (stream_mode) {
        /* File-based streaming: load audio fully, then stream-transcribe */
        int ns = 0;
        float *samps = qwen_load_wav(input_wav, &ns);
        if (samps) {
            text = qwen_transcribe_stream(ctx, samps, ns);
            free(samps);
        }
    } else if (use_stdin) {
        text = qwen_transcribe_stdin(ctx);
    } else {
        text = qwen_transcribe(ctx, input_wav);
    }

    if (text) {
        if (emit_tokens) printf("\n");
        else printf("%s\n", text);
        free(text);
    } else {
        fprintf(stderr, "Transcription failed\n");
        qwen_free(ctx);
        return 1;
    }

    if (timeline_file) {
        const qwen_timeline_entry_t *entries = NULL;
        int n_entries = qwen_get_timeline(ctx, &entries);
        if (n_entries <= 0) {
            fprintf(stderr, "Timeline: no cues collected\n");
        } else if (write_timeline_file(timeline_file, timeline_fmt, entries, n_entries) != 0) {
            fprintf(stderr, "Timeline: failed to write %s\n", timeline_file);
        } else if (verbosity >= 1) {
            fprintf(stderr, "Timeline: %d cues written to %s\n", n_entries, timeline_file);
        }
    }

    if (verbosity >= 1) {
        double tokens_per_sec = 0.0;
        if (ctx->perf_total_ms > 0) {
            tokens_per_sec = (1000.0 * (double)ctx->perf_text_tokens) / ctx->perf_total_ms;
        }
        fprintf(stderr,
                "Inference: %.0f ms, %d text tokens (%.2f tok/s, encoding: %.0fms, decoding: %.0fms)\n",
                ctx->perf_total_ms, ctx->perf_text_tokens, tokens_per_sec,
                ctx->perf_encode_ms, ctx->perf_decode_ms);
        if (ctx->perf_audio_ms > 0 && ctx->perf_total_ms > 0) {
            double audio_s = ctx->perf_audio_ms / 1000.0;
            double infer_s = ctx->perf_total_ms / 1000.0;
            double realtime_x = audio_s / infer_s;
            fprintf(stderr, "Audio: %.1f s processed in %.1f s (%.2fx realtime)\n",
                    audio_s, infer_s, realtime_x);
        }
    }

    qwen_free(ctx);
    return 0;
}
