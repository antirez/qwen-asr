/*
 * server.cpp — Qwen3-ASR HTTP inference server
 *
 * Single-process HTTP server exposing the Qwen3-ASR inference engine
 * over a simple REST API. Uses cpp-httplib for HTTP transport.
 *
 * Endpoints:
 *   GET  /          — built-in HTML page (or static files if public/ has index.html)
 *   POST /inference — transcribe uploaded audio, returns JSON or plain text
 *   POST /load      — hot-swap the loaded model directory at runtime
 *   GET  /health    — readiness probe
 *
 * Requests are serialized: only one inference runs at a time.
 */

#include "httplib.h"
#include "json.hpp"

extern "C" {
#include "../../qwen_asr.h"
#include "../../qwen_asr_audio.h"
#include "../../qwen_asr_kernels.h"
}

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>

#if defined(_WIN32)
#include <windows.h>
#endif

using namespace httplib;
using json = nlohmann::json;

// ─────────────────────────────────────────────────────────────────────────────
// Server state
// ─────────────────────────────────────────────────────────────────────────────

enum server_state {
    SERVER_STATE_LOADING_MODEL,
    SERVER_STATE_READY,
};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static std::string json_err(const std::string &msg) {
    return json{{"error", msg}}.dump();
}

// ─────────────────────────────────────────────────────────────────────────────
// Signal handling
// ─────────────────────────────────────────────────────────────────────────────

namespace {
std::function<void(int)> shutdown_handler;
std::atomic_flag is_terminating = ATOMIC_FLAG_INIT;

void signal_handler(int sig) {
    if (is_terminating.test_and_set()) {
        fprintf(stderr, "Received second interrupt, terminating immediately.\n");
        exit(1);
    }
    shutdown_handler(sig);
}
} // namespace

// ─────────────────────────────────────────────────────────────────────────────
// FFmpeg helpers
// ─────────────────────────────────────────────────────────────────────────────

static bool check_ffmpeg() {
    int r = system("ffmpeg -version > /dev/null 2>&1");
    if (r != 0) {
        fprintf(stderr, "error: ffmpeg not found on PATH\n");
        return false;
    }
    return true;
}

static std::string generate_temp_filename(const std::string &dir,
                                           const std::string &prefix) {
    auto now = std::chrono::system_clock::now();
    auto t   = std::chrono::system_clock::to_time_t(now);
    static std::mt19937 rng{std::random_device{}()};
    std::uniform_int_distribution<long long> dist(0, 1000000000LL);
    std::ostringstream ss;
    ss << dir << std::filesystem::path::preferred_separator
       << prefix << "-"
       << std::put_time(std::localtime(&t), "%Y%m%d-%H%M%S")
       << "-" << dist(rng) << ".wav";
    return ss.str();
}

static bool convert_to_wav(const std::string &path, std::string &err_resp) {
    std::string tmp = path + "_tmp.wav";
    std::string cmd = "ffmpeg -i \"" + path + "\" -y -ar 16000 -ac 1 -c:a pcm_s16le \""
                    + tmp + "\" > /dev/null 2>&1";
    if (system(cmd.c_str()) != 0) {
        err_resp = json_err("FFmpeg conversion failed.");
        return false;
    }
    remove(path.c_str());
    if (rename(tmp.c_str(), path.c_str()) != 0) {
        err_resp = json_err("Failed to rename converted file.");
        return false;
    }
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Parameters
// ─────────────────────────────────────────────────────────────────────────────

struct server_params {
    std::string hostname     = "127.0.0.1";
    std::string public_path  = "examples/server/public";
    std::string request_path = "";
    std::string tmp_dir      = ".";
    int32_t     port          = 8080;
    int32_t     read_timeout  = 600;
    int32_t     write_timeout = 600;
    bool        ffmpeg_converter = false;
};

struct qwen_server_params {
    std::string model_dir       = "";
    std::string language        = "";     // empty = auto-detect
    std::string prompt          = "";
    std::string response_format = "json"; // "json" | "text"
    int32_t     n_threads       = 0;      // 0 = auto
};

static void print_usage(const char *prog,
                        const qwen_server_params &qp,
                        const server_params &sp) {
    fprintf(stderr, "\nUsage: %s -d <model_dir> [options]\n\n", prog);
    fprintf(stderr, "Required:\n");
    fprintf(stderr, "  -d DIR, --model-dir DIR   Model directory (*.safetensors + vocab.json)\n");
    fprintf(stderr, "\nInference options:\n");
    fprintf(stderr, "  -t N, --threads N         [%d] Number of threads (0 = auto)\n",
            qp.n_threads);
    fprintf(stderr, "  --language LANG           [\"%s\"] Default forced language (empty = auto)\n",
            qp.language.c_str());
    fprintf(stderr, "                            Supported: %s\n",
            qwen_supported_languages_csv());
    fprintf(stderr, "  --prompt TEXT             [\"%s\"] Default system prompt\n",
            qp.prompt.c_str());
    fprintf(stderr, "\nServer options:\n");
    fprintf(stderr, "  --host HOST               [%s] Hostname / IP to bind\n",
            sp.hostname.c_str());
    fprintf(stderr, "  --port PORT               [%d] Port number\n", sp.port);
    fprintf(stderr, "  --public PATH             [%s] Directory for static files\n",
            sp.public_path.c_str());
    fprintf(stderr, "  --convert                 Accept non-WAV input; convert via ffmpeg\n");
    fprintf(stderr, "  --tmp-dir DIR             [%s] Temp directory for ffmpeg output\n",
            sp.tmp_dir.c_str());
    fprintf(stderr, "\n");
}

static bool parse_args(int argc, char **argv,
                       qwen_server_params &qp, server_params &sp) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0], qp, sp);
            exit(0);
        }
        else if ((arg == "-d" || arg == "--model-dir") && i + 1 < argc) { qp.model_dir  = argv[++i]; }
        else if ((arg == "-t" || arg == "--threads")   && i + 1 < argc) { qp.n_threads  = std::stoi(argv[++i]); }
        else if (arg == "--language"  && i + 1 < argc) { qp.language = argv[++i]; }
        else if (arg == "--prompt"    && i + 1 < argc) { qp.prompt   = argv[++i]; }
        else if (arg == "--host"      && i + 1 < argc) { sp.hostname    = argv[++i]; }
        else if (arg == "--port"      && i + 1 < argc) { sp.port        = std::stoi(argv[++i]); }
        else if (arg == "--public"    && i + 1 < argc) { sp.public_path = argv[++i]; }
        else if (arg == "--tmp-dir"   && i + 1 < argc) { sp.tmp_dir     = argv[++i]; }
        else if (arg == "--convert")                    { sp.ffmpeg_converter = true; }
        else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            print_usage(argv[0], qp, sp);
            return false;
        }
    }
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char **argv) {
    qwen_server_params qparams;
    server_params      sparams;

    if (!parse_args(argc, argv, qparams, sparams)) return 1;

    if (qparams.model_dir.empty()) {
        fprintf(stderr, "error: -d <model_dir> is required\n");
        print_usage(argv[0], qparams, sparams);
        return 1;
    }

    if (sparams.ffmpeg_converter && !check_ffmpeg()) return 1;

    // Show per-request timing summary on stderr (same as CLI default)
    qwen_verbose = 1;

    // Thread pool
    if (qparams.n_threads > 0)
        qwen_set_threads(qparams.n_threads);

    // Load model
    std::mutex              qwen_mutex;
    std::atomic<server_state> state{SERVER_STATE_LOADING_MODEL};

    fprintf(stderr, "Loading model from %s ...\n", qparams.model_dir.c_str());
    qwen_ctx_t *ctx = qwen_load(qparams.model_dir.c_str());
    if (!ctx) {
        fprintf(stderr, "error: failed to load model from '%s'\n",
                qparams.model_dir.c_str());
        return 1;
    }

    // Apply server-level defaults to context
    if (!qparams.language.empty()) {
        if (qwen_set_force_language(ctx, qparams.language.c_str()) != 0) {
            fprintf(stderr, "error: unsupported language '%s'\n  supported: %s\n",
                    qparams.language.c_str(), qwen_supported_languages_csv());
            qwen_free(ctx);
            return 1;
        }
    }
    if (!qparams.prompt.empty())
        qwen_set_prompt(ctx, qparams.prompt.c_str());

    state.store(SERVER_STATE_READY);

    // HTTP server
    auto svr = std::make_unique<httplib::Server>();
    svr->set_default_headers({
        {"Server",                       "qwen-asr-server"},
        {"Access-Control-Allow-Origin",  "*"},
        {"Access-Control-Allow-Headers", "content-type, authorization"},
    });

    // ── Built-in fallback page (used when public/ has no index.html) ──────
    std::string default_content =
        "<html><head><title>Qwen3-ASR Server</title>"
        "<meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width\">"
        "<style>body{font-family:sans-serif}pre{background:#f4f4f4;padding:1em}"
        "form label{display:block;margin:.6em 0}"
        "button{margin-top:.8em}</style></head><body>"
        "<h1>Qwen3-ASR Server</h1>"
        "<h2>POST /inference</h2><pre>"
        "curl 127.0.0.1:" + std::to_string(sparams.port) + "/inference \\\n"
        "  -F file=\"@audio.wav\" \\\n"
        "  -F response_format=\"json\"</pre>"
        "<h2>POST /load</h2><pre>"
        "curl 127.0.0.1:" + std::to_string(sparams.port) + "/load \\\n"
        "  -F model=\"/path/to/model_dir\"</pre>"
        "<h2>Try it</h2>"
        "<form action=\"/inference\" method=\"POST\" enctype=\"multipart/form-data\">"
        "<label>Audio file: <input type=\"file\" name=\"file\" accept=\"audio/*\" required></label>"
        "<label>Language (optional):"
        " <input type=\"text\" name=\"language\" placeholder=\"e.g. English\"></label>"
        "<label>Response format:"
        " <select name=\"response_format\">"
        "<option value=\"json\">JSON</option>"
        "<option value=\"text\">Text</option>"
        "</select></label>"
        "<button type=\"submit\">Transcribe</button>"
        "</form></body></html>";

    svr->Get(sparams.request_path + "/", [&](const Request &, Response &res) {
        res.set_content(default_content, "text/html; charset=utf-8");
        return false;
    });

    // ── CORS preflight ────────────────────────────────────────────────────
    svr->Options(sparams.request_path + "/inference",
                 [](const Request &, Response &) {});

    // ── POST /inference ───────────────────────────────────────────────────
    svr->Post(sparams.request_path + "/inference",
              [&](const Request &req, Response &res) {
        std::lock_guard<std::mutex> lock(qwen_mutex);

        // Per-request overrides (fall back to server-level defaults)
        std::string req_language        = qparams.language;
        std::string req_prompt          = qparams.prompt;
        std::string req_response_format = qparams.response_format;

        if (req.has_file("language"))        req_language        = req.get_file_value("language").content;
        if (req.has_file("prompt"))          req_prompt          = req.get_file_value("prompt").content;
        if (req.has_file("response_format")) req_response_format = req.get_file_value("response_format").content;

        // Validate and apply per-request language
        if (qwen_set_force_language(ctx, req_language.empty() ? nullptr
                                                              : req_language.c_str()) != 0) {
            res.set_content(json_err("unsupported language: " + req_language),
                            "application/json");
            goto restore_defaults;
        }
        qwen_set_prompt(ctx, req_prompt.empty() ? nullptr : req_prompt.c_str());

        {
            if (!req.has_file("file")) {
                res.set_content(json_err("no 'file' field in the request"),
                                "application/json");
                goto restore_defaults;
            }

            const auto &audio_file = req.get_file_value("file");
            fprintf(stderr, "Received: %s (%zu bytes)\n",
                    audio_file.filename.c_str(), audio_file.content.size());

            // Load audio into float samples
            float *samples   = nullptr;
            int    n_samples = 0;

            if (sparams.ffmpeg_converter) {
                const std::string tmp =
                    generate_temp_filename(sparams.tmp_dir, "qwen-server");
                {
                    std::ofstream f(tmp, std::ios::binary);
                    f.write(audio_file.content.data(),
                            (std::streamsize)audio_file.content.size());
                }
                std::string err;
                if (!convert_to_wav(tmp, err)) {
                    remove(tmp.c_str());
                    res.set_content(err, "application/json");
                    goto restore_defaults;
                }
                samples = qwen_load_wav(tmp.c_str(), &n_samples);
                remove(tmp.c_str());
            } else {
                samples = qwen_parse_wav_buffer(
                    reinterpret_cast<const uint8_t *>(audio_file.content.data()),
                    audio_file.content.size(),
                    &n_samples);
            }

            if (!samples) {
                res.set_content(json_err("failed to read audio data"),
                                "application/json");
                goto restore_defaults;
            }

            // Transcribe
            char *result = qwen_transcribe_audio(ctx, samples, n_samples);
            free(samples);

            if (!result) {
                res.status = 500;
                res.set_content(json_err("failed to process audio"),
                                "application/json");
                goto restore_defaults;
            }

            std::string text(result);
            free(result);

            if (req_response_format == "text") {
                res.set_content(text, "text/plain; charset=utf-8");
            } else {
                double tok_s = (ctx->perf_total_ms > 0 && ctx->perf_text_tokens > 0)
                               ? ctx->perf_text_tokens / (ctx->perf_total_ms / 1000.0) : 0.0;
                double rt_factor = (ctx->perf_audio_ms > 0)
                                   ? ctx->perf_total_ms / ctx->perf_audio_ms : 0.0;
                json jres = {
                    {"text",       text},
                    {"total_ms",   ctx->perf_total_ms},
                    {"encode_ms",  ctx->perf_encode_ms},
                    {"decode_ms",  ctx->perf_decode_ms},
                    {"tokens",     ctx->perf_text_tokens},
                    {"tok_s",      tok_s},
                    {"rt_factor",  rt_factor},
                };
                res.set_content(jres.dump(), "application/json");
            }
        }

    restore_defaults:
        // Always restore server-level defaults for the next request
        qwen_set_force_language(ctx, qparams.language.empty() ? nullptr
                                                              : qparams.language.c_str());
        qwen_set_prompt(ctx, qparams.prompt.empty() ? nullptr : qparams.prompt.c_str());
    });

    // ── POST /load ────────────────────────────────────────────────────────
    svr->Post(sparams.request_path + "/load",
              [&](const Request &req, Response &res) {
        std::lock_guard<std::mutex> lock(qwen_mutex);
        state.store(SERVER_STATE_LOADING_MODEL);

        if (!req.has_file("model")) {
            res.set_content(json_err("no 'model' field in the request"),
                            "application/json");
            state.store(SERVER_STATE_READY);
            return;
        }

        std::string new_model_dir = req.get_file_value("model").content;
        fprintf(stderr, "Loading new model from %s ...\n", new_model_dir.c_str());

        qwen_free(ctx);
        ctx = qwen_load(new_model_dir.c_str());
        if (!ctx) {
            fprintf(stderr, "error: failed to load model from '%s', exiting\n",
                    new_model_dir.c_str());
            exit(1); // no fallback, same behavior as whisper-server
        }

        qparams.model_dir = new_model_dir;
        // Re-apply server-level defaults to the new context
        if (!qparams.language.empty())
            qwen_set_force_language(ctx, qparams.language.c_str());
        if (!qparams.prompt.empty())
            qwen_set_prompt(ctx, qparams.prompt.c_str());

        state.store(SERVER_STATE_READY);
        res.set_content("Load successful!", "text/plain");
    });

    // ── GET /health ───────────────────────────────────────────────────────
    svr->Get(sparams.request_path + "/health",
             [&](const Request &, Response &res) {
        if (state.load() == SERVER_STATE_READY) {
            res.set_content("{\"status\":\"ok\"}", "application/json");
        } else {
            res.status = 503;
            res.set_content("{\"status\":\"loading model\"}", "application/json");
        }
    });

    // ── Error handlers ────────────────────────────────────────────────────
    svr->set_exception_handler([](const Request &, Response &res,
                                   std::exception_ptr ep) {
        char buf[1024];
        try { std::rethrow_exception(std::move(ep)); }
        catch (std::exception &e)
            { snprintf(buf, sizeof(buf), "500 Internal Server Error\n%s", e.what()); }
        catch (...)
            { snprintf(buf, sizeof(buf), "500 Internal Server Error\nUnknown exception"); }
        res.set_content(buf, "text/plain");
        res.status = 500;
    });

    svr->set_error_handler([](const Request &req, Response &res) {
        if (res.status == 400) {
            res.set_content("Invalid request", "text/plain");
        } else if (res.status != 500) {
            res.set_content("Not Found (" + req.path + ")", "text/plain");
            res.status = 404;
        }
    });

    // ── Bind & start listening ────────────────────────────────────────────
    svr->set_read_timeout(sparams.read_timeout);
    svr->set_write_timeout(sparams.write_timeout);

    if (!svr->bind_to_port(sparams.hostname, sparams.port)) {
        fprintf(stderr, "error: couldn't bind to %s:%d\n",
                sparams.hostname.c_str(), sparams.port);
        qwen_free(ctx);
        return 1;
    }

    svr->set_base_dir(sparams.public_path);
    printf("\nqwen-asr server listening at http://%s:%d\n\n",
           sparams.hostname.c_str(), sparams.port);

    shutdown_handler = [&](int /*sig*/) {
        printf("\nShutting down...\n");
        if (svr) svr->stop();
    };

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
    {
        struct sigaction sa{};
        sa.sa_handler = signal_handler;
        sigemptyset(&sa.sa_mask);
        sigaction(SIGINT,  &sa, nullptr);
        sigaction(SIGTERM, &sa, nullptr);
    }
#elif defined(_WIN32)
    SetConsoleCtrlHandler(
        +[](DWORD t) -> BOOL {
            return t == CTRL_C_EVENT ? (signal_handler(SIGINT), true) : false;
        },
        true);
#endif

    std::thread srv_thread([&] {
        if (!svr->listen_after_bind())
            fprintf(stderr, "error: server listen failed\n");
    });

    svr->wait_until_ready();
    srv_thread.join();

    qwen_free(ctx);
    return 0;
}
