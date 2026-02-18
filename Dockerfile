# Qwen3-ASR C implementation: production image with streaming endpoint
# This image does NOT include model files; mount a model dir at /models
# (e.g. -v /path/to/qwen3-asr-0.6b:/models).

# -----------------------------------------------------------------------------
# Stage 1: build qwen_asr binary
# -----------------------------------------------------------------------------
FROM debian:bookworm-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY main.c qwen_asr.c qwen_asr.h qwen_asr_audio.c qwen_asr_audio.h \
     qwen_asr_decoder.c qwen_asr_encoder.c qwen_asr_kernels.c qwen_asr_kernels.h \
     qwen_asr_kernels_impl.h qwen_asr_kernels_generic.c qwen_asr_kernels_neon.c qwen_asr_kernels_avx.c \
     qwen_asr_safetensors.c qwen_asr_safetensors.h qwen_asr_tokenizer.c qwen_asr_tokenizer.h \
     Makefile .

# Portable build (no -march=native) so the image runs on any x86_64/arm64 host
RUN make clean && make qwen_asr \
    CFLAGS="-Wall -Wextra -O3 -ffast-math -DUSE_BLAS -DUSE_OPENBLAS -I/usr/include/openblas" \
    LDFLAGS="-lm -lpthread -lopenblas"

# -----------------------------------------------------------------------------
# Stage 2: runtime with streaming server
# -----------------------------------------------------------------------------
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Binary on PATH
COPY --from=builder /build/qwen_asr /usr/local/bin/qwen_asr

# Streaming server (venv for Debian PEP 668)
WORKDIR /app
COPY server/requirements.txt server/stream_server.py ./
RUN python3 -m venv /app/venv && /app/venv/bin/pip install --no-cache-dir -r requirements.txt

ENV MODEL_DIR=/models \
    PORT=2020 \
    PATH="/app/venv/bin:$PATH"

EXPOSE 2020

# Default: run the streaming endpoint. Override entrypoint to run CLI only, e.g.:
#   docker run --rm --entrypoint /usr/local/bin/qwen_asr -v ... qwen-asr -d /models -i /data/audio.wav --silent
CMD ["python3", "/app/stream_server.py"]
