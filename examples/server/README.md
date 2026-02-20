# Qwen ASR Server Example

Simple HTTP server for Qwen3-ASR transcription. Audio files are uploaded via HTTP and transcribed on the server using the native qwen-asr inference engine.

## Build

```bash
# From root
make server
```

The resulting binary is `./qwen_asr_server` in the project root.

## Usage

```
./qwen_asr_server -d <model_dir> [options]
```

Options:

```
./qwen_asr_server -h

Required:
  -d DIR, --model-dir DIR   Model directory (*.safetensors + vocab.json)

Inference options:
  -t N, --threads N         Number of threads (default: auto)
  --language LANG           Default forced language (default: auto-detect)
                            Supported: Chinese,English,Cantonese,Arabic,...
  --prompt TEXT             Default system prompt text

Server options:
  --host HOST               Hostname / IP to bind (default: 127.0.0.1)
  --port PORT               Port number (default: 8080)
  --public PATH             Directory for static files (default: examples/server/public)
  --convert                 Accept non-WAV audio; convert via ffmpeg
  --tmp-dir DIR             Temp directory for ffmpeg output (default: .)
```

### Example usage

```bash
./qwen_asr_server -d qwen3-asr-1.7b
```

Then open http://127.0.0.1:8080 in a browser for the live microphone UI.

## API endpoints

### `POST /inference`

Transcribe an audio file. Accepts `multipart/form-data`.

**Request fields:**

| Field             | Type   | Default         | Description                    |
|-------------------|--------|-----------------|--------------------------------|
| `file`            | binary | None (required) | Audio to transcribe            |
| `language`        | string | Empty (auto)    | Language name (e.g. `English`) |
| `prompt`          | string | server default  | System prompt for decoder bias |
| `response_format` | string | `json`          | `json` or `text`               |

**Response formats:**

| `response_format` | Content-Type       | Body                    |
|-------------------|--------------------|-------------------------|
| `json` (default)  | `application/json` | JSON object (see below) |
| `text`            | `text/plain`       | Raw transcription text  |

A real world curl example:

```bash
curl 127.0.0.1:8080/inference \
  -H "Content-Type: multipart/form-data" \
  -F file="@samples/jfk.wav" \
  -F response_format="json"
```

The `json` response includes the transcription and per-request timing:

```
{"decode_ms":5519.23490234375,"encode_ms":968.870849609375,"rt_factor":0.5898475454545454,"text":"And so, my fellow Americans, ask not what your country can do for you; ask what you can do for your country.","tok_s":4.007198778482514,"tokens":26,"total_ms":6488.323}
```

**Response Fields:**

| Field       | Description                                           |
|-------------|-------------------------------------------------------|
| `text`      | Full transcription                                    |
| `total_ms`  | End-to-end inference time (ms)                        |
| `encode_ms` | Mel extraction + audio encoder time (ms)              |
| `decode_ms` | Decoder prefill + autoregressive generation time (ms) |
| `tokens`    | Number of text tokens generated                       |
| `tok_s`     | Tokens per second (`tokens / total_ms * 1000`)        |
| `rt_factor` | Real-time factor: `total_ms / audio_duration_ms`.     |

RT values below `1.0` mean faster than real-time; `2.0` means it took twice as long as the audio.

With ffmpeg conversion (any format accepted):

```bash
./qwen_asr_server -d qwen3-asr-1.7b --convert

curl 127.0.0.1:8080/inference -F file="@recording.mp3"
```

### `POST /load`

Hot-swap the loaded model directory at runtime.

```bash
curl 127.0.0.1:8080/load \
  -F model="/path/to/qwen3-asr-0.6b"
```

### `GET /health`

Readiness probe.

| State         | HTTP status | Body                         |
|---------------|-------------|------------------------------|
| Model loaded  | 200         | `{"status":"ok"}`            |
| Loading model | 503         | `{"status":"loading model"}` |

## Audio format

Without `--convert`: the uploaded file must be **16-bit PCM WAV, 16 kHz, mono**. Any other format will fail to parse.

With `--convert`: any format ffmpeg can decode is accepted. The server invokes:

```
ffmpeg -i <input> -y -ar 16000 -ac 1 -c:a pcm_s16le <output.wav>
```

## Load testing with k6

```bash
k6 run examples/server/bench.js \
  --env FILE_PATH=/absolute/path/to/samples/jfk.wav \
  --env BASE_URL=http://127.0.0.1:8080 \
  --env CONCURRENCY=4
```

**Environment variables:**

| Variable          | Default                 | Description                   |
|-------------------|-------------------------|-------------------------------|
| `FILE_PATH`       | (required)              | Absolute path to audio file   |
| `BASE_URL`        | `http://127.0.0.1:8080` | Server base URL               |
| `ENDPOINT`        | `/inference`            | API endpoint                  |
| `CONCURRENCY`     | `4`                     | Number of concurrent requests |
| `LANGUAGE`        | (empty = auto)          | Language name override        |
| `RESPONSE_FORMAT` | `json`                  | Response format               |

> **Note:** Install [k6](https://k6.io/) before running the benchmark script.
