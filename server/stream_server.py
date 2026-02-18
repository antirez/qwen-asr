#!/usr/bin/env python3
"""
Streaming ASR server for Qwen3-ASR C implementation.

Exposes a WebSocket endpoint at GET /stream (binary audio in, transcript text out).
POST and other methods to /stream return 405 so health checks or mistaken clients
don't trigger WebSocket handshake errors.

Environment:
  MODEL_DIR  Model directory (default /models)
  PORT       Listen port (default 2020)
  QWEN_ASR   Path to qwen_asr binary (default qwen_asr from PATH)
  QWEN_N_THREADS  If set, pass -t N to qwen_asr (default: use binary default, all CPUs)
"""

import asyncio
import os
import sys


async def handle_stream_ws(
    websocket, model_dir: str, binary: str, n_threads: int | None = None
) -> None:
    """Handle one WebSocket connection: pipe binary audio to qwen_asr, stream text back."""
    cmd = [binary, "-d", model_dir, "--stdin", "--stream"]
    if n_threads is not None and n_threads > 0:
        cmd.extend(["-t", str(n_threads)])
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
    except FileNotFoundError:
        await websocket.send_text("Error: qwen_asr binary not found")
        return
    except Exception as e:
        await websocket.send_text(f"Error: {e}")
        return

    async def read_stdout_and_forward():
        try:
            while proc.stdout:
                data = await proc.stdout.read(4096)
                if not data:
                    break
                text = data.decode("utf-8", errors="replace")
                if text:
                    try:
                        await websocket.send_text(text)
                    except Exception:
                        return
        except (ConnectionResetError, BrokenPipeError):
            pass

    send_task = asyncio.create_task(read_stdout_and_forward())

    try:
        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive(), timeout=3600.0)
            except asyncio.TimeoutError:
                break
            if msg.get("type") == "websocket.disconnect":
                break
            text = msg.get("text")
            data = msg.get("bytes")
            if text is not None:
                if text.strip().lower() in ("end", "close", "done"):
                    break
                continue
            if data is not None and proc.stdin:
                try:
                    proc.stdin.write(data)
                    await proc.stdin.drain()
                except (BrokenPipeError, ConnectionResetError):
                    break
    except Exception:
        pass
    finally:
        if proc.stdin:
            try:
                proc.stdin.close()
                await proc.stdin.wait_closed()
            except Exception:
                pass
        await asyncio.wait_for(asyncio.shield(send_task), timeout=60.0)
        try:
            await asyncio.wait_for(proc.wait(), timeout=30.0)
        except asyncio.TimeoutExpired:
            proc.kill()
            await proc.wait()


def create_app(model_dir: str, binary: str, n_threads: int | None = None):
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
    from fastapi.responses import PlainTextResponse

    app = FastAPI(title="Qwen3-ASR streaming", docs_url=None, redoc_url=None)

    @app.get("/stream")
    async def stream_get_reject():
        """GET without Upgrade: websocket gets a hint."""
        return PlainTextResponse(
            "Use a WebSocket client to connect to ws://host:port/stream\n",
            status_code=400,
        )

    @app.api_route("/stream", methods=["POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
    async def stream_method_not_allowed():
        """Reject non-WebSocket methods with 405."""
        return PlainTextResponse(
            "Method not allowed. Connect via WebSocket (GET with Upgrade: websocket) to /stream\n",
            status_code=405,
        )

    @app.websocket("/stream")
    async def stream_websocket(websocket: WebSocket):
        await websocket.accept()
        try:
            await handle_stream_ws(websocket, model_dir, binary, n_threads)
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            try:
                await websocket.close()
            except Exception:
                pass

    @app.get("/")
    async def root():
        return PlainTextResponse(
            "Qwen3-ASR streaming server. WebSocket endpoint: ws://host:port/stream\n",
            status_code=200,
        )

    return app


async def main() -> None:
    try:
        import uvicorn
    except ImportError:
        print("Install server deps: pip install -r server/requirements.txt", file=sys.stderr)
        sys.exit(1)

    model_dir = os.environ.get("MODEL_DIR", "/models")
    port = int(os.environ.get("PORT", "2020"))
    binary = os.environ.get("QWEN_ASR", "qwen_asr")
    n_threads = None
    if os.environ.get("QWEN_N_THREADS"):
        try:
            n_threads = int(os.environ["QWEN_N_THREADS"])
        except ValueError:
            pass

    if not os.path.isdir(model_dir):
        print(f"Model dir not found: {model_dir}", file=sys.stderr)
        print("Mount the model directory at /models (e.g. -v /path/to/qwen3-asr-0.6b:/models)", file=sys.stderr)
        sys.exit(1)

    app = create_app(model_dir, binary, n_threads)
    print(f"ASR streaming server: ws://0.0.0.0:{port}/stream (model: {model_dir})", file=sys.stderr)
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
