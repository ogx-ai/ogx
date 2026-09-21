#!/usr/bin/env python3
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
CLOSE_WAIT leak repro (issue #6437).

Starts a mock OpenAI server and an ogx server proxying to it.
The client opens one stream, reads the first chunk, then disconnects.
Use --no-exit to leave the servers running for inspection with ss -tanp.

Usage:
    uv run scripts/repro_close_wait.py
"""

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from openai import AsyncOpenAI

MOCK_PORT = 18080
OGX_PORT = 18321
PROVIDER_ID = "mock-openai"
MODEL_ID = "test-model"
ROUTED_MODEL_ID = f"{PROVIDER_ID}/{MODEL_ID}"
CHUNK_DELAY = 2.0
STREAM_CHUNKS = 3


class MockHandler(BaseHTTPRequestHandler):
    server_version = "repro/1.0"

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/v1/models":
            body = json.dumps({"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        if not body.get("stream"):
            self.send_error(400, "this mock only supports stream=true")
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        for i in range(STREAM_CHUNKS):
            payload = {
                "id": f"chatcmpl-{i}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": MODEL_ID,
                "choices": [{"index": 0, "delta": {"content": f"chunk {i}"}, "finish_reason": None}],
            }
            try:
                self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
                self.wfile.flush()
            except (BrokenPipeError, OSError):
                return
            time.sleep(CHUNK_DELAY)

    def log_message(self, fmt: str, *args: object) -> None:
        pass


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-exit", action="store_true", help="keep servers running after the check")
    args = parser.parse_args()

    # Mock upstream
    mock = ThreadingHTTPServer(("127.0.0.1", MOCK_PORT), MockHandler)
    mock.daemon_threads = True
    threading.Thread(target=mock.serve_forever, daemon=True).start()
    print(f"mock openai server: http://127.0.0.1:{MOCK_PORT}")

    # Ogx server
    config_dir = tempfile.mkdtemp(prefix="ogx-repro-")
    config_path = Path(config_dir) / "config.yaml"
    config_path.write_text(f"""\
version: 2
distro_name: close-wait-repro
apis:
- inference
providers:
  inference:
  - provider_id: {PROVIDER_ID}
    provider_type: remote::openai
    config:
      base_url: http://127.0.0.1:{MOCK_PORT}/v1
      api_key: repro-key
registered_resources:
  models:
  - metadata: {{}}
    model_id: {MODEL_ID}
    provider_id: {PROVIDER_ID}
    provider_model_id: {MODEL_ID}
    model_type: llm
  vector_stores: []
server:
  port: {OGX_PORT}
""")
    env = dict(os.environ)
    env["OGX_CONFIG"] = str(config_path)
    env["OGX_DISABLE_VERSION_CHECK"] = "1"

    ogx = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "ogx.core.server.server:create_app",
            "--factory",
            "--host",
            "127.0.0.1",
            "--port",
            str(OGX_PORT),
            "--log-level",
            "warning",
        ],
        cwd=repo_root(),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    try:
        print(f"ogx server: http://127.0.0.1:{OGX_PORT} (pid={ogx.pid})")

        # Wait for ogx
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{OGX_PORT}/v1/models", timeout=3):
                    break
            except Exception:
                time.sleep(1)
        else:
            print("ogx did not start in 60s")
            return 1
        print("ogx ready")

        # Client: open 3 streams, read 1 chunk each, close
        client = AsyncOpenAI(base_url=f"http://127.0.0.1:{OGX_PORT}/v1", api_key="test", max_retries=0)

        async def abandon() -> None:
            for i in range(3):
                stream = await client.chat.completions.create(
                    model=ROUTED_MODEL_ID,
                    messages=[{"role": "user", "content": "hi"}],
                    stream=True,
                )
                async for _chunk in stream:
                    print(f"request {i + 1}: got first chunk, disconnecting")
                    break
                await stream.close()
            await client.close()

        asyncio.run(abandon())

        wait_seconds = STREAM_CHUNKS * CHUNK_DELAY + 1
        print(f"\nwaiting {wait_seconds:.0f}s for mock to finish...")
        time.sleep(wait_seconds)

        out = subprocess.run(["ss", "-tanp"], capture_output=True, text=True, timeout=10, check=True).stdout
        lines = [line for line in out.splitlines() if f":{MOCK_PORT}" in line or line.startswith("State")]
        print(f"\nconnections to mock (:{MOCK_PORT}):")
        for line in lines:
            print(f"  {line}")
        close_wait = sum(1 for line in lines if "CLOSE-WAIT" in line)
        if close_wait:
            print(f"LEAK: {close_wait} socket(s) in CLOSE_WAIT")
        else:
            print("no CLOSE_WAIT: ogx closed upstream connections on abandon")

        if args.no_exit:
            print("\nservers left running (Ctrl-C to stop)")
            print(f"  mock:  :{MOCK_PORT}")
            print(f"  ogx:   :{OGX_PORT} (pid={ogx.pid})")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
        return 1 if close_wait else 0
    finally:
        ogx.terminate()
        try:
            ogx.wait(timeout=15)
        except subprocess.TimeoutExpired:
            ogx.kill()
            ogx.wait()
        mock.shutdown()
        mock.server_close()
        shutil.rmtree(config_dir)


if __name__ == "__main__":
    raise SystemExit(main())
