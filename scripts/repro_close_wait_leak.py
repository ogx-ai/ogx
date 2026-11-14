#!/usr/bin/env python3
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Reproducer for the abandoned-stream CLOSE_WAIT leak (issue #6437).

Starts a mock OpenAI-compatible upstream, boots a real ogx server that
proxies chat completions to it, abandons streams after the first chunk,
and counts the sockets left in CLOSE_WAIT with ss(8). Streams routed
through the ogx wrappers are closed on abandon, so no CLOSE_WAIT sockets
should accumulate between ogx and the mock upstream. The --leaky mode
skips the ogx server and abandons streams straight against the mock
without closing them, demonstrating the leak this fix closes.

Usage:
    uv run scripts/repro_close_wait_leak.py
    uv run scripts/repro_close_wait_leak.py --requests 10
    uv run scripts/repro_close_wait_leak.py --leaky
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

CHUNK_DELAY_SECONDS = 0.3
STREAM_CHUNKS = 4
CLOSE_WAIT_POLL_SECONDS = 20.0

MOCK_PORT = 18080
OGX_PORT = 18321
PROVIDER_ID = "mock-openai"
MODEL_ID = "test-model"
# The routing table prefixes listed models with their provider id.
ROUTED_MODEL_ID = f"{PROVIDER_ID}/{MODEL_ID}"

# Uvicorn may need a while for the provider registry and routing tables to
# come up before the first request succeeds.
READY_TIMEOUT_SECONDS = 180.0

OGX_CONFIG_YAML = """\
version: 2
distro_name: close-wait-repro
apis:
- inference
providers:
  inference:
  - provider_id: {provider_id}
    provider_type: remote::openai
    config:
      base_url: http://127.0.0.1:{mock_port}/v1
      api_key: repro-key
registered_resources:
  models:
  - metadata: {{}}
    model_id: {model_id}
    provider_id: {provider_id}
    provider_model_id: {model_id}
    model_type: llm
  vector_stores: []
server:
  port: {ogx_port}
"""


class StreamingHandler(BaseHTTPRequestHandler):
    server_version = "repro-server/1.0"

    def do_GET(self):  # noqa: N802
        if self.path == "/v1/models":
            body = json.dumps({"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def do_POST(self):  # noqa: N802
        # Drain the request body so the connection closes with FIN instead of
        # RST when the handler returns.
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            for i in range(STREAM_CHUNKS):
                payload = {
                    "id": f"chatcmpl-repro-{i}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": MODEL_ID,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": f"chunk {i}"},
                            "finish_reason": None,
                        }
                    ],
                }
                self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
                self.wfile.flush()
                time.sleep(CHUNK_DELAY_SECONDS)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, format, *args):
        pass


def start_mock_server(port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), StreamingHandler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def start_ogx_server(mock_port: int, ogx_port: int) -> tuple[subprocess.Popen, str]:
    """Boot a real ogx server proxying to the mock upstream.

    Returns the server process and the directory holding the generated
    config (kept alive until the caller removes it).
    """
    config_dir = tempfile.mkdtemp(prefix="ogx-close-wait-repro-")
    config_path = Path(config_dir) / "config.yaml"
    config_path.write_text(
        OGX_CONFIG_YAML.format(
            provider_id=PROVIDER_ID,
            model_id=MODEL_ID,
            mock_port=mock_port,
            ogx_port=ogx_port,
        )
    )
    env = dict(os.environ)
    env["OGX_CONFIG"] = str(config_path)
    env["OGX_DISABLE_VERSION_CHECK"] = "1"
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "ogx.core.server.server:create_app",
            "--factory",
            "--host",
            "127.0.0.1",
            "--port",
            str(ogx_port),
            "--log-level",
            "warning",
        ],
        cwd=repo_root(),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    return proc, config_dir


async def wait_for_ogx_server(ogx_port: int) -> None:
    """Poll /v1/models until the ogx server answers or the deadline passes."""
    deadline = time.monotonic() + READY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                f"import urllib.request;urllib.request.urlopen('http://127.0.0.1:{ogx_port}/v1/models', timeout=5)",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            code = await proc.wait()
            if code == 0:
                return
        except Exception:
            pass
        await asyncio.sleep(2)
    raise SystemExit(
        f"ogx server did not become ready on port {ogx_port} within "
        f"{READY_TIMEOUT_SECONDS:.0f}s; is the ogx package importable from "
        f"'{sys.executable}'? Run with 'uv run' from the repo root."
    )


def count_sockets(port: int) -> tuple[int, int]:
    """Return (close_wait, established) socket counts for connections to port."""
    try:
        out = subprocess.run(["ss", "-tanH"], capture_output=True, text=True, timeout=10, check=True).stdout
    except FileNotFoundError:
        raise SystemExit("ss(8) not found: install iproute2 to monitor connections") from None
    close_wait = established = 0
    for line in out.splitlines():
        fields = line.split()
        if len(fields) < 5 or not fields[4].endswith(f":{port}"):
            continue
        state = fields[0]
        if state == "CLOSE-WAIT":
            close_wait += 1
        elif state == "ESTAB":
            established += 1
    return close_wait, established


def poll_close_wait_zero(port: int, timeout: float) -> int:
    """Poll until no sockets to port are in CLOSE_WAIT; return the last count."""
    deadline = time.monotonic() + timeout
    close_wait = 0
    while time.monotonic() < deadline:
        close_wait, _ = count_sockets(port)
        if close_wait == 0:
            return 0
        time.sleep(1.0)
    return close_wait


async def run_ogx_client(ogx_port: int, mock_port: int, requests: int) -> int:
    """Abandon streams through the ogx server and wait for upstream cleanup."""
    client = AsyncOpenAI(
        base_url=f"http://127.0.0.1:{ogx_port}/v1",
        api_key="test",
        timeout=10.0,
        max_retries=0,
    )
    abandoned: list[Any] = []
    for _ in range(requests):
        stream = await client.chat.completions.create(
            model=ROUTED_MODEL_ID,
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
        abandoned.append(stream)
        async for _chunk in stream:
            break
        # Abandon: close the downstream response without draining the stream.
        # ogx must then close its upstream connection to the mock.
        await stream.close()
    await client.close()
    # The mock finishes each stream a moment after the last chunk; give ogx
    # time to close its side of every upstream connection.
    await asyncio.sleep(STREAM_CHUNKS * CHUNK_DELAY_SECONDS + 1)
    remaining = poll_close_wait_zero(mock_port, CLOSE_WAIT_POLL_SECONDS)
    return remaining


async def run_leaky_client(mock_port: int, requests: int) -> int:
    """Abandon streams against the mock directly, without closing them."""
    client = AsyncOpenAI(
        base_url=f"http://127.0.0.1:{mock_port}/v1",
        api_key="test",
        timeout=10.0,
        max_retries=0,
    )
    abandoned: list[Any] = []
    for _ in range(requests):
        stream = await client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
        # Keep a reference so an unclosed upstream stream cannot be garbage
        # collected (and its connection closed by the finalizer) before the
        # sockets are counted.
        abandoned.append(stream)
        iterator = stream.__aiter__()
        abandoned.append(iterator)
        await iterator.__anext__()
    # Let the mock finish streaming and send FIN to every abandoned socket.
    await asyncio.sleep(2 * CHUNK_DELAY_SECONDS * STREAM_CHUNKS)
    close_wait, _ = count_sockets(mock_port)
    return close_wait


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=8, help="streams to abandon (default: 8)")
    parser.add_argument("--mock-port", type=int, default=MOCK_PORT, help=f"mock upstream port (default: {MOCK_PORT})")
    parser.add_argument("--ogx-port", type=int, default=OGX_PORT, help=f"ogx server port (default: {OGX_PORT})")
    parser.add_argument("--leaky", action="store_true", help="demonstrate the leak without the ogx server")
    args = parser.parse_args()

    mock_server = start_mock_server(args.mock_port)
    ogx_proc = None
    config_dir = None
    try:
        if args.leaky:
            close_wait = asyncio.run(run_leaky_client(args.mock_port, args.requests))
            print(f"abandoned streams: {args.requests}, ogx wrapper: skipped")
            print(f"sockets: CLOSE_WAIT={close_wait} (client -> mock)")
            if close_wait == 0:
                print("expected the leak but no CLOSE_WAIT sockets were observed;")
                print("increase --requests or CHUNK_DELAY_SECONDS")
                return 1
            print(f"leak reproduced: {close_wait} sockets stuck in CLOSE_WAIT")
            return 0

        ogx_proc, config_dir = start_ogx_server(args.mock_port, args.ogx_port)
        asyncio.run(wait_for_ogx_server(args.ogx_port))
        if ogx_proc.poll() is not None:
            raise SystemExit(f"ogx server exited during startup with code {ogx_proc.returncode}")
        # Let the registry refresh settle so the run does not race it.
        time.sleep(2)

        remaining = asyncio.run(run_ogx_client(args.ogx_port, args.mock_port, args.requests))
        print(f"abandoned streams: {args.requests}, through ogx wrapper")
        print(f"sockets: CLOSE_WAIT={remaining} (ogx -> mock)")
        if remaining != 0:
            print(f"regression: {remaining} sockets stuck in CLOSE_WAIT after wrapped streams were abandoned")
            return 1
        print("no CLOSE_WAIT sockets: the ogx wrappers closed the upstream streams on abandon")
        return 0
    finally:
        if ogx_proc is not None:
            ogx_proc.terminate()
            try:
                ogx_proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                ogx_proc.kill()
        if config_dir is not None:
            import shutil

            shutil.rmtree(config_dir, ignore_errors=True)
        mock_server.shutdown()
        mock_server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
