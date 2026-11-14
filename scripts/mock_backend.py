#!/usr/bin/env python3
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Plain-HTTP mock model backend for OGX.

A deterministic stand-in for a model server, used by CI (the mock test setup)
so client smoke tests can drive OGX end to end without downloading or
generating with a real model. It serves just enough of the OpenAI-compatible
wire protocol for OGX's remote::llama-cpp-server provider:

  - GET  /props              llama.cpp server signature for the health check
  - GET  /v1/models          one chat model plus one embedding model
  - POST /v1/chat/completions  fixed sentinel, JSON or SSE
  - POST /v1/embeddings        a fixed vector, so vector-store setup works

Every chat completion replies with the same fixed SENTINEL string, which
tests assert on to prove the request round-tripped mock -> OGX -> client.
The chat and embedding model ids are models.dev-known names so the
llama-cpp-server adapter classifies them (and picks up the embedding
dimension) without extra configuration.

Run it standalone with: python3 scripts/mock_backend.py --port 8090
"""

import argparse
import http.server
import json
import threading
import time

# Keep in sync with the "sentinel" default of the "mock" setup in
# tests/integration/suites.py.
SENTINEL = "OGX-E2E-SENTINEL-42"
# Model ids are names the llama-cpp-server adapter's models.dev lookup knows:
# the embedding name classifies with its dimension, the chat name classifies
# as a plain LLM.
CHAT_MODEL = "Llama-3.1-8B-Instruct"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536

_log_lock = threading.Lock()


def log_request(line: str) -> None:
    with _log_lock:
        print(f"mock-backend: {time.strftime('%Y-%m-%dT%H:%M:%S')} {line}", flush=True)


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a: object) -> None:
        pass

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            return json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return {}

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        path = self.path.split("?")[0]
        log_request(f"GET {self.path}")
        if path == "/props":
            self._send_json({"total_slots": 1, "default_generation_settings": {}})
        elif path == "/v1/models":
            self._send_json(
                {
                    "data": [
                        {"id": CHAT_MODEL, "object": "model", "owned_by": "ogx-mock"},
                        {"id": EMBEDDING_MODEL, "object": "model", "owned_by": "ogx-mock"},
                    ]
                }
            )
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        body = self._read_body()
        path = self.path.split("?")[0]
        log_request(f"POST {self.path} model={body.get('model')} stream={body.get('stream')}")
        if path == "/v1/chat/completions":
            model = body.get("model") or CHAT_MODEL
            if body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                chunk = {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": model}
                self.wfile.write(
                    f"data: {json.dumps({**chunk, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': SENTINEL}, 'finish_reason': None}]})}\n\n".encode()
                )
                self.wfile.write(
                    f"data: {json.dumps({**chunk, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n".encode()
                )
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                self.close_connection = True
            else:
                self._send_json(
                    {
                        "id": "c1",
                        "object": "chat.completion",
                        "created": 1,
                        "model": model,
                        "choices": [
                            {"index": 0, "message": {"role": "assistant", "content": SENTINEL}, "finish_reason": "stop"}
                        ],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    }
                )
        elif path == "/v1/embeddings":
            self._send_json(
                {
                    "object": "list",
                    "data": [{"object": "embedding", "index": 0, "embedding": [0.1] * EMBEDDING_DIMENSION}],
                    "model": body.get("model") or EMBEDDING_MODEL,
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                }
            )
        else:
            self._send_json({"error": "not found"}, 404)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, required=True)
    args = p.parse_args()
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"mock backend on {args.port}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
