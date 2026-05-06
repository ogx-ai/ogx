# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

MOCK_RESPONSE = {
    "id": "chatcmpl-mock123",
    "object": "chat.completion",
    "created": int(time.time()),
    "model": "openai/mock-chat-model",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "This is a mock response from the benchmark server.",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
    },
}


class MockHandler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        if self.path == "/v1/chat/completions":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            response = MOCK_RESPONSE.copy()
            response["created"] = int(time.time())
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):  # noqa: N802
        if self.path == "/v1/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    import sys

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    server = HTTPServer(("0.0.0.0", port), MockHandler)
    print(f"Mock server listening on port {port}")
    server.serve_forever()
