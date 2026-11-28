#!/usr/bin/env python3
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""OpenAI-compatible /v1/moderations endpoint backed by Llama Guard on Ollama.

The builtin responses provider implements guardrails by calling an OpenAI-compatible
/v1/moderations endpoint (its moderation_endpoint config). Ollama serves no such endpoint,
and Llama Guard answers through chat completions with a verdict ("safe", or "unsafe"
followed by violated category codes) rather than a moderation result. This shim translates
between the two, so the responses guardrail integration tests can record against the
llama-guard3:1b model the ollama setup already pulls.

scripts/integration-tests.sh starts it when OGX_TEST_MODERATION_ENDPOINT is set and the
inference mode records; in replay, the recorder serves the recorded moderation responses
and the shim is not needed.

  - POST /v1/moderations  {"input": str | [str, ...]} -> {"id", "model", "results": [...]}

Run it standalone with:
  python3 scripts/moderation_shim.py --port 11435 --ollama-url http://localhost:11434/v1
"""

import argparse
import http.server
import json
import os
import threading
import time
import urllib.parse
import urllib.request
import uuid

DEFAULT_GUARD_MODEL = "llama-guard3:1b"

# Llama Guard 3's hazard taxonomy, keyed by the category codes it emits.
LLAMA_GUARD_CATEGORIES = {
    "S1": "violent_crimes",
    "S2": "non_violent_crimes",
    "S3": "sex_related_crimes",
    "S4": "child_sexual_exploitation",
    "S5": "defamation",
    "S6": "specialized_advice",
    "S7": "privacy",
    "S8": "intellectual_property",
    "S9": "indiscriminate_weapons",
    "S10": "hate",
    "S11": "suicide_and_self_harm",
    "S12": "sexual_content",
    "S13": "elections",
    "S14": "code_interpreter_abuse",
}

_log_lock = threading.Lock()


def log_request(line: str) -> None:
    with _log_lock:
        print(f"moderation-shim: {time.strftime('%Y-%m-%dT%H:%M:%S')} {line}", flush=True)


def parse_verdict(verdict: str) -> dict:
    """Convert a Llama Guard verdict into one OpenAI moderation result.

    Llama Guard replies "safe", or "unsafe" with the violated category codes on the next
    line (e.g. "unsafe\\nS1,S10"). Anything else is treated as flagged with no category,
    so an unexpected reply fails closed like the provider itself does.
    """
    lines = [line.strip() for line in verdict.strip().splitlines() if line.strip()]
    first = lines[0].lower() if lines else ""
    codes: set[str] = set()
    if first == "safe":
        flagged = False
    else:
        flagged = True
        if first == "unsafe" and len(lines) > 1:
            codes = {code.strip().upper() for code in lines[1].split(",") if code.strip()}

    categories = {name: code in codes for code, name in LLAMA_GUARD_CATEGORIES.items()}
    return {
        "flagged": flagged,
        "categories": categories,
        "category_scores": {name: 1.0 if hit else 0.0 for name, hit in categories.items()},
    }


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    ollama_url: str = "http://localhost:11434/v1"
    guard_model: str = DEFAULT_GUARD_MODEL

    def log_message(self, *a: object) -> None:
        pass

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _classify(self, text: str) -> dict:
        request = urllib.request.Request(  # noqa: S310 - http(s) only, validated in main()
            f"{self.ollama_url.rstrip('/')}/chat/completions",
            data=json.dumps(
                {
                    "model": self.guard_model,
                    "messages": [{"role": "user", "content": text}],
                    "temperature": 0,
                    "stream": False,
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        # main() only accepts an http(s) --ollama-url, so no file: or custom scheme reaches here.
        with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
            completion = json.load(response)
        verdict = completion["choices"][0]["message"]["content"] or ""
        result = parse_verdict(verdict)
        log_request(f"verdict={verdict.strip()!r} flagged={result['flagged']}")
        return result

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        if self.path.split("?", 1)[0] != "/v1/moderations":
            self._send_json({"error": {"message": f"Unknown path {self.path}"}}, status=404)
            return

        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send_json({"error": {"message": "Request body is not valid JSON"}}, status=400)
            return

        raw_input = payload.get("input")
        inputs = [raw_input] if isinstance(raw_input, str) else raw_input
        if not isinstance(inputs, list) or not inputs or not all(isinstance(item, str) for item in inputs):
            self._send_json({"error": {"message": "'input' must be a string or a list of strings"}}, status=400)
            return

        try:
            results = [self._classify(text) for text in inputs]
        except Exception as e:  # report the guard model failure instead of dropping the connection
            log_request(f"guard model call failed: {e}")
            self._send_json({"error": {"message": f"Failed to classify input: {e}"}}, status=502)
            return

        self._send_json({"id": f"modr-{uuid.uuid4().hex}", "model": self.guard_model, "results": results})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11435)
    parser.add_argument("--ollama-url", default="http://localhost:11434/v1", help="Ollama's OpenAI-compatible base URL")
    parser.add_argument("--model", default=DEFAULT_GUARD_MODEL, help="Llama Guard model served by Ollama")
    parser.add_argument(
        "--exit-with-pid",
        type=int,
        default=None,
        help="Exit once this process is gone, so a runner that is killed does not leak the shim",
    )
    args = parser.parse_args()
    if urllib.parse.urlparse(args.ollama_url).scheme not in ("http", "https"):
        parser.error(f"--ollama-url must be an http(s) URL, got {args.ollama_url!r}")

    Handler.ollama_url = args.ollama_url
    Handler.guard_model = args.model
    server = http.server.ThreadingHTTPServer((args.host, args.port), Handler)
    log_request(f"listening on {args.host}:{args.port}, guard model {args.model} at {args.ollama_url}")
    if args.exit_with_pid is not None:
        threading.Thread(target=_shutdown_when_gone, args=(server, args.exit_with_pid), daemon=True).start()
    server.serve_forever()


def _shutdown_when_gone(server: http.server.ThreadingHTTPServer, pid: int) -> None:
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            log_request(f"process {pid} exited, shutting down")
            server.shutdown()
            return
        except PermissionError:
            pass  # the process exists but belongs to another user
        time.sleep(2)


if __name__ == "__main__":
    main()
