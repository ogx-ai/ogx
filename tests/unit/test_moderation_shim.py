# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""scripts/moderation_shim.py turns Llama Guard verdicts from Ollama's chat completions into
OpenAI /v1/moderations results, the format the responses provider's guardrails expect (#6950)."""

import http.server
import importlib.util
import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest

from ogx.providers.inline.responses.builtin.responses.utils import run_guardrails

SHIM_PATH = Path(__file__).resolve().parents[2] / "scripts" / "moderation_shim.py"
_spec = importlib.util.spec_from_file_location("moderation_shim", SHIM_PATH)
assert _spec is not None and _spec.loader is not None
moderation_shim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(moderation_shim)


class TestParseVerdict:
    def test_safe(self):
        result = moderation_shim.parse_verdict("safe")

        assert result["flagged"] is False
        assert not any(result["categories"].values())

    def test_unsafe_with_categories(self):
        result = moderation_shim.parse_verdict("unsafe\nS1,S10")

        assert result["flagged"] is True
        flagged = {name for name, hit in result["categories"].items() if hit}
        assert flagged == {"violent_crimes", "hate"}
        assert result["category_scores"]["violent_crimes"] == 1.0
        assert result["category_scores"]["privacy"] == 0.0

    def test_surrounding_whitespace_and_case_are_ignored(self):
        assert moderation_shim.parse_verdict("\n  Safe \n")["flagged"] is False
        assert moderation_shim.parse_verdict(" unsafe \n s11 ")["categories"]["suicide_and_self_harm"] is True

    @pytest.mark.parametrize("verdict", ["", "I cannot help with that.", "unsafe"])
    def test_anything_but_safe_fails_closed(self, verdict):
        assert moderation_shim.parse_verdict(verdict)["flagged"] is True


def _serve(handler_class: type[http.server.BaseHTTPRequestHandler]) -> tuple[http.server.ThreadingHTTPServer, str]:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler_class)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


@pytest.fixture
def fake_ollama() -> Iterator[tuple[str, list[dict]]]:
    """An OpenAI-compatible chat completions stand-in answering like Llama Guard."""
    requests: list[dict] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a: object) -> None:
            pass

        def do_POST(self) -> None:  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            verdict = "unsafe\nS1" if "hurt" in body["messages"][0]["content"] else "safe"
            payload = json.dumps({"choices": [{"message": {"role": "assistant", "content": verdict}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server, url = _serve(Handler)
    yield f"{url}/v1", requests
    server.shutdown()


@pytest.fixture
def shim(fake_ollama) -> Iterator[str]:
    ollama_url, _ = fake_ollama

    class Handler(moderation_shim.Handler):
        pass

    Handler.ollama_url = ollama_url
    server, url = _serve(Handler)
    yield f"{url}/v1/moderations"
    server.shutdown()


def _post(url: str, payload: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


class TestShimEndpoint:
    def test_classifies_each_input_with_the_guard_model(self, shim, fake_ollama):
        _, ollama_requests = fake_ollama

        status, body = _post(shim, {"input": ["How can I hurt someone?", "What's your name?"]})

        assert status == 200
        assert body["model"] == moderation_shim.DEFAULT_GUARD_MODEL
        assert [result["flagged"] for result in body["results"]] == [True, False]
        assert body["results"][0]["categories"]["violent_crimes"] is True
        assert [request["model"] for request in ollama_requests] == [moderation_shim.DEFAULT_GUARD_MODEL] * 2
        assert all(request["temperature"] == 0 for request in ollama_requests)

    def test_accepts_a_single_string_input(self, shim):
        status, body = _post(shim, {"input": "What's your name?"})

        assert status == 200
        assert len(body["results"]) == 1

    @pytest.mark.parametrize("payload", [{}, {"input": []}, {"input": [1]}])
    def test_rejects_invalid_input(self, shim, payload):
        status, _ = _post(shim, payload)

        assert status == 400

    def test_guard_model_failure_is_a_502(self, fake_ollama):
        class Handler(moderation_shim.Handler):
            pass

        Handler.ollama_url = "http://127.0.0.1:1/v1"
        server, url = _serve(Handler)
        try:
            status, body = _post(f"{url}/v1/moderations", {"input": "hi"})
        finally:
            server.shutdown()

        assert status == 502
        assert body["error"]["message"].startswith("Failed to classify input")

    async def test_responses_guardrails_accept_the_shim_response(self, shim):
        """The shim's output is what run_guardrails parses: blocked input yields the refusal
        message, allowed input yields None."""
        assert await run_guardrails(shim, "How can I hurt someone?") == (
            "Content blocked by safety guardrails (flagged for: violent_crimes)"
        )
        assert await run_guardrails(shim, "What's your name?") is None
