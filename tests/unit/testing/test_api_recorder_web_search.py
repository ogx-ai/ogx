# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Tests for web-search backend call recording/replay at the HTTP layer.

The recorder intercepts the raw httpx calls of the web-search backends. Tavily
is the only backend that carries its API key in the request body, so these
tests pin the credential-redaction contract: the live request keeps the real
key, the stored recording never contains it, and the request hash is
independent of the key, so a recording made under one key replays under any."""

import httpx2
import pytest

from ogx.core.testing_context import reset_test_context, set_test_context
from ogx.testing import api_recorder
from ogx.testing.api_recorder import (
    APIRecordingMode,
    _record_replay_web_search,
    _redact_web_search_credentials,
)

TEST_ID = "tests/unit/testing/test_api_recorder_web_search.py::test"
TAVILY_URL = "https://api.tavily.com/search"
SEARCH_BODY = {"results": [{"title": "Example", "url": "https://example.com", "content": "A result."}]}


class _FakeStorage:
    """In-memory stand-in for ResponseStorage (hash-keyed, no directory logic)."""

    def __init__(self):
        self.by_hash: dict[str, dict] = {}

    def store_recording(self, request_hash: str, request: dict, response: dict):
        self.by_hash[request_hash] = {"request": request, "response": response}

    def find_recording(self, request_hash: str):
        return self.by_hash.get(request_hash)


@pytest.fixture
def test_context():
    token = set_test_context(TEST_ID)
    yield
    reset_test_context(token)


@pytest.fixture
def storage():
    return _FakeStorage()


def test_redact_web_search_credentials_removes_api_key_and_keeps_the_query():
    payload = {
        "query": "what is ogx?",
        "api_key": "tvly-secret",
        "include_domains": ["example.com"],
        "search_depth": "advanced",
    }

    redacted = _redact_web_search_credentials(payload)

    assert "api_key" not in redacted
    assert redacted == {
        "query": "what is ogx?",
        "include_domains": ["example.com"],
        "search_depth": "advanced",
    }
    # The caller's payload is untouched: the live request still carries the key.
    assert payload["api_key"] == "tvly-secret"


async def test_recording_is_independent_of_the_api_key(test_context, monkeypatch, storage):
    sent: list[dict] = []

    async def fake_original(self, url, **kwargs):
        sent.append(kwargs["json"])
        return httpx2.Response(200, json=SEARCH_BODY, request=httpx2.Request("POST", url))

    monkeypatch.setattr(api_recorder, "_current_mode", APIRecordingMode.RECORD)
    monkeypatch.setattr(api_recorder, "_current_storage", storage)

    recorded_payload = {"query": "what is ogx?", "api_key": "tvly-record-key"}
    response = await _record_replay_web_search(
        fake_original, object(), TAVILY_URL, "POST", recorded_payload, json=recorded_payload
    )
    assert response.status_code == 200

    # The live call went out with the real key...
    assert sent == [{"query": "what is ogx?", "api_key": "tvly-record-key"}]
    # ...and exactly one recording was stored, without the key.
    (stored,) = storage.by_hash.values()
    assert stored["request"]["payload"] == {"query": "what is ogx?"}
    assert stored["response"]["body"] == SEARCH_BODY

    # Replay under a *different* key: the hash must not depend on it, so the
    # recording is found without touching the network.
    monkeypatch.setattr(api_recorder, "_current_mode", APIRecordingMode.REPLAY)
    replay_payload = {"query": "what is ogx?", "api_key": "tvly-replay-key"}
    replayed = await _record_replay_web_search(
        fake_original, object(), TAVILY_URL, "POST", replay_payload, json=replay_payload
    )
    assert replayed.json() == SEARCH_BODY
    assert sent == [{"query": "what is ogx?", "api_key": "tvly-record-key"}]
