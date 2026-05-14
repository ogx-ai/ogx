# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Integration tests for the Anthropic Message Batches API.

Covers POST/GET/cancel/results under /v1/messages/batches. Each batch request
reuses the Anthropic↔OpenAI translation path, so these tests are intended to
run under the `messages-openai` suite.
"""

import json
import time

from .conftest import _build_headers


def _poll_until_ended(client, batch_id: str, timeout: float = 60.0) -> dict:
    """Poll the batch until processing_status is 'ended' or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = client.get(f"/v1/messages/batches/{batch_id}", headers=_build_headers())
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()
        if data["processing_status"] == "ended":
            return data
        time.sleep(0.2)
    raise AssertionError(f"Batch {batch_id} did not end within {timeout}s")


def _create_batch(client, text_model_id: str, prompts: list[tuple[str, str]]) -> dict:
    """POST a batch with the given (custom_id, prompt) pairs."""
    body = {
        "requests": [
            {
                "custom_id": cid,
                "params": {
                    "model": text_model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 32,
                },
            }
            for cid, prompt in prompts
        ],
    }
    response = client.post("/v1/messages/batches", headers=_build_headers(), json=body)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    return response.json()


def test_messages_batch_create_and_retrieve(messages_client, text_model_id):
    """Creating a batch returns a MessageBatch object that is retrievable by ID."""
    batch = _create_batch(
        messages_client,
        text_model_id,
        [
            ("req-1", "What is 2+2? Reply with just the number."),
            ("req-2", "What is 3+3? Reply with just the number."),
        ],
    )

    assert batch["id"].startswith("msgbatch_")
    assert batch["type"] == "message_batch"
    assert batch["processing_status"] in ("in_progress", "ended")
    assert batch["request_counts"]["processing"] + batch["request_counts"]["succeeded"] == 2
    assert "created_at" in batch
    assert "expires_at" in batch

    final = _poll_until_ended(messages_client, batch["id"])
    assert final["processing_status"] == "ended"
    assert final["request_counts"]["succeeded"] == 2
    assert final["request_counts"]["errored"] == 0
    assert final["ended_at"] is not None
    assert final["results_url"] == f"/v1/messages/batches/{batch['id']}/results"


def test_messages_batch_results(messages_client, text_model_id):
    """Batch results endpoint streams JSONL with one line per request."""
    batch = _create_batch(
        messages_client,
        text_model_id,
        [
            ("r1", "Say 'hello' and nothing else."),
            ("r2", "Say 'world' and nothing else."),
        ],
    )
    _poll_until_ended(messages_client, batch["id"])

    response = messages_client.get(
        f"/v1/messages/batches/{batch['id']}/results",
        headers=_build_headers(),
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-jsonl")

    lines = [line for line in response.text.splitlines() if line.strip()]
    assert len(lines) == 2

    custom_ids = set()
    for line in lines:
        entry = json.loads(line)
        assert "custom_id" in entry
        assert "result" in entry
        assert entry["result"]["type"] == "succeeded"
        assert entry["result"]["message"]["type"] == "message"
        assert entry["result"]["message"]["role"] == "assistant"
        assert len(entry["result"]["message"]["content"]) > 0
        custom_ids.add(entry["custom_id"])

    assert custom_ids == {"r1", "r2"}


def test_messages_batch_list_includes_new_batch(messages_client, text_model_id):
    """Listing batches returns the newly-created batch."""
    batch = _create_batch(
        messages_client,
        text_model_id,
        [("list-1", "Say 'ok' and nothing else.")],
    )
    _poll_until_ended(messages_client, batch["id"])

    response = messages_client.get("/v1/messages/batches", headers=_build_headers())
    assert response.status_code == 200
    data = response.json()

    assert "data" in data
    assert "has_more" in data
    assert any(b["id"] == batch["id"] for b in data["data"]), f"Batch {batch['id']} not found in list response"


def test_messages_batch_cancel_after_end_fails(messages_client, text_model_id):
    """Attempting to cancel a batch that has already ended returns 400."""
    batch = _create_batch(
        messages_client,
        text_model_id,
        [("done-1", "Say 'done' and nothing else.")],
    )
    _poll_until_ended(messages_client, batch["id"])

    response = messages_client.post(
        f"/v1/messages/batches/{batch['id']}/cancel",
        headers=_build_headers(),
    )
    assert response.status_code == 400
    err = response.json()
    assert err["type"] == "error"
    assert err["error"]["type"] == "invalid_request_error"


def test_messages_batch_duplicate_custom_id(messages_client, text_model_id):
    """Batch creation rejects duplicate custom_ids with 400."""
    body = {
        "requests": [
            {
                "custom_id": "dup",
                "params": {
                    "model": text_model_id,
                    "messages": [{"role": "user", "content": "Say 'a'."}],
                    "max_tokens": 16,
                },
            },
            {
                "custom_id": "dup",
                "params": {
                    "model": text_model_id,
                    "messages": [{"role": "user", "content": "Say 'b'."}],
                    "max_tokens": 16,
                },
            },
        ],
    }
    response = messages_client.post("/v1/messages/batches", headers=_build_headers(), json=body)
    assert response.status_code == 400
    err = response.json()
    assert err["type"] == "error"
    assert "dup" in err["error"]["message"]


def test_messages_batch_retrieve_nonexistent(messages_client):
    """Retrieving an unknown batch ID returns 404."""
    response = messages_client.get(
        "/v1/messages/batches/msgbatch_does_not_exist",
        headers=_build_headers(),
    )
    assert response.status_code == 404
    err = response.json()
    assert err["type"] == "error"
    assert err["error"]["type"] == "not_found_error"
