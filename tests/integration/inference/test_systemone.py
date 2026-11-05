# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""End-to-end record/replay for the TypeSafe SystemOne (judgment) provider.

Run in replay mode with the ``typesafe`` setup (no live TypeSafe API or real key needed):

    ./scripts/integration-tests.sh --stack-config server:ci-tests --setup typesafe \
        --file tests/integration/inference/test_systemone.py

The model list is served from a shared (test-id-free) recording under
``tests/integration/common/recordings``; each judgment call is a test-scoped recording under
``tests/integration/inference/recordings``. Every judgment test sends a distinct request so it
captures a distinct recording. To (re)record the judgment tests against the live API, set a real
``TYPESAFE_API_KEY`` and run:

    ./scripts/integration-tests.sh --stack-config server:ci-tests --setup typesafe \
        --inference-mode record --file tests/integration/inference/test_systemone.py

Assertions check the SystemOne wire contract (the shape of the response), not exact values, so
they hold for both the committed placeholder recordings and freshly recorded live responses.

The raw-wire tests (model listing shape and judgment) call the server's HTTP surface, so they
need the server client; the ogx_client fixture carries the server URL for them, and they skip in
in-process (library client) runs.
"""

import json

import pytest
import requests

from ogx.core.library_client import OGXAsLibraryClient
from ogx.core.testing_context import get_test_context
from tests.integration.fixtures.common import _client_timeout


def _base_url(ogx_client) -> str:
    """The running server's base URL, from the same source as the ogx_client fixture.

    ``instantiate_ogx_client`` binds the server client to the running server and carries its URL
    on ``base_url``; the in-process library client has no HTTP surface, so the raw-wire tests skip
    there.
    """
    if isinstance(ogx_client, OGXAsLibraryClient):
        pytest.skip("SystemOne wire tests need the server client; the in-process library client has no HTTP surface")
    return str(ogx_client.base_url)


def _test_context_headers() -> dict[str, str]:
    """Carry the active test id to the server so the server-side recorder keys the TypeSafe call to this test."""
    test_id = get_test_context()
    if not test_id:
        return {}
    return {"X-OGX-Provider-Data": json.dumps({"__test_id": test_id})}


def _judge(ogx_client, payload: dict[str, object]) -> "requests.Response":
    """POST /v1/systemone with the active test id, returning the raw response."""
    return requests.post(
        f"{_base_url(ogx_client)}/v1/systemone",
        headers=_test_context_headers(),
        json=payload,
        timeout=_client_timeout(),
    )


def _assert_envelope(body: dict[str, object], question_names: frozenset[str], model_id: str) -> dict[str, object]:
    """Assert the top-level SystemOne response envelope and return the ``answers`` mapping."""
    assert body.get("model") == model_id, (
        f"response model {body.get('model')!r} must be the id the client sent ({model_id!r}), "
        "not a provider-internal id"
    )
    usage = body.get("usage")
    assert isinstance(usage, dict), "response must carry usage"
    assert int(usage.get("input_tokens", 0)) >= 0
    assert int(usage.get("output_tokens", 0)) >= 0
    answers = body.get("answers")
    assert isinstance(answers, dict), "response must carry answers"
    assert set(answers) == question_names, f"answers {sorted(answers)} != questions {sorted(question_names)}"
    return answers


# --- model registry ----------------------------------------------------------


def test_systemone_model_is_listed(ogx_client, judge_model_id):
    """GET /v1/models surfaces the TypeSafe SystemOne model in the standard listing."""
    ids = {m.id for m in ogx_client.models.list().data}
    assert judge_model_id in ids, f"{judge_model_id} not in {sorted(ids)}"


def test_systemone_models_listing_typesafe_shape(ogx_client, judge_model_id):
    """GET /v1/models with the x-typesafe-sdk header returns TypeSafe's model-listing shape."""
    resp = requests.get(
        f"{_base_url(ogx_client)}/v1/models",
        headers={"x-typesafe-sdk": "1"},
        timeout=_client_timeout(),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    models = body.get("models")
    assert isinstance(models, list) and models, "TypeSafe listing must carry a non-empty models list"
    for entry in models:
        assert set(entry) >= {"name", "description", "release_date"}, f"unexpected entry shape: {entry}"
    # `name` is the OGX identifier, which is what POST /v1/systemone accepts in its model field.
    assert any(entry["name"] == judge_model_id for entry in models)


# --- judgment: noul ----------------------------------------------------------


def test_systemone_judgment_is_replayed(ogx_client, judge_model_id):
    """A noul (yes/no) question returns a calibrated probability in [0, 1]."""
    resp = _judge(
        ogx_client,
        {
            "model": judge_model_id,
            "state": "I was charged twice. Please help.",
            "questions": {"billing": {"type": "noul", "instructions": "Is this about billing?"}},
        },
    )
    assert resp.status_code == 200, resp.text
    answers = _assert_envelope(resp.json(), frozenset({"billing"}), judge_model_id)
    answer = answers["billing"]
    assert answer["type"] == "noul"
    assert 0.0 <= float(answer["noul"]) <= 1.0


# --- judgment: choice --------------------------------------------------------


def test_systemone_judgment_choice(ogx_client, judge_model_id):
    """A choice question returns a selected option, per-option probabilities, and confidence."""
    resp = _judge(
        ogx_client,
        {
            "model": judge_model_id,
            "state": "The caller is shouting and using profanity.",
            "questions": {
                "tone": {
                    "type": "choice",
                    "instructions": "What is the tone of the caller?",
                    "criteria": {"calm": "Level and measured", "angry": "Loud and hostile", "neutral": None},
                }
            },
        },
    )
    assert resp.status_code == 200, resp.text
    answers = _assert_envelope(resp.json(), frozenset({"tone"}), judge_model_id)
    answer = answers["tone"]
    assert answer["type"] == "choice"
    assert answer["choice"] in {"calm", "angry", "neutral"}
    assert isinstance(answer["probabilities"], dict)
    assert 0.0 <= float(answer["confidence"]) <= 1.0


# --- judgment: score ---------------------------------------------------------


def test_systemone_judgment_score(ogx_client, judge_model_id):
    """A score question returns a rubric-weighted score, legend, probabilities, and confidence."""
    resp = _judge(
        ogx_client,
        {
            "model": judge_model_id,
            "state": "The agent was slightly rude but did resolve the issue.",
            "questions": {
                "quality": {
                    "type": "score",
                    "instructions": "Rate the quality of the interaction.",
                    "criteria": ["Unacceptable", "Poor", "Acceptable", "Good", "Excellent"],
                }
            },
        },
    )
    assert resp.status_code == 200, resp.text
    answers = _assert_envelope(resp.json(), frozenset({"quality"}), judge_model_id)
    answer = answers["quality"]
    assert answer["type"] == "score"
    assert isinstance(answer["score"], int | float)
    assert isinstance(answer["legend"], dict)
    assert isinstance(answer["probabilities"], dict)
    assert 0.0 <= float(answer["confidence"]) <= 1.0


# --- judgment: mixed questions ----------------------------------------------


def test_systemone_judgment_mixed_questions(ogx_client, judge_model_id):
    """One request can mix noul, choice, and score questions; each is answered by its name."""
    resp = _judge(
        ogx_client,
        {
            "model": judge_model_id,
            "state": "I was charged twice and the agent was rude about it.",
            "questions": {
                "billing": {"type": "noul", "instructions": "Is this about billing?"},
                "tone": {
                    "type": "choice",
                    "instructions": "What is the tone?",
                    "criteria": {"calm": None, "angry": None},
                },
                "quality": {
                    "type": "score",
                    "instructions": "Rate the interaction.",
                    "criteria": ["Bad", "Ok", "Good"],
                },
            },
        },
    )
    assert resp.status_code == 200, resp.text
    answers = _assert_envelope(resp.json(), frozenset({"billing", "tone", "quality"}), judge_model_id)
    assert answers["billing"]["type"] == "noul"
    assert answers["tone"]["type"] == "choice"
    assert answers["quality"]["type"] == "score"


# --- judgment: structured state ----------------------------------------------


def test_systemone_judgment_structured_state(ogx_client, judge_model_id):
    """`state` may be a JSON object (not just a string); questions evaluate against it."""
    resp = _judge(
        ogx_client,
        {
            "model": judge_model_id,
            "state": {"message": "I was charged twice. Please help.", "channel": "phone", "customer_tier": "premium"},
            "questions": {"billing": {"type": "noul", "instructions": "Is this about billing?"}},
        },
    )
    assert resp.status_code == 200, resp.text
    answers = _assert_envelope(resp.json(), frozenset({"billing"}), judge_model_id)
    assert answers["billing"]["type"] == "noul"
