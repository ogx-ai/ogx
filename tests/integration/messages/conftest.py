# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import os
from typing import Any

import httpx2
import pytest

from ogx.core.library_client import OGXAsLibraryClient
from ogx.core.testing_context import get_test_context

# Import fixtures from common module to make them available in this test directory
from tests.integration.fixtures.common import (  # noqa: F401
    openai_client,
    require_server,
)


def pytest_configure(config):
    """Disable stderr pipe to prevent Rich logging from blocking on buffer saturation."""
    os.environ["OGX_TEST_LOG_STDERR"] = "0"


@pytest.fixture(scope="session")
def messages_base_url(ogx_client):
    """Provide the base URL for the Messages API, skipping library client mode."""
    if isinstance(ogx_client, OGXAsLibraryClient):
        pytest.skip("Messages API tests are not supported in library client mode")
    return ogx_client.base_url


@pytest.fixture(scope="session")
def mock_sentinel():
    """The fixed reply of the mock model backend, or None when the selected setup uses a real model."""
    from tests.integration.suites import SETUP_DEFINITIONS

    setup_obj = SETUP_DEFINITIONS.get(os.environ.get("OGX_TEST_SETUP", ""))
    if setup_obj is None:
        return None
    sentinel = setup_obj.defaults.get("sentinel")
    return str(sentinel) if sentinel else None


# Single timeout for every messages request in this suite. Mirrors
# AnthropicMixin.anthropic_messages_timeout (300s, the provider's own upstream
# timeout unless network.timeout is set) so the test client is never the first
# link to give up: on slow backends (e.g. CPU vllm with thinking) a request can
# legitimately take well over a minute.
_REQUEST_TIMEOUT = 300.0


@pytest.fixture
def messages_client(messages_base_url):
    """Provide an httpx2 client configured for Anthropic Messages API calls."""
    client = httpx2.Client(base_url=messages_base_url, timeout=_REQUEST_TIMEOUT)
    yield client
    client.close()


# Setups whose text model thinks by default. The suite asserts on text content,
# so thinking must be steered per provider:
#   - Fireworks: disable it (its /v1/messages documents {"type": "disabled"}).
#   - Meta Muse: cannot be disabled (reasoning_effort 'none' is unsupported);
#     enable it with an explicit budget and raise max_tokens so text follows
#     the thinking blocks (verified against api.meta.ai).
#   - vllm (Qwen3-0.6B): thinks by default in a typical deployment; keep the
#     deployment standard and just raise max_tokens so text follows the
#     thinking blocks.
#   - gpt (gpt-5-nano): OpenAI reasoning models charge reasoning tokens
#     against max_tokens (translated to max_completion_tokens), so a small
#     explicit budget is exhausted before any text is emitted; raise
#     max_tokens to leave headroom for the reasoning.
# Other providers default to non-thinking models.
_THINKING_DISABLED_SETUPS = frozenset({"fireworks"})
_THINKING_BUDGET_SETUPS = frozenset({"meta"})
_THINKING_FLOOR_SETUPS = frozenset({"meta", "vllm", "gpt"})
_THINKING_BUDGET_TOKENS = 1024
_THINKING_MAX_TOKENS = 2048


def _thinking_config() -> dict[str, Any] | None:
    setup = os.environ.get("OGX_TEST_SETUP")
    if setup in _THINKING_DISABLED_SETUPS:
        return {"type": "disabled"}
    if setup in _THINKING_BUDGET_SETUPS:
        return {"type": "enabled", "budget_tokens": _THINKING_BUDGET_TOKENS}
    return None


def _max_tokens_floor(max_tokens: int) -> int:
    """Raise max_tokens for setups whose model thinks (thinking plus the asserted
    text must fit within the budget)."""
    if os.environ.get("OGX_TEST_SETUP") in _THINKING_FLOOR_SETUPS:
        return max(max_tokens, _THINKING_MAX_TOKENS)
    return max_tokens


def _thinking_params_fragment() -> dict[str, Any]:
    """Fragment to merge into message-creation params (e.g. batch items)."""
    if (thinking := _thinking_config()) is None:
        return {}
    return {"thinking": thinking}


# Models released after Claude Opus 4.6 (e.g. claude-haiku-5-5) reject any
# temperature other than 1.0 with a 400. Other setups keep 0.0 to preserve
# existing recordings (the request body is hashed for replay).
_TEMP_LOCKED_SETUPS = frozenset({"anthropic"})

# Models known to restrict sampling parameters in chat completions:
# non-default temperature values are rejected ("Only the default (1) value
# is supported") and the stop parameter is rejected entirely ("Unsupported
# parameter: 'stop'"). Verified against gpt-5-nano; add models here as
# their restrictions are verified.
_LIMITED_SAMPLING_MODELS = frozenset({"gpt-5-nano"})


def _model_name(model_id: str) -> str:
    return model_id.rsplit("/", 1)[-1]


def _is_limited_sampling_model(model_id: str) -> bool:
    return _model_name(model_id) in _LIMITED_SAMPLING_MODELS


def _temperature_value(model_id: str, default: float) -> float:
    if os.environ.get("OGX_TEST_SETUP") in _TEMP_LOCKED_SETUPS:
        return 1.0
    if _is_limited_sampling_model(model_id):
        return 1.0
    return default


# Meta Muse's /v1/messages rejects stop_sequences entirely and only accepts
# tool_choice "auto" (not "any"/"none"/named). Adapt those params for meta.
_META_LIMITED_SETUPS = frozenset({"meta"})


def _is_meta_limited() -> bool:
    return os.environ.get("OGX_TEST_SETUP") in _META_LIMITED_SETUPS


def _build_messages_body(
    *,
    model: str,
    messages: list[dict],
    max_tokens: int = 256,
    stream: bool = False,
    system: str | None = None,
    tools: list[dict] | None = None,
    tool_choice: dict | str | None = None,
    temperature: float | None = None,
    stop_sequences: list[str] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": _max_tokens_floor(max_tokens),
        "stream": stream,
    }
    if (thinking := _thinking_config()) is not None:
        body["thinking"] = thinking
    if system is not None:
        body["system"] = system
    if tools is not None:
        body["tools"] = tools
    if tool_choice is not None:
        if _is_meta_limited():
            tool_choice = {"type": "auto"}
        body["tool_choice"] = tool_choice
    if temperature is not None:
        body["temperature"] = _temperature_value(model, temperature)
    if stop_sequences is not None and not _is_meta_limited():
        body["stop_sequences"] = stop_sequences
    return body


def _build_headers() -> dict[str, str]:
    headers = {
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    test_id = get_test_context()
    if test_id:
        provider_data = {"__test_id": test_id}
        headers["X-OGX-Provider-Data"] = json.dumps(provider_data)
    return headers


def make_messages_request(
    client: httpx2.Client,
    **kwargs: Any,
) -> httpx2.Response:
    """Make a non-streaming POST request to /v1/messages."""
    body = _build_messages_body(**kwargs)
    return client.post("/v1/messages", headers=_build_headers(), json=body)


def _build_count_tokens_body(
    *,
    model: str,
    messages: list[dict],
    system: str | None = None,
    tools: list[dict] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if system is not None:
        body["system"] = system
    if tools is not None:
        body["tools"] = tools
    return body


def make_count_tokens_request(
    client: httpx2.Client,
    **kwargs: Any,
) -> httpx2.Response:
    """Make a POST request to /v1/messages/count_tokens."""
    body = _build_count_tokens_body(**kwargs)
    return client.post("/v1/messages/count_tokens", headers=_build_headers(), json=body)


def make_streaming_messages_request(
    client: httpx2.Client,
    **kwargs: Any,
) -> list[dict]:
    """Make a streaming POST request to /v1/messages and return parsed SSE events.

    Raises AssertionError if the response status is not 200.
    """
    kwargs["stream"] = True
    body = _build_messages_body(**kwargs)
    headers = _build_headers()

    events: list[dict] = []
    current_event_type: str | None = None

    with client.stream("POST", "/v1/messages", headers=headers, json=body) as response:
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        for line in response.iter_lines():
            if line.startswith("event: "):
                current_event_type = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
                if current_event_type:
                    data["_event_type"] = current_event_type
                events.append(data)
                current_event_type = None

    return events
