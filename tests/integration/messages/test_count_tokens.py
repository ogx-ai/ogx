# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Integration tests for the Anthropic count_tokens API (/v1/messages/count_tokens).

These tests exercise both count_tokens paths end-to-end: the provider's native
/v1/messages/count_tokens endpoint (anthropic, meta, vllm, sglang) and the
fallback that counts via a max_tokens=1 message round trip (ollama, fireworks,
deepseek). Assertions are deliberately path-agnostic because exact counts vary
by provider tokenizer; they pin response shape, the anthropic-version header,
and monotonicity as the prompt grows.
"""

from .conftest import make_count_tokens_request


def _count_tokens(messages_client, text_model_id: str, text: str, **kwargs) -> int:
    """Count tokens for a single user message and return input_tokens."""
    response = make_count_tokens_request(
        messages_client,
        model=text_model_id,
        messages=[{"role": "user", "content": text}],
        **kwargs,
    )
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert isinstance(data["input_tokens"], int), f"input_tokens must be an int, got {data}"
    assert data["input_tokens"] > 0, f"Expected a positive token count, got {data}"
    return data["input_tokens"]


def test_count_tokens_basic(messages_client, text_model_id):
    """A basic count returns a positive input_tokens and the anthropic-version header."""
    response = make_count_tokens_request(
        messages_client,
        model=text_model_id,
        messages=[{"role": "user", "content": "Hello, world!"}],
    )

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert isinstance(data["input_tokens"], int)
    assert data["input_tokens"] > 0
    assert response.headers.get("anthropic-version") == "2023-06-01"


def test_count_tokens_monotonic_in_prompt_length(messages_client, text_model_id):
    """A longer prompt counts strictly more tokens than a shorter one."""
    short = _count_tokens(messages_client, text_model_id, "Hello.")
    long_text = " ".join(f"word{i}" for i in range(100))
    long = _count_tokens(messages_client, text_model_id, long_text)

    assert long > short, f"Expected count of {len(long_text)}-char prompt to exceed short count; got {long} <= {short}"


def test_count_tokens_with_system(messages_client, text_model_id):
    """A system prompt is included in the count."""
    without = _count_tokens(messages_client, text_model_id, "Hello.")
    with_system = _count_tokens(
        messages_client,
        text_model_id,
        "Hello.",
        system="You are a concise assistant. Always answer in one short sentence.",
    )

    assert with_system > without, f"Expected count with system to exceed count without; got {with_system} <= {without}"


def test_count_tokens_with_tools(messages_client, text_model_id):
    """Tool definitions are accepted in the count body and the count stays positive."""
    tools = [
        {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                },
                "required": ["location"],
            },
        }
    ]

    response = make_count_tokens_request(
        messages_client,
        model=text_model_id,
        messages=[{"role": "user", "content": "What's the weather in San Francisco?"}],
        tools=tools,
    )

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert isinstance(data["input_tokens"], int)
    assert data["input_tokens"] > 0


def test_count_tokens_error_missing_model(messages_client):
    """Request without model returns a validation error without reaching the provider."""
    response = messages_client.post(
        "/v1/messages/count_tokens",
        headers={
            "content-type": "application/json",
            "anthropic-version": "2023-06-01",
        },
        json={
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    assert response.status_code in (400, 422)
