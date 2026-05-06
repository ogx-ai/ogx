# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Integration tests for the Anthropic Messages API (/v1/messages).

These tests verify the full request/response cycle through the server,
including translation between Anthropic and OpenAI formats.
"""

from .conftest import make_messages_request, make_streaming_messages_request


def _get_text_blocks(content: list[dict]) -> list[dict]:
    """Extract text blocks from a content list, skipping thinking blocks."""
    return [b for b in content if b["type"] == "text"]


def test_messages_non_streaming_basic(messages_client, text_model_id):
    """Basic non-streaming message creation returns a valid Anthropic response."""
    response = make_messages_request(
        messages_client,
        model=text_model_id,
        messages=[{"role": "user", "content": "What is 2+2? Reply with just the number."}],
        max_tokens=64,
    )

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    data = response.json()
    assert data["type"] == "message"
    assert data["role"] == "assistant"
    assert data["id"].startswith("msg_")
    assert len(data["content"]) > 0

    # Content may include thinking blocks; find first text block
    text_blocks = _get_text_blocks(data["content"])
    assert len(text_blocks) > 0, f"No text blocks found in content: {data['content']}"
    assert len(text_blocks[0]["text"]) > 0

    assert data["stop_reason"] in ("end_turn", "max_tokens")
    assert "usage" in data
    assert data["usage"]["input_tokens"] > 0
    assert data["usage"]["output_tokens"] > 0

    # All content blocks must be valid types
    for block in data["content"]:
        assert block["type"] in ("text", "thinking", "tool_use")


def test_messages_non_streaming_with_system(messages_client, text_model_id):
    """Non-streaming message with a system prompt."""
    response = make_messages_request(
        messages_client,
        model=text_model_id,
        messages=[{"role": "user", "content": "What are you?"}],
        system="You are a helpful pirate. Always respond in pirate speak.",
        max_tokens=128,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "message"
    assert len(data["content"]) > 0

    text_blocks = _get_text_blocks(data["content"])
    assert len(text_blocks) > 0
    assert len(text_blocks[0]["text"]) > 0


def test_messages_non_streaming_multi_turn(messages_client, text_model_id):
    """Non-streaming multi-turn conversation."""
    response = make_messages_request(
        messages_client,
        model=text_model_id,
        messages=[
            {"role": "user", "content": "My name is Alice."},
            {"role": "assistant", "content": "Hello Alice! Nice to meet you."},
            {"role": "user", "content": "What is my name?"},
        ],
        max_tokens=64,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "message"
    assert len(data["content"]) > 0

    text_blocks = _get_text_blocks(data["content"])
    assert len(text_blocks) > 0
    text = text_blocks[0]["text"].lower()
    assert "alice" in text


def test_messages_streaming_basic(messages_client, text_model_id):
    """Streaming message creation returns proper Anthropic SSE events."""
    events = make_streaming_messages_request(
        messages_client,
        model=text_model_id,
        messages=[{"role": "user", "content": "Say hello in one sentence."}],
        max_tokens=64,
    )

    assert len(events) > 0

    event_types = [e.get("_event_type") or e.get("type") for e in events]

    # Verify the required event sequence
    assert "message_start" in event_types, f"Missing message_start in {event_types}"
    assert "message_stop" in event_types, f"Missing message_stop in {event_types}"

    # Verify message_start event structure
    msg_start = next(e for e in events if e.get("_event_type") == "message_start")
    assert "message" in msg_start
    assert msg_start["message"]["role"] == "assistant"

    # Verify we got content deltas
    content_deltas = [e for e in events if e.get("_event_type") == "content_block_delta"]
    assert len(content_deltas) > 0, "Expected at least one content_block_delta event"

    # Verify content_block_delta structure
    for delta in content_deltas:
        assert "delta" in delta
        assert delta["delta"]["type"] in ("text_delta", "thinking_delta")


def test_messages_streaming_collects_full_text(messages_client, text_model_id):
    """Streaming response text deltas can be concatenated into the full response."""
    events = make_streaming_messages_request(
        messages_client,
        model=text_model_id,
        messages=[{"role": "user", "content": "Count from 1 to 5, separated by commas."}],
        max_tokens=64,
    )

    # Collect text from content_block_delta events
    text_parts = []
    for event in events:
        if event.get("_event_type") == "content_block_delta":
            delta = event.get("delta", {})
            if delta.get("type") == "text_delta":
                text_parts.append(delta["text"])

    full_text = "".join(text_parts)
    assert len(full_text) > 0


def test_messages_non_streaming_with_temperature(messages_client, text_model_id):
    """Non-streaming with explicit temperature parameter."""
    response = make_messages_request(
        messages_client,
        model=text_model_id,
        messages=[{"role": "user", "content": "Say hello."}],
        max_tokens=32,
        temperature=0.0,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "message"
    assert len(data["content"]) > 0


def test_messages_non_streaming_with_stop_sequences(messages_client, text_model_id):
    """Non-streaming with stop_sequences parameter."""
    response = make_messages_request(
        messages_client,
        model=text_model_id,
        messages=[{"role": "user", "content": "Count: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10"}],
        max_tokens=128,
        stop_sequences=[","],
    )

    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "message"


def test_messages_with_tool_definitions(messages_client, text_model_id):
    """Non-streaming message with tool definitions."""
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

    response = make_messages_request(
        messages_client,
        model=text_model_id,
        messages=[{"role": "user", "content": "What's the weather in San Francisco?"}],
        tools=tools,
        max_tokens=256,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "message"
    assert len(data["content"]) > 0

    # The model may or may not call the tool; thinking, text, and tool_use are all valid
    for block in data["content"]:
        assert block["type"] in ("text", "tool_use", "thinking")
        if block["type"] == "tool_use":
            assert "id" in block
            assert block["name"] == "get_weather"
            assert "input" in block


def test_messages_tool_use_round_trip(messages_client, text_model_id):
    """Full tool use round trip: request -> tool_use -> tool_result -> response."""
    tools = [
        {
            "name": "calculator",
            "description": "Perform basic arithmetic. Use this for any math question.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "The math expression to evaluate"},
                },
                "required": ["expression"],
            },
        }
    ]

    # First request -- ask a math question
    response = make_messages_request(
        messages_client,
        model=text_model_id,
        messages=[
            {"role": "user", "content": "Use the calculator tool to compute 15 * 7."},
        ],
        tools=tools,
        tool_choice={"type": "any"},
        max_tokens=256,
    )

    assert response.status_code == 200
    data = response.json()

    # Find tool_use block
    tool_use_blocks = [b for b in data["content"] if b["type"] == "tool_use"]
    if not tool_use_blocks:
        # Model didn't use the tool -- skip the rest
        return

    tool_use = tool_use_blocks[0]
    tool_use_id = tool_use["id"]

    # Second request -- provide tool result
    response2 = make_messages_request(
        messages_client,
        model=text_model_id,
        messages=[
            {"role": "user", "content": "Use the calculator tool to compute 15 * 7."},
            {"role": "assistant", "content": data["content"]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": "105",
                    }
                ],
            },
        ],
        tools=tools,
        max_tokens=256,
    )

    assert response2.status_code == 200
    data2 = response2.json()
    assert data2["type"] == "message"
    assert len(data2["content"]) > 0


def test_messages_error_missing_model(messages_client):
    """Request without model returns an error."""
    headers = {
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
    }

    response = messages_client.post(
        "/v1/messages",
        headers=headers,
        json={
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 64,
        },
    )

    assert response.status_code in (400, 422)


def test_messages_error_empty_messages(messages_client, text_model_id):
    """Request with empty messages list returns an error."""
    headers = {
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
    }

    response = messages_client.post(
        "/v1/messages",
        headers=headers,
        json={
            "model": text_model_id,
            "messages": [],
            "max_tokens": 64,
        },
    )

    # Should fail validation or return an error
    assert response.status_code in (400, 422, 500)


def test_messages_response_headers(messages_client, text_model_id):
    """Response includes anthropic-version header."""
    response = make_messages_request(
        messages_client,
        model=text_model_id,
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=16,
    )

    assert response.status_code == 200
    assert response.headers.get("anthropic-version") == "2023-06-01"


def test_messages_content_block_array(messages_client, text_model_id):
    """Message with content as an array of content blocks."""
    response = make_messages_request(
        messages_client,
        model=text_model_id,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is 1+1? Reply with just the number."},
                ],
            }
        ],
        max_tokens=32,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "message"
    assert len(data["content"]) > 0
