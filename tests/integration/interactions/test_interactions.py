# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Integration tests for the Google Interactions API (/v1alpha/interactions).

These tests verify the full request/response cycle through the server
using the official Google GenAI SDK, proving that ADK/Gemini ecosystem
clients can call OGX natively.
"""

import warnings

import pytest


@pytest.fixture(autouse=True)
def _suppress_experimental_warning():
    """Suppress the google-genai experimental usage warning."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Interactions usage is experimental")
        yield


def _get_text_output(interaction):
    """Extract the first text output, skipping any thought content."""
    for output in interaction.outputs:
        if _get_field(output, "type") == "text":
            return output
    return None


def _get_field(value, key, default=None):
    """Get a field from either an SDK object or a dict payload."""
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _get_event_type(event):
    """Get a canonical event type string across google-genai versions."""
    return _get_field(event, "event_type", type(event).__name__)


def _get_content_delta_text(event):
    """Extract text from a content.delta event for both SDK object styles."""
    delta = _get_field(event, "delta")
    if delta is None:
        return None
    return _get_field(delta, "text")


def test_interactions_non_streaming_basic(genai_client, text_model_id):
    """Basic non-streaming interaction returns a valid Google Interactions response."""
    interaction = genai_client.interactions.create(
        model=text_model_id,
        input="What is 2+2? Reply with just the number.",
    )

    assert interaction.id is not None, "ID should be present"
    assert interaction.status == "completed", f"Status should be 'completed', got: {interaction.status}"
    assert len(interaction.outputs) > 0, "Expected at least one output"
    text_output = _get_text_output(interaction)
    assert text_output is not None, (
        f"Expected a text output, got types: {[_get_field(o, 'type') for o in interaction.outputs]}"
    )
    assert len(_get_field(text_output, "text", "")) > 0
    assert interaction.usage.total_input_tokens > 0
    assert interaction.usage.total_output_tokens > 0
    assert (
        interaction.usage.total_tokens == interaction.usage.total_input_tokens + interaction.usage.total_output_tokens
    )


def test_interactions_non_streaming_system_instruction(genai_client, text_model_id):
    """Non-streaming interaction with a system instruction."""
    interaction = genai_client.interactions.create(
        model=text_model_id,
        input="What are you?",
        system_instruction="You are a pirate. Always respond in pirate speak. Keep it short.",
    )

    assert interaction.status == "completed"
    assert len(interaction.outputs) > 0
    text_output = _get_text_output(interaction)
    assert text_output is not None
    assert len(_get_field(text_output, "text", "")) > 0


def test_interactions_non_streaming_multi_turn(genai_client, text_model_id):
    """Non-streaming multi-turn conversation with 'model' role."""
    interaction = genai_client.interactions.create(
        model=text_model_id,
        input=[
            {"role": "user", "content": [{"type": "text", "text": "My name is Alice."}]},
            {"role": "model", "content": [{"type": "text", "text": "Hello Alice! Nice to meet you."}]},
            {"role": "user", "content": [{"type": "text", "text": "What is my name?"}]},
        ],
    )

    assert interaction.status == "completed"
    assert len(interaction.outputs) > 0
    text_output = _get_text_output(interaction)
    assert text_output is not None
    assert "alice" in _get_field(text_output, "text", "").lower()


def test_interactions_non_streaming_generation_config(genai_client, text_model_id):
    """Non-streaming interaction with generation config parameters."""
    interaction = genai_client.interactions.create(
        model=text_model_id,
        input="Say hello.",
        generation_config={
            "temperature": 0.0,
            "max_output_tokens": 32,
        },
    )

    assert interaction.status == "completed"
    assert len(interaction.outputs) > 0
    text_output = _get_text_output(interaction)
    assert text_output is not None
    assert len(_get_field(text_output, "text", "")) > 0


def test_interactions_non_streaming_response_shape(genai_client, text_model_id):
    """Non-streaming response includes all required fields matching Google's real API."""
    interaction = genai_client.interactions.create(
        model=text_model_id,
        input="Hi",
    )

    assert interaction.id is not None
    assert interaction.status == "completed"
    assert interaction.model is not None
    assert interaction.role == "model"
    assert interaction.outputs is not None
    assert interaction.usage is not None


def test_interactions_streaming_basic(genai_client, text_model_id):
    """Streaming interaction returns proper Google SSE events via the SDK."""
    stream = genai_client.interactions.create(
        model=text_model_id,
        input="Count from 1 to 5, separated by commas.",
        stream=True,
    )

    event_types = []
    text_parts = []
    interaction_id = None

    for event in stream:
        event_type = _get_event_type(event)
        event_types.append(event_type)

        if event_type == "interaction.start":
            interaction = _get_field(event, "interaction")
            interaction_id = _get_field(interaction, "id")

        if event_type == "content.delta":
            text = _get_content_delta_text(event)
            if text:
                text_parts.append(text)

    full_text = "".join(text_parts)
    assert len(full_text) > 0, "Streaming should produce text"
    assert interaction_id is not None, "Should have received an interaction ID"

    # Verify event sequence contains expected types
    assert "interaction.start" in event_types
    assert "content.start" in event_types
    assert "content.delta" in event_types
    assert "content.stop" in event_types
    assert "interaction.complete" in event_types


def test_interactions_streaming_text_concatenation(genai_client, text_model_id):
    """Streaming text deltas can be concatenated into the full response."""
    stream = genai_client.interactions.create(
        model=text_model_id,
        input="Say hello in one sentence.",
        stream=True,
    )

    text_parts = []
    for event in stream:
        if _get_event_type(event) == "content.delta":
            text = _get_content_delta_text(event)
            if text:
                text_parts.append(text)

    full_text = "".join(text_parts)
    assert len(full_text) > 0


def test_interactions_streaming_event_order(genai_client, text_model_id):
    """Streaming events contain required types in the correct relative order."""
    stream = genai_client.interactions.create(
        model=text_model_id,
        input="Hi",
        stream=True,
    )

    events = list(stream)
    event_types = [_get_event_type(e) for e in events]
    assert len(events) >= 4, f"Expected at least 4 events, got {len(events)}: {event_types}"

    # Verify all required event types are present
    required = ["interaction.start", "content.start", "content.delta", "content.stop", "interaction.complete"]
    for req in required:
        assert req in event_types, f"Missing required event type {req}, got: {event_types}"

    # Verify relative ordering: start before content, content before complete
    def _first(name):
        return event_types.index(name)

    assert _first("interaction.start") < _first("content.start"), "interaction.start should precede content.start"
    assert _first("content.start") < _first("content.delta"), "content.start should precede content.delta"
    assert _first("content.stop") < _first("interaction.complete"), "content.stop should precede interaction.complete"


def test_interactions_tool_calling_function_call_output(genai_client, text_model_id):
    """Tool calling: model returns a function_call output when given tools."""
    interaction = genai_client.interactions.create(
        model=text_model_id,
        input="What is the weather in Paris right now? Use the get_weather tool.",
        tools=[
            {
                "function_declarations": [
                    {
                        "name": "get_weather",
                        "description": "Get the current weather for a location.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string", "description": "City name"},
                            },
                            "required": ["location"],
                        },
                    }
                ]
            }
        ],
    )

    assert interaction.status in ("completed", "requires_action")
    assert len(interaction.outputs) > 0

    # Model should produce a function_call output
    function_calls = [o for o in interaction.outputs if _get_field(o, "type") == "function_call"]
    if not function_calls:
        pytest.skip("Model answered directly without calling the tool")

    fc = function_calls[0]
    assert _get_field(fc, "name") == "get_weather"
    assert _get_field(fc, "id") is not None
    # SDK uses 'arguments' attribute for function call args
    fc_args = _get_field(fc, "arguments") or _get_field(fc, "args") or {}
    assert isinstance(fc_args, dict)


@pytest.mark.xfail(
    reason="Round-trip requires exact Interactions API wire format for multi-turn with function_call/result. "
    "Passthrough parse→dump cycle transforms field names in ways Gemini rejects. "
    "Will be fixed when previous_interaction_id is supported.",
    strict=False,
)
def test_interactions_tool_calling_round_trip(genai_client, text_model_id):
    """Tool calling round-trip: function_call → function_response → text answer."""
    # Step 1: Get a function_call
    interaction = genai_client.interactions.create(
        model=text_model_id,
        input="What is the weather in Tokyo? You must use the get_weather tool.",
        tools=[
            {
                "function_declarations": [
                    {
                        "name": "get_weather",
                        "description": "Get the current weather for a location.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string", "description": "City name"},
                            },
                            "required": ["location"],
                        },
                    }
                ]
            }
        ],
    )

    function_calls = [o for o in interaction.outputs if _get_field(o, "type") == "function_call"]
    if not function_calls:
        pytest.skip("Model answered directly without calling the tool")

    fc = function_calls[0]
    fc_args = _get_field(fc, "arguments") or _get_field(fc, "args") or {}
    fc_id = _get_field(fc, "id")
    fc_name = _get_field(fc, "name")

    # Step 2: Send function_response and get a final text answer
    interaction2 = genai_client.interactions.create(
        model=text_model_id,
        input=[
            {"role": "user", "content": "What is the weather in Tokyo?"},
            {
                "role": "model",
                "content": [
                    {
                        "type": "function_call",
                        "id": fc_id,
                        "name": fc_name,
                        "arguments": fc_args,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "function_result",
                        "call_id": fc_id,
                        "name": fc_name,
                        "result": {"temperature_celsius": 22, "condition": "Cloudy"},
                    }
                ],
            },
        ],
        tools=[
            {
                "function_declarations": [
                    {
                        "name": "get_weather",
                        "description": "Get the current weather for a location.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string", "description": "City name"},
                            },
                            "required": ["location"],
                        },
                    }
                ]
            }
        ],
    )

    assert interaction2.status == "completed"
    text_outputs = [o for o in interaction2.outputs if _get_field(o, "type") == "text"]
    assert len(text_outputs) > 0, "Expected a text response after providing function result"


def test_interactions_tool_calling_multiple_tools(genai_client, text_model_id):
    """Tool calling with multiple function declarations available."""
    interaction = genai_client.interactions.create(
        model=text_model_id,
        input="What time is it in London? Use the get_time tool.",
        tools=[
            {
                "function_declarations": [
                    {
                        "name": "get_weather",
                        "description": "Get the current weather for a location.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"},
                            },
                            "required": ["location"],
                        },
                    },
                    {
                        "name": "get_time",
                        "description": "Get the current time in a timezone.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "timezone": {"type": "string", "description": "IANA timezone"},
                            },
                            "required": ["timezone"],
                        },
                    },
                ]
            }
        ],
    )

    assert interaction.status in ("completed", "requires_action")
    assert len(interaction.outputs) > 0

    function_calls = [o for o in interaction.outputs if _get_field(o, "type") == "function_call"]
    if function_calls:
        # If the model called a tool, it should have picked get_time
        assert _get_field(function_calls[0], "name") == "get_time"


def test_interactions_no_tools_no_function_call(genai_client, text_model_id):
    """Without tools, the model should never produce function_call outputs."""
    interaction = genai_client.interactions.create(
        model=text_model_id,
        input="What is 2+2? Reply with just the number.",
    )

    assert interaction.status == "completed"
    function_calls = [o for o in interaction.outputs if _get_field(o, "type") == "function_call"]
    assert len(function_calls) == 0, "Should not produce function_call without tools"


def test_interactions_error_missing_model(genai_client):
    """Request without model returns an error."""
    with pytest.raises(Exception):  # noqa: B017
        genai_client.interactions.create(
            model="",
            input="Hello",
        )


def test_interactions_error_invalid_model(genai_client):
    """Request with invalid model returns an error."""
    with pytest.raises(Exception):  # noqa: B017
        genai_client.interactions.create(
            model="nonexistent-model-12345",
            input="Hello",
        )
