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
        if output.type == "text":
            return output
    return None


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
    assert text_output is not None, f"Expected a text output, got types: {[o.type for o in interaction.outputs]}"
    assert len(text_output.text) > 0
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
    assert len(text_output.text) > 0


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
    assert "alice" in text_output.text.lower()


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
    assert len(text_output.text) > 0


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
        event_name = type(event).__name__
        event_types.append(event_name)

        if event_name == "InteractionStartEvent" and hasattr(event, "interaction") and event.interaction:
            interaction_id = event.interaction.id

        if event_name == "ContentDelta" and hasattr(event, "delta") and event.delta and hasattr(event.delta, "text"):
            text_parts.append(event.delta.text)

    full_text = "".join(text_parts)
    assert len(full_text) > 0, "Streaming should produce text"
    assert interaction_id is not None, "Should have received an interaction ID"

    # Verify event sequence contains expected types
    assert "InteractionStartEvent" in event_types
    assert "ContentStart" in event_types
    assert "ContentDelta" in event_types
    assert "ContentStop" in event_types
    assert "InteractionCompleteEvent" in event_types


def test_interactions_streaming_text_concatenation(genai_client, text_model_id):
    """Streaming text deltas can be concatenated into the full response."""
    stream = genai_client.interactions.create(
        model=text_model_id,
        input="Say hello in one sentence.",
        stream=True,
    )

    text_parts = []
    for event in stream:
        if type(event).__name__ == "ContentDelta" and hasattr(event, "delta") and event.delta:
            if hasattr(event.delta, "text"):
                text_parts.append(event.delta.text)

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
    event_names = [type(e).__name__ for e in events]
    assert len(events) >= 4, f"Expected at least 4 events, got {len(events)}: {event_names}"

    # Verify all required event types are present
    required = ["InteractionStartEvent", "ContentStart", "ContentDelta", "ContentStop", "InteractionCompleteEvent"]
    for req in required:
        assert req in event_names, f"Missing required event type {req}, got: {event_names}"

    # Verify relative ordering: start before content, content before complete
    def _first(name):
        return event_names.index(name)

    assert _first("InteractionStartEvent") < _first("ContentStart"), "InteractionStartEvent should precede ContentStart"
    assert _first("ContentStart") < _first("ContentDelta"), "ContentStart should precede ContentDelta"
    assert _first("ContentStop") < _first("InteractionCompleteEvent"), (
        "ContentStop should precede InteractionCompleteEvent"
    )


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
