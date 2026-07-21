# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest

from .streaming_assertions import StreamingValidator


def provider_from_model(client_with_models, text_model_id):
    models = {m.id: m for m in client_with_models.models.list()}
    models.update(
        {m.custom_metadata["provider_resource_id"]: m for m in client_with_models.models.list() if m.custom_metadata}
    )
    provider_id = models[text_model_id].custom_metadata["provider_id"]
    providers = {p.provider_id: p for p in client_with_models.providers.list()}
    return providers[provider_id]


def skip_if_reasoning_content_not_provided(client_with_models, text_model_id):
    provider_type = provider_from_model(client_with_models, text_model_id).provider_type
    if provider_type in ("remote::openai", "remote::azure", "remote::watsonx"):
        pytest.skip(f"{provider_type} doesn't return reasoning content.")


def test_reasoning_basic_streaming(client_with_models, text_model_id):
    """Test handling of reasoning content in streaming responses."""

    skip_if_reasoning_content_not_provided(client_with_models, text_model_id)

    input = "What is 2 + 2?"

    # Create a streaming response using a reasoning model
    stream = client_with_models.responses.create(
        model=text_model_id,
        input=input,
        stream=True,
    )

    chunks = []
    # Collect all chunks
    for chunk in stream:
        chunks.append(chunk)

    # Validate common streaming events
    validator = StreamingValidator(chunks)
    validator.assert_basic_event_sequence()
    validator.assert_response_consistency()
    validator.assert_rich_streaming()

    # Verify reasoning streaming events are present
    reasoning_text_delta_events = [chunk for chunk in chunks if chunk.type == "response.reasoning_text.delta"]
    reasoning_text_done_events = [chunk for chunk in chunks if chunk.type == "response.reasoning_text.done"]

    event_types = [chunk.type for chunk in chunks]

    assert len(reasoning_text_delta_events) > 0, (
        f"Expected response.reasoning_text.delta events, got chunk types: {event_types}"
    )
    assert len(reasoning_text_done_events) > 0, (
        f"Expected response.reasoning_text.done events, got chunk types: {event_types}"
    )

    assert hasattr(reasoning_text_done_events[-1], "text"), "Reasoning done event should have text field"
    assert len(reasoning_text_done_events[-1].text) > 0, "Reasoning text should not be empty"
