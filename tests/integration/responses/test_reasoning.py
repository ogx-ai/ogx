# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest

from .streaming_assertions import StreamingValidator


def _get_attr(item, key, default=None):
    """Get attribute from typed object or dict — works with both
    the current OGX client (returns dicts) and OpenAI client
    (returns typed objects)."""
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def provider_from_model(client_with_models, text_model_id):
    models = {m.id: m for m in client_with_models.models.list().data}
    models.update(
        {
            m.custom_metadata["provider_resource_id"]: m
            for m in client_with_models.models.list().data
            if m.custom_metadata
        }
    )
    provider_id = models[text_model_id].custom_metadata["provider_id"]
    providers = {p.provider_id: p for p in client_with_models.providers.list()}
    return providers[provider_id]


def skip_if_reasoning_content_not_provided(client_with_models, text_model_id):
    provider_type = provider_from_model(client_with_models, text_model_id).provider_type
    if provider_type in (
        "remote::openai",
        "remote::azure",
        "remote::watsonx",
        "remote::vertexai",
        "remote::vllm",
        "remote::bedrock",
    ):
        pytest.skip(f"{provider_type} doesn't return reasoning content.")


def test_reasoning_basic_streaming(client_with_models, text_model_id):
    """Test handling of reasoning content in streaming responses."""

    skip_if_reasoning_content_not_provided(client_with_models, text_model_id)

    input = "What is 2 + 2? Think Step by Step !"

    # Create a streaming response using a reasoning model
    stream = client_with_models.responses.create(
        model=text_model_id,
        input=input,
        stream=True,
        reasoning={"effort": "high"},
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


def test_reasoning_non_streaming(client_with_models, text_model_id):
    """Test that ReasoningItem appears in non-streaming response output."""

    skip_if_reasoning_content_not_provided(client_with_models, text_model_id)

    response = client_with_models.responses.create(
        model=text_model_id,
        input="What is 2 + 2? Think step by step.",
        reasoning={"effort": "medium"},
        stream=False,
    )

    output_types = [_get_attr(item, "type") for item in response.output]
    reasoning_items = [item for item in response.output if _get_attr(item, "type") == "reasoning"]

    assert len(reasoning_items) > 0, f"Expected reasoning items in output, got types: {output_types}"

    reasoning_item = reasoning_items[0]
    assert _get_attr(reasoning_item, "id"), "Reasoning item should have an id"
    content = _get_attr(reasoning_item, "content")
    assert content is not None, "Reasoning item should have content"
    assert len(content) > 0, "Reasoning item should have at least one content entry"
    assert _get_attr(content[0], "type") == "reasoning_text"
    assert len(_get_attr(content[0], "text", "")) > 0, "Reasoning content text should not be empty"


def test_reasoning_multi_turn_passthrough(client_with_models, text_model_id):
    """Test that reasoning output survives a round-trip when passed back as input.

    Turn 1: send a prompt, assert ReasoningItem in output.
    Turn 2: pass the full output (including reasoning) back as input with
            a follow-up question, assert the model responds coherently.
    """

    skip_if_reasoning_content_not_provided(client_with_models, text_model_id)

    # Turn 1
    input_messages = [
        {"role": "user", "content": "What is 2 + 2? Think step by step."},
    ]
    resp1 = client_with_models.responses.create(
        model=text_model_id,
        input=input_messages,
        reasoning={"effort": "medium"},
        stream=False,
    )

    output_types = [_get_attr(item, "type") for item in resp1.output]
    reasoning_items = [item for item in resp1.output if _get_attr(item, "type") == "reasoning"]
    assert len(reasoning_items) > 0, f"Expected reasoning items in turn 1, got types: {output_types}"

    # Turn 2: pass previous output back as input with a follow-up
    turn2_input = list(input_messages) + list(resp1.output)
    turn2_input.append({"role": "user", "content": "Now multiply that result by 3."})

    resp2 = client_with_models.responses.create(
        model=text_model_id,
        input=turn2_input,
        reasoning={"effort": "medium"},
        stream=False,
    )

    assert resp2.output, "Expected non-empty output in turn 2"
    message_items = [item for item in resp2.output if _get_attr(item, "type") == "message"]
    assert len(message_items) > 0, "Expected a message in turn 2 output"


# ---------------------------------------------------------------------------
# Reasoning summary integration tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("summary_mode", ["concise", "detailed", "auto"])
def test_reasoning_summary_streaming(client_with_models, text_model_id, summary_mode):
    """Test that reasoning summary is attached to the reasoning item in streaming mode."""

    skip_if_reasoning_content_not_provided(client_with_models, text_model_id)

    stream = client_with_models.responses.create(
        model=text_model_id,
        input="What is 2 + 2? Think step by step.",
        stream=True,
        reasoning={"effort": "high", "summary": summary_mode},
    )

    chunks = list(stream)
    event_types = [chunk.type for chunk in chunks]

    validator = StreamingValidator(chunks)
    validator.assert_basic_event_sequence()
    validator.assert_response_consistency()

    reasoning_item_done = [
        c
        for c in chunks
        if c.type == "response.output_item.done" and _get_attr(_get_attr(c, "item"), "type") == "reasoning"
    ]
    assert len(reasoning_item_done) > 0, f"Expected output_item.done for reasoning, got types: {event_types}"

    item = _get_attr(reasoning_item_done[0], "item")
    summary = _get_attr(item, "summary")
    assert summary is not None and len(summary) > 0, "Reasoning item should have a non-empty summary"

    summary_text = _get_attr(summary[0], "text", "")
    assert len(summary_text) > 0, "Summary text should not be empty"


def test_reasoning_summary_non_streaming(client_with_models, text_model_id):
    """Test that reasoning summary appears in non-streaming response output."""

    skip_if_reasoning_content_not_provided(client_with_models, text_model_id)

    response = client_with_models.responses.create(
        model=text_model_id,
        input="What is 2 + 2? Think step by step.",
        reasoning={"effort": "medium", "summary": "concise"},
        stream=False,
    )

    reasoning_items = [item for item in response.output if _get_attr(item, "type") == "reasoning"]
    assert len(reasoning_items) > 0, "Expected reasoning items in output"

    reasoning_item = reasoning_items[0]
    summary = _get_attr(reasoning_item, "summary")
    assert summary is not None, "Reasoning item should have a summary when summary is requested"
    assert len(summary) > 0, "Summary list should not be empty"

    summary_entry = summary[0]
    summary_text = _get_attr(summary_entry, "text", "")
    assert len(summary_text) > 0, "Summary text should not be empty"


def test_reasoning_summary_event_ordering(client_with_models, text_model_id):
    """Verify that the reasoning item's OutputItemDone carries the summary and comes after OutputItemAdded."""

    skip_if_reasoning_content_not_provided(client_with_models, text_model_id)

    stream = client_with_models.responses.create(
        model=text_model_id,
        input="What is 2 + 2? Think step by step.",
        stream=True,
        reasoning={"effort": "high", "summary": "concise"},
    )

    chunks = list(stream)
    event_types = [chunk.type for chunk in chunks]

    reasoning_item_added_idx = None
    reasoning_item_done_idx = None
    for i, c in enumerate(chunks):
        item_type = _get_attr(_get_attr(c, "item"), "type")
        if item_type == "reasoning":
            if c.type == "response.output_item.added":
                reasoning_item_added_idx = i
            elif c.type == "response.output_item.done":
                reasoning_item_done_idx = i

    assert reasoning_item_added_idx is not None, (
        f"Expected output_item.added for reasoning item, got types: {event_types}"
    )
    assert reasoning_item_done_idx is not None, (
        f"Expected output_item.done for reasoning item, got types: {event_types}"
    )
    assert reasoning_item_done_idx > reasoning_item_added_idx, (
        "output_item.done should come after output_item.added for reasoning"
    )

    done_item = _get_attr(chunks[reasoning_item_done_idx], "item")
    summary = _get_attr(done_item, "summary")
    assert summary is not None and len(summary) > 0, "OutputItemDone reasoning item should carry the summary"


def test_reasoning_no_summary_without_request(client_with_models, text_model_id):
    """Verify that no summary is attached when summary is not requested."""

    skip_if_reasoning_content_not_provided(client_with_models, text_model_id)

    stream = client_with_models.responses.create(
        model=text_model_id,
        input="What is 2 + 2? Think step by step.",
        stream=True,
        reasoning={"effort": "high"},
    )

    chunks = list(stream)
    reasoning_item_done = [
        c
        for c in chunks
        if c.type == "response.output_item.done" and _get_attr(_get_attr(c, "item"), "type") == "reasoning"
    ]

    if reasoning_item_done:
        item = _get_attr(reasoning_item_done[0], "item")
        summary = _get_attr(item, "summary")
        assert summary is None or len(summary) == 0, f"Expected no summary when not requested, got: {summary}"


def test_reasoning_summary_usage_included(client_with_models, text_model_id):
    """Test that token usage in the final response includes tokens from the summary inference call."""

    skip_if_reasoning_content_not_provided(client_with_models, text_model_id)

    response = client_with_models.responses.create(
        model=text_model_id,
        input="What is 2 + 2? Think step by step.",
        reasoning={"effort": "medium", "summary": "concise"},
        stream=False,
    )

    usage = _get_attr(response, "usage")
    assert usage is not None, "Response with summary should have usage data"

    total_tokens = _get_attr(usage, "total_tokens", 0)
    assert total_tokens > 0, "Total tokens should be positive when summary is generated"
