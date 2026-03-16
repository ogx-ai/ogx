# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import time

import pytest

from .fixtures.test_cases import basic_test_cases, image_test_cases, multi_turn_image_test_cases, multi_turn_test_cases
from .streaming_assertions import StreamingValidator


def provider_from_model(client_with_models, text_model_id):
    models = {m.id: m for m in client_with_models.models.list()}
    models.update(
        {m.custom_metadata["provider_resource_id"]: m for m in client_with_models.models.list() if m.custom_metadata}
    )
    provider_id = models[text_model_id].custom_metadata["provider_id"]
    providers = {p.provider_id: p for p in client_with_models.providers.list()}
    return providers[provider_id]


def skip_if_provider_isnt_vllm(client_with_models, text_model_id):
    provider = provider_from_model(client_with_models, text_model_id)
    if provider.provider_type != "remote::vllm":
        pytest.skip(
            f"Model {text_model_id} hosted by {provider.provider_type} doesn't support vllm extra_body parameters."
        )


def skip_if_chat_completions_logprobs_not_supported(client_with_models, text_model_id):
    provider_type = provider_from_model(client_with_models, text_model_id).provider_type
    if provider_type in ("remote::ollama", "remote::watsonx"):
        pytest.skip(f"Model {text_model_id} hosted by {provider_type} doesn't support /v1/chat/completions logprobs.")


@pytest.mark.parametrize("case", basic_test_cases)
def test_response_non_streaming_basic(responses_client, text_model_id, case):
    response = responses_client.responses.create(
        model=text_model_id,
        input=case.input,
        stream=False,
    )
    output_text = response.output_text.lower().strip()
    assert len(output_text) > 0
    assert case.expected.lower() in output_text

    # Verify usage is reported
    assert response.usage is not None, "Response should include usage information"
    assert response.usage.input_tokens > 0, "Input tokens should be greater than 0"
    assert response.usage.output_tokens > 0, "Output tokens should be greater than 0"
    assert response.usage.total_tokens == response.usage.input_tokens + response.usage.output_tokens, (
        "Total tokens should equal input + output tokens"
    )

    retrieved_response = responses_client.responses.retrieve(response_id=response.id)
    assert retrieved_response.output_text == response.output_text

    next_response = responses_client.responses.create(
        model=text_model_id,
        input="Repeat your previous response in all caps.",
        previous_response_id=response.id,
    )
    next_output_text = next_response.output_text.strip()
    assert case.expected.upper() in next_output_text


@pytest.mark.parametrize("case", basic_test_cases)
def test_response_streaming_basic(responses_client, text_model_id, case):
    response = responses_client.responses.create(
        model=text_model_id,
        input=case.input,
        stream=True,
    )

    # Track events and timing to verify proper streaming
    events = []
    event_times = []
    response_id = ""

    start_time = time.time()

    for chunk in response:
        current_time = time.time()
        event_times.append(current_time - start_time)
        events.append(chunk)

        if chunk.type == "response.created":
            # Verify response.created is emitted first and immediately
            assert len(events) == 1, "response.created should be the first event"
            assert event_times[0] < 0.2, (
                f"response.created should be emitted immediately (took {event_times[0]} seconds)"
            )
            assert chunk.response.status == "in_progress"
            response_id = chunk.response.id

        elif chunk.type == "response.completed":
            # Verify response.completed comes after response.created
            assert len(events) >= 2, "response.completed should come after response.created"
            assert chunk.response.status == "completed"
            assert chunk.response.id == response_id, "Response ID should be consistent"

            # Verify content quality
            output_text = chunk.response.output_text.lower().strip()
            assert len(output_text) > 0, "Response should have content"
            assert case.expected.lower() in output_text, f"Expected '{case.expected}' in response"

            # Verify usage is reported in final response
            assert chunk.response.usage is not None, "Completed response should include usage information"
            assert chunk.response.usage.input_tokens > 0, "Input tokens should be greater than 0"
            assert chunk.response.usage.output_tokens > 0, "Output tokens should be greater than 0"
            assert (
                chunk.response.usage.total_tokens
                == chunk.response.usage.input_tokens + chunk.response.usage.output_tokens
            ), "Total tokens should equal input + output tokens"

    # Use validator for common checks
    validator = StreamingValidator(events)
    validator.assert_basic_event_sequence()
    validator.assert_response_consistency()

    # Verify stored response matches streamed response
    retrieved_response = responses_client.responses.retrieve(response_id=response_id)
    final_event = events[-1]
    assert retrieved_response.output_text == final_event.response.output_text


@pytest.mark.parametrize("case", basic_test_cases)
def test_response_streaming_incremental_content(responses_client, text_model_id, case):
    """Test that streaming actually delivers content incrementally, not just at the end."""
    response = responses_client.responses.create(
        model=text_model_id,
        input=case.input,
        stream=True,
    )

    # Track all events and their content to verify incremental streaming
    events = []
    content_snapshots = []
    event_times = []

    start_time = time.time()

    for chunk in response:
        current_time = time.time()
        event_times.append(current_time - start_time)
        events.append(chunk)

        # Track content at each event based on event type
        if chunk.type == "response.output_text.delta":
            # For delta events, track the delta content
            content_snapshots.append(chunk.delta)
        elif hasattr(chunk, "response") and hasattr(chunk.response, "output_text"):
            # For response.created/completed events, track the full output_text
            content_snapshots.append(chunk.response.output_text)
        else:
            content_snapshots.append("")

    validator = StreamingValidator(events)
    validator.assert_basic_event_sequence()

    # Check if we have incremental content updates
    event_types = [event.type for event in events]
    created_index = event_types.index("response.created")
    completed_index = event_types.index("response.completed")

    # The key test: verify content progression
    created_content = content_snapshots[created_index]
    completed_content = content_snapshots[completed_index]

    # Verify that response.created has empty or minimal content
    assert len(created_content) == 0, f"response.created should have empty content, got: {repr(created_content[:100])}"

    # Verify that response.completed has the full content
    assert len(completed_content) > 0, "response.completed should have content"
    assert case.expected.lower() in completed_content.lower(), f"Expected '{case.expected}' in final content"

    # Use validator for incremental content checks
    delta_content_total = validator.assert_has_incremental_content()

    # Verify that the accumulated delta content matches the final content
    assert delta_content_total.strip() == completed_content.strip(), (
        f"Delta content '{delta_content_total}' should match final content '{completed_content}'"
    )

    # Verify timing: delta events should come between created and completed
    delta_events = [i for i, event_type in enumerate(event_types) if event_type == "response.output_text.delta"]
    for delta_idx in delta_events:
        assert created_index < delta_idx < completed_index, (
            f"Delta event at index {delta_idx} should be between created ({created_index}) and completed ({completed_index})"
        )


@pytest.mark.parametrize("case", multi_turn_test_cases)
def test_response_non_streaming_multi_turn(responses_client, text_model_id, case):
    previous_response_id = None
    for turn_input, turn_expected in case.turns:
        response = responses_client.responses.create(
            model=text_model_id,
            input=turn_input,
            previous_response_id=previous_response_id,
        )
        previous_response_id = response.id
        output_text = response.output_text.lower()
        assert turn_expected.lower() in output_text


@pytest.mark.parametrize("case", image_test_cases)
def test_response_non_streaming_image(responses_client, text_model_id, case):
    if text_model_id.startswith("watsonx/"):
        pytest.skip("WatsonX text model does not support vision/image inputs")
    response = responses_client.responses.create(
        model=text_model_id,
        input=case.input,
        stream=False,
    )
    output_text = response.output_text.lower()
    assert case.expected.lower() in output_text


@pytest.mark.parametrize("case", multi_turn_image_test_cases)
def test_response_non_streaming_multi_turn_image(responses_client, text_model_id, case):
    if text_model_id.startswith("watsonx/"):
        pytest.skip("WatsonX text model does not support vision/image inputs")
    previous_response_id = None
    for turn_input, turn_expected in case.turns:
        response = responses_client.responses.create(
            model=text_model_id,
            input=turn_input,
            previous_response_id=previous_response_id,
        )
        previous_response_id = response.id
        output_text = response.output_text.lower()
        assert turn_expected.lower() in output_text


def test_include_logprobs_non_streaming(client_with_models, text_model_id):
    """Test logprobs inclusion in responses with the include parameter."""

    skip_if_chat_completions_logprobs_not_supported(client_with_models, text_model_id)

    input = "Which planet do humans live on?"
    include = ["message.output_text.logprobs"]

    # Create a response without include["message.output_text.logprobs"]
    response_w_o_logprobs = client_with_models.responses.create(
        model=text_model_id,
        input=input,
        stream=False,
    )

    # Verify we got one output message and no logprobs
    assert len(response_w_o_logprobs.output) == 1
    message_outputs = [output for output in response_w_o_logprobs.output if output.type == "message"]
    assert len(message_outputs) == 1, f"Expected one message output, got {len(message_outputs)}"
    assert message_outputs[0].content[0].logprobs == [], "Expected no logprobs in the returned response"

    # Create a response with include["message.output_text.logprobs"]
    response_with_logprobs = client_with_models.responses.create(
        model=text_model_id,
        input=input,
        stream=False,
        include=include,
    )

    # Verify we got one output message and output message has logprobs
    assert len(response_with_logprobs.output) == 1
    message_outputs = [output for output in response_with_logprobs.output if output.type == "message"]
    assert len(message_outputs) == 1, f"Expected one message output, got {len(message_outputs)}"
    assert message_outputs[0].content[0].logprobs is not None, (
        "Expected logprobs in the returned response, but none were returned"
    )


def test_include_logprobs_streaming(client_with_models, text_model_id):
    """Test logprobs inclusion in responses with the include parameter."""

    skip_if_chat_completions_logprobs_not_supported(client_with_models, text_model_id)

    input = "Which planet do humans live on?"
    include = ["message.output_text.logprobs"]

    # Create a streaming response with include["message.output_text.logprobs"]
    stream = client_with_models.responses.create(
        model=text_model_id,
        input=input,
        stream=True,
        include=include,
    )

    for chunk in stream:
        if chunk.type == "response.completed":
            message_outputs = [output for output in chunk.response.output if output.type == "message"]
            assert len(message_outputs) == 1, f"Expected one message output, got {len(message_outputs)}"
            assert message_outputs[0].content[0].logprobs is not None, (
                f"Expected logprobs in the returned chunk ({chunk.type=}), but none were returned"
            )
        elif chunk.type == "response.output_item.done":
            content = chunk.item.content
            assert len(content) == 1, f"Expected one content object, got {len(content)}"
            assert content[0].logprobs is not None, (
                f"Expected logprobs in the returned chunk ({chunk.type=}), but none were returned"
            )
        elif chunk.type in ["response.output_text.delta", "response.output_text.done"]:
            assert chunk.logprobs is not None, (
                f"Expected logprobs in the returned chunk ({chunk.type=}), but none were returned"
            )
        elif chunk.type == "response.content_part.done":
            assert chunk.part.logprobs == [], f"Expected no logprobs in the returned chunk ({chunk.type=})"


def test_include_logprobs_with_web_search(client_with_models, text_model_id):
    """Test include logprobs with built-in tool."""

    skip_if_chat_completions_logprobs_not_supported(client_with_models, text_model_id)

    input = "Search for a positive news story from today."
    include = ["message.output_text.logprobs"]
    tools = [
        {
            "type": "web_search",
        }
    ]

    # Create a response with built-in tool and include["message.output_text.logprobs"]
    response = client_with_models.responses.create(
        model=text_model_id,
        input=input,
        stream=False,
        include=include,
        tools=tools,
    )

    # Verify we got one built-in tool call and output message has logprobs
    assert len(response.output) >= 2
    assert response.output[0].type == "web_search_call"
    assert response.output[0].status == "completed"
    message_outputs = [output for output in response.output if output.type == "message"]
    assert len(message_outputs) == 1, f"Expected one message output, got {len(message_outputs)}"
    assert message_outputs[0].content[0].logprobs is not None, (
        "Expected logprobs in the returned response, but none were returned"
    )


def test_include_logprobs_with_function_tools(client_with_models, text_model_id):
    """Test include logprobs with function tools."""

    skip_if_chat_completions_logprobs_not_supported(client_with_models, text_model_id)

    input = "What is the weather in Paris?"
    include = ["message.output_text.logprobs"]
    tools = [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get weather information for a specified location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name (e.g., 'New York', 'London')",
                    },
                },
            },
        },
    ]

    # Create a response with function tool and include["message.output_text.logprobs"]
    response = client_with_models.responses.create(
        model=text_model_id,
        input=input,
        stream=False,
        include=include,
        tools=tools,
    )

    # Verify we got one function tool call and no logprobs
    assert len(response.output) == 1
    assert response.output[0].type == "function_call"
    assert response.output[0].name == "get_weather"
    assert response.output[0].status == "completed"
    message_outputs = [output for output in response.output if output.type == "message"]
    assert len(message_outputs) == 0, f"Expected no message output, got {len(message_outputs)}"


def test_response_extra_body_guided_choice(client_with_models, text_model_id):
    """Test that extra_body parameters pass through the responses API to the backend (see #3777)."""
    skip_if_provider_isnt_vllm(client_with_models, text_model_id)

    response = client_with_models.responses.create(
        model=text_model_id,
        input="I am feeling really sad today.",
        stream=False,
        extra_body={"structured_outputs": {"choice": ["joy", "sadness"]}},
    )
    assert len(response.output) > 0
    output_text = response.output_text.strip()
    assert output_text in ["joy", "sadness"]
