# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import patch

import pytest
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)

from ogx_api import (
    InternalServerError,
    OpenAISystemMessageParam,
)
from ogx_api.openai_responses import (
    OpenAIResponseError,
    OpenAIResponseInputToolWebSearch,
    OpenAIResponseObject,
    OpenAIResponseObjectStreamResponseFailed,
)
from ogx_api.tools import ToolDef, ToolInvocationResult


async def test_failed_stream_persists_non_system_messages(openai_responses_impl, mock_responses_store):
    input_text = "Hello"
    model = "meta-llama/Llama-3.1-8B-Instruct"

    failed_response = OpenAIResponseObject(
        created_at=1,
        id="resp_failed",
        model=model,
        output=[],
        status="failed",
        error=OpenAIResponseError(code="server_error", message="boom"),
        store=True,
    )

    class FakeOrchestrator:
        def __init__(self, *, ctx, **_kwargs):
            self.ctx = ctx
            self.final_messages = None

        async def create_response(self):
            yield OpenAIResponseObjectStreamResponseFailed(response=failed_response, sequence_number=0)

    with patch(
        "ogx.providers.inline.responses.builtin.responses.openai_responses.StreamingResponseOrchestrator",
        FakeOrchestrator,
    ):
        stream = await openai_responses_impl.create_openai_response(
            input=input_text,
            model=model,
            instructions="system instructions",
            stream=True,
            store=True,
        )
        chunks = [chunk async for chunk in stream]

    assert chunks[-1].type == "response.failed"
    mock_responses_store.upsert_response_object.assert_awaited()

    # Find the call that corresponds to the failed response
    call_args_list = mock_responses_store.upsert_response_object.call_args_list
    failed_call = None
    for call in call_args_list:
        _, kwargs = call
        if kwargs.get("response_object") and kwargs["response_object"].status == "failed":
            failed_call = call
            break

    assert failed_call is not None, "Expected upsert_response_object to be called with failed response"
    _, kwargs = failed_call
    messages = kwargs["messages"]
    assert messages, "Expected non-system messages to be persisted on failure"
    assert all(not isinstance(m, OpenAISystemMessageParam) for m in messages)
    assert any(getattr(m, "role", None) == "user" for m in messages)


async def test_failed_stream_raises_internal_server_error_in_non_streaming_mode(openai_responses_impl):
    """Test that a response.failed event in non-streaming mode raises InternalServerError.

    When stream=False, the caller expects a fully resolved response object, not a stream.
    If the underlying stream emits a response.failed event, the implementation must raise
    InternalServerError so the caller gets a typed, predictable error rather than a raw
    RuntimeError or ValueError.

    Unlike other InternalServerError sites in this file (which guard against internal bugs),
    response.failed carries a structured, curated message from the inference backend that
    may be directly actionable by the caller (e.g. context window exceeded, invalid prompt).
    The message is surfaced to maintain consistency with streaming mode, where the same
    response.failed event is returned directly to the caller with the error message visible.
    """
    model = "meta-llama/Llama-3.1-8B-Instruct"
    provider_error_message = "This model's maximum context length is 4096 tokens"

    failed_response = OpenAIResponseObject(
        created_at=1,
        id="resp_failed_nonstream",
        model=model,
        output=[],
        status="failed",
        error=OpenAIResponseError(code="server_error", message=provider_error_message),
        store=False,
    )

    class FakeOrchestrator:
        def __init__(self, *, ctx, **_kwargs):
            self.ctx = ctx
            self.final_messages = None

        async def create_response(self):
            yield OpenAIResponseObjectStreamResponseFailed(response=failed_response, sequence_number=0)

    with patch(
        "ogx.providers.inline.responses.builtin.responses.openai_responses.StreamingResponseOrchestrator",
        FakeOrchestrator,
    ):
        with pytest.raises(InternalServerError) as exc_info:
            await openai_responses_impl.create_openai_response(
                input="Hello",
                model=model,
                stream=False,
                store=False,
            )

    # The provider message is surfaced to the caller: response.failed errors are
    # structured and may be actionable (e.g. context window, invalid prompt).
    # This is consistent with streaming mode where the same message is visible.
    assert provider_error_message in str(exc_info.value)


def test_response_object_incomplete_details_null_when_completed():
    """Test that completed response has incomplete_details as null."""
    from ogx_api.openai_responses import OpenAIResponseObject

    response = OpenAIResponseObject(
        created_at=1234567890,
        id="resp_123",
        model="gpt-4o",
        object="response",
        output=[],
        status="completed",
        store=False,
    )

    assert response.incomplete_details is None

    # Verify JSON serialization
    json_data = response.model_dump(mode="json")
    assert json_data["incomplete_details"] is None


def test_response_object_incomplete_details_with_max_output_tokens_reason():
    """Test that incomplete response has incomplete_details with max_output_tokens reason."""
    from ogx_api.openai_responses import OpenAIResponseIncompleteDetails, OpenAIResponseObject

    response = OpenAIResponseObject(
        created_at=1234567890,
        id="resp_456",
        model="gpt-4o",
        object="response",
        output=[],
        status="incomplete",
        store=False,
        incomplete_details=OpenAIResponseIncompleteDetails(reason="max_output_tokens"),
    )

    assert response.incomplete_details is not None
    assert response.incomplete_details.reason == "max_output_tokens"

    # Verify JSON serialization
    json_data = response.model_dump(mode="json")
    assert json_data["incomplete_details"] == {"reason": "max_output_tokens"}


def test_response_object_incomplete_details_with_length_reason():
    """Test that incomplete response has incomplete_details with length reason."""
    from ogx_api.openai_responses import OpenAIResponseIncompleteDetails, OpenAIResponseObject

    response = OpenAIResponseObject(
        created_at=1234567890,
        id="resp_length",
        model="gpt-4o",
        object="response",
        output=[],
        status="incomplete",
        store=False,
        incomplete_details=OpenAIResponseIncompleteDetails(reason="length"),
    )

    assert response.incomplete_details is not None
    assert response.incomplete_details.reason == "length"

    # Verify JSON serialization
    json_data = response.model_dump(mode="json")
    assert json_data["incomplete_details"] == {"reason": "length"}


def test_response_object_incomplete_details_with_max_iterations_exceeded_reason():
    """Test that incomplete response has incomplete_details with max_iterations_exceeded reason."""
    from ogx_api.openai_responses import OpenAIResponseIncompleteDetails, OpenAIResponseObject

    response = OpenAIResponseObject(
        created_at=1234567890,
        id="resp_iters",
        model="gpt-4o",
        object="response",
        output=[],
        status="incomplete",
        store=False,
        incomplete_details=OpenAIResponseIncompleteDetails(reason="max_iterations_exceeded"),
    )

    assert response.incomplete_details is not None
    assert response.incomplete_details.reason == "max_iterations_exceeded"

    # Verify JSON serialization
    json_data = response.model_dump(mode="json")
    assert json_data["incomplete_details"] == {"reason": "max_iterations_exceeded"}


async def test_agent_loop_incomplete_due_to_max_output_tokens(
    openai_responses_impl, mock_inference_api, mock_tool_groups_api, mock_tool_runtime_api
):
    """Test that agent loop marks response incomplete when max_output_tokens is reached."""
    from openai.types.completion_usage import CompletionUsage

    model = "gpt-4o"
    max_output_tokens = 25  # Set low enough to be exceeded by first tool call

    # Setup tool mocks - the tool call will trigger a second iteration
    mock_tool_groups_api.get_tool.return_value = ToolDef(
        name="web_search", description="Search the web", input_schema={}
    )
    mock_tool_runtime_api.invoke_tool.return_value = ToolInvocationResult(content="Search results")

    # First stream: returns a tool call after consuming 30 tokens (exceeds limit of 25)
    async def first_stream_with_tool_call():
        yield ChatCompletionChunk(
            id="test_123",
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(
                        role="assistant",
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                id="call_abc",
                                function=ChoiceDeltaToolCallFunction(name="web_search", arguments='{"query":"test"}'),
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
            created=1234567890,
            model=model,
            object="chat.completion.chunk",
            usage=CompletionUsage(prompt_tokens=10, completion_tokens=30, total_tokens=40),
        )

    # Second stream would consume more tokens, but should be skipped because we already have 30 tokens
    # and the next iteration check will see we're at/above the limit
    async def second_stream():
        yield ChatCompletionChunk(
            id="test_456",
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(content="More content", role="assistant"),
                    finish_reason="stop",
                )
            ],
            created=1234567890,
            model=model,
            object="chat.completion.chunk",
            usage=CompletionUsage(prompt_tokens=10, completion_tokens=25, total_tokens=35),
        )

    # Mock returns first stream with tool call, then would return second stream but should be blocked
    mock_inference_api.openai_chat_completion.side_effect = [first_stream_with_tool_call(), second_stream()]

    # Execute with max_output_tokens that will be exceeded after first tool call
    result = await openai_responses_impl.create_openai_response(
        input="Test input",
        model=model,
        max_output_tokens=max_output_tokens,
        tools=[OpenAIResponseInputToolWebSearch(type="web_search")],
        stream=True,
    )

    # Collect all events
    events = [event async for event in result]

    # Find the final event (should be response.incomplete)
    final_event = events[-1]
    assert final_event.type == "response.incomplete"
    assert final_event.response.status == "incomplete"
    assert final_event.response.incomplete_details is not None
    assert final_event.response.incomplete_details.reason == "max_output_tokens"


async def test_agent_loop_incomplete_due_to_max_iterations(
    openai_responses_impl, mock_inference_api, mock_tool_groups_api, mock_tool_runtime_api
):
    """Test that agent loop marks response incomplete when max iterations is exceeded via tool calls."""
    from openai.types.completion_usage import CompletionUsage

    model = "gpt-4o"

    # Setup tool mocks
    mock_tool_groups_api.get_tool.return_value = ToolDef(
        name="web_search", description="Search the web", input_schema={}
    )
    mock_tool_runtime_api.invoke_tool.return_value = ToolInvocationResult(content="Search results")

    # Create a stream generator factory that returns a tool call (to trigger another iteration)
    call_counter = {"count": 0}

    def fake_stream_factory():
        async def fake_stream_with_tool_call():
            call_id = f"call_abc{call_counter['count']}"
            call_counter["count"] += 1
            # First chunk with tool call
            yield ChatCompletionChunk(
                id="test_123",
                choices=[
                    Choice(
                        index=0,
                        delta=ChoiceDelta(
                            role="assistant",
                            tool_calls=[
                                ChoiceDeltaToolCall(
                                    index=0,
                                    id=call_id,
                                    function=ChoiceDeltaToolCallFunction(
                                        name="web_search", arguments='{"query":"test"}'
                                    ),
                                )
                            ],
                        ),
                        finish_reason="tool_calls",
                    )
                ],
                created=1234567890,
                model=model,
                object="chat.completion.chunk",
                usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            )

        return fake_stream_with_tool_call()

    # Mock the inference to repeatedly return tool calls to exceed max iterations
    # Default max_infer_iters is 5, so we need to keep returning tool calls
    mock_inference_api.openai_chat_completion.side_effect = lambda *args, **kwargs: fake_stream_factory()

    # Execute with tool configuration
    result = await openai_responses_impl.create_openai_response(
        input="Test input",
        model=model,
        tools=[OpenAIResponseInputToolWebSearch(type="web_search")],
        stream=True,
    )

    # Collect all events
    events = [event async for event in result]

    # Find the final event (should be response.incomplete)
    final_event = events[-1]
    assert final_event.type == "response.incomplete"
    assert final_event.response.status == "incomplete"
    assert final_event.response.incomplete_details is not None
    assert final_event.response.incomplete_details.reason == "max_iterations_exceeded"


async def test_agent_loop_incomplete_due_to_length_finish_reason(openai_responses_impl, mock_inference_api):
    """Test that agent loop marks response incomplete when model returns finish_reason='length'."""
    from openai.types.completion_usage import CompletionUsage

    model = "gpt-4o"

    # Create a stream that returns finish_reason="length"
    async def fake_stream_with_length_finish():
        yield ChatCompletionChunk(
            id="test_123",
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(content="This is a response that was cut off due to", role="assistant"),
                    finish_reason=None,
                )
            ],
            created=1234567890,
            model=model,
            object="chat.completion.chunk",
            usage=CompletionUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )
        # Final chunk with finish_reason="length"
        yield ChatCompletionChunk(
            id="test_123",
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(content=" length"),
                    finish_reason="length",  # This indicates the response was truncated
                )
            ],
            created=1234567890,
            model=model,
            object="chat.completion.chunk",
            usage=CompletionUsage(prompt_tokens=10, completion_tokens=25, total_tokens=35),
        )

    mock_inference_api.openai_chat_completion.return_value = fake_stream_with_length_finish()

    # Execute
    result = await openai_responses_impl.create_openai_response(
        input="Test input",
        model=model,
        stream=True,
    )

    # Collect all events
    events = [event async for event in result]

    # Find the final event (should be response.incomplete)
    final_event = events[-1]
    assert final_event.type == "response.incomplete"
    assert final_event.response.status == "incomplete"
    assert final_event.response.incomplete_details is not None
    assert final_event.response.incomplete_details.reason == "length"
