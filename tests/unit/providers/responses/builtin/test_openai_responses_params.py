# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)

from ogx.providers.inline.responses.builtin.responses.openai_responses import (
    OpenAIResponsesImpl,
)
from ogx.providers.remote.inference.openai.config import OpenAIConfig
from ogx.providers.remote.inference.openai.openai import OpenAIInferenceAdapter
from ogx.providers.utils.responses.responses_store import (
    _OpenAIResponseObjectWithInputAndMessages,
)
from ogx_api import (
    ResponseStreamOptions,
    ResponseTruncation,
)
from ogx_api.inference import (
    OpenAIAssistantMessageParam,
    OpenAIUserMessageParam,
    ServiceTier,
)
from ogx_api.openai_responses import (
    OpenAIResponseInputToolFunction,
    OpenAIResponseMessage,
    OpenAIResponseText,
    OpenAIResponseTextFormat,
)
from ogx_api.tools import ToolDef, ToolInvocationResult
from tests.unit.providers.responses.builtin.test_openai_responses_helpers import fake_stream


async def test_create_openai_response_with_max_output_tokens_non_streaming(
    openai_responses_impl, mock_inference_api, mock_responses_store
):
    """Test that max_output_tokens is properly handled in non-streaming responses."""
    input_text = "Write a long story about AI."
    model = "meta-llama/Llama-3.1-8B-Instruct"
    max_tokens = 100

    mock_inference_api.openai_chat_completion.return_value = fake_stream()

    # Execute
    result = await openai_responses_impl.create_openai_response(
        input=input_text,
        model=model,
        max_output_tokens=max_tokens,
        stream=False,
        store=True,
    )

    # Verify response includes the max_output_tokens
    assert result.max_output_tokens == max_tokens
    assert result.model == model
    assert result.status == "completed"

    # Verify the max_output_tokens was passed to inference API
    mock_inference_api.openai_chat_completion.assert_called()
    call_args = mock_inference_api.openai_chat_completion.call_args
    params = call_args.args[0]
    assert params.max_completion_tokens == max_tokens

    # Verify the max_output_tokens was stored
    mock_responses_store.upsert_response_object.assert_called()
    store_call_args = mock_responses_store.upsert_response_object.call_args
    stored_response = store_call_args.kwargs["response_object"]
    assert stored_response.max_output_tokens == max_tokens


async def test_create_openai_response_with_max_output_tokens_streaming(
    openai_responses_impl, mock_inference_api, mock_responses_store
):
    """Test that max_output_tokens is properly handled in streaming responses."""
    input_text = "Explain machine learning in detail."
    model = "meta-llama/Llama-3.1-8B-Instruct"
    max_tokens = 200

    mock_inference_api.openai_chat_completion.return_value = fake_stream()

    # Execute
    result = await openai_responses_impl.create_openai_response(
        input=input_text,
        model=model,
        max_output_tokens=max_tokens,
        stream=True,
        store=True,
    )

    # Collect all chunks
    chunks = [chunk async for chunk in result]

    # Verify max_output_tokens is in the created event
    created_event = chunks[0]
    assert created_event.type == "response.created"
    assert created_event.response.max_output_tokens == max_tokens

    # Verify max_output_tokens is in the completed event
    completed_event = chunks[-1]
    assert completed_event.type == "response.completed"
    assert completed_event.response.max_output_tokens == max_tokens

    # Verify the max_output_tokens was passed to inference API
    mock_inference_api.openai_chat_completion.assert_called()
    call_args = mock_inference_api.openai_chat_completion.call_args
    params = call_args.args[0]
    assert params.max_completion_tokens == max_tokens

    # Verify the max_output_tokens was stored
    mock_responses_store.upsert_response_object.assert_called()
    store_call_args = mock_responses_store.upsert_response_object.call_args
    stored_response = store_call_args.kwargs["response_object"]
    assert stored_response.max_output_tokens == max_tokens


async def test_create_openai_response_with_max_output_tokens_boundary_value(openai_responses_impl, mock_inference_api):
    """Test that max_output_tokens accepts the minimum valid value of 16."""
    input_text = "Hi"
    model = "meta-llama/Llama-3.1-8B-Instruct"

    mock_inference_api.openai_chat_completion.return_value = fake_stream()

    # Execute with minimum valid value
    result = await openai_responses_impl.create_openai_response(
        input=input_text,
        model=model,
        max_output_tokens=16,
        stream=False,
    )

    # Verify it accepts 16
    assert result.max_output_tokens == 16
    assert result.status == "completed"

    # Verify the inference API was called with max_completion_tokens=16
    mock_inference_api.openai_chat_completion.assert_called()
    call_args = mock_inference_api.openai_chat_completion.call_args
    params = call_args.args[0]
    assert params.max_completion_tokens == 16


async def test_create_openai_response_with_max_output_tokens_and_tools(openai_responses_impl, mock_inference_api):
    """Test that max_output_tokens works correctly with tool calls."""
    input_text = "What's the weather in San Francisco?"
    model = "meta-llama/Llama-3.1-8B-Instruct"
    max_tokens = 150

    openai_responses_impl.tool_groups_api.get_tool.return_value = ToolDef(
        name="get_weather",
        toolgroup_id="weather",
        description="Get weather information",
        input_schema={
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    )

    openai_responses_impl.tool_runtime_api.invoke_tool.return_value = ToolInvocationResult(
        status="completed",
        content="Sunny, 72°F",
    )

    # Mock two inference calls: one for tool call, one for final response
    mock_inference_api.openai_chat_completion.side_effect = [
        fake_stream("tool_call_completion.yaml"),
        fake_stream(),
    ]

    # Execute
    result = await openai_responses_impl.create_openai_response(
        input=input_text,
        model=model,
        max_output_tokens=max_tokens,
        stream=False,
        tools=[
            OpenAIResponseInputToolFunction(
                name="get_weather",
                description="Get weather information",
                parameters={"location": "string"},
            )
        ],
    )

    # Verify max_output_tokens is preserved
    assert result.max_output_tokens == max_tokens
    assert result.status == "completed"

    # Verify both inference calls received max_completion_tokens
    assert mock_inference_api.openai_chat_completion.call_count == 2
    for call in mock_inference_api.openai_chat_completion.call_args_list:
        params = call.args[0]
        # The first call gets the full max_tokens, subsequent calls get remaining tokens
        assert params.max_completion_tokens is not None
        assert params.max_completion_tokens <= max_tokens


@pytest.mark.parametrize("store", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "param_name,param_value,backend_param_name,backend_expected_value,response_expected_value,stored_expected_value",
    [
        ("temperature", 1.5, "temperature", 1.5, 1.5, 1.5),
        ("safety_identifier", "user-123", "safety_identifier", "user-123", "user-123", "user-123"),
        ("max_output_tokens", 500, "max_completion_tokens", 500, 500, 500),
        (
            "prompt_cache_key",
            "geography-cache-001",
            "prompt_cache_key",
            "geography-cache-001",
            "geography-cache-001",
            "geography-cache-001",
        ),
        ("service_tier", ServiceTier.flex, "service_tier", "flex", "flex", ServiceTier.default.value),
        ("top_p", 0.9, "top_p", 0.9, 0.9, 0.9),
        ("frequency_penalty", 0.5, "frequency_penalty", 0.5, 0.5, 0.5),
        ("presence_penalty", 0.3, "presence_penalty", 0.3, 0.3, 0.3),
        ("top_logprobs", 5, "top_logprobs", 5, 5, 5),
        (
            "extra_body",
            {"chat_template_kwargs": {"thinking": True}},
            "extra_body",
            {"chat_template_kwargs": {"thinking": True}},
            None,
            None,
        ),
    ],
)
async def test_params_passed_through_full_chain_to_backend_service(
    param_name,
    param_value,
    backend_param_name,
    backend_expected_value,
    response_expected_value,
    stored_expected_value,
    stream,
    store,
    mock_responses_store,
):
    """Test that parameters which pass through to the backend service are correctly propagated.

    Only parameters that are forwarded as kwargs to the underlying chat completions API belong
    here. Parameters handled internally by the responses layer (e.g. truncation) should be
    tested separately since they don't produce a backend kwarg assertion.

    This test should not act differently based on the param_name/param_value/etc. Needing changes
    in behavior based on those params suggests a bug in the implementation.

    This test may act differently based on :
      - stream: whether the response is streamed or not
      - store: whether the response is persisted via the responses store
    """
    config = OpenAIConfig(api_key="test-key")
    openai_adapter = OpenAIInferenceAdapter(config=config)
    openai_adapter.provider_data_api_key_field = None

    mock_model_store = AsyncMock()
    mock_model_store.has_model = AsyncMock(return_value=False)
    openai_adapter.model_store = mock_model_store

    openai_responses_impl = OpenAIResponsesImpl(
        inference_api=openai_adapter,
        tool_groups_api=AsyncMock(),
        tool_runtime_api=AsyncMock(),
        responses_store=mock_responses_store,
        vector_io_api=AsyncMock(),
        safety_api=AsyncMock(),
        conversations_api=AsyncMock(),
        prompts_api=AsyncMock(),
        files_api=AsyncMock(),
        connectors_api=AsyncMock(),
    )

    with patch("ogx.providers.utils.inference.openai_mixin.AsyncOpenAI") as mock_openai_class:
        mock_client = MagicMock()
        mock_chat_completions = AsyncMock()
        mock_client.chat.completions.create = mock_chat_completions
        mock_openai_class.return_value = mock_client

        if stream:
            mock_chat_completions.return_value = fake_stream()
        else:
            mock_response = MagicMock()
            mock_response.id = "chatcmpl-123"
            mock_response.choices = [
                MagicMock(
                    index=0,
                    message=MagicMock(content="Test response", role="assistant", tool_calls=None),
                    finish_reason="stop",
                )
            ]
            mock_response.model = "fake-model"
            mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
            mock_chat_completions.return_value = mock_response

        result = await openai_responses_impl.create_openai_response(
            **{
                "input": "Test message",
                "model": "fake-model",
                "stream": stream,
                "store": store,
                param_name: param_value,
            }
        )
        if stream:
            chunks = [chunk async for chunk in result]
            created_event = chunks[0]
            assert created_event.type == "response.created"
            assert getattr(created_event.response, param_name, None) == response_expected_value, (
                f"Expected created {param_name}={response_expected_value}, got {getattr(created_event.response, param_name, None)}"
            )
            completed_event = chunks[-1]
            assert completed_event.type == "response.completed"
            assert getattr(completed_event.response, param_name, None) == stored_expected_value, (
                f"Expected completed {param_name}={stored_expected_value}, got {getattr(completed_event.response, param_name, None)}"
            )

        mock_chat_completions.assert_called_once()
        call_kwargs = mock_chat_completions.call_args[1]

        assert backend_param_name in call_kwargs, f"{backend_param_name} not found in backend call"
        assert call_kwargs[backend_param_name] == backend_expected_value, (
            f"Expected {backend_param_name}={backend_expected_value}, got {call_kwargs[backend_param_name]}"
        )

        if store:
            mock_responses_store.upsert_response_object.assert_called()
            stored_response = mock_responses_store.upsert_response_object.call_args.kwargs["response_object"]
            assert getattr(stored_response, param_name, None) == stored_expected_value, (
                f"Expected stored {param_name}={stored_expected_value}, got {getattr(stored_response, param_name, None)}"
            )
        else:
            mock_responses_store.upsert_response_object.assert_not_called()


async def test_create_openai_response_with_truncation_disabled_streaming(
    openai_responses_impl, mock_inference_api, mock_responses_store
):
    """Test that truncation='disabled' is properly handled in streaming responses."""
    input_text = "Explain machine learning comprehensively."
    model = "meta-llama/Llama-3.1-8B-Instruct"

    mock_inference_api.openai_chat_completion.return_value = fake_stream()

    # Execute
    result = await openai_responses_impl.create_openai_response(
        input=input_text,
        model=model,
        truncation=ResponseTruncation.disabled,
        stream=True,
        store=True,
    )

    # Collect all chunks
    chunks = [chunk async for chunk in result]

    # Verify truncation is in the created event
    created_event = chunks[0]
    assert created_event.type == "response.created"
    assert created_event.response.truncation == ResponseTruncation.disabled

    # Verify truncation is in the completed event
    completed_event = chunks[-1]
    assert completed_event.type == "response.completed"
    assert completed_event.response.truncation == ResponseTruncation.disabled

    mock_inference_api.openai_chat_completion.assert_called()

    # Verify the truncation was stored
    mock_responses_store.upsert_response_object.assert_called()
    store_call_args = mock_responses_store.upsert_response_object.call_args
    stored_response = store_call_args.kwargs["response_object"]
    assert stored_response.truncation == ResponseTruncation.disabled


async def test_create_openai_response_with_truncation_auto_streaming(
    openai_responses_impl, mock_inference_api, mock_responses_store
):
    """Test that truncation='auto' raises an error since it's not yet supported."""
    input_text = "Tell me about quantum computing."
    model = "meta-llama/Llama-3.1-8B-Instruct"

    mock_inference_api.openai_chat_completion.return_value = fake_stream()

    # Execute
    result = await openai_responses_impl.create_openai_response(
        input=input_text,
        model=model,
        truncation=ResponseTruncation.auto,
        stream=True,
        store=True,
    )

    # Collect all chunks
    chunks = [chunk async for chunk in result]

    # Verify truncation is in the created event
    created_event = chunks[0]
    assert created_event.type == "response.created"
    assert created_event.response.truncation == ResponseTruncation.auto

    # Verify the response failed due to unsupported truncation mode
    failed_event = chunks[-1]
    assert failed_event.type == "response.failed"
    assert failed_event.response.truncation == ResponseTruncation.auto
    assert failed_event.response.error is not None
    assert failed_event.response.error.code == "server_error"
    assert "Truncation mode 'auto' is not supported" in failed_event.response.error.message

    # Inference API should not be called since error occurs before inference
    mock_inference_api.openai_chat_completion.assert_not_called()

    # Verify the failed response was stored
    mock_responses_store.upsert_response_object.assert_called()
    store_call_args = mock_responses_store.upsert_response_object.call_args
    stored_response = store_call_args.kwargs["response_object"]
    assert stored_response.truncation == ResponseTruncation.auto
    assert stored_response.status == "failed"


async def test_create_openai_response_with_prompt_cache_key_and_previous_response(
    openai_responses_impl, mock_responses_store, mock_inference_api
):
    """Test that prompt_cache_key works correctly with previous_response_id."""
    # Setup previous response
    previous_response = _OpenAIResponseObjectWithInputAndMessages(
        id="resp-prev-123",
        object="response",
        created_at=1234567890,
        model="meta-llama/Llama-3.1-8B-Instruct",
        status="completed",
        text=OpenAIResponseText(format=OpenAIResponseTextFormat(type="text")),
        input=[OpenAIResponseMessage(id="msg-1", role="user", content="First question")],
        output=[OpenAIResponseMessage(id="msg-2", role="assistant", content="First answer")],
        messages=[
            OpenAIUserMessageParam(content="First question"),
            OpenAIAssistantMessageParam(content="First answer"),
        ],
        prompt_cache_key="conversation-cache-001",
        store=True,
    )

    mock_responses_store.get_response_object.return_value = previous_response
    mock_inference_api.openai_chat_completion.return_value = fake_stream()

    # Create a new response with the same cache key
    result = await openai_responses_impl.create_openai_response(
        input="Second question",
        model="meta-llama/Llama-3.1-8B-Instruct",
        previous_response_id="resp-prev-123",
        prompt_cache_key="conversation-cache-001",
        store=True,
    )

    # Verify cache key is preserved
    assert result.prompt_cache_key == "conversation-cache-001"
    assert result.status == "completed"

    # Verify the cache key was stored
    mock_responses_store.upsert_response_object.assert_called()
    store_call_args = mock_responses_store.upsert_response_object.call_args
    stored_response = store_call_args.kwargs["response_object"]
    assert stored_response.prompt_cache_key == "conversation-cache-001"


async def test_create_openai_response_with_service_tier(openai_responses_impl, mock_inference_api):
    """Test creating an OpenAI response with service_tier parameter."""
    # Setup
    input_text = "What is the capital of France?"
    model = "meta-llama/Llama-3.1-8B-Instruct"
    service_tier = ServiceTier.flex

    # Load the chat completion fixture
    mock_inference_api.openai_chat_completion.return_value = fake_stream()

    # Execute - non-streaming to get final response directly
    result = await openai_responses_impl.create_openai_response(
        input=input_text,
        model=model,
        service_tier=service_tier,
        stream=False,
    )

    # Verify service_tier is preserved in the response (as string)
    assert result.service_tier == ServiceTier.default.value
    assert result.status == "completed"

    # Verify inference call received service_tier
    mock_inference_api.openai_chat_completion.assert_called_once()
    params = mock_inference_api.openai_chat_completion.call_args.args[0]
    assert params.service_tier == service_tier


async def test_create_openai_response_service_tier_auto_transformation(openai_responses_impl, mock_inference_api):
    """Test that service_tier 'auto' is transformed to actual tier from provider response."""
    # Setup
    input_text = "Hello"
    model = "meta-llama/Llama-3.1-8B-Instruct"

    # Mock a response that returns actual service tier when "auto" was requested
    async def fake_stream_with_service_tier():
        yield ChatCompletionChunk(
            id="chatcmpl-123",
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(content="Hi there!", role="assistant"),
                    finish_reason="stop",
                )
            ],
            created=1234567890,
            model=model,
            object="chat.completion.chunk",
            service_tier="default",  # Provider returns actual tier used
        )

    mock_inference_api.openai_chat_completion.return_value = fake_stream_with_service_tier()

    # Execute with "auto" service tier
    result = await openai_responses_impl.create_openai_response(
        input=input_text,
        model=model,
        service_tier=ServiceTier.auto,
        stream=False,
    )

    # Verify the response has the actual tier from provider, not "auto"
    assert result.service_tier == "default", "service_tier should be transformed from 'auto' to actual tier"
    assert result.service_tier != ServiceTier.auto.value, "service_tier should not remain as 'auto'"
    assert result.status == "completed"

    # Verify inference was called with "auto"
    mock_inference_api.openai_chat_completion.assert_called_once()
    params = mock_inference_api.openai_chat_completion.call_args.args[0]
    assert params.service_tier == "auto"


async def test_create_openai_response_service_tier_propagation_streaming(openai_responses_impl, mock_inference_api):
    """Test that service_tier from chat completion is propagated to response object in streaming mode."""
    # Setup
    input_text = "Tell me about AI"
    model = "meta-llama/Llama-3.1-8B-Instruct"

    # Mock streaming response with service_tier
    async def fake_stream_with_service_tier():
        yield ChatCompletionChunk(
            id="chatcmpl-456",
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(content="AI is", role="assistant"),
                    finish_reason=None,
                )
            ],
            created=1234567890,
            model=model,
            object="chat.completion.chunk",
            service_tier="priority",  # First chunk with service_tier
        )
        yield ChatCompletionChunk(
            id="chatcmpl-456",
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(content=" amazing!"),
                    finish_reason="stop",
                )
            ],
            created=1234567890,
            model=model,
            object="chat.completion.chunk",
        )

    mock_inference_api.openai_chat_completion.return_value = fake_stream_with_service_tier()

    # Execute with "auto" but provider returns "priority"
    result = await openai_responses_impl.create_openai_response(
        input=input_text,
        model=model,
        service_tier=ServiceTier.auto,
        stream=True,
    )

    # Collect all chunks
    chunks = [chunk async for chunk in result]
    # Verify service_tier is propagated to all events
    created_event = chunks[0]
    assert created_event.type == "response.created"
    # Initially should have "auto" value
    assert created_event.response.service_tier == "auto"

    # Check final response has the actual tier from provider
    completed_event = chunks[-1]
    assert completed_event.type == "response.completed"
    assert completed_event.response.service_tier == "priority", "Final response should have actual tier from provider"


async def test_create_openai_response_with_top_logprobs_boundary_values(
    openai_responses_impl, mock_inference_api, mock_responses_store
):
    """Test that top_logprobs works with boundary values (0 and 20)."""
    input_text = "Test message"
    model = "meta-llama/Llama-3.1-8B-Instruct"

    # Test with minimum value (0)
    mock_inference_api.openai_chat_completion.return_value = fake_stream()
    result = await openai_responses_impl.create_openai_response(
        input=input_text,
        model=model,
        top_logprobs=0,
        stream=False,
        store=True,
    )
    assert result.top_logprobs == 0

    # Test with maximum value (20)
    mock_inference_api.openai_chat_completion.return_value = fake_stream()
    result = await openai_responses_impl.create_openai_response(
        input=input_text,
        model=model,
        top_logprobs=20,
        stream=False,
        store=True,
    )
    assert result.top_logprobs == 20


async def test_create_openai_response_with_frequency_penalty_default(openai_responses_impl, mock_inference_api):
    """Test that frequency_penalty defaults to 0.0 when not provided."""
    input_text = "Hello"
    model = "meta-llama/Llama-3.1-8B-Instruct"

    mock_inference_api.openai_chat_completion.return_value = fake_stream()

    # Execute without frequency_penalty
    result = await openai_responses_impl.create_openai_response(
        input=input_text,
        model=model,
        stream=False,
    )

    # Verify response has 0.0 for frequency_penalty (non-null default for OpenResponses conformance)
    assert result.frequency_penalty == 0.0

    # Verify inference API was called with None
    mock_inference_api.openai_chat_completion.assert_called()
    call_args = mock_inference_api.openai_chat_completion.call_args
    params = call_args.args[0]
    assert params.frequency_penalty is None


async def test_create_openai_response_with_presence_penalty_default(openai_responses_impl, mock_inference_api):
    """Test that presence_penalty defaults to 0.0 when not provided."""
    input_text = "Hi"
    model = "meta-llama/Llama-3.1-8B-Instruct"

    mock_inference_api.openai_chat_completion.return_value = fake_stream()

    # Execute without presence_penalty
    result = await openai_responses_impl.create_openai_response(
        input=input_text,
        model=model,
        stream=False,
    )

    # Verify presence_penalty is 0.0 (non-null default for OpenResponses conformance)
    assert result.presence_penalty == 0.0
    assert result.status == "completed"

    # Verify the inference API was called with presence_penalty=None
    mock_inference_api.openai_chat_completion.assert_called()
    call_args = mock_inference_api.openai_chat_completion.call_args
    params = call_args.args[0]
    assert params.presence_penalty is None


async def test_hallucinated_tool_call_does_not_cause_500(openai_responses_impl, mock_inference_api):
    """Regression test: a hallucinated tool name should not produce a 500 (InternalServerError).

    When the LLM calls a tool name that is not in the registered tools list the server
    was raising ValueError from _coordinate_tool_execution which then propagated as an
    InternalServerError (HTTP 500). The correct behaviour is to surface the unknown call
    as a regular function-tool-call output so the client can respond, exactly as OpenAI
    does for any function tool call.
    """
    input_text = "What is the capital of Ireland?"
    model = "meta-llama/Llama-3.1-8B-Instruct"

    async def fake_stream_hallucinated_tool():
        # The LLM calls "lookup_capital_city" which is NOT in the registered tools list.
        yield ChatCompletionChunk(
            id="hallucinated-123",
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                id="tc_hall_123",
                                function=ChoiceDeltaToolCallFunction(
                                    name="lookup_capital_city",
                                    arguments='{"country": "Ireland"}',
                                ),
                                type="function",
                            )
                        ]
                    ),
                ),
            ],
            created=1,
            model=model,
            object="chat.completion.chunk",
        )

    mock_inference_api.openai_chat_completion.return_value = fake_stream_hallucinated_tool()

    # The only registered tool is "get_weather".  The LLM hallucinated "lookup_capital_city".
    # The response should complete without raising InternalServerError, and the hallucinated
    # call should appear in the output as a function_call item so the client can handle it.
    result = await openai_responses_impl.create_openai_response(
        input=input_text,
        model=model,
        tools=[
            OpenAIResponseInputToolFunction(
                name="get_weather",
                description="Get current temperature for a given location.",
                parameters={
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            )
        ],
    )

    assert result is not None
    assert result.status == "completed"
    assert len(result.output) == 1
    assert result.output[0].type == "function_call"
    assert result.output[0].name == "lookup_capital_city"


async def test_create_openai_response_with_stream_options_merges_with_default(
    openai_responses_impl, mock_inference_api
):
    """Test that stream_options merges with default include_usage."""
    input_text = "Test stream options"
    model = "meta-llama/Llama-3.1-8B-Instruct"
    stream_options = ResponseStreamOptions(include_obfuscation=False)

    mock_inference_api.openai_chat_completion.return_value = fake_stream()

    # Execute
    result = await openai_responses_impl.create_openai_response(
        input=input_text,
        model=model,
        stream_options=stream_options,
        stream=True,
    )

    # Collect chunks (consume the async iterator)
    _ = [chunk async for chunk in result]

    # Verify the stream_options was merged properly
    mock_inference_api.openai_chat_completion.assert_called()
    call_args = mock_inference_api.openai_chat_completion.call_args
    params = call_args.args[0]
    assert params.stream_options is not None
    # Should have both default include_usage and user's option
    assert params.stream_options["include_usage"] is True
    assert params.stream_options["include_obfuscation"] is False


async def test_create_openai_response_with_empty_stream_options(openai_responses_impl, mock_inference_api):
    """Test that default stream_options still merges with default include_usage."""
    input_text = "Test empty options"
    model = "meta-llama/Llama-3.1-8B-Instruct"
    stream_options = ResponseStreamOptions()  # Uses default include_obfuscation=True

    mock_inference_api.openai_chat_completion.return_value = fake_stream()

    # Execute
    result = await openai_responses_impl.create_openai_response(
        input=input_text,
        model=model,
        stream_options=stream_options,
        stream=True,
    )

    # Collect chunks (consume the async iterator)
    _ = [chunk async for chunk in result]

    # Verify the stream_options has both defaults
    mock_inference_api.openai_chat_completion.assert_called()
    call_args = mock_inference_api.openai_chat_completion.call_args
    params = call_args.args[0]
    assert params.stream_options is not None
    assert params.stream_options["include_usage"] is True
    assert params.stream_options["include_obfuscation"] is True
