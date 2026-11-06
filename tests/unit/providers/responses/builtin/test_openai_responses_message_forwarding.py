# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Tests for message forwarding to openai_chat_completion across all input types."""

from ogx.providers.utils.responses.responses_store import (
    _OpenAIResponseObjectWithInputAndMessages,
)
from ogx_api.inference import (
    OpenAIAssistantMessageParam,
    OpenAIChatCompletionContentPartImageParam,
    OpenAIUserMessageParam,
)
from ogx_api.openai_responses import (
    OpenAIResponseCompaction,
    OpenAIResponseInputFunctionToolCallOutput,
    OpenAIResponseInputMessageContentImage,
    OpenAIResponseMessage,
    OpenAIResponseOutputMessageContentOutputText,
    OpenAIResponseOutputMessageFunctionToolCall,
    OpenAIResponseOutputMessageMCPCall,
    OpenAIResponseOutputMessageReasoningContent,
    OpenAIResponseOutputMessageReasoningItem,
    OpenAIResponseOutputMessageReasoningSummary,
)
from ogx_api.responses.models import CreateResponseRequest, ResponseTruncation
from tests.unit.providers.responses.builtin.test_openai_responses_helpers import fake_stream


async def test_function_call_and_output_forwarded_to_inference(
    openai_responses_impl,
    mock_inference_api,
    mock_responses_store,
):
    """Test that function tool call and its output are correctly converted and
    forwarded to openai_chat_completion as assistant message with tool_calls and
    a tool result message."""
    # Setup: function call + its output
    input_items = [
        OpenAIResponseMessage(role="user", content="What's the weather in Tokyo?", name=None),
        OpenAIResponseOutputMessageFunctionToolCall(
            call_id="call_123",
            name="get_weather",
            arguments='{"city": "Tokyo"}',
            type="function_call",
        ),
        OpenAIResponseInputFunctionToolCallOutput(
            call_id="call_123",
            output="25°C and sunny",
            type="function_call_output",
        ),
    ]
    model = "meta-llama/Llama-3.1-8B-Instruct"

    mock_inference_api.openai_chat_completion.return_value = fake_stream()

    await openai_responses_impl.create_openai_response(
        CreateResponseRequest(input=input_items, model=model, temperature=0.1)
    )

    params = mock_inference_api.openai_chat_completion.call_args[0][0]
    messages = params.messages

    # Should be: user, assistant (with tool_call), tool result
    assert len(messages) == 3

    # User message
    assert messages[0].role == "user"
    assert messages[0].content == "What's the weather in Tokyo?"

    # Assistant message with tool_calls
    assert messages[1].role == "assistant"
    assert len(messages[1].tool_calls) == 1
    assert messages[1].tool_calls[0].id == "call_123"
    assert messages[1].tool_calls[0].function.name == "get_weather"
    assert messages[1].tool_calls[0].function.arguments == '{"city": "Tokyo"}'

    # Tool result message
    assert messages[2].role == "tool"
    assert messages[2].tool_call_id == "call_123"


async def test_mcp_call_forwarded_to_inference(
    openai_responses_impl,
    mock_inference_api,
    mock_responses_store,
):
    """Test that MCP call is correctly converted and forwarded to
    openai_chat_completion as assistant message with tool_calls and a tool
    result message."""
    # Setup: MCP call with output
    input_items = [
        OpenAIResponseMessage(role="user", content="What's the temp in Tokyo?", name=None),
        OpenAIResponseOutputMessageMCPCall(
            id="mcp_789",
            type="mcp_call",
            server_label="weather_server",
            name="get_temperature",
            arguments='{"location": "Tokyo"}',
            output="25C",
        ),
    ]
    model = "meta-llama/Llama-3.1-8B-Instruct"

    mock_inference_api.openai_chat_completion.return_value = fake_stream()

    await openai_responses_impl.create_openai_response(
        CreateResponseRequest(input=input_items, model=model, temperature=0.1)
    )

    params = mock_inference_api.openai_chat_completion.call_args[0][0]
    messages = params.messages

    # Should be: user, assistant (with tool_call), tool result
    assert len(messages) == 3

    # User message
    assert messages[0].role == "user"
    assert messages[0].content == "What's the temp in Tokyo?"

    # Assistant message with tool_calls
    assert messages[1].role == "assistant"
    assert len(messages[1].tool_calls) == 1
    assert messages[1].tool_calls[0].id == "mcp_789"
    assert messages[1].tool_calls[0].function.name == "get_temperature"
    assert messages[1].tool_calls[0].function.arguments == '{"location": "Tokyo"}'

    # Tool result message
    assert messages[2].role == "tool"
    assert messages[2].tool_call_id == "mcp_789"


async def test_reasoning_and_assistant_forwarded_to_inference(
    openai_responses_impl,
    mock_inference_api,
    mock_responses_store,
):
    """Test that reasoning item followed by assistant message is correctly
    converted to AssistantMessageWithReasoning and forwarded to inference."""
    # Reasoning + assistant message

    # Setup: reasoning + assistant message
    input_items = [
        OpenAIResponseMessage(role="user", content="What's 2+2?", name=None),
        OpenAIResponseOutputMessageReasoningItem(
            id="reason_1",
            type="reasoning",
            summary=[OpenAIResponseOutputMessageReasoningSummary(text="The model reasoned step by step.")],
            content=[
                OpenAIResponseOutputMessageReasoningContent(text="Let me think about this."),
                OpenAIResponseOutputMessageReasoningContent(text="2 + 2 = 4"),
            ],
        ),
        OpenAIResponseMessage(
            role="assistant",
            content=[OpenAIResponseOutputMessageContentOutputText(text="4")],
            name=None,
        ),
    ]
    model = "meta-llama/Llama-3.1-8B-Instruct"

    mock_inference_api.openai_chat_completion.return_value = fake_stream()

    await openai_responses_impl.create_openai_response(
        CreateResponseRequest(input=input_items, model=model, temperature=0.1)
    )

    params = mock_inference_api.openai_chat_completion.call_args[0][0]
    messages = params.messages

    # Should be: user, assistant (with reasoning_content)
    assert len(messages) == 2

    # User message
    assert messages[0].role == "user"
    assert messages[0].content == "What's 2+2?"

    # Assistant message with reasoning_content
    assert messages[1].role == "assistant"
    assert messages[1].content[0].text == "4"
    assert messages[1].reasoning_content == "Let me think about this. 2 + 2 = 4"


async def test_image_content_forwarded_to_inference(
    openai_responses_impl,
    mock_inference_api,
    mock_responses_store,
):
    """Test that user message with image content is correctly converted and
    forwarded to openai_chat_completion as a multi-part content with image."""
    input_item = OpenAIResponseMessage(
        role="user",
        content=[
            {"type": "input_text", "text": "What is in this image?"},
            OpenAIResponseInputMessageContentImage(
                type="input_image",
                image_url="https://example.com/photo.jpg",
                detail="low",
            ),
        ],
        name=None,
    )
    model = "meta-llama/Llama-3.1-8B-Instruct"

    mock_inference_api.openai_chat_completion.return_value = fake_stream()

    await openai_responses_impl.create_openai_response(
        CreateResponseRequest(input=[input_item], model=model, temperature=0.1)
    )

    params = mock_inference_api.openai_chat_completion.call_args[0][0]
    messages = params.messages

    assert len(messages) == 1
    assert messages[0].role == "user"
    assert isinstance(messages[0].content, list)
    assert len(messages[0].content) == 2
    assert messages[0].content[0].type == "text"
    assert messages[0].content[0].text == "What is in this image?"
    assert isinstance(messages[0].content[1], OpenAIChatCompletionContentPartImageParam)
    assert messages[0].content[1].image_url.url == "https://example.com/photo.jpg"
    assert messages[0].content[1].image_url.detail == "low"


async def test_compaction_forwarded_to_inference(
    openai_responses_impl,
    mock_inference_api,
    mock_responses_store,
):
    """Test that compaction item is correctly converted to an assistant message
    and forwarded to openai_chat_completion."""
    # Setup: compaction used as previous context
    input_items = [
        OpenAIResponseCompaction(
            id="compact_1",
            type="compaction",
            encrypted_content="Compact summary of prior conversation.",
            output=[],
        ),
        OpenAIResponseMessage(role="user", content="Now answer this: 42?", name=None),
    ]
    model = "meta-llama/Llama-3.1-8B-Instruct"

    mock_inference_api.openai_chat_completion.return_value = fake_stream()

    await openai_responses_impl.create_openai_response(
        CreateResponseRequest(input=input_items, model=model, temperature=0.1)
    )

    params = mock_inference_api.openai_chat_completion.call_args[0][0]
    messages = params.messages

    # Should be: assistant (compaction summary), user
    assert len(messages) == 2

    # Compaction converted to assistant message
    assert messages[0].role == "assistant"
    assert messages[0].content == "Compact summary of prior conversation."

    # User message
    assert messages[1].role == "user"
    assert messages[1].content == "Now answer this: 42?"


async def test_function_call_with_multiple_outputs_forwarded_to_inference(
    openai_responses_impl,
    mock_inference_api,
    mock_responses_store,
):
    """Test that multiple function calls and their outputs are correctly
    ordered and forwarded to openai_chat_completion."""

    input_items = [
        OpenAIResponseMessage(role="user", content="What's the weather in Tokyo and Paris?", name=None),
        OpenAIResponseOutputMessageFunctionToolCall(
            call_id="call_tokyo",
            name="get_weather",
            arguments='{"city": "Tokyo"}',
            type="function_call",
        ),
        OpenAIResponseOutputMessageFunctionToolCall(
            call_id="call_paris",
            name="get_weather",
            arguments='{"city": "Paris"}',
            type="function_call",
        ),
        OpenAIResponseInputFunctionToolCallOutput(
            call_id="call_tokyo",
            output="25C",
            type="function_call_output",
        ),
        OpenAIResponseInputFunctionToolCallOutput(
            call_id="call_paris",
            output="18C",
            type="function_call_output",
        ),
    ]
    model = "meta-llama/Llama-3.1-8B-Instruct"

    mock_inference_api.openai_chat_completion.return_value = fake_stream()

    await openai_responses_impl.create_openai_response(
        CreateResponseRequest(input=input_items, model=model, temperature=0.1)
    )

    params = mock_inference_api.openai_chat_completion.call_args[0][0]
    messages = params.messages

    # Each function call creates its own assistant message, followed by its tool result
    # user, assistant (tok), tool (tok), assistant (paris), tool (paris)
    assert len(messages) == 5

    assert messages[0].role == "user"
    assert messages[0].content == "What's the weather in Tokyo and Paris?"

    # First function call + output
    assert messages[1].role == "assistant"
    assert len(messages[1].tool_calls) == 1
    assert messages[1].tool_calls[0].id == "call_tokyo"
    assert messages[1].tool_calls[0].function.name == "get_weather"
    assert messages[2].role == "tool"
    assert messages[2].tool_call_id == "call_tokyo"

    # Second function call + output
    assert messages[3].role == "assistant"
    assert len(messages[3].tool_calls) == 1
    assert messages[3].tool_calls[0].id == "call_paris"
    assert messages[3].tool_calls[0].function.name == "get_weather"
    assert messages[4].role == "tool"
    assert messages[4].tool_call_id == "call_paris"


async def test_truncation_retry_drops_oldest_turn(
    openai_responses_impl,
    mock_inference_api,
    mock_responses_store,
):
    """When truncation='auto' and the provider raises a context-length error,
    the responses layer drops the oldest turn and retries inference.  Verify
    both calls to openai_chat_completion carry the expected messages."""
    prev_id = "resp-prev"
    previous_response = _OpenAIResponseObjectWithInputAndMessages(
        id=prev_id,
        object="response",
        created_at=1234567890,
        model="meta-llama/Llama-3.1-8B-Instruct",
        status="completed",
        input=[OpenAIResponseMessage(id="msg-1", role="user", content="First question")],
        output=[OpenAIResponseMessage(id="msg-2", role="assistant", content="First answer")],
        messages=[
            OpenAIUserMessageParam(role="user", content="First question"),
            OpenAIAssistantMessageParam(role="assistant", content="First answer"),
            OpenAIUserMessageParam(role="user", content="Second question"),
            OpenAIAssistantMessageParam(role="assistant", content="Second answer"),
        ],
        store=True,
    )
    mock_responses_store.get_response_object.return_value = previous_response

    error_body = {
        "error": {
            "code": 400,
            "message": "This model's maximum context length is 128000 tokens. However your input contains 200000 tokens.",
            "type": "BadRequestError",
            "param": "input_text",
        }
    }

    class ContextLengthError(Exception):
        status_code = 400
        body = error_body

    call_count = 0

    async def mock_chat_completion(params):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ContextLengthError("context length exceeded")
        return fake_stream()

    mock_inference_api.openai_chat_completion.side_effect = mock_chat_completion

    result = await openai_responses_impl.create_openai_response(
        CreateResponseRequest(
            input="Third question",
            model="meta-llama/Llama-3.1-8B-Instruct",
            previous_response_id=prev_id,
            truncation=ResponseTruncation.auto,
            stream=True,
            store=True,
        )
    )

    # Consume the stream so the request completes
    chunks = [chunk async for chunk in result]

    # Verify response completed (not failed)
    assert chunks[-1].type == "response.completed"
    assert chunks[-1].response.error is None

    # Verify inference was called twice
    assert call_count == 2

    calls = mock_inference_api.openai_chat_completion.call_args_list

    # First call: full history (4 from previous response + 1 new user message)
    first_messages = calls[0].args[0].messages
    assert len(first_messages) == 5
    assert first_messages[0].content == "First question"
    assert first_messages[1].content == "First answer"
    assert first_messages[2].content == "Second question"
    assert first_messages[3].content == "Second answer"
    assert first_messages[4].content == "Third question"

    # Retry: oldest turn (user "First question" + assistant "First answer") dropped
    retry_messages = calls[1].args[0].messages
    assert len(retry_messages) == 3
    assert retry_messages[0].content == "Second question"
    assert retry_messages[1].content == "Second answer"
    assert retry_messages[2].content == "Third question"
