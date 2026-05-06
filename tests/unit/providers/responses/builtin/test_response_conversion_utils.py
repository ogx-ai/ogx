# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from unittest.mock import AsyncMock, Mock

import pytest

from ogx.providers.inline.responses.builtin.responses.utils import (
    _extract_citations_from_text,
    convert_chat_choice_to_response_message,
    convert_response_content_to_chat_content,
    convert_response_input_to_chat_messages,
    convert_response_text_to_chat_response_format,
    get_message_type_by_role,
    is_function_tool_call,
)
from ogx_api import RetrieveFileContentRequest, RetrieveFileRequest
from ogx_api.inference import (
    OpenAIAssistantMessageParam,
    OpenAIChatCompletionContentPartImageParam,
    OpenAIChatCompletionContentPartTextParam,
    OpenAIChatCompletionResponseMessage,
    OpenAIChatCompletionToolCall,
    OpenAIChatCompletionToolCallFunction,
    OpenAIChoice,
    OpenAIDeveloperMessageParam,
    OpenAIResponseFormatJSONObject,
    OpenAIResponseFormatJSONSchema,
    OpenAIResponseFormatText,
    OpenAISystemMessageParam,
    OpenAIToolMessageParam,
    OpenAIUserMessageParam,
)
from ogx_api.openai_responses import (
    MCPListToolsTool,
    OpenAIResponseAnnotationFileCitation,
    OpenAIResponseInputFunctionToolCallOutput,
    OpenAIResponseInputMessageContentFile,
    OpenAIResponseInputMessageContentImage,
    OpenAIResponseInputMessageContentText,
    OpenAIResponseInputToolFunction,
    OpenAIResponseInputToolWebSearch,
    OpenAIResponseMessage,
    OpenAIResponseOutputMessageContentOutputText,
    OpenAIResponseOutputMessageFunctionToolCall,
    OpenAIResponseOutputMessageMCPCall,
    OpenAIResponseOutputMessageMCPListTools,
    OpenAIResponseOutputMessageReasoningContent,
    OpenAIResponseOutputMessageReasoningItem,
    OpenAIResponseText,
    OpenAIResponseTextFormat,
)


@pytest.fixture
def mock_files_api():
    """Mock files API for testing."""
    return AsyncMock()


class TestConvertChatChoiceToResponseMessage:
    async def test_convert_string_content(self):
        choice = OpenAIChoice(
            message=OpenAIChatCompletionResponseMessage(content="Test message"),
            finish_reason="stop",
            index=0,
        )

        result = await convert_chat_choice_to_response_message(choice)

        assert result.role == "assistant"
        assert result.status == "completed"
        assert len(result.content) == 1
        assert isinstance(result.content[0], OpenAIResponseOutputMessageContentOutputText)
        assert result.content[0].text == "Test message"

    async def test_convert_none_content(self):
        choice = OpenAIChoice(
            message=OpenAIChatCompletionResponseMessage(
                content=None,
            ),
            finish_reason="stop",
            index=0,
        )

        result = await convert_chat_choice_to_response_message(choice)

        assert result.role == "assistant"
        assert result.status == "completed"
        assert len(result.content) == 1
        assert isinstance(result.content[0], OpenAIResponseOutputMessageContentOutputText)
        assert result.content[0].text == ""


class TestConvertResponseContentToChatContent:
    async def test_convert_string_content(self, mock_files_api):
        result = await convert_response_content_to_chat_content("Simple string", mock_files_api)
        assert result == "Simple string"

    async def test_convert_text_content_parts(self, mock_files_api):
        content = [
            OpenAIResponseInputMessageContentText(text="First part"),
            OpenAIResponseOutputMessageContentOutputText(text="Second part"),
        ]

        result = await convert_response_content_to_chat_content(content, mock_files_api)

        assert len(result) == 2
        assert isinstance(result[0], OpenAIChatCompletionContentPartTextParam)
        assert result[0].text == "First part"
        assert isinstance(result[1], OpenAIChatCompletionContentPartTextParam)
        assert result[1].text == "Second part"

    async def test_convert_image_content(self, mock_files_api):
        content = [OpenAIResponseInputMessageContentImage(image_url="https://example.com/image.jpg", detail="high")]

        result = await convert_response_content_to_chat_content(content, mock_files_api)

        assert len(result) == 1
        assert isinstance(result[0], OpenAIChatCompletionContentPartImageParam)
        assert result[0].image_url.url == "https://example.com/image.jpg"
        assert result[0].image_url.detail == "high"

    async def test_convert_image_content_with_file_id_calls_retrieve_with_request_objects(self, mock_files_api):
        mock_files_api.openai_retrieve_file.return_value = Mock(filename="photo.png")
        mock_files_api.openai_retrieve_file_content.return_value = Mock(body=b"\x89PNG\r\n")

        content = [OpenAIResponseInputMessageContentImage(file_id="file-abc123")]
        await convert_response_content_to_chat_content(content, mock_files_api)

        mock_files_api.openai_retrieve_file.assert_called_once_with(RetrieveFileRequest(file_id="file-abc123"))
        mock_files_api.openai_retrieve_file_content.assert_called_once_with(
            RetrieveFileContentRequest(file_id="file-abc123")
        )

    async def test_convert_file_content_with_file_id_calls_retrieve_with_request_objects(self, mock_files_api):
        mock_files_api.openai_retrieve_file.return_value = Mock(filename="report.txt")
        mock_files_api.openai_retrieve_file_content.return_value = Mock(body=b"report content")

        content = [OpenAIResponseInputMessageContentFile(file_id="file-def456")]
        await convert_response_content_to_chat_content(content, mock_files_api)

        mock_files_api.openai_retrieve_file.assert_called_once_with(RetrieveFileRequest(file_id="file-def456"))
        mock_files_api.openai_retrieve_file_content.assert_called_once_with(
            RetrieveFileContentRequest(file_id="file-def456")
        )


class TestConvertResponseInputToChatMessages:
    async def test_convert_string_input(self):
        result = await convert_response_input_to_chat_messages("User message")

        assert len(result) == 1
        assert isinstance(result[0], OpenAIUserMessageParam)
        assert result[0].content == "User message"

    async def test_convert_function_tool_call_output(self):
        input_items = [
            OpenAIResponseOutputMessageFunctionToolCall(
                call_id="call_123",
                name="test_function",
                arguments='{"param": "value"}',
            ),
            OpenAIResponseInputFunctionToolCallOutput(
                output="Tool output",
                call_id="call_123",
            ),
        ]

        result = await convert_response_input_to_chat_messages(input_items)

        assert len(result) == 2
        assert isinstance(result[0], OpenAIAssistantMessageParam)
        assert result[0].tool_calls[0].id == "call_123"
        assert result[0].tool_calls[0].function.name == "test_function"
        assert result[0].tool_calls[0].function.arguments == '{"param": "value"}'
        assert isinstance(result[1], OpenAIToolMessageParam)
        assert result[1].content == "Tool output"
        assert result[1].tool_call_id == "call_123"

    async def test_convert_function_tool_call_output_with_list_content(self):
        # The OpenAI Responses API spec uses "input_text" as the type discriminator
        # for text content blocks in function_call_output.
        input_items = [
            OpenAIResponseOutputMessageFunctionToolCall(
                call_id="call_123",
                name="search_parks",
                arguments='{"state_code": "RI"}',
            ),
            OpenAIResponseInputFunctionToolCallOutput(
                output=[{"type": "input_text", "text": '{"parks": ["Park A", "Park B"]}'}],
                call_id="call_123",
            ),
        ]

        result = await convert_response_input_to_chat_messages(input_items)

        assert len(result) == 2
        assert isinstance(result[1], OpenAIToolMessageParam)
        assert result[1].content == [OpenAIChatCompletionContentPartTextParam(text='{"parks": ["Park A", "Park B"]}')]
        assert result[1].tool_call_id == "call_123"

    async def test_convert_function_tool_call_output_with_multi_block_list_content(self):
        # Multiple text blocks should each become a separate content part.
        input_items = [
            OpenAIResponseOutputMessageFunctionToolCall(
                call_id="call_456",
                name="search",
                arguments="{}",
            ),
            OpenAIResponseInputFunctionToolCallOutput(
                output=[{"type": "input_text", "text": "first"}, {"type": "input_text", "text": "second"}],
                call_id="call_456",
            ),
        ]

        result = await convert_response_input_to_chat_messages(input_items)

        assert len(result) == 2
        assert isinstance(result[1], OpenAIToolMessageParam)
        assert result[1].content == [
            OpenAIChatCompletionContentPartTextParam(text="first"),
            OpenAIChatCompletionContentPartTextParam(text="second"),
        ]

    async def test_convert_function_tool_call_output_with_image_content(self):
        # Image parts in function_call_output can't go in OpenAIToolMessageParam (text-only).
        # They should be split: tool message gets placeholder text, image goes in a user message.
        input_items = [
            OpenAIResponseOutputMessageFunctionToolCall(
                call_id="call_789",
                name="capture_screenshot",
                arguments='{"url": "https://example.com"}',
            ),
            OpenAIResponseInputFunctionToolCallOutput(
                output=[{"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgo="}],
                call_id="call_789",
            ),
        ]

        result = await convert_response_input_to_chat_messages(input_items)

        # Should produce 3 messages: assistant (tool_call), tool (text placeholder), user (image)
        assert len(result) == 3
        assert isinstance(result[0], OpenAIAssistantMessageParam)
        assert isinstance(result[1], OpenAIToolMessageParam)
        assert isinstance(result[2], OpenAIUserMessageParam)
        # Tool message should have placeholder text since there were no text parts
        assert result[1].content == [OpenAIChatCompletionContentPartTextParam(text="[image]")]
        # User message should carry the image
        assert isinstance(result[2].content, list)
        assert len(result[2].content) == 1
        assert isinstance(result[2].content[0], OpenAIChatCompletionContentPartImageParam)

    async def test_convert_function_tool_call_output_with_mixed_text_and_image(self):
        # When output has both text and image parts, text stays in tool message, image in user message.
        input_items = [
            OpenAIResponseOutputMessageFunctionToolCall(
                call_id="call_mix",
                name="analyze",
                arguments="{}",
            ),
            OpenAIResponseInputFunctionToolCallOutput(
                output=[
                    {"type": "input_text", "text": "Screenshot captured successfully"},
                    {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgo="},
                ],
                call_id="call_mix",
            ),
        ]

        result = await convert_response_input_to_chat_messages(input_items)

        assert len(result) == 3
        assert isinstance(result[0], OpenAIAssistantMessageParam)
        assert isinstance(result[1], OpenAIToolMessageParam)
        assert result[1].content == [OpenAIChatCompletionContentPartTextParam(text="Screenshot captured successfully")]
        assert isinstance(result[2], OpenAIUserMessageParam)

    async def test_convert_function_tool_call(self):
        input_items = [
            OpenAIResponseOutputMessageFunctionToolCall(
                call_id="call_456",
                name="test_function",
                arguments='{"param": "value"}',
            )
        ]

        result = await convert_response_input_to_chat_messages(input_items)

        assert len(result) == 1
        assert isinstance(result[0], OpenAIAssistantMessageParam)
        assert len(result[0].tool_calls) == 1
        assert result[0].tool_calls[0].id == "call_456"
        assert result[0].tool_calls[0].function.name == "test_function"
        assert result[0].tool_calls[0].function.arguments == '{"param": "value"}'

    async def test_convert_function_call_ordering(self):
        input_items = [
            OpenAIResponseOutputMessageFunctionToolCall(
                call_id="call_123",
                name="test_function_a",
                arguments='{"param": "value"}',
            ),
            OpenAIResponseOutputMessageFunctionToolCall(
                call_id="call_456",
                name="test_function_b",
                arguments='{"param": "value"}',
            ),
            OpenAIResponseInputFunctionToolCallOutput(
                output="AAA",
                call_id="call_123",
            ),
            OpenAIResponseInputFunctionToolCallOutput(
                output="BBB",
                call_id="call_456",
            ),
        ]

        result = await convert_response_input_to_chat_messages(input_items)
        assert len(result) == 4
        assert isinstance(result[0], OpenAIAssistantMessageParam)
        assert len(result[0].tool_calls) == 1
        assert result[0].tool_calls[0].id == "call_123"
        assert result[0].tool_calls[0].function.name == "test_function_a"
        assert result[0].tool_calls[0].function.arguments == '{"param": "value"}'
        assert isinstance(result[1], OpenAIToolMessageParam)
        assert result[1].content == "AAA"
        assert result[1].tool_call_id == "call_123"
        assert isinstance(result[2], OpenAIAssistantMessageParam)
        assert len(result[2].tool_calls) == 1
        assert result[2].tool_calls[0].id == "call_456"
        assert result[2].tool_calls[0].function.name == "test_function_b"
        assert result[2].tool_calls[0].function.arguments == '{"param": "value"}'
        assert isinstance(result[3], OpenAIToolMessageParam)
        assert result[3].content == "BBB"
        assert result[3].tool_call_id == "call_456"

    async def test_convert_response_message(self):
        input_items = [
            OpenAIResponseMessage(
                role="user",
                content=[OpenAIResponseInputMessageContentText(text="User text")],
            )
        ]

        result = await convert_response_input_to_chat_messages(input_items)

        assert len(result) == 1
        assert isinstance(result[0], OpenAIUserMessageParam)
        # Content should be converted to chat content format
        assert len(result[0].content) == 1
        assert result[0].content[0].text == "User text"


class TestConvertResponseTextToChatResponseFormat:
    async def test_convert_text_format(self):
        text = OpenAIResponseText(format=OpenAIResponseTextFormat(type="text"))
        result = await convert_response_text_to_chat_response_format(text)

        assert isinstance(result, OpenAIResponseFormatText)
        assert result.type == "text"

    async def test_convert_json_object_format(self):
        text = OpenAIResponseText(format={"type": "json_object"})
        result = await convert_response_text_to_chat_response_format(text)

        assert isinstance(result, OpenAIResponseFormatJSONObject)

    async def test_convert_json_schema_format(self):
        schema_def = {"type": "object", "properties": {"test": {"type": "string"}}}
        text = OpenAIResponseText(
            format={
                "type": "json_schema",
                "name": "test_schema",
                "schema": schema_def,
            }
        )
        result = await convert_response_text_to_chat_response_format(text)

        assert isinstance(result, OpenAIResponseFormatJSONSchema)
        assert result.json_schema["name"] == "test_schema"
        assert result.json_schema["schema"] == schema_def

    async def test_default_text_format(self):
        text = OpenAIResponseText()
        result = await convert_response_text_to_chat_response_format(text)

        assert isinstance(result, OpenAIResponseFormatText)
        assert result.type == "text"


class TestGetMessageTypeByRole:
    async def test_user_role(self):
        result = await get_message_type_by_role("user")
        assert result == OpenAIUserMessageParam

    async def test_system_role(self):
        result = await get_message_type_by_role("system")
        assert result == OpenAISystemMessageParam

    async def test_assistant_role(self):
        result = await get_message_type_by_role("assistant")
        assert result == OpenAIAssistantMessageParam

    async def test_developer_role(self):
        result = await get_message_type_by_role("developer")
        assert result == OpenAIDeveloperMessageParam

    async def test_unknown_role(self):
        result = await get_message_type_by_role("unknown")
        assert result is None


class TestIsFunctionToolCall:
    def test_is_function_tool_call_true(self):
        tool_call = OpenAIChatCompletionToolCall(
            index=0,
            id="call_123",
            function=OpenAIChatCompletionToolCallFunction(
                name="test_function",
                arguments="{}",
            ),
        )
        tools = [
            OpenAIResponseInputToolFunction(
                type="function", name="test_function", parameters={"type": "object", "properties": {}}
            ),
            OpenAIResponseInputToolWebSearch(type="web_search"),
        ]

        result = is_function_tool_call(tool_call, tools)
        assert result is True

    def test_is_function_tool_call_false_different_name(self):
        tool_call = OpenAIChatCompletionToolCall(
            index=0,
            id="call_123",
            function=OpenAIChatCompletionToolCallFunction(
                name="other_function",
                arguments="{}",
            ),
        )
        tools = [
            OpenAIResponseInputToolFunction(
                type="function", name="test_function", parameters={"type": "object", "properties": {}}
            ),
        ]

        result = is_function_tool_call(tool_call, tools)
        assert result is False

    def test_is_function_tool_call_false_no_function(self):
        tool_call = OpenAIChatCompletionToolCall(
            index=0,
            id="call_123",
            function=None,
        )
        tools = [
            OpenAIResponseInputToolFunction(
                type="function", name="test_function", parameters={"type": "object", "properties": {}}
            ),
        ]

        result = is_function_tool_call(tool_call, tools)
        assert result is False

    def test_is_function_tool_call_false_wrong_type(self):
        tool_call = OpenAIChatCompletionToolCall(
            index=0,
            id="call_123",
            function=OpenAIChatCompletionToolCallFunction(
                name="web_search",
                arguments="{}",
            ),
        )
        tools = [
            OpenAIResponseInputToolWebSearch(type="web_search"),
        ]

        result = is_function_tool_call(tool_call, tools)
        assert result is False


class TestReasoningSupportInConversion:
    """Tests for reasoning look-back in convert_response_input_to_chat_messages.

    When a ReasoningItem appears in the input, it should be:
    1. Skipped (not converted to its own CC message)
    2. Attached as `reasoning=<text>` on the NEXT assistant message via _get_preceding_reasoning()

    This applies to FunctionToolCalls, McpCalls, and ResponseMessages with role='assistant'.
    """

    async def test_reasoning_attached_to_function_tool_call(self):
        """ReasoningItem before a FunctionToolCall should attach reasoning to the assistant message.

        Scenario: Model reasons about which tool to call, then calls get_weather.
        """
        input_items = [
            OpenAIResponseMessage(role="user", content="What's the weather in Tokyo?"),
            OpenAIResponseOutputMessageReasoningItem(
                id="rs_001",
                summary=[],
                content=[OpenAIResponseOutputMessageReasoningContent(text="Need to call get_weather for Tokyo.")],
                status="completed",
            ),
            OpenAIResponseOutputMessageFunctionToolCall(
                call_id="call_abc",
                name="get_weather",
                arguments='{"location":"Tokyo","unit":"celsius"}',
            ),
            OpenAIResponseInputFunctionToolCallOutput(
                output='{"temperature": 27, "condition": "humid"}',
                call_id="call_abc",
            ),
        ]

        result = await convert_response_input_to_chat_messages(input_items)

        assert len(result) == 3
        # user message
        assert isinstance(result[0], OpenAIUserMessageParam)
        # assistant with reasoning attached
        assert isinstance(result[1], OpenAIAssistantMessageParam)
        assert result[1].tool_calls[0].function.name == "get_weather"
        assert result[1].reasoning_content == "Need to call get_weather for Tokyo."
        # tool result
        assert isinstance(result[2], OpenAIToolMessageParam)
        assert result[2].tool_call_id == "call_abc"

    async def test_reasoning_attached_to_assistant_content_message(self):
        """ReasoningItem before an assistant ResponseMessage should attach reasoning.

        Scenario: resp_2 output from the MCP notebook -- model reasons then produces content.
        Input: [user("Hi!"), McpListTools, ReasoningItem("greeting"), ResponseMessage(assistant, "Hi there!")]
        Expected CC: [user("Hi!"), assistant(content="Hi there!", reasoning="greeting")]
        """
        input_items = [
            OpenAIResponseMessage(role="user", content="Hi !"),
            OpenAIResponseOutputMessageMCPListTools(
                id="mcp_list_001",
                server_label="gitmcp",
                tools=[
                    MCPListToolsTool(
                        input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
                        name="search_docs",
                        description="Search documentation",
                    ),
                ],
            ),
            OpenAIResponseOutputMessageReasoningItem(
                id="rs_002",
                summary=[],
                content=[
                    OpenAIResponseOutputMessageReasoningContent(
                        text='The user says "Hi !". We respond politely.',
                    )
                ],
                status="completed",
            ),
            OpenAIResponseMessage(
                role="assistant",
                content=[OpenAIResponseOutputMessageContentOutputText(text="Hi there! How can I help you today?")],
                id="msg_001",
                status="completed",
            ),
        ]

        result = await convert_response_input_to_chat_messages(input_items)

        assert len(result) == 2
        # user
        assert isinstance(result[0], OpenAIUserMessageParam)
        assert result[0].content == "Hi !"
        # assistant with reasoning from look-back
        assert isinstance(result[1], OpenAIAssistantMessageParam)
        assert result[1].reasoning_content == 'The user says "Hi !". We respond politely.'

    async def test_mcp_tool_call_reasoning_lookback(self):
        """ReasoningItem before an McpCall should attach reasoning to the assistant message.

        This is the resp_3 scenario from the MCP notebook: resp_2 returned
        [McpListTools, ReasoningItem, McpCall(search), ReasoningItem, McpCall(fetch),
         ReasoningItem, OutputMessage]. When passed back as input for resp_3,
        the McpCall assistant messages should carry their preceding reasoning.

        Input (simplified from debug logs):
          [user("Hi!"), McpListTools, ReasoningItem("greeting"), assistant("Hi there!"),
           user("overview?"), McpListTools,
           ReasoningItem("First search"), McpCall(search, output="Found: overview"),
           ReasoningItem("Now fetch"), McpCall(fetch, output="tiktoken is a fast BPE..."),
           ReasoningItem("Produce overview"), assistant("**tiktoken**..."),
           user("thanks!")]

        Expected CC output:
          [user("Hi!"), assistant("Hi there!", reasoning="greeting"),
           user("overview?"),
           assistant(tool_calls=[search], reasoning="First search"), tool("Found: overview"),
           assistant(tool_calls=[fetch], reasoning="Now fetch"), tool("tiktoken is..."),
           assistant("**tiktoken**...", reasoning="Produce overview"),
           user("thanks!")]
        """
        input_items = [
            # Turn 1: greeting
            OpenAIResponseMessage(role="user", content="Hi !"),
            OpenAIResponseOutputMessageMCPListTools(
                id="mcp_list_001",
                server_label="gitmcp",
                tools=[
                    MCPListToolsTool(
                        input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
                        name="search_tiktoken_documentation",
                        description="Search tiktoken docs",
                    ),
                    MCPListToolsTool(
                        input_schema={"type": "object", "properties": {"topic": {"type": "string"}}},
                        name="fetch_tiktoken_documentation",
                        description="Fetch tiktoken docs for a topic",
                    ),
                ],
            ),
            OpenAIResponseOutputMessageReasoningItem(
                id="rs_greeting",
                summary=[],
                content=[OpenAIResponseOutputMessageReasoningContent(text="Friendly greeting back.")],
                status="completed",
            ),
            OpenAIResponseMessage(
                role="assistant",
                content=[OpenAIResponseOutputMessageContentOutputText(text="Hi there! How can I help you today?")],
                id="msg_greeting",
                status="completed",
            ),
            # Turn 2: user asks about tiktoken
            OpenAIResponseMessage(
                role="user",
                content="Give me very brief overview of tiktoken?",
            ),
            # Turn 2 output: MCP tool calls with reasoning
            OpenAIResponseOutputMessageMCPListTools(
                id="mcp_list_002",
                server_label="gitmcp",
                tools=[
                    MCPListToolsTool(
                        input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
                        name="search_tiktoken_documentation",
                        description="Search tiktoken docs",
                    ),
                    MCPListToolsTool(
                        input_schema={"type": "object", "properties": {"topic": {"type": "string"}}},
                        name="fetch_tiktoken_documentation",
                        description="Fetch tiktoken docs for a topic",
                    ),
                ],
            ),
            OpenAIResponseOutputMessageReasoningItem(
                id="rs_search",
                summary=[],
                content=[
                    OpenAIResponseOutputMessageReasoningContent(
                        text="We need to use the provided tool functions. First search.",
                    )
                ],
                status="completed",
            ),
            OpenAIResponseOutputMessageMCPCall(
                id="fc_search",
                name="search_tiktoken_documentation",
                arguments='{"query":"tiktoken overview"}',
                server_label="gitmcp",
                output="Found Section: `overview`",
            ),
            OpenAIResponseOutputMessageReasoningItem(
                id="rs_fetch",
                summary=[],
                content=[OpenAIResponseOutputMessageReasoningContent(text="Now fetch overview.")],
                status="completed",
            ),
            OpenAIResponseOutputMessageMCPCall(
                id="fc_fetch",
                name="fetch_tiktoken_documentation",
                arguments='{"topic":"overview"}',
                server_label="gitmcp",
                output="tiktoken is a fast BPE tokeniser for OpenAI models.",
            ),
            OpenAIResponseOutputMessageReasoningItem(
                id="rs_summary",
                summary=[],
                content=[OpenAIResponseOutputMessageReasoningContent(text="We need to produce brief overview.")],
                status="completed",
            ),
            OpenAIResponseMessage(
                role="assistant",
                content=[
                    OpenAIResponseOutputMessageContentOutputText(
                        text="**tiktoken** - a fast BPE tokenizer for OpenAI models.",
                    )
                ],
                id="msg_summary",
                status="completed",
            ),
            # Turn 3: user thanks
            OpenAIResponseMessage(role="user", content="hmm. thanks ! this is helpful"),
        ]

        result = await convert_response_input_to_chat_messages(input_items)

        # Expected: 9 messages
        # [0] user("Hi!")
        # [1] assistant("Hi there!", reasoning="Friendly greeting back.")
        # [2] user("overview?")
        # [3] assistant(tool_calls=[search], reasoning="...First search.")
        # [4] tool("Found Section: overview")
        # [5] assistant(tool_calls=[fetch], reasoning="Now fetch overview.")
        # [6] tool("tiktoken is a fast BPE...")
        # [7] assistant("**tiktoken**...", reasoning="...produce brief overview.")
        # [8] user("thanks!")
        assert len(result) == 9

        # [0] user
        assert isinstance(result[0], OpenAIUserMessageParam)
        assert result[0].content == "Hi !"

        # [1] assistant with greeting reasoning
        assert isinstance(result[1], OpenAIAssistantMessageParam)
        assert result[1].reasoning_content == "Friendly greeting back."

        # [2] user
        assert isinstance(result[2], OpenAIUserMessageParam)
        assert result[2].content == "Give me very brief overview of tiktoken?"

        # [3] assistant from McpCall(search) -- should have reasoning from look-back
        assert isinstance(result[3], OpenAIAssistantMessageParam)
        assert result[3].tool_calls is not None
        assert result[3].tool_calls[0].function.name == "search_tiktoken_documentation"
        assert result[3].reasoning_content == "We need to use the provided tool functions. First search."

        # [4] tool result for search
        assert isinstance(result[4], OpenAIToolMessageParam)
        assert result[4].content == "Found Section: `overview`"
        assert result[4].tool_call_id == "fc_search"

        # [5] assistant from McpCall(fetch) -- should have reasoning from look-back
        assert isinstance(result[5], OpenAIAssistantMessageParam)
        assert result[5].tool_calls is not None
        assert result[5].tool_calls[0].function.name == "fetch_tiktoken_documentation"
        assert result[5].reasoning_content == "Now fetch overview."

        # [6] tool result for fetch
        assert isinstance(result[6], OpenAIToolMessageParam)
        assert result[6].content == "tiktoken is a fast BPE tokeniser for OpenAI models."
        assert result[6].tool_call_id == "fc_fetch"

        # [7] assistant content message with reasoning
        assert isinstance(result[7], OpenAIAssistantMessageParam)
        assert result[7].reasoning_content == "We need to produce brief overview."

        # [8] user
        assert isinstance(result[8], OpenAIUserMessageParam)
        assert result[8].content == "hmm. thanks ! this is helpful"

    async def test_no_reasoning_when_no_preceding_reasoning_item(self):
        """When there is no ReasoningItem before a tool call, reasoning should not be set."""
        input_items = [
            OpenAIResponseMessage(role="user", content="What's the weather?"),
            OpenAIResponseOutputMessageFunctionToolCall(
                call_id="call_no_reason",
                name="get_weather",
                arguments='{"location":"Tokyo"}',
            ),
            OpenAIResponseInputFunctionToolCallOutput(
                output='{"temperature": 27}',
                call_id="call_no_reason",
            ),
        ]

        result = await convert_response_input_to_chat_messages(input_items)

        assert len(result) == 3
        assert isinstance(result[1], OpenAIAssistantMessageParam)
        assert not hasattr(result[1], "reasoning")


class TestExtractCitationsFromText:
    def test_extract_citations_and_annotations(self):
        text = "Start [not-a-file]. New source <|file-abc123|>. "
        text += "Other source <|file-def456|>? Repeat source <|file-abc123|>! No citation."
        file_mapping = {"file-abc123": "doc1.pdf", "file-def456": "doc2.txt"}

        annotations, cleaned_text = _extract_citations_from_text(text, file_mapping)

        expected_annotations = [
            OpenAIResponseAnnotationFileCitation(file_id="file-abc123", filename="doc1.pdf", index=30),
            OpenAIResponseAnnotationFileCitation(file_id="file-def456", filename="doc2.txt", index=44),
            OpenAIResponseAnnotationFileCitation(file_id="file-abc123", filename="doc1.pdf", index=59),
        ]
        expected_clean_text = "Start [not-a-file]. New source. Other source? Repeat source! No citation."

        assert cleaned_text == expected_clean_text
        assert annotations == expected_annotations
        # OpenAI cites at the end of the sentence
        assert cleaned_text[expected_annotations[0].index] == "."
        assert cleaned_text[expected_annotations[1].index] == "?"
        assert cleaned_text[expected_annotations[2].index] == "!"
