# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for Anthropic<->OpenAI translation utilities in anthropic_translation."""

from unittest.mock import MagicMock

import pytest

from ogx.providers.utils.inference.anthropic_translation import (
    anthropic_request_to_openai,
    convert_tool_choice_to_openai,
    openai_response_to_anthropic,
    openai_stream_to_anthropic,
    parse_anthropic_sse_event,
)
from ogx_api.messages.models import (
    AnthropicBase64ImageSource,
    AnthropicBashTool,
    AnthropicCacheControl,
    AnthropicCreateMessageRequest,
    AnthropicCustomToolDef,
    AnthropicImageBlock,
    AnthropicMessage,
    AnthropicRedactedThinkingBlock,
    AnthropicTextBlock,
    AnthropicTextEditorTool,
    AnthropicThinkingBlock,
    AnthropicThinkingConfig,
    AnthropicToolResultBlock,
    AnthropicToolUseBlock,
    AnthropicURLImageSource,
    AnthropicWebSearchTool,
    _ToolChoiceAny,
    _ToolChoiceAuto,
    _ToolChoiceNone,
    _ToolChoiceTool,
)


def _msg_to_dict(msg):
    """Convert a Pydantic message model to dict for easy assertion."""
    if hasattr(msg, "model_dump"):
        return msg.model_dump(exclude_none=True)
    return dict(msg)


# -- Request (Anthropic -> OpenAI) --


class TestRequestTranslation:
    def test_simple_text_message(self):
        request = AnthropicCreateMessageRequest(
            model="claude-sonnet-4-20250514",
            messages=[AnthropicMessage(role="user", content="Hello")],
            max_tokens=100,
        )
        result = anthropic_request_to_openai(request)

        assert result.model == "claude-sonnet-4-20250514"
        assert result.max_tokens == 100
        assert len(result.messages) == 1
        m = _msg_to_dict(result.messages[0])
        assert m["role"] == "user"
        assert m["content"] == "Hello"

    def test_system_string(self):
        request = AnthropicCreateMessageRequest(
            model="m",
            messages=[AnthropicMessage(role="user", content="Hi")],
            max_tokens=100,
            system="You are helpful.",
        )
        result = anthropic_request_to_openai(request)

        m0 = _msg_to_dict(result.messages[0])
        m1 = _msg_to_dict(result.messages[1])
        assert m0["role"] == "system"
        assert m0["content"] == "You are helpful."
        assert m1["role"] == "user"

    def test_system_text_blocks(self):
        request = AnthropicCreateMessageRequest(
            model="m",
            messages=[AnthropicMessage(role="user", content="Hi")],
            max_tokens=100,
            system=[
                AnthropicTextBlock(text="Line 1."),
                AnthropicTextBlock(text="Line 2."),
            ],
        )
        result = anthropic_request_to_openai(request)

        m0 = _msg_to_dict(result.messages[0])
        assert m0["role"] == "system"
        assert m0["content"] == "Line 1.\nLine 2."

    def test_inline_system_message_string(self):
        # Clients such as the Claude Code CLI interleave system-role messages
        # inside the conversation rather than using the top-level system field.
        request = AnthropicCreateMessageRequest(
            model="m",
            messages=[
                AnthropicMessage(role="user", content="Hi"),
                AnthropicMessage(role="system", content="Stay terse."),
            ],
            max_tokens=100,
        )
        result = anthropic_request_to_openai(request)

        m1 = _msg_to_dict(result.messages[1])
        assert m1["role"] == "system"
        assert m1["content"] == "Stay terse."

    def test_inline_system_message_text_blocks(self):
        request = AnthropicCreateMessageRequest(
            model="m",
            messages=[
                AnthropicMessage(role="user", content="Hi"),
                AnthropicMessage(
                    role="system",
                    content=[
                        AnthropicTextBlock(text="Line 1."),
                        AnthropicTextBlock(text="Line 2."),
                    ],
                ),
            ],
            max_tokens=100,
        )
        result = anthropic_request_to_openai(request)

        m1 = _msg_to_dict(result.messages[1])
        assert m1["role"] == "system"
        assert m1["content"] == "Line 1.\nLine 2."

    def test_tool_definitions(self):
        request = AnthropicCreateMessageRequest(
            model="m",
            messages=[AnthropicMessage(role="user", content="Hi")],
            max_tokens=100,
            tools=[
                AnthropicCustomToolDef(
                    name="get_weather",
                    description="Get weather",
                    input_schema={"type": "object", "properties": {"location": {"type": "string"}}},
                ),
            ],
        )
        result = anthropic_request_to_openai(request)

        assert len(result.tools) == 1
        tool = result.tools[0]
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "get_weather"
        assert tool["function"]["parameters"]["type"] == "object"

    def test_tool_choice_any(self):
        assert convert_tool_choice_to_openai(_ToolChoiceAny()) == "required"

    def test_tool_choice_none(self):
        assert convert_tool_choice_to_openai(_ToolChoiceNone()) == "none"

    def test_tool_choice_auto(self):
        assert convert_tool_choice_to_openai(_ToolChoiceAuto()) == "auto"

    def test_tool_choice_specific(self):
        result = convert_tool_choice_to_openai(_ToolChoiceTool(name="get_weather"))
        assert result == {"type": "function", "function": {"name": "get_weather"}}

    def test_disable_parallel_tool_use_translated(self):
        request = AnthropicCreateMessageRequest(
            model="m",
            messages=[AnthropicMessage(role="user", content="Hi")],
            max_tokens=100,
            tools=[
                AnthropicCustomToolDef(
                    name="search",
                    description="Search",
                    input_schema={"type": "object", "properties": {}},
                ),
            ],
            tool_choice={"type": "auto", "disable_parallel_tool_use": True},
        )
        result = anthropic_request_to_openai(request)
        assert result.parallel_tool_calls is False

    def test_parallel_tool_calls_default_when_not_disabled(self):
        request = AnthropicCreateMessageRequest(
            model="m",
            messages=[AnthropicMessage(role="user", content="Hi")],
            max_tokens=100,
            tool_choice={"type": "auto"},
        )
        result = anthropic_request_to_openai(request)
        assert result.parallel_tool_calls is None

    def test_stop_sequences(self):
        request = AnthropicCreateMessageRequest(
            model="m",
            messages=[AnthropicMessage(role="user", content="Hi")],
            max_tokens=100,
            stop_sequences=["STOP", "END"],
        )
        result = anthropic_request_to_openai(request)
        assert result.stop == ["STOP", "END"]

    def test_tool_use_in_assistant_message(self):
        import json

        request = AnthropicCreateMessageRequest(
            model="m",
            messages=[
                AnthropicMessage(
                    role="assistant",
                    content=[
                        AnthropicTextBlock(text="Let me check the weather."),
                        AnthropicToolUseBlock(
                            id="toolu_123",
                            name="get_weather",
                            input={"location": "SF"},
                        ),
                    ],
                ),
            ],
            max_tokens=100,
        )
        result = anthropic_request_to_openai(request)

        msg = _msg_to_dict(result.messages[0])
        assert msg["role"] == "assistant"
        assert msg["content"] == "Let me check the weather."
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["id"] == "toolu_123"
        assert msg["tool_calls"][0]["function"]["name"] == "get_weather"
        assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"location": "SF"}

    def test_tool_result_in_user_message(self):
        request = AnthropicCreateMessageRequest(
            model="m",
            messages=[
                AnthropicMessage(
                    role="user",
                    content=[
                        AnthropicToolResultBlock(
                            tool_use_id="toolu_123",
                            content="72F and sunny",
                        ),
                    ],
                ),
            ],
            max_tokens=100,
        )
        result = anthropic_request_to_openai(request)

        msg = _msg_to_dict(result.messages[0])
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "toolu_123"
        assert msg["content"] == "72F and sunny"

    def test_base64_image_in_user_message(self):
        request = AnthropicCreateMessageRequest(
            model="m",
            messages=[
                AnthropicMessage(
                    role="user",
                    content=[
                        AnthropicTextBlock(text="What is in this image?"),
                        AnthropicImageBlock(
                            source=AnthropicBase64ImageSource(
                                media_type="image/png",
                                data="abc123",
                            )
                        ),
                    ],
                ),
            ],
            max_tokens=100,
        )
        result = anthropic_request_to_openai(request)

        assert len(result.messages) == 1
        msg = _msg_to_dict(result.messages[0])
        assert msg["role"] == "user"
        assert isinstance(msg["content"], list)
        assert msg["content"][0] == {"type": "text", "text": "What is in this image?"}
        assert msg["content"][1] == {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,abc123"},
        }

    def test_url_image_in_user_message(self):
        request = AnthropicCreateMessageRequest(
            model="m",
            messages=[
                AnthropicMessage(
                    role="user",
                    content=[
                        AnthropicImageBlock(source=AnthropicURLImageSource(url="https://example.com/img.jpg")),
                    ],
                ),
            ],
            max_tokens=100,
        )
        result = anthropic_request_to_openai(request)

        assert len(result.messages) == 1
        msg = _msg_to_dict(result.messages[0])
        assert msg["content"] == [{"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}}]

    def test_image_in_tool_result_promoted_to_user_message(self):
        request = AnthropicCreateMessageRequest(
            model="m",
            messages=[
                AnthropicMessage(
                    role="user",
                    content=[
                        AnthropicToolResultBlock(
                            tool_use_id="toolu_abc",
                            content=[
                                AnthropicTextBlock(text="Screenshot taken"),
                                AnthropicImageBlock(
                                    source=AnthropicBase64ImageSource(
                                        media_type="image/png",
                                        data="screenshotdata",
                                    )
                                ),
                            ],
                        ),
                    ],
                ),
            ],
            max_tokens=100,
        )
        result = anthropic_request_to_openai(request)

        assert len(result.messages) == 2
        tool_msg = _msg_to_dict(result.messages[0])
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == "toolu_abc"
        assert tool_msg["content"] == "Screenshot taken"
        image_msg = _msg_to_dict(result.messages[1])
        assert image_msg["role"] == "user"
        assert image_msg["content"] == [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,screenshotdata"}}
        ]

    def test_top_k_passed_as_extra(self):
        request = AnthropicCreateMessageRequest(
            model="m",
            messages=[AnthropicMessage(role="user", content="Hi")],
            max_tokens=100,
            top_k=40,
        )
        result = anthropic_request_to_openai(request)
        assert result.model_extra.get("top_k") == 40

    def test_redacted_thinking_skipped_in_assistant_message(self):
        request = AnthropicCreateMessageRequest(
            model="m",
            messages=[
                AnthropicMessage(
                    role="assistant",
                    content=[
                        AnthropicThinkingBlock(thinking="reasoning here", signature="sig123"),
                        AnthropicRedactedThinkingBlock(data="opaque-data"),
                        AnthropicTextBlock(text="The answer is 42."),
                    ],
                ),
            ],
            max_tokens=100,
        )
        result = anthropic_request_to_openai(request)

        msg = _msg_to_dict(result.messages[0])
        assert msg["role"] == "assistant"
        assert msg["content"] == "The answer is 42."
        assert "tool_calls" not in msg

    def test_cache_control_on_text_block_parses(self):
        request = AnthropicCreateMessageRequest(
            model="m",
            messages=[
                AnthropicMessage(
                    role="user",
                    content=[AnthropicTextBlock(text="Hello", cache_control=AnthropicCacheControl())],
                )
            ],
            max_tokens=100,
            system=[AnthropicTextBlock(text="You are helpful.", cache_control=AnthropicCacheControl())],
        )
        result = anthropic_request_to_openai(request)
        assert len(result.messages) == 2
        assert _msg_to_dict(result.messages[0])["role"] == "system"

    def test_cache_control_on_tool_def_parses(self):
        request = AnthropicCreateMessageRequest(
            model="m",
            messages=[AnthropicMessage(role="user", content="Hi")],
            max_tokens=100,
            tools=[
                AnthropicCustomToolDef(
                    name="get_weather",
                    description="Get weather",
                    input_schema={"type": "object", "properties": {}},
                    cache_control=AnthropicCacheControl(),
                )
            ],
        )
        result = anthropic_request_to_openai(request)
        assert len(result.tools) == 1
        assert result.tools[0]["function"]["name"] == "get_weather"

    def test_server_tools_accepted_in_request(self):
        request = AnthropicCreateMessageRequest(
            model="m",
            messages=[AnthropicMessage(role="user", content="Hi")],
            max_tokens=100,
            tools=[
                AnthropicCustomToolDef(
                    name="get_weather",
                    input_schema={"type": "object", "properties": {}},
                ),
                AnthropicWebSearchTool(),
                AnthropicBashTool(),
                AnthropicTextEditorTool(type="text_editor_20250728", name="str_replace_based_edit_tool"),
            ],
        )
        assert len(request.tools) == 4

    def test_server_tools_dropped_in_translation(self):
        request = AnthropicCreateMessageRequest(
            model="m",
            messages=[AnthropicMessage(role="user", content="Hi")],
            max_tokens=100,
            tools=[
                AnthropicCustomToolDef(
                    name="get_weather",
                    input_schema={"type": "object", "properties": {}},
                ),
                AnthropicWebSearchTool(),
                AnthropicBashTool(),
            ],
        )
        result = anthropic_request_to_openai(request)
        assert len(result.tools) == 1
        assert result.tools[0]["function"]["name"] == "get_weather"

    def test_only_server_tools_yields_no_tools_in_translation(self):
        request = AnthropicCreateMessageRequest(
            model="m",
            messages=[AnthropicMessage(role="user", content="Hi")],
            max_tokens=100,
            tools=[AnthropicWebSearchTool(), AnthropicBashTool()],
        )
        result = anthropic_request_to_openai(request)
        assert result.tools is None

    def test_tool_choice_dropped_when_all_tools_filtered(self):
        request = AnthropicCreateMessageRequest(
            model="m",
            messages=[AnthropicMessage(role="user", content="Hi")],
            max_tokens=100,
            tools=[AnthropicWebSearchTool()],
            tool_choice="any",
        )
        result = anthropic_request_to_openai(request)
        assert result.tools is None
        assert result.tool_choice is None


# -- Response (OpenAI -> Anthropic) --


class TestResponseTranslation:
    def test_simple_text_response(self):
        openai_resp = MagicMock()
        openai_resp.choices = [MagicMock()]
        openai_resp.choices[0].message = MagicMock()
        openai_resp.choices[0].message.content = "Hello!"
        openai_resp.choices[0].message.tool_calls = None
        openai_resp.choices[0].finish_reason = "stop"
        openai_resp.usage = MagicMock()
        openai_resp.usage.prompt_tokens = 10
        openai_resp.usage.completion_tokens = 5

        result = openai_response_to_anthropic(openai_resp, "claude-sonnet-4-20250514")

        assert result.id.startswith("msg_")
        assert result.type == "message"
        assert result.role == "assistant"
        assert result.model == "claude-sonnet-4-20250514"
        assert result.stop_reason == "end_turn"
        assert len(result.content) == 1
        assert result.content[0].type == "text"
        assert result.content[0].text == "Hello!"
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5

    def test_tool_call_response(self):
        tc = MagicMock()
        tc.id = "call_123"
        tc.function.name = "get_weather"
        tc.function.arguments = '{"location": "SF"}'

        openai_resp = MagicMock()
        openai_resp.choices = [MagicMock()]
        openai_resp.choices[0].message = MagicMock()
        openai_resp.choices[0].message.content = None
        openai_resp.choices[0].message.tool_calls = [tc]
        openai_resp.choices[0].finish_reason = "tool_calls"
        openai_resp.usage = MagicMock()
        openai_resp.usage.prompt_tokens = 20
        openai_resp.usage.completion_tokens = 10

        result = openai_response_to_anthropic(openai_resp, "m")

        assert result.stop_reason == "tool_use"
        assert len(result.content) == 1
        assert result.content[0].type == "tool_use"
        assert result.content[0].name == "get_weather"
        assert result.content[0].input == {"location": "SF"}

    def test_length_stop_reason(self):
        openai_resp = MagicMock()
        openai_resp.choices = [MagicMock()]
        openai_resp.choices[0].message = MagicMock()
        openai_resp.choices[0].message.content = "truncated"
        openai_resp.choices[0].message.tool_calls = None
        openai_resp.choices[0].finish_reason = "length"
        openai_resp.usage = MagicMock()
        openai_resp.usage.prompt_tokens = 5
        openai_resp.usage.completion_tokens = 100

        result = openai_response_to_anthropic(openai_resp, "m")
        assert result.stop_reason == "max_tokens"

    def test_cache_metrics_mapping(self):
        openai_resp = MagicMock()
        openai_resp.choices = [MagicMock()]
        openai_resp.choices[0].message = MagicMock()
        openai_resp.choices[0].message.content = "response"
        openai_resp.choices[0].message.tool_calls = None
        openai_resp.choices[0].finish_reason = "stop"
        openai_resp.usage = MagicMock()
        openai_resp.usage.prompt_tokens = 100
        openai_resp.usage.completion_tokens = 50
        openai_resp.usage.prompt_tokens_details = MagicMock()
        openai_resp.usage.prompt_tokens_details.cached_tokens = 75

        result = openai_response_to_anthropic(openai_resp, "m")
        assert result.usage.input_tokens == 100
        assert result.usage.output_tokens == 50
        assert result.usage.cache_read_input_tokens == 75
        assert result.usage.cache_creation_input_tokens is None

    def test_cache_metrics_missing(self):
        openai_resp = MagicMock()
        openai_resp.choices = [MagicMock()]
        openai_resp.choices[0].message = MagicMock()
        openai_resp.choices[0].message.content = "response"
        openai_resp.choices[0].message.tool_calls = None
        openai_resp.choices[0].finish_reason = "stop"
        openai_resp.usage = MagicMock()
        openai_resp.usage.prompt_tokens = 100
        openai_resp.usage.completion_tokens = 50
        openai_resp.usage.prompt_tokens_details = None

        result = openai_response_to_anthropic(openai_resp, "m")
        assert result.usage.input_tokens == 100
        assert result.usage.output_tokens == 50
        assert result.usage.cache_read_input_tokens is None
        assert result.usage.cache_creation_input_tokens is None


# -- Streaming (OpenAI -> Anthropic) --


class TestStreamingTranslation:
    async def test_text_streaming(self):
        chunks = []

        for i, text in enumerate(["Hello", " world", "!"]):
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta = MagicMock()
            chunk.choices[0].delta.content = text
            chunk.choices[0].delta.tool_calls = None
            chunk.choices[0].finish_reason = "stop" if i == 2 else None
            chunk.usage = None
            chunks.append(chunk)

        async def mock_stream():
            for c in chunks:
                yield c

        events = []
        async for event in openai_stream_to_anthropic(mock_stream(), "m"):
            events.append(event)

        assert events[0].type == "message_start"
        assert events[1].type == "ping"
        assert events[2].type == "content_block_start"
        assert events[2].content_block.type == "text"
        assert events[3].type == "content_block_delta"
        assert events[3].delta.text == "Hello"
        assert events[4].type == "content_block_delta"
        assert events[4].delta.text == " world"
        assert events[5].type == "content_block_delta"
        assert events[5].delta.text == "!"
        assert events[6].type == "content_block_stop"
        assert events[7].type == "ping"
        assert events[8].type == "message_delta"
        assert events[8].delta.stop_reason == "end_turn"
        assert events[9].type == "message_stop"

    async def test_tool_call_streaming(self):
        chunks = []

        tc_delta = MagicMock()
        tc_delta.index = 0
        tc_delta.id = "call_abc"
        tc_delta.function = MagicMock()
        tc_delta.function.name = "search"
        tc_delta.function.arguments = None
        tc_delta.type = "function"

        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta = MagicMock()
        chunk1.choices[0].delta.content = None
        chunk1.choices[0].delta.tool_calls = [tc_delta]
        chunk1.choices[0].finish_reason = None
        chunk1.usage = None
        chunks.append(chunk1)

        tc_delta2 = MagicMock()
        tc_delta2.index = 0
        tc_delta2.id = None
        tc_delta2.function = MagicMock()
        tc_delta2.function.name = None
        tc_delta2.function.arguments = '{"query": "test"}'

        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta = MagicMock()
        chunk2.choices[0].delta.content = None
        chunk2.choices[0].delta.tool_calls = [tc_delta2]
        chunk2.choices[0].finish_reason = "tool_calls"
        chunk2.usage = None
        chunks.append(chunk2)

        async def mock_stream():
            for c in chunks:
                yield c

        events = []
        async for event in openai_stream_to_anthropic(mock_stream(), "m"):
            events.append(event)

        assert events[0].type == "message_start"
        tool_start = [e for e in events if e.type == "content_block_start" and hasattr(e.content_block, "name")]
        assert len(tool_start) == 1
        assert tool_start[0].content_block.name == "search"

        json_deltas = [e for e in events if e.type == "content_block_delta" and hasattr(e.delta, "partial_json")]
        assert len(json_deltas) == 1
        assert json_deltas[0].delta.partial_json == '{"query": "test"}'

        msg_delta = [e for e in events if e.type == "message_delta"]
        assert msg_delta[0].delta.stop_reason == "tool_use"


# -- SSE Parsing --


class TestSSEParsing:
    def test_signature_delta_parsed(self):
        event = parse_anthropic_sse_event(
            "content_block_delta",
            {
                "index": 0,
                "delta": {"type": "signature_delta", "signature": "ErUBCkYIAxgCIkA"},
            },
        )
        assert event is not None
        assert event.type == "content_block_delta"
        assert event.delta.type == "signature_delta"
        assert event.delta.signature == "ErUBCkYIAxgCIkA"

    def test_redacted_thinking_block_start_parsed(self):
        event = parse_anthropic_sse_event(
            "content_block_start",
            {
                "index": 0,
                "content_block": {
                    "type": "redacted_thinking",
                    "data": "opaque-redacted-data-string",
                },
            },
        )
        assert event is not None
        assert event.type == "content_block_start"
        assert event.content_block.type == "redacted_thinking"
        assert event.content_block.data == "opaque-redacted-data-string"

    def test_pause_turn_stop_reason_passthrough(self):
        event = parse_anthropic_sse_event(
            "message_delta",
            {
                "delta": {"stop_reason": "pause_turn"},
                "usage": {"output_tokens": 10},
            },
        )
        assert event is not None
        assert event.type == "message_delta"
        assert event.delta.stop_reason == "pause_turn"

    def test_ping_event_parsed(self):
        event = parse_anthropic_sse_event("ping", {})
        assert event is not None
        assert event.type == "ping"

    def test_error_event_parsed(self):
        event = parse_anthropic_sse_event(
            "error",
            {"error": {"type": "api_error", "message": "something broke"}},
        )
        assert event is not None
        assert event.type == "error"
        assert event.error.type == "api_error"
        assert event.error.message == "something broke"


# -- Thinking Config --


class TestThinkingConfig:
    def test_budget_tokens_below_minimum_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            AnthropicThinkingConfig(type="enabled", budget_tokens=500)

    def test_budget_tokens_at_minimum_accepted(self):
        config = AnthropicThinkingConfig(type="enabled", budget_tokens=1024)
        assert config.budget_tokens == 1024

    def test_budget_tokens_above_minimum_accepted(self):
        config = AnthropicThinkingConfig(type="enabled", budget_tokens=4096)
        assert config.budget_tokens == 4096

    def test_thinking_enabled_raises_in_translation_mode(self):
        request = AnthropicCreateMessageRequest(
            model="m",
            messages=[AnthropicMessage(role="user", content="Think about this")],
            max_tokens=8192,
            thinking=AnthropicThinkingConfig(type="enabled", budget_tokens=4096),
        )
        with pytest.raises(ValueError, match="extended thinking requires a native Anthropic-compatible provider"):
            anthropic_request_to_openai(request)

    def test_thinking_disabled_allowed_in_translation_mode(self):
        request = AnthropicCreateMessageRequest(
            model="m",
            messages=[AnthropicMessage(role="user", content="Hello")],
            max_tokens=100,
            thinking=AnthropicThinkingConfig(type="disabled"),
        )
        result = anthropic_request_to_openai(request)
        assert result.model == "m"

    def test_thinking_none_allowed_in_translation_mode(self):
        request = AnthropicCreateMessageRequest(
            model="m",
            messages=[AnthropicMessage(role="user", content="Hello")],
            max_tokens=100,
        )
        result = anthropic_request_to_openai(request)
        assert result.model == "m"


# -- Ping Events --


class TestPingEvents:
    async def test_ping_after_message_start_in_text_stream(self):
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta = MagicMock()
        chunk.choices[0].delta.content = "Hi"
        chunk.choices[0].delta.tool_calls = None
        chunk.choices[0].finish_reason = "stop"
        chunk.usage = None

        async def mock_stream():
            yield chunk

        events = []
        async for event in openai_stream_to_anthropic(mock_stream(), "m"):
            events.append(event)

        assert events[0].type == "message_start"
        assert events[1].type == "ping"
        assert events[2].type == "content_block_start"

    async def test_ping_after_content_block_stop_in_text_stream(self):
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta = MagicMock()
        chunk.choices[0].delta.content = "Hi"
        chunk.choices[0].delta.tool_calls = None
        chunk.choices[0].finish_reason = "stop"
        chunk.usage = None

        async def mock_stream():
            yield chunk

        events = []
        async for event in openai_stream_to_anthropic(mock_stream(), "m"):
            events.append(event)

        stop_indices = [i for i, e in enumerate(events) if e.type == "content_block_stop"]
        for idx in stop_indices:
            assert events[idx + 1].type == "ping"

    async def test_ping_after_content_block_stop_in_tool_stream(self):
        tc_delta = MagicMock()
        tc_delta.index = 0
        tc_delta.id = "call_abc"
        tc_delta.function = MagicMock()
        tc_delta.function.name = "search"
        tc_delta.function.arguments = '{"q": "x"}'

        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta = MagicMock()
        chunk.choices[0].delta.content = None
        chunk.choices[0].delta.tool_calls = [tc_delta]
        chunk.choices[0].finish_reason = "tool_calls"
        chunk.usage = None

        async def mock_stream():
            yield chunk

        events = []
        async for event in openai_stream_to_anthropic(mock_stream(), "m"):
            events.append(event)

        stop_indices = [i for i, e in enumerate(events) if e.type == "content_block_stop"]
        assert len(stop_indices) >= 1
        for idx in stop_indices:
            assert events[idx + 1].type == "ping"


# -- Error Stream Events --


class TestErrorStreamEvent:
    async def test_error_event_on_mid_stream_exception(self):
        async def failing_stream():
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta = MagicMock()
            chunk.choices[0].delta.content = "partial"
            chunk.choices[0].delta.tool_calls = None
            chunk.choices[0].finish_reason = None
            chunk.usage = None
            yield chunk
            raise RuntimeError("connection lost")

        events = []
        async for event in openai_stream_to_anthropic(failing_stream(), "m"):
            events.append(event)

        assert events[0].type == "message_start"
        assert events[-1].type == "error"
        assert events[-1].error.type == "api_error"

    async def test_error_event_terminates_stream(self):
        async def failing_stream():
            raise RuntimeError("immediate failure")
            yield  # unreachable

        events = []
        async for event in openai_stream_to_anthropic(failing_stream(), "m"):
            events.append(event)

        assert events[0].type == "message_start"
        assert events[1].type == "ping"
        assert events[2].type == "error"
        assert len(events) == 3
