# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for the BuiltinMessagesImpl translation logic."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from ogx.providers.inline.messages.config import MessagesConfig
from ogx.providers.inline.messages.impl import BuiltinMessagesImpl
from ogx_api.messages.models import (
    AnthropicCreateMessageRequest,
    AnthropicMessage,
    AnthropicTextBlock,
    AnthropicToolDef,
    AnthropicToolResultBlock,
    AnthropicToolUseBlock,
)


def _msg_to_dict(msg):
    """Convert a Pydantic message model to dict for easy assertion."""
    if hasattr(msg, "model_dump"):
        return msg.model_dump(exclude_none=True)
    return dict(msg)


@pytest.fixture
def impl():
    mock_inference = AsyncMock()
    return BuiltinMessagesImpl(config=MessagesConfig(), inference_api=mock_inference)


class TestRequestTranslation:
    def test_simple_text_message(self, impl):
        request = AnthropicCreateMessageRequest(
            model="claude-sonnet-4-20250514",
            messages=[AnthropicMessage(role="user", content="Hello")],
            max_tokens=100,
        )
        result = impl._anthropic_to_openai(request)

        assert result.model == "claude-sonnet-4-20250514"
        assert result.max_tokens == 100
        assert len(result.messages) == 1
        m = _msg_to_dict(result.messages[0])
        assert m["role"] == "user"
        assert m["content"] == "Hello"

    def test_system_string(self, impl):
        request = AnthropicCreateMessageRequest(
            model="m",
            messages=[AnthropicMessage(role="user", content="Hi")],
            max_tokens=100,
            system="You are helpful.",
        )
        result = impl._anthropic_to_openai(request)

        m0 = _msg_to_dict(result.messages[0])
        m1 = _msg_to_dict(result.messages[1])
        assert m0["role"] == "system"
        assert m0["content"] == "You are helpful."
        assert m1["role"] == "user"

    def test_system_text_blocks(self, impl):
        request = AnthropicCreateMessageRequest(
            model="m",
            messages=[AnthropicMessage(role="user", content="Hi")],
            max_tokens=100,
            system=[
                AnthropicTextBlock(text="Line 1."),
                AnthropicTextBlock(text="Line 2."),
            ],
        )
        result = impl._anthropic_to_openai(request)

        m0 = _msg_to_dict(result.messages[0])
        assert m0["role"] == "system"
        assert m0["content"] == "Line 1.\nLine 2."

    def test_tool_definitions(self, impl):
        request = AnthropicCreateMessageRequest(
            model="m",
            messages=[AnthropicMessage(role="user", content="Hi")],
            max_tokens=100,
            tools=[
                AnthropicToolDef(
                    name="get_weather",
                    description="Get weather",
                    input_schema={"type": "object", "properties": {"location": {"type": "string"}}},
                ),
            ],
        )
        result = impl._anthropic_to_openai(request)

        assert len(result.tools) == 1
        tool = result.tools[0]
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "get_weather"
        assert tool["function"]["parameters"]["type"] == "object"

    def test_tool_choice_any(self, impl):
        assert impl._convert_tool_choice_to_openai("any") == "required"

    def test_tool_choice_none(self, impl):
        assert impl._convert_tool_choice_to_openai("none") == "none"

    def test_tool_choice_auto(self, impl):
        assert impl._convert_tool_choice_to_openai("auto") == "auto"

    def test_tool_choice_specific(self, impl):
        result = impl._convert_tool_choice_to_openai({"type": "tool", "name": "get_weather"})
        assert result == {"type": "function", "function": {"name": "get_weather"}}

    def test_stop_sequences(self, impl):
        request = AnthropicCreateMessageRequest(
            model="m",
            messages=[AnthropicMessage(role="user", content="Hi")],
            max_tokens=100,
            stop_sequences=["STOP", "END"],
        )
        result = impl._anthropic_to_openai(request)
        assert result.stop == ["STOP", "END"]

    def test_tool_use_in_assistant_message(self, impl):
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
        result = impl._anthropic_to_openai(request)

        msg = _msg_to_dict(result.messages[0])
        assert msg["role"] == "assistant"
        assert msg["content"] == "Let me check the weather."
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["id"] == "toolu_123"
        assert msg["tool_calls"][0]["function"]["name"] == "get_weather"
        assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"location": "SF"}

    def test_tool_result_in_user_message(self, impl):
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
        result = impl._anthropic_to_openai(request)

        msg = _msg_to_dict(result.messages[0])
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "toolu_123"
        assert msg["content"] == "72F and sunny"

    def test_top_k_passed_as_extra(self, impl):
        request = AnthropicCreateMessageRequest(
            model="m",
            messages=[AnthropicMessage(role="user", content="Hi")],
            max_tokens=100,
            top_k=40,
        )
        result = impl._anthropic_to_openai(request)
        assert result.model_extra.get("top_k") == 40


class TestResponseTranslation:
    def test_simple_text_response(self, impl):
        openai_resp = MagicMock()
        openai_resp.choices = [MagicMock()]
        openai_resp.choices[0].message = MagicMock()
        openai_resp.choices[0].message.content = "Hello!"
        openai_resp.choices[0].message.tool_calls = None
        openai_resp.choices[0].finish_reason = "stop"
        openai_resp.usage = MagicMock()
        openai_resp.usage.prompt_tokens = 10
        openai_resp.usage.completion_tokens = 5

        result = impl._openai_to_anthropic(openai_resp, "claude-sonnet-4-20250514")

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

    def test_tool_call_response(self, impl):
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

        result = impl._openai_to_anthropic(openai_resp, "m")

        assert result.stop_reason == "tool_use"
        assert len(result.content) == 1
        assert result.content[0].type == "tool_use"
        assert result.content[0].name == "get_weather"
        assert result.content[0].input == {"location": "SF"}

    def test_length_stop_reason(self, impl):
        openai_resp = MagicMock()
        openai_resp.choices = [MagicMock()]
        openai_resp.choices[0].message = MagicMock()
        openai_resp.choices[0].message.content = "truncated"
        openai_resp.choices[0].message.tool_calls = None
        openai_resp.choices[0].finish_reason = "length"
        openai_resp.usage = MagicMock()
        openai_resp.usage.prompt_tokens = 5
        openai_resp.usage.completion_tokens = 100

        result = impl._openai_to_anthropic(openai_resp, "m")
        assert result.stop_reason == "max_tokens"

    def test_cache_metrics_mapping(self, impl):
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

        result = impl._openai_to_anthropic(openai_resp, "m")
        assert result.usage.input_tokens == 100
        assert result.usage.output_tokens == 50
        assert result.usage.cache_read_input_tokens == 75
        assert result.usage.cache_creation_input_tokens is None

    def test_cache_metrics_missing(self, impl):
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

        result = impl._openai_to_anthropic(openai_resp, "m")
        assert result.usage.input_tokens == 100
        assert result.usage.output_tokens == 50
        assert result.usage.cache_read_input_tokens is None
        assert result.usage.cache_creation_input_tokens is None


class TestStreamingTranslation:
    async def test_text_streaming(self, impl):
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
        async for event in impl._stream_openai_to_anthropic(mock_stream(), "m"):
            events.append(event)

        assert events[0].type == "message_start"
        assert events[1].type == "content_block_start"
        assert events[1].content_block.type == "text"
        assert events[2].type == "content_block_delta"
        assert events[2].delta.text == "Hello"
        assert events[3].type == "content_block_delta"
        assert events[3].delta.text == " world"
        assert events[4].type == "content_block_delta"
        assert events[4].delta.text == "!"
        assert events[5].type == "content_block_stop"
        assert events[6].type == "message_delta"
        assert events[6].delta.stop_reason == "end_turn"
        assert events[7].type == "message_stop"

    async def test_tool_call_streaming(self, impl):
        chunks = []

        # Tool call start
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

        # Tool call arguments
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
        async for event in impl._stream_openai_to_anthropic(mock_stream(), "m"):
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
