# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for the VertexAI OpenAI ↔ google-genai conversion module.

All google-genai types are mocked via SimpleNamespace — no SDK installation required.
"""

import base64
from types import SimpleNamespace
from typing import Any, cast

import pytest

from ogx.providers.remote.inference.vertexai import converters as vertexai_converters
from ogx.providers.remote.inference.vertexai.converters import (
    _convert_user_message,
    _extract_text_content,
    convert_deprecated_function_call_to_tool_choice,
    convert_deprecated_functions_to_tools,
    convert_finish_reason,
    convert_gemini_response_to_openai,
    convert_gemini_stream_chunk_to_openai,
    convert_openai_messages_to_gemini,
    convert_openai_tools_to_gemini,
    convert_response_format,
)

_convert_image_url_part = getattr(vertexai_converters, "_convert_image_url_part", None)

convert_gemini_response_to_openai = cast(Any, convert_gemini_response_to_openai)
convert_gemini_stream_chunk_to_openai = cast(Any, convert_gemini_stream_chunk_to_openai)

FAKE_IMAGE_BYTES = b"fake image bytes"
FAKE_IMAGE_B64 = base64.b64encode(FAKE_IMAGE_BYTES).decode()


@pytest.fixture
def weather_tool_call() -> dict[str, Any]:
    """Provide a reusable weather tool-call payload."""
    return {
        "id": "call_weather",
        "type": "function",
        "function": {
            "name": "get_weather",
            "arguments": '{"location": "Boston"}',
        },
    }


def _make_text_part(text: str) -> Any:
    """Build ext part."""
    return SimpleNamespace(text=text, thought=None, function_call=None)


def _make_thought_part(thought_text: str) -> Any:
    """Build hought part."""
    return SimpleNamespace(text=thought_text, thought=True, function_call=None)


def _make_function_call_part(name: str, args: dict) -> Any:
    """Build unction call part."""
    return SimpleNamespace(
        text=None,
        function_call=SimpleNamespace(name=name, args=args),
    )


def _make_candidate(
    parts: list | None = None,
    finish_reason: str | None = "STOP",
    index: int = 0,
    logprobs_result: Any = None,
) -> Any:
    """Build andidate."""
    content = SimpleNamespace(parts=parts or [])
    return SimpleNamespace(
        content=content,
        finish_reason=finish_reason,
        index=index,
        logprobs_result=logprobs_result,
    )


def _make_response(
    candidates: list | None = None,
    prompt_token_count: int = 10,
    candidates_token_count: int = 20,
    total_token_count: int = 30,
) -> Any:
    """Build esponse."""
    usage = SimpleNamespace(
        prompt_token_count=prompt_token_count,
        candidates_token_count=candidates_token_count,
        total_token_count=total_token_count,
    )
    return SimpleNamespace(candidates=candidates, usage_metadata=usage)


def _make_function_call_response() -> Any:
    """Build unction call response."""
    return _make_response(
        candidates=[
            _make_candidate(
                parts=[_make_function_call_part("get_weather", {"location": "NYC"})],
                finish_reason="STOP",
            )
        ]
    )


def _make_logprob_candidate(
    token: str,
    log_probability: float,
    token_id: int | None = None,
) -> Any:
    """Build ogprob candidate."""
    return SimpleNamespace(token=token, log_probability=log_probability, token_id=token_id)


def _make_top_candidates_entry(candidates: list) -> Any:
    """Build op candidates entry."""
    return SimpleNamespace(candidates=candidates)


def _make_logprobs_result(
    chosen: list | None = None,
    top: list | None = None,
) -> Any:
    """Build ogprobs result."""
    return SimpleNamespace(
        chosen_candidates=chosen or [],
        top_candidates=top or [],
    )


class TestConvertFinishReason:
    @pytest.mark.parametrize(
        "input_reason,expected",
        [
            ("STOP", "stop"),
            ("MAX_TOKENS", "length"),
            ("SAFETY", "content_filter"),
            ("RECITATION", "content_filter"),
            ("LANGUAGE", "content_filter"),
            ("BLOCKLIST", "content_filter"),
            ("PROHIBITED_CONTENT", "content_filter"),
            ("SPII", "content_filter"),
            ("MALFORMED_FUNCTION_CALL", "stop"),
            ("OTHER", "stop"),
        ],
    )
    def test_standard_mappings(self, input_reason, expected):
        """Test that standard mappings."""
        assert convert_finish_reason(input_reason) == expected

    def test_none(self):
        """Test that none."""
        assert convert_finish_reason(None) == "stop"

    def test_unknown_value(self):
        """Test that unknown value."""
        assert convert_finish_reason("TOTALLY_NEW_REASON") == "stop"

    @pytest.mark.parametrize("input_reason", ["stop", "Stop"])
    def test_case_insensitive(self, input_reason):
        # FinishReason values from SDK are uppercase but let's be defensive
        """Test that case insensitive."""
        assert convert_finish_reason(input_reason) == "stop"


class TestConvertResponseFormat:
    @pytest.mark.parametrize(
        "response_format,expected",
        [
            (None, {}),
            ({"type": "text"}, {}),
            ({"type": "json_object"}, {"response_mime_type": "application/json"}),
            (
                {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "test",
                        "schema": {"type": "object", "properties": {"name": {"type": "string"}}},
                    },
                },
                {
                    "response_mime_type": "application/json",
                    "response_schema": {"type": "object", "properties": {"name": {"type": "string"}}},
                },
            ),
            ({"type": "json_schema", "json_schema": {"name": "test"}}, {"response_mime_type": "application/json"}),
            ({"type": "unknown"}, {}),
        ],
    )
    def test_convert_response_format(self, response_format, expected):
        """Test that convert response format."""
        assert convert_response_format(response_format) == expected


class TestExtractTextContent:
    @pytest.mark.parametrize(
        "input_content,expected",
        [
            ("hello", "hello"),
            (None, ""),
            ([], ""),
            (
                [
                    {"type": "text", "text": "hello "},
                    {"type": "text", "text": "world"},
                ],
                "hello world",
            ),
            (
                [
                    {"type": "text", "text": "hello"},
                    {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
                ],
                "hello",
            ),
        ],
    )
    def test_extract_text_content(self, input_content, expected):
        """Test that extract text content."""
        assert _extract_text_content(input_content) == expected


class TestConvertImageUrlPart:
    def _convert_part(self, part: dict[str, Any]) -> dict[str, Any] | None:
        """Convert art."""
        assert _convert_image_url_part is not None
        return _convert_image_url_part(part)

    @pytest.mark.parametrize(
        "fmt",
        [
            pytest.param("jpeg", id="jpeg"),
            pytest.param("png", id="png"),
            pytest.param("gif", id="gif"),
            pytest.param("webp", id="webp"),
        ],
    )
    def test_data_uri_to_inline_data(self, fmt):
        """Test that data uri to inline data."""
        part = {"type": "image_url", "image_url": {"url": f"data:image/{fmt};base64,{FAKE_IMAGE_B64}"}}
        result = self._convert_part(part)
        assert result == {"inline_data": {"data": FAKE_IMAGE_BYTES, "mime_type": f"image/{fmt}"}}

    def test_image_detail_parameter_ignored(self):
        """Test that image detail parameter ignored."""
        part = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{FAKE_IMAGE_B64}",
                "detail": "high",
            },
        }
        result = self._convert_part(part)
        assert result == {"inline_data": {"data": FAKE_IMAGE_BYTES, "mime_type": "image/jpeg"}}

    @pytest.mark.parametrize(
        "url",
        [
            pytest.param("file:///path/to/img.png", id="file_scheme"),
            pytest.param("ftp://example.com/img.png", id="ftp_scheme"),
        ],
    )
    def test_unsupported_url_scheme_returns_none(self, url):
        """Test that unsupported url scheme returns none."""
        part = {"type": "image_url", "image_url": {"url": url}}
        assert self._convert_part(part) is None


class TestConvertUserMessageWithImages:
    @pytest.mark.parametrize(
        "message,expected_parts",
        [
            pytest.param({"role": "user", "content": "hello"}, [{"text": "hello"}], id="text_string"),
            pytest.param(
                {"role": "user", "content": [{"type": "text", "text": "hello"}]},
                [{"text": "hello"}],
                id="text_list",
            ),
            pytest.param(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{FAKE_IMAGE_B64}"},
                        }
                    ],
                },
                [{"inline_data": {"data": FAKE_IMAGE_BYTES, "mime_type": "image/jpeg"}}],
                id="single_image",
            ),
            pytest.param(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{FAKE_IMAGE_B64}"},
                        },
                    ],
                },
                [
                    {"text": "hello"},
                    {"inline_data": {"data": FAKE_IMAGE_BYTES, "mime_type": "image/jpeg"}},
                ],
                id="text_then_image",
            ),
            pytest.param(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{FAKE_IMAGE_B64}"},
                        },
                        {"type": "text", "text": "hello"},
                    ],
                },
                [
                    {"inline_data": {"data": FAKE_IMAGE_BYTES, "mime_type": "image/jpeg"}},
                    {"text": "hello"},
                ],
                id="image_then_text",
            ),
            pytest.param(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "before"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{FAKE_IMAGE_B64}"},
                        },
                        {"type": "text", "text": "after"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{FAKE_IMAGE_B64}"},
                        },
                    ],
                },
                [
                    {"text": "before"},
                    {"inline_data": {"data": FAKE_IMAGE_BYTES, "mime_type": "image/jpeg"}},
                    {"text": "after"},
                    {"inline_data": {"data": FAKE_IMAGE_BYTES, "mime_type": "image/png"}},
                ],
                id="interleaved_text_images",
            ),
        ],
    )
    def test_user_message_conversion(self, message, expected_parts):
        """Test that user message conversion."""
        assert _convert_user_message(message) == {"role": "user", "parts": expected_parts}

    def test_image_only_no_text(self):
        """Test that image only no text."""
        message = {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{FAKE_IMAGE_B64}"},
                }
            ],
        }
        assert _convert_user_message(message) == {
            "role": "user",
            "parts": [{"inline_data": {"data": FAKE_IMAGE_BYTES, "mime_type": "image/jpeg"}}],
        }

    def test_unsupported_url_scheme_skipped(self):
        """Test that unsupported url scheme skipped."""
        message = {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "file:///path/to/img.png"},
                }
            ],
        }
        assert _convert_user_message(message) == {"role": "user", "parts": []}

    def test_user_message_with_image_in_full_conversion(self):
        """Test that user message with image in full conversion."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "analyze this image"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{FAKE_IMAGE_B64}"},
                    },
                ],
            }
        ]
        system, contents = convert_openai_messages_to_gemini(messages)
        assert system is None
        assert contents == [
            {
                "role": "user",
                "parts": [
                    {"text": "analyze this image"},
                    {"inline_data": {"data": FAKE_IMAGE_BYTES, "mime_type": "image/jpeg"}},
                ],
            }
        ]


class TestConvertOpenAIMessagesToGemini:
    @pytest.mark.parametrize(
        "messages,expected_system",
        [
            ([{"role": "user", "content": "Hello"}], None),
            (
                [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hi"},
                ],
                "You are helpful.",
            ),
            (
                [
                    {"role": "system", "content": "Rule 1."},
                    {"role": "system", "content": "Rule 2."},
                    {"role": "user", "content": "Hi"},
                ],
                "Rule 1.\nRule 2.",
            ),
            (
                [
                    {"role": "developer", "content": "Be concise."},
                    {"role": "user", "content": "Hi"},
                ],
                "Be concise.",
            ),
        ],
    )
    def test_system_and_user_message_conversion(self, messages, expected_system):
        """Test that system and user message conversion."""
        system, contents = convert_openai_messages_to_gemini(messages)
        assert system == expected_system
        assert len(contents) == 1
        assert contents[0]["role"] == "user"

    def test_assistant_message(self):
        """Test that assistant message."""
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello there!"},
        ]
        system, contents = convert_openai_messages_to_gemini(messages)
        assert len(contents) == 2
        assert contents[1]["role"] == "model"
        assert contents[1]["parts"] == [{"text": "Hello there!"}]

    def test_assistant_with_tool_calls(self, weather_tool_call):
        """Test that assistant with tool calls."""
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{**weather_tool_call, "id": "call_123"}],
            },
        ]
        system, contents = convert_openai_messages_to_gemini(messages)
        assert len(contents) == 2
        model_msg = contents[1]
        assert model_msg["role"] == "model"
        assert len(model_msg["parts"]) == 1
        fc = model_msg["parts"][0]["function_call"]
        assert fc["name"] == "get_weather"
        assert fc["args"] == {"location": "Boston"}

    def test_tool_response_message(self, weather_tool_call):
        """Test that tool response message."""
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        **weather_tool_call,
                        "id": "call_abc",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_abc",
                "content": '{"temperature": 72}',
            },
        ]
        system, contents = convert_openai_messages_to_gemini(messages)
        assert len(contents) == 3
        tool_msg = contents[2]
        assert tool_msg["role"] == "user"
        fr = tool_msg["parts"][0]["function_response"]
        assert fr["name"] == "get_weather"
        assert fr["response"] == {"temperature": 72}

    def test_tool_response_non_json(self):
        """Test that tool response non json."""
        messages = [
            {"role": "user", "content": "Hi"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_xyz",
                        "type": "function",
                        "function": {"name": "some_tool", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_xyz",
                "content": "plain text result",
            },
        ]
        system, contents = convert_openai_messages_to_gemini(messages)
        tool_msg = contents[2]
        fr = tool_msg["parts"][0]["function_response"]
        assert fr["response"] == {"result": "plain text result"}

    def test_tool_response_json_array_wrapped_in_dict(self):
        """Test that tool response json array wrapped in dict."""
        messages = [
            {"role": "user", "content": "Hi"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_arr",
                        "type": "function",
                        "function": {"name": "list_items", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_arr",
                "content": "[1, 2, 3]",
            },
        ]
        system, contents = convert_openai_messages_to_gemini(messages)
        tool_msg = contents[2]
        fr = tool_msg["parts"][0]["function_response"]
        assert fr["response"] == {"result": [1, 2, 3]}

    def test_empty_messages(self):
        """Test that empty messages."""
        system, contents = convert_openai_messages_to_gemini([])
        assert system is None
        assert contents == []

    def test_assistant_with_text_and_tool_calls(self):
        """Test that assistant with text and tool calls."""
        messages = [
            {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "search", "arguments": '{"q": "test"}'},
                    }
                ],
            }
        ]
        system, contents = convert_openai_messages_to_gemini(messages)
        model_msg = contents[0]
        assert model_msg["role"] == "model"
        # Should have both text and function_call parts
        assert len(model_msg["parts"]) == 2
        assert model_msg["parts"][0] == {"text": "Let me check."}
        assert "function_call" in model_msg["parts"][1]

    def test_tool_call_id_not_found(self):
        """When tool_call_id doesn't match any assistant message, use 'unknown' as name."""
        messages = [
            {
                "role": "tool",
                "tool_call_id": "nonexistent",
                "content": "result",
            }
        ]
        system, contents = convert_openai_messages_to_gemini(messages)
        fr = contents[0]["parts"][0]["function_response"]
        assert fr["name"] == "unknown"


class TestConvertOpenAIToolsToGemini:
    def test_single_function_tool(self):
        """Test that single function tool."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                },
            }
        ]
        result = convert_openai_tools_to_gemini(tools)
        assert result is not None
        assert len(result) == 1
        fds = result[0]["function_declarations"]
        assert len(fds) == 1
        assert fds[0]["name"] == "get_weather"
        assert fds[0]["description"] == "Get current weather"
        assert "properties" in fds[0]["parameters_json_schema"]

    def test_multiple_tools(self):
        """Test that multiple tools."""
        tools = [
            {"type": "function", "function": {"name": "tool_a", "description": "A"}},
            {"type": "function", "function": {"name": "tool_b", "description": "B"}},
        ]
        result = convert_openai_tools_to_gemini(tools)
        assert result is not None
        fds = result[0]["function_declarations"]
        assert len(fds) == 2
        assert fds[0]["name"] == "tool_a"
        assert fds[1]["name"] == "tool_b"

    @pytest.mark.parametrize("tools", [None, [], [{"type": "code_interpreter", "other": "data"}]])
    def test_no_convertible_tools_returns_none(self, tools):
        """Test that no convertible tools returns none."""
        assert convert_openai_tools_to_gemini(tools) is None

    def test_tool_without_parameters(self):
        """Test that tool without parameters."""
        tools = [{"type": "function", "function": {"name": "noop", "description": "Does nothing"}}]
        result = convert_openai_tools_to_gemini(tools)
        assert result is not None
        fd = result[0]["function_declarations"][0]
        assert "parameters_json_schema" not in fd


class TestConvertDeprecatedFunctions:
    def test_single_function_converts_to_tool(self):
        """Test that single function converts to tool."""
        functions = [{"name": "get_weather", "description": "Get weather", "parameters": {"type": "object"}}]
        result = convert_deprecated_functions_to_tools(functions)
        assert result == [
            {
                "type": "function",
                "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object"}},
            }
        ]

    def test_multiple_functions(self):
        """Test that multiple functions."""
        functions = [
            {"name": "func_a", "description": "A"},
            {"name": "func_b", "description": "B"},
        ]
        result = convert_deprecated_functions_to_tools(functions)
        assert len(result) == 2
        assert result[0] == {"type": "function", "function": {"name": "func_a", "description": "A"}}
        assert result[1] == {"type": "function", "function": {"name": "func_b", "description": "B"}}

    def test_empty_functions_returns_empty(self):
        """Test that empty functions returns empty."""
        assert convert_deprecated_functions_to_tools([]) == []


class TestConvertDeprecatedFunctionCall:
    def test_auto_passthrough(self):
        """Test that auto passthrough."""
        assert convert_deprecated_function_call_to_tool_choice("auto") == "auto"

    def test_none_passthrough(self):
        """Test that none passthrough."""
        assert convert_deprecated_function_call_to_tool_choice("none") == "none"

    def test_named_function_converts_to_tool_choice(self):
        """Test that named function converts to tool choice."""
        result = convert_deprecated_function_call_to_tool_choice({"name": "get_weather"})
        assert result == {"type": "function", "function": {"name": "get_weather"}}

    def test_unknown_string_passthrough(self):
        """Any unrecognised string passes through unchanged (forward-compat)."""
        assert convert_deprecated_function_call_to_tool_choice("required") == "required"

    def test_dict_without_name_passthrough(self):
        """A dict without a 'name' key is returned as-is (fallback path)."""
        payload: dict[str, Any] = {"mode": "auto"}
        assert convert_deprecated_function_call_to_tool_choice(payload) == payload
