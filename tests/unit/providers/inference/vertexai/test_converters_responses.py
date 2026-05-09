# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for the VertexAI OpenAI ↔ google-genai conversion module.

All google-genai types are mocked via SimpleNamespace — no SDK installation required.
"""

import base64
import json
from types import SimpleNamespace
from typing import Any, cast

import pytest

from ogx.providers.remote.inference.vertexai import converters as vertexai_converters
from ogx.providers.remote.inference.vertexai.converters import (
    _extract_logprobs,
    convert_gemini_response_to_openai,
    convert_gemini_stream_chunk_to_openai,
    extract_usage,
    generate_completion_id,
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


class TestConvertGeminiResponseToOpenAI:
    def test_simple_text_response(self):
        """Test that simple text response."""
        response = _make_response(candidates=[_make_candidate(parts=[_make_text_part("Hello!")])])
        result = convert_gemini_response_to_openai(response, "gemini-2.5-flash")

        assert result.object == "chat.completion"
        assert result.model == "gemini-2.5-flash"
        assert result.id.startswith("chatcmpl-")
        assert len(result.choices) == 1
        assert result.choices[0].message.role == "assistant"
        assert result.choices[0].message.content == "Hello!"
        assert result.choices[0].finish_reason == "stop"
        assert result.choices[0].message.tool_calls is None

    def test_function_call_response_sets_finish_reason_and_type(self):
        """Test that function call response sets finish reason and type."""
        response = _make_function_call_response()
        result = convert_gemini_response_to_openai(response, "gemini-2.5-flash")

        assert result.choices[0].finish_reason == "tool_calls"
        tool_calls = result.choices[0].message.tool_calls
        assert tool_calls is not None
        assert len(tool_calls) == 1
        tc = tool_calls[0]
        assert tc.type == "function"

    def test_function_call_response_sets_function_payload(self):
        """Test that function call response sets function payload."""
        response = _make_function_call_response()
        result = convert_gemini_response_to_openai(response, "gemini-2.5-flash")

        tool_calls = result.choices[0].message.tool_calls
        assert tool_calls is not None
        tc = tool_calls[0]
        assert tc.function is not None
        assert tc.function.name == "get_weather"
        assert tc.function.arguments is not None
        assert json.loads(tc.function.arguments) == {"location": "NYC"}
        assert tc.id is not None
        assert tc.id.startswith("call_")

    def test_multi_part_response(self):
        """Test that multi part response."""
        response = _make_response(
            candidates=[_make_candidate(parts=[_make_text_part("Part 1 "), _make_text_part("Part 2")])]
        )
        result = convert_gemini_response_to_openai(response, "gemini-2.5-flash")
        assert result.choices[0].message.content == "Part 1 Part 2"

    def test_text_and_function_call_response(self):
        """Test that text and function call response."""
        response = _make_response(
            candidates=[
                _make_candidate(
                    parts=[
                        _make_text_part("Let me check."),
                        _make_function_call_part("search", {"q": "test"}),
                    ]
                )
            ]
        )
        result = convert_gemini_response_to_openai(response, "gemini-2.5-flash")
        assert result.choices[0].message.content == "Let me check."
        assert result.choices[0].finish_reason == "tool_calls"
        tc_list = result.choices[0].message.tool_calls
        assert tc_list is not None
        assert len(tc_list) == 1

    def test_no_candidates_safety_filtered(self):
        """Test that no candidates safety filtered."""
        response = _make_response(candidates=[])
        result = convert_gemini_response_to_openai(response, "gemini-2.5-flash")
        assert len(result.choices) == 1
        assert result.choices[0].finish_reason == "content_filter"
        assert result.choices[0].message.content is None

    def test_none_candidates(self):
        """Test that none candidates."""
        response: Any = SimpleNamespace(candidates=None, usage_metadata=None)
        result = convert_gemini_response_to_openai(response, "gemini-2.5-flash")
        assert len(result.choices) == 1
        assert result.choices[0].finish_reason == "content_filter"

    def test_usage_metadata(self):
        """Test that usage metadata."""
        response = _make_response(
            candidates=[_make_candidate(parts=[_make_text_part("hi")])],
            prompt_token_count=15,
            candidates_token_count=5,
            total_token_count=20,
        )
        result = convert_gemini_response_to_openai(response, "gemini-2.5-flash")
        assert result.usage is not None
        assert result.usage.prompt_tokens == 15
        assert result.usage.completion_tokens == 5
        assert result.usage.total_tokens == 20

    def test_no_usage_metadata(self):
        """Test that no usage metadata."""
        response: Any = SimpleNamespace(
            candidates=[_make_candidate(parts=[_make_text_part("hi")])],
            usage_metadata=None,
        )
        result = convert_gemini_response_to_openai(response, "model")
        assert result.usage is None

    def test_empty_parts(self):
        """Test that empty parts."""
        response = _make_response(candidates=[_make_candidate(parts=[])])
        result = convert_gemini_response_to_openai(response, "model")
        assert result.choices[0].message.content is None
        assert result.choices[0].message.tool_calls is None

    def test_candidate_with_none_content(self):
        """Test that candidate with none content."""
        candidate: Any = SimpleNamespace(content=None, finish_reason="STOP", index=0)
        response = _make_response(candidates=[candidate])
        result = convert_gemini_response_to_openai(response, "model")
        assert result.choices[0].message.content is None

    def test_multiple_function_calls(self):
        """Test that multiple function calls."""
        response = _make_response(
            candidates=[
                _make_candidate(
                    parts=[
                        _make_function_call_part("func_a", {"x": 1}),
                        _make_function_call_part("func_b", {"y": 2}),
                    ]
                )
            ]
        )
        result = convert_gemini_response_to_openai(response, "model")
        tool_calls = result.choices[0].message.tool_calls
        assert tool_calls is not None
        assert len(tool_calls) == 2
        assert tool_calls[0].index == 0
        assert tool_calls[1].index == 1
        assert tool_calls[0].function is not None
        assert tool_calls[0].function.name == "func_a"
        assert tool_calls[1].function is not None
        assert tool_calls[1].function.name == "func_b"


class TestExtractUsage:
    def test_extract_usage_with_cached_tokens(self):
        """prompt_tokens_details populated when cached_content_token_count is present."""
        response = SimpleNamespace(
            usage_metadata=SimpleNamespace(
                prompt_token_count=100,
                candidates_token_count=50,
                total_token_count=150,
                cached_content_token_count=30,
            )
        )
        usage = extract_usage(cast(Any, response))
        assert usage is not None
        assert usage.prompt_tokens_details is not None
        assert usage.prompt_tokens_details.cached_tokens == 30
        assert usage.completion_tokens_details is None

    def test_extract_usage_with_reasoning_tokens(self):
        """completion_tokens_details populated when thoughts_token_count is present."""
        response = SimpleNamespace(
            usage_metadata=SimpleNamespace(
                prompt_token_count=100,
                candidates_token_count=50,
                total_token_count=150,
                thoughts_token_count=20,
            )
        )
        usage = extract_usage(cast(Any, response))
        assert usage is not None
        assert usage.completion_tokens_details is not None
        assert usage.completion_tokens_details.reasoning_tokens == 20
        assert usage.prompt_tokens_details is None

    def test_extract_usage_no_details_when_fields_absent(self):
        """Both details are None when usage_metadata lacks optional fields."""
        response = SimpleNamespace(
            usage_metadata=SimpleNamespace(
                prompt_token_count=100,
                candidates_token_count=50,
                total_token_count=150,
            )
        )
        usage = extract_usage(cast(Any, response))
        assert usage is not None
        assert usage.prompt_tokens_details is None
        assert usage.completion_tokens_details is None

    def test_extract_usage_with_both_details(self):
        """Both details populated when both optional fields present."""
        response = SimpleNamespace(
            usage_metadata=SimpleNamespace(
                prompt_token_count=100,
                candidates_token_count=50,
                total_token_count=150,
                cached_content_token_count=30,
                thoughts_token_count=20,
            )
        )
        usage = extract_usage(cast(Any, response))
        assert usage is not None
        assert usage.prompt_tokens_details is not None
        assert usage.completion_tokens_details is not None
        assert usage.prompt_tokens_details.cached_tokens == 30
        assert usage.completion_tokens_details.reasoning_tokens == 20


class TestConvertGeminiStreamChunkToOpenAI:
    def test_first_chunk_has_role(self):
        """Test that first chunk has role."""
        chunk = _make_response(candidates=[_make_candidate(parts=[_make_text_part("Hel")], finish_reason=None)])
        result = convert_gemini_stream_chunk_to_openai(
            chunk, "gemini-2.5-flash", "chatcmpl-test123", is_first_chunk=True
        )
        assert result.object == "chat.completion.chunk"
        assert result.id == "chatcmpl-test123"
        assert result.choices[0].delta.role == "assistant"
        assert result.choices[0].delta.content == "Hel"
        assert result.choices[0].finish_reason is None

    def test_subsequent_chunk_no_role(self):
        """Test that subsequent chunk no role."""
        chunk = _make_response(candidates=[_make_candidate(parts=[_make_text_part("lo")], finish_reason=None)])
        result = convert_gemini_stream_chunk_to_openai(chunk, "model", "chatcmpl-test123", is_first_chunk=False)
        assert result.choices[0].delta.role is None
        assert result.choices[0].delta.content == "lo"

    def test_final_chunk_with_finish_reason(self):
        """Test that final chunk with finish reason."""
        chunk = _make_response(candidates=[_make_candidate(parts=[_make_text_part("!")], finish_reason="STOP")])
        result = convert_gemini_stream_chunk_to_openai(chunk, "model", "chatcmpl-test123", is_first_chunk=False)
        assert result.choices[0].finish_reason == "stop"

    def test_chunk_with_tool_call(self):
        """Test that chunk with tool call."""
        chunk = _make_response(
            candidates=[
                _make_candidate(
                    parts=[_make_function_call_part("search", {"q": "test"})],
                    finish_reason=None,
                )
            ]
        )
        result = convert_gemini_stream_chunk_to_openai(chunk, "model", "chatcmpl-tc", is_first_chunk=True)
        assert result.choices[0].delta.tool_calls is not None
        assert len(result.choices[0].delta.tool_calls) == 1
        assert result.choices[0].finish_reason == "tool_calls"

    def test_empty_chunk_no_candidates(self):
        """Test that empty chunk no candidates."""
        chunk = _make_response(candidates=[])
        result = convert_gemini_stream_chunk_to_openai(chunk, "model", "chatcmpl-empty", is_first_chunk=True)
        assert len(result.choices) == 1
        assert result.choices[0].delta.role == "assistant"
        assert result.choices[0].delta.content is None
        assert result.choices[0].finish_reason is None

    def test_stream_usage(self):
        """Test that stream usage."""
        chunk = _make_response(
            candidates=[_make_candidate(parts=[_make_text_part("done")], finish_reason="STOP")],
            prompt_token_count=10,
            candidates_token_count=5,
            total_token_count=15,
        )
        result = convert_gemini_stream_chunk_to_openai(chunk, "model", "chatcmpl-u", is_first_chunk=False)
        assert result.usage is not None
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5

    def test_safety_filtered_chunk(self):
        """Test that safety filtered chunk."""
        chunk = _make_response(candidates=[_make_candidate(parts=[], finish_reason="SAFETY")])
        result = convert_gemini_stream_chunk_to_openai(chunk, "model", "chatcmpl-sf", is_first_chunk=False)
        assert result.choices[0].finish_reason == "content_filter"


class TestGenerateCompletionId:
    def test_format(self):
        """Test that format."""
        cid = generate_completion_id()
        assert cid.startswith("chatcmpl-")
        # UUID part should be valid
        uuid_part = cid[len("chatcmpl-") :]
        assert len(uuid_part) == 36  # standard UUID string length

    def test_uniqueness(self):
        """Test that uniqueness."""
        ids = {generate_completion_id() for _ in range(100)}
        assert len(ids) == 100


class TestExtractLogprobs:
    def test_logprobs_none_when_not_present(self):
        """Candidate with no logprobs_result returns None."""
        candidate = _make_candidate(parts=[_make_text_part("hello")])
        result = _extract_logprobs(candidate)
        assert result is None

    def test_logprobs_extracted_from_response(self):
        """Full logprobs extraction from a non-streaming response."""
        chosen = [_make_logprob_candidate("hello", -0.5)]
        logprobs_result = _make_logprobs_result(chosen=chosen)
        candidate = _make_candidate(
            parts=[_make_text_part("hello")],
            logprobs_result=logprobs_result,
        )

        result = _extract_logprobs(candidate)
        assert result is not None
        assert result.content is not None
        assert len(result.content) == 1
        assert result.content[0].token == "hello"
        assert result.content[0].logprob == -0.5

    def test_logprobs_bytes_is_utf8_encoding(self):
        """bytes field is UTF-8 encoding of the token string."""
        chosen = [_make_logprob_candidate("hi", -1.0)]
        logprobs_result = _make_logprobs_result(chosen=chosen)
        candidate = _make_candidate(logprobs_result=logprobs_result)

        result = _extract_logprobs(candidate)
        assert result is not None
        assert result.content is not None
        expected_bytes = list(b"hi")
        assert result.content[0].bytes == expected_bytes

    def test_top_logprobs_extracted(self):
        """top_logprobs list is populated from top_candidates."""
        chosen = [_make_logprob_candidate("hello", -0.5)]
        top = [
            _make_top_candidates_entry(
                [
                    _make_logprob_candidate("hello", -0.5),
                    _make_logprob_candidate("world", -1.2),
                ]
            )
        ]
        logprobs_result = _make_logprobs_result(chosen=chosen, top=top)
        candidate = _make_candidate(logprobs_result=logprobs_result)

        result = _extract_logprobs(candidate)
        assert result is not None
        assert result.content is not None
        assert result.content[0].top_logprobs is not None
        assert len(result.content[0].top_logprobs) == 2
        assert result.content[0].top_logprobs[0].token == "hello"
        assert result.content[0].top_logprobs[1].token == "world"

    def test_logprobs_in_streaming_chunk(self):
        """Logprobs extraction works in convert_gemini_stream_chunk_to_openai."""
        chosen = [_make_logprob_candidate("hey", -0.3)]
        logprobs_result = _make_logprobs_result(chosen=chosen)
        chunk = _make_response(
            candidates=[
                _make_candidate(
                    parts=[_make_text_part("hey")],
                    finish_reason="STOP",
                    logprobs_result=logprobs_result,
                )
            ]
        )

        result = convert_gemini_stream_chunk_to_openai(
            chunk=chunk,
            model="gemini-2.0-flash",
            completion_id="chatcmpl-test",
            is_first_chunk=True,
        )
        assert result.choices[0].logprobs is not None
        assert result.choices[0].logprobs.content is not None
        assert result.choices[0].logprobs.content[0].token == "hey"


class TestReasoningContent:
    def test_thought_parts_separated_from_text(self):
        """Candidate with both thought and text parts: text goes to text, thought to reasoning_content."""
        response = _make_response(
            candidates=[_make_candidate(parts=[_make_thought_part("Let me think"), _make_text_part("Answer")])]
        )
        result = convert_gemini_stream_chunk_to_openai(
            response, "gemini-2.5-flash", "chatcmpl-test", is_first_chunk=True
        )
        assert result.choices[0].delta.content == "Answer"
        assert result.choices[0].delta.reasoning_content == "Let me think"

    def test_multiple_thought_parts_concatenated(self):
        """Multiple thinking parts are concatenated into single reasoning_content."""
        response = _make_response(
            candidates=[
                _make_candidate(
                    parts=[
                        _make_thought_part("Step 1. "),
                        _make_thought_part("Step 2."),
                        _make_text_part("Done"),
                    ]
                )
            ]
        )
        result = convert_gemini_stream_chunk_to_openai(response, "model", "chatcmpl-test", is_first_chunk=True)
        assert result.choices[0].delta.reasoning_content == "Step 1. Step 2."
        assert result.choices[0].delta.content == "Done"

    def test_no_thought_parts_reasoning_content_none(self):
        """Regular candidate without thinking parts has reasoning_content=None."""
        response = _make_response(candidates=[_make_candidate(parts=[_make_text_part("Hello!")])])
        result = convert_gemini_stream_chunk_to_openai(response, "model", "chatcmpl-test", is_first_chunk=True)
        assert result.choices[0].delta.reasoning_content is None
        assert result.choices[0].delta.content == "Hello!"

    def test_only_thought_parts_text_none(self):
        """Candidate with only thinking parts: text=None, reasoning_content has the thinking text."""
        response = _make_response(candidates=[_make_candidate(parts=[_make_thought_part("I am thinking...")])])
        result = convert_gemini_stream_chunk_to_openai(response, "model", "chatcmpl-test", is_first_chunk=True)
        assert result.choices[0].delta.content is None
        assert result.choices[0].delta.reasoning_content == "I am thinking..."

    def test_thought_parts_in_streaming_chunk(self):
        """Streaming chunk: delta.reasoning_content is populated from thought parts."""
        chunk = _make_response(
            candidates=[
                _make_candidate(
                    parts=[_make_thought_part("Thinking..."), _make_text_part("Result")],
                    finish_reason="STOP",
                )
            ]
        )
        result = convert_gemini_stream_chunk_to_openai(chunk, "model", "chatcmpl-test", is_first_chunk=False)
        assert result.choices[0].delta.reasoning_content == "Thinking..."
        assert result.choices[0].delta.content == "Result"

    def test_thought_parts_in_non_streaming_response(self):
        """Non-streaming response: thinking text is REJOINED into content (no reasoning_content field on message)."""
        response = _make_response(
            candidates=[_make_candidate(parts=[_make_thought_part("Thinking"), _make_text_part("Answer")])]
        )
        result = convert_gemini_response_to_openai(response, "gemini-2.5-flash")
        # Thinking text is prepended to content for non-streaming
        assert result.choices[0].message.content == "ThinkingAnswer"
        # Message type has no reasoning_content field
        assert not hasattr(result.choices[0].message, "reasoning_content")

    def test_thought_and_tool_calls(self):
        """Thinking parts + function call: reasoning_content has thinking, tool_calls has function call."""
        response = _make_response(
            candidates=[
                _make_candidate(
                    parts=[
                        _make_thought_part("Let me call the tool"),
                        _make_function_call_part("get_weather", {"city": "NYC"}),
                    ]
                )
            ]
        )
        result = convert_gemini_stream_chunk_to_openai(response, "model", "chatcmpl-test", is_first_chunk=True)
        assert result.choices[0].delta.reasoning_content == "Let me call the tool"
        assert result.choices[0].delta.tool_calls is not None
        assert len(result.choices[0].delta.tool_calls) == 1
        assert result.choices[0].delta.content is None
