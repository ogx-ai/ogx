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
    convert_completion_prompt_to_contents,
    convert_gemini_response_to_openai,
    convert_gemini_response_to_openai_completion,
    convert_gemini_stream_chunk_to_openai,
    convert_gemini_stream_chunk_to_openai_completion,
)
from ogx_api import OpenAICompletion

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


class TestConvertCompletionPromptToContents:
    """Tests for convert_completion_prompt_to_contents()."""

    def test_string_prompt_to_user_message(self):
        """Plain text prompt is wrapped as a single user message."""
        result = convert_completion_prompt_to_contents("hello world")
        assert result == [{"role": "user", "parts": [{"text": "hello world"}]}]

    def test_empty_prompt(self):
        """Empty string prompt still produces a valid user message."""
        result = convert_completion_prompt_to_contents("")
        assert result == [{"role": "user", "parts": [{"text": ""}]}]


class TestConvertGeminiResponseToCompletion:
    """Tests for convert_gemini_response_to_openai_completion()."""

    def test_simple_text_completion(self):
        """Single candidate with text maps to OpenAICompletion correctly."""
        response = _make_response(
            candidates=[_make_candidate(parts=[_make_text_part("The answer is 42")], finish_reason=None)],
        )
        result = convert_gemini_response_to_openai_completion(response, model="gemini-2.5-flash", prompt="What is 42?")
        assert isinstance(result, OpenAICompletion)
        assert result.choices[0].text == "The answer is 42"
        assert result.model == "gemini-2.5-flash"

    def test_no_candidates_returns_empty_text(self):
        """Empty candidates list produces a content_filter choice with empty text."""
        response = _make_response(candidates=[])
        result = convert_gemini_response_to_openai_completion(response, model="gemini-2.5-flash", prompt="hi")
        assert result.choices[0].text == ""
        assert result.choices[0].finish_reason == "content_filter"

    def test_finish_reason_stop(self):
        """STOP finish reason maps to 'stop'."""
        response = _make_response(
            candidates=[_make_candidate(parts=[_make_text_part("done")], finish_reason="STOP")],
        )
        result = convert_gemini_response_to_openai_completion(response, model="gemini-2.5-flash", prompt="hi")
        assert result.choices[0].finish_reason == "stop"

    def test_completion_logprobs_extracted_when_present(self):
        """Logprobs from candidate logprobs_result are converted to OpenAIChoiceLogprobs."""
        chosen = [_make_logprob_candidate("hello", -0.5)]
        logprobs_result = _make_logprobs_result(chosen=chosen)
        response = _make_response(
            candidates=[
                _make_candidate(
                    parts=[_make_text_part("hello")],
                    finish_reason="STOP",
                    logprobs_result=logprobs_result,
                )
            ]
        )
        result = convert_gemini_response_to_openai_completion(response, model="test-model", prompt="hi")
        logprobs = result.choices[0].logprobs
        assert logprobs is not None
        assert logprobs.content is not None
        assert len(logprobs.content) == 1
        assert logprobs.content[0].token == "hello"
        assert logprobs.content[0].logprob == -0.5

    def test_completion_logprobs_none_when_absent(self):
        """Logprobs is None when candidate has no logprobs_result."""
        response = _make_response(
            candidates=[_make_candidate(parts=[_make_text_part("hello")], finish_reason="STOP")],
        )
        result = convert_gemini_response_to_openai_completion(response, model="test-model", prompt="hi")
        assert result.choices[0].logprobs is None

    def test_multiple_candidates_return_multiple_choices(self):
        """Test that multiple candidates return multiple choices."""
        response = _make_response(
            candidates=[
                _make_candidate(parts=[_make_text_part("first")], finish_reason="STOP", index=0),
                _make_candidate(parts=[_make_text_part("second")], finish_reason="STOP", index=1),
            ]
        )
        result = convert_gemini_response_to_openai_completion(response, model="test-model", prompt="hi")
        assert len(result.choices) == 2
        assert result.choices[0].text == "first"
        assert result.choices[0].index == 0
        assert result.choices[1].text == "second"
        assert result.choices[1].index == 1


class TestConvertGeminiStreamChunkToOpenAICompletion:
    """Tests for convert_gemini_stream_chunk_to_openai_completion()."""

    def test_stream_completion_chunk_with_text(self):
        """Chunk with a text candidate maps to OpenAICompletion with that text."""
        chunk = _make_response(
            candidates=[_make_candidate(parts=[_make_text_part("Hello stream")], finish_reason="STOP")],
        )

        result = convert_gemini_stream_chunk_to_openai_completion(
            chunk, model="gemini-2.5-flash", completion_id="cmpl-abc"
        )

        assert isinstance(result, OpenAICompletion)
        assert result.id == "cmpl-abc"
        assert result.model == "gemini-2.5-flash"
        assert len(result.choices) == 1
        assert result.choices[0].text == "Hello stream"
        assert result.choices[0].finish_reason == "stop"
        assert result.choices[0].index == 0

    def test_stream_completion_chunk_mid_stream_uses_stop_default(self):
        """Mid-stream chunk with no finish_reason defaults to 'stop' (required field)."""
        chunk = _make_response(
            candidates=[_make_candidate(parts=[_make_text_part("partial")], finish_reason=None)],
        )

        result = convert_gemini_stream_chunk_to_openai_completion(chunk, model="test-model", completion_id="cmpl-xyz")

        # finish_reason must be a valid value (not None) since OpenAICompletionChoice requires it
        assert result.choices[0].finish_reason == "stop"
        assert result.choices[0].text == "partial"

    def test_stream_completion_chunk_with_logprobs(self):
        """Chunk with logprobs_result populates logprobs on the choice."""
        chosen = [_make_logprob_candidate("hello", -0.3)]
        logprobs_result = _make_logprobs_result(chosen=chosen)
        chunk = _make_response(
            candidates=[
                _make_candidate(
                    parts=[_make_text_part("hello")],
                    finish_reason="STOP",
                    logprobs_result=logprobs_result,
                )
            ]
        )

        result = convert_gemini_stream_chunk_to_openai_completion(chunk, model="test-model", completion_id="cmpl-lp")

        logprobs = result.choices[0].logprobs
        assert logprobs is not None
        assert logprobs.content is not None
        assert len(logprobs.content) == 1
        assert logprobs.content[0].token == "hello"
        assert logprobs.content[0].logprob == -0.3

    def test_stream_completion_chunk_empty_candidates(self):
        """Chunk with no candidates produces a single fallback choice with empty text."""
        chunk = _make_response(candidates=[])

        result = convert_gemini_stream_chunk_to_openai_completion(chunk, model="test-model", completion_id="cmpl-empty")

        assert len(result.choices) == 1
        assert result.choices[0].text == ""
        assert result.choices[0].finish_reason == "stop"
        assert result.choices[0].index == 0

    def test_stream_completion_chunk_none_candidates(self):
        """Chunk with candidates=None also produces a fallback choice."""
        chunk = _make_response(candidates=None)

        result = convert_gemini_stream_chunk_to_openai_completion(chunk, model="test-model", completion_id="cmpl-none")

        assert len(result.choices) == 1
        assert result.choices[0].text == ""
        assert result.choices[0].finish_reason == "stop"

    def test_stream_completion_chunk_multiple_candidates(self):
        """Multiple candidates produce multiple choices with correct indices."""
        chunk = _make_response(
            candidates=[
                _make_candidate(parts=[_make_text_part("first")], finish_reason="STOP", index=0),
                _make_candidate(parts=[_make_text_part("second")], finish_reason="STOP", index=1),
            ]
        )

        result = convert_gemini_stream_chunk_to_openai_completion(chunk, model="test-model", completion_id="cmpl-multi")

        assert len(result.choices) == 2
        assert result.choices[0].text == "first"
        assert result.choices[0].index == 0
        assert result.choices[1].text == "second"
        assert result.choices[1].index == 1
