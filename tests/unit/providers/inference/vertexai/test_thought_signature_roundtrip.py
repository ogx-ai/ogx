# Regression tests for thought_signature preservation across the
# Gemini -> OpenAI -> Gemini conversion round trip.
#
# Vertex Gemini 3 rejects follow-up turns whose function-call parts are missing
# the thought_signature emitted on the original tool call ("Function call is
# missing a thought_signature in functionCall parts"). These tests pin every hop
# of the chain that carries the signature through OGX:
#   1. Gemini candidate part -> OpenAI tool call + adapter-local id->sig cache
#   2. OpenAI chat completion assistant message -> Gemini Content parts
#
# The signature travels through an adapter-local map keyed by the synthetic
# call id (converters._THOUGHT_SIGNATURE_CACHE); the shared OpenAI model layer
# stays untouched.
#
# All google-genai types are mocked via SimpleNamespace — no SDK installation required.

from types import SimpleNamespace
from typing import Any

import pytest

from ogx.providers.remote.inference.vertexai import converters


@pytest.fixture(autouse=True)
def _reset_signature_cache():
    converters._clear_thought_signature_cache()
    yield
    converters._clear_thought_signature_cache()


def _make_function_call_part(name: str, args: dict, thought_signature: Any = None) -> Any:
    return SimpleNamespace(
        text=None,
        thought=None,
        function_call=SimpleNamespace(name=name, args=args),
        thought_signature=thought_signature,
    )


def _make_candidate(parts: list) -> Any:
    content = SimpleNamespace(parts=parts)
    return SimpleNamespace(content=content, finish_reason="STOP")


class TestGeminiToOpenAICarriesSignature:
    def test_function_call_part_signature_extracted(self):
        candidate = _make_candidate(
            [_make_function_call_part("get_weather", {"city": "Paris"}, thought_signature="sig-abc")]
        )
        _, _, tool_calls = converters._extract_candidate_parts(candidate)

        assert len(tool_calls) == 1
        assert converters._lookup_thought_signature(tool_calls[0].id) == "sig-abc"
        assert not hasattr(tool_calls[0].function, "thought_signature") or getattr(
            tool_calls[0].function, "thought_signature", None
        ) is None

    def test_bytes_signature_base64_encoded(self):
        raw = b"\x9a\x04\x00"
        encoded = "mgQA"
        candidate = _make_candidate([_make_function_call_part("get_weather", {"city": "Paris"}, thought_signature=raw)])
        _, _, tool_calls = converters._extract_candidate_parts(candidate)

        assert converters._lookup_thought_signature(tool_calls[0].id) == encoded

    def test_absent_signature_is_none(self):
        candidate = _make_candidate([_make_function_call_part("get_weather", {"city": "Paris"})])
        _, _, tool_calls = converters._extract_candidate_parts(candidate)

        assert converters._lookup_thought_signature(tool_calls[0].id) is None

    def test_completion_response_carries_signature_in_tool_calls(self):
        """End-to-end: Gemini response -> OpenAI chat completion response."""
        candidate = _make_candidate(
            [_make_function_call_part("get_weather", {"city": "Paris"}, thought_signature="sig-xyz")]
        )
        response = SimpleNamespace(candidates=[candidate], usage_metadata=None)

        completion = converters.convert_gemini_response_to_openai(response, model="gemini-3.5-flash")

        tool_calls = completion.choices[0].message.tool_calls
        assert tool_calls is not None and len(tool_calls) == 1
        assert converters._lookup_thought_signature(tool_calls[0].id) == "sig-xyz"


class TestOpenAIToGeminiEmitsSignature:
    def _convert_single_assistant(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        system, contents = converters.convert_openai_messages_to_gemini([message])
        assert system is None
        return contents

    def test_signature_emitted_on_function_call_part(self):
        converters._cache_thought_signature("call_1", "sig-abc")
        contents = self._convert_single_assistant(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                    }
                ],
            }
        )

        assert len(contents) == 1
        parts = contents[0]["parts"]
        assert len(parts) == 1
        assert parts[0]["function_call"]["name"] == "get_weather"
        assert parts[0]["thought_signature"] == "sig-abc"

    def test_no_signature_omits_key(self):
        contents = self._convert_single_assistant(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
            }
        )

        assert "thought_signature" not in contents[0]["parts"][0]

    def test_empty_signature_omits_key(self):
        contents = self._convert_single_assistant(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{}",
                            "thought_signature": "",
                        },
                    }
                ],
            }
        )

        assert "thought_signature" not in contents[0]["parts"][0]


class TestFullRoundTrip:
    def test_gemini_to_openai_to_gemini_keeps_signature(self):
        """The reported scenario: a tool call generated by Gemini must reach the
        follow-up generateContent request with its thought_signature intact."""

        # Hop 1: Gemini emits a function call with a signature.
        candidate = _make_candidate(
            [_make_function_call_part("get_weather", {"city": "Paris"}, thought_signature="round-trip-sig")]
        )
        _, _, tool_calls = converters._extract_candidate_parts(candidate)
        assistant_message = {
            "role": "assistant",
            "tool_calls": [tc.model_dump() for tc in tool_calls],
        }

        # Hop 2: the reconstructed history is converted back for the next turn.
        system, contents = converters.convert_openai_messages_to_gemini([assistant_message])

        parts = contents[0]["parts"]
        assert parts[0]["thought_signature"] == "round-trip-sig"


class TestSignatureCacheBehavior:
    def test_unknown_call_id_yields_none(self):
        assert converters._lookup_thought_signature("call_does_not_exist") is None

    def test_cache_is_bounded(self):
        for i in range(converters._THOUGHT_SIGNATURE_CACHE_MAX + 50):
            converters._cache_thought_signature(f"call_{i}", f"sig-{i}")

        assert len(converters._THOUGHT_SIGNATURE_CACHE) == converters._THOUGHT_SIGNATURE_CACHE_MAX
        # oldest entries evicted first
        assert converters._lookup_thought_signature("call_0") is None
        last = converters._THOUGHT_SIGNATURE_CACHE_MAX + 49
        assert converters._lookup_thought_signature(f"call_{last}") == f"sig-{last}"



if __name__ == "__main__":
    pytest.main([__file__, "-v"])
