# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for VertexAI thought_signature persistence across tool-call round trips."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from ogx.core.storage.kvstore.kvstore import InmemoryKVStoreImpl
from ogx.providers.remote.inference.vertexai import converters
from ogx.providers.remote.inference.vertexai.config import VertexAIConfig
from ogx.providers.remote.inference.vertexai.thought_signature_store import ThoughtSignatureStore
from ogx.providers.remote.inference.vertexai.vertexai import VertexAIInferenceAdapter
from ogx_api.inference.models import OpenAIChatCompletionRequestWithExtraBody

_WEATHER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }
]


def _make_function_call_part(name: str, args: dict, thought_signature: Any = None) -> Any:
    return SimpleNamespace(
        text=None,
        thought=None,
        function_call=SimpleNamespace(name=name, args=args),
        thought_signature=thought_signature,
    )


def _make_candidate(parts: list) -> Any:
    return SimpleNamespace(content=SimpleNamespace(parts=parts), finish_reason="STOP")


def _make_gemini_response(parts: list) -> Any:
    return SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=parts),
                finish_reason="STOP",
                index=0,
                logprobs_result=None,
            )
        ],
        usage_metadata=None,
    )


class TestNormalizeThoughtSignature:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (b"\x9a\x04\x00", "mgQA"),
            ("sig-abc", "sig-abc"),
            (None, None),
            ("", None),
        ],
    )
    def test_normalize(self, value, expected):
        assert converters._normalize_thought_signature(value) == expected


class TestExtractSignatures:
    def test_extracts_string_signature(self):
        signatures: dict[str, str] = {}
        _, _, tool_calls = converters._extract_candidate_parts(
            _make_candidate([_make_function_call_part("get_weather", {"city": "Paris"}, thought_signature="sig-abc")]),
            signatures,
        )

        assert len(tool_calls) == 1
        assert signatures[tool_calls[0].id] == "sig-abc"
        assert getattr(tool_calls[0].function, "thought_signature", None) is None

    def test_extracts_bytes_as_base64(self):
        signatures: dict[str, str] = {}
        _, _, tool_calls = converters._extract_candidate_parts(
            _make_candidate(
                [_make_function_call_part("get_weather", {"city": "Paris"}, thought_signature=b"\x9a\x04\x00")]
            ),
            signatures,
        )

        assert signatures[tool_calls[0].id] == "mgQA"

    def test_extracts_camel_case_attribute(self):
        part = SimpleNamespace(
            text=None,
            thought=None,
            function_call=SimpleNamespace(name="get_weather", args={"city": "Paris"}),
            thoughtSignature="sig-camel",
        )
        signatures: dict[str, str] = {}
        _, _, tool_calls = converters._extract_candidate_parts(_make_candidate([part]), signatures)

        assert signatures[tool_calls[0].id] == "sig-camel"

    def test_absent_signature_not_recorded(self):
        signatures: dict[str, str] = {}
        _, _, tool_calls = converters._extract_candidate_parts(
            _make_candidate([_make_function_call_part("get_weather", {"city": "Paris"})]),
            signatures,
        )

        assert len(tool_calls) == 1
        assert signatures == {}


class TestEmitSignatures:
    def test_emits_stored_signature(self):
        _, contents = converters.convert_openai_messages_to_gemini(
            [
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
            ],
            {"call_1": "sig-abc"},
        )

        assert contents[0]["parts"][0]["thought_signature"] == "sig-abc"

    def test_parallel_sibling_omits_missing_signature(self):
        _, contents = converters.convert_openai_messages_to_gemini(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "London"}'},
                        },
                    ],
                }
            ],
            {"call_1": "sig-first"},
        )

        assert contents[0]["parts"][0]["thought_signature"] == "sig-first"
        assert "thought_signature" not in contents[0]["parts"][1]


class TestThoughtSignatureStore:
    async def test_put_get_across_store_instances(self):
        kv = InmemoryKVStoreImpl(namespace="shared")
        store_a = ThoughtSignatureStore(kv)
        store_b = ThoughtSignatureStore(kv)

        await store_a.put("call_1", "sig-abc")
        assert await store_b.get("call_1") == "sig-abc"

    async def test_expired_entry_is_miss(self):
        kv = InmemoryKVStoreImpl()
        store = ThoughtSignatureStore(kv)
        await store.put("call_1", "sig-abc")
        await kv.set(
            "thought_sig:call_1",
            "sig-abc",
            expiration=datetime.now(tz=UTC) - timedelta(seconds=1),
        )

        assert await store.get("call_1") is None


class TestAdapterRoundTrip:
    async def test_chat_completion_persists_and_reloads_across_workers(self, monkeypatch):
        shared_kv = InmemoryKVStoreImpl(namespace="e2e")

        adapter_a = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        cast(Any, adapter_a).__provider_id__ = "vertexai"
        monkeypatch.setattr(adapter_a, "_thought_signature_store", ThoughtSignatureStore(shared_kv))
        monkeypatch.setattr(adapter_a, "_validate_model_allowed", lambda _: None)
        monkeypatch.setattr(
            adapter_a,
            "_get_client",
            lambda: SimpleNamespace(
                aio=SimpleNamespace(
                    models=SimpleNamespace(
                        generate_content=AsyncMock(
                            return_value=_make_gemini_response(
                                [
                                    _make_function_call_part(
                                        "get_weather",
                                        {"city": "Paris"},
                                        thought_signature="e2e-sig",
                                    )
                                ]
                            )
                        )
                    )
                )
            ),
        )

        turn1 = await adapter_a.openai_chat_completion(
            OpenAIChatCompletionRequestWithExtraBody(
                model="publishers/google/models/gemini-3-flash",
                messages=cast(Any, [{"role": "user", "content": "weather?"}]),
                tools=cast(Any, _WEATHER_TOOLS),
            )
        )
        tool_calls = turn1.choices[0].message.tool_calls
        assert tool_calls is not None and len(tool_calls) == 1
        call_id = tool_calls[0].id
        assert await shared_kv.get(f"thought_sig:{call_id}") == "e2e-sig"

        adapter_b = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        cast(Any, adapter_b).__provider_id__ = "vertexai"
        monkeypatch.setattr(adapter_b, "_thought_signature_store", ThoughtSignatureStore(shared_kv))
        monkeypatch.setattr(adapter_b, "_validate_model_allowed", lambda _: None)

        captured: dict[str, Any] = {}

        async def _generate(model, contents, config):
            captured["contents"] = contents
            return _make_gemini_response([SimpleNamespace(text="Sunny", thought=None, function_call=None)])

        monkeypatch.setattr(
            adapter_b,
            "_get_client",
            lambda: SimpleNamespace(
                aio=SimpleNamespace(models=SimpleNamespace(generate_content=AsyncMock(side_effect=_generate)))
            ),
        )

        turn2 = await adapter_b.openai_chat_completion(
            OpenAIChatCompletionRequestWithExtraBody(
                model="publishers/google/models/gemini-3-flash",
                messages=cast(
                    Any,
                    [
                        {"role": "user", "content": "weather?"},
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [tc.model_dump() for tc in tool_calls],
                        },
                        {"role": "tool", "tool_call_id": call_id, "content": '{"temp": 22}'},
                    ],
                ),
                tools=cast(Any, _WEATHER_TOOLS),
            )
        )
        assert turn2.choices[0].message.content == "Sunny"

        model_msg = next(content for content in captured["contents"] if content["role"] == "model")
        fc_part = next(part for part in model_msg["parts"] if "function_call" in part)
        assert fc_part["thought_signature"] == "e2e-sig"
