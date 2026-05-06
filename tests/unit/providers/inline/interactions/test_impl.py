# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for the BuiltinInteractionsImpl translation and passthrough logic."""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from ogx.providers.inline.interactions.config import InteractionsConfig
from ogx.providers.inline.interactions.impl import BuiltinInteractionsImpl
from ogx_api.interactions.models import (
    GoogleCreateInteractionRequest,
    GoogleGenerationConfig,
    GoogleInputTurn,
    GoogleTextContent,
)


def _msg_to_dict(msg):
    """Convert a Pydantic message model to dict for easy assertion."""
    if hasattr(msg, "model_dump"):
        return msg.model_dump(exclude_none=True)
    return dict(msg)


@pytest.fixture
def impl():
    mock_inference = AsyncMock()
    return BuiltinInteractionsImpl(config=InteractionsConfig(), inference_api=mock_inference)


class TestRequestTranslation:
    def test_simple_string_input(self, impl):
        request = GoogleCreateInteractionRequest(
            model="gemini-2.5-flash",
            input="Hello",
        )
        result = impl._google_to_openai(request)

        assert result.model == "gemini-2.5-flash"
        assert len(result.messages) == 1
        m = _msg_to_dict(result.messages[0])
        assert m["role"] == "user"
        assert m["content"] == "Hello"

    def test_conversation_turns(self, impl):
        request = GoogleCreateInteractionRequest(
            model="m",
            input=[
                GoogleInputTurn(role="user", content=[GoogleTextContent(text="Question 1")]),
                GoogleInputTurn(role="model", content=[GoogleTextContent(text="Answer 1")]),
                GoogleInputTurn(role="user", content=[GoogleTextContent(text="Question 2")]),
            ],
        )
        result = impl._google_to_openai(request)

        assert len(result.messages) == 3
        m0 = _msg_to_dict(result.messages[0])
        m1 = _msg_to_dict(result.messages[1])
        m2 = _msg_to_dict(result.messages[2])
        assert m0["role"] == "user"
        assert m0["content"] == "Question 1"
        assert m1["role"] == "assistant"
        assert m1["content"] == "Answer 1"
        assert m2["role"] == "user"
        assert m2["content"] == "Question 2"

    def test_model_role_mapped_to_assistant(self, impl):
        request = GoogleCreateInteractionRequest(
            model="m",
            input=[
                GoogleInputTurn(role="model", content=[GoogleTextContent(text="I am the model")]),
            ],
        )
        result = impl._google_to_openai(request)

        m = _msg_to_dict(result.messages[0])
        assert m["role"] == "assistant"

    def test_system_instruction(self, impl):
        request = GoogleCreateInteractionRequest(
            model="m",
            input="Hi",
            system_instruction="You are helpful.",
        )
        result = impl._google_to_openai(request)

        assert len(result.messages) == 2
        m0 = _msg_to_dict(result.messages[0])
        m1 = _msg_to_dict(result.messages[1])
        assert m0["role"] == "system"
        assert m0["content"] == "You are helpful."
        assert m1["role"] == "user"

    def test_generation_config_temperature(self, impl):
        request = GoogleCreateInteractionRequest(
            model="m",
            input="Hi",
            generation_config=GoogleGenerationConfig(temperature=0.7),
        )
        result = impl._google_to_openai(request)
        assert result.temperature == 0.7

    def test_generation_config_top_p(self, impl):
        request = GoogleCreateInteractionRequest(
            model="m",
            input="Hi",
            generation_config=GoogleGenerationConfig(top_p=0.9),
        )
        result = impl._google_to_openai(request)
        assert result.top_p == 0.9

    def test_generation_config_max_output_tokens(self, impl):
        request = GoogleCreateInteractionRequest(
            model="m",
            input="Hi",
            generation_config=GoogleGenerationConfig(max_output_tokens=500),
        )
        result = impl._google_to_openai(request)
        assert result.max_tokens == 500

    def test_generation_config_top_k_extra_body(self, impl):
        request = GoogleCreateInteractionRequest(
            model="m",
            input="Hi",
            generation_config=GoogleGenerationConfig(top_k=40),
        )
        result = impl._google_to_openai(request)
        assert result.model_extra.get("top_k") == 40

    def test_stream_flag(self, impl):
        request = GoogleCreateInteractionRequest(
            model="m",
            input="Hi",
            stream=True,
        )
        result = impl._google_to_openai(request)
        assert result.stream is True

    def test_multi_content_turn(self, impl):
        request = GoogleCreateInteractionRequest(
            model="m",
            input=[
                GoogleInputTurn(
                    role="user",
                    content=[
                        GoogleTextContent(text="Line 1"),
                        GoogleTextContent(text="Line 2"),
                    ],
                ),
            ],
        )
        result = impl._google_to_openai(request)

        m = _msg_to_dict(result.messages[0])
        assert m["content"] == "Line 1\nLine 2"


class TestResponseTranslation:
    def test_simple_text_response(self, impl):
        openai_resp = MagicMock()
        openai_resp.choices = [MagicMock()]
        openai_resp.choices[0].message = MagicMock()
        openai_resp.choices[0].message.content = "Hello!"
        openai_resp.choices[0].finish_reason = "stop"
        openai_resp.usage = MagicMock()
        openai_resp.usage.prompt_tokens = 10
        openai_resp.usage.completion_tokens = 5

        result = impl._openai_to_google(openai_resp, "gemini-2.5-flash")

        assert result.id.startswith("interaction-")
        assert result.status == "completed"
        assert result.model == "gemini-2.5-flash"
        assert result.role == "model"
        assert result.object == "interaction"
        assert result.created is not None
        assert result.updated is not None
        assert len(result.outputs) == 1
        assert result.outputs[0].type == "text"
        assert result.outputs[0].text == "Hello!"
        assert result.usage.total_input_tokens == 10
        assert result.usage.total_output_tokens == 5
        assert result.usage.total_tokens == 15

    def test_empty_response(self, impl):
        openai_resp = MagicMock()
        openai_resp.choices = [MagicMock()]
        openai_resp.choices[0].message = MagicMock()
        openai_resp.choices[0].message.content = None
        openai_resp.choices[0].finish_reason = "stop"
        openai_resp.usage = MagicMock()
        openai_resp.usage.prompt_tokens = 5
        openai_resp.usage.completion_tokens = 0

        result = impl._openai_to_google(openai_resp, "m")

        assert result.status == "completed"
        assert len(result.outputs) == 0

    def test_missing_usage(self, impl):
        openai_resp = MagicMock()
        openai_resp.choices = [MagicMock()]
        openai_resp.choices[0].message = MagicMock()
        openai_resp.choices[0].message.content = "Hi"
        openai_resp.choices[0].finish_reason = "stop"
        openai_resp.usage = None

        result = impl._openai_to_google(openai_resp, "m")

        assert result.usage.total_input_tokens == 0
        assert result.usage.total_output_tokens == 0
        assert result.usage.total_tokens == 0


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
        async for event in impl._stream_openai_to_google(mock_stream(), "m"):
            events.append(event)

        # interaction.start wraps in interaction object
        assert events[0].event_type == "interaction.start"
        assert events[0].interaction.id.startswith("interaction-")
        assert events[0].interaction.status == "in_progress"
        assert events[0].interaction.model == "m"
        assert events[0].interaction.object == "interaction"
        # content.start wraps type in content object
        assert events[1].event_type == "content.start"
        assert events[1].content.type == "text"
        # content.delta unchanged
        assert events[2].event_type == "content.delta"
        assert events[2].delta.text == "Hello"
        assert events[3].event_type == "content.delta"
        assert events[3].delta.text == " world"
        assert events[4].event_type == "content.delta"
        assert events[4].delta.text == "!"
        assert events[5].event_type == "content.stop"
        # interaction.complete wraps in interaction object
        assert events[6].event_type == "interaction.complete"
        assert events[6].interaction.status == "completed"
        assert events[6].interaction.model == "m"
        assert events[6].interaction.role == "model"
        assert events[6].interaction.object == "interaction"

    async def test_streaming_with_usage(self, impl):
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta = MagicMock()
        chunk1.choices[0].delta.content = "Hi"
        chunk1.choices[0].delta.tool_calls = None
        chunk1.choices[0].finish_reason = None
        chunk1.usage = None

        # Usage-only chunk
        chunk2 = MagicMock()
        chunk2.choices = []
        chunk2.usage = MagicMock()
        chunk2.usage.prompt_tokens = 10
        chunk2.usage.completion_tokens = 5

        async def mock_stream():
            yield chunk1
            yield chunk2

        events = []
        async for event in impl._stream_openai_to_google(mock_stream(), "m"):
            events.append(event)

        complete_event = [e for e in events if e.event_type == "interaction.complete"][0]
        assert complete_event.interaction.usage.total_input_tokens == 10
        assert complete_event.interaction.usage.total_output_tokens == 5
        assert complete_event.interaction.usage.total_tokens == 15

    async def test_empty_streaming(self, impl):
        async def mock_stream():
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta = MagicMock()
            chunk.choices[0].delta.content = None
            chunk.choices[0].delta.tool_calls = None
            chunk.choices[0].finish_reason = "stop"
            chunk.usage = None
            yield chunk

        events = []
        async for event in impl._stream_openai_to_google(mock_stream(), "m"):
            events.append(event)

        assert events[0].event_type == "interaction.start"
        # No content.start/delta/stop since no content
        assert events[1].event_type == "interaction.complete"


class TestPassthroughDetection:
    """Tests for the native passthrough detection logic."""

    def _make_impl_with_router(
        self,
        provider_module: str,
        base_url: str,
        auth_headers: dict[str, str] | None = None,
        network_config=None,
    ):
        """Create an impl with a mocked routing table and provider."""
        if auth_headers is None:
            auth_headers = {"x-goog-api-key": "test-key"}

        mock_inference = AsyncMock()
        mock_inference.routing_table = AsyncMock()

        mock_obj = MagicMock()
        mock_obj.identifier = "gemini/gemini-2.5-flash"
        mock_obj.provider_resource_id = "gemini-2.5-flash"
        mock_inference.routing_table.get_object_by_identifier = AsyncMock(return_value=mock_obj)

        mock_provider = MagicMock()
        mock_provider.__class__.__module__ = provider_module
        mock_provider.get_base_url = MagicMock(return_value=base_url)
        mock_provider.get_passthrough_auth_headers = MagicMock(return_value=auth_headers)
        mock_provider.config = MagicMock()
        mock_provider.config.network = network_config
        mock_inference.routing_table.get_provider_impl = AsyncMock(return_value=mock_provider)

        return BuiltinInteractionsImpl(config=InteractionsConfig(), inference_api=mock_inference)

    async def test_gemini_provider_detected(self):
        impl = self._make_impl_with_router(
            provider_module="ogx.providers.remote.inference.gemini.gemini",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        result = await impl._get_passthrough_info("gemini/gemini-2.5-flash")

        assert result is not None
        assert result["base_url"] == "https://generativelanguage.googleapis.com/v1beta"
        assert result["auth_headers"] == {"x-goog-api-key": "test-key"}
        assert result["provider_resource_id"] == "gemini-2.5-flash"

    async def test_non_gemini_provider_returns_none(self):
        impl = self._make_impl_with_router(
            provider_module="ogx.providers.remote.inference.openai",
            base_url="https://api.openai.com/v1",
        )
        result = await impl._get_passthrough_info("openai/gpt-4o")

        assert result is None

    async def test_no_auth_headers_returns_none(self):
        impl = self._make_impl_with_router(
            provider_module="ogx.providers.remote.inference.gemini.gemini",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            auth_headers={},
        )
        result = await impl._get_passthrough_info("gemini/gemini-2.5-flash")

        assert result is None

    async def test_no_routing_table_returns_none(self):
        mock_inference = AsyncMock(spec=[])
        impl = BuiltinInteractionsImpl(config=InteractionsConfig(), inference_api=mock_inference)
        result = await impl._get_passthrough_info("gemini/gemini-2.5-flash")

        assert result is None

    async def test_openai_suffix_stripped(self):
        impl = self._make_impl_with_router(
            provider_module="ogx.providers.remote.inference.gemini.gemini",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        )
        result = await impl._get_passthrough_info("gemini/gemini-2.5-flash")

        assert result is not None
        assert result["base_url"] == "https://generativelanguage.googleapis.com/v1beta"

    async def test_network_config_propagated(self):
        network_config = MagicMock()
        impl = self._make_impl_with_router(
            provider_module="ogx.providers.remote.inference.gemini.gemini",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            network_config=network_config,
        )
        result = await impl._get_passthrough_info("gemini/gemini-2.5-flash")

        assert result is not None
        assert result["network_config"] is network_config


class TestCreateInteractionPassthrough:
    def _make_impl_with_router(
        self,
        provider_module: str,
        base_url: str,
        auth_headers: dict[str, str] | None = None,
        network_config=None,
    ):
        if auth_headers is None:
            auth_headers = {"x-goog-api-key": "test-key"}

        mock_inference = AsyncMock()
        mock_inference.routing_table = AsyncMock()

        mock_obj = MagicMock()
        mock_obj.identifier = "gemini/gemini-2.5-flash"
        mock_obj.provider_resource_id = "gemini-2.5-flash"
        mock_inference.routing_table.get_object_by_identifier = AsyncMock(return_value=mock_obj)

        mock_provider = MagicMock()
        mock_provider.__class__.__module__ = provider_module
        mock_provider.get_base_url = MagicMock(return_value=base_url)
        mock_provider.get_passthrough_auth_headers = MagicMock(return_value=auth_headers)
        mock_provider.config = MagicMock()
        mock_provider.config.network = network_config
        mock_inference.routing_table.get_provider_impl = AsyncMock(return_value=mock_provider)

        return BuiltinInteractionsImpl(config=InteractionsConfig(), inference_api=mock_inference)

    async def test_non_streaming_uses_native_passthrough(self):
        impl = self._make_impl_with_router(
            provider_module="ogx.providers.remote.inference.gemini.gemini",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        )
        expected = MagicMock()
        impl._passthrough_request = AsyncMock(return_value=expected)

        request = GoogleCreateInteractionRequest(model="gemini/gemini-2.5-flash", input="hello", stream=False)
        result = await impl.create_interaction(request)

        assert result is expected
        impl._passthrough_request.assert_awaited_once()
        impl.inference_api.openai_chat_completion.assert_not_awaited()

    async def test_streaming_uses_native_passthrough(self):
        impl = self._make_impl_with_router(
            provider_module="ogx.providers.remote.inference.gemini.gemini",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        )
        expected = MagicMock()
        impl._passthrough_request = AsyncMock(return_value=expected)

        request = GoogleCreateInteractionRequest(model="gemini/gemini-2.5-flash", input="hello", stream=True)
        result = await impl.create_interaction(request)

        assert result is expected
        impl._passthrough_request.assert_awaited_once()
        impl.inference_api.openai_chat_completion.assert_not_awaited()


class TestPassthroughRequest:
    async def test_non_streaming_uses_header_auth_and_no_query_params(self, monkeypatch):
        impl = BuiltinInteractionsImpl(config=InteractionsConfig(), inference_api=AsyncMock())
        passthrough = {
            "base_url": "https://generativelanguage.googleapis.com/v1beta",
            "auth_headers": {"x-goog-api-key": "test-key"},
            "provider_resource_id": "gemini-2.5-flash",
            "network_config": None,
        }
        request = GoogleCreateInteractionRequest(model="gemini/gemini-2.5-flash", input="hello", stream=False)

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "id": "interaction-test",
            "model": "gemini-2.5-flash",
            "outputs": [{"text": "hello"}],
        }

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.post = AsyncMock(return_value=mock_response)
        async_client_ctor = MagicMock(return_value=mock_client)
        monkeypatch.setattr("ogx.providers.inline.interactions.impl.httpx.AsyncClient", async_client_ctor)

        result = await impl._passthrough_request(passthrough, request)

        assert result.model == "gemini-2.5-flash"
        ctor_kwargs = async_client_ctor.call_args.kwargs
        assert ctor_kwargs["headers"]["x-goog-api-key"] == "test-key"
        assert ctor_kwargs["headers"]["content-type"] == "application/json"

        post_kwargs = mock_client.post.call_args.kwargs
        assert "params" not in post_kwargs
        assert post_kwargs["json"]["model"] == "gemini-2.5-flash"

    async def test_non_streaming_applies_network_config_client_kwargs(self, monkeypatch):
        impl = BuiltinInteractionsImpl(config=InteractionsConfig(), inference_api=AsyncMock())
        network_config = MagicMock()
        passthrough = {
            "base_url": "https://generativelanguage.googleapis.com/v1beta",
            "auth_headers": {"x-goog-api-key": "test-key"},
            "provider_resource_id": "gemini-2.5-flash",
            "network_config": network_config,
        }
        request = GoogleCreateInteractionRequest(model="gemini/gemini-2.5-flash", input="hello", stream=False)

        built_kwargs = {"headers": {"x-custom-header": "enabled"}, "timeout": httpx.Timeout(42.0)}
        build_kwargs_mock = MagicMock(return_value=built_kwargs)
        monkeypatch.setattr(
            "ogx.providers.inline.interactions.impl.build_network_client_kwargs",
            build_kwargs_mock,
        )

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "id": "interaction-test",
            "model": "gemini-2.5-flash",
            "outputs": [{"text": "hello"}],
        }

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.post = AsyncMock(return_value=mock_response)
        async_client_ctor = MagicMock(return_value=mock_client)
        monkeypatch.setattr("ogx.providers.inline.interactions.impl.httpx.AsyncClient", async_client_ctor)

        await impl._passthrough_request(passthrough, request)

        build_kwargs_mock.assert_called_once_with(network_config)
        ctor_kwargs = async_client_ctor.call_args.kwargs
        assert ctor_kwargs["headers"]["x-custom-header"] == "enabled"
        assert ctor_kwargs["headers"]["x-goog-api-key"] == "test-key"
        assert ctor_kwargs["timeout"] == built_kwargs["timeout"]

    async def test_non_streaming_accepts_thought_outputs(self, monkeypatch):
        impl = BuiltinInteractionsImpl(config=InteractionsConfig(), inference_api=AsyncMock())
        passthrough = {
            "base_url": "https://generativelanguage.googleapis.com/v1beta",
            "auth_headers": {"x-goog-api-key": "test-key"},
            "provider_resource_id": "gemini-2.5-flash",
            "network_config": None,
        }
        request = GoogleCreateInteractionRequest(model="gemini/gemini-2.5-flash", input="hello", stream=False)

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "id": "interaction-test",
            "status": "completed",
            "model": "gemini-2.5-flash",
            "outputs": [
                {
                    "type": "thought",
                    "signature": "sig-123",
                },
                {
                    "type": "text",
                    "text": "4",
                },
            ],
            "usage": {
                "total_input_tokens": 10,
                "total_output_tokens": 1,
                "total_tokens": 11,
            },
            "role": "model",
            "object": "interaction",
        }

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.post = AsyncMock(return_value=mock_response)
        async_client_ctor = MagicMock(return_value=mock_client)
        monkeypatch.setattr("ogx.providers.inline.interactions.impl.httpx.AsyncClient", async_client_ctor)

        result = await impl._passthrough_request(passthrough, request)

        assert len(result.outputs) == 2
        assert result.outputs[0].type == "thought"
        assert getattr(result.outputs[0], "signature", None) == "sig-123"
        assert result.outputs[1].type == "text"
        assert getattr(result.outputs[1], "text", None) == "4"


class TestSSEParsing:
    """Tests for the _parse_sse_event method used in passthrough streaming."""

    @pytest.fixture
    def impl(self):
        mock_inference = AsyncMock()
        return BuiltinInteractionsImpl(config=InteractionsConfig(), inference_api=mock_inference)

    def test_parse_interaction_start(self, impl):
        event = impl._parse_sse_event(
            "interaction.start",
            {"interaction": {"id": "test-123", "status": "in_progress", "model": "gemini-2.5-flash"}},
        )
        assert event is not None
        assert event.event_type == "interaction.start"
        assert event.interaction.id == "test-123"

    def test_parse_content_start(self, impl):
        event = impl._parse_sse_event("content.start", {"index": 0, "content": {"type": "text"}})
        assert event is not None
        assert event.event_type == "content.start"
        assert event.index == 0
        assert event.content.type == "text"

    def test_parse_content_delta(self, impl):
        event = impl._parse_sse_event("content.delta", {"index": 0, "delta": {"type": "text", "text": "Hello"}})
        assert event is not None
        assert event.event_type == "content.delta"
        assert event.delta.text == "Hello"

    def test_parse_content_stop(self, impl):
        event = impl._parse_sse_event("content.stop", {"index": 0})
        assert event is not None
        assert event.event_type == "content.stop"

    def test_parse_interaction_complete(self, impl):
        event = impl._parse_sse_event(
            "interaction.complete",
            {
                "interaction": {
                    "id": "test-123",
                    "status": "completed",
                    "model": "gemini-2.5-flash",
                    "usage": {"total_input_tokens": 10, "total_output_tokens": 5, "total_tokens": 15},
                }
            },
        )
        assert event is not None
        assert event.event_type == "interaction.complete"
        assert event.interaction.usage.total_input_tokens == 10

    def test_parse_unknown_event_returns_none(self, impl):
        event = impl._parse_sse_event("unknown.event", {"foo": "bar"})
        assert event is None
