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
    GoogleFunctionCallContent,
    GoogleFunctionDeclaration,
    GoogleFunctionResponseContent,
    GoogleGenerationConfig,
    GoogleInputTurn,
    GoogleTextContent,
    GoogleTool,
)


def _msg_to_dict(msg):
    """Convert a Pydantic message model to dict for easy assertion."""
    if hasattr(msg, "model_dump"):
        return msg.model_dump(exclude_none=True)
    return dict(msg)


@pytest.fixture
def impl():
    mock_inference = AsyncMock()
    instance = BuiltinInteractionsImpl(config=InteractionsConfig(), inference_api=mock_inference, policy=[])
    instance.store = AsyncMock()
    instance.store.get_interaction = AsyncMock(return_value=None)
    instance.store.store_interaction = AsyncMock()
    return instance


def _build_and_translate(impl, request):
    """Helper: build messages from request and translate to OpenAI params."""
    messages = impl._convert_input_to_openai(request.system_instruction, request.input)
    return impl._google_to_openai(request, messages)


class TestRequestTranslation:
    def test_simple_string_input(self, impl):
        request = GoogleCreateInteractionRequest(
            model="gemini-2.5-flash",
            input="Hello",
        )
        result = _build_and_translate(impl, request)

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
        result = _build_and_translate(impl, request)

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
        result = _build_and_translate(impl, request)

        m = _msg_to_dict(result.messages[0])
        assert m["role"] == "assistant"

    def test_system_instruction(self, impl):
        request = GoogleCreateInteractionRequest(
            model="m",
            input="Hi",
            system_instruction="You are helpful.",
        )
        result = _build_and_translate(impl, request)

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
        result = _build_and_translate(impl, request)
        assert result.temperature == 0.7

    def test_generation_config_top_p(self, impl):
        request = GoogleCreateInteractionRequest(
            model="m",
            input="Hi",
            generation_config=GoogleGenerationConfig(top_p=0.9),
        )
        result = _build_and_translate(impl, request)
        assert result.top_p == 0.9

    def test_generation_config_max_output_tokens(self, impl):
        request = GoogleCreateInteractionRequest(
            model="m",
            input="Hi",
            generation_config=GoogleGenerationConfig(max_output_tokens=500),
        )
        result = _build_and_translate(impl, request)
        assert result.max_tokens == 500

    def test_generation_config_top_k_extra_body(self, impl):
        request = GoogleCreateInteractionRequest(
            model="m",
            input="Hi",
            generation_config=GoogleGenerationConfig(top_k=40),
        )
        result = _build_and_translate(impl, request)
        assert result.model_extra.get("top_k") == 40

    def test_stream_flag(self, impl):
        request = GoogleCreateInteractionRequest(
            model="m",
            input="Hi",
            stream=True,
        )
        result = _build_and_translate(impl, request)
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
        result = _build_and_translate(impl, request)

        m = _msg_to_dict(result.messages[0])
        assert m["content"] == "Line 1\nLine 2"


class TestResponseTranslation:
    async def test_simple_text_response(self, impl):
        openai_resp = MagicMock()
        openai_resp.choices = [MagicMock()]
        openai_resp.choices[0].message = MagicMock()
        openai_resp.choices[0].message.content = "Hello!"
        openai_resp.choices[0].finish_reason = "stop"
        openai_resp.usage = MagicMock()
        openai_resp.usage.prompt_tokens = 10
        openai_resp.usage.completion_tokens = 5

        result = await impl._openai_to_google(openai_resp, "gemini-2.5-flash", [{"role": "user", "content": "Hi"}])

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

    async def test_empty_response(self, impl):
        openai_resp = MagicMock()
        openai_resp.choices = [MagicMock()]
        openai_resp.choices[0].message = MagicMock()
        openai_resp.choices[0].message.content = None
        openai_resp.choices[0].finish_reason = "stop"
        openai_resp.usage = MagicMock()
        openai_resp.usage.prompt_tokens = 5
        openai_resp.usage.completion_tokens = 0

        result = await impl._openai_to_google(openai_resp, "m", [])

        assert result.status == "completed"
        assert len(result.outputs) == 0

    async def test_missing_usage(self, impl):
        openai_resp = MagicMock()
        openai_resp.choices = [MagicMock()]
        openai_resp.choices[0].message = MagicMock()
        openai_resp.choices[0].message.content = "Hi"
        openai_resp.choices[0].finish_reason = "stop"
        openai_resp.usage = None

        result = await impl._openai_to_google(openai_resp, "m", [])

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
        async for event in impl._stream_openai_to_google(mock_stream(), "m", []):
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
        async for event in impl._stream_openai_to_google(mock_stream(), "m", []):
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
        async for event in impl._stream_openai_to_google(mock_stream(), "m", []):
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

        return BuiltinInteractionsImpl(config=InteractionsConfig(), inference_api=mock_inference, policy=[])

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
        impl = BuiltinInteractionsImpl(config=InteractionsConfig(), inference_api=mock_inference, policy=[])
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

        return BuiltinInteractionsImpl(config=InteractionsConfig(), inference_api=mock_inference, policy=[])

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
        impl = BuiltinInteractionsImpl(config=InteractionsConfig(), inference_api=AsyncMock(), policy=[])
        passthrough = {
            "base_url": "https://generativelanguage.googleapis.com/v1beta",
            "auth_headers": {"x-goog-api-key": "test-key"},
            "provider_resource_id": "gemini-2.5-flash",
            "network_config": None,
        }
        request = GoogleCreateInteractionRequest(model="gemini/gemini-2.5-flash", input="hello", stream=False)

        mock_response = MagicMock()
        mock_response.status_code = 200
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

        assert result.status_code == 200
        ctor_kwargs = async_client_ctor.call_args.kwargs
        assert ctor_kwargs["headers"]["x-goog-api-key"] == "test-key"
        assert ctor_kwargs["headers"]["content-type"] == "application/json"

        post_kwargs = mock_client.post.call_args.kwargs
        assert "params" not in post_kwargs
        assert post_kwargs["json"]["model"] == "gemini-2.5-flash"

    async def test_non_streaming_applies_network_config_client_kwargs(self, monkeypatch):
        impl = BuiltinInteractionsImpl(config=InteractionsConfig(), inference_api=AsyncMock(), policy=[])
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
        mock_response.status_code = 200
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
        impl = BuiltinInteractionsImpl(config=InteractionsConfig(), inference_api=AsyncMock(), policy=[])
        passthrough = {
            "base_url": "https://generativelanguage.googleapis.com/v1beta",
            "auth_headers": {"x-goog-api-key": "test-key"},
            "provider_resource_id": "gemini-2.5-flash",
            "network_config": None,
        }
        request = GoogleCreateInteractionRequest(model="gemini/gemini-2.5-flash", input="hello", stream=False)

        mock_response = MagicMock()
        mock_response.status_code = 200
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
            "role": "model",
            "usage": {"total_input_tokens": 10, "total_output_tokens": 5, "total_tokens": 15},
        }
        mock_response.status_code = 200

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.post = AsyncMock(return_value=mock_response)
        async_client_ctor = MagicMock(return_value=mock_client)
        monkeypatch.setattr("ogx.providers.inline.interactions.impl.httpx.AsyncClient", async_client_ctor)

        result = await impl._passthrough_request(passthrough, request)

        assert result.status_code == 200
        body = result.body
        import json as _json

        data = _json.loads(body)
        assert len(data["outputs"]) == 2
        assert data["outputs"][0]["type"] == "thought"
        assert data["outputs"][1]["type"] == "text"


class TestToolCallingRequestTranslation:
    def test_tools_converted_to_openai(self, impl):
        request = GoogleCreateInteractionRequest(
            model="m",
            input="What's the weather?",
            tools=[
                GoogleTool(
                    function_declarations=[
                        GoogleFunctionDeclaration(
                            name="get_weather",
                            description="Get current weather",
                            parameters={
                                "type": "object",
                                "properties": {"location": {"type": "string"}},
                                "required": ["location"],
                            },
                        ),
                    ]
                )
            ],
        )
        result = _build_and_translate(impl, request)

        assert result.tools is not None
        assert len(result.tools) == 1
        tool = _msg_to_dict(result.tools[0])
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "get_weather"
        assert tool["function"]["description"] == "Get current weather"
        assert tool["function"]["parameters"]["required"] == ["location"]

    def test_multiple_function_declarations(self, impl):
        request = GoogleCreateInteractionRequest(
            model="m",
            input="Help me",
            tools=[
                GoogleTool(
                    function_declarations=[
                        GoogleFunctionDeclaration(name="fn_a", description="First"),
                        GoogleFunctionDeclaration(name="fn_b", description="Second"),
                    ]
                )
            ],
        )
        result = _build_and_translate(impl, request)

        assert result.tools is not None
        assert len(result.tools) == 2

    def test_function_response_input(self, impl):
        request = GoogleCreateInteractionRequest(
            model="m",
            input=[
                GoogleInputTurn(
                    role="user",
                    content=[GoogleTextContent(text="What's the weather?")],
                ),
                GoogleInputTurn(
                    role="model",
                    content=[
                        GoogleFunctionCallContent(
                            id="call_123",
                            name="get_weather",
                            args={"location": "NYC"},
                        )
                    ],
                ),
                GoogleInputTurn(
                    role="user",
                    content=[
                        GoogleFunctionResponseContent(
                            call_id="call_123",
                            name="get_weather",
                            response={"temperature": 72},
                        )
                    ],
                ),
            ],
        )
        result = _build_and_translate(impl, request)

        # user message, assistant with tool_calls, tool message
        assert len(result.messages) == 3
        m0 = _msg_to_dict(result.messages[0])
        m1 = _msg_to_dict(result.messages[1])
        m2 = _msg_to_dict(result.messages[2])

        assert m0["role"] == "user"
        assert m0["content"] == "What's the weather?"

        assert m1["role"] == "assistant"
        assert "tool_calls" in m1
        assert m1["tool_calls"][0]["id"] == "call_123"
        assert m1["tool_calls"][0]["function"]["name"] == "get_weather"

        assert m2["role"] == "tool"
        assert m2["tool_call_id"] == "call_123"
        assert '"temperature": 72' in m2["content"]

    def test_model_turn_with_text_and_function_call(self, impl):
        request = GoogleCreateInteractionRequest(
            model="m",
            input=[
                GoogleInputTurn(
                    role="model",
                    content=[
                        GoogleTextContent(text="Let me check the weather."),
                        GoogleFunctionCallContent(
                            id="call_abc",
                            name="get_weather",
                            args={"location": "Paris"},
                        ),
                    ],
                ),
            ],
        )
        result = _build_and_translate(impl, request)

        assert len(result.messages) == 1
        m = _msg_to_dict(result.messages[0])
        assert m["role"] == "assistant"
        assert m["content"] == "Let me check the weather."
        assert len(m["tool_calls"]) == 1
        assert m["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_no_tools_when_none(self, impl):
        request = GoogleCreateInteractionRequest(model="m", input="Hello")
        result = _build_and_translate(impl, request)
        assert result.tools is None


class TestToolCallingResponseTranslation:
    async def test_function_call_output(self, impl):
        openai_resp = MagicMock()
        openai_resp.choices = [MagicMock()]
        openai_resp.choices[0].message = MagicMock()
        openai_resp.choices[0].message.content = None

        tc = MagicMock()
        tc.id = "call_xyz"
        tc.function = MagicMock()
        tc.function.name = "get_weather"
        tc.function.arguments = '{"location": "NYC"}'
        openai_resp.choices[0].message.tool_calls = [tc]
        openai_resp.choices[0].finish_reason = "tool_calls"
        openai_resp.usage = MagicMock()
        openai_resp.usage.prompt_tokens = 20
        openai_resp.usage.completion_tokens = 10

        result = await impl._openai_to_google(openai_resp, "m", [])

        assert len(result.outputs) == 1
        fc = result.outputs[0]
        assert fc.type == "function_call"
        assert fc.id == "call_xyz"
        assert fc.name == "get_weather"
        assert fc.args == {"location": "NYC"}

    async def test_text_and_function_call_output(self, impl):
        openai_resp = MagicMock()
        openai_resp.choices = [MagicMock()]
        openai_resp.choices[0].message = MagicMock()
        openai_resp.choices[0].message.content = "I'll check the weather."

        tc = MagicMock()
        tc.id = "call_abc"
        tc.function = MagicMock()
        tc.function.name = "get_weather"
        tc.function.arguments = '{"location": "London"}'
        openai_resp.choices[0].message.tool_calls = [tc]
        openai_resp.choices[0].finish_reason = "tool_calls"
        openai_resp.usage = MagicMock()
        openai_resp.usage.prompt_tokens = 10
        openai_resp.usage.completion_tokens = 15

        result = await impl._openai_to_google(openai_resp, "m", [])

        assert len(result.outputs) == 2
        assert result.outputs[0].type == "text"
        assert result.outputs[0].text == "I'll check the weather."
        assert result.outputs[1].type == "function_call"
        assert result.outputs[1].name == "get_weather"

    async def test_invalid_function_arguments_json(self, impl):
        openai_resp = MagicMock()
        openai_resp.choices = [MagicMock()]
        openai_resp.choices[0].message = MagicMock()
        openai_resp.choices[0].message.content = None

        tc = MagicMock()
        tc.id = "call_bad"
        tc.function = MagicMock()
        tc.function.name = "fn"
        tc.function.arguments = "not valid json"
        openai_resp.choices[0].message.tool_calls = [tc]
        openai_resp.choices[0].finish_reason = "tool_calls"
        openai_resp.usage = None

        result = await impl._openai_to_google(openai_resp, "m", [])

        assert result.outputs[0].args == {}


class TestToolCallingStreamingTranslation:
    async def test_streaming_function_call(self, impl):
        chunks = []

        # First chunk: tool_call start with name
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta = MagicMock()
        chunk1.choices[0].delta.content = None
        tc_delta1 = MagicMock()
        tc_delta1.index = 0
        tc_delta1.id = "call_stream1"
        tc_delta1.function = MagicMock()
        tc_delta1.function.name = "get_weather"
        tc_delta1.function.arguments = ""
        chunk1.choices[0].delta.tool_calls = [tc_delta1]
        chunk1.choices[0].finish_reason = None
        chunk1.usage = None
        chunks.append(chunk1)

        # Second chunk: arguments part 1
        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta = MagicMock()
        chunk2.choices[0].delta.content = None
        tc_delta2 = MagicMock()
        tc_delta2.index = 0
        tc_delta2.id = None
        tc_delta2.function = MagicMock()
        tc_delta2.function.name = None
        tc_delta2.function.arguments = '{"location":'
        chunk2.choices[0].delta.tool_calls = [tc_delta2]
        chunk2.choices[0].finish_reason = None
        chunk2.usage = None
        chunks.append(chunk2)

        # Third chunk: arguments part 2
        chunk3 = MagicMock()
        chunk3.choices = [MagicMock()]
        chunk3.choices[0].delta = MagicMock()
        chunk3.choices[0].delta.content = None
        tc_delta3 = MagicMock()
        tc_delta3.index = 0
        tc_delta3.id = None
        tc_delta3.function = MagicMock()
        tc_delta3.function.name = None
        tc_delta3.function.arguments = ' "NYC"}'
        chunk3.choices[0].delta.tool_calls = [tc_delta3]
        chunk3.choices[0].finish_reason = "tool_calls"
        chunk3.usage = None
        chunks.append(chunk3)

        async def mock_stream():
            for c in chunks:
                yield c

        events = []
        async for event in impl._stream_openai_to_google(mock_stream(), "m", []):
            events.append(event)

        assert events[0].event_type == "interaction.start"
        # content.start for function_call
        assert events[1].event_type == "content.start"
        assert events[1].content.type == "function_call"
        assert events[1].content.id == "call_stream1"
        assert events[1].content.name == "get_weather"
        # content.delta with args
        assert events[2].event_type == "content.delta"
        assert events[2].delta.type == "function_call"
        assert events[2].delta.args == '{"location":'
        assert events[3].event_type == "content.delta"
        assert events[3].delta.args == ' "NYC"}'
        # content.stop
        assert events[4].event_type == "content.stop"
        # interaction.complete
        assert events[5].event_type == "interaction.complete"

    async def test_streaming_text_then_function_call(self, impl):
        chunks = []

        # Text chunk
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta = MagicMock()
        chunk1.choices[0].delta.content = "Let me check."
        chunk1.choices[0].delta.tool_calls = None
        chunk1.choices[0].finish_reason = None
        chunk1.usage = None
        chunks.append(chunk1)

        # Tool call chunk
        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta = MagicMock()
        chunk2.choices[0].delta.content = None
        tc_delta = MagicMock()
        tc_delta.index = 0
        tc_delta.id = "call_mixed"
        tc_delta.function = MagicMock()
        tc_delta.function.name = "search"
        tc_delta.function.arguments = '{"q": "test"}'
        chunk2.choices[0].delta.tool_calls = [tc_delta]
        chunk2.choices[0].finish_reason = "tool_calls"
        chunk2.usage = None
        chunks.append(chunk2)

        async def mock_stream():
            for c in chunks:
                yield c

        events = []
        async for event in impl._stream_openai_to_google(mock_stream(), "m", []):
            events.append(event)

        event_types = [e.event_type for e in events]
        assert event_types == [
            "interaction.start",
            "content.start",  # text block
            "content.delta",  # text delta
            "content.stop",  # text block closed
            "content.start",  # function_call block
            "content.delta",  # args delta
            "content.stop",  # function_call block closed
            "interaction.complete",
        ]
        # Text block
        assert events[1].content.type == "text"
        assert events[2].delta.text == "Let me check."
        # Function call block
        assert events[4].content.type == "function_call"
        assert events[4].content.name == "search"
        assert events[5].delta.args == '{"q": "test"}'
