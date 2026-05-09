# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import base64
import logging  # allow-direct-logging
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from ogx.providers.remote.inference.vertexai.config import VertexAIConfig
from ogx.providers.remote.inference.vertexai.vertexai import VertexAIInferenceAdapter
from ogx_api import OpenAIChatCompletionChunk, OpenAIEmbeddingsRequestWithExtraBody
from ogx_api.inference.models import OpenAIChatCompletionRequestWithExtraBody

from .conftest import _async_pager


class TestVertexAIEmbeddings:
    """Tests for openai_embeddings() implementation."""

    async def test_single_string_returns_one_embedding(self, make_adapter_with_mock_embed):
        """Test that single string returns one embedding."""
        adapter, _ = make_adapter_with_mock_embed([[0.1, 0.2, 0.3]])

        params = OpenAIEmbeddingsRequestWithExtraBody(model="text-embedding-004", input="hello world")
        result = await adapter.openai_embeddings(params)

        assert len(result.data) == 1
        assert result.data[0].embedding == [0.1, 0.2, 0.3]
        assert result.data[0].index == 0
        assert result.data[0].object == "embedding"
        assert result.model == "text-embedding-004"

    async def test_batch_strings_returns_multiple_embeddings(self, make_adapter_with_mock_embed):
        """Test that batch strings returns multiple embeddings."""
        adapter, _ = make_adapter_with_mock_embed([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

        params = OpenAIEmbeddingsRequestWithExtraBody(model="text-embedding-004", input=["a", "b", "c"])
        result = await adapter.openai_embeddings(params)

        assert len(result.data) == 3
        assert result.data[0].embedding == [0.1, 0.2]
        assert result.data[1].embedding == [0.3, 0.4]
        assert result.data[2].embedding == [0.5, 0.6]
        for i, item in enumerate(result.data):
            assert item.index == i

    async def test_dimensions_forwarded_as_output_dimensionality(self, make_adapter_with_mock_embed):
        """Test that dimensions forwarded as output dimensionality."""
        capture: dict = {}
        adapter, _ = make_adapter_with_mock_embed([[0.1]], capture)

        params = OpenAIEmbeddingsRequestWithExtraBody(model="text-embedding-004", input="hello", dimensions=512)
        await adapter.openai_embeddings(params)

        assert "config" in capture
        assert capture["config"].output_dimensionality == 512

    async def test_user_forwarded_as_labels(self, make_adapter_with_mock_embed):
        """Test that user forwarded as labels."""
        capture: dict = {}
        adapter, _ = make_adapter_with_mock_embed([[0.1]], capture)

        params = OpenAIEmbeddingsRequestWithExtraBody(model="text-embedding-004", input="hello", user="alice")
        await adapter.openai_embeddings(params)

        assert "config" in capture
        assert capture["config"].labels == {"user": "alice"}

    async def test_base64_encoding_format(self, make_adapter_with_mock_embed):
        """Test that base64 encoding format."""
        import struct as _struct

        values = [0.1, 0.2, 0.3]
        adapter, _ = make_adapter_with_mock_embed([values])

        params = OpenAIEmbeddingsRequestWithExtraBody(
            model="text-embedding-004", input="hello", encoding_format="base64"
        )
        result = await adapter.openai_embeddings(params)

        embedding = result.data[0].embedding
        assert isinstance(embedding, str)
        decoded_bytes = base64.b64decode(embedding)
        decoded_floats = list(_struct.unpack(f"{len(values)}f", decoded_bytes))
        assert len(decoded_floats) == len(values)
        for orig, dec in zip(values, decoded_floats, strict=False):
            assert abs(orig - dec) < 1e-6

    async def test_token_array_input_raises_value_error(self, make_adapter_with_mock_embed):
        """Test that token array input raises value error."""
        adapter, _ = make_adapter_with_mock_embed([[0.1]])

        params = OpenAIEmbeddingsRequestWithExtraBody(
            model="text-embedding-004",
            input=cast(Any, [1, 2, 3]),  # token array, not text
        )
        with pytest.raises((ValueError, AttributeError)):
            await adapter.openai_embeddings(params)

    async def test_usage_returns_zeros(self, make_adapter_with_mock_embed):
        """Test that usage returns zeros."""
        adapter, _ = make_adapter_with_mock_embed([[0.1, 0.2]])

        params = OpenAIEmbeddingsRequestWithExtraBody(model="text-embedding-004", input="test")
        result = await adapter.openai_embeddings(params)

        assert result.usage.prompt_tokens == 0
        assert result.usage.total_tokens == 0

    async def test_no_config_when_no_options(self, make_adapter_with_mock_embed):
        """When no dimensions or user are set, config should be None."""
        capture: dict = {}
        adapter, _ = make_adapter_with_mock_embed([[0.1]], capture)

        params = OpenAIEmbeddingsRequestWithExtraBody(model="text-embedding-004", input="hello")
        await adapter.openai_embeddings(params)

        assert capture.get("config") is None

    async def test_embedding_usage_with_real_tokens(self, make_adapter_with_mock_embed):
        """When response has usage_metadata, usage shows real token counts."""
        usage_metadata = SimpleNamespace(prompt_token_count=10, total_token_count=15)
        adapter, _ = make_adapter_with_mock_embed([[0.1, 0.2]], usage_metadata=usage_metadata)

        params = OpenAIEmbeddingsRequestWithExtraBody(model="text-embedding-004", input="test")
        result = await adapter.openai_embeddings(params)

        assert result.usage.prompt_tokens == 10
        assert result.usage.total_tokens == 15

    async def test_embedding_usage_fallback_when_no_metadata(self, make_adapter_with_mock_embed):
        """When response has no usage_metadata, usage falls back to zeros."""
        adapter, _ = make_adapter_with_mock_embed([[0.1, 0.2]])

        params = OpenAIEmbeddingsRequestWithExtraBody(model="text-embedding-004", input="test")
        result = await adapter.openai_embeddings(params)

        assert result.usage.prompt_tokens == 0
        assert result.usage.total_tokens == 0

    async def test_embedding_usage_fallback_when_metadata_missing_fields(self, make_adapter_with_mock_embed):
        """When usage_metadata exists but fields are missing, fall back to zeros."""
        usage_metadata = SimpleNamespace()  # No token count fields
        adapter, _ = make_adapter_with_mock_embed([[0.1, 0.2]], usage_metadata=usage_metadata)

        params = OpenAIEmbeddingsRequestWithExtraBody(model="text-embedding-004", input="test")
        result = await adapter.openai_embeddings(params)

        assert result.usage.prompt_tokens == 0
        assert result.usage.total_tokens == 0


class TestDroppedParameterWarnings:
    """Test that unsupported parameters generate appropriate warning/debug logs."""

    @pytest.mark.parametrize(
        "param_name,param_value,log_level,expected_text",
        [
            pytest.param("logit_bias", {"50256": -100.0}, logging.WARNING, "logit_bias", id="logit_bias"),
            pytest.param("prompt_cache_key", "mykey", logging.WARNING, "prompt_cache_key", id="prompt_cache_key"),
            pytest.param("user", "test-user", logging.DEBUG, "user", id="user"),
            pytest.param("safety_identifier", "test-id", logging.DEBUG, "safety_identifier", id="safety_identifier"),
        ],
    )
    async def test_unsupported_param_logged(
        self, monkeypatch, caplog, patch_chat_completion_dependencies, param_name, param_value, log_level, expected_text
    ):
        """Test that unsupported param logged."""
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        patch_chat_completion_dependencies(adapter)

        payload: dict[str, Any] = {
            "model": "google/gemini-2.5-flash",
            "messages": [{"role": "user", "content": "hi"}],
            param_name: param_value,
        }
        params = OpenAIChatCompletionRequestWithExtraBody.model_validate(payload)

        with caplog.at_level(logging.DEBUG, logger="ogx.providers.remote.inference.vertexai.vertexai"):
            await adapter.openai_chat_completion(params)

        assert any(expected_text in r.message for r in caplog.records if r.levelno == log_level)

    @pytest.mark.parametrize(
        "value,should_warn",
        [
            pytest.param(False, True, id="false_warns"),
            pytest.param(True, False, id="true_no_warn"),
        ],
    )
    async def test_parallel_tool_calls_warning(
        self, monkeypatch, caplog, patch_chat_completion_dependencies, value, should_warn
    ):
        """Test that parallel tool calls warning."""
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        patch_chat_completion_dependencies(adapter)

        params = OpenAIChatCompletionRequestWithExtraBody(
            model="google/gemini-2.5-flash",
            messages=cast(Any, [{"role": "user", "content": "hi"}]),
            parallel_tool_calls=value,
        )

        with caplog.at_level(logging.WARNING):
            await adapter.openai_chat_completion(params)

        found = any("parallel tool calls" in r.message for r in caplog.records if r.levelno == logging.WARNING)
        assert found == should_warn


class TestServiceTier:
    """Test that service_tier is mapped and forwarded to GenerateContentConfig."""

    _OMITTED = object()  # sentinel: service_tier should not appear on the config

    @pytest.fixture
    def capture_generation_config(self, monkeypatch, patch_chat_completion_dependencies):
        """Fixture that runs a chat completion and returns the GenerateContentConfig produced.

        Returns an async callable: ``config = await fn(service_tier=...)``
        Pass ``service_tier=None`` (or omit) to test the "not set" case.
        """

        async def _run(*, service_tier: str | None = None) -> Any:
            """Run a chat completion with the given service_tier and return the config."""
            adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
            captured: dict[str, Any] = {}
            original_build = adapter._build_generation_config

            def _capturing_build(*args, **kwargs):
                """Intercept _build_generation_config to capture the config object."""
                config = original_build(*args, **kwargs)
                captured["config"] = config
                return config

            patch_chat_completion_dependencies(adapter)
            monkeypatch.setattr(adapter, "_build_generation_config", _capturing_build)

            payload: dict[str, Any] = {
                "model": "gemini-2.5-flash",
                "messages": [{"role": "user", "content": "hi"}],
            }
            if service_tier is not None:
                payload["service_tier"] = service_tier

            params = OpenAIChatCompletionRequestWithExtraBody.model_validate(payload)
            await adapter.openai_chat_completion(params)
            return captured["config"]

        return _run

    @pytest.mark.parametrize(
        "tier_input,expected",
        [
            pytest.param("flex", "flex", id="flex"),
            pytest.param("priority", "priority", id="priority"),
            pytest.param("default", "standard", id="default_maps_to_standard"),
            pytest.param("auto", _OMITTED, id="auto_omitted"),
            pytest.param(None, _OMITTED, id="none_omitted"),
        ],
    )
    async def test_service_tier_mapping(self, capture_generation_config, tier_input, expected):
        """Verify each service_tier value is correctly mapped (or omitted) on the config."""
        config = await capture_generation_config(service_tier=tier_input)

        if expected is self._OMITTED:
            assert getattr(config, "service_tier", None) is None
        else:
            assert config.service_tier == expected

    async def test_unknown_service_tier_returns_none(self):
        """An unrecognized service_tier value returns None (caller omits the field)."""
        assert VertexAIInferenceAdapter._convert_service_tier("bogus_tier") is None

    @pytest.mark.parametrize(
        "tier_value",
        [
            pytest.param("flex", id="flex"),
            pytest.param("priority", id="priority"),
            pytest.param("default", id="default"),
            pytest.param("auto", id="auto"),
        ],
    )
    async def test_supported_values_produce_no_warnings(self, caplog, patch_chat_completion_dependencies, tier_value):
        """Supported service_tier values should not produce any warnings."""
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        patch_chat_completion_dependencies(adapter)

        params = OpenAIChatCompletionRequestWithExtraBody.model_validate(
            {
                "model": "gemini-2.5-flash",
                "messages": [{"role": "user", "content": "hi"}],
                "service_tier": tier_value,
            }
        )

        with caplog.at_level(logging.WARNING, logger="ogx.providers.remote.inference.vertexai.vertexai"):
            await adapter.openai_chat_completion(params)

        assert not any("service_tier" in r.message for r in caplog.records if r.levelno == logging.WARNING)


class TestTelemetryStreamOptions:
    """Test that telemetry stream options are injected when appropriate."""

    def _patch_stream_chat_completion(self, monkeypatch, adapter: VertexAIInferenceAdapter) -> dict[str, Any]:
        """Patch dependencies for streaming chat completion."""
        fake_client = SimpleNamespace(
            aio=SimpleNamespace(models=SimpleNamespace(generate_content=AsyncMock(return_value=None)))
        )
        stream_call_args: dict[str, Any] = {}

        async def _provider_model_id(_: str) -> str:
            """Return a fixed provider model identifier."""
            return "gemini-2.5-flash"

        async def _stream_chat_completion(client, model_id, contents, config, model, stream_options=None):
            """Handle stream chat completion."""
            stream_call_args["stream_options"] = stream_options
            return _async_pager([])

        monkeypatch.setattr(adapter, "_get_provider_model_id", _provider_model_id)
        monkeypatch.setattr(adapter, "_validate_model_allowed", lambda _: None)
        monkeypatch.setattr(adapter, "_get_client", lambda: fake_client)
        monkeypatch.setattr(adapter, "_build_generation_config", lambda *_args, **_kwargs: object())
        monkeypatch.setattr(
            "ogx.providers.remote.inference.vertexai.vertexai.converters.convert_openai_messages_to_gemini",
            lambda messages: (None, [{"role": "user", "parts": [{"text": "ok"}]}]),
        )
        monkeypatch.setattr(
            "ogx.providers.remote.inference.vertexai.vertexai.converters.convert_openai_tools_to_gemini",
            lambda _tools: None,
        )
        monkeypatch.setattr(adapter, "_stream_chat_completion", _stream_chat_completion)

        return stream_call_args

    async def test_stream_options_injected_when_telemetry_active(self, monkeypatch):
        """When telemetry span is recording, stream_options should include include_usage=True."""
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        stream_call_args = self._patch_stream_chat_completion(monkeypatch, adapter)

        # Mock opentelemetry to return a recording span
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        mock_trace = MagicMock()
        mock_trace.get_current_span.return_value = mock_span
        monkeypatch.setattr("opentelemetry.trace", mock_trace)

        params = OpenAIChatCompletionRequestWithExtraBody(
            model="google/gemini-2.5-flash",
            messages=cast(Any, [{"role": "user", "content": "hi"}]),
            stream=True,
        )

        # Consume the async generator
        result = await adapter.openai_chat_completion(params)
        async for _ in cast(AsyncIterator[OpenAIChatCompletionChunk], result):
            pass

        # Verify stream_options was passed with include_usage=True
        assert stream_call_args.get("stream_options") is not None
        assert stream_call_args["stream_options"].get("include_usage") is True

    async def test_stream_options_not_injected_when_telemetry_inactive(self, monkeypatch):
        """When telemetry span is not recording, stream_options should be None."""
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        stream_call_args = self._patch_stream_chat_completion(monkeypatch, adapter)

        # Mock opentelemetry to return a non-recording span
        mock_span = MagicMock()
        mock_span.is_recording.return_value = False
        mock_trace = MagicMock()
        mock_trace.get_current_span.return_value = mock_span
        monkeypatch.setattr("opentelemetry.trace", mock_trace)

        params = OpenAIChatCompletionRequestWithExtraBody(
            model="google/gemini-2.5-flash",
            messages=cast(Any, [{"role": "user", "content": "hi"}]),
            stream=True,
        )

        # Consume the async generator
        result = await adapter.openai_chat_completion(params)
        async for _ in cast(AsyncIterator[OpenAIChatCompletionChunk], result):
            pass

        # Verify stream_options was not modified
        assert stream_call_args.get("stream_options") is None


class TestEmbeddingsModelExtra:
    """Test that model_extra parameter generates debug logs in embeddings."""

    async def test_model_extra_debug_log(self, make_adapter_with_mock_embed, caplog):
        """model_extra parameter should generate a DEBUG log."""
        adapter, _ = make_adapter_with_mock_embed([[0.1, 0.2]])

        params = OpenAIEmbeddingsRequestWithExtraBody.model_validate(
            {
                "model": "text-embedding-004",
                "input": "hello",
                "custom_param": "value",
                "another_param": 123,
            }
        )

        with caplog.at_level(logging.DEBUG, logger="ogx.providers.remote.inference.vertexai.vertexai"):
            await adapter.openai_embeddings(params)

        # Should have a DEBUG log mentioning model_extra or extra body parameters
        assert any(
            ("model_extra" in record.message or "extra body parameters" in record.message)
            for record in caplog.records
            if record.levelno == logging.DEBUG
        )

    async def test_no_model_extra_no_debug_log(self, make_adapter_with_mock_embed, caplog):
        """When model_extra is empty, no debug log should be generated."""
        adapter, _ = make_adapter_with_mock_embed([[0.1, 0.2]])

        params = OpenAIEmbeddingsRequestWithExtraBody(
            model="text-embedding-004",
            input="hello",
        )

        with caplog.at_level(logging.DEBUG, logger="ogx.providers.remote.inference.vertexai.vertexai"):
            await adapter.openai_embeddings(params)

        # Should NOT have a debug log about model_extra
        assert not any(
            ("model_extra" in record.message or "extra body parameters" in record.message)
            for record in caplog.records
            if record.levelno == logging.DEBUG
        )


class TestDeprecatedFunctionCalling:
    async def test_functions_converted_to_tools_when_tools_absent(
        self, monkeypatch, caplog, patch_chat_completion_dependencies
    ):
        """Test that functions converted to tools when tools absent."""
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        captured = patch_chat_completion_dependencies(
            adapter,
            capture_tools=True,
            capture_generation_kwargs=True,
        )

        functions = [{"name": "get_weather", "description": "Get weather", "parameters": {"type": "object"}}]
        params = OpenAIChatCompletionRequestWithExtraBody.model_validate(
            {
                "model": "gemini-2.5-flash",
                "messages": [{"role": "user", "content": "hi"}],
                "functions": functions,
            }
        )

        with caplog.at_level(logging.WARNING, logger="ogx.providers.remote.inference.vertexai.vertexai"):
            await adapter.openai_chat_completion(params)

        assert any("functions" in record.message and "deprecated" in record.message for record in caplog.records)
        tools_passed = captured["tools_passed"]
        assert tools_passed is not None
        assert len(tools_passed) == 1
        assert tools_passed[0]["type"] == "function"
        assert tools_passed[0]["function"]["name"] == "get_weather"

    async def test_tools_takes_priority_over_functions(self, monkeypatch, caplog, patch_chat_completion_dependencies):
        """Test that tools takes priority over functions."""
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        captured = patch_chat_completion_dependencies(
            adapter,
            capture_tools=True,
            capture_generation_kwargs=True,
        )

        modern_tools = [{"type": "function", "function": {"name": "modern_tool", "description": "Modern"}}]
        functions = [{"name": "legacy_func", "description": "Legacy"}]
        params = OpenAIChatCompletionRequestWithExtraBody.model_validate(
            {
                "model": "gemini-2.5-flash",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": modern_tools,
                "functions": functions,
            }
        )

        with caplog.at_level(logging.WARNING, logger="ogx.providers.remote.inference.vertexai.vertexai"):
            await adapter.openai_chat_completion(params)

        assert not any("functions" in record.message and "deprecated" in record.message for record in caplog.records)
        tools_passed = captured["tools_passed"]
        assert tools_passed is not None
        assert len(tools_passed) == 1
        assert tools_passed[0]["function"]["name"] == "modern_tool"

    async def test_function_call_converted_to_tool_choice(
        self, monkeypatch, caplog, patch_chat_completion_dependencies
    ):
        """Test that function call converted to tool choice."""
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        captured = patch_chat_completion_dependencies(
            adapter,
            capture_tools=True,
            capture_generation_kwargs=True,
        )

        params = OpenAIChatCompletionRequestWithExtraBody.model_validate(
            {
                "model": "gemini-2.5-flash",
                "messages": [{"role": "user", "content": "hi"}],
                "function_call": {"name": "get_weather"},
            }
        )

        with caplog.at_level(logging.WARNING, logger="ogx.providers.remote.inference.vertexai.vertexai"):
            await adapter.openai_chat_completion(params)

        assert any("function_call" in record.message and "deprecated" in record.message for record in caplog.records)
        kwargs = captured["build_generation_config_kwargs"]
        assert kwargs["tool_choice"] == {"type": "function", "function": {"name": "get_weather"}}

    async def test_tool_choice_takes_priority_over_function_call(
        self, monkeypatch, caplog, patch_chat_completion_dependencies
    ):
        """Test that tool choice takes priority over function call."""
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        captured = patch_chat_completion_dependencies(
            adapter,
            capture_tools=True,
            capture_generation_kwargs=True,
        )

        params = OpenAIChatCompletionRequestWithExtraBody.model_validate(
            {
                "model": "gemini-2.5-flash",
                "messages": [{"role": "user", "content": "hi"}],
                "tool_choice": "required",
                "function_call": {"name": "get_weather"},
            }
        )

        with caplog.at_level(logging.WARNING, logger="ogx.providers.remote.inference.vertexai.vertexai"):
            await adapter.openai_chat_completion(params)

        assert not any(
            "function_call" in record.message and "deprecated" in record.message for record in caplog.records
        )
        kwargs = captured["build_generation_config_kwargs"]
        assert kwargs["tool_choice"] == "required"
