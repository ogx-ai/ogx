# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import base64
import logging  # allow-direct-logging
import ssl
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from ogx.providers.remote.inference.vertexai.config import VertexAIConfig
from ogx.providers.remote.inference.vertexai.vertexai import VertexAIInferenceAdapter, _build_http_options
from ogx.providers.utils.inference.model_registry import NetworkConfig, ProxyConfig, TimeoutConfig, TLSConfig
from ogx_api.inference.models import OpenAIChatCompletionRequestWithExtraBody

from .conftest import _make_fake_streaming_chunk


class TestOpenAIChatCompletionImagePreDownload:
    async def test_http_image_url_is_downloaded(self, monkeypatch, patch_chat_completion_dependencies):
        """Test that http image url is downloaded."""
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        conversion_inputs = patch_chat_completion_dependencies(adapter, capture_messages=True)

        localize_image_content_mock = AsyncMock(return_value=(b"fake image bytes", "jpeg"))
        monkeypatch.setattr(
            "ogx.providers.remote.inference.vertexai.vertexai.localize_image_content",
            localize_image_content_mock,
        )

        params = OpenAIChatCompletionRequestWithExtraBody(
            model="google/gemini-2.5-flash",
            messages=cast(
                Any,
                [
                    {
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": "https://example.com/image.jpeg"}}],
                    }
                ],
            ),
        )

        await adapter.openai_chat_completion(params)

        localize_image_content_mock.assert_awaited_once_with("https://example.com/image.jpeg")
        converted_messages = conversion_inputs["messages"]
        image_url = converted_messages[0].content[0].image_url.url
        expected_data = base64.b64encode(b"fake image bytes").decode("utf-8")
        assert image_url == f"data:image/jpeg;base64,{expected_data}"

    async def test_data_uri_not_re_downloaded(self, monkeypatch, patch_chat_completion_dependencies):
        """Test that data uri not re downloaded."""
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        conversion_inputs = patch_chat_completion_dependencies(adapter, capture_messages=True)

        localize_image_content_mock = AsyncMock()
        monkeypatch.setattr(
            "ogx.providers.remote.inference.vertexai.vertexai.localize_image_content",
            localize_image_content_mock,
        )

        data_uri = "data:image/png;base64,ZmFrZQ=="
        params = OpenAIChatCompletionRequestWithExtraBody(
            model="google/gemini-2.5-flash",
            messages=cast(
                Any,
                [
                    {
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": data_uri}}],
                    }
                ],
            ),
        )

        await adapter.openai_chat_completion(params)

        localize_image_content_mock.assert_not_called()
        converted_messages = conversion_inputs["messages"]
        assert converted_messages[0].content[0].image_url.url == data_uri

    async def test_failed_download_raises_value_error(self, monkeypatch, patch_chat_completion_dependencies):
        """Test that failed download raises value error."""
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        patch_chat_completion_dependencies(adapter)

        localize_image_content_mock = AsyncMock(return_value=None)
        monkeypatch.setattr(
            "ogx.providers.remote.inference.vertexai.vertexai.localize_image_content",
            localize_image_content_mock,
        )

        params = OpenAIChatCompletionRequestWithExtraBody(
            model="google/gemini-2.5-flash",
            messages=cast(
                Any,
                [
                    {
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": "http://example.com/image.png"}}],
                    }
                ],
            ),
        )

        with pytest.raises(ValueError, match="Failed to localize image content"):
            await adapter.openai_chat_completion(params)


class TestCollectSamplingParams:
    """Test VertexAIInferenceAdapter._collect_sampling_params() for new sampling parameters."""

    @pytest.mark.parametrize(
        "param_name,param_value,expected_key,expected_value",
        [
            pytest.param("frequency_penalty", 0.5, "frequency_penalty", 0.5, id="frequency_penalty"),
            pytest.param("presence_penalty", 0.5, "presence_penalty", 0.5, id="presence_penalty"),
            pytest.param("seed", 42, "seed", 42, id="seed"),
        ],
    )
    def test_single_sampling_param_forwarded(self, param_name, param_value, expected_key, expected_value):
        """Test that single sampling param forwarded."""
        params = OpenAIChatCompletionRequestWithExtraBody(
            model="test-model",
            messages=cast(Any, [{"role": "user", "content": "hi"}]),
            **{param_name: param_value},
        )
        result = VertexAIInferenceAdapter._collect_sampling_params(params)
        assert expected_key in result
        assert result[expected_key] == expected_value

    def test_all_three_params_combined(self):
        """Test that all three new params are collected together."""
        params = OpenAIChatCompletionRequestWithExtraBody(
            model="test-model",
            messages=cast(Any, [{"role": "user", "content": "hi"}]),
            frequency_penalty=0.3,
            presence_penalty=0.4,
            seed=123,
        )
        result = VertexAIInferenceAdapter._collect_sampling_params(params)
        assert result["frequency_penalty"] == 0.3
        assert result["presence_penalty"] == 0.4
        assert result["seed"] == 123

    def test_none_values_are_excluded(self):
        """Test that None values are not included in result."""
        params = OpenAIChatCompletionRequestWithExtraBody(
            model="test-model",
            messages=cast(Any, [{"role": "user", "content": "hi"}]),
        )
        result = VertexAIInferenceAdapter._collect_sampling_params(params)
        assert "frequency_penalty" not in result
        assert "presence_penalty" not in result
        assert "seed" not in result

    def test_zero_values_are_forwarded(self):
        """Test that zero values are forwarded (not dropped as falsy)."""
        params = OpenAIChatCompletionRequestWithExtraBody(
            model="test-model",
            messages=cast(Any, [{"role": "user", "content": "hi"}]),
            frequency_penalty=0.0,
            presence_penalty=0.0,
            seed=0,
        )
        result = VertexAIInferenceAdapter._collect_sampling_params(params)
        assert "frequency_penalty" in result
        assert result["frequency_penalty"] == 0.0
        assert "presence_penalty" in result
        assert result["presence_penalty"] == 0.0
        assert "seed" in result
        assert result["seed"] == 0

    def test_existing_params_unaffected(self):
        """Test that existing params (temperature, top_p) still work alongside new params."""
        params = OpenAIChatCompletionRequestWithExtraBody(
            model="test-model",
            messages=cast(Any, [{"role": "user", "content": "hi"}]),
            temperature=0.7,
            top_p=0.9,
            frequency_penalty=0.2,
            presence_penalty=0.3,
            seed=99,
        )
        result = VertexAIInferenceAdapter._collect_sampling_params(params)
        assert result["temperature"] == 0.7
        assert result["top_p"] == 0.9
        assert result["frequency_penalty"] == 0.2
        assert result["presence_penalty"] == 0.3
        assert result["seed"] == 99

    @pytest.mark.parametrize(
        "input_param,input_value,expected_key,expected_value",
        [
            pytest.param("logprobs", True, "response_logprobs", True, id="logprobs_to_response_logprobs"),
            pytest.param("top_logprobs", 5, "logprobs", 5, id="top_logprobs_to_logprobs"),
        ],
    )
    def test_logprobs_mapping(self, input_param, input_value, expected_key, expected_value):
        """Test that logprobs mapping."""
        params = OpenAIChatCompletionRequestWithExtraBody(
            model="test-model",
            messages=cast(Any, [{"role": "user", "content": "hi"}]),
            **{input_param: input_value},
        )
        result = VertexAIInferenceAdapter._collect_sampling_params(params)
        assert expected_key in result
        assert result[expected_key] == expected_value

    def test_logprobs_none_excluded(self):
        """Test that logprobs and top_logprobs keys are absent when params are None."""
        params = OpenAIChatCompletionRequestWithExtraBody(
            model="test-model",
            messages=cast(Any, [{"role": "user", "content": "hi"}]),
        )
        result = VertexAIInferenceAdapter._collect_sampling_params(params)
        assert "response_logprobs" not in result
        assert "logprobs" not in result


class TestVertexAINetworkConfig:
    """Tests for _build_http_options() conversion from NetworkConfig to HttpOptions."""

    def test_build_http_options_none_returns_none(self):
        """Test that build http options none returns none."""
        assert _build_http_options(None) is None

    def test_build_http_options_empty_config_returns_none(self):
        # All fields are None by default — nothing to configure
        """Test that build http options empty config returns none."""
        assert _build_http_options(NetworkConfig()) is None

    def test_build_http_options_headers_only(self):
        """Test that build http options headers only."""
        result = _build_http_options(NetworkConfig(headers={"X-Custom": "value"}))
        assert result is not None
        assert result.headers == {"X-Custom": "value"}

    def test_build_http_options_float_timeout_converts_to_ms(self):
        """Test that build http options float timeout converts to ms."""
        result = _build_http_options(NetworkConfig(timeout=30.0))
        assert result is not None
        assert result.timeout == 30000

    def test_build_http_options_timeout_config_read_wins(self):
        """Test that build http options timeout config read wins."""
        result = _build_http_options(NetworkConfig(timeout=TimeoutConfig(read=60.0)))
        assert result is not None
        assert result.timeout == 60000

    def test_build_http_options_timeout_config_connect_fallback(self):
        # When read is None, fall back to connect
        """Test that build http options timeout config connect fallback."""
        result = _build_http_options(NetworkConfig(timeout=TimeoutConfig(connect=5.0)))
        assert result is not None
        assert result.timeout == 5000

    def test_build_http_options_timeout_config_both_uses_read(self):
        # read takes priority over connect
        """Test that build http options timeout config both uses read."""
        result = _build_http_options(NetworkConfig(timeout=TimeoutConfig(connect=5.0, read=30.0)))
        assert result is not None
        assert result.timeout == 30000

    def test_build_http_options_timeout_config_both_none_returns_none(self):
        # If both connect and read are None, no timeout added → no other fields → None
        """Test that build http options timeout config both none returns none."""
        assert _build_http_options(NetworkConfig(timeout=TimeoutConfig())) is None

    def test_build_http_options_tls_verify_false_produces_ssl_context(self):
        # verify=False should produce an explicit ssl.SSLContext with CERT_NONE so it's
        # truthy and won't be replaced by the SDK's _ensure_httpx_ssl_ctx().
        """Test that build http options tls verify false produces ssl context."""
        result = _build_http_options(NetworkConfig(tls=TLSConfig(verify=False)))
        assert result is not None
        assert hasattr(result, "httpx_async_client")
        client = result.httpx_async_client
        assert client is not None
        # The verify kwarg should be an SSLContext (not bare False) — access via internal pool
        pool = client._transport._pool  # type: ignore[union-attr]  # httpx.AsyncHTTPTransport has _pool
        assert isinstance(pool._ssl_context, ssl.SSLContext)
        assert pool._ssl_context.verify_mode == ssl.CERT_NONE

    def test_build_http_options_tls_verify_true_still_sets_httpx_client(self):
        # Even verify=True triggers httpx_async_client so we can set follow_redirects
        """Test that build http options tls verify true still sets httpx client."""
        result = _build_http_options(NetworkConfig(tls=TLSConfig(verify=True)))
        assert result is not None
        assert hasattr(result, "httpx_async_client")
        assert result.httpx_async_client is not None

    def test_build_http_options_tls_verify_false_httpx_client_has_follow_redirects(self):
        """Test that build http options tls verify false httpx client has follow redirects."""
        result = _build_http_options(NetworkConfig(tls=TLSConfig(verify=False)))
        assert result is not None
        client = result.httpx_async_client
        assert client is not None
        assert client.follow_redirects is True or getattr(client, "_follow_redirects", True) is True

    def test_build_http_options_proxy_url_produces_httpx_client(self):
        """Test that build http options proxy url produces httpx client."""
        result = _build_http_options(
            NetworkConfig(proxy=ProxyConfig(url="http://proxy.example.com:8080"))  # type: ignore[arg-type]  # Pydantic coerces str→HttpUrl
        )
        assert result is not None
        assert hasattr(result, "httpx_async_client")
        assert result.httpx_async_client is not None

    def test_build_http_options_no_proxy_logs_warning(self, caplog):
        """Test that build http options no proxy logs warning."""
        with caplog.at_level(logging.WARNING):
            _build_http_options(NetworkConfig(proxy=ProxyConfig(no_proxy=["localhost"])))
        assert any("no_proxy" in record.message for record in caplog.records)

    def test_build_http_options_combined_all_fields(self):
        """Test that build http options combined all fields."""
        result = _build_http_options(
            NetworkConfig(
                headers={"X-Custom": "val"},
                timeout=30.0,
                tls=TLSConfig(verify=False),
                proxy=ProxyConfig(url="http://proxy.example.com:8080"),  # type: ignore[arg-type]  # Pydantic coerces str→HttpUrl
            )
        )
        assert result is not None
        assert result.headers == {"X-Custom": "val"}
        assert result.timeout == 30000
        assert hasattr(result, "httpx_async_client")
        assert result.httpx_async_client is not None


class TestVertexAIClientWithNetworkConfig:
    """Tests that client factory methods thread http_options into Client() calls."""

    def test_create_client_with_token_passes_http_options(self, monkeypatch):
        """Test that create client with token passes http options (lazy initialization)."""
        client_ctor = MagicMock(return_value=object())
        monkeypatch.setattr("ogx.providers.remote.inference.vertexai.vertexai.Client", client_ctor)

        adapter = VertexAIInferenceAdapter(
            config=VertexAIConfig(project="p", location="l", network=NetworkConfig(headers={"X-Test": "1"}))
        )
        # _create_client triggers _ensure_http_options which builds from config.network
        adapter._create_client(project="p", location="l", access_token="tok")

        kwargs = client_ctor.call_args.kwargs
        assert "http_options" in kwargs
        assert kwargs["http_options"].headers == {"X-Test": "1"}

    def test_create_adc_client_passes_http_options(self, monkeypatch):
        """Test that create adc client passes http options (lazy initialization)."""
        client_ctor = MagicMock(return_value=object())
        monkeypatch.setattr("ogx.providers.remote.inference.vertexai.vertexai.Client", client_ctor)

        adapter = VertexAIInferenceAdapter(
            config=VertexAIConfig(project="p", location="l", network=NetworkConfig(headers={"X-Test": "1"}))
        )
        # _create_adc_client triggers _ensure_http_options which builds from config.network
        adapter._create_adc_client(project="p", location="l")

        kwargs = client_ctor.call_args.kwargs
        assert "http_options" in kwargs
        assert kwargs["http_options"].headers == {"X-Test": "1"}

    def test_create_client_no_network_config_no_http_options(self, monkeypatch):
        """Test that create client no network config no http options."""
        client_ctor = MagicMock(return_value=object())
        monkeypatch.setattr("ogx.providers.remote.inference.vertexai.vertexai.Client", client_ctor)

        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        # _http_options is None by default (no network config)
        assert adapter._http_options is None

        adapter._create_client(project="p", location="l", access_token="tok")

        kwargs = client_ctor.call_args.kwargs
        # http_options should not be in kwargs when network config is not set
        assert "http_options" not in kwargs

    async def test_initialize_does_not_build_http_options(self, monkeypatch):
        """Test that initialize does NOT build http options (lazy initialization)."""
        adapter = VertexAIInferenceAdapter(
            config=VertexAIConfig(
                project="p",
                location="l",
                network=NetworkConfig(headers={"X-Custom": "val"}),
            )
        )
        mock_client = object()
        monkeypatch.setattr(adapter, "_create_client", lambda **kw: mock_client)

        await adapter.initialize()

        # With lazy initialization, _http_options should NOT be populated during initialize()
        assert adapter._http_options is None
        assert adapter._http_options_initialized is False

    async def test_shutdown_closes_managed_httpx_async_client(self, monkeypatch):
        """Test that shutdown closes managed httpx async client."""
        adapter = VertexAIInferenceAdapter(
            config=VertexAIConfig(
                project="p",
                location="l",
                network=NetworkConfig(tls=TLSConfig(verify=False)),
            )
        )
        adapter._http_options = _build_http_options(adapter.config.network)
        assert adapter._http_options is not None
        client = adapter._http_options.httpx_async_client
        assert client is not None
        aclose_mock = AsyncMock()
        monkeypatch.setattr(client, "aclose", aclose_mock)

        await adapter.shutdown()

        aclose_mock.assert_awaited_once()
        assert adapter._http_options is None
        assert adapter._default_client is None

    async def test_ensure_http_options_is_idempotent(self, monkeypatch):
        """Test that _ensure_http_options() is idempotent (only initializes once)."""
        adapter = VertexAIInferenceAdapter(
            config=VertexAIConfig(
                project="p",
                location="l",
                network=NetworkConfig(headers={"X-Test": "1"}),
            )
        )

        # First call should initialize
        adapter._ensure_http_options()
        assert adapter._http_options_initialized is True
        first_options = adapter._http_options

        # Second call should not reinitialize
        adapter._ensure_http_options()
        assert adapter._http_options is first_options  # Same object

        # Third call should also not reinitialize
        adapter._ensure_http_options()
        assert adapter._http_options is first_options  # Still same object


class TestBuildThinkingConfig:
    def test_none_returns_none(self):
        """Test that none returns none."""
        result = VertexAIInferenceAdapter._build_thinking_config(None)
        assert result is None

    def test_none_string_uses_budget_zero(self):
        """Test that none string uses budget zero."""
        result = VertexAIInferenceAdapter._build_thinking_config("none")
        assert result is not None
        assert result.thinking_budget == 0

    @pytest.mark.parametrize(
        "effort,expected_level",
        [
            ("minimal", "MINIMAL"),
            ("low", "LOW"),
            ("medium", "MEDIUM"),
            ("high", "HIGH"),
            ("xhigh", "HIGH"),
        ],
    )
    def test_effort_maps_to_thinking_level(self, effort: str, expected_level: str):
        """Test that effort maps to thinking level."""
        result = VertexAIInferenceAdapter._build_thinking_config(effort)
        assert result is not None
        assert result.thinking_level == expected_level

    def test_model_extra_overrides_thinking_config(self):
        """Test that model extra overrides thinking config."""
        override = {"thinking_budget": 9999}
        params = OpenAIChatCompletionRequestWithExtraBody(
            model="test-model",
            messages=cast(Any, [{"role": "user", "content": "hi"}]),
            reasoning_effort="low",
        )
        params = OpenAIChatCompletionRequestWithExtraBody.model_validate(
            {
                **params.model_dump(exclude_none=True),
                "thinking_config": override,
            }
        )
        adapter = VertexAIInferenceAdapter(
            config=VertexAIConfig(project="proj", location="us-central1"),
        )
        config = adapter._build_generation_config(
            params,
            system_instruction=None,
            tools_input=None,
        )
        assert config.thinking_config is not None
        assert config.thinking_config.thinking_budget == 9999


class TestStreamChatCompletion:
    """Test _stream_chat_completion() stream_options include_usage behavior."""

    def _make_fake_chunk(
        self,
        text: str = "hello",
        prompt_tokens: int = 10,
        completion_tokens: int = 5,
        total_tokens: int = 15,
    ) -> Any:
        """Build a fake chunk with usage metadata."""
        usage = SimpleNamespace(
            prompt_token_count=prompt_tokens,
            candidates_token_count=completion_tokens,
            total_token_count=total_tokens,
        )
        chunk = _make_fake_streaming_chunk(text)
        candidate = chunk.candidates[0]
        return SimpleNamespace(candidates=[candidate], usage_metadata=usage)

    async def test_stream_with_include_usage_emits_final_chunk(self, monkeypatch):
        """When stream_options include_usage=True, a final usage-only chunk is emitted."""
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))

        chunk1 = self._make_fake_chunk("Hello", prompt_tokens=10, completion_tokens=3, total_tokens=13)
        chunk2 = self._make_fake_chunk(" world", prompt_tokens=10, completion_tokens=5, total_tokens=15)

        async def fake_stream(**kwargs):
            """Yield predefined fake stream chunks."""
            yield chunk1
            yield chunk2

        fake_client = SimpleNamespace(
            aio=SimpleNamespace(models=SimpleNamespace(generate_content_stream=AsyncMock(return_value=fake_stream())))
        )
        monkeypatch.setattr(adapter, "_get_client", lambda: fake_client)

        stream = await adapter._stream_chat_completion(
            client=cast(Any, fake_client),
            provider_model_id="gemini-2.0-flash",
            contents=[],
            config=cast(Any, SimpleNamespace()),
            model="gemini-2.0-flash",
            stream_options={"include_usage": True},
        )

        chunks = [chunk async for chunk in stream]

        # 2 content chunks + 1 final usage-only chunk
        assert len(chunks) == 3

        # Final chunk has empty choices and usage populated
        final = chunks[-1]
        assert final.choices == []
        assert final.usage is not None
        assert final.usage.completion_tokens == 5
        assert final.usage.total_tokens == 15

    async def test_stream_without_stream_options_no_extra_chunk(self, monkeypatch):
        """When stream_options is None, only original content chunks are yielded."""
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))

        chunk1 = self._make_fake_chunk("Hello")
        chunk2 = self._make_fake_chunk(" world")

        async def fake_stream(**kwargs):
            """Yield predefined fake stream chunks."""
            yield chunk1
            yield chunk2

        fake_client = SimpleNamespace(
            aio=SimpleNamespace(models=SimpleNamespace(generate_content_stream=AsyncMock(return_value=fake_stream())))
        )
        monkeypatch.setattr(adapter, "_get_client", lambda: fake_client)

        stream = await adapter._stream_chat_completion(
            client=cast(Any, fake_client),
            provider_model_id="gemini-2.0-flash",
            contents=[],
            config=cast(Any, SimpleNamespace()),
            model="gemini-2.0-flash",
            stream_options=None,
        )

        chunks = [chunk async for chunk in stream]

        # Exactly 2 chunks — no extra usage chunk
        assert len(chunks) == 2

    async def test_stream_with_include_usage_false_no_extra_chunk(self, monkeypatch):
        """When stream_options include_usage=False, no extra usage chunk is emitted."""
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))

        chunk1 = self._make_fake_chunk("Hi")

        async def fake_stream(**kwargs):
            """Yield predefined fake stream chunks."""
            yield chunk1

        fake_client = SimpleNamespace(
            aio=SimpleNamespace(models=SimpleNamespace(generate_content_stream=AsyncMock(return_value=fake_stream())))
        )
        monkeypatch.setattr(adapter, "_get_client", lambda: fake_client)

        stream = await adapter._stream_chat_completion(
            client=cast(Any, fake_client),
            provider_model_id="gemini-2.0-flash",
            contents=[],
            config=cast(Any, SimpleNamespace()),
            model="gemini-2.0-flash",
            stream_options={"include_usage": False},
        )

        chunks = [chunk async for chunk in stream]

        # Exactly 1 chunk — include_usage=False means no extra chunk
        assert len(chunks) == 1


@pytest.fixture
def make_adapter_with_mock_embed(monkeypatch):
    """Create adapter with mock embed."""

    def factory(
        embeddings: list[list[float]], capture: dict | None = None, usage_metadata: SimpleNamespace | None = None
    ):
        """Create an adapter with a mocked client."""
        a = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        cast(Any, a).__provider_id__ = "vertexai"

        fake_embedding_objects = [SimpleNamespace(value=vec) for vec in embeddings]
        fake_response = SimpleNamespace(embeddings=fake_embedding_objects)
        if usage_metadata is not None:
            fake_response.usage_metadata = usage_metadata

        embed_mock = AsyncMock(return_value=fake_response)

        if capture is not None:

            async def capturing_embed(*args, **kwargs):
                """Handle capturing embed."""
                capture.update(kwargs)
                return fake_response

            embed_mock = capturing_embed

        fake_client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(embed_content=embed_mock)))
        monkeypatch.setattr(a, "_get_client", lambda: fake_client)
        return a, embed_mock

    return factory
