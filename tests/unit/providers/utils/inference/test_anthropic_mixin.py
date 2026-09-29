# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""AnthropicMixin: the shared native /v1/messages passthrough, exercised against all six
providers that use it (anthropic, meta, ollama, vllm, fireworks, deepseek)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import SecretStr

from ogx.providers.remote.inference.anthropic.anthropic import AnthropicInferenceAdapter
from ogx.providers.remote.inference.anthropic.config import AnthropicConfig
from ogx.providers.remote.inference.deepseek.config import DeepSeekImplConfig
from ogx.providers.remote.inference.deepseek.deepseek import DeepSeekInferenceAdapter
from ogx.providers.remote.inference.fireworks.config import FireworksImplConfig
from ogx.providers.remote.inference.fireworks.fireworks import FireworksInferenceAdapter
from ogx.providers.remote.inference.meta.config import MetaConfig
from ogx.providers.remote.inference.meta.meta import MetaInferenceAdapter
from ogx.providers.remote.inference.ollama.config import OllamaImplConfig
from ogx.providers.remote.inference.ollama.ollama import OllamaInferenceAdapter
from ogx.providers.remote.inference.vllm.config import VLLMInferenceAdapterConfig
from ogx.providers.remote.inference.vllm.vllm import VLLMInferenceAdapter
from ogx.providers.utils.inference.anthropic_mixin import AnthropicAPIError
from ogx.providers.utils.inference.network_config import NetworkConfig
from ogx_api.messages.models import ANTHROPIC_VERSION, AnthropicCountTokensRequest, AnthropicCreateMessageRequest

MESSAGE_BODY = {
    "id": "msg-1",
    "type": "message",
    "role": "assistant",
    "model": "test-model",
    "content": [{"type": "text", "text": "Hello"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 5, "output_tokens": 2},
}


def _adapter(provider: str, *, api_key: str | None = "config-key", base_url: str = "https://api.example.com/v1"):
    """A real provider adapter, with provider-data lookups stubbed out.

    Note: anthropic ignores base_url (its endpoint is fixed), ollama never uses a configured
    key (its get_api_key() returns the 'NO KEY REQUIRED' sentinel), and deepseek's Messages
    endpoint comes from anthropic_base_url, not base_url.
    """
    if provider == "anthropic":
        adapter = AnthropicInferenceAdapter(config=AnthropicConfig(api_key=api_key))
    elif provider == "meta":
        adapter = MetaInferenceAdapter(config=MetaConfig(base_url=base_url, api_key=api_key))
    elif provider == "ollama":
        adapter = OllamaInferenceAdapter(config=OllamaImplConfig(base_url=base_url))
    elif provider == "vllm":
        adapter = VLLMInferenceAdapter(config=VLLMInferenceAdapterConfig(base_url=base_url, api_token=api_key))
    elif provider == "fireworks":
        adapter = FireworksInferenceAdapter(config=FireworksImplConfig(base_url=base_url, api_key=api_key))
    elif provider == "deepseek":
        adapter = DeepSeekInferenceAdapter(
            config=DeepSeekImplConfig(
                base_url=base_url, anthropic_base_url="https://anthropic.example.com", api_key=api_key
            )
        )
    else:
        raise ValueError(f"unknown provider {provider}")
    if provider != "ollama":
        adapter.get_request_provider_data = MagicMock(return_value=None)
    return adapter


PROVIDERS = ["anthropic", "meta", "ollama", "vllm", "fireworks", "deepseek"]
BASE_URLS = ["https://api.example.com/v1", "https://api.example.com"]


def _message_request(**overrides):
    fields = {"model": "test-model", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 16}
    return AnthropicCreateMessageRequest(**{**fields, **overrides})


def _count_tokens_request():
    return AnthropicCountTokensRequest(model="test-model", messages=[{"role": "user", "content": "Hi"}])


@pytest.fixture
def mock_client():
    """A patched httpx.AsyncClient; returns (client_class, client, response) for assertions."""
    with patch("httpx.AsyncClient") as client_class:
        response = MagicMock()
        response.json.return_value = MESSAGE_BODY
        client = MagicMock()
        client.post = AsyncMock(return_value=response)
        client_class.return_value.__aenter__.return_value = client
        yield SimpleNamespace(client_class=client_class, client=client, response=response)


class TestAuthHeaders:
    @pytest.mark.parametrize("provider", PROVIDERS)
    async def test_auth_header_styles(self, provider, mock_client):
        adapter = _adapter(provider)
        await adapter.anthropic_messages(_message_request())

        headers = mock_client.client.post.call_args.kwargs["headers"]
        if provider == "vllm":
            assert headers["Authorization"] == "Bearer config-key"
            assert "x-api-key" not in headers
        elif provider == "ollama":
            # Ollama's get_api_key() returns the 'NO KEY REQUIRED' sentinel, which normalizes
            # to the placeholder rather than being sent as a key.
            assert headers["x-api-key"] == "no-key-required"
        else:
            assert headers["x-api-key"] == "config-key"

    async def test_anthropic_missing_key_raises_with_provider_data_hint(self, mock_client):
        adapter = _adapter("anthropic", api_key=None)

        with pytest.raises(ValueError, match="anthropic_api_key"):
            await adapter.anthropic_messages(_message_request())

        mock_client.client.post.assert_not_called()

    async def test_x_api_key_providers_send_placeholder_without_key(self, mock_client):
        for provider in ("meta", "fireworks", "deepseek"):
            adapter = _adapter(provider, api_key=None)
            await adapter.anthropic_messages(_message_request())
            assert mock_client.client.post.call_args.kwargs["headers"]["x-api-key"] == "no-key-required"

    async def test_vllm_sends_no_auth_header_without_key(self, mock_client):
        adapter = _adapter("vllm", api_key=None)
        await adapter.anthropic_messages(_message_request())

        headers = mock_client.client.post.call_args.kwargs["headers"]
        assert "Authorization" not in headers
        assert "x-api-key" not in headers

    @pytest.mark.parametrize("provider", [p for p in PROVIDERS if p != "ollama"])
    async def test_provider_data_key_overrides_config(self, provider, mock_client):
        adapter = _adapter(provider)
        field = {
            "anthropic": "anthropic_api_key",
            "meta": "meta_api_key",
            "vllm": "vllm_api_token",
            "fireworks": "fireworks_api_key",
            "deepseek": "deepseek_api_key",
        }[provider]
        adapter.get_request_provider_data = MagicMock(return_value=MagicMock(**{field: SecretStr("per-request-key")}))
        await adapter.anthropic_messages(_message_request())

        if provider == "vllm":
            assert mock_client.client.post.call_args.kwargs["headers"]["Authorization"] == "Bearer per-request-key"
        else:
            assert mock_client.client.post.call_args.kwargs["headers"]["x-api-key"] == "per-request-key"


class TestUrls:
    def _expected_base(self, provider: str, base_url: str) -> str:
        # Anthropic's endpoint is fixed regardless of config, and deepseek's comes from
        # anthropic_base_url. For the rest, the mixin strips a trailing /v1 so both
        # "https://host" and "https://host/v1" yield the same message URL.
        if provider == "anthropic":
            return "https://api.anthropic.com"
        if provider == "deepseek":
            return "https://anthropic.example.com"
        return base_url.rstrip("/").removesuffix("/v1")

    @pytest.mark.parametrize("provider", PROVIDERS)
    @pytest.mark.parametrize("base_url", BASE_URLS)
    async def test_messages_url(self, provider, base_url, mock_client):
        adapter = _adapter(provider, base_url=base_url)
        await adapter.anthropic_messages(_message_request())

        url = mock_client.client.post.call_args.args[0]
        assert url == f"{self._expected_base(provider, base_url)}/v1/messages"
        assert "/v1/v1/" not in url

    @pytest.mark.parametrize("provider", ["anthropic", "meta", "ollama", "vllm"])
    @pytest.mark.parametrize("base_url", BASE_URLS)
    async def test_count_tokens_url(self, provider, base_url, mock_client):
        mock_client.response.json.return_value = {"input_tokens": 1}
        adapter = _adapter(provider, base_url=base_url)
        await adapter.anthropic_count_tokens(_count_tokens_request())

        url = mock_client.client.post.call_args.args[0]
        assert url == f"{self._expected_base(provider, base_url)}/v1/messages/count_tokens"
        assert "/v1/v1/" not in url

    @pytest.mark.parametrize("provider", ["fireworks", "deepseek"])
    @pytest.mark.parametrize("base_url", BASE_URLS)
    async def test_no_native_count_endpoint_falls_back_to_messages_with_max_tokens_one(
        self, provider, base_url, mock_client
    ):
        adapter = _adapter(provider, base_url=base_url)
        await adapter.anthropic_count_tokens(_count_tokens_request())

        assert mock_client.client.post.call_args.args[0] == f"{self._expected_base(provider, base_url)}/v1/messages"
        assert mock_client.client.post.call_args.kwargs["json"]["max_tokens"] == 1


class TestHeadersAndBody:
    async def test_sends_anthropic_headers(self, mock_client):
        await _adapter("meta").anthropic_messages(_message_request())

        headers = mock_client.client.post.call_args.kwargs["headers"]
        assert headers["content-type"] == "application/json"
        assert headers["anthropic-version"] == ANTHROPIC_VERSION

    async def test_body_is_the_request_model_dump(self, mock_client):
        await _adapter("meta").anthropic_messages(_message_request(metadata={"user_id": "u1"}))

        body = mock_client.client.post.call_args.kwargs["json"]
        assert body["model"] == "test-model"
        assert body["max_tokens"] == 16
        assert body["metadata"] == {"user_id": "u1"}
        assert body["stream"] is False


class TestErrors:
    async def test_error_preserves_status_and_body_message(self, mock_client):
        response = httpx.Response(
            400,
            json={"type": "error", "error": {"type": "invalid_request_error", "message": "bad request"}},
            request=httpx.Request("POST", "https://api.example.com/v1/messages"),
        )
        mock_client.client.post = AsyncMock(return_value=response)

        with pytest.raises(AnthropicAPIError, match="bad request") as exc_info:
            await _adapter("meta").anthropic_messages(_message_request())

        assert exc_info.value.status_code == 400

    async def test_error_without_json_body_falls_back_to_status(self, mock_client):
        response = httpx.Response(
            529, text="not json", request=httpx.Request("POST", "https://api.example.com/v1/messages")
        )
        mock_client.client.post = AsyncMock(return_value=response)

        with pytest.raises(AnthropicAPIError, match="status 529") as exc_info:
            await _adapter("fireworks").anthropic_count_tokens(_count_tokens_request())

        assert exc_info.value.status_code == 529

    async def test_streaming_error_keeps_status(self, monkeypatch):
        async def failing(**_kwargs):
            raise httpx.HTTPStatusError(
                "429",
                request=httpx.Request("POST", "https://api.example.com/v1/messages"),
                response=httpx.Response(429, request=httpx.Request("POST", "https://api.example.com/v1/messages")),
            )
            yield

        monkeypatch.setattr(
            "ogx.providers.utils.inference.anthropic_mixin.passthrough_anthropic_stream",
            MagicMock(side_effect=failing),
        )

        result = await _adapter("meta").anthropic_messages(_message_request(stream=True))
        with pytest.raises(AnthropicAPIError) as exc_info:
            async for _ in result:
                pass

        assert exc_info.value.status_code == 429

    async def test_streaming_missing_key_fails_before_streaming(self, monkeypatch):
        mock = MagicMock()
        monkeypatch.setattr("ogx.providers.utils.inference.anthropic_mixin.passthrough_anthropic_stream", mock)

        with pytest.raises(ValueError, match="anthropic_api_key"):
            await _adapter("anthropic", api_key=None).anthropic_messages(_message_request(stream=True))

        mock.assert_not_called()


class TestAllowedModels:
    @pytest.mark.parametrize("provider", PROVIDERS)
    async def test_messages_allows_listed_model(self, provider, mock_client):
        adapter = _adapter(provider)
        adapter.config.allowed_models = ["test-model"]

        await adapter.anthropic_messages(_message_request())

        mock_client.client.post.assert_called_once()

    @pytest.mark.parametrize("provider", PROVIDERS)
    async def test_messages_rejects_disallowed_model(self, provider, mock_client):
        adapter = _adapter(provider)
        adapter.config.allowed_models = ["other-model"]

        with pytest.raises(ValueError, match="not in the allowed models list"):
            await adapter.anthropic_messages(_message_request())

        mock_client.client.post.assert_not_called()

    @pytest.mark.parametrize("provider", PROVIDERS)
    async def test_count_tokens_rejects_disallowed_model(self, provider, mock_client):
        adapter = _adapter(provider)
        adapter.config.allowed_models = ["other-model"]

        with pytest.raises(ValueError, match="not in the allowed models list"):
            await adapter.anthropic_count_tokens(_count_tokens_request())

        mock_client.client.post.assert_not_called()


class TestTimeouts:
    async def test_default_timeouts_per_call(self, mock_client):
        adapter = _adapter("meta")
        await adapter.anthropic_messages(_message_request())
        assert mock_client.client_class.call_args.kwargs["timeout"] == httpx.Timeout(300.0)

        mock_client.response.json.return_value = {"input_tokens": 1}
        await adapter.anthropic_count_tokens(_count_tokens_request())
        assert mock_client.client_class.call_args.kwargs["timeout"] == httpx.Timeout(30.0)

    @pytest.mark.parametrize("provider", PROVIDERS)
    async def test_network_timeout_takes_precedence(self, provider, mock_client):
        adapter = _adapter(provider)
        adapter.config.network = NetworkConfig(timeout=12.0)

        await adapter.anthropic_messages(_message_request())
        assert mock_client.client_class.call_args.kwargs["timeout"] == httpx.Timeout(12.0)

        if provider not in ("fireworks", "deepseek"):
            # Fireworks and DeepSeek have no native endpoint and count via anthropic_messages,
            # which expects a message response rather than a count_tokens one.
            mock_client.response.json.return_value = {"input_tokens": 1}
        await adapter.anthropic_count_tokens(_count_tokens_request())
        assert mock_client.client_class.call_args.kwargs["timeout"] == httpx.Timeout(12.0)

    async def test_streaming_passes_client_kwargs_to_the_shared_helper(self, monkeypatch):
        async def no_events(**_kwargs):
            return
            yield

        mock = MagicMock(side_effect=no_events)
        monkeypatch.setattr("ogx.providers.utils.inference.anthropic_mixin.passthrough_anthropic_stream", mock)

        adapter = _adapter("meta")
        adapter.config.network = NetworkConfig(timeout=12.0)

        result = await adapter.anthropic_messages(_message_request(stream=True))
        async for _ in result:
            pass

        client_kwargs = mock.call_args.kwargs["httpx_client_kwargs"]
        assert client_kwargs["timeout"] == httpx.Timeout(12.0)
