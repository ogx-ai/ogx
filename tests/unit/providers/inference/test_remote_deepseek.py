# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Native /v1/messages passthrough of the DeepSeek adapter (no chat-completions translation)."""

import ssl
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import SecretStr

from ogx.providers.remote.inference.deepseek.config import DeepSeekImplConfig
from ogx.providers.remote.inference.deepseek.deepseek import DeepSeekInferenceAdapter
from ogx_api.messages.models import (
    AnthropicCountTokensRequest,
    AnthropicCreateMessageRequest,
    AnthropicMessageResponse,
)


def _adapter(**config_kwargs) -> DeepSeekInferenceAdapter:
    config_kwargs.setdefault("base_url", "https://api.deepseek.com/v1")
    config = DeepSeekImplConfig(**config_kwargs)
    adapter = DeepSeekInferenceAdapter(config=config)
    adapter.get_request_provider_data = MagicMock(return_value=None)
    return adapter


_MESSAGE_RESPONSE_BODY = {
    "id": "msg-1",
    "content": [{"type": "text", "text": "Hello"}],
    "role": "assistant",
    "stop_reason": "end_turn",
    "type": "message",
    "model": "test-model",
    "stop_sequences": None,
    "usage": {"input_tokens": 5, "output_tokens": 5},
}


class TestBuildHttpClientKwargs:
    async def test_default_returns_ssl_context(self):
        adapter = _adapter()

        kwargs = adapter._build_httpx_client_kwargs()

        assert isinstance(kwargs["verify"], ssl.SSLContext)

    async def test_verify_false(self):
        adapter = _adapter(network={"tls": {"verify": False}})

        kwargs = adapter._build_httpx_client_kwargs()

        assert kwargs["verify"] is False


class TestMessagesPassthrough:
    """anthropic_messages() forwards to DeepSeek's own anthropic_base_url, not base_url."""

    async def test_posts_to_the_anthropic_base_url_not_the_openai_one(self):
        adapter = _adapter()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.post = AsyncMock(return_value=MagicMock(json=lambda: _MESSAGE_RESPONSE_BODY))
            mock_client_class.return_value.__aenter__.return_value = mock_client

            request = AnthropicCreateMessageRequest(
                messages=[{"role": "user", "content": "Hi"}], model="test-model", max_tokens=16, stream=False
            )
            result = await adapter.anthropic_messages(request)

            assert mock_client.post.call_args.args[0] == "https://api.deepseek.com/anthropic/v1/messages"
            assert isinstance(result, AnthropicMessageResponse)

    async def test_custom_anthropic_base_url_is_used_independently_of_base_url(self):
        adapter = _adapter(
            base_url="https://gateway.internal/deepseek/v1",
            anthropic_base_url="https://gateway.internal/deepseek-anthropic",
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.post = AsyncMock(return_value=MagicMock(json=lambda: _MESSAGE_RESPONSE_BODY))
            mock_client_class.return_value.__aenter__.return_value = mock_client

            request = AnthropicCreateMessageRequest(
                messages=[{"role": "user", "content": "Hi"}], model="test-model", max_tokens=16, stream=False
            )
            await adapter.anthropic_messages(request)

            assert mock_client.post.call_args.args[0] == "https://gateway.internal/deepseek-anthropic/v1/messages"

    async def test_sends_anthropic_headers_and_config_api_key(self):
        adapter = _adapter(api_key="config-key")

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.post = AsyncMock(return_value=MagicMock(json=lambda: _MESSAGE_RESPONSE_BODY))
            mock_client_class.return_value.__aenter__.return_value = mock_client

            request = AnthropicCreateMessageRequest(
                messages=[{"role": "user", "content": "Hi"}], model="test-model", max_tokens=16, stream=False
            )
            await adapter.anthropic_messages(request)

            headers = mock_client.post.call_args.kwargs["headers"]
            assert headers["content-type"] == "application/json"
            assert "anthropic-version" in headers
            assert headers["x-api-key"] == "config-key"

    async def test_provider_data_api_key_overrides_config(self):
        adapter = _adapter(api_key="config-key")
        adapter.get_request_provider_data = MagicMock(
            return_value=SimpleNamespace(deepseek_api_key=SecretStr("per-request-key"))
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.post = AsyncMock(return_value=MagicMock(json=lambda: _MESSAGE_RESPONSE_BODY))
            mock_client_class.return_value.__aenter__.return_value = mock_client

            request = AnthropicCreateMessageRequest(
                messages=[{"role": "user", "content": "Hi"}], model="test-model", max_tokens=16, stream=False
            )
            await adapter.anthropic_messages(request)

            headers = mock_client.post.call_args.kwargs["headers"]
            assert headers["x-api-key"] == "per-request-key"

    async def test_no_api_key_sends_no_key_required(self):
        adapter = _adapter()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.post = AsyncMock(return_value=MagicMock(json=lambda: _MESSAGE_RESPONSE_BODY))
            mock_client_class.return_value.__aenter__.return_value = mock_client

            request = AnthropicCreateMessageRequest(
                messages=[{"role": "user", "content": "Hi"}], model="test-model", max_tokens=16, stream=False
            )
            await adapter.anthropic_messages(request)

            headers = mock_client.post.call_args.kwargs["headers"]
            assert headers["x-api-key"] == "no-key-required"

    async def test_request_body_keeps_fields_the_model_does_not_declare(self):
        """extra="allow" on AnthropicCreateMessageRequest forwards fields like thinking and
        output_config unchanged, which is the whole point of native passthrough over
        chat-completions translation."""
        adapter = _adapter()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.post = AsyncMock(return_value=MagicMock(json=lambda: _MESSAGE_RESPONSE_BODY))
            mock_client_class.return_value.__aenter__.return_value = mock_client

            request = AnthropicCreateMessageRequest.model_validate(
                {
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 16,
                    "thinking": {"type": "enabled", "budget_tokens": 2048},
                    "output_config": {"effort": "high"},
                }
            )
            await adapter.anthropic_messages(request)

            body = mock_client.post.call_args.kwargs["json"]
            assert body["thinking"] == {"type": "enabled", "budget_tokens": 2048}
            assert body["output_config"] == {"effort": "high"}

    async def test_returns_the_parsed_response(self):
        adapter = _adapter()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.post = AsyncMock(return_value=MagicMock(json=lambda: _MESSAGE_RESPONSE_BODY))
            mock_client_class.return_value.__aenter__.return_value = mock_client

            request = AnthropicCreateMessageRequest(
                messages=[{"role": "user", "content": "Hi"}], model="test-model", max_tokens=16, stream=False
            )
            result = await adapter.anthropic_messages(request)

            assert isinstance(result, AnthropicMessageResponse)
            assert result.usage.input_tokens == 5


@pytest.fixture
def mock_passthrough(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr(
        "ogx.providers.remote.inference.deepseek.deepseek.passthrough_anthropic_stream",
        mock,
    )
    return mock


class TestStreamingPassthrough:
    async def test_stream_calls_passthrough_anthropic_stream(self, mock_passthrough):
        adapter = _adapter()

        async def empty_gen():
            return
            yield

        mock_passthrough.return_value = empty_gen()

        request = AnthropicCreateMessageRequest(
            messages=[{"role": "user", "content": "Hi"}], model="test-model", max_tokens=16, stream=True
        )
        result = await adapter.anthropic_messages(request)
        events = [event async for event in result]

        assert events == []
        mock_passthrough.assert_called_once()
        call_kwargs = mock_passthrough.call_args.kwargs
        assert call_kwargs["url"] == "https://api.deepseek.com/anthropic/v1/messages"
        assert call_kwargs["req_body"]["model"] == "test-model"
        assert "verify" in call_kwargs["httpx_client_kwargs"]


class TestUpstreamErrors:
    async def test_error_response_raises_http_status_error(self):
        adapter = _adapter()

        with patch("httpx.AsyncClient") as mock_client_class:
            error_response = httpx.Response(
                429,
                json={"type": "error", "error": {"type": "rate_limit_error", "message": "rate limited"}},
                request=httpx.Request("POST", "https://api.deepseek.com/anthropic/v1/messages"),
            )
            mock_client = MagicMock()
            mock_client.post = AsyncMock(return_value=error_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client

            request = AnthropicCreateMessageRequest(
                messages=[{"role": "user", "content": "Hi"}], model="test-model", max_tokens=16, stream=False
            )
            with pytest.raises(httpx.HTTPStatusError) as exc_info:
                await adapter.anthropic_messages(request)

            assert exc_info.value.response.status_code == 429


class TestCountTokensFallsBackToMessages:
    """DeepSeek has no /v1/messages/count_tokens endpoint (#6674), so anthropic_count_tokens is
    not overridden: it must inherit OpenAIMixin's default, which counts by calling
    anthropic_messages() -- our native override, not the chat-completions translation -- with
    max_tokens=1."""

    async def test_count_tokens_posts_to_messages_not_count_tokens(self):
        adapter = _adapter()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.post = AsyncMock(return_value=MagicMock(json=lambda: _MESSAGE_RESPONSE_BODY))
            mock_client_class.return_value.__aenter__.return_value = mock_client

            request = AnthropicCountTokensRequest(model="test-model", messages=[{"role": "user", "content": "Hi"}])
            result = await adapter.anthropic_count_tokens(request)

            assert mock_client.post.call_args.args[0] == "https://api.deepseek.com/anthropic/v1/messages"
            body = mock_client.post.call_args.kwargs["json"]
            assert body["max_tokens"] == 1
            assert result.input_tokens == 5

    async def test_count_tokens_is_not_overridden(self):
        """Guards the design decision itself: no /v1/messages/count_tokens endpoint exists to call."""
        assert "anthropic_count_tokens" not in DeepSeekInferenceAdapter.__dict__
