# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import ssl
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from ogx.providers.remote.inference.meta.config import MetaConfig
from ogx.providers.remote.inference.meta.meta import MetaInferenceAdapter
from ogx_api.messages.models import (
    AnthropicCountTokensRequest,
    AnthropicCreateMessageRequest,
    AnthropicMessageResponse,
)


@pytest.fixture(scope="function")
async def meta_adapter():
    config = MetaConfig(base_url="https://api.meta.ai/v1")
    adapter = MetaInferenceAdapter(config=config)
    await adapter.initialize()
    return adapter


class TestBuildHttpClientKwargs:
    """Tests for _build_httpx_client_kwargs() on Meta adapter."""

    async def test_default_returns_ssl_context(self):
        config = MetaConfig(base_url="https://api.meta.ai/v1")
        adapter = MetaInferenceAdapter(config=config)
        await adapter.initialize()

        kwargs = adapter._build_httpx_client_kwargs()
        assert "verify" in kwargs
        assert isinstance(kwargs["verify"], ssl.SSLContext)

    async def test_verify_false(self):
        config = MetaConfig(
            base_url="https://api.meta.ai/v1",
            network={"tls": {"verify": False}},
        )
        adapter = MetaInferenceAdapter(config=config)
        await adapter.initialize()

        kwargs = adapter._build_httpx_client_kwargs()
        assert kwargs["verify"] is False

    async def test_verify_with_custom_cert_path(self):
        with tempfile.NamedTemporaryFile(suffix=".crt") as f:
            f.write(b"fake cert")
            cert_path = f.name
            config = MetaConfig(
                base_url="https://api.meta.ai/v1",
                network={"tls": {"verify": cert_path}},
            )
            adapter = MetaInferenceAdapter(config=config)
            await adapter.initialize()

            kwargs = adapter._build_httpx_client_kwargs()
            assert "verify" in kwargs
            assert kwargs["verify"] == Path(cert_path).resolve().__str__()

    async def test_timeout_config_in_kwargs(self):
        config = MetaConfig(
            base_url="https://api.meta.ai/v1",
            network={"timeout": {"connect": 5.0, "read": 30.0}},
        )
        adapter = MetaInferenceAdapter(config=config)
        await adapter.initialize()

        kwargs = adapter._build_httpx_client_kwargs()
        assert "timeout" in kwargs
        assert kwargs["timeout"].connect == 5.0
        assert kwargs["timeout"].read == 30.0

    async def test_proxy_mounts_in_kwargs(self):
        config = MetaConfig(
            base_url="https://api.meta.ai/v1",
            network={"proxy": {"url": "http://proxy.example.com:8080"}},
        )
        adapter = MetaInferenceAdapter(config=config)
        await adapter.initialize()

        kwargs = adapter._build_httpx_client_kwargs()
        assert "mounts" in kwargs
        assert "http://" in kwargs["mounts"]
        assert "https://" in kwargs["mounts"]

    async def test_combined_config(self):
        config = MetaConfig(
            base_url="https://api.meta.ai/v1",
            network={
                "tls": {"verify": False},
                "timeout": {"connect": 10.0, "read": 60.0},
                "proxy": {"url": "http://proxy:8080"},
            },
        )
        adapter = MetaInferenceAdapter(config=config)
        await adapter.initialize()

        kwargs = adapter._build_httpx_client_kwargs()
        assert kwargs["verify"] is False
        assert kwargs["timeout"].connect == 10.0
        assert kwargs["timeout"].read == 60.0
        assert "mounts" in kwargs
        assert "http://" in kwargs["mounts"]
        assert "https://" in kwargs["mounts"]


class TestPassthroughMessagesNetworkKwargs:
    """Tests that _passthrough_anthropic_messages passes httpx kwargs."""

    async def test_stream_false_passes_kwargs(self, meta_adapter):
        meta_adapter.get_request_provider_data = MagicMock(return_value=None)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "id": "msg-1",
                "content": [{"type": "text", "text": "Hello"}],
                "role": "assistant",
                "stop_reason": "end_turn",
                "type": "message",
                "model": "test-model",
                "stop_sequences": None,
                "usage": {"input_tokens": 5, "output_tokens": 5},
            }

            mock_client_instance = MagicMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client_instance

            request = AnthropicCreateMessageRequest(
                messages=[{"role": "user", "content": "Hi"}],
                model="test-model",
                max_tokens=256,
                stream=False,
            )
            result = await meta_adapter.anthropic_messages(request)

            mock_client_class.assert_called_once()
            assert isinstance(result, AnthropicMessageResponse)

    async def test_stream_false_with_tls_config(self):
        config = MetaConfig(
            base_url="https://api.meta.ai/v1",
            network={"tls": {"verify": False}},
        )
        adapter = MetaInferenceAdapter(config=config)
        await adapter.initialize()
        adapter.get_request_provider_data = MagicMock(return_value=None)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "id": "msg-1",
                "content": [{"type": "text", "text": "Response"}],
                "role": "assistant",
                "stop_reason": "end_turn",
                "type": "message",
                "model": "test-model",
                "stop_sequences": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }

            mock_client_instance = MagicMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client_instance

            request = AnthropicCreateMessageRequest(
                messages=[{"role": "user", "content": "Hi"}],
                model="test-model",
                max_tokens=256,
                stream=False,
            )
            await adapter.anthropic_messages(request)

            mock_client_class.assert_called_once()
            call_kwargs = mock_client_class.call_args.kwargs
            assert call_kwargs.get("verify") is False

    async def test_stream_false_url_with_v1_stripped(self):
        adapter = MetaInferenceAdapter(config=MetaConfig(base_url="https://api.meta.ai/v1"))
        await adapter.initialize()
        adapter.get_request_provider_data = MagicMock(return_value=None)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "id": "msg-1",
                "content": [{"type": "text", "text": "OK"}],
                "role": "assistant",
                "stop_reason": "end_turn",
                "type": "message",
                "model": "test-model",
                "stop_sequences": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }

            mock_client_instance = MagicMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client_instance

            request = AnthropicCreateMessageRequest(
                messages=[{"role": "user", "content": "Hi"}],
                model="test-model",
                max_tokens=256,
                stream=False,
            )
            await adapter.anthropic_messages(request)

            call_args = mock_client_instance.post.call_args
            assert call_args.args[0] == "https://api.meta.ai/v1/messages"

    async def test_stream_false_url_without_v1_suffix(self):
        adapter = MetaInferenceAdapter(config=MetaConfig(base_url="https://api.meta.ai"))
        await adapter.initialize()
        adapter.get_request_provider_data = MagicMock(return_value=None)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "id": "msg-1",
                "content": [{"type": "text", "text": "OK"}],
                "role": "assistant",
                "stop_reason": "end_turn",
                "type": "message",
                "model": "test-model",
                "stop_sequences": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }

            mock_client_instance = MagicMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client_instance

            request = AnthropicCreateMessageRequest(
                messages=[{"role": "user", "content": "Hi"}],
                model="test-model",
                max_tokens=256,
                stream=False,
            )
            await adapter.anthropic_messages(request)

            call_args = mock_client_instance.post.call_args
            assert call_args.args[0] == "https://api.meta.ai/v1/messages"


class TestPassthroughCountTokensNetworkKwargs:
    """Tests that anthropic_count_tokens passes httpx kwargs."""

    async def test_count_tokens_passes_kwargs(self):
        config = MetaConfig(
            base_url="https://api.meta.ai/v1",
            network={"tls": {"verify": False}},
        )
        adapter = MetaInferenceAdapter(config=config)
        await adapter.initialize()
        adapter.get_request_provider_data = MagicMock(return_value=None)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "input_tokens": 42,
            }

            mock_client_instance = MagicMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client_instance

            request = AnthropicCountTokensRequest(
                model="test-model",
                messages=[{"role": "user", "content": "Hello world"}],
            )
            await adapter.anthropic_count_tokens(request)

            mock_client_class.assert_called_once()
            call_kwargs = mock_client_class.call_args.kwargs
            assert call_kwargs.get("verify") is False

    async def test_count_tokens_url(self):
        adapter = MetaInferenceAdapter(config=MetaConfig(base_url="https://api.meta.ai/v1"))
        await adapter.initialize()
        adapter.get_request_provider_data = MagicMock(return_value=None)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"input_tokens": 10}

            mock_client_instance = MagicMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client_instance

            request = AnthropicCountTokensRequest(
                model="test-model",
                messages=[{"role": "user", "content": "test"}],
            )
            await adapter.anthropic_count_tokens(request)

            call_args = mock_client_instance.post.call_args
            assert call_args.args[0] == "https://api.meta.ai/v1/messages/count_tokens"

    async def test_count_tokens_api_key_in_headers(self):
        adapter = MetaInferenceAdapter(config=MetaConfig(base_url="https://api.meta.ai/v1"))
        await adapter.initialize()
        adapter.get_request_provider_data = MagicMock(return_value=None)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"input_tokens": 5}

            mock_client_instance = MagicMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client_instance

            request = AnthropicCountTokensRequest(
                model="test-model",
                messages=[{"role": "user", "content": "test"}],
            )
            await adapter.anthropic_count_tokens(request)

            call_args = mock_client_instance.post.call_args
            headers = call_args.kwargs.get("headers", {})
            assert headers["x-api-key"] == "no-key-required"


class TestApiKeyHeader:
    """Tests for x-api-key header in Anthropic passthrough."""

    async def test_sends_meta_api_key_from_provider_data(self):
        adapter = MetaInferenceAdapter(config=MetaConfig(base_url="https://api.meta.ai/v1"))
        await adapter.initialize()

        adapter.get_request_provider_data = MagicMock(
            return_value=SimpleNamespace(meta_api_key=SecretStr("secret-key-123"))
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "id": "msg-1",
                "content": [{"type": "text", "text": "Hello"}],
                "role": "assistant",
                "stop_reason": "end_turn",
                "type": "message",
                "model": "test-model",
                "stop_sequences": None,
                "usage": {"input_tokens": 5, "output_tokens": 5},
            }

            mock_client_instance = MagicMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client_instance

            request = AnthropicCreateMessageRequest(
                messages=[{"role": "user", "content": "Hi"}],
                model="test-model",
                max_tokens=256,
                stream=False,
            )
            await adapter.anthropic_messages(request)

            call_args = mock_client_instance.post.call_args
            headers = call_args.kwargs.get("headers", {})
            assert headers["x-api-key"] == "secret-key-123"

    async def test_count_tokens_sends_api_key_from_provider_data(self):
        adapter = MetaInferenceAdapter(config=MetaConfig(base_url="https://api.meta.ai/v1"))
        await adapter.initialize()

        adapter.get_request_provider_data = MagicMock(
            return_value=SimpleNamespace(meta_api_key=SecretStr("secret-key-456"))
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"input_tokens": 10}

            mock_client_instance = MagicMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client_instance

            request = AnthropicCountTokensRequest(
                model="test-model",
                messages=[{"role": "user", "content": "test"}],
            )
            await adapter.anthropic_count_tokens(request)

            call_args = mock_client_instance.post.call_args
            headers = call_args.kwargs.get("headers", {})
            assert headers["x-api-key"] == "secret-key-456"

    async def test_default_no_key_when_no_provider_data(self):
        adapter = MetaInferenceAdapter(config=MetaConfig(base_url="https://api.meta.ai/v1"))
        await adapter.initialize()
        adapter.get_request_provider_data = MagicMock(return_value=None)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "id": "msg-1",
                "content": [{"type": "text", "text": "OK"}],
                "role": "assistant",
                "stop_reason": "end_turn",
                "type": "message",
                "model": "test-model",
                "stop_sequences": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }

            mock_client_instance = MagicMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client_instance

            request = AnthropicCreateMessageRequest(
                messages=[{"role": "user", "content": "Hi"}],
                model="test-model",
                max_tokens=256,
                stream=False,
            )
            await adapter.anthropic_messages(request)

            call_args = mock_client_instance.post.call_args
            headers = call_args.kwargs.get("headers", {})
            assert headers["x-api-key"] == "no-key-required"

    async def test_anthropic_version_header(self):
        adapter = MetaInferenceAdapter(config=MetaConfig(base_url="https://api.meta.ai/v1"))
        await adapter.initialize()
        adapter.get_request_provider_data = MagicMock(return_value=None)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "id": "msg-1",
                "content": [{"type": "text", "text": "OK"}],
                "role": "assistant",
                "stop_reason": "end_turn",
                "type": "message",
                "model": "test-model",
                "stop_sequences": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }

            mock_client_instance = MagicMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client_instance

            request = AnthropicCreateMessageRequest(
                messages=[{"role": "user", "content": "Hi"}],
                model="test-model",
                max_tokens=256,
                stream=False,
            )
            await adapter.anthropic_messages(request)

            call_args = mock_client_instance.post.call_args
            headers = call_args.kwargs.get("headers", {})
            assert headers["content-type"] == "application/json"
            assert "anthropic-version" in headers
            assert "x-api-key" in headers


@pytest.fixture
def mock_passthrough(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr(
        "ogx.providers.remote.inference.meta.meta.passthrough_anthropic_stream",
        mock,
    )
    return mock


class TestPassthroughStreamNetworkKwargs:
    """Tests that streaming passthrough passes httpx kwargs."""

    async def test_stream_calls_passthrough_anthropic_stream_with_kwargs(self, mock_passthrough):
        config = MetaConfig(
            base_url="https://api.meta.ai/v1",
            network={"tls": {"verify": False}},
        )
        adapter = MetaInferenceAdapter(config=config)
        await adapter.initialize()
        adapter.get_request_provider_data = MagicMock(return_value=None)

        async def empty_gen():
            return
            yield

        mock_passthrough.return_value = empty_gen()

        request = AnthropicCreateMessageRequest(
            messages=[{"role": "user", "content": "Hi"}],
            model="test-model",
            max_tokens=256,
            stream=True,
        )

        result = await adapter.anthropic_messages(request)
        events = []
        async for event in result:
            events.append(event)

        mock_passthrough.assert_called_once()
        call_kwargs = mock_passthrough.call_args.kwargs
        assert call_kwargs["url"] == "https://api.meta.ai/v1/messages"
        assert call_kwargs["req_body"]["model"] == "test-model"
        assert call_kwargs["httpx_client_kwargs"]["verify"] is False

    async def test_stream_uses_shared_ssl_context_by_default(self):
        """Without network config, _build_httpx_client_kwargs should return shared_ssl_context as verify."""
        adapter = MetaInferenceAdapter(config=MetaConfig(base_url="https://api.meta.ai/v1"))
        await adapter.initialize()

        kwargs = adapter._build_httpx_client_kwargs()
        assert "verify" in kwargs
        assert isinstance(kwargs["verify"], ssl.SSLContext)
