# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Native /v1/messages passthrough of the Anthropic adapter (no chat-completions translation)."""

import json
import ssl
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import SecretStr

from ogx.providers.remote.inference.anthropic.anthropic import AnthropicAPIError, AnthropicInferenceAdapter
from ogx.providers.remote.inference.anthropic.config import AnthropicConfig
from ogx_api.messages.models import (
    ANTHROPIC_VERSION,
    AnthropicCountTokensRequest,
    AnthropicCreateMessageRequest,
    AnthropicMessageResponse,
    AnthropicThinkingConfig,
)

MESSAGES_URL = "https://api.anthropic.com/v1/messages"
COUNT_TOKENS_URL = "https://api.anthropic.com/v1/messages/count_tokens"

MESSAGE_RESPONSE = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-4-5",
    "content": [{"type": "text", "text": "Paris."}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 5, "output_tokens": 2},
}

NETWORK_CONFIG = {
    "tls": {"verify": False},
    "proxy": {"url": "http://proxy.example.com:3128"},
    "headers": {"X-Route": "team-a"},
    "timeout": 12.0,
    "limits": {"max_connections": 7},
}


def _adapter(**config_kwargs) -> AnthropicInferenceAdapter:
    config_kwargs.setdefault("api_key", "config-key")
    adapter = AnthropicInferenceAdapter(config=AnthropicConfig(**config_kwargs))
    adapter.get_request_provider_data = MagicMock(return_value=None)
    return adapter


def _request(**overrides) -> AnthropicCreateMessageRequest:
    fields = {
        "model": "claude-haiku-4-5",
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "max_tokens": 256,
    }
    return AnthropicCreateMessageRequest(**{**fields, **overrides})


@pytest.fixture
def mock_client():
    """A patched httpx.AsyncClient whose post() returns a successful messages response."""
    with patch("httpx.AsyncClient") as client_class:
        response = MagicMock()
        response.is_error = False
        response.json.return_value = MESSAGE_RESPONSE
        client = MagicMock()
        client.post = AsyncMock(return_value=response)
        client_class.return_value.__aenter__.return_value = client
        client_class.client = client
        client_class.response = response
        yield client_class


@pytest.fixture
def mock_passthrough(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr("ogx.providers.remote.inference.anthropic.anthropic.passthrough_anthropic_stream", mock)
    return mock


@pytest.fixture
def wire(monkeypatch):
    """Route the adapter's httpx clients through a MockTransport; returns the captured requests."""
    captured: list[httpx.Request] = []
    responder = {"handler": lambda request: httpx.Response(200, json=MESSAGE_RESPONSE)}
    real_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return responder["handler"](request)

    def make_client(*args, **kwargs):
        kwargs.pop("verify", None)
        kwargs.pop("mounts", None)
        return real_client(*args, transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", make_client)
    captured_holder = SimpleNamespace(requests=captured, respond=lambda fn: responder.update(handler=fn))
    return captured_holder


class TestDoesNotTranslate:
    async def test_messages_do_not_go_through_chat_completions(self, mock_client):
        adapter = _adapter()
        adapter.openai_chat_completion = AsyncMock()

        await adapter.anthropic_messages(_request())

        adapter.openai_chat_completion.assert_not_awaited()

    async def test_extended_thinking_is_accepted(self, mock_client):
        """Regression: the translation fallback rejected `thinking`, so the claude CLI could not use ogx."""
        adapter = _adapter()

        response = await adapter.anthropic_messages(
            _request(thinking=AnthropicThinkingConfig(type="enabled", budget_tokens=4096))
        )

        assert isinstance(response, AnthropicMessageResponse)
        assert mock_client.client.post.call_args.kwargs["json"]["thinking"] == {
            "type": "enabled",
            "budget_tokens": 4096,
        }


class TestMessagesRequest:
    async def test_posts_to_the_anthropic_messages_url(self, mock_client):
        await _adapter().anthropic_messages(_request())

        assert mock_client.client.post.call_args.args[0] == MESSAGES_URL

    async def test_sends_anthropic_headers_and_config_api_key(self, mock_client):
        await _adapter().anthropic_messages(_request())

        assert mock_client.client.post.call_args.kwargs["headers"] == {
            "content-type": "application/json",
            "anthropic-version": ANTHROPIC_VERSION,
            "x-api-key": "config-key",
        }

    async def test_provider_data_api_key_overrides_config(self, mock_client):
        adapter = _adapter()
        adapter.get_request_provider_data = MagicMock(
            return_value=SimpleNamespace(anthropic_api_key=SecretStr("per-request-key"))
        )

        await adapter.anthropic_messages(_request())

        assert mock_client.client.post.call_args.kwargs["headers"]["x-api-key"] == "per-request-key"

    async def test_missing_api_key_fails_with_a_hint_and_sends_nothing(self, mock_client):
        adapter = _adapter(api_key=None)

        with pytest.raises(ValueError, match="anthropic_api_key"):
            await adapter.anthropic_messages(_request())

        mock_client.client.post.assert_not_awaited()

    async def test_request_body_keeps_fields_the_model_does_not_declare(self, mock_client):
        await _adapter().anthropic_messages(_request(metadata={"user_id": "u1"}, output_config={"effort": "high"}))

        body = mock_client.client.post.call_args.kwargs["json"]
        assert body["metadata"] == {"user_id": "u1"}
        assert body["output_config"] == {"effort": "high"}

    async def test_model_outside_allowed_models_is_rejected(self, mock_client):
        adapter = _adapter(allowed_models=["claude-sonnet-4-6"])

        with pytest.raises(ValueError, match="not in the allowed models list"):
            await adapter.anthropic_messages(_request(model="claude-haiku-4-5"))

        mock_client.client.post.assert_not_awaited()

    async def test_returns_the_parsed_response(self, mock_client):
        response = await _adapter().anthropic_messages(_request())

        assert response.id == "msg_1"
        assert response.content[0].text == "Paris."
        assert response.usage.input_tokens == 5


class TestNetworkConfig:
    async def test_applies_network_config_to_the_client(self, mock_client):
        await _adapter(network=NETWORK_CONFIG).anthropic_messages(_request())

        kwargs = mock_client.call_args.kwargs
        assert kwargs["verify"] is False
        assert set(kwargs["mounts"]) == {"http://", "https://"}
        assert kwargs["headers"] == {"X-Route": "team-a"}
        assert kwargs["timeout"] == httpx.Timeout(12.0)
        assert kwargs["limits"].max_connections == 7

    async def test_uses_shared_ssl_context_and_call_timeout_without_network_config(self, mock_client):
        adapter = _adapter()

        await adapter.anthropic_messages(_request())

        kwargs = mock_client.call_args.kwargs
        assert kwargs["verify"] is adapter.shared_ssl_context
        assert isinstance(kwargs["verify"], ssl.SSLContext)
        assert kwargs["timeout"] == httpx.Timeout(300.0)

    async def test_streaming_passes_network_kwargs_to_the_shared_helper(self, mock_passthrough):
        async def no_events():
            return
            yield

        mock_passthrough.return_value = no_events()

        result = await _adapter(network=NETWORK_CONFIG).anthropic_messages(_request(stream=True))
        async for _ in result:
            pass

        call_kwargs = mock_passthrough.call_args.kwargs
        assert call_kwargs["url"] == MESSAGES_URL
        assert call_kwargs["headers"]["x-api-key"] == "config-key"
        assert call_kwargs["req_body"]["stream"] is True
        client_kwargs = call_kwargs["httpx_client_kwargs"]
        assert client_kwargs["verify"] is False
        assert set(client_kwargs["mounts"]) == {"http://", "https://"}
        assert client_kwargs["timeout"] == httpx.Timeout(12.0)

    async def test_streaming_uses_shared_ssl_context_without_network_config(self, mock_passthrough):
        async def no_events():
            return
            yield

        mock_passthrough.return_value = no_events()
        adapter = _adapter()

        result = await adapter.anthropic_messages(_request(stream=True))
        async for _ in result:
            pass

        assert mock_passthrough.call_args.kwargs["httpx_client_kwargs"] == {"verify": adapter.shared_ssl_context}

    async def test_streaming_with_a_network_timeout_builds_a_real_client(self, wire):
        """The helper's own default timeout used to collide with network.timeout as a duplicate argument."""
        wire.respond(lambda request: httpx.Response(200, text=""))

        result = await _adapter(network={"timeout": 12.0}).anthropic_messages(_request(stream=True))
        events = [event async for event in result]

        assert events == []
        assert len(wire.requests) == 1


class TestCountTokens:
    async def test_posts_to_count_tokens_with_headers(self, mock_client):
        mock_client.response.json.return_value = {"input_tokens": 14}

        result = await _adapter().anthropic_count_tokens(
            AnthropicCountTokensRequest(model="claude-haiku-4-5", messages=[{"role": "user", "content": "Hi"}])
        )

        post = mock_client.client.post
        assert post.call_args.args[0] == COUNT_TOKENS_URL
        assert post.call_args.kwargs["headers"]["x-api-key"] == "config-key"
        assert post.call_args.kwargs["headers"]["anthropic-version"] == ANTHROPIC_VERSION
        assert result.input_tokens == 14

    async def test_uses_a_shorter_default_timeout_and_the_network_config(self, mock_client):
        mock_client.response.json.return_value = {"input_tokens": 1}
        request = AnthropicCountTokensRequest(model="claude-haiku-4-5", messages=[{"role": "user", "content": "Hi"}])

        await _adapter().anthropic_count_tokens(request)
        assert mock_client.call_args.kwargs["timeout"] == httpx.Timeout(30.0)

        await _adapter(network=NETWORK_CONFIG).anthropic_count_tokens(request)
        assert mock_client.call_args.kwargs["timeout"] == httpx.Timeout(12.0)
        assert mock_client.call_args.kwargs["verify"] is False

    async def test_provider_data_api_key_overrides_config(self, mock_client):
        mock_client.response.json.return_value = {"input_tokens": 1}
        adapter = _adapter()
        adapter.get_request_provider_data = MagicMock(
            return_value=SimpleNamespace(anthropic_api_key=SecretStr("per-request-key"))
        )

        await adapter.anthropic_count_tokens(
            AnthropicCountTokensRequest(model="claude-haiku-4-5", messages=[{"role": "user", "content": "Hi"}])
        )

        assert mock_client.client.post.call_args.kwargs["headers"]["x-api-key"] == "per-request-key"


class TestUpstreamErrors:
    """Anthropic's own status and message must reach the caller instead of a generic 500."""

    async def test_error_response_keeps_status_and_message(self, mock_client):
        mock_client.response.is_error = True
        mock_client.response.status_code = 401
        mock_client.response.json.return_value = {
            "type": "error",
            "error": {"type": "authentication_error", "message": "invalid x-api-key"},
        }

        with pytest.raises(AnthropicAPIError, match="invalid x-api-key") as exc_info:
            await _adapter().anthropic_messages(_request())

        assert exc_info.value.status_code == 401

    async def test_error_response_without_a_json_body_falls_back_to_the_status(self, mock_client):
        mock_client.response.is_error = True
        mock_client.response.status_code = 529
        mock_client.response.json.side_effect = ValueError("not json")

        with pytest.raises(AnthropicAPIError, match="status 529") as exc_info:
            await _adapter().anthropic_messages(_request())

        assert exc_info.value.status_code == 529

    async def test_count_tokens_error_keeps_status(self, mock_client):
        mock_client.response.is_error = True
        mock_client.response.status_code = 400
        mock_client.response.json.return_value = {"type": "error", "error": {"message": "bad request"}}

        with pytest.raises(AnthropicAPIError, match="bad request") as exc_info:
            await _adapter().anthropic_count_tokens(
                AnthropicCountTokensRequest(model="claude-haiku-4-5", messages=[{"role": "user", "content": "Hi"}])
            )

        assert exc_info.value.status_code == 400

    async def test_streaming_error_keeps_status(self, mock_passthrough):
        async def failing():
            raise httpx.HTTPStatusError(
                "429",
                request=httpx.Request("POST", MESSAGES_URL),
                response=httpx.Response(429, request=httpx.Request("POST", MESSAGES_URL)),
            )
            yield

        mock_passthrough.return_value = failing()

        result = await _adapter().anthropic_messages(_request(stream=True))
        with pytest.raises(AnthropicAPIError) as exc_info:
            async for _ in result:
                pass

        assert exc_info.value.status_code == 429

    async def test_streaming_without_an_api_key_fails_before_streaming(self, mock_passthrough):
        with pytest.raises(ValueError, match="anthropic_api_key"):
            await _adapter(api_key=None).anthropic_messages(_request(stream=True))

        mock_passthrough.assert_not_called()

    def test_server_preserves_the_status_of_the_error(self):
        from ogx.core.exceptions.translation import translate_exception

        http_exc = translate_exception(AnthropicAPIError(429, "rate limited"))

        assert http_exc.status_code == 429
        assert http_exc.detail == "rate limited"


class TestOnTheWire:
    """Real httpx requests and SSE parsing against a mock transport."""

    async def test_non_streaming_request_body_and_headers(self, wire):
        request = _request(thinking=AnthropicThinkingConfig(type="enabled", budget_tokens=2048), temperature=1.0)

        response = await _adapter().anthropic_messages(request)

        sent = wire.requests[0]
        assert sent.method == "POST"
        assert str(sent.url) == MESSAGES_URL
        assert sent.headers["x-api-key"] == "config-key"
        assert sent.headers["anthropic-version"] == ANTHROPIC_VERSION
        body = json.loads(sent.content)
        assert body["thinking"] == {"type": "enabled", "budget_tokens": 2048}
        assert body["model"] == "claude-haiku-4-5"
        assert body["max_tokens"] == 256
        assert response.content[0].text == "Paris."

    async def test_streaming_relays_thinking_events(self, wire):
        sse = "\n".join(
            [
                "event: message_start",
                'data: {"type": "message_start", "message": {"id": "msg_1", "type": "message", "role": "assistant",'
                ' "model": "claude-haiku-4-5", "content": [], "stop_reason": null,'
                ' "usage": {"input_tokens": 5, "output_tokens": 0}}}',
                "",
                "event: content_block_start",
                'data: {"type": "content_block_start", "index": 0,'
                ' "content_block": {"type": "thinking", "thinking": "", "signature": ""}}',
                "",
                "event: content_block_delta",
                'data: {"type": "content_block_delta", "index": 0,'
                ' "delta": {"type": "thinking_delta", "thinking": "Capital of France."}}',
                "",
                "event: content_block_stop",
                'data: {"type": "content_block_stop", "index": 0}',
                "",
                "event: message_stop",
                'data: {"type": "message_stop"}',
                "",
            ]
        )
        wire.respond(lambda request: httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"}))

        result = await _adapter().anthropic_messages(
            _request(stream=True, thinking=AnthropicThinkingConfig(type="enabled", budget_tokens=1024))
        )
        events = [event async for event in result]

        assert [e.type for e in events] == [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_stop",
        ]
        assert events[1].content_block.type == "thinking"
        assert events[2].delta.thinking == "Capital of France."
        assert json.loads(wire.requests[0].content)["stream"] is True
        assert wire.requests[0].headers["x-api-key"] == "config-key"

    async def test_upstream_error_reaches_the_caller_with_its_message(self, wire):
        wire.respond(
            lambda request: httpx.Response(
                401,
                json={"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}},
            )
        )

        with pytest.raises(AnthropicAPIError, match="invalid x-api-key") as exc_info:
            await _adapter().anthropic_messages(_request())

        assert exc_info.value.status_code == 401
