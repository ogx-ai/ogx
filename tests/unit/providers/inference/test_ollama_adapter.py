# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import ssl
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import httpx
import pytest

from ogx.providers.inline.responses.builtin.responses.types import AssistantMessageWithReasoning
from ogx.providers.remote.inference.ollama.config import OllamaImplConfig
from ogx.providers.remote.inference.ollama.ollama import OllamaInferenceAdapter
from ogx.providers.utils.inference.openai_compat import prepare_openai_completion_params
from ogx_api import (
    HealthStatus,
    OpenAIAssistantMessageParam,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAIUserMessageParam,
)
from ogx_api.messages.models import AnthropicCountTokensRequest, AnthropicCreateMessageRequest


async def _empty_stream():
    if False:
        yield None


async def test_openai_chat_completions_with_reasoning_keeps_messages_typed():
    """Ollama should remap reasoning fields without widening messages to raw dicts."""
    adapter = OllamaInferenceAdapter(config=OllamaImplConfig(base_url="http://localhost:11434/v1"))
    adapter.__provider_id__ = "ollama"

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=_empty_stream())

    with patch.object(type(adapter), "client", new_callable=PropertyMock, return_value=mock_client):
        with patch("ogx.providers.utils.inference.openai_mixin.localize_image_content") as mock_localize:
            mock_localize.return_value = (b"fake_image_data", "jpeg")

            captured_messages = None

            async def _capture_prepare_params(**kwargs):
                nonlocal captured_messages
                captured_messages = kwargs["messages"]
                return await prepare_openai_completion_params(**kwargs)

            with patch(
                "ogx.providers.utils.inference.openai_mixin.prepare_openai_completion_params",
                new=AsyncMock(side_effect=_capture_prepare_params),
            ):
                result = await adapter.openai_chat_completions_with_reasoning(
                    OpenAIChatCompletionRequestWithExtraBody(
                        model="test-model",
                        stream=True,
                        messages=[
                            AssistantMessageWithReasoning(
                                role="assistant",
                                content="Previous answer",
                                reasoning_content="Step 1",
                            ),
                            OpenAIUserMessageParam(
                                role="user",
                                content=[
                                    {"type": "text", "text": "What's in this image?"},
                                    {"type": "image_url", "image_url": {"url": "http://example.com/image.jpg"}},
                                ],
                            ),
                        ],
                    )
                )

    assert result is not None
    mock_localize.assert_called_once_with("http://example.com/image.jpg")

    assert captured_messages is not None
    assert type(captured_messages[0]) is OpenAIAssistantMessageParam
    assert captured_messages[0].model_dump(exclude_none=True)["reasoning"] == "Step 1"
    assert "reasoning_content" not in captured_messages[0].model_dump(exclude_none=True)

    mock_client.chat.completions.create.assert_called_once()
    processed_messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
    assert processed_messages[0]["reasoning"] == "Step 1"
    assert "reasoning_content" not in processed_messages[0]
    assert processed_messages[1]["content"][1]["image_url"]["url"] == "data:image/jpeg;base64,ZmFrZV9pbWFnZV9kYXRh"


async def test_health_ok():
    """health() probes the unauthenticated GET /api/version endpoint and reports OK on success."""
    adapter = OllamaInferenceAdapter(config=OllamaImplConfig(base_url="http://localhost:11434/v1"))
    adapter.__provider_id__ = "ollama"

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_client_instance = MagicMock()
        mock_client_instance.get = AsyncMock(return_value=mock_response)
        mock_client_class.return_value.__aenter__.return_value = mock_client_instance

        health_response = await adapter.health()

    assert health_response["status"] == HealthStatus.OK
    mock_client_instance.get.assert_called_once()
    # The /v1 suffix must be stripped so we hit the Ollama root /api/version endpoint.
    assert mock_client_instance.get.call_args[0][0] == "http://localhost:11434/api/version"


async def test_health_error():
    """health() reports ERROR with a message when the version probe fails."""
    adapter = OllamaInferenceAdapter(config=OllamaImplConfig(base_url="http://localhost:11434/v1"))
    adapter.__provider_id__ = "ollama"

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client_instance = MagicMock()
        mock_client_instance.get = AsyncMock(side_effect=Exception("Connection failed"))
        mock_client_class.return_value.__aenter__.return_value = mock_client_instance

        health_response = await adapter.health()

    assert health_response["status"] == HealthStatus.ERROR
    assert "Connection failed" in health_response["message"]


NETWORK_CONFIG = {
    "tls": {"verify": False},
    "proxy": {"url": "http://proxy.example.com:3128"},
    "headers": {"X-Route": "team-a"},
    "timeout": 12.0,
    "limits": {"max_connections": 7},
}

_MESSAGE_RESPONSE = {
    "id": "msg-1",
    "content": [{"type": "text", "text": "Hello"}],
    "role": "assistant",
    "stop_reason": "end_turn",
    "type": "message",
    "model": "test-model",
    "stop_sequences": None,
    "usage": {"input_tokens": 5, "output_tokens": 5},
}


def _make_adapter(**config_kwargs) -> OllamaInferenceAdapter:
    adapter = OllamaInferenceAdapter(config=OllamaImplConfig(base_url="http://localhost:11434/v1", **config_kwargs))
    adapter.__provider_id__ = "ollama"
    adapter.get_request_provider_data = MagicMock(return_value=None)
    return adapter


async def _call_health(adapter: OllamaInferenceAdapter) -> None:
    await adapter.health()


async def _call_count_tokens(adapter: OllamaInferenceAdapter) -> None:
    await adapter.anthropic_count_tokens(
        AnthropicCountTokensRequest(model="test-model", messages=[{"role": "user", "content": "Hi"}])
    )


async def _call_messages(adapter: OllamaInferenceAdapter) -> None:
    await adapter.anthropic_messages(
        AnthropicCreateMessageRequest(
            model="test-model", messages=[{"role": "user", "content": "Hi"}], max_tokens=16, stream=False
        )
    )


# (call, the call's own default timeout in seconds)
ADHOC_CALLS = [
    pytest.param(_call_health, 30.0, id="health"),
    pytest.param(_call_count_tokens, 30.0, id="anthropic_count_tokens"),
    pytest.param(_call_messages, 300.0, id="anthropic_messages"),
]


async def _client_kwargs_used_by(call, adapter: OllamaInferenceAdapter) -> dict:
    """Run one ad-hoc call against a mocked httpx and return the kwargs the client was built with."""
    with patch("httpx.AsyncClient") as mock_client_class:
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.side_effect = lambda: {"input_tokens": 3, **_MESSAGE_RESPONSE}
        client = MagicMock()
        client.get = AsyncMock(return_value=response)
        client.post = AsyncMock(return_value=response)
        mock_client_class.return_value.__aenter__.return_value = client

        await call(adapter)

    mock_client_class.assert_called_once()
    return mock_client_class.call_args.kwargs


@pytest.mark.parametrize("call,_default_timeout", ADHOC_CALLS)
async def test_adhoc_calls_apply_network_config(call, _default_timeout):
    """Health and the Anthropic passthrough must reach Ollama through the configured proxy/TLS/headers."""
    kwargs = await _client_kwargs_used_by(call, _make_adapter(network=NETWORK_CONFIG))

    assert kwargs["verify"] is False
    assert set(kwargs["mounts"]) == {"http://", "https://"}
    assert kwargs["headers"] == {"X-Route": "team-a"}
    assert kwargs["limits"].max_connections == 7


@pytest.mark.parametrize("call,_default_timeout", ADHOC_CALLS)
async def test_adhoc_calls_prefer_configured_network_timeout(call, _default_timeout):
    kwargs = await _client_kwargs_used_by(call, _make_adapter(network=NETWORK_CONFIG))

    assert kwargs["timeout"] == httpx.Timeout(12.0)


@pytest.mark.parametrize("call,default_timeout", ADHOC_CALLS)
async def test_adhoc_calls_use_shared_ssl_context_and_own_timeout_without_network_config(call, default_timeout):
    adapter = _make_adapter()

    kwargs = await _client_kwargs_used_by(call, adapter)

    assert kwargs["verify"] is adapter.shared_ssl_context
    assert isinstance(kwargs["verify"], ssl.SSLContext)
    assert kwargs["timeout"] == httpx.Timeout(default_timeout)
    assert "mounts" not in kwargs


async def test_streaming_passthrough_applies_network_config():
    """The streaming Anthropic passthrough builds its client in a shared helper, so the
    network kwargs must be handed to it."""
    adapter = _make_adapter(network=NETWORK_CONFIG)

    async def _no_events(**_kwargs):
        return
        yield

    with patch(
        "ogx.providers.remote.inference.ollama.ollama.passthrough_anthropic_stream", side_effect=_no_events
    ) as mock_stream:
        result = await adapter.anthropic_messages(
            AnthropicCreateMessageRequest(
                model="test-model", messages=[{"role": "user", "content": "Hi"}], max_tokens=16, stream=True
            )
        )
        async for _ in result:
            pass

    stream_kwargs = mock_stream.call_args.kwargs["httpx_client_kwargs"]
    assert stream_kwargs["verify"] is False
    assert set(stream_kwargs["mounts"]) == {"http://", "https://"}
    assert stream_kwargs["headers"] == {"X-Route": "team-a"}
    assert stream_kwargs["timeout"] == httpx.Timeout(12.0)


@pytest.mark.parametrize("call,_default_timeout", ADHOC_CALLS)
async def test_adhoc_calls_keep_the_shared_ssl_context_when_only_a_proxy_is_configured(call, _default_timeout):
    adapter = _make_adapter(network={"proxy": {"url": "http://proxy.example.com:3128"}})

    kwargs = await _client_kwargs_used_by(call, adapter)

    assert set(kwargs["mounts"]) == {"http://", "https://"}
    assert kwargs["verify"] is adapter.shared_ssl_context
