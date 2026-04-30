# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ogx.providers.remote.inference.vllm.config import VLLMInferenceAdapterConfig
from ogx.providers.remote.inference.vllm.vllm import VLLMInferenceAdapter
from ogx_api import (
    CreateResponseRequest,
    OpenAIResponseObject,
    OpenAIResponseReasoning,
)


@pytest.fixture
def vllm_config_native():
    return VLLMInferenceAdapterConfig(
        base_url="http://vllm-server:8000/v1",
        max_tokens=4096,
        api_token="test-token",
        native_responses=True,
    )


@pytest.fixture
def vllm_config_disabled():
    return VLLMInferenceAdapterConfig(
        base_url="http://vllm-server:8000/v1",
        max_tokens=4096,
        native_responses=False,
    )


@pytest.fixture
def adapter_native(vllm_config_native):
    return VLLMInferenceAdapter(config=vllm_config_native)


@pytest.fixture
def adapter_disabled(vllm_config_disabled):
    return VLLMInferenceAdapter(config=vllm_config_disabled)


# --- Config flag behavior ---


async def test_native_responses_disabled_raises(adapter_disabled):
    request = CreateResponseRequest(
        model="test-model",
        input="Hello",
        stream=False,
    )
    with pytest.raises(NotImplementedError, match="Native responses disabled"):
        await adapter_disabled.openai_response(request)


async def test_native_responses_enabled_does_not_raise(adapter_native):
    request = CreateResponseRequest(
        model="test-model",
        input="Hello",
        stream=False,
    )
    response_json = {
        "id": "resp_123",
        "created_at": 1000,
        "model": "test-model",
        "object": "response",
        "output": [],
        "status": "completed",
        "store": True,
    }
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = response_json

    adapter_native._get_api_key_from_config_or_provider_data = MagicMock(return_value="test-token")

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        result = await adapter_native.openai_response(request)
        assert isinstance(result, OpenAIResponseObject)
        assert result.id == "resp_123"


# --- Payload construction ---


async def test_payload_always_sets_store_false(adapter_native):
    request = CreateResponseRequest(
        model="test-model",
        input="Hello",
        stream=False,
        store=True,
    )
    payload = adapter_native._build_response_payload(request)
    assert payload["store"] is False


async def test_payload_includes_basic_fields(adapter_native):
    request = CreateResponseRequest(
        model="test-model",
        input="Hello world",
        stream=True,
        temperature=0.5,
        top_p=0.9,
        max_output_tokens=256,
    )
    payload = adapter_native._build_response_payload(request)
    assert payload["model"] == "test-model"
    assert payload["input"] == "Hello world"
    assert payload["stream"] is True
    assert payload["temperature"] == 0.5
    assert payload["top_p"] == 0.9
    assert payload["max_output_tokens"] == 256


async def test_payload_includes_tools(adapter_native):
    request = CreateResponseRequest(
        model="test-model",
        input="Hello",
        stream=False,
        tools=[
            {
                "type": "function",
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            }
        ],
    )
    payload = adapter_native._build_response_payload(request)
    assert payload["tools"] is not None
    assert len(payload["tools"]) == 1


async def test_payload_includes_reasoning(adapter_native):
    request = CreateResponseRequest(
        model="test-model",
        input="Hello",
        stream=False,
        reasoning=OpenAIResponseReasoning(effort="high"),
    )
    payload = adapter_native._build_response_payload(request)
    assert payload["reasoning"]["effort"] == "high"


async def test_payload_includes_frequency_penalty(adapter_native):
    request = CreateResponseRequest(
        model="test-model",
        input="Hello",
        stream=False,
        frequency_penalty=0.5,
    )
    payload = adapter_native._build_response_payload(request)
    assert payload["frequency_penalty"] == 0.5


async def test_payload_omits_none_optional_fields(adapter_native):
    request = CreateResponseRequest(
        model="test-model",
        input="Hello",
        stream=False,
    )
    payload = adapter_native._build_response_payload(request)
    assert "instructions" not in payload
    assert "temperature" not in payload
    assert "tools" not in payload
    assert "reasoning" not in payload
    assert "metadata" not in payload


async def test_payload_serializes_message_list_input(adapter_native):
    request = CreateResponseRequest(
        model="test-model",
        input=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ],
        stream=False,
    )
    payload = adapter_native._build_response_payload(request)
    assert isinstance(payload["input"], list)
    assert len(payload["input"]) == 2


# --- Auth headers ---


async def test_auth_headers_with_token(adapter_native):
    adapter_native._get_api_key_from_config_or_provider_data = MagicMock(return_value="my-secret-key")
    headers = adapter_native._build_auth_headers()
    assert headers["Authorization"] == "Bearer my-secret-key"


async def test_auth_headers_without_token(adapter_native):
    adapter_native._get_api_key_from_config_or_provider_data = MagicMock(return_value="NO KEY REQUIRED")
    headers = adapter_native._build_auth_headers()
    assert "Authorization" not in headers


# --- Non-streaming fetch ---


async def test_fetch_response_404_raises_not_implemented(adapter_native):
    mock_response = MagicMock()
    mock_response.status_code = 404

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        with pytest.raises(NotImplementedError, match="does not support"):
            await adapter_native._fetch_response("http://test/v1/responses", {}, {"model": "m"})


async def test_fetch_response_500_raises_runtime_error(adapter_native):
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        with pytest.raises(RuntimeError, match="Failed to get response from vLLM"):
            await adapter_native._fetch_response("http://test/v1/responses", {}, {"model": "m"})


async def test_fetch_response_parses_valid_json(adapter_native):
    response_json = {
        "id": "resp_abc",
        "created_at": 1000,
        "model": "test-model",
        "object": "response",
        "output": [
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "Hello!", "annotations": []}],
            }
        ],
        "status": "completed",
        "store": True,
    }
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = response_json

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        result = await adapter_native._fetch_response("http://test/v1/responses", {}, {"model": "m"})
        assert isinstance(result, OpenAIResponseObject)
        assert result.id == "resp_abc"
        assert result.status == "completed"
        assert len(result.output) == 1


# --- Streaming ---


async def test_stream_response_parses_sse_events(adapter_native):
    sse_lines = [
        'data: {"type":"response.created","response":{"id":"resp_1","created_at":1,"model":"m","object":"response","output":[],"status":"in_progress","store":true},"sequence_number":0}',
        'data: {"type":"response.output_text.delta","delta":"Hello","item_id":"msg_1","output_index":0,"content_index":0,"sequence_number":1}',
        'data: {"type":"response.completed","response":{"id":"resp_1","created_at":1,"model":"m","object":"response","output":[],"status":"completed","store":true},"sequence_number":2}',
        "data: [DONE]",
    ]

    async def mock_aiter_lines():
        for line in sse_lines:
            yield line

    mock_stream_response = AsyncMock()
    mock_stream_response.status_code = 200
    mock_stream_response.aiter_lines = mock_aiter_lines

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.stream = MagicMock(return_value=AsyncMock())

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        stream_ctx = AsyncMock()
        stream_ctx.__aenter__ = AsyncMock(return_value=mock_stream_response)
        stream_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_client.stream = MagicMock(return_value=stream_ctx)

        mock_client_cls.return_value = mock_client

        result = await adapter_native._stream_response("http://test/v1/responses", {}, {"model": "m"})

        events = []
        async for event in result:
            events.append(event)

        assert len(events) == 3
        assert events[0].type == "response.created"
        assert events[1].type == "response.output_text.delta"
        assert events[1].delta == "Hello"
        assert events[2].type == "response.completed"


async def test_stream_response_404_raises_not_implemented(adapter_native):
    mock_stream_response = AsyncMock()
    mock_stream_response.status_code = 404

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        stream_ctx = AsyncMock()
        stream_ctx.__aenter__ = AsyncMock(return_value=mock_stream_response)
        stream_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_client.stream = MagicMock(return_value=stream_ctx)

        mock_client_cls.return_value = mock_client

        result = await adapter_native._stream_response("http://test/v1/responses", {}, {"model": "m"})

        with pytest.raises(NotImplementedError, match="does not support"):
            async for _ in result:
                pass


# --- Endpoint construction ---


async def test_endpoint_uses_base_url(adapter_native):
    request = CreateResponseRequest(
        model="test-model",
        input="Hello",
        stream=False,
    )
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "resp_1",
        "created_at": 1,
        "model": "test-model",
        "object": "response",
        "output": [],
        "status": "completed",
        "store": True,
    }

    adapter_native._get_api_key_from_config_or_provider_data = MagicMock(return_value="test-token")

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        await adapter_native.openai_response(request)

        call_args = mock_client.post.call_args
        assert call_args[0][0] == "http://vllm-server:8000/v1/responses"
