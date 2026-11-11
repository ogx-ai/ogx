# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for the shared SystemOne (judgment) util used by TypeSafe-compatible providers.

The official typesafe SDK client is mocked at its import boundary; no network is touched.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx2
import pytest
from pydantic import ValidationError
from typesafe_sdk import (
    TypeSafeAPIConnectionError,
    TypeSafeAPIError,
    TypeSafeAPIResponseValidationError,
    TypeSafeAPITimeoutError,
)

from ogx.providers.utils.inference.network_config import NetworkConfig
from ogx.providers.utils.inference.systemone import (
    LOCAL_ENGINE_API_KEY,
    SystemOneProviderError,
    systemone_judge,
    to_provider_error,
)
from ogx_api import SystemOneRequest, SystemOneResponse

_UTIL_CLIENT = "ogx.providers.utils.inference.systemone.AsyncTypeSafeClient"


def _patch_client(client: MagicMock):
    return patch(_UTIL_CLIENT, return_value=client)


def _mock_client() -> MagicMock:
    client = MagicMock()
    client.system_one = AsyncMock()
    client.aclose = AsyncMock()
    return client


def _judge_request() -> SystemOneRequest:
    return SystemOneRequest(
        model="laya",
        state="I was charged twice. Please help.",
        questions={"billing": {"type": "noul", "instructions": "Is this about billing?"}},
    )


def _sdk_response(body: dict[str, object]) -> SimpleNamespace:
    """Stand-in for the SDK response: exposes the raw response body bytes, like the real client."""
    return SimpleNamespace(raw_http_response=SimpleNamespace(content=json.dumps(body).encode()))


_BODY = {
    "model": "laya",
    "answers": {"billing": {"type": "noul", "noul": 0.9}},
    "usage": {"input_tokens": 12, "output_tokens": 0},
}


async def test_judge_forwards_to_sdk_and_returns_wire_response():
    client = _mock_client()
    client.system_one.return_value = _sdk_response(_BODY)

    with _patch_client(client):
        result = await systemone_judge("http://localhost:11434", LOCAL_ENGINE_API_KEY, _judge_request())

    assert isinstance(result, SystemOneResponse)
    assert result.model == "laya"
    assert result.answers["billing"].noul == 0.9
    assert result.usage.input_tokens == 12
    client.system_one.assert_awaited_once()
    kwargs = client.system_one.call_args.kwargs
    assert kwargs["state"] == "I was charged twice. Please help."
    assert kwargs["model"] == "laya"
    assert kwargs["questions"] == {"billing": {"type": "noul", "instructions": "Is this about billing?"}}
    client.aclose.assert_awaited_once()


def _mock_client_with_response() -> MagicMock:
    client = _mock_client()
    client.system_one.return_value = _sdk_response(_BODY)
    return client


async def test_judge_builds_client_with_explicit_key_and_stripped_base_url():
    # Both are passed explicitly so the SDK never falls back to its environment variables,
    # and trailing slashes must not produce a doubled path segment.
    with _patch_client(_mock_client_with_response()) as mock_cls:
        await systemone_judge("http://localhost:11434/", LOCAL_ENGINE_API_KEY, _judge_request())
    assert mock_cls.call_args.kwargs["api_key"] == LOCAL_ENGINE_API_KEY
    assert mock_cls.call_args.kwargs["base_url"] == "http://localhost:11434"


async def test_judge_without_network_config_passes_no_http_client():
    with _patch_client(_mock_client_with_response()) as mock_cls:
        await systemone_judge("http://localhost:11434", LOCAL_ENGINE_API_KEY, _judge_request())
    assert "http_client" not in mock_cls.call_args.kwargs


async def test_judge_with_network_config_passes_http_client():
    network = NetworkConfig(proxy={"url": "http://proxy.example.com:3128"})
    with _patch_client(_mock_client_with_response()) as mock_cls:
        await systemone_judge("http://localhost:11434", LOCAL_ENGINE_API_KEY, _judge_request(), network=network)
    http_client = mock_cls.call_args.kwargs["http_client"]
    assert isinstance(http_client, httpx2.AsyncClient)


async def test_judge_passes_4xx_status_through():
    client = _mock_client()
    client.system_one.side_effect = TypeSafeAPIError(
        status=400, body=None, headers=httpx2.Headers(), message="bad request"
    )
    with _patch_client(client):
        with pytest.raises(SystemOneProviderError) as exc_info:
            await systemone_judge("http://localhost:11434", LOCAL_ENGINE_API_KEY, _judge_request())
    assert exc_info.value.status_code == 400
    client.aclose.assert_awaited_once()


async def test_judge_maps_upstream_5xx_to_bad_gateway():
    client = _mock_client()
    client.system_one.side_effect = TypeSafeAPIError(
        status=503, body=None, headers=httpx2.Headers(), message="upstream unavailable"
    )
    with _patch_client(client):
        with pytest.raises(SystemOneProviderError) as exc_info:
            await systemone_judge("http://localhost:11434", LOCAL_ENGINE_API_KEY, _judge_request())
    assert exc_info.value.status_code == 502


async def test_judge_maps_200_with_invalid_body_to_bad_gateway():
    # A 200 whose body does not match the wire contract is an upstream failure, not a client error.
    client = _mock_client()
    client.system_one.side_effect = TypeSafeAPIResponseValidationError(
        status=200, body=None, headers=httpx2.Headers(), field_path="answers.billing.noul"
    )
    with _patch_client(client):
        with pytest.raises(SystemOneProviderError) as exc_info:
            await systemone_judge("http://localhost:11434", LOCAL_ENGINE_API_KEY, _judge_request())
    assert exc_info.value.status_code == 502
    client.aclose.assert_awaited_once()


async def test_judge_maps_connection_error_to_bad_gateway():
    client = _mock_client()
    client.system_one.side_effect = TypeSafeAPIConnectionError("connection refused")
    with _patch_client(client):
        with pytest.raises(SystemOneProviderError) as exc_info:
            await systemone_judge("http://localhost:11434", LOCAL_ENGINE_API_KEY, _judge_request())
    assert exc_info.value.status_code == 502
    client.aclose.assert_awaited_once()


async def test_judge_maps_timeout_to_bad_gateway():
    client = _mock_client()
    client.system_one.side_effect = TypeSafeAPITimeoutError(timeout=30.0)
    with _patch_client(client):
        with pytest.raises(SystemOneProviderError) as exc_info:
            await systemone_judge("http://localhost:11434", LOCAL_ENGINE_API_KEY, _judge_request())
    assert exc_info.value.status_code == 502
    client.aclose.assert_awaited_once()


async def test_judge_closes_client_when_response_does_not_match_wire_model():
    # An answer type the wire model does not know must surface as a validation error (and the
    # client must still be closed) rather than being cast through unchecked.
    client = _mock_client()
    client.system_one.return_value = _sdk_response(
        {"model": "laya", "answers": {"mystery": {"type": "unknown", "value": 1}}}
    )
    with pytest.raises(ValidationError):
        with _patch_client(client):
            await systemone_judge("http://localhost:11434", LOCAL_ENGINE_API_KEY, _judge_request())
    client.aclose.assert_awaited_once()


async def test_judge_drops_declared_extension_fields_before_validation():
    # SGLang reports x_label_mass on every answer; a provider that declares it gets a valid
    # wire response, with the extension gone from the answers.
    client = _mock_client()
    client.system_one.return_value = _sdk_response(
        {
            "model": "Qwen/Qwen3-0.6B",
            "answers": {
                "billing": {"type": "noul", "noul": 0.99, "x_label_mass": 0.35},
                "team": {
                    "type": "choice",
                    "choice": "billing",
                    "confidence": 0.998,
                    "probabilities": {"billing": 0.999, "technical": 0.001},
                    "x_label_mass": 0.97,
                },
            },
            "usage": {"input_tokens": 109, "output_tokens": 0},
        }
    )
    with _patch_client(client):
        result = await systemone_judge(
            "http://localhost:30000",
            LOCAL_ENGINE_API_KEY,
            _judge_request(),
            extension_fields=frozenset({"x_label_mass"}),
        )
    assert result.answers["billing"].noul == 0.99
    assert result.answers["team"].choice == "billing"
    assert "x_label_mass" not in result.answers["team"].model_dump()


async def test_judge_still_rejects_undeclared_extra_fields():
    client = _mock_client()
    client.system_one.return_value = _sdk_response(
        {"model": "laya", "answers": {"billing": {"type": "noul", "noul": 0.9, "x_label_mass": 0.35}}}
    )
    with pytest.raises(ValidationError):
        with _patch_client(client):
            await systemone_judge("http://localhost:11434", LOCAL_ENGINE_API_KEY, _judge_request())
    client.aclose.assert_awaited_once()


def test_to_provider_error_keeps_4xx_and_maps_5xx_and_none():
    make = lambda status: TypeSafeAPIError(status=status, body=None, headers=httpx2.Headers())  # noqa: E731
    assert to_provider_error(make(404)).status_code == 404
    assert to_provider_error(make(500)).status_code == 502
    assert to_provider_error(make(502)).status_code == 502
