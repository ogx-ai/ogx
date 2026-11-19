# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for the TypeSafe SystemOne (judgment) inference provider.

The official typesafe SDK client is mocked at its import boundary; no network is touched.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx2
import pytest
from pydantic import ValidationError
from typesafe_sdk import TypeSafeAPIError, TypeSafeAuthenticationError

import ogx.providers.remote.inference.typesafe.typesafe as typesafe_module
from ogx.core.request_headers import PROVIDER_DATA_VAR
from ogx.providers.remote.inference.typesafe.config import TypeSafeConfig
from ogx.providers.remote.inference.typesafe.typesafe import TypeSafeInferenceAdapter
from ogx.providers.utils.inference.systemone import SystemOneProviderError
from ogx_api import Model, ModelType, SystemOneRequest, SystemOneResponse

_VALIDATOR = "ogx.providers.remote.inference.typesafe.config.TypeSafeProviderDataValidator"

# judge() delegates to the shared util, so its SDK client is patched in the util's namespace;
# list_models() still builds its own client, patched in the provider module's namespace.
_JUDGE_CLIENT = "ogx.providers.utils.inference.systemone.AsyncTypeSafeClient"


def _patch_judge_client(client: MagicMock):
    return patch(_JUDGE_CLIENT, return_value=client)


def _patch_list_client(client: MagicMock):
    return patch.object(typesafe_module, "AsyncTypeSafeClient", return_value=client)


def _make_config(api_key: str | None = "test-key", **kwargs) -> TypeSafeConfig:
    return TypeSafeConfig(api_key=api_key, **kwargs)


def _make_adapter(config: TypeSafeConfig | None = None) -> TypeSafeInferenceAdapter:
    adapter = TypeSafeInferenceAdapter(config=config or _make_config())
    adapter.__provider_id__ = "typesafe"
    adapter.__provider_spec__ = MagicMock(provider_type="remote::typesafe", provider_data_validator=_VALIDATOR)
    return adapter


def _mock_client() -> MagicMock:
    client = MagicMock()
    client.system_one = AsyncMock()
    client.aclose = AsyncMock()
    client.models = MagicMock()
    client.models.list = AsyncMock()
    return client


def _judge_request() -> SystemOneRequest:
    return SystemOneRequest(
        model="jev-latest",
        state="I was charged twice. Please help.",
        questions={"billing": {"type": "noul", "instructions": "Is this about billing?"}},
    )


def _sdk_response(body: dict[str, object]) -> SimpleNamespace:
    """Stand-in for the SDK response: exposes the raw response body bytes, like the real client."""
    return SimpleNamespace(raw_http_response=SimpleNamespace(content=json.dumps(body).encode()))


async def test_judge_forwards_to_sdk_and_returns_wire_response():
    body = {
        "model": "jev-latest",
        "answers": {"billing": {"type": "noul", "noul": 0.9}},
        "usage": {"input_tokens": 12, "output_tokens": 4},
    }
    client = _mock_client()
    client.system_one.return_value = _sdk_response(body)
    adapter = _make_adapter()

    with _patch_judge_client(client):
        result = await adapter.judge(_judge_request())

    # The response is reconstructed from the raw body into the OGX wire model, not cast through.
    assert isinstance(result, SystemOneResponse)
    assert result.model == "jev-latest"
    assert result.answers["billing"].noul == 0.9
    assert result.usage.input_tokens == 12
    assert result.usage.output_tokens == 4
    client.system_one.assert_awaited_once()
    kwargs = client.system_one.call_args.kwargs
    assert kwargs["state"] == "I was charged twice. Please help."
    assert kwargs["model"] == "jev-latest"
    # Questions are forwarded as normalized raw dictionaries (the SDK re-validates them).
    assert kwargs["questions"] == {"billing": {"type": "noul", "instructions": "Is this about billing?"}}


async def test_judge_with_no_api_key_raises():
    adapter = _make_adapter(_make_config(api_key=None))
    with pytest.raises(ValueError, match="no API key"):
        await adapter.judge(_judge_request())


async def test_judge_translates_sdk_error_to_provider_error():
    # The provider translates the SDK's typed errors into a general error carrying the upstream
    # status, so the core can surface it without importing the SDK. 4xx is passed through.
    client = _mock_client()
    client.system_one.side_effect = TypeSafeAuthenticationError(
        status=401, body=None, headers=httpx2.Headers(), message="Invalid API key"
    )
    adapter = _make_adapter(_make_config(api_key="test-key"))
    with _patch_judge_client(client):
        with pytest.raises(SystemOneProviderError) as exc_info:
            await adapter.judge(_judge_request())
    assert exc_info.value.status_code == 401


async def test_judge_maps_upstream_5xx_to_bad_gateway():
    # Upstream 5xx is mapped to a bad-gateway (an upstream failure, not an OGX error).
    client = _mock_client()
    client.system_one.side_effect = TypeSafeAPIError(
        status=503, body=None, headers=httpx2.Headers(), message="upstream unavailable"
    )
    adapter = _make_adapter(_make_config(api_key="test-key"))
    with _patch_judge_client(client):
        with pytest.raises(SystemOneProviderError) as exc_info:
            await adapter.judge(_judge_request())
    assert exc_info.value.status_code == 502


async def test_list_models_maps_metadata_to_systemone_models():
    client = _mock_client()
    client.models.list.return_value = SimpleNamespace(
        models=(
            SimpleNamespace(name="jev-latest", description="General decision model", release_date="2026-09-15"),
            SimpleNamespace(name="jev-mini", description="Small decision model", release_date="2026-09-16"),
        )
    )
    adapter = _make_adapter()

    with _patch_list_client(client):
        models = await adapter.list_models()

    assert [m.identifier for m in models] == ["jev-latest", "jev-mini"]
    assert all(m.model_type == ModelType.systemone for m in models)
    assert all(isinstance(m, Model) for m in models)
    assert models[0].provider_id == "typesafe"
    assert models[0].provider_resource_id == "jev-latest"
    assert models[0].metadata == {"description": "General decision model", "release_date": "2026-09-15"}


async def test_list_models_returns_none_without_api_key():
    adapter = _make_adapter(_make_config(api_key=None))
    assert await adapter.list_models() is None


async def test_list_models_respects_allowed_models():
    client = _mock_client()
    client.models.list.return_value = SimpleNamespace(
        models=(
            SimpleNamespace(name="jev-latest", description="d", release_date="2026-09-15"),
            SimpleNamespace(name="jev-mini", description="d", release_date="2026-09-16"),
        )
    )
    adapter = _make_adapter(_make_config(allowed_models=["jev-mini"]))

    with _patch_list_client(client):
        models = await adapter.list_models()

    assert [m.identifier for m in models] == ["jev-mini"]


def test_build_client_is_fresh_each_call():
    # No client cache: every _build_client() constructs a new client.
    adapter = _make_adapter()
    instances: list[MagicMock] = []

    def _new_client(*args, **kwargs):
        client = _mock_client()
        instances.append(client)
        return client

    with patch.object(typesafe_module, "AsyncTypeSafeClient", side_effect=_new_client):
        adapter._build_client()
        adapter._build_client()

    assert len(instances) == 2


def test_build_client_uses_per_request_override_key():
    adapter = _make_adapter(_make_config(api_key="config-key"))
    token = PROVIDER_DATA_VAR.set({"typesafe_api_key": "override-key"})
    try:
        with patch.object(typesafe_module, "AsyncTypeSafeClient") as mock_cls:
            mock_cls.return_value = _mock_client()
            adapter._build_client()
            assert mock_cls.call_args.kwargs["api_key"] == "override-key"
    finally:
        PROVIDER_DATA_VAR.reset(token)


def test_per_request_provider_data_key_overrides_config():
    adapter = _make_adapter(_make_config(api_key="config-key"))
    token = PROVIDER_DATA_VAR.set({"typesafe_api_key": "override-key"})
    try:
        assert adapter._get_api_key() == "override-key"
    finally:
        PROVIDER_DATA_VAR.reset(token)


def test_per_request_provider_data_absent_uses_config_key():
    adapter = _make_adapter(_make_config(api_key="config-key"))
    token = PROVIDER_DATA_VAR.set(None)
    try:
        assert adapter._get_api_key() == "config-key"
    finally:
        PROVIDER_DATA_VAR.reset(token)


@pytest.mark.parametrize(
    "method",
    [
        "rerank",
        "openai_completion",
        "openai_chat_completion",
        "openai_chat_completions_with_reasoning",
        "openai_embeddings",
        "anthropic_messages",
        "anthropic_count_tokens",
    ],
)
async def test_unsupported_operations_raise_not_implemented(method):
    adapter = _make_adapter()
    with pytest.raises(NotImplementedError):
        await getattr(adapter, method)(object())


async def test_judge_closes_client_after_call():
    client = _mock_client()
    client.system_one.return_value = _sdk_response(
        {"model": "jev-latest", "answers": {"billing": {"type": "noul", "noul": 0.9}}}
    )
    adapter = _make_adapter()

    with _patch_judge_client(client):
        await adapter.judge(_judge_request())

    client.aclose.assert_awaited_once()


async def test_judge_raises_when_response_does_not_match_wire_model():
    # An answer type the wire model does not know is an incongruity that must surface (and the
    # client must still be closed) rather than being cast through unchecked.
    client = _mock_client()
    client.system_one.return_value = _sdk_response(
        {"model": "jev-latest", "answers": {"mystery": {"type": "unknown", "value": 1}}}
    )
    adapter = _make_adapter()

    with pytest.raises(ValidationError):
        with _patch_judge_client(client):
            await adapter.judge(_judge_request())

    client.aclose.assert_awaited_once()


async def test_list_models_closes_client_after_call():
    client = _mock_client()
    client.models.list.return_value = SimpleNamespace(
        models=(SimpleNamespace(name="jev-latest", description="d", release_date="2026-09-15"),)
    )
    adapter = _make_adapter()

    with _patch_list_client(client):
        await adapter.list_models()

    client.aclose.assert_awaited_once()


async def test_shutdown_is_noop_without_persisted_client():
    adapter = _make_adapter()
    await adapter.shutdown()
