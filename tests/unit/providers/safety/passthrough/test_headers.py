# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock

import httpx
import pytest
from pydantic import ValidationError

from ogx.providers.remote.safety.passthrough.config import PassthroughSafetyConfig
from ogx_api import (
    OpenAIUserMessageParam,
    ResourceType,
    RunModerationRequest,
    RunShieldRequest,
    Shield,
)

from .conftest import (
    FakePassthroughSafetyAdapter,
    _stub_provider_spec,
    mock_httpx_response,
    provider_data_ctx,
)

_SAFE_RESPONSE = {
    "id": "modr-hdr",
    "model": "text-moderation-latest",
    "results": [{"flagged": False, "categories": {}, "category_scores": {}}],
}


def _make_adapter_with_headers(
    forward_headers: dict[str, str],
    shield_store: AsyncMock,
) -> FakePassthroughSafetyAdapter:
    config = PassthroughSafetyConfig(
        base_url="https://safety.example.com/v1",
        forward_headers=forward_headers,
    )
    a = FakePassthroughSafetyAdapter(config, shield_store)
    _stub_provider_spec(a)
    a._client = AsyncMock(spec=httpx.AsyncClient)
    return a


async def test_forward_headers_maps_keys(adapter_with_forward_headers: FakePassthroughSafetyAdapter) -> None:
    with provider_data_ctx({"maas_api_token": "Bearer tok-123", "tenant_id": "t-456"}):
        headers = adapter_with_forward_headers._build_forward_headers()

    assert headers == {"Authorization": "Bearer tok-123", "X-Tenant-Id": "t-456"}


async def test_forward_headers_skips_missing_keys(
    adapter_with_forward_headers: FakePassthroughSafetyAdapter,
) -> None:
    with provider_data_ctx({"maas_api_token": "Bearer tok-123"}):
        headers = adapter_with_forward_headers._build_forward_headers()

    assert headers == {"Authorization": "Bearer tok-123"}
    assert "X-Tenant-Id" not in headers


async def test_forward_headers_sent_in_run_shield(shield_store: AsyncMock) -> None:
    adapter = _make_adapter_with_headers({"maas_api_token": "Authorization"}, shield_store)
    shield_store.get_shield.return_value = Shield(
        provider_id="passthrough",
        type=ResourceType.shield,
        identifier="test-shield",
        provider_resource_id="text-moderation-latest",
    )

    adapter._client.post.return_value = mock_httpx_response(_SAFE_RESPONSE)
    with provider_data_ctx({"maas_api_token": "Bearer my-token"}):
        await adapter.run_shield(
            RunShieldRequest(shield_id="test-shield", messages=[OpenAIUserMessageParam(content="hello")])
        )

    sent_headers = adapter._client.post.call_args.kwargs.get("headers") or adapter._client.post.call_args[1].get(
        "headers"
    )
    assert sent_headers["Authorization"] == "Bearer my-token"


async def test_forward_headers_sent_in_run_moderation(shield_store: AsyncMock) -> None:
    adapter = _make_adapter_with_headers({"tenant_id": "X-Tenant-Id"}, shield_store)

    adapter._client.post.return_value = mock_httpx_response(_SAFE_RESPONSE)
    with provider_data_ctx({"tenant_id": "tenant-789"}):
        await adapter.run_moderation(RunModerationRequest(input="test", model="text-moderation-latest"))

    sent_headers = adapter._client.post.call_args.kwargs.get("headers") or adapter._client.post.call_args[1].get(
        "headers"
    )
    assert sent_headers["X-Tenant-Id"] == "tenant-789"


async def test_forward_headers_does_not_leak_unlisted_keys(shield_store: AsyncMock) -> None:
    """Only keys listed in forward_headers are sent; other provider data stays local."""
    adapter = _make_adapter_with_headers({"maas_api_token": "Authorization"}, shield_store)

    with provider_data_ctx({"maas_api_token": "Bearer tok", "secret_key": "should-not-appear"}):
        headers = adapter._build_forward_headers()

    assert "Authorization" in headers
    for v in headers.values():
        assert "should-not-appear" not in v


async def test_url_trailing_slash_stripped(shield_store: AsyncMock) -> None:
    config = PassthroughSafetyConfig(base_url="https://safety.example.com/v1/")
    adapter = FakePassthroughSafetyAdapter(config, shield_store)
    _stub_provider_spec(adapter)
    adapter._client = AsyncMock(spec=httpx.AsyncClient)

    shield_store.get_shield.return_value = Shield(
        provider_id="passthrough",
        type=ResourceType.shield,
        identifier="test-shield",
        provider_resource_id="text-moderation-latest",
    )

    adapter._client.post.return_value = mock_httpx_response(_SAFE_RESPONSE)
    await adapter.run_shield(
        RunShieldRequest(shield_id="test-shield", messages=[OpenAIUserMessageParam(content="test")])
    )

    url = (
        adapter._client.post.call_args.args[0]
        if adapter._client.post.call_args.args
        else adapter._client.post.call_args.kwargs.get("url")
    )
    assert "/v1/moderations" in url
    assert "//moderations" not in url


async def test_forward_headers_coerces_non_string_values(shield_store: AsyncMock) -> None:
    adapter = _make_adapter_with_headers(
        {"retry_count": "X-Retry-Count", "debug": "X-Debug"},
        shield_store,
    )

    with provider_data_ctx({"retry_count": 3, "debug": True}):
        headers = adapter._build_forward_headers()

    assert headers["X-Retry-Count"] == "3"
    assert headers["X-Debug"] == "True"


async def test_forward_headers_crlf_stripped(shield_store: AsyncMock) -> None:
    """CRLF sequences are stripped to prevent header injection."""
    adapter = _make_adapter_with_headers({"token": "Authorization"}, shield_store)

    with provider_data_ctx({"token": "Bearer tok\r\nX-Injected: evil"}):
        headers = adapter._build_forward_headers()

    assert headers["Authorization"] == "Bearer tokX-Injected: evil"


async def test_forward_headers_rejects_dunder_keys() -> None:
    with pytest.raises(ValidationError, match="reserved __ prefix"):
        PassthroughSafetyConfig(
            base_url="https://safety.example.com/v1",
            forward_headers={"__authenticated_user": "X-Auth-User"},
        )


async def test_config_api_key_cannot_be_overridden_by_forward_headers(shield_store: AsyncMock) -> None:
    """Config api_key wins even when forward_headers maps something to Authorization."""
    config = PassthroughSafetyConfig(
        base_url="https://safety.example.com/v1",
        api_key="config-secret",
        forward_headers={"user_token": "Authorization"},
    )
    adapter = FakePassthroughSafetyAdapter(config, shield_store)
    _stub_provider_spec(adapter)

    with provider_data_ctx({"user_token": "Bearer user-tok"}):
        headers = adapter._build_request_headers()

    assert headers["Authorization"] == "Bearer config-secret"


@pytest.mark.parametrize(
    "header_name",
    ["Host", "Content-Type", "Content-Length", "Transfer-Encoding", "Cookie", "Set-Cookie", "Connection"],
)
async def test_forward_headers_rejects_blocked_headers(header_name: str) -> None:
    with pytest.raises(ValidationError, match="blocked"):
        PassthroughSafetyConfig(
            base_url="https://safety.example.com/v1",
            forward_headers={"some_key": header_name},
        )


async def test_forward_headers_rejects_multiple_violations() -> None:
    """Both dunder keys and blocked headers are reported in a single error."""
    with pytest.raises(ValidationError, match="reserved __ prefix.*blocked|blocked.*reserved __ prefix"):
        PassthroughSafetyConfig(
            base_url="https://safety.example.com/v1",
            forward_headers={"__internal": "X-Internal", "host_key": "Host"},
        )


async def test_forward_headers_rejects_operator_blocked_headers() -> None:
    with pytest.raises(ValidationError, match="blocked"):
        PassthroughSafetyConfig(
            base_url="https://safety.example.com/v1",
            forward_headers={"trace_id": "X-Internal-Debug"},
            extra_blocked_headers=["x-internal-debug"],
        )


async def test_forward_headers_rejects_operator_blocked_headers_case_insensitive() -> None:
    with pytest.raises(ValidationError, match="blocked"):
        PassthroughSafetyConfig(
            base_url="https://safety.example.com/v1",
            forward_headers={"trace_id": "X-INTERNAL-DEBUG"},
            extra_blocked_headers=["x-internal-debug"],
        )


async def test_forward_headers_rejects_empty_extra_blocked_header_names() -> None:
    with pytest.raises(ValidationError, match="empty header name"):
        PassthroughSafetyConfig(
            base_url="https://safety.example.com/v1",
            extra_blocked_headers=["   "],
        )
