# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from contextlib import AbstractContextManager
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from ogx.core.request_headers import request_provider_data_context
from ogx.providers.remote.safety.passthrough.config import PassthroughSafetyConfig
from ogx.providers.remote.safety.passthrough.passthrough import PassthroughSafetyAdapter


class FakePassthroughSafetyAdapter(PassthroughSafetyAdapter):
    """Test subclass that injects a mock shield_store."""

    def __init__(self, config: PassthroughSafetyConfig, shield_store: AsyncMock) -> None:
        super().__init__(config)
        self.shield_store = shield_store


def _stub_provider_spec(adapter: PassthroughSafetyAdapter) -> None:
    adapter.__provider_spec__ = MagicMock()
    adapter.__provider_spec__.provider_data_validator = (
        "ogx.providers.remote.safety.passthrough.config.PassthroughProviderDataValidator"
    )


def mock_httpx_response(json_data: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status.return_value = None
    return resp


@pytest.fixture
def shield_store() -> AsyncMock:
    store = AsyncMock()
    store.get_shield = AsyncMock()
    return store


@pytest.fixture
def config() -> PassthroughSafetyConfig:
    return PassthroughSafetyConfig(base_url="https://safety.example.com/v1")


@pytest.fixture
def config_with_api_key() -> PassthroughSafetyConfig:
    return PassthroughSafetyConfig(
        base_url="https://safety.example.com/v1",
        api_key="config-key-123",
    )


@pytest.fixture
def config_with_forward_headers() -> PassthroughSafetyConfig:
    return PassthroughSafetyConfig(
        base_url="https://safety.example.com/v1",
        forward_headers={"maas_api_token": "Authorization", "tenant_id": "X-Tenant-Id"},
    )


@pytest.fixture
def adapter(config: PassthroughSafetyConfig, shield_store: AsyncMock) -> FakePassthroughSafetyAdapter:
    a = FakePassthroughSafetyAdapter(config, shield_store)
    _stub_provider_spec(a)
    a._client = AsyncMock(spec=httpx.AsyncClient)
    return a


@pytest.fixture
def adapter_with_api_key(
    config_with_api_key: PassthroughSafetyConfig, shield_store: AsyncMock
) -> FakePassthroughSafetyAdapter:
    a = FakePassthroughSafetyAdapter(config_with_api_key, shield_store)
    _stub_provider_spec(a)
    a._client = AsyncMock(spec=httpx.AsyncClient)
    return a


@pytest.fixture
def adapter_with_forward_headers(
    config_with_forward_headers: PassthroughSafetyConfig, shield_store: AsyncMock
) -> FakePassthroughSafetyAdapter:
    a = FakePassthroughSafetyAdapter(config_with_forward_headers, shield_store)
    _stub_provider_spec(a)
    a._client = AsyncMock(spec=httpx.AsyncClient)
    return a


def provider_data_ctx(data: dict) -> AbstractContextManager:
    return request_provider_data_context({"x-ogx-provider-data": json.dumps(data)})
