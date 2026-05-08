# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import logging  # allow-direct-logging
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from ogx.core.datatypes import (
    AuthProviderType,
    OAuth2IntrospectionConfig,
    OAuth2TokenAuthConfig,
)
from ogx.core.server.auth_providers import OAuth2TokenAuthProvider


class MockResponse:
    def __init__(self, status_code, json_data):
        self.status_code = status_code
        self._json_data = json_data

    def json(self):
        return self._json_data


@pytest.fixture
def mock_introspection_endpoint():
    return "http://mock-authz-service/token/introspect"


@pytest.fixture
def introspection_provider(mock_introspection_endpoint):
    return OAuth2TokenAuthProvider(
        OAuth2TokenAuthConfig(
            type=AuthProviderType.OAUTH2_TOKEN,
            introspection=OAuth2IntrospectionConfig(
                url=mock_introspection_endpoint,
                client_id="myclient",
                client_secret="abcdefg",
            ),
        )
    )


def _mock_async_client(mock_client_cls):
    """Wire up an AsyncClient mock that supports async context manager usage."""
    mock_client = AsyncMock()
    mock_client.post.return_value = MockResponse(200, {"active": True, "sub": "user1", "username": "user1"})
    mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
    return mock_client


async def test_introspection_uses_verify_true_by_default(introspection_provider):
    """Introspection uses system CA bundle (verify=True) when verify_tls is True and no tls_cafile."""
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = _mock_async_client(mock_client_cls)

        await introspection_provider.introspect_token("some-token")

        mock_client_cls.assert_called_once_with(verify=True, timeout=httpx.Timeout(10.0, connect=5.0))
        assert "timeout" not in mock_client.post.call_args.kwargs


async def test_introspection_uses_custom_ca_file(mock_introspection_endpoint, tmp_path):
    """Introspection uses custom CA file when tls_cafile is configured."""
    ca_file = tmp_path / "ca.pem"
    ca_file.write_text("fake-ca-cert")

    provider = OAuth2TokenAuthProvider(
        OAuth2TokenAuthConfig(
            type=AuthProviderType.OAUTH2_TOKEN,
            introspection=OAuth2IntrospectionConfig(
                url=mock_introspection_endpoint,
                client_id="myclient",
                client_secret="abcdefg",
            ),
            tls_cafile=ca_file,
        )
    )

    with patch("httpx.AsyncClient") as mock_client_cls, patch("ssl.create_default_context") as mock_ssl_ctx:
        mock_ctx = Mock()
        mock_ssl_ctx.return_value = mock_ctx
        _mock_async_client(mock_client_cls)

        await provider.introspect_token("some-token")

        mock_ssl_ctx.assert_called_once_with(cafile=ca_file.as_posix())
        mock_client_cls.assert_called_once_with(verify=mock_ctx, timeout=httpx.Timeout(10.0, connect=5.0))


async def test_introspection_disables_verification_when_verify_tls_false(mock_introspection_endpoint, caplog):
    """Introspection uses verify=False and logs WARNING when verify_tls is False."""
    provider = OAuth2TokenAuthProvider(
        OAuth2TokenAuthConfig(
            type=AuthProviderType.OAUTH2_TOKEN,
            introspection=OAuth2IntrospectionConfig(
                url=mock_introspection_endpoint,
                client_id="myclient",
                client_secret="abcdefg",
            ),
            verify_tls=False,
        )
    )

    with patch("httpx.AsyncClient") as mock_client_cls:
        _mock_async_client(mock_client_cls)

        with caplog.at_level(logging.WARNING):
            await provider.introspect_token("some-token")

        mock_client_cls.assert_called_once_with(verify=False, timeout=httpx.Timeout(10.0, connect=5.0))
        assert any("TLS verification is disabled" in r.message for r in caplog.records)
