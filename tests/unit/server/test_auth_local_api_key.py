# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ogx.core.datatypes import (
    AuthenticationConfig,
    LocalApiKeyAuthConfig,
    StackConfig,
    TenancyConfig,
)
from ogx.core.server.auth import AuthenticationMiddleware
from ogx.core.server.auth_providers import LocalApiKeyAuthProvider, TokenValidationError
from ogx.core.server.server import validate_auth_security

KEY1 = "ogk_abc123"
KEY2 = "ogk_def456"
INVALID_KEY = "ogk_invalid"

_CONFIG = LocalApiKeyAuthConfig(api_keys=[KEY1, KEY2])


async def test_valid_token_returns_user_with_attributes():
    provider = LocalApiKeyAuthProvider(_CONFIG)
    user = await provider.validate_token(KEY1)
    assert user.principal == KEY1
    assert user.attributes == {"roles": ["admin", "owner"], "teams": [KEY1]}


async def test_invalid_token_raises():
    provider = LocalApiKeyAuthProvider(_CONFIG)
    with pytest.raises(TokenValidationError, match="Invalid or missing API key"):
        await provider.validate_token(INVALID_KEY)


async def test_all_keys_work():
    provider = LocalApiKeyAuthProvider(_CONFIG)
    u1 = await provider.validate_token(KEY1)
    u2 = await provider.validate_token(KEY2)
    assert u1.attributes["roles"] == ["admin", "owner"]
    assert u2.attributes["roles"] == ["admin", "owner"]


async def test_attributes_are_immutable():
    provider = LocalApiKeyAuthProvider(_CONFIG)
    user = await provider.validate_token(KEY1)
    # mutating the returned dict should not affect future validations
    user.attributes.pop("roles")
    user2 = await provider.validate_token(KEY1)
    assert user2.attributes == {"roles": ["admin", "owner"], "teams": [KEY1]}


# --- Authentication middleware integration tests ---


@pytest.fixture
def local_api_key_app():
    app = FastAPI()

    auth_config = AuthenticationConfig(
        provider_config=LocalApiKeyAuthConfig(
            type="local_api_key",
            api_keys=["test-api-key-12345", "secondary-key-67890", "third-key-abcde"],
        ),
    )

    app.add_middleware(
        AuthenticationMiddleware,
        auth_config=auth_config,
    )

    @app.get("/test")
    def test_endpoint():
        return {"message": "Authentication successful"}

    return app


@pytest.fixture
def local_api_key_client(local_api_key_app):
    return TestClient(local_api_key_app)


def test_authenticated_endpoint_without_token(local_api_key_client):
    """Test accessing protected endpoint without token"""
    response = local_api_key_client.get("/test")
    assert response.status_code == 401
    assert "Authentication required" in response.json()["error"]["message"]


def test_authenticated_endpoint_with_invalid_bearer_format(local_api_key_client):
    """Test accessing protected endpoint with invalid bearer format"""
    response = local_api_key_client.get("/test", headers={"Authorization": "InvalidFormat token123"})
    assert response.status_code == 401
    assert "Invalid Authorization header format" in response.json()["error"]["message"]


def test_authenticated_endpoint_with_invalid_api_key(local_api_key_client):
    """Test accessing protected endpoint with wrong API key"""
    response = local_api_key_client.get("/test", headers={"Authorization": "Bearer wrong-key"})
    assert response.status_code == 401
    assert "Invalid or missing API key" in response.json()["error"]["message"]


def test_authenticated_endpoint_with_valid_api_key(local_api_key_client):
    """Test accessing protected endpoint with correct API key"""
    response = local_api_key_client.get(
        "/test",
        headers={"Authorization": "Bearer test-api-key-12345"},
    )
    assert response.status_code == 200
    assert response.json()["message"] == "Authentication successful"


def test_authenticated_endpoint_with_valid_api_key_secondary(local_api_key_client):
    """Test accessing protected endpoint with secondary API key"""
    response = local_api_key_client.get(
        "/test",
        headers={"Authorization": "Bearer secondary-key-67890"},
    )
    assert response.status_code == 200
    assert response.json()["message"] == "Authentication successful"


def test_authenticated_endpoint_empty_bearer_token(local_api_key_client):
    """Test accessing protected endpoint with empty bearer token"""
    response = local_api_key_client.get(
        "/test",
        headers={"Authorization": "Bearer "},
    )
    assert response.status_code == 401
    assert "Invalid or missing API key" in response.json()["error"]["message"]


# --- Startup validation ---


class TestLocalApiKeyTenancyValidation:
    def _make_config(self, tenancy_mode, default_tenant_id=None):
        return StackConfig(
            version=2,
            distro_name="test",
            providers={},
            server={
                "insecure": True,
                "auth": AuthenticationConfig(
                    provider_config=LocalApiKeyAuthConfig(
                        api_keys=["ogk_test123"],
                    ),
                ),
                "tenancy": TenancyConfig(mode=tenancy_mode, default_tenant_id=default_tenant_id),
            },
        )

    def test_multi_tenancy_errors(self):
        config = self._make_config("multi")
        with pytest.raises(SystemExit, match="local_api_key.*multi"):
            validate_auth_security(config)

    def test_single_tenancy_passes(self):
        config = self._make_config("single", default_tenant_id="acme-corp")
        validate_auth_security(config)

    def test_disabled_tenancy_passes(self):
        config = self._make_config("disabled")
        validate_auth_security(config)
