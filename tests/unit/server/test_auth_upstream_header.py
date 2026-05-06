# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import logging  # allow-direct-logging

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from ogx.core.datatypes import (
    AuthenticationConfig,
    AuthProviderType,
    UpstreamHeaderAuthConfig,
)
from ogx.core.server.auth import AuthenticationMiddleware
from ogx.core.server.auth_providers import UpstreamHeaderAuthProvider


@pytest.fixture
def suppress_auth_errors(caplog):
    """Suppress expected ERROR/WARNING logs for tests that deliberately trigger authentication errors"""
    caplog.set_level(logging.CRITICAL, logger="ogx.core.server.auth")
    caplog.set_level(logging.CRITICAL, logger="ogx.core.server.auth_providers")


@pytest.fixture
def upstream_header_app():
    app = FastAPI()
    auth_config = AuthenticationConfig(
        provider_config=UpstreamHeaderAuthConfig(
            type=AuthProviderType.UPSTREAM_HEADER,
            principal_header="x-auth-user-id",
            attributes_header="x-auth-attributes",
        ),
        access_policy=[],
    )
    app.add_middleware(AuthenticationMiddleware, auth_config=auth_config, impls={})

    @app.get("/test")
    def test_endpoint():
        return {"message": "Authentication successful"}

    return app


@pytest.fixture
def upstream_header_client(upstream_header_app):
    return TestClient(upstream_header_app)


@pytest.fixture
def upstream_header_app_no_attributes():
    app = FastAPI()
    auth_config = AuthenticationConfig(
        provider_config=UpstreamHeaderAuthConfig(
            type=AuthProviderType.UPSTREAM_HEADER,
            principal_header="x-auth-user-id",
        ),
        access_policy=[],
    )
    app.add_middleware(AuthenticationMiddleware, auth_config=auth_config, impls={})

    @app.get("/test")
    def test_endpoint():
        return {"message": "Authentication successful"}

    return app


@pytest.fixture
def upstream_header_client_no_attributes(upstream_header_app_no_attributes):
    return TestClient(upstream_header_app_no_attributes)


def test_valid_upstream_header_auth(upstream_header_client):
    """Test successful authentication with principal and attributes headers."""
    attributes = json.dumps({"roles": ["admin", "user"], "teams": ["ml-team"]})
    response = upstream_header_client.get(
        "/test",
        headers={
            "x-auth-user-id": "alice",
            "x-auth-attributes": attributes,
        },
    )
    assert response.status_code == 200
    assert response.json() == {"message": "Authentication successful"}


def test_valid_upstream_header_auth_principal_only(upstream_header_client):
    """Test successful authentication with only the principal header (no attributes)."""
    response = upstream_header_client.get(
        "/test",
        headers={"x-auth-user-id": "alice"},
    )
    assert response.status_code == 200
    assert response.json() == {"message": "Authentication successful"}


def test_missing_principal_header(upstream_header_client, suppress_auth_errors):
    """Test that missing principal header returns 401."""
    response = upstream_header_client.get("/test")
    assert response.status_code == 401
    assert "Missing required authentication header" in response.json()["error"]["message"]
    assert "x-auth-user-id" in response.json()["error"]["message"]


def test_invalid_attributes_json(upstream_header_client, suppress_auth_errors):
    """Test that invalid JSON in attributes header returns 401."""
    response = upstream_header_client.get(
        "/test",
        headers={
            "x-auth-user-id": "alice",
            "x-auth-attributes": "not-valid-json{",
        },
    )
    assert response.status_code == 401
    assert "Failed to parse authentication attributes header" in response.json()["error"]["message"]


def test_attributes_not_object(upstream_header_client, suppress_auth_errors):
    """Test that non-object JSON in attributes header returns 401."""
    response = upstream_header_client.get(
        "/test",
        headers={
            "x-auth-user-id": "alice",
            "x-auth-attributes": '["not", "an", "object"]',
        },
    )
    assert response.status_code == 401
    assert "expected JSON object" in response.json()["error"]["message"]


def test_no_bearer_token_required(upstream_header_client):
    """Test that upstream header auth does NOT require a Bearer token."""
    response = upstream_header_client.get(
        "/test",
        headers={"x-auth-user-id": "alice"},
    )
    assert response.status_code == 200


def test_bearer_token_ignored(upstream_header_client):
    """Test that a Bearer token is ignored when upstream header auth is configured."""
    response = upstream_header_client.get(
        "/test",
        headers={
            "Authorization": "Bearer some-token",
            "x-auth-user-id": "alice",
        },
    )
    assert response.status_code == 200


def test_no_attributes_header_configured(upstream_header_client_no_attributes):
    """Test that when attributes_header is not configured, auth succeeds without it."""
    response = upstream_header_client_no_attributes.get(
        "/test",
        headers={"x-auth-user-id": "bob"},
    )
    assert response.status_code == 200


def test_case_insensitive_headers(upstream_header_client):
    """Test that header matching is case-insensitive (HTTP standard)."""
    attributes = json.dumps({"roles": ["viewer"]})
    response = upstream_header_client.get(
        "/test",
        headers={
            "X-Auth-User-Id": "alice",
            "X-Auth-Attributes": attributes,
        },
    )
    assert response.status_code == 200


def test_attributes_string_values_normalized(upstream_header_client):
    """Test that string attribute values are normalized to lists."""
    attributes = json.dumps({"roles": "admin", "teams": ["ml-team", "infra"]})
    response = upstream_header_client.get(
        "/test",
        headers={
            "x-auth-user-id": "alice",
            "x-auth-attributes": attributes,
        },
    )
    assert response.status_code == 200


def test_error_message_includes_header_name(upstream_header_client, suppress_auth_errors):
    """Test that the error message for missing principal includes the header name."""
    response = upstream_header_client.get("/test")
    error_msg = response.json()["error"]["message"]
    assert "x-auth-user-id" in error_msg


def test_authenticated_client_id_uses_principal(upstream_header_app):
    """Test that middleware sets authenticated_client_id to principal when no Bearer token is used."""

    @upstream_header_app.get("/scope")
    def scope_endpoint(request: Request):
        return {"authenticated_client_id": request.scope.get("authenticated_client_id")}

    client = TestClient(upstream_header_app)
    response = client.get("/scope", headers={"x-auth-user-id": "alice"})
    assert response.status_code == 200
    assert response.json()["authenticated_client_id"] == "alice"


# Provider unit tests (without middleware)


async def test_provider_requires_http_bearer_false():
    """Test that UpstreamHeaderAuthProvider.requires_http_bearer returns False."""
    config = UpstreamHeaderAuthConfig(
        principal_header="x-auth-user-id",
    )
    provider = UpstreamHeaderAuthProvider(config)
    assert provider.requires_http_bearer is False


async def test_provider_validate_token_extracts_principal():
    """Test that validate_token extracts principal from headers in scope."""
    config = UpstreamHeaderAuthConfig(
        principal_header="x-auth-user-id",
    )
    provider = UpstreamHeaderAuthProvider(config)
    scope = {
        "headers": [
            (b"x-auth-user-id", b"alice"),
        ],
    }
    user = await provider.validate_token("", scope)
    assert user.principal == "alice"
    assert user.attributes is None


async def test_provider_validate_token_extracts_attributes():
    """Test that validate_token extracts and parses attributes from headers."""
    config = UpstreamHeaderAuthConfig(
        principal_header="x-auth-user-id",
        attributes_header="x-auth-attributes",
    )
    provider = UpstreamHeaderAuthProvider(config)
    attributes = json.dumps({"roles": ["admin", "user"], "teams": ["ml-team"]})
    scope = {
        "headers": [
            (b"x-auth-user-id", b"alice"),
            (b"x-auth-attributes", attributes.encode()),
        ],
    }
    user = await provider.validate_token("", scope)
    assert user.principal == "alice"
    assert user.attributes == {"roles": ["admin", "user"], "teams": ["ml-team"]}


async def test_provider_validate_token_missing_principal():
    """Test that validate_token raises ValueError when principal header is missing."""
    config = UpstreamHeaderAuthConfig(
        principal_header="x-auth-user-id",
    )
    provider = UpstreamHeaderAuthProvider(config)
    scope = {
        "headers": [
            (b"other-header", b"value"),
        ],
    }
    with pytest.raises(ValueError, match="Missing required authentication header"):
        await provider.validate_token("", scope)


async def test_provider_validate_token_none_scope():
    """Test that validate_token raises ValueError when scope is None."""
    config = UpstreamHeaderAuthConfig(
        principal_header="x-auth-user-id",
    )
    provider = UpstreamHeaderAuthProvider(config)
    with pytest.raises(ValueError, match="Missing required authentication header"):
        await provider.validate_token("", None)
