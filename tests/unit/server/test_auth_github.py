# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import logging  # allow-direct-logging
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ogx.core.datatypes import AuthenticationConfig, AuthProviderType, GitHubTokenAuthConfig
from ogx.core.server.auth import AuthenticationMiddleware


@pytest.fixture
def suppress_auth_errors(caplog):
    """Suppress expected ERROR logs for tests that deliberately trigger authentication errors"""
    caplog.set_level(logging.CRITICAL, logger="ogx.core.server.auth")
    caplog.set_level(logging.CRITICAL, logger="ogx.core.server.auth_providers")


class MockResponse:
    def __init__(self, status_code, json_data):
        self.status_code = status_code
        self._json_data = json_data

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code != 200:
            # Create a mock request for the HTTPStatusError
            mock_request = httpx.Request("GET", "https://api.github.com/user")
            raise httpx.HTTPStatusError(f"HTTP error: {self.status_code}", request=mock_request, response=self)


@pytest.fixture
def github_token_app():
    app = FastAPI()

    # Configure GitHub token auth
    auth_config = AuthenticationConfig(
        provider_config=GitHubTokenAuthConfig(
            type=AuthProviderType.GITHUB_TOKEN,
            github_api_base_url="https://api.github.com",
            claims_mapping={
                "login": "username",
                "id": "user_id",
                "organizations": "teams",
            },
        ),
        access_policy=[],
    )

    # Add auth middleware
    app.add_middleware(AuthenticationMiddleware, auth_config=auth_config, impls={})

    @app.get("/test")
    def test_endpoint():
        return {"message": "Authentication successful"}

    return app


@pytest.fixture
def github_token_client(github_token_app):
    return TestClient(github_token_app)


def test_authenticated_endpoint_without_token(github_token_client):
    """Test accessing protected endpoint without token"""
    response = github_token_client.get("/test")
    assert response.status_code == 401
    assert "Authentication required" in response.json()["error"]["message"]
    assert "GitHub access token" in response.json()["error"]["message"]


def test_authenticated_endpoint_with_invalid_bearer_format(github_token_client):
    """Test accessing protected endpoint with invalid bearer format"""
    response = github_token_client.get("/test", headers={"Authorization": "InvalidFormat token123"})
    assert response.status_code == 401
    assert "Invalid Authorization header format" in response.json()["error"]["message"]


@patch("ogx.core.server.auth_providers.httpx.AsyncClient")
def test_authenticated_endpoint_with_valid_github_token(mock_client_class, github_token_client):
    """Test accessing protected endpoint with valid GitHub token"""
    # Mock the GitHub API responses
    mock_client = AsyncMock()
    mock_client_class.return_value.__aenter__.return_value = mock_client

    # Mock successful user API response
    mock_client.get.side_effect = [
        MockResponse(
            200,
            {
                "login": "testuser",
                "id": 12345,
                "email": "test@example.com",
                "name": "Test User",
            },
        ),
        MockResponse(
            200,
            [
                {"login": "test-org-1"},
                {"login": "test-org-2"},
            ],
        ),
    ]

    response = github_token_client.get("/test", headers={"Authorization": "Bearer github_token_123"})
    assert response.status_code == 200
    assert response.json()["message"] == "Authentication successful"

    # Verify both /user and /user/orgs endpoints were called
    assert mock_client.get.call_count == 2
    calls = mock_client.get.call_args_list
    assert calls[0][0][0] == "https://api.github.com/user"
    assert calls[1][0][0] == "https://api.github.com/user/orgs"
    assert calls[1][1]["params"] == {"per_page": 100, "page": 1}

    # Check authorization header was passed
    assert calls[0][1]["headers"]["Authorization"] == "Bearer github_token_123"


@patch("ogx.core.server.auth_providers.httpx.AsyncClient")
def test_authenticated_endpoint_with_invalid_github_token(mock_client_class, github_token_client, suppress_auth_errors):
    """Test accessing protected endpoint with invalid GitHub token"""
    # Mock the GitHub API to return 401 Unauthorized
    mock_client = AsyncMock()
    mock_client_class.return_value.__aenter__.return_value = mock_client

    # Mock failed user API response
    mock_client.get.return_value = MockResponse(401, {"message": "Bad credentials"})

    response = github_token_client.get("/test", headers={"Authorization": "Bearer invalid_token"})
    assert response.status_code == 401
    assert (
        "GitHub token validation failed. Please check your token and try again." in response.json()["error"]["message"]
    )


@patch("ogx.core.server.auth_providers.httpx.AsyncClient")
def test_github_enterprise_support(mock_client_class):
    """Test GitHub Enterprise support with custom API base URL"""
    app = FastAPI()

    # Configure GitHub token auth with enterprise URL
    auth_config = AuthenticationConfig(
        provider_config=GitHubTokenAuthConfig(
            type=AuthProviderType.GITHUB_TOKEN,
            github_api_base_url="https://github.enterprise.com/api/v3",
        ),
        access_policy=[],
    )

    app.add_middleware(AuthenticationMiddleware, auth_config=auth_config, impls={})

    @app.get("/test")
    def test_endpoint():
        return {"message": "Authentication successful"}

    client = TestClient(app)

    # Mock the GitHub Enterprise API responses
    mock_client = AsyncMock()
    mock_client_class.return_value.__aenter__.return_value = mock_client

    # Mock successful user API response
    mock_client.get.side_effect = [
        MockResponse(
            200,
            {
                "login": "enterprise_user",
                "id": 99999,
                "email": "user@enterprise.com",
            },
        ),
        MockResponse(
            200,
            [
                {"login": "enterprise-org"},
            ],
        ),
    ]

    response = client.get("/test", headers={"Authorization": "Bearer enterprise_token"})
    assert response.status_code == 200

    # Verify both /user and /user/orgs endpoints were called with enterprise URL
    assert mock_client.get.call_count == 2
    calls = mock_client.get.call_args_list
    assert calls[0][0][0] == "https://github.enterprise.com/api/v3/user"
    assert calls[1][0][0] == "https://github.enterprise.com/api/v3/user/orgs"
    assert calls[1][1]["params"] == {"per_page": 100, "page": 1}


@patch("ogx.core.server.auth_providers.httpx.AsyncClient")
def test_github_token_extracts_org_attributes(mock_client_class):
    """Test that organization memberships are extracted into user attributes"""
    from ogx.core.server.auth_providers import GitHubTokenAuthProvider

    config = GitHubTokenAuthConfig(
        type=AuthProviderType.GITHUB_TOKEN,
        github_api_base_url="https://api.github.com",
        claims_mapping={
            "login": "roles",
            "organizations": "teams",
        },
    )
    provider = GitHubTokenAuthProvider(config)

    mock_client = AsyncMock()
    mock_client_class.return_value.__aenter__.return_value = mock_client
    mock_client.get.side_effect = [
        MockResponse(200, {"login": "testuser", "id": 12345}),
        MockResponse(200, [{"login": "org-a"}, {"login": "org-b"}]),
    ]

    user = asyncio.run(provider.validate_token("token123"))
    assert user.principal == "testuser"
    assert user.attributes is not None
    assert "teams" in user.attributes
    assert "org-a" in user.attributes["teams"]
    assert "org-b" in user.attributes["teams"]


@patch("ogx.core.server.auth_providers.httpx.AsyncClient")
def test_github_token_handles_org_fetch_failure(mock_client_class, caplog):
    """Test that authentication succeeds even if fetching orgs fails"""
    from ogx.core.server.auth_providers import GitHubTokenAuthProvider

    config = GitHubTokenAuthConfig(
        type=AuthProviderType.GITHUB_TOKEN,
        github_api_base_url="https://api.github.com",
        claims_mapping={
            "login": "roles",
            "organizations": "teams",
        },
    )
    provider = GitHubTokenAuthProvider(config)

    mock_client = AsyncMock()
    mock_client_class.return_value.__aenter__.return_value = mock_client

    mock_request = httpx.Request("GET", "https://api.github.com/user/orgs")
    mock_client.get.side_effect = [
        MockResponse(200, {"login": "testuser", "id": 12345}),
        httpx.HTTPStatusError("403 Forbidden", request=mock_request, response=MockResponse(403, {})),
    ]

    user = asyncio.run(provider.validate_token("token123"))
    assert user.principal == "testuser"
    assert user.attributes is not None
    assert user.attributes.get("teams", []) == []


@patch("ogx.core.server.auth_providers.httpx.AsyncClient")
def test_github_token_multiple_orgs(mock_client_class):
    """Test with user who belongs to multiple organizations"""
    from ogx.core.server.auth_providers import GitHubTokenAuthProvider

    config = GitHubTokenAuthConfig(
        type=AuthProviderType.GITHUB_TOKEN,
        github_api_base_url="https://api.github.com",
        claims_mapping={
            "login": "roles",
            "organizations": "teams",
        },
    )
    provider = GitHubTokenAuthProvider(config)

    mock_client = AsyncMock()
    mock_client_class.return_value.__aenter__.return_value = mock_client
    mock_client.get.side_effect = [
        MockResponse(200, {"login": "multiorguser", "id": 99999}),
        MockResponse(
            200,
            [{"login": "org-1"}, {"login": "org-2"}, {"login": "org-3"}, {"login": "org-4"}],
        ),
    ]

    user = asyncio.run(provider.validate_token("token123"))
    assert user.principal == "multiorguser"
    assert user.attributes is not None
    assert "teams" in user.attributes
    assert len(user.attributes["teams"]) == 4
    assert set(user.attributes["teams"]) == {"org-1", "org-2", "org-3", "org-4"}


@patch("ogx.core.server.auth_providers.httpx.AsyncClient")
def test_github_token_fetches_paginated_orgs(mock_client_class):
    """Test that organization memberships are fetched across all GitHub org pages."""
    from ogx.core.server.auth_providers import GitHubTokenAuthProvider

    config = GitHubTokenAuthConfig(
        type=AuthProviderType.GITHUB_TOKEN,
        github_api_base_url="https://api.github.com",
        claims_mapping={
            "login": "roles",
            "organizations": "teams",
        },
    )
    provider = GitHubTokenAuthProvider(config)

    first_page_orgs = [{"login": f"org-{index}"} for index in range(1, 101)]
    second_page_orgs = [{"login": "org-101"}, {"login": "org-102"}]

    mock_client = AsyncMock()
    mock_client_class.return_value.__aenter__.return_value = mock_client
    mock_client.get.side_effect = [
        MockResponse(200, {"login": "multiorguser", "id": 99999}),
        MockResponse(200, first_page_orgs),
        MockResponse(200, second_page_orgs),
    ]

    user = asyncio.run(provider.validate_token("token123"))
    assert user.principal == "multiorguser"
    assert user.attributes is not None
    assert "teams" in user.attributes
    assert len(user.attributes["teams"]) == 102
    assert "org-101" in user.attributes["teams"]
    assert "org-102" in user.attributes["teams"]

    calls = mock_client.get.call_args_list
    assert calls[1][1]["params"] == {"per_page": 100, "page": 1}
    assert calls[2][1]["params"] == {"per_page": 100, "page": 2}


@patch("ogx.core.server.auth_providers.httpx.AsyncClient")
def test_github_token_preserves_partial_orgs_if_later_page_fails(mock_client_class):
    """Test that already-fetched organizations are preserved if a later org page fails."""
    from ogx.core.server.auth_providers import GitHubTokenAuthProvider

    config = GitHubTokenAuthConfig(
        type=AuthProviderType.GITHUB_TOKEN,
        github_api_base_url="https://api.github.com",
        claims_mapping={
            "login": "roles",
            "organizations": "teams",
        },
    )
    provider = GitHubTokenAuthProvider(config)

    first_page_orgs = [{"login": f"org-{index}"} for index in range(1, 101)]
    mock_request = httpx.Request("GET", "https://api.github.com/user/orgs")

    mock_client = AsyncMock()
    mock_client_class.return_value.__aenter__.return_value = mock_client
    mock_client.get.side_effect = [
        MockResponse(200, {"login": "multiorguser", "id": 99999}),
        MockResponse(200, first_page_orgs),
        httpx.HTTPStatusError("503 Service Unavailable", request=mock_request, response=MockResponse(503, {})),
    ]

    user = asyncio.run(provider.validate_token("token123"))
    assert user.principal == "multiorguser"
    assert user.attributes is not None
    assert "teams" in user.attributes
    assert len(user.attributes["teams"]) == 100
    assert "org-1" in user.attributes["teams"]
    assert "org-100" in user.attributes["teams"]


def test_github_token_auth_error_message_format(github_token_client):
    """Test that the error message for missing auth is properly formatted"""
    response = github_token_client.get("/test")
    assert response.status_code == 401

    error_data = response.json()
    assert "error" in error_data
    assert "message" in error_data["error"]
    assert "Authentication required" in error_data["error"]["message"]
    assert "https://docs.github.com" in error_data["error"]["message"]  # Contains link to GitHub docs
