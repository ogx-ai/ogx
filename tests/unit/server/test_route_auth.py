# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import Mock

import pytest
from fastapi import FastAPI

from ogx.core.access_control.datatypes import RouteAccessRule, RouteScope
from ogx.core.datatypes import (
    AuthenticationConfig,
    AuthProviderType,
    CustomAuthConfig,
    User,
)
from ogx.core.server.auth import (
    AuthenticationMiddleware,
    RouteAuthorizationMiddleware,
)


@pytest.fixture
def admin_user():
    return User(
        principal="admin@example.com",
        attributes={
            "roles": ["admin"],
            "teams": ["platform"],
        },
    )


@pytest.fixture
def developer_user():
    return User(
        principal="dev@example.com",
        attributes={
            "roles": ["developer"],
            "teams": ["ml-team"],
        },
    )


@pytest.fixture
def regular_user():
    return User(
        principal="user@example.com",
        attributes={
            "roles": ["user"],
            "teams": ["ml-team"],
        },
    )


def create_mock_auth_provider(user: User):
    """Create a mock auth provider that returns the specified user"""
    mock_provider = Mock()
    mock_provider.validate_token = Mock(return_value=user)
    return mock_provider


def create_app_with_route_policy(route_policy: list[RouteAccessRule], user: User):
    """Create a FastAPI app with route authorization middleware"""
    app = FastAPI()

    # Create auth config
    auth_config = AuthenticationConfig(
        provider_config=CustomAuthConfig(
            type=AuthProviderType.CUSTOM,
            endpoint="http://mock-auth/validate",
        ),
        route_policy=route_policy,
        access_policy=[],
    )

    # Add authentication middleware
    auth_middleware = AuthenticationMiddleware(app, auth_config, {})
    # Mock the auth provider to return our test user
    auth_middleware.auth_provider = create_mock_auth_provider(user)

    # Create the middleware stack
    app.add_middleware(RouteAuthorizationMiddleware, route_policy=route_policy)

    # Replace the app in the auth middleware
    async def app_with_auth(scope, receive, send):
        return await auth_middleware(scope, receive, send)

    # Add test endpoints
    @app.get("/v1/chat/completions")
    def chat_completions():
        return {"message": "chat completions"}

    @app.get("/v1/models/list")
    def models_list():
        return {"message": "models list"}

    @app.post("/v1/files/upload")
    def files_upload():
        return {"message": "file uploaded"}

    @app.delete("/v1/admin/reset")
    def admin_reset():
        return {"message": "admin reset"}

    return app, app_with_auth


async def test_no_route_policy_allows_all(regular_user):
    """Test backward compatibility: empty route_policy allows all routes"""
    app = FastAPI()

    # No route policy
    route_policy = []

    middleware = RouteAuthorizationMiddleware(app, route_policy)

    # Mock scope with any path
    scope = {
        "type": "http",
        "path": "/v1/chat/completions",
        "method": "GET",
        "principal": regular_user.principal,
        "user_attributes": regular_user.attributes,
    }

    # Track if next middleware was called
    called = False

    async def mock_app(scope, receive, send):
        nonlocal called
        called = True

    async def receive():
        return {}

    async def send(msg):
        pass

    middleware.app = mock_app
    await middleware(scope, receive, send)

    # Should pass through without blocking
    assert called


async def test_exact_path_match(developer_user):
    """Test exact path matching"""
    route_policy = [
        RouteAccessRule(
            permit=RouteScope(paths="/v1/chat/completions"),
            when="user with developer in roles",
        )
    ]

    middleware = RouteAuthorizationMiddleware(None, route_policy)

    # Should match
    assert middleware._route_matches("/v1/chat/completions", "/v1/chat/completions")

    # Should not match
    assert not middleware._route_matches("/v1/chat/completions/stream", "/v1/chat/completions")
    assert not middleware._route_matches("/v1/models/list", "/v1/chat/completions")


async def test_wildcard_prefix_match():
    """Test wildcard prefix matching"""
    route_policy = []
    middleware = RouteAuthorizationMiddleware(None, route_policy)

    # Test prefix wildcard
    assert middleware._route_matches("/v1/files/upload", "/v1/files*")
    assert middleware._route_matches("/v1/files/delete", "/v1/files*")
    assert middleware._route_matches("/v1/files/list/all", "/v1/files*")
    # Should also match the exact prefix
    assert middleware._route_matches("/v1/files", "/v1/files*")

    # Should not match different prefix
    assert not middleware._route_matches("/v1/models/list", "/v1/files*")


async def test_full_wildcard_match():
    """Test full wildcard matching"""
    route_policy = []
    middleware = RouteAuthorizationMiddleware(None, route_policy)

    # Full wildcard should match everything
    assert middleware._route_matches("/v1/chat/completions", "*")
    assert middleware._route_matches("/v1/files/upload", "*")
    assert middleware._route_matches("/admin/reset", "*")
    assert middleware._route_matches("/anything/goes", "*")


async def test_multiple_paths_in_rule(regular_user):
    """Test rule with multiple paths"""
    route_policy = [
        RouteAccessRule(
            permit=RouteScope(paths=["/v1/files*", "/v1/models*"]),
            when="user with user in roles",
        )
    ]

    middleware = RouteAuthorizationMiddleware(None, route_policy)

    # Test that the policy allows these paths for regular_user
    assert middleware._is_route_allowed("/v1/files/upload", regular_user)
    assert middleware._is_route_allowed("/v1/models/list", regular_user)

    # Should not match other paths
    assert not middleware._is_route_allowed("/v1/chat/completions", regular_user)


async def test_condition_evaluation_with_roles(developer_user, regular_user):
    """Test condition evaluation with role attributes"""
    route_policy = [
        RouteAccessRule(
            permit=RouteScope(paths="/v1/chat/completions"),
            when="user with developer in roles",
        )
    ]

    middleware = RouteAuthorizationMiddleware(None, route_policy)

    # Developer should pass
    assert middleware._is_route_allowed("/v1/chat/completions", developer_user)

    # Regular user should not pass
    assert not middleware._is_route_allowed("/v1/chat/completions", regular_user)


async def test_admin_full_wildcard_access(admin_user, developer_user):
    """Test admin with full wildcard access"""
    route_policy = [
        RouteAccessRule(
            permit=RouteScope(paths="/v1/chat/completions"),
            when="user with developer in roles",
        ),
        RouteAccessRule(
            permit=RouteScope(paths="*"),
            when="user with admin in roles",
        ),
    ]

    middleware = RouteAuthorizationMiddleware(None, route_policy)

    # Admin should access everything
    assert middleware._is_route_allowed("/v1/chat/completions", admin_user)
    assert middleware._is_route_allowed("/v1/files/upload", admin_user)
    assert middleware._is_route_allowed("/v1/admin/reset", admin_user)

    # Developer should only access chat completions
    assert middleware._is_route_allowed("/v1/chat/completions", developer_user)
    assert not middleware._is_route_allowed("/v1/files/upload", developer_user)


async def test_forbid_rule(admin_user, developer_user):
    """Test forbid rules with unless conditions"""
    route_policy = [
        RouteAccessRule(
            forbid=RouteScope(paths="/v1/admin*"),
            unless="user with admin in roles",
        ),
        RouteAccessRule(
            permit=RouteScope(paths="*"),
            when="user with admin in roles",
            description="Admins can access everything",
        ),
        RouteAccessRule(
            permit=RouteScope(paths="*"),
            when="user with developer in roles",
            description="Developers can access everything except what's forbidden",
        ),
    ]

    middleware = RouteAuthorizationMiddleware(None, route_policy)

    # Developer is forbidden from admin routes by the first forbid rule
    assert not middleware._is_route_allowed("/v1/admin/reset", developer_user)

    # Admin bypasses the forbid rule (due to 'unless' condition) and matches the permit rule
    assert middleware._is_route_allowed("/v1/admin/reset", admin_user)

    # Developer can access non-admin routes
    assert middleware._is_route_allowed("/v1/chat/completions", developer_user)


async def test_no_matching_rule_denies_access(regular_user):
    """Test that no matching rule results in denied access"""
    route_policy = [
        RouteAccessRule(
            permit=RouteScope(paths="/v1/chat/completions"),
            when="user with developer in roles",
        )
    ]

    middleware = RouteAuthorizationMiddleware(None, route_policy)

    # Regular user should be denied (doesn't have developer role)
    assert not middleware._is_route_allowed("/v1/chat/completions", regular_user)

    # Any user should be denied for non-matching path
    assert not middleware._is_route_allowed("/v1/models/list", regular_user)


async def test_multiple_conditions(admin_user):
    """Test multiple conditions in a rule"""
    route_policy = [
        RouteAccessRule(
            permit=RouteScope(paths="*"),
            when=["user with admin in roles", "user with platform in teams"],
        )
    ]

    middleware = RouteAuthorizationMiddleware(None, route_policy)

    # Admin with platform team should pass
    assert middleware._is_route_allowed("/v1/anything", admin_user)


async def test_rule_order_matters(developer_user):
    """Test that rules are evaluated in order"""
    route_policy = [
        # First rule: developers can access chat
        RouteAccessRule(
            permit=RouteScope(paths="/v1/chat/completions"),
            when="user with developer in roles",
        ),
        # Second rule: deny all developers from everything (should not apply to chat)
        RouteAccessRule(
            forbid=RouteScope(paths="*"),
            when="user with developer in roles",
        ),
    ]

    middleware = RouteAuthorizationMiddleware(None, route_policy)

    # First matching rule should win
    assert middleware._is_route_allowed("/v1/chat/completions", developer_user)

    # Second rule should match for other paths
    assert not middleware._is_route_allowed("/v1/models/list", developer_user)


async def test_websocket_passthrough():
    """Test that websocket requests pass through without blocking"""
    route_policy = [
        RouteAccessRule(
            permit=RouteScope(paths="/v1/chat/completions"),
            when="user with developer in roles",
        )
    ]

    app = FastAPI()
    middleware = RouteAuthorizationMiddleware(app, route_policy)

    # Mock websocket scope
    scope = {
        "type": "websocket",
        "path": "/ws",
    }

    # Track if next middleware was called
    called = False

    async def mock_app(scope, receive, send):
        nonlocal called
        called = True

    async def receive():
        return {}

    async def send(msg):
        pass

    middleware.app = mock_app
    await middleware(scope, receive, send)

    # Websocket requests should pass through
    assert called


async def test_route_blocking_without_auth():
    """Test that route policy can block routes without authentication configured"""
    route_policy = [
        RouteAccessRule(
            permit=RouteScope(paths="/v1/health"),
            description="Allow health check route",
        ),
        RouteAccessRule(
            permit=RouteScope(paths="/v1/models*"),
            description="Allow model routes",
        ),
        # All other routes denied by default (no matching rule)
    ]

    app = FastAPI()
    middleware = RouteAuthorizationMiddleware(app, route_policy)

    # No user (no authentication)
    user = None

    # Should allow health check
    assert middleware._is_route_allowed("/v1/health", user)

    # Should allow model routes
    assert middleware._is_route_allowed("/v1/models/list", user)

    # Should deny other routes (no matching rule)
    assert not middleware._is_route_allowed("/v1/chat/completions", user)
    assert not middleware._is_route_allowed("/v1/admin/reset", user)


async def test_forbid_rule_without_auth():
    """Test forbid rules work without authentication"""
    route_policy = [
        RouteAccessRule(
            forbid=RouteScope(paths="/v1/admin*"),
            description="Block admin routes",
        ),
        RouteAccessRule(
            permit=RouteScope(paths="*"),
            description="Allow all other routes",
        ),
    ]

    middleware = RouteAuthorizationMiddleware(None, route_policy)

    # No user (no authentication)
    user = None

    # Should forbid admin routes
    assert not middleware._is_route_allowed("/v1/admin/reset", user)
    assert not middleware._is_route_allowed("/v1/admin/users", user)

    # Should allow other routes
    assert middleware._is_route_allowed("/v1/chat/completions", user)
    assert middleware._is_route_allowed("/v1/models/list", user)


async def test_rule_with_condition_requires_user():
    """Test that rules with user conditions require authentication"""
    route_policy = [
        RouteAccessRule(
            permit=RouteScope(paths="*"),
            when="user with admin in roles",
        )
    ]

    middleware = RouteAuthorizationMiddleware(None, route_policy)

    # No user (no authentication)
    user = None

    # Should be denied because rule has condition but no user available
    assert not middleware._is_route_allowed("/v1/chat/completions", user)


async def test_mixed_rules_with_and_without_conditions(admin_user, regular_user):
    """Test mixing rules with and without user conditions"""
    route_policy = [
        # Public routes (no condition)
        RouteAccessRule(
            permit=RouteScope(paths=["/v1/health", "/v1/version"]),
            description="Public routes",
        ),
        # Admin-only routes (requires user)
        RouteAccessRule(
            permit=RouteScope(paths="/v1/admin*"),
            when="user with admin in roles",
            description="Admin routes require admin role",
        ),
        # Default: deny everything else
    ]

    middleware = RouteAuthorizationMiddleware(None, route_policy)

    # No user can access public routes
    assert middleware._is_route_allowed("/v1/health", None)
    assert middleware._is_route_allowed("/v1/version", None)

    # No user cannot access admin routes (condition requires user)
    assert not middleware._is_route_allowed("/v1/admin/reset", None)

    # Admin can access admin routes
    assert middleware._is_route_allowed("/v1/admin/reset", admin_user)

    # Regular user cannot access admin routes (lacks admin role)
    assert not middleware._is_route_allowed("/v1/admin/reset", regular_user)

    # No one can access other routes (no matching rule)
    assert not middleware._is_route_allowed("/v1/chat/completions", None)
    assert not middleware._is_route_allowed("/v1/chat/completions", admin_user)
    assert not middleware._is_route_allowed("/v1/chat/completions", regular_user)


async def test_regex_path_matching():
    """Test regex pattern matching for routes"""
    route_policy = []
    middleware = RouteAuthorizationMiddleware(None, route_policy)

    # Test exact match (backward compatibility)
    assert middleware._route_matches("/v1/chat/completions", "/v1/chat/completions")

    # Test wildcard (backward compatibility)
    assert middleware._route_matches("/v1/files/upload", "/v1/files*")
    assert middleware._route_matches("/anything", "*")

    # Test regex patterns
    assert middleware._route_matches("/v1/chat/completions", "regex:/v1/(chat|inference)/.*")
    assert middleware._route_matches("/v1/inference/run", "regex:/v1/(chat|inference)/.*")
    assert middleware._route_matches("/v1/models/list", "regex:/v1/models/.*")
    assert middleware._route_matches("/v1/files/123", r"regex:/v1/files/\d+")

    # Test that regex doesn't match when it shouldn't
    assert not middleware._route_matches("/v1/other/endpoint", "regex:/v1/(chat|inference)/.*")
    assert not middleware._route_matches("/v2/chat/completions", "regex:/v1/chat/.*")

    # Test list of patterns including regex
    assert middleware._route_matches("/v1/chat/completions", ["/v1/models*", "regex:/v1/(chat|inference)/.*"])


async def test_route_policy_with_regex(developer_user, regular_user):
    """Test route authorization with regex patterns"""
    route_policy = [
        RouteAccessRule(
            permit=RouteScope(paths="regex:/v1/(chat|inference)/.*"),
            when="user with developer in roles",
            description="Developers can access chat and inference endpoints",
        ),
        RouteAccessRule(
            permit=RouteScope(paths=r"regex:/v1/files/\d+"),
            when="user with user in roles",
            description="Regular users can access specific file IDs",
        ),
    ]

    middleware = RouteAuthorizationMiddleware(None, route_policy)

    # Developer can access chat and inference endpoints
    assert middleware._is_route_allowed("/v1/chat/completions", developer_user)
    assert middleware._is_route_allowed("/v1/inference/run", developer_user)

    # Developer cannot access file endpoints (not in their policy)
    assert not middleware._is_route_allowed("/v1/files/123", developer_user)

    # Regular user can access numbered file IDs
    assert middleware._is_route_allowed("/v1/files/123", regular_user)
    assert middleware._is_route_allowed("/v1/files/456", regular_user)

    # Regular user cannot access non-numeric file paths
    assert not middleware._is_route_allowed("/v1/files/upload", regular_user)

    # Regular user cannot access chat endpoints
    assert not middleware._is_route_allowed("/v1/chat/completions", regular_user)


async def test_invalid_regex_pattern_logs_warning(caplog):
    """Test that invalid regex patterns log a warning and don't crash"""
    import logging  # allow-direct-logging

    route_policy = []
    middleware = RouteAuthorizationMiddleware(None, route_policy)

    # Test various invalid regex patterns
    invalid_patterns = [
        "regex:**",  # Invalid: ** is not valid regex
        "regex:(?P<invalid",  # Invalid: unclosed group
        "regex:[z-a]",  # Invalid: bad character range
    ]

    with caplog.at_level(logging.WARNING):
        for pattern in invalid_patterns:
            # Should not crash, should return False (no match)
            result = middleware._route_matches("/v1/chat/completions", pattern)
            assert result is False

    # Check that warnings were logged
    assert "Invalid regex pattern in route_policy" in caplog.text
