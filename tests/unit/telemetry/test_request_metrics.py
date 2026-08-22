# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from ogx.core.server.metrics import (
    RequestMetricsMiddleware,
    RouteInfo,
    _compile_route_patterns,
    build_route_to_api_map,
)


@pytest.fixture
def sample_route_to_api():
    return {
        "POST:/v1/chat/completions": RouteInfo("inference", "openai_chat_completion"),
        "GET:/v1/chat/completions": RouteInfo("inference", "list_chat_completions"),
        "GET:/v1/chat/completions/{completion_id}/messages": RouteInfo("inference", "list_chat_completion_messages"),
        "POST:/v1/completions": RouteInfo("inference", "openai_completion"),
        "POST:/v1/embeddings": RouteInfo("inference", "openai_embeddings"),
        "GET:/v1/models": RouteInfo("models", "openai_list_models"),
        "POST:/v1/models": RouteInfo("models", "register_model"),
        "GET:/v1/models/{model_id}": RouteInfo("models", "get_model"),
        "DELETE:/v1/models/{model_id}": RouteInfo("models", "unregister_model"),
        "GET:/v1/vector-stores": RouteInfo("vector_io", "list_vector_stores"),
        "GET:/v1/agents": RouteInfo("agents", "list_agents"),
        "POST:/v1/agents/{agent_id}/sessions/{session_id}/turns": RouteInfo("agents", "create_agent_turn"),
    }


class TestResolveRoute:
    def test_same_path_different_methods(self, sample_route_to_api):
        """GET /v1/models and POST /v1/models should resolve to different methods."""
        patterns = _compile_route_patterns(sample_route_to_api)
        middleware = RequestMetricsMiddleware.__new__(RequestMetricsMiddleware)
        middleware._patterns = patterns

        route = middleware._resolve_route("GET", "/v1/models")
        assert route.api == "models"
        assert route.method == "openai_list_models"

        route = middleware._resolve_route("POST", "/v1/models")
        assert route.api == "models"
        assert route.method == "register_model"

    def test_same_path_different_methods_chat(self, sample_route_to_api):
        patterns = _compile_route_patterns(sample_route_to_api)
        middleware = RequestMetricsMiddleware.__new__(RequestMetricsMiddleware)
        middleware._patterns = patterns

        route = middleware._resolve_route("POST", "/v1/chat/completions")
        assert route.method == "openai_chat_completion"

        route = middleware._resolve_route("GET", "/v1/chat/completions")
        assert route.method == "list_chat_completions"

    def test_delete_vs_get(self, sample_route_to_api):
        patterns = _compile_route_patterns(sample_route_to_api)
        middleware = RequestMetricsMiddleware.__new__(RequestMetricsMiddleware)
        middleware._patterns = patterns

        route = middleware._resolve_route("GET", "/v1/models/llama3")
        assert route.method == "get_model"

        route = middleware._resolve_route("DELETE", "/v1/models/llama3")
        assert route.method == "unregister_model"

    def test_exact_path(self, sample_route_to_api):
        patterns = _compile_route_patterns(sample_route_to_api)
        middleware = RequestMetricsMiddleware.__new__(RequestMetricsMiddleware)
        middleware._patterns = patterns

        route = middleware._resolve_route("POST", "/v1/embeddings")
        assert route.api == "inference"
        assert route.method == "openai_embeddings"

    def test_parameterized_path(self, sample_route_to_api):
        patterns = _compile_route_patterns(sample_route_to_api)
        middleware = RequestMetricsMiddleware.__new__(RequestMetricsMiddleware)
        middleware._patterns = patterns

        route = middleware._resolve_route("GET", "/v1/models/my-model")
        assert route.api == "models"
        assert route.method == "get_model"

    def test_nested_parameterized_path(self, sample_route_to_api):
        patterns = _compile_route_patterns(sample_route_to_api)
        middleware = RequestMetricsMiddleware.__new__(RequestMetricsMiddleware)
        middleware._patterns = patterns

        route = middleware._resolve_route("POST", "/v1/agents/agent-123/sessions/sess-456/turns")
        assert route.api == "agents"
        assert route.method == "create_agent_turn"

    def test_nested_parameterized_path_messages(self, sample_route_to_api):
        patterns = _compile_route_patterns(sample_route_to_api)
        middleware = RequestMetricsMiddleware.__new__(RequestMetricsMiddleware)
        middleware._patterns = patterns

        route = middleware._resolve_route("GET", "/v1/chat/completions/chatcmpl-123/messages")
        assert route.api == "inference"
        assert route.method == "list_chat_completion_messages"

    def test_unknown_path(self, sample_route_to_api):
        patterns = _compile_route_patterns(sample_route_to_api)
        middleware = RequestMetricsMiddleware.__new__(RequestMetricsMiddleware)
        middleware._patterns = patterns

        route = middleware._resolve_route("GET", "/v1/nonexistent")
        assert route.api == "unknown"
        assert route.method == "unknown"


def _make_fake_stack_app(route_to_api):
    """Create a mock Starlette app with a StackApp-compatible stack."""
    from ogx.core.server.metrics import _compile_route_patterns

    patterns = _compile_route_patterns(route_to_api)
    stack_mock = AsyncMock()
    stack_mock.impls = {"inference": {}, "models": {}}
    mock_stack = AsyncMock()
    mock_stack.stack = stack_mock

    async def inner_app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200})
        await send({"type": "http.response.body", "body": b"ok"})

    return inner_app, mock_stack, patterns


class TestRequestMetricsMiddleware:
    async def test_skips_non_http(self):
        mock_app = AsyncMock()
        middleware = RequestMetricsMiddleware(mock_app)
        scope = {"type": "lifespan"}
        receive = AsyncMock()
        send = AsyncMock()

        await middleware(scope, receive, send)
        mock_app.assert_called_once_with(scope, receive, send)

    async def test_skips_excluded_paths(self):
        mock_app = AsyncMock()
        middleware = RequestMetricsMiddleware(mock_app)
        mock_stack = AsyncMock()
        mock_stack.impls = {}
        mock_stack_mock = AsyncMock()
        mock_stack_mock.stack = mock_stack
        for path in ["/docs", "/redoc", "/openapi.json", "/favicon.ico", "/static/foo.js"]:
            await middleware(
                {"type": "http", "path": path, "method": "GET", "app": mock_stack_mock},
                AsyncMock(),
                AsyncMock(),
            )
            mock_app.assert_called()
            mock_app.reset_mock()

    async def _build_middleware_with_routes(self, route_to_api):
        """Helper to create middleware with pre-built patterns for tests."""
        inner_app, mock_stack, patterns = _make_fake_stack_app(route_to_api)

        def _fake_app(scope, receive, send):
            return inner_app(scope, receive, send)

        middleware = RequestMetricsMiddleware(_fake_app)
        middleware._patterns = patterns
        return middleware

    async def test_tracks_successful_request(self, sample_route_to_api):
        middleware = await self._build_middleware_with_routes(sample_route_to_api)
        scope = {
            "type": "http",
            "path": "/v1/chat/completions",
            "method": "POST",
            "app": AsyncMock(),
        }
        receive = AsyncMock()
        send = AsyncMock()

        await middleware(scope, receive, send)

    async def test_tracks_error_request(self, sample_route_to_api):
        async def error_app(scope, receive, send):
            raise ValueError("test error")

        middleware = RequestMetricsMiddleware(error_app)
        middleware._patterns = _compile_route_patterns(sample_route_to_api)
        mock_stack = AsyncMock()
        mock_stack.impls = {}
        mock_stack_mock = AsyncMock()
        mock_stack_mock.stack = mock_stack
        scope = {"type": "http", "path": "/v1/models", "method": "GET", "app": mock_stack_mock}
        receive = AsyncMock()
        send = AsyncMock()

        with pytest.raises(ValueError, match="test error"):
            await middleware(scope, receive, send)

    async def test_concurrent_requests(self):
        event = asyncio.Event()

        async def slow_app(scope, receive, send):
            await event.wait()
            await send({"type": "http.response.start", "status": 200})

        patterns = _compile_route_patterns(
            {
                "POST:/v1/chat/completions": RouteInfo("inference", "openai_chat_completion"),
            }
        )
        middleware = RequestMetricsMiddleware(slow_app)
        middleware._patterns = patterns
        mock_stack = AsyncMock()
        mock_stack.impls = {}
        mock_stack_mock = AsyncMock()
        mock_stack_mock.stack = mock_stack
        scope = {"type": "http", "path": "/v1/chat/completions", "method": "POST", "app": mock_stack_mock}
        receive = AsyncMock()
        send = AsyncMock()

        tasks = [asyncio.create_task(middleware(scope, receive, send)) for _ in range(3)]
        await asyncio.sleep(0.01)
        event.set()
        await asyncio.gather(*tasks)

    async def test_lazy_build_from_scope_app(self, sample_route_to_api):
        """Middleware should build patterns from scope['app'].stack.impls on first request."""
        mock_app_inner = AsyncMock()
        middleware = RequestMetricsMiddleware(mock_app_inner)
        assert middleware._patterns is None

        stack_mock = AsyncMock()
        stack_mock.impls = {}
        fastapi_app = AsyncMock()
        fastapi_app.stack = stack_mock

        scope = {
            "type": "http",
            "path": "/v1/models",
            "method": "GET",
            "app": fastapi_app,
        }

        async def mock_receive():
            return {"type": "http.request", "body": b""}

        send_mock = AsyncMock()

        # Patch build_route_to_api_map to return known data
        with patch(
            "ogx.core.server.metrics.build_route_to_api_map",
            return_value=sample_route_to_api,
        ) as mock_build:
            await middleware(scope, mock_receive, send_mock)

        mock_build.assert_called_once()
        assert middleware._patterns is not None
        mock_app_inner.assert_called_once()


class TestBuildRouteToApiMap:
    def test_builds_map_from_router_factories(self):
        """Smoke test that build_route_to_api_map doesn't crash with empty inputs."""
        result = build_route_to_api_map({}, {})
        assert result == {}
