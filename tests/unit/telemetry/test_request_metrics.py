# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
from unittest.mock import AsyncMock

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
        "POST:/v1/completions": RouteInfo("inference", "openai_completion"),
        "POST:/v1/embeddings": RouteInfo("inference", "openai_embeddings"),
        "GET:/v1/models": RouteInfo("models", "openai_list_models"),
        "POST:/v1/models": RouteInfo("models", "register_model"),
        "GET:/v1/models/{model_id}": RouteInfo("models", "get_model"),
        "DELETE:/v1/models/{model_id}": RouteInfo("models", "unregister_model"),
        "GET:/v1/shields": RouteInfo("shields", "list_shields"),
        "GET:/v1/shields/{identifier:path}": RouteInfo("shields", "get_shield"),
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

        route = middleware._resolve_route("GET", "/v1/shields/my-shield")
        assert route.api == "shields"
        assert route.method == "get_shield"

        # :path param should match slashes
        route = middleware._resolve_route("GET", "/v1/shields/namespace/my-shield")
        assert route.api == "shields"
        assert route.method == "get_shield"

    def test_nested_parameterized_path(self, sample_route_to_api):
        patterns = _compile_route_patterns(sample_route_to_api)
        middleware = RequestMetricsMiddleware.__new__(RequestMetricsMiddleware)
        middleware._patterns = patterns

        route = middleware._resolve_route("POST", "/v1/agents/agent-123/sessions/sess-456/turns")
        assert route.api == "agents"
        assert route.method == "create_agent_turn"

    def test_unknown_path(self, sample_route_to_api):
        patterns = _compile_route_patterns(sample_route_to_api)
        middleware = RequestMetricsMiddleware.__new__(RequestMetricsMiddleware)
        middleware._patterns = patterns

        route = middleware._resolve_route("GET", "/v1/nonexistent")
        assert route.api == "unknown"
        assert route.method == "unknown"


class TestRequestMetricsMiddleware:
    @pytest.fixture
    def middleware(self, sample_route_to_api):
        mock_app = AsyncMock()
        return RequestMetricsMiddleware(mock_app, route_to_api=sample_route_to_api)

    async def test_skips_non_http(self, middleware):
        scope = {"type": "lifespan"}
        receive = AsyncMock()
        send = AsyncMock()

        await middleware(scope, receive, send)
        middleware.app.assert_called_once_with(scope, receive, send)

    async def test_skips_excluded_paths(self, middleware):
        for path in ["/docs", "/redoc", "/openapi.json", "/favicon.ico", "/static/foo.js"]:
            middleware.app.reset_mock()
            scope = {"type": "http", "path": path, "method": "GET"}
            receive = AsyncMock()
            send = AsyncMock()
            await middleware(scope, receive, send)
            middleware.app.assert_called_once()

    async def test_tracks_successful_request(self, sample_route_to_api):
        async def mock_app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200})
            await send({"type": "http.response.body", "body": b"ok"})

        middleware = RequestMetricsMiddleware(mock_app, route_to_api=sample_route_to_api)
        scope = {"type": "http", "path": "/v1/chat/completions", "method": "POST"}
        receive = AsyncMock()
        send = AsyncMock()

        await middleware(scope, receive, send)

    async def test_tracks_error_request(self, sample_route_to_api):
        async def mock_app(scope, receive, send):
            raise ValueError("test error")

        middleware = RequestMetricsMiddleware(mock_app, route_to_api=sample_route_to_api)
        scope = {"type": "http", "path": "/v1/models", "method": "GET"}
        receive = AsyncMock()
        send = AsyncMock()

        with pytest.raises(ValueError, match="test error"):
            await middleware(scope, receive, send)

    async def test_concurrent_requests(self, sample_route_to_api):
        event = asyncio.Event()

        async def slow_app(scope, receive, send):
            await event.wait()
            await send({"type": "http.response.start", "status": 200})

        middleware = RequestMetricsMiddleware(slow_app, route_to_api=sample_route_to_api)
        scope = {"type": "http", "path": "/v1/chat/completions", "method": "POST"}
        receive = AsyncMock()
        send = AsyncMock()

        tasks = [asyncio.create_task(middleware(scope, receive, send)) for _ in range(3)]
        await asyncio.sleep(0.01)
        event.set()
        await asyncio.gather(*tasks)


class TestBuildRouteToApiMap:
    def test_builds_map_from_router_factories(self):
        """Smoke test that build_route_to_api_map doesn't crash with empty inputs."""
        result = build_route_to_api_map({}, {})
        assert result == {}
