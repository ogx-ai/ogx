# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import re
import time
from collections.abc import MutableMapping
from typing import Any

from opentelemetry import metrics
from opentelemetry.metrics import Counter, Histogram, UpDownCounter
from starlette.types import ASGIApp, Receive, Scope, Send

from ogx.log import get_logger
from ogx.telemetry.constants import (
    CONCURRENT_REQUESTS,
    REQUEST_DURATION_SECONDS,
    REQUESTS_TOTAL,
)

logger = get_logger(name=__name__, category="core::server")

meter = metrics.get_meter("ogx.server", version="1.0.0")

requests_total: Counter = meter.create_counter(
    name=REQUESTS_TOTAL,
    description="Total number of API requests",
    unit="1",
)

request_duration_seconds: Histogram = meter.create_histogram(
    name=REQUEST_DURATION_SECONDS,
    description="Duration of API requests in seconds",
    unit="s",
)

concurrent_requests: UpDownCounter = meter.create_up_down_counter(
    name=CONCURRENT_REQUESTS,
    description="Number of concurrent API requests",
    unit="1",
)

# Paths excluded from metrics collection
_EXCLUDED_PATHS = ("/docs", "/redoc", "/openapi.json", "/favicon.ico", "/static")


class RouteInfo:
    """Holds API name and method name for a route."""

    __slots__ = ("api", "method")

    def __init__(self, api: str, method: str):
        self.api = api
        self.method = method


def build_route_to_api_map(
    router_factories: dict[str, Any],
    impls: dict[Any, Any],
) -> dict[str, RouteInfo]:
    """Build a mapping from route path patterns to API and method names.

    This is called once at server startup to create a lookup table
    that maps each registered route (keyed by "HTTP_METHOD:path", e.g. "GET:/v1/models")
    to its API name (e.g. "models") and method name (e.g. "openai_list_models").

    The HTTP method is included in the key because the same path can have different
    handlers for different methods (e.g. GET /v1/models -> list, POST /v1/models -> register).

    Args:
        router_factories: Dict of api_name -> router factory function
        impls: Dict of Api -> implementation instances

    Returns:
        Dict mapping "HTTP_METHOD:path" strings to RouteInfo(api, method)
    """
    from fastapi.routing import APIRoute

    from ogx.core.server.fastapi_router_registry import build_fastapi_router
    from ogx_api import Api

    route_to_api: dict[str, RouteInfo] = {}

    for api_name in router_factories:
        api = Api(api_name)
        if api not in impls:
            continue
        router = build_fastapi_router(api, impls[api])
        if router:
            for route in router.routes:
                if isinstance(route, APIRoute):
                    info = RouteInfo(api_name, route.name or "unknown")
                    for http_method in route.methods or {"GET"}:
                        if http_method == "HEAD":
                            continue
                        route_to_api[f"{http_method}:{route.path}"] = info

    return route_to_api


def _compile_route_patterns(
    route_to_api: dict[str, RouteInfo],
) -> list[tuple[re.Pattern[str], RouteInfo]]:
    """Compile route path templates into regex patterns for matching.

    Keys are "HTTP_METHOD:/v1/models/{model_id}" format.
    Converts to regex like "^GET:/v1/models/[^/]+$".
    """
    patterns = []
    for key, route_info in route_to_api.items():
        # Convert {param:path} to .+ and {param} to [^/]+
        regex = re.sub(
            r"\{\w+:path\}",
            ".+",
            key,
        )
        regex = re.sub(
            r"\{\w+\}",
            "[^/]+",
            regex,
        )
        patterns.append((re.compile(f"^{regex}$"), route_info))
    return patterns


_UNKNOWN_ROUTE = RouteInfo("unknown", "unknown")


class RequestMetricsMiddleware:
    """ASGI middleware that tracks request-level metrics.

    Uses a pre-built route-to-API mapping keyed by HTTP method + path for
    accurate API and method identification.

    Metrics tracked:
    - ogx.requests_total: counter by api, method, status
    - ogx.request_duration_seconds: histogram by api, method
    - ogx.concurrent_requests: up-down counter by api
    """

    def __init__(self, app: ASGIApp, route_to_api: dict[str, RouteInfo] | None = None) -> None:
        self.app = app
        self._patterns: list[tuple[re.Pattern[str], RouteInfo]] = _compile_route_patterns(route_to_api or {})

    def _resolve_route(self, http_method: str, path: str) -> RouteInfo:
        """Resolve HTTP method + path to RouteInfo using compiled route patterns."""
        lookup = f"{http_method}:{path}"
        for pattern, route_info in self._patterns:
            if pattern.match(lookup):
                return route_info
        return _UNKNOWN_ROUTE

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> Any:
        if scope.get("type") != "http":
            return await self.app(scope, receive, send)

        path = scope.get("path", "")

        if any(path.startswith(excluded) for excluded in _EXCLUDED_PATHS):
            return await self.app(scope, receive, send)

        http_method = scope.get("method", "GET")
        route_info = self._resolve_route(http_method, path)
        base_attrs = {"api": route_info.api, "method": route_info.method}
        status_code = 500

        async def send_wrapper(message: MutableMapping[str, Any]) -> None:
            nonlocal status_code
            if message.get("type") == "http.response.start":
                status_code = message.get("status", 200)
            await send(message)

        concurrent_requests.add(1, {"api": route_info.api})
        start_time = time.perf_counter()
        try:
            await self.app(scope, receive, send_wrapper)
            status = "success" if status_code < 400 else "error"
        except asyncio.CancelledError:
            status = "error"
            raise
        except Exception:
            status = "error"
            raise
        finally:
            duration = time.perf_counter() - start_time
            requests_total.add(1, {**base_attrs, "status": status, "status_code": str(status_code)})
            request_duration_seconds.record(duration, base_attrs)
            concurrent_requests.add(-1, {"api": route_info.api})
