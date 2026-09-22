# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Router utilities for FastAPI routers.

This module auto-discovers FastAPI routers from ogx_api packages.
Each API package that has a `fastapi_routes` submodule with a `create_router`
factory is automatically registered.

External APIs can also provide a `create_router` function in their module
(the same module that provides `available_providers`).
"""

import copy
import importlib
import importlib.util
from collections.abc import Callable
from typing import Any, cast

import fastapi.routing
from fastapi import APIRouter
from fastapi.routing import APIRoute
from starlette.routing import compile_path

from ogx.log import get_logger
from ogx_api.datatypes import Api, ExternalApiSpec

# fastapi resolves included routers itself from 0.138 on. Read through `getattr` rather
# than a guarded import so this type-checks against either version.
iter_route_contexts = getattr(fastapi.routing, "iter_route_contexts", None)

logger = get_logger(__name__, category="core")

# Api enum values that don't match their package name in ogx_api
_API_TO_PACKAGE: dict[str, str] = {
    "inspect": "inspect_api",
    "tool_groups": "tools",
}


def _discover_router_factories() -> dict[str, Callable[[Any], APIRouter]]:
    """Auto-discover router factories from ogx_api packages.

    For each Api enum value, try to import
    `ogx_api.<package>.fastapi_routes.create_router`.
    APIs without a fastapi_routes module (e.g. vector_stores, tool_runtime)
    are silently skipped.
    """
    factories: dict[str, Callable[[Any], APIRouter]] = {}
    for api in Api:
        package_name = _API_TO_PACKAGE.get(api.value, api.value)
        module_path = f"ogx_api.{package_name}.fastapi_routes"
        # Check if the module exists before importing — a missing module is
        # expected (not all APIs have routers), but a broken import is a bug
        # that should be surfaced.
        try:
            spec = importlib.util.find_spec(module_path)
        except (ModuleNotFoundError, ValueError):
            continue
        if spec is None:
            continue
        try:
            module = importlib.import_module(module_path)
            create_router = getattr(module, "create_router", None)
            if create_router is not None:
                factories[api.value] = create_router
        except Exception:
            logger.warning("Failed to import router module", module=module_path, exc_info=True)
    return factories


_ROUTER_FACTORIES: dict[str, Callable[[Any], APIRouter]] = _discover_router_factories()


def register_external_api_routers(external_apis: dict[Api, ExternalApiSpec]) -> None:
    """Register router factories from external API modules.

    External APIs can provide a `create_router(impl) -> APIRouter` function
    in their module to define FastAPI routes.
    """
    for api, api_spec in external_apis.items():
        if api.value in _ROUTER_FACTORIES:
            continue
        try:
            module = importlib.import_module(api_spec.module)
            create_router = getattr(module, "create_router", None)
            if create_router is not None:
                _ROUTER_FACTORIES[api.value] = create_router
        except Exception:
            logger.warning("Failed to import external API router", api=api.value, module=api_spec.module, exc_info=True)


def build_fastapi_router(api: "Api", impl: Any) -> APIRouter | None:
    """Build a router for an API using its auto-discovered router factory."""
    router_factory = _ROUTER_FACTORIES.get(api.value)
    if router_factory is None:
        return None

    return cast(APIRouter, router_factory(impl))


def _with_path(route: APIRoute, path: str) -> APIRoute:
    """Copy a route carrying `path`, the way include_router() serves it."""
    served = copy.copy(route)
    served.path = path
    served.path_regex, served.path_format, served.param_convertors = compile_path(path)
    return served


def _with_path_prefix(route: APIRoute, prefix: str) -> APIRoute:
    """Copy a route with `prefix` prepended to its path, the way include_router() serves it."""
    return _with_path(route, prefix + route.path)


def collect_api_routes(routes: list[Any], prefix: str = "") -> list[APIRoute]:
    """Collect all APIRoute objects, recursing into included routers.

    On fastapi >= 0.138 this defers to `fastapi.routing.iter_route_contexts`, which resolves
    included routers and their prefixes itself. Below that the traversal is ours, because the
    two shapes fastapi has had need different handling: < 0.137 `include_router()` flattened
    the sub-router's routes into the parent with the prefix already baked into each path,
    while 0.137 wraps them and keeps the prefix on the wrapper, so it has to be reapplied.

    The other include-time options (tags, deprecated, include_in_schema) are not merged —
    no OGX router passes them, and every consumer of this helper keys off the path.
    """
    if iter_route_contexts is not None and not prefix:
        collected: list[APIRoute] = []
        for context in iter_route_contexts(routes):
            route = context.original_route
            if not isinstance(route, APIRoute):
                continue
            path = context.path
            collected.append(_with_path(route, path) if path != route.path else route)
        return collected
    return _collect_api_routes_legacy(routes, prefix)


def _collect_api_routes_legacy(routes: list[Any], prefix: str = "") -> list[APIRoute]:
    """Traverse included routers by hand, for fastapi < 0.138."""
    api_routes: list[APIRoute] = []
    for route in routes:
        if isinstance(route, APIRoute):
            api_routes.append(_with_path_prefix(route, prefix) if prefix else route)
        elif hasattr(route, "original_router"):
            # FastAPI 0.137 wraps include_router() results in _IncludedRouter, whose
            # include_context.prefix already covers the prefix of the router it was included into.
            include_context = getattr(route, "include_context", None)
            api_routes.extend(
                _collect_api_routes_legacy(
                    route.original_router.routes, prefix + getattr(include_context, "prefix", "")
                )
            )
        elif hasattr(route, "routes"):
            api_routes.extend(_collect_api_routes_legacy(route.routes, prefix))
    return api_routes


def get_router_routes(router: APIRouter) -> list[APIRoute]:
    """Extract APIRoute objects from a FastAPI router, including those of included routers."""
    return collect_api_routes(router.routes)
