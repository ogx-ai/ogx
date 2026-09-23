# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import APIRouter, FastAPI
from fastapi.openapi.utils import get_openapi

from ogx.core.server.fastapi_router_registry import (
    _collect_api_routes_legacy,
    build_fastapi_router,
    collect_api_routes,
    get_router_routes,
)
from ogx.core.server.routes import find_matching_route, initialize_route_impls
from ogx_api.datatypes import Api


def test_get_router_routes_collects_included_router_routes() -> None:
    """FastAPI >= 0.137 keeps included routes behind a wrapper instead of flattening them."""
    nested_router = APIRouter()

    @nested_router.get("/v1/nested")
    async def nested_endpoint() -> None:
        return None

    router = APIRouter()

    @router.get("/v1/top")
    async def top_endpoint() -> None:
        return None

    router.include_router(nested_router)

    assert sorted(route.path for route in get_router_routes(router)) == ["/v1/nested", "/v1/top"]


def test_legacy_traversal_applies_include_wrapper_prefix() -> None:
    """The fallback for fastapi < 0.138, where `iter_route_contexts` does not exist yet.

    Stands in for the 0.137 wrapper so that path stays covered whatever fastapi the
    environment resolves; above 0.138 `collect_api_routes` defers to fastapi instead.
    """
    nested_router = APIRouter()

    @nested_router.get("/items/{item_id}")
    async def nested_endpoint(item_id: str) -> None:
        return None

    wrapper = SimpleNamespace(original_router=nested_router, include_context=SimpleNamespace(prefix="/v1/nested"))

    assert [route.path for route in _collect_api_routes_legacy([wrapper])] == ["/v1/nested/items/{item_id}"]


def test_both_traversals_agree_on_a_prefixed_include() -> None:
    """Whichever path runs, the collected paths are the same — and they are fastapi's own."""
    nested_router = APIRouter()

    @nested_router.get("/items/{item_id}")
    async def nested_endpoint(item_id: str) -> None:
        return None

    router = APIRouter()
    router.include_router(nested_router, prefix="/v1/nested")

    app = FastAPI()
    app.include_router(router)
    schema = get_openapi(title="test", version="1.0.0", routes=app.routes)

    assert sorted(r.path for r in collect_api_routes(router.routes)) == sorted(schema["paths"])
    assert sorted(r.path for r in _collect_api_routes_legacy(router.routes)) == sorted(schema["paths"])


def test_get_router_routes_matches_served_paths_of_prefixed_include() -> None:
    """A prefix passed to include_router() is part of the served path, as fastapi's own schema shows."""
    nested_router = APIRouter()

    @nested_router.get("/items/{item_id}")
    async def nested_endpoint(item_id: str) -> None:
        return None

    router = APIRouter()
    router.include_router(nested_router, prefix="/v1/nested")

    app = FastAPI()
    app.include_router(router)
    schema = get_openapi(title="test", version="1.0.0", routes=app.routes)

    assert sorted(route.path for route in get_router_routes(router)) == sorted(schema["paths"])


def test_get_router_routes_collects_admin_router_routes() -> None:
    """The admin router nests its versioned sub-routers, so route listings must recurse."""
    router = build_fastapi_router(Api.admin, None)
    assert router is not None

    paths = {route.path for route in get_router_routes(router)}
    assert "/v1/admin/tools" in paths
    assert "/v1alpha/admin/connectors" in paths


def test_initialize_route_impls_dispatches_nested_routes() -> None:
    """The library client dispatches in-process against this table, nested routes included."""
    route_impls = initialize_route_impls({Api.admin: AsyncMock()})

    _func, path_params, route_path, _auth_info = find_matching_route(
        "GET", "/v1alpha/admin/connectors/my-connector", route_impls
    )

    assert route_path == "/v1alpha/admin/connectors/{connector_id}"
    assert path_params == {"connector_id": "my-connector"}
