# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock

from ogx.core.datatypes import TenancyConfig, TenancyMode
from ogx.core.request_headers import get_authenticated_user
from ogx.core.server.auth import TenancyMiddleware
from ogx.core.server.routes import RouteAuthInfo
from ogx.core.server.server import ProviderDataMiddleware


async def _receive():
    return {"type": "http.request", "body": b"", "more_body": False}


async def test_multi_tenancy_allows_public_route_without_tenant(monkeypatch):
    app = AsyncMock()
    send = AsyncMock()
    middleware = TenancyMiddleware(app, TenancyConfig(mode=TenancyMode.MULTI), impls={})

    monkeypatch.setattr("ogx.core.server.auth.initialize_route_impls", lambda impls: {})
    monkeypatch.setattr(
        "ogx.core.server.auth.find_matching_route",
        lambda method, path, route_impls: (None, {}, path, RouteAuthInfo(require_authentication=False)),
    )

    await middleware({"type": "http", "method": "GET", "path": "/v1/health"}, _receive, send)

    app.assert_awaited_once()
    send.assert_not_awaited()


async def test_multi_tenancy_rejects_private_route_without_tenant(monkeypatch):
    app = AsyncMock()
    send = AsyncMock()
    middleware = TenancyMiddleware(app, TenancyConfig(mode=TenancyMode.MULTI), impls={})

    monkeypatch.setattr("ogx.core.server.auth.initialize_route_impls", lambda impls: {})
    monkeypatch.setattr(
        "ogx.core.server.auth.find_matching_route",
        lambda method, path, route_impls: (None, {}, path, RouteAuthInfo(require_authentication=True)),
    )

    await middleware({"type": "http", "method": "GET", "path": "/v1/models"}, _receive, send)

    app.assert_not_awaited()
    assert send.await_args_list[0].args[0]["status"] == 401


async def test_provider_data_context_wraps_websocket_authenticated_user():
    seen = []

    async def app(scope, receive, send):
        seen.append(get_authenticated_user())

    middleware = ProviderDataMiddleware(app)

    await middleware(
        {
            "type": "websocket",
            "headers": [],
            "principal": "alice",
            "tenant_id": "tenant-a",
            "user_attributes": {"roles": ["admin"]},
        },
        _receive,
        AsyncMock(),
    )

    assert seen[0] is not None
    assert seen[0].principal == "alice"
    assert seen[0].tenant_id == "tenant-a"
    assert seen[0].attributes == {"roles": ["admin"]}
