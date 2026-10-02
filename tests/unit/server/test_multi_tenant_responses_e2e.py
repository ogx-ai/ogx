# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Cross-tenant isolation tests for the Responses API through the full HTTP middleware stack.

Exercises: AuthenticationMiddleware → TenancyMiddleware → ProviderDataMiddleware
           → TenancyTestResponsesImpl → ResponsesStore → AuthorizedSqlStore

Uses a thin Responses protocol implementation that wraps a real ResponsesStore
backed by SQLite, bypassing the inference-heavy BuiltinResponsesImpl while still
exercising the full middleware-to-storage chain for CRUD operations.
"""

import asyncio
import logging  # allow-direct-logging
import time
from collections.abc import AsyncIterator
from tempfile import TemporaryDirectory
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ogx.core.access_control.access_control import default_policy
from ogx.core.datatypes import (
    AuthenticationConfig,
    AuthProviderType,
    TenancyConfig,
    TenancyMode,
    UpstreamHeaderAuthConfig,
)
from ogx.core.server.auth import AuthenticationMiddleware, TenancyMiddleware
from ogx.core.server.fastapi_router_registry import build_fastapi_router
from ogx.core.server.routes import RouteAuthInfo
from ogx.core.server.server import ProviderDataMiddleware
from ogx.core.storage.datatypes import (
    SqliteSqlStoreConfig,
    SqlStoreReference,
)
from ogx.core.storage.sqlstore.authorized_sqlstore import (
    get_default_tenancy_config,
    set_default_tenancy_config,
)
from ogx.core.storage.sqlstore.sqlstore import (
    _SQLSTORE_BACKENDS,
    _SQLSTORE_INSTANCES,
    _SQLSTORE_LOCKS,
    register_sqlstore_backends,
)
from ogx.providers.utils.responses.responses_store import ResponsesStore
from ogx_api import Api
from ogx_api.openai_responses import (
    OpenAICompactedResponse,
    OpenAIDeleteResponseObject,
    OpenAIResponseObject,
    OpenAIResponseObjectStream,
    OpenAIResponseText,
    OpenAIResponseTextFormat,
)
from ogx_api.responses.models import (
    CancelResponseRequest,
    CompactResponseRequest,
    CreateResponseRequest,
    DeleteResponseRequest,
    ListResponseInputItemsRequest,
    ListResponsesRequest,
    RetrieveResponseRequest,
)


@pytest.fixture
def suppress_auth_errors(caplog):
    caplog.set_level(logging.CRITICAL, logger="ogx.core.server.auth")
    caplog.set_level(logging.CRITICAL, logger="ogx.core.server.auth_providers")


def _headers(principal: str, tenant_id: str) -> dict[str, str]:
    return {"x-auth-user-id": principal, "x-tenant-id": tenant_id}


class TenancyTestResponsesImpl:
    """Thin Responses implementation that delegates CRUD to a real ResponsesStore.

    Avoids the 10+ dependency BuiltinResponsesImpl while still exercising
    the full HTTP → middleware → AuthorizedSqlStore chain.
    """

    def __init__(self, store: ResponsesStore):
        self.store = store

    async def create_openai_response(
        self, request: CreateResponseRequest
    ) -> OpenAIResponseObject | AsyncIterator[OpenAIResponseObjectStream]:
        response = OpenAIResponseObject(
            id=f"resp_{uuid4().hex[:12]}",
            created_at=int(time.time()),
            model=request.model or "test-model",
            output=[],
            status="completed",
            store=True,
            text=OpenAIResponseText(format=OpenAIResponseTextFormat(type="text")),
        )
        await self.store.store_response_object(response, [], [])
        return response

    async def get_openai_response(self, request: RetrieveResponseRequest) -> OpenAIResponseObject:
        resp = await self.store.get_response_object(request.response_id)
        return OpenAIResponseObject(**resp.model_dump())

    async def list_openai_responses(self, request: ListResponsesRequest):
        return await self.store.list_responses(
            after=request.after,
            limit=request.limit,
            model=request.model,
            order=request.order,
        )

    async def delete_openai_response(self, request: DeleteResponseRequest) -> OpenAIDeleteResponseObject:
        return await self.store.delete_response_object(request.response_id)

    async def list_openai_response_input_items(self, request: ListResponseInputItemsRequest):
        raise NotImplementedError

    async def compact_openai_response(self, request: CompactResponseRequest) -> OpenAICompactedResponse:
        raise NotImplementedError

    async def cancel_openai_response(self, request: CancelResponseRequest) -> OpenAIResponseObject:
        raise NotImplementedError


def _make_app_and_client(
    tenancy_config: TenancyConfig,
    responses_impl: TenancyTestResponsesImpl,
    monkeypatch,
):
    app = FastAPI()

    router = build_fastapi_router(Api.responses, responses_impl)
    assert router is not None
    app.include_router(router)

    auth_config = AuthenticationConfig(
        provider_config=UpstreamHeaderAuthConfig(
            type=AuthProviderType.UPSTREAM_HEADER,
            principal_header="x-auth-user-id",
            tenant_header="x-tenant-id",
        ),
        access_policy=[],
    )

    app.add_middleware(ProviderDataMiddleware)
    app.add_middleware(TenancyMiddleware, tenancy_config=tenancy_config)
    app.add_middleware(AuthenticationMiddleware, auth_config=auth_config)

    monkeypatch.setattr(
        "ogx.core.server.auth.find_matching_route",
        lambda method, path, route_impls: (None, {}, path, RouteAuthInfo(require_authentication=True)),
    )

    return TestClient(app, raise_server_exceptions=False)


def _setup_globals(tmp: str, tenancy_config: TenancyConfig):
    saved_backends = dict(_SQLSTORE_BACKENDS)
    saved_instances = dict(_SQLSTORE_INSTANCES)
    saved_locks = dict(_SQLSTORE_LOCKS)
    saved_tenancy = get_default_tenancy_config()

    register_sqlstore_backends({"sql_test": SqliteSqlStoreConfig(db_path=f"{tmp}/e2e.db")})
    set_default_tenancy_config(tenancy_config)

    return saved_backends, saved_instances, saved_locks, saved_tenancy


def _restore_globals(saved):
    saved_backends, saved_instances, saved_locks, saved_tenancy = saved
    _SQLSTORE_BACKENDS.clear()
    _SQLSTORE_BACKENDS.update(saved_backends)
    _SQLSTORE_INSTANCES.clear()
    _SQLSTORE_INSTANCES.update(saved_instances)
    _SQLSTORE_LOCKS.clear()
    _SQLSTORE_LOCKS.update(saved_locks)
    set_default_tenancy_config(saved_tenancy)


def _create_responses_store(tmp: str) -> ResponsesStore:
    store = ResponsesStore(
        SqlStoreReference(backend="sql_test", table_name="responses"),
        default_policy(),
    )
    asyncio.run(store.initialize())
    return store


@pytest.fixture
def multi_tenant_setup(monkeypatch):
    with TemporaryDirectory() as tmp:
        tenancy_config = TenancyConfig(mode=TenancyMode.MULTI)
        saved = _setup_globals(tmp, tenancy_config)
        try:
            store = _create_responses_store(tmp)
            impl = TenancyTestResponsesImpl(store)
            client = _make_app_and_client(tenancy_config, impl, monkeypatch)
            yield client, impl
        finally:
            _restore_globals(saved)


def _create_response(client: TestClient, principal: str, tenant_id: str) -> str:
    resp = client.post(
        "/v1/responses",
        json={"model": "test-model", "input": "hello"},
        headers=_headers(principal, tenant_id),
    )
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    return resp.json()["id"]


def test_cross_tenant_get_response_isolation(multi_tenant_setup):
    """Create as tenant-a/alice, GET as tenant-b/bob -> 404."""
    client, _ = multi_tenant_setup

    response_id = _create_response(client, "alice", "tenant-a")

    bob_resp = client.get(f"/v1/responses/{response_id}", headers=_headers("bob", "tenant-b"))
    assert bob_resp.status_code == 404

    alice_resp = client.get(f"/v1/responses/{response_id}", headers=_headers("alice", "tenant-a"))
    assert alice_resp.status_code == 200
    assert alice_resp.json()["id"] == response_id


def test_cross_tenant_delete_response_isolation(multi_tenant_setup):
    """Cross-tenant delete returns 404 and response survives."""
    client, _ = multi_tenant_setup

    response_id = _create_response(client, "alice", "tenant-a")

    bob_resp = client.delete(f"/v1/responses/{response_id}", headers=_headers("bob", "tenant-b"))
    assert bob_resp.status_code == 404

    alice_resp = client.get(f"/v1/responses/{response_id}", headers=_headers("alice", "tenant-a"))
    assert alice_resp.status_code == 200


def test_cross_tenant_list_responses_isolation(multi_tenant_setup):
    """Each tenant only sees their own responses in list results."""
    client, _ = multi_tenant_setup

    _create_response(client, "alice", "tenant-a")
    _create_response(client, "alice", "tenant-a")
    _create_response(client, "bob", "tenant-b")

    alice_list = client.get("/v1/responses", headers=_headers("alice", "tenant-a"))
    assert alice_list.status_code == 200
    alice_data = alice_list.json()["data"]
    assert len(alice_data) == 2

    bob_list = client.get("/v1/responses", headers=_headers("bob", "tenant-b"))
    assert bob_list.status_code == 200
    bob_data = bob_list.json()["data"]
    assert len(bob_data) == 1


def test_multi_mode_rejects_missing_tenant_responses(multi_tenant_setup, suppress_auth_errors):
    """In multi mode, a request with principal but no tenant header gets 401."""
    client, _ = multi_tenant_setup

    response = client.post(
        "/v1/responses",
        json={"model": "test-model", "input": "hello"},
        headers={"x-auth-user-id": "alice"},
    )
    assert response.status_code == 401


def test_same_tenant_different_users_responses(multi_tenant_setup):
    """Within the same tenant, ABAC (user is owner) still applies.
    Each user sees only their own responses."""
    client, _ = multi_tenant_setup

    alice_resp_id = _create_response(client, "alice", "tenant-a")
    bob_resp_id = _create_response(client, "bob", "tenant-a")

    alice_get_own = client.get(f"/v1/responses/{alice_resp_id}", headers=_headers("alice", "tenant-a"))
    assert alice_get_own.status_code == 200

    alice_get_bob = client.get(f"/v1/responses/{bob_resp_id}", headers=_headers("alice", "tenant-a"))
    assert alice_get_bob.status_code == 404

    bob_get_own = client.get(f"/v1/responses/{bob_resp_id}", headers=_headers("bob", "tenant-a"))
    assert bob_get_own.status_code == 200

    bob_get_alice = client.get(f"/v1/responses/{alice_resp_id}", headers=_headers("bob", "tenant-a"))
    assert bob_get_alice.status_code == 404
