# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import logging  # allow-direct-logging
from tempfile import TemporaryDirectory

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ogx.core.access_control.access_control import default_policy
from ogx.core.conversations.conversations import ConversationServiceConfig, ConversationServiceImpl
from ogx.core.datatypes import (
    AuthenticationConfig,
    AuthProviderType,
    StackConfig,
    TenancyConfig,
    TenancyMode,
    UpstreamHeaderAuthConfig,
)
from ogx.core.server.auth import AuthenticationMiddleware, TenancyMiddleware
from ogx.core.server.fastapi_router_registry import build_fastapi_router
from ogx.core.server.routes import RouteAuthInfo
from ogx.core.server.server import ProviderDataMiddleware
from ogx.core.storage.datatypes import (
    ServerStoresConfig,
    SqliteSqlStoreConfig,
    SqlStoreReference,
    StorageConfig,
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
from ogx_api import Api


@pytest.fixture
def suppress_auth_errors(caplog):
    caplog.set_level(logging.CRITICAL, logger="ogx.core.server.auth")
    caplog.set_level(logging.CRITICAL, logger="ogx.core.server.auth_providers")


def _headers(principal: str, tenant_id: str) -> dict[str, str]:
    return {"x-auth-user-id": principal, "x-tenant-id": tenant_id}


def _make_app_and_client(
    tenancy_config: TenancyConfig,
    conv_impl: ConversationServiceImpl,
    monkeypatch,
    *,
    include_auth: bool = True,
    include_tenancy: bool = True,
):
    app = FastAPI()

    router = build_fastapi_router(Api.conversations, conv_impl)
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

    # Middleware wraps in reverse add order — last added runs first.
    # Production order (server.py:296-319): Request → Auth → Tenancy → ProviderData → App
    # So add order is: ProviderData first (innermost), then Tenancy, then Auth last (outermost).
    app.add_middleware(ProviderDataMiddleware)
    if include_tenancy and tenancy_config.mode != TenancyMode.DISABLED:
        app.add_middleware(TenancyMiddleware, tenancy_config=tenancy_config)
    if include_auth:
        app.add_middleware(AuthenticationMiddleware, auth_config=auth_config)

    monkeypatch.setattr(
        "ogx.core.server.auth.find_matching_route",
        lambda method, path, route_impls: (None, {}, path, RouteAuthInfo(require_authentication=True)),
    )

    return TestClient(app, raise_server_exceptions=False)


def _setup_globals(tmp: str, tenancy_config: TenancyConfig):
    """Register SQLite backend and set tenancy config globals.

    Returns saved state for cleanup.
    """
    saved_backends = dict(_SQLSTORE_BACKENDS)
    saved_instances = dict(_SQLSTORE_INSTANCES)
    saved_locks = dict(_SQLSTORE_LOCKS)
    saved_tenancy = get_default_tenancy_config()

    register_sqlstore_backends({"sql_test": SqliteSqlStoreConfig(db_path=f"{tmp}/e2e.db")})
    set_default_tenancy_config(tenancy_config)

    return saved_backends, saved_instances, saved_locks, saved_tenancy


def _restore_globals(saved):
    """Restore process-wide globals after test."""
    saved_backends, saved_instances, saved_locks, saved_tenancy = saved
    _SQLSTORE_BACKENDS.clear()
    _SQLSTORE_BACKENDS.update(saved_backends)
    _SQLSTORE_INSTANCES.clear()
    _SQLSTORE_INSTANCES.update(saved_instances)
    _SQLSTORE_LOCKS.clear()
    _SQLSTORE_LOCKS.update(saved_locks)
    set_default_tenancy_config(saved_tenancy)


def _create_conv_impl(tmp: str) -> ConversationServiceImpl:
    """Create a real ConversationServiceImpl backed by SQLite."""
    stack_config = StackConfig(
        distro_name="test-tenancy",
        providers={},
        storage=StorageConfig(
            backends={"sql_test": SqliteSqlStoreConfig(db_path=f"{tmp}/e2e.db")},
            stores=ServerStoresConfig(
                metadata=None,
                inference=None,
                conversations=SqlStoreReference(backend="sql_test", table_name="openai_conversations"),
                prompts=None,
                connectors=None,
            ),
        ),
    )
    conv_config = ConversationServiceConfig(config=stack_config, policy=default_policy())
    impl = ConversationServiceImpl(conv_config, deps={})
    asyncio.run(impl.initialize())
    return impl


@pytest.fixture
def multi_tenant_setup(monkeypatch):
    with TemporaryDirectory() as tmp:
        tenancy_config = TenancyConfig(mode=TenancyMode.MULTI)
        saved = _setup_globals(tmp, tenancy_config)
        try:
            conv_impl = _create_conv_impl(tmp)
            client = _make_app_and_client(tenancy_config, conv_impl, monkeypatch)
            yield client, conv_impl
        finally:
            _restore_globals(saved)


def _create_conversation(client: TestClient, principal: str, tenant_id: str) -> str:
    """Create a conversation and return its ID."""
    resp = client.post(
        "/v1/conversations",
        json={},
        headers=_headers(principal, tenant_id),
    )
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    return resp.json()["id"]


def test_cross_tenant_read_isolation(multi_tenant_setup):
    """Full pipeline: Auth → Tenancy → ProviderData → ConversationServiceImpl → AuthorizedSqlStore.
    Cross-tenant reads return 404."""
    client, _ = multi_tenant_setup

    conv_id = _create_conversation(client, "alice", "tenant-a")

    bob_resp = client.get(f"/v1/conversations/{conv_id}", headers=_headers("bob", "tenant-b"))
    assert bob_resp.status_code == 404

    alice_resp = client.get(f"/v1/conversations/{conv_id}", headers=_headers("alice", "tenant-a"))
    assert alice_resp.status_code == 200
    assert alice_resp.json()["id"] == conv_id


def test_cross_tenant_delete_isolation(multi_tenant_setup):
    """Cross-tenant delete returns 404 and does not remove the row."""
    client, _ = multi_tenant_setup

    conv_id = _create_conversation(client, "alice", "tenant-a")

    bob_resp = client.delete(f"/v1/conversations/{conv_id}", headers=_headers("bob", "tenant-b"))
    assert bob_resp.status_code == 404

    alice_resp = client.get(f"/v1/conversations/{conv_id}", headers=_headers("alice", "tenant-a"))
    assert alice_resp.status_code == 200


def test_cross_tenant_update_isolation(multi_tenant_setup):
    """Cross-tenant update returns 404 and does not modify the row."""
    client, _ = multi_tenant_setup

    conv_id = _create_conversation(client, "alice", "tenant-a")

    bob_resp = client.post(
        f"/v1/conversations/{conv_id}",
        json={"metadata": {"hacked": "true"}},
        headers=_headers("bob", "tenant-b"),
    )
    assert bob_resp.status_code == 404

    alice_resp = client.get(f"/v1/conversations/{conv_id}", headers=_headers("alice", "tenant-a"))
    assert alice_resp.status_code == 200
    assert alice_resp.json().get("metadata") is None


def test_multi_mode_rejects_missing_tenant(multi_tenant_setup, suppress_auth_errors):
    """In multi mode, a request with principal but no tenant header gets 401."""
    client, _ = multi_tenant_setup

    response = client.post(
        "/v1/conversations",
        json={},
        headers={"x-auth-user-id": "alice"},
    )
    assert response.status_code == 401
    assert "Tenant context required" in response.json()["error"]["message"]


def test_single_mode_stamps_default_tenant(monkeypatch):
    """In single mode, all data is stamped with the configured default tenant."""
    with TemporaryDirectory() as tmp:
        tenancy_config = TenancyConfig(mode=TenancyMode.SINGLE, default_tenant_id="default-org")
        saved = _setup_globals(tmp, tenancy_config)
        try:
            conv_impl = _create_conv_impl(tmp)
            client = _make_app_and_client(tenancy_config, conv_impl, monkeypatch)

            conv_id = _create_conversation(client, "alice", "any-tenant")

            raw = asyncio.run(conv_impl.sql_store.sql_store.fetch_all("openai_conversations", where={"id": conv_id}))
            assert raw.data[0]["tenant_id"] == "default-org"
        finally:
            _restore_globals(saved)


def test_disabled_mode_no_tenant_filtering(monkeypatch):
    """In disabled mode, no tenant column exists. Same user sees all their own rows
    regardless of tenant headers (ABAC owner isolation still applies)."""
    with TemporaryDirectory() as tmp:
        tenancy_config = TenancyConfig(mode=TenancyMode.DISABLED)
        saved = _setup_globals(tmp, tenancy_config)
        try:
            conv_impl = _create_conv_impl(tmp)
            client = _make_app_and_client(tenancy_config, conv_impl, monkeypatch, include_tenancy=False)

            _create_conversation(client, "alice", "tenant-a")
            _create_conversation(client, "alice", "tenant-b")

            raw = asyncio.run(conv_impl.sql_store.sql_store.fetch_all("openai_conversations"))
            assert len(raw.data) == 2
        finally:
            _restore_globals(saved)


def test_same_tenant_different_users_see_only_own(multi_tenant_setup):
    """Within the same tenant, ABAC (user is owner) still applies.
    Each user sees only their own conversations even though they share the tenant partition."""
    client, conv_impl = multi_tenant_setup

    alice_id = _create_conversation(client, "alice", "tenant-a")
    bob_id = _create_conversation(client, "bob", "tenant-a")

    alice_get = client.get(f"/v1/conversations/{alice_id}", headers=_headers("alice", "tenant-a"))
    assert alice_get.status_code == 200

    alice_cross = client.get(f"/v1/conversations/{bob_id}", headers=_headers("alice", "tenant-a"))
    assert alice_cross.status_code == 404

    bob_get = client.get(f"/v1/conversations/{bob_id}", headers=_headers("bob", "tenant-a"))
    assert bob_get.status_code == 200

    bob_cross = client.get(f"/v1/conversations/{alice_id}", headers=_headers("bob", "tenant-a"))
    assert bob_cross.status_code == 404

    raw = asyncio.run(conv_impl.sql_store.sql_store.fetch_all("openai_conversations", where={"tenant_id": "tenant-a"}))
    assert len(raw.data) == 2
