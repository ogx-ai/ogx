# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Cross-tenant isolation tests for the Files API through the full HTTP middleware stack.

Exercises: AuthenticationMiddleware → TenancyMiddleware → ProviderDataMiddleware
           → LocalfsFilesImpl → AuthorizedSqlStore

Mirrors the pattern from test_multi_tenant_e2e.py (conversations) but for file
uploads, reads, deletes, and content retrieval.
"""

import asyncio
import logging  # allow-direct-logging
from tempfile import TemporaryDirectory

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
from ogx.core.server.server import ProviderDataMiddleware, global_exception_handler
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
from ogx.providers.inline.files.localfs.config import LocalfsFilesImplConfig
from ogx.providers.inline.files.localfs.files import LocalfsFilesImpl
from ogx_api import Api, ResourceNotFoundError


@pytest.fixture
def suppress_auth_errors(caplog):
    caplog.set_level(logging.CRITICAL, logger="ogx.core.server.auth")
    caplog.set_level(logging.CRITICAL, logger="ogx.core.server.auth_providers")


def _headers(principal: str, tenant_id: str) -> dict[str, str]:
    return {"x-auth-user-id": principal, "x-tenant-id": tenant_id}


def _make_app_and_client(
    tenancy_config: TenancyConfig,
    files_impl: LocalfsFilesImpl,
    monkeypatch,
):
    app = FastAPI()

    router = build_fastapi_router(Api.files, files_impl)
    assert router is not None
    app.include_router(router)

    # The files router does not use ExceptionTranslatingRoute, so we need
    # the global exception handler to translate ResourceNotFoundError to 404.
    app.exception_handler(ResourceNotFoundError)(global_exception_handler)
    app.exception_handler(Exception)(global_exception_handler)

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


def _create_files_impl(tmp: str) -> LocalfsFilesImpl:
    config = LocalfsFilesImplConfig(
        storage_dir=f"{tmp}/files",
        metadata_store=SqlStoreReference(backend="sql_test", table_name="openai_files"),
    )
    impl = LocalfsFilesImpl(config, default_policy())
    asyncio.run(impl.initialize())
    return impl


@pytest.fixture
def multi_tenant_setup(monkeypatch):
    with TemporaryDirectory() as tmp:
        tenancy_config = TenancyConfig(mode=TenancyMode.MULTI)
        saved = _setup_globals(tmp, tenancy_config)
        try:
            files_impl = _create_files_impl(tmp)
            client = _make_app_and_client(tenancy_config, files_impl, monkeypatch)
            yield client, files_impl
        finally:
            _restore_globals(saved)


def _upload_file(
    client: TestClient,
    principal: str,
    tenant_id: str,
    content: bytes = b"hello world",
    filename: str = "test.txt",
    purpose: str = "assistants",
) -> str:
    resp = client.post(
        "/v1/files",
        files={"file": (filename, content, "text/plain")},
        data={"purpose": purpose},
        headers=_headers(principal, tenant_id),
    )
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    return resp.json()["id"]


def test_cross_tenant_read_file_isolation(multi_tenant_setup):
    """Upload as tenant-a/alice, GET as tenant-b/bob -> 404."""
    client, _ = multi_tenant_setup

    file_id = _upload_file(client, "alice", "tenant-a")

    bob_resp = client.get(f"/v1/files/{file_id}", headers=_headers("bob", "tenant-b"))
    assert bob_resp.status_code == 404

    alice_resp = client.get(f"/v1/files/{file_id}", headers=_headers("alice", "tenant-a"))
    assert alice_resp.status_code == 200
    assert alice_resp.json()["id"] == file_id


def test_cross_tenant_delete_file_isolation(multi_tenant_setup):
    """Cross-tenant delete returns 404 and file survives."""
    client, _ = multi_tenant_setup

    file_id = _upload_file(client, "alice", "tenant-a")

    bob_resp = client.delete(f"/v1/files/{file_id}", headers=_headers("bob", "tenant-b"))
    assert bob_resp.status_code == 404

    alice_resp = client.get(f"/v1/files/{file_id}", headers=_headers("alice", "tenant-a"))
    assert alice_resp.status_code == 200


def test_cross_tenant_list_files_isolation(multi_tenant_setup):
    """Each tenant only sees their own files in list results."""
    client, _ = multi_tenant_setup

    _upload_file(client, "alice", "tenant-a", content=b"alice file 1")
    _upload_file(client, "alice", "tenant-a", content=b"alice file 2")
    _upload_file(client, "bob", "tenant-b", content=b"bob file 1")

    alice_list = client.get("/v1/files", headers=_headers("alice", "tenant-a"))
    assert alice_list.status_code == 200
    alice_files = alice_list.json()["data"]
    assert len(alice_files) == 2

    bob_list = client.get("/v1/files", headers=_headers("bob", "tenant-b"))
    assert bob_list.status_code == 200
    bob_files = bob_list.json()["data"]
    assert len(bob_files) == 1


def test_cross_tenant_file_content_isolation(multi_tenant_setup):
    """Cross-tenant content retrieval returns 404."""
    client, _ = multi_tenant_setup

    file_id = _upload_file(client, "alice", "tenant-a", content=b"secret content")

    bob_resp = client.get(f"/v1/files/{file_id}/content", headers=_headers("bob", "tenant-b"))
    assert bob_resp.status_code == 404

    alice_resp = client.get(f"/v1/files/{file_id}/content", headers=_headers("alice", "tenant-a"))
    assert alice_resp.status_code == 200


def test_multi_mode_rejects_missing_tenant_files(multi_tenant_setup, suppress_auth_errors):
    """In multi mode, a request with principal but no tenant header gets 401."""
    client, _ = multi_tenant_setup

    response = client.post(
        "/v1/files",
        files={"file": ("test.txt", b"hello", "text/plain")},
        data={"purpose": "assistants"},
        headers={"x-auth-user-id": "alice"},
    )
    assert response.status_code == 401


def test_same_tenant_different_users_files(multi_tenant_setup):
    """Within the same tenant, ABAC (user is owner) still applies.
    Each user sees only their own files."""
    client, _ = multi_tenant_setup

    alice_file = _upload_file(client, "alice", "tenant-a", content=b"alice data")
    bob_file = _upload_file(client, "bob", "tenant-a", content=b"bob data")

    alice_get_own = client.get(f"/v1/files/{alice_file}", headers=_headers("alice", "tenant-a"))
    assert alice_get_own.status_code == 200

    alice_get_bob = client.get(f"/v1/files/{bob_file}", headers=_headers("alice", "tenant-a"))
    assert alice_get_bob.status_code == 404

    bob_get_own = client.get(f"/v1/files/{bob_file}", headers=_headers("bob", "tenant-a"))
    assert bob_get_own.status_code == 200

    bob_get_alice = client.get(f"/v1/files/{alice_file}", headers=_headers("bob", "tenant-a"))
    assert bob_get_alice.status_code == 404
