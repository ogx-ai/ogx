# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import logging  # allow-direct-logging
from uuid import uuid4

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

from ogx.core.datatypes import QuotaConfig, QuotaPeriod
from ogx.core.server.quota import QuotaMiddleware
from ogx.core.storage.datatypes import KVStoreReference, SqliteKVStoreConfig
from ogx.core.storage.kvstore import register_kvstore_backends


@pytest.fixture
def suppress_quota_warnings(caplog):
    """Suppress expected WARNING logs for SQLite backend and quota exceeded"""
    caplog.set_level(logging.CRITICAL, logger="ogx.core.server.quota")


class InjectClientIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware that injects 'authenticated_client_id' to mimic AuthenticationMiddleware.
    """

    def __init__(self, app, client_id="client1"):
        super().__init__(app)
        self.client_id = client_id

    async def dispatch(self, request: Request, call_next):
        request.scope["authenticated_client_id"] = self.client_id
        return await call_next(request)


def build_quota_config(db_path) -> QuotaConfig:
    backend_name = f"kv_quota_{uuid4().hex}"
    register_kvstore_backends({backend_name: SqliteKVStoreConfig(db_path=str(db_path))})
    return QuotaConfig(
        kvstore=KVStoreReference(backend=backend_name, namespace="quota"),
        anonymous_max_requests=1,
        authenticated_max_requests=2,
        period=QuotaPeriod.DAY,
    )


@pytest.fixture
def auth_app(tmp_path, request):
    """
    FastAPI app with InjectClientIDMiddleware and QuotaMiddleware for authenticated testing.
    Each test gets its own DB file.
    """
    inner_app = FastAPI()

    @inner_app.get("/test")
    async def test_endpoint():
        return {"message": "ok"}

    db_path = tmp_path / f"quota_{request.node.name}.db"
    quota = build_quota_config(db_path)

    app = InjectClientIDMiddleware(
        QuotaMiddleware(
            inner_app,
            kv_config=quota.kvstore,
            anonymous_max_requests=quota.anonymous_max_requests,
            authenticated_max_requests=quota.authenticated_max_requests,
            window_seconds=86400,
        ),
        client_id=f"client_{request.node.name}",
    )
    return app


def test_authenticated_quota_allows_up_to_limit(auth_app, suppress_quota_warnings):
    client = TestClient(auth_app)
    assert client.get("/test").status_code == 200
    assert client.get("/test").status_code == 200


def test_authenticated_quota_blocks_after_limit(auth_app, suppress_quota_warnings):
    client = TestClient(auth_app)
    client.get("/test")
    client.get("/test")
    resp = client.get("/test")
    assert resp.status_code == 429
    assert resp.json()["error"]["message"] == "Quota exceeded"


def test_anonymous_quota_allows_up_to_limit(tmp_path, request, suppress_quota_warnings):
    inner_app = FastAPI()

    @inner_app.get("/test")
    async def test_endpoint():
        return {"message": "ok"}

    db_path = tmp_path / f"quota_anon_{request.node.name}.db"
    quota = build_quota_config(db_path)

    app = QuotaMiddleware(
        inner_app,
        kv_config=quota.kvstore,
        anonymous_max_requests=quota.anonymous_max_requests,
        authenticated_max_requests=quota.authenticated_max_requests,
        window_seconds=86400,
    )

    client = TestClient(app)
    assert client.get("/test").status_code == 200


def test_anonymous_quota_blocks_after_limit(tmp_path, request, suppress_quota_warnings):
    inner_app = FastAPI()

    @inner_app.get("/test")
    async def test_endpoint():
        return {"message": "ok"}

    db_path = tmp_path / f"quota_anon_{request.node.name}.db"
    quota = build_quota_config(db_path)

    app = QuotaMiddleware(
        inner_app,
        kv_config=quota.kvstore,
        anonymous_max_requests=quota.anonymous_max_requests,
        authenticated_max_requests=quota.authenticated_max_requests,
        window_seconds=86400,
    )

    client = TestClient(app)
    client.get("/test")
    resp = client.get("/test")
    assert resp.status_code == 429
    assert resp.json()["error"]["message"] == "Quota exceeded"
