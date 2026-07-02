# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ogx.core.server.server import ClientVersionMiddleware


def _client(monkeypatch, api_version: str = "1.1.4.dev0") -> tuple[TestClient, list[str]]:
    requested_packages: list[str] = []

    def fake_parse_version(package_name: str) -> str:
        requested_packages.append(package_name)
        return api_version

    monkeypatch.setattr("ogx.core.server.server.parse_version", fake_parse_version)

    app = FastAPI()
    app.add_middleware(ClientVersionMiddleware)

    @app.get("/test")
    def test_endpoint() -> dict[str, bool]:
        return {"ok": True}

    return TestClient(app), requested_packages


def test_client_version_uses_ogx_api_package(monkeypatch):
    client, requested_version_packages = _client(monkeypatch)

    response = client.get("/test", headers={"x-ogx-client-version": "1.1.1.dev0"})

    assert response.status_code == 200
    assert requested_version_packages == ["ogx-api"]


def test_client_version_rejects_incompatible_major_minor(monkeypatch):
    client, _ = _client(monkeypatch, api_version="1.1.4.dev0")

    response = client.get("/test", headers={"x-ogx-client-version": "1.2.0"})

    assert response.status_code == 426
    assert response.json()["error"]["message"] == (
        "Client version 1.2.0 is not compatible with server version 1.1.4.dev0. Please update your client."
    )
