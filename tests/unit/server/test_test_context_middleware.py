# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import os

import pytest
from fastapi import APIRouter, FastAPI
from starlette.testclient import TestClient

from ogx.core.server.server import ProviderDataMiddleware
from ogx.core.testing_context import get_test_context


@pytest.fixture
def app_with_middleware():
    """Create a minimal FastAPI app with ProviderDataMiddleware."""
    app = FastAPI()

    router = APIRouter()

    @router.get("/test-context")
    def get_current_test_context():
        return {"test_id": get_test_context()}

    app.include_router(router)
    app.add_middleware(ProviderDataMiddleware)

    return app


@pytest.fixture
def test_mode_env(monkeypatch):
    """Set environment variables required for test context extraction."""
    monkeypatch.setenv("OGX_TEST_INFERENCE_MODE", "replay")
    monkeypatch.setenv("OGX_TEST_STACK_CONFIG_TYPE", "server")


def test_middleware_returns_none_without_header(app_with_middleware, test_mode_env):
    """Without the provider data header, test context should be None."""
    client = TestClient(app_with_middleware)
    response = client.get("/test-context")

    assert response.status_code == 200
    assert response.json()["test_id"] is None


def test_middleware_extracts_test_id_from_header(app_with_middleware, test_mode_env):
    """With the provider data header containing __test_id, it should be extracted."""
    client = TestClient(app_with_middleware)

    provider_data = json.dumps({"__test_id": "test-abc-123"})
    response = client.get(
        "/test-context",
        headers={"X-OGX-Provider-Data": provider_data},
    )

    assert response.status_code == 200
    assert response.json()["test_id"] == "test-abc-123"


def test_middleware_handles_empty_provider_data(app_with_middleware, test_mode_env):
    """Empty provider data should result in None test context."""
    client = TestClient(app_with_middleware)

    response = client.get(
        "/test-context",
        headers={"X-OGX-Provider-Data": "{}"},
    )

    assert response.status_code == 200
    assert response.json()["test_id"] is None


def test_middleware_handles_invalid_json(app_with_middleware, test_mode_env):
    """Invalid JSON in header should not crash, test context should be None."""
    client = TestClient(app_with_middleware)

    response = client.get(
        "/test-context",
        headers={"X-OGX-Provider-Data": "not-valid-json"},
    )

    assert response.status_code == 200
    assert response.json()["test_id"] is None


def test_middleware_noop_without_test_mode(app_with_middleware):
    """Without test mode env vars, middleware should not extract test context."""
    # Ensure env vars are not set
    os.environ.pop("OGX_TEST_INFERENCE_MODE", None)
    os.environ.pop("OGX_TEST_STACK_CONFIG_TYPE", None)

    client = TestClient(app_with_middleware)

    provider_data = json.dumps({"__test_id": "test-abc-123"})
    response = client.get(
        "/test-context",
        headers={"X-OGX-Provider-Data": provider_data},
    )

    assert response.status_code == 200
    assert response.json()["test_id"] is None
