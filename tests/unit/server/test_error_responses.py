# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from collections.abc import Iterator
from unittest.mock import AsyncMock

import pytest
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from ogx.core.datatypes import (
    AuthenticationConfig,
    AuthProviderType,
    CustomAuthConfig,
    TenancyConfig,
    TenancyMode,
)
from ogx.core.server.auth import AuthenticationMiddleware, RouteAuthorizationMiddleware, TenancyMiddleware
from ogx.core.server.server import _send_error_response, create_app, register_exception_handlers
from ogx_api.common.errors import OpenAIErrorResponse, openai_error_type_for_status
from ogx_api.inference.fastapi_routes import _format_inference_sse_error_event

AUTH_CONFIG = AuthenticationConfig(
    provider_config=CustomAuthConfig(type=AuthProviderType.CUSTOM, endpoint="http://mock-auth/validate"),
    access_policy=[],
)

MINIMAL_CONFIG = {
    "version": 2,
    "distro_name": "test",
    "apis": [],
    "providers": {},
    "storage": {
        "backends": {
            "kv_default": {"type": "kv_sqlite", "db_path": ":memory:"},
            "sql_default": {"type": "sql_sqlite", "db_path": ":memory:"},
        },
        "stores": {
            "metadata": {"backend": "kv_default", "namespace": "registry"},
            "inference": {"backend": "sql_default", "table_name": "inference_store"},
            "conversations": {"backend": "sql_default", "table_name": "conversations"},
            "prompts": {"backend": "kv_default", "namespace": "prompts"},
        },
    },
}


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()

    @app.get("/v1/registered")
    async def registered() -> dict[str, str]:
        return {"ok": "yes"}

    @app.get("/v1/teapot")
    async def teapot() -> None:
        raise HTTPException(status_code=418, detail="I'm a teapot")

    @app.post("/v1alpha/interactions/boom")
    async def interactions_boom() -> None:
        raise HTTPException(status_code=404, detail="No such interaction")

    @app.get("/v1/kaboom")
    async def kaboom() -> None:
        raise ValueError("bad value")

    register_exception_handlers(app)
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def server_client(tmp_path, monkeypatch) -> Iterator[TestClient]:
    """A client for the real application, built the way `ogx run` builds it.

    Nothing here stands in for the handler registration under test — `create_app` does
    it, so deleting that line fails these tests.
    """
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(MINIMAL_CONFIG))
    monkeypatch.setenv("OGX_CONFIG", str(config_file))
    monkeypatch.setenv("OGX_DISABLE_VERSION_CHECK", "1")

    with TestClient(create_app(), raise_server_exceptions=False) as client:
        yield client


def test_real_app_returns_openai_error_shape_for_unrouted_path(server_client: TestClient) -> None:
    """The 404 an OpenAI client hits against a partially configured stack."""
    response = server_client.get("/v1/no-such-endpoint")

    assert response.status_code == 404
    assert response.json() == {"error": {"message": "Not Found", "type": "invalid_request_error"}}


def test_real_app_keeps_the_google_envelope_for_unrouted_interactions_paths(server_client: TestClient) -> None:
    response = server_client.post("/v1alpha/interactions/nope:generate")

    assert response.status_code == 404
    assert response.json() == {"error": {"code": 404, "message": "Not Found"}}


def test_unregistered_path_returns_openai_error_shape(client: TestClient) -> None:
    response = client.get("/v1/conversations")

    assert response.status_code == 404
    assert response.json() == {"error": {"message": "Not Found", "type": "invalid_request_error"}}


def test_unsupported_method_returns_openai_error_shape(client: TestClient) -> None:
    response = client.post("/v1/registered")

    assert response.status_code == 405
    assert response.json() == {"error": {"message": "Method Not Allowed", "type": "invalid_request_error"}}
    # Starlette sets Allow on the exception it raises; the handler must not drop it.
    assert response.headers["allow"] == "GET"


def test_handler_http_exception_returns_openai_error_shape(client: TestClient) -> None:
    response = client.get("/v1/teapot")

    assert response.status_code == 418
    assert response.json() == {"error": {"message": "I'm a teapot", "type": "invalid_request_error"}}


def test_interactions_paths_keep_the_google_error_envelope(client: TestClient) -> None:
    response = client.post("/v1alpha/interactions/boom")

    assert response.status_code == 404
    assert response.json() == {"error": {"code": 404, "message": "No such interaction"}}


def test_translated_exceptions_carry_an_error_type(client: TestClient) -> None:
    response = client.get("/v1/kaboom")

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_inference_sse_error_event_carries_an_error_type() -> None:
    event = _format_inference_sse_error_event(ValueError("bad value"))

    payload = json.loads(event.removeprefix("data: ").strip())

    assert payload["error"]["type"] == "invalid_request_error"
    assert payload["error"]["code"] == "400"


async def _capture_body(coro) -> dict:
    """Run an ASGI error sender and return the JSON body it wrote."""
    messages: list[dict] = []

    async def send(message: dict) -> None:
        messages.append(message)

    await coro(send)
    return json.loads(next(m["body"] for m in messages if m["type"] == "http.response.body"))


@pytest.mark.parametrize(
    "sender,expected_type",
    [
        (
            lambda send: AuthenticationMiddleware(AsyncMock(), AUTH_CONFIG)._send_auth_error(send, "nope", 401),
            "invalid_request_error",
        ),
        (
            lambda send: RouteAuthorizationMiddleware(AsyncMock(), [])._send_error(send, "nope", 403),
            "invalid_request_error",
        ),
        (
            lambda send: TenancyMiddleware(AsyncMock(), TenancyConfig(mode=TenancyMode.MULTI))._send_error(
                send, "nope"
            ),
            "invalid_request_error",
        ),
        (lambda send: _send_error_response(send, 503, "nope"), "server_error"),
    ],
    ids=["authentication", "route_authorization", "tenancy", "asgi_helper"],
)
async def test_asgi_error_senders_carry_an_error_type(sender, expected_type: str) -> None:
    """The middlewares write their bodies directly, bypassing the exception handlers."""
    body = await _capture_body(sender)

    assert body["error"] == {"message": "nope", "type": expected_type}


@pytest.mark.parametrize(
    "status_code,expected",
    [
        (400, "invalid_request_error"),
        (401, "invalid_request_error"),
        (404, "invalid_request_error"),
        (429, "rate_limit_error"),
        (500, "server_error"),
        (503, "server_error"),
    ],
)
def test_error_type_for_status(status_code: int, expected: str) -> None:
    assert openai_error_type_for_status(status_code) == expected
    assert OpenAIErrorResponse.for_status(status_code, "boom").error.type == expected
