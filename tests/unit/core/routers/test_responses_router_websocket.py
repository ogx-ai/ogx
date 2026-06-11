# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from unittest.mock import AsyncMock

from fastapi import FastAPI
from starlette.testclient import TestClient

from ogx.core.server.fastapi_router_registry import build_fastapi_router
from ogx_api import Api, Responses
from ogx_api.openai_responses import (
    OpenAIResponseObject,
    OpenAIResponseObjectStreamResponseCompleted,
    OpenAIResponseObjectStreamResponseCreated,
)

# WebSocket transport tests


def _ws_app(impl: Responses) -> FastAPI:
    app = FastAPI()
    router = build_fastapi_router(Api.responses, impl)
    assert router is not None
    app.include_router(router)
    return app


def _ws_response(response_id: str, status: str = "completed") -> OpenAIResponseObject:
    return OpenAIResponseObject(
        id=response_id,
        created_at=1234567890,
        model="test-model",
        object="response",
        output=[],
        status=status,
        store=False,
    )


def test_websocket_invalid_json_returns_error_envelope():
    impl = AsyncMock(spec=Responses)
    client = TestClient(_ws_app(impl))

    with client.websocket_connect("/v1/responses") as ws:
        ws.send_text("this is not json")
        event = ws.receive_json()

    assert event["type"] == "error"
    assert event["error"]["code"] == "invalid_json"
    impl.create_openai_response.assert_not_called()


def test_websocket_validation_error_returns_invalid_request():
    impl = AsyncMock(spec=Responses)
    client = TestClient(_ws_app(impl))

    with client.websocket_connect("/v1/responses") as ws:
        # Missing required model/input fields.
        ws.send_text(json.dumps({"type": "response.create"}))
        event = ws.receive_json()

    assert event["type"] == "error"
    assert event["error"]["code"] == "invalid_request"
    impl.create_openai_response.assert_not_called()


def test_websocket_unknown_previous_response_not_found():
    impl = AsyncMock(spec=Responses)
    client = TestClient(_ws_app(impl))

    with client.websocket_connect("/v1/responses") as ws:
        ws.send_text(
            json.dumps(
                {
                    "type": "response.create",
                    "model": "test",
                    "store": False,
                    "previous_response_id": "resp_does_not_exist",
                    "input": "continue please",
                }
            )
        )
        event = ws.receive_json()

    assert event["type"] == "error"
    assert event["status"] == 404
    assert event["error"]["code"] == "previous_response_not_found"
    # No inference is attempted for a connection-local cache miss.
    impl.create_openai_response.assert_not_called()


def test_websocket_impl_exception_returns_server_error():
    impl = AsyncMock(spec=Responses)
    impl.create_openai_response.side_effect = RuntimeError("boom")
    client = TestClient(_ws_app(impl))

    with client.websocket_connect("/v1/responses") as ws:
        ws.send_text(json.dumps({"type": "response.create", "model": "test", "input": "hi"}))
        event = ws.receive_json()

    assert event["type"] == "error"
    assert event["error"]["code"] == "server_error"


def test_websocket_streams_events_until_terminal():
    impl = AsyncMock(spec=Responses)

    async def _stream():
        yield OpenAIResponseObjectStreamResponseCreated(response=_ws_response("resp_ws_1"), sequence_number=0)
        yield OpenAIResponseObjectStreamResponseCompleted(response=_ws_response("resp_ws_1"), sequence_number=1)

    impl.create_openai_response.return_value = _stream()
    client = TestClient(_ws_app(impl))

    with client.websocket_connect("/v1/responses") as ws:
        ws.send_text(json.dumps({"type": "response.create", "model": "test", "input": "hi"}))
        first = ws.receive_json()
        second = ws.receive_json()

    assert first["type"] == "response.created"
    assert second["type"] == "response.completed"
    # The HTTP-only streaming discriminator must not leak into the request.
    sent_request = impl.create_openai_response.call_args.args[0]
    assert sent_request.stream is True
    assert not hasattr(sent_request, "type") or "type" not in sent_request.model_dump()
