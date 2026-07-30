# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock

from fastapi import FastAPI
from starlette.testclient import TestClient

from ogx.core.server.fastapi_router_registry import build_fastapi_router
from ogx_api import Api
from ogx_api.conversations import Conversations


def test_consecutive_errors_keep_connection_alive():
    """Route-level ExceptionTranslatingRoute prevents connection resets on repeated errors.

    Without the route class, unhandled exceptions reach ServerErrorMiddleware which
    can close the TCP transport on Linux (RST). This test verifies the conversations
    router has ExceptionTranslatingRoute applied by sending two requests on the same
    connection and confirming both requests get proper JSON responses.
    """
    app = FastAPI()
    impl = AsyncMock(spec=Conversations)
    impl.get_conversation.side_effect = ValueError("bad request")

    router = build_fastapi_router(Api.conversations, impl)
    assert router is not None
    app.include_router(router)

    client = TestClient(app, raise_server_exceptions=False)

    resp1 = client.get("/v1/conversations/conv_abc")
    assert resp1.status_code == 400

    resp2 = client.get("/v1/conversations/conv_abc")
    assert resp2.status_code == 400
    assert resp2.json()["detail"] == "bad request"
