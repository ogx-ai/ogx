# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from collections.abc import AsyncGenerator

import pytest
from fastapi.responses import StreamingResponse

from ogx.core.library_client import _route_call_in_process
from ogx.core.request_headers import PROVIDER_DATA_VAR
from ogx.core.server.routes import RouteAuthInfo, RouteImpls


async def test_async_streaming_preserves_provider_data_context() -> None:
    """Provider data must survive lazy iteration of the SSE stream.

    AsyncOGXAsLibraryClient.call_api returns a response whose stream is
    consumed after request_provider_data_context has exited, so the body
    iterator must re-enter the captured context on each chunk.
    """

    async def endpoint() -> StreamingResponse:
        async def gen() -> AsyncGenerator[str, None]:
            yield json.dumps({"provider_data": PROVIDER_DATA_VAR.get()})

        return StreamingResponse(gen(), media_type="text/event-stream")

    route_impls: RouteImpls = {
        "post": {
            r"^/v1/test$": (endpoint, "/v1/test", RouteAuthInfo()),
        }
    }

    rest_response = await _route_call_in_process(
        method="POST",
        url="http://localhost/v1/test",
        header_params=None,
        body=None,
        post_params=None,
        route_impls=route_impls,
        provider_data={"provider": "test"},
        sanitize_headers=lambda headers: dict(headers or {}),
        convert_body=lambda _func, body, **_kwargs: body,
        async_streaming=True,
    )

    assert PROVIDER_DATA_VAR.get() is None

    payload = b"".join([chunk async for chunk in rest_response.response.stream]).decode("utf-8")
    assert '"provider_data": {"provider": "test"}' in payload


@pytest.mark.parametrize("as_iterable", [False, True])
async def test_async_streaming_close_closes_body_in_provider_context(as_iterable: bool) -> None:
    closed_in_context = []

    async def endpoint() -> StreamingResponse:
        async def gen() -> AsyncGenerator[str, None]:
            try:
                yield "first"
                yield "second"
            finally:
                closed_in_context.append(PROVIDER_DATA_VAR.get())

        class SSEIterable:
            """Return a separate iterator, as permitted by StreamingResponse."""

            def __aiter__(self) -> AsyncGenerator[str, None]:
                return gen()

        return StreamingResponse(SSEIterable() if as_iterable else gen(), media_type="text/event-stream")

    rest_response = await _route_call_in_process(
        method="POST",
        url="http://localhost/v1/test",
        header_params=None,
        body=None,
        post_params=None,
        route_impls={"post": {r"^/v1/test$": (endpoint, "/v1/test", RouteAuthInfo())}},
        provider_data={"provider": "test"},
        sanitize_headers=lambda headers: dict(headers or {}),
        convert_body=lambda _func, body, **_kwargs: body,
        async_streaming=True,
    )

    response = rest_response.response
    iterator = response.aiter_bytes()
    assert await anext(iterator) == b"first"
    await response.aclose()
    assert closed_in_context == [{"provider": "test"}]
    assert PROVIDER_DATA_VAR.get() is None
