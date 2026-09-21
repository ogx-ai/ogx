# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Tests for ogx_api.utils.sse_stream abandonment handling."""

import asyncio
from collections.abc import AsyncGenerator
from typing import cast

import anyio
import pytest
from fastapi import HTTPException
from fastapi.routing import APIRoute
from starlette.responses import StreamingResponse
from starlette.types import Message

from ogx_api import Inference, OpenAIChatCompletionRequestWithExtraBody
from ogx_api.inference.fastapi_routes import _format_inference_sse_error_event, create_router
from ogx_api.interactions.fastapi_routes import _format_google_sse_error_event
from ogx_api.messages.fastapi_routes import _format_anthropic_sse_error_event
from ogx_api.responses.fastapi_routes import sse_generator
from ogx_api.utils import create_sse_event, sse_stream


async def test_sse_stream_forwards_events() -> None:
    async def event_gen():
        yield "one"
        yield "two"

    seen = []
    async for event in sse_stream(event_gen(), create_sse_event, lambda e: f"error: {e}"):
        seen.append(event)
    assert seen == [create_sse_event("one"), create_sse_event("two")]


async def test_sse_stream_closes_inner_generator_on_aclose() -> None:
    """aclose (GeneratorExit) must propagate to the wrapped generator."""
    closed = asyncio.Event()

    async def event_gen():
        try:
            while True:
                yield "event"
                await asyncio.sleep(0)
        finally:
            closed.set()

    sse = sse_stream(event_gen(), create_sse_event, lambda e: f"error: {e}")
    assert await anext(sse) == create_sse_event("event")
    await sse.aclose()
    assert closed.is_set()


async def test_sse_stream_closes_inner_generator_on_cancellation() -> None:
    closed = asyncio.Event()
    release = asyncio.Event()

    async def event_gen():
        try:
            yield "first"
            while True:
                await release.wait()
                yield "event"
        finally:
            closed.set()

    sse = sse_stream(event_gen(), create_sse_event, lambda e: f"error: {e}")
    assert await anext(sse) == create_sse_event("first")
    task = asyncio.create_task(sse.__anext__())
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set()


async def test_sse_stream_reports_errors_as_events() -> None:
    async def event_gen():
        yield "ok"
        raise ValueError("boom")

    seen = []
    async for event in sse_stream(event_gen(), create_sse_event, lambda e: f"error: {e}"):
        seen.append(event)
    assert seen == [create_sse_event("ok"), "error: boom"]


async def test_inference_disconnect_finishes_upstream_cleanup() -> None:
    """Disconnect cancellation must not interrupt shielded upstream cleanup."""
    first_chunk_sent = anyio.Event()
    upstream_closed = anyio.Event()

    async def upstream() -> AsyncGenerator[str, None]:
        try:
            yield "first"
            await anyio.sleep_forever()
        finally:
            # HTTP transports shield their asynchronous socket close. A raw
            # asyncio task for each chunk bypasses AnyIO's task cancellation
            # tracking, allowing repeated cancellation to interrupt this close.
            with anyio.CancelScope(shield=True):
                await anyio.sleep(0)
                await anyio.sleep(0)
                upstream_closed.set()

    class InferenceImpl:
        """Provide a cancellable stream for the real inference route."""

        async def openai_chat_completion(
            self, _params: OpenAIChatCompletionRequestWithExtraBody
        ) -> AsyncGenerator[str, None]:
            return upstream()

    router = create_router(cast(Inference, InferenceImpl()))
    route = next(
        route for route in router.routes if isinstance(route, APIRoute) and route.name == "openai_chat_completion"
    )
    response = await route.endpoint(
        OpenAIChatCompletionRequestWithExtraBody(
            model="test", messages=[{"role": "user", "content": "hi"}], stream=True
        )
    )
    assert isinstance(response, StreamingResponse)

    async def receive() -> Message:
        await first_chunk_sent.wait()
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        if message["type"] == "http.response.body" and message.get("body"):
            first_chunk_sent.set()

    with anyio.fail_after(2):
        await response({"type": "http", "asgi": {"spec_version": "2.3"}}, receive, send)
    assert upstream_closed.is_set()


@pytest.mark.parametrize("protocol", ["inference", "responses", "messages", "interactions"])
@pytest.mark.parametrize("error_kind", ["unexpected", "http", "provider"])
async def test_sse_server_errors_hide_internal_details(
    protocol: str, error_kind: str, caplog: pytest.LogCaptureFixture
) -> None:
    private_detail = "private-backend-detail"

    class ProviderError(Exception):
        """Represent an upstream service failure with an HTTP status."""

        status_code = 503

    errors = {
        "unexpected": RuntimeError(private_detail),
        "http": HTTPException(status_code=500, detail=private_detail),
        "provider": ProviderError(private_detail),
    }
    events = await _collect_protocol_error(protocol, errors[error_kind])
    assert private_detail not in events
    assert "Internal server error: An unexpected error occurred." in events
    assert private_detail in caplog.text


@pytest.mark.parametrize("protocol", ["inference", "responses", "messages", "interactions"])
@pytest.mark.parametrize("error", [ValueError("invalid model"), HTTPException(status_code=404, detail="invalid model")])
async def test_sse_client_errors_keep_actionable_details(protocol: str, error: Exception) -> None:
    events = await _collect_protocol_error(protocol, error)
    assert "invalid model" in events


async def _collect_protocol_error(protocol: str, error: Exception) -> str:
    async def source() -> AsyncGenerator[str, None]:
        raise error
        yield "unreachable"

    if protocol == "responses":
        stream = sse_generator(source())
    else:
        formatter = {
            "inference": _format_inference_sse_error_event,
            "messages": _format_anthropic_sse_error_event,
            "interactions": _format_google_sse_error_event,
        }[protocol]
        stream = sse_stream(source(), create_sse_event, formatter)
    return "".join([event async for event in stream])
