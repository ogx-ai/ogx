# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import logging  # allow-direct-logging

import pytest

from ogx_api.responses.fastapi_routes import create_sse_event, sse_generator


@pytest.fixture
def suppress_sse_errors(caplog):
    """Suppress expected ERROR logs for tests that deliberately trigger SSE errors"""
    caplog.set_level(logging.CRITICAL, logger="ogx_api.responses.fastapi_routes")


async def test_sse_generator_basic():
    async def event_gen():
        yield "Test event 1"
        yield "Test event 2"

    sse_gen = sse_generator(event_gen())
    assert sse_gen is not None

    seen_events = []
    async for event in sse_gen:
        seen_events.append(event)
    assert len(seen_events) == 2
    assert seen_events[0] == create_sse_event("Test event 1")
    assert seen_events[1] == create_sse_event("Test event 2")


async def test_sse_generator_client_disconnected():
    async def event_gen():
        yield "Test event 1"
        raise asyncio.CancelledError()

    sse_gen = sse_generator(event_gen())
    assert sse_gen is not None

    seen_events = []
    with pytest.raises(asyncio.CancelledError):
        async for event in sse_gen:
            seen_events.append(event)

    # We should see 1 event before the client disconnected
    assert len(seen_events) == 1
    assert seen_events[0] == create_sse_event("Test event 1")


async def test_sse_generator_client_disconnected_before_response_starts():
    async def event_gen():
        raise asyncio.CancelledError()
        yield  # make this an async generator  # noqa: E303

    sse_gen = sse_generator(event_gen())
    assert sse_gen is not None

    seen_events = []
    with pytest.raises(asyncio.CancelledError):
        async for event in sse_gen:
            seen_events.append(event)

    assert len(seen_events) == 0


async def test_sse_generator_error_before_response_starts(suppress_sse_errors):
    async def event_gen():
        raise Exception("Test error")
        yield  # make this an async generator  # noqa: E303

    sse_gen = sse_generator(event_gen())
    assert sse_gen is not None

    seen_events = []
    async for event in sse_gen:
        seen_events.append(event)

    # We should have 1 error event with the spec-compliant error format
    assert len(seen_events) == 1
    assert '"type":"error"' in seen_events[0]
    assert '"code":"server_error"' in seen_events[0]
    assert '"message":"Internal server error: An unexpected error occurred."' in seen_events[0]
    assert '"sequence_number":1' in seen_events[0]


async def test_create_sse_event_string():
    event = create_sse_event("hello")
    assert event == 'data: "hello"\n\n'


async def test_create_sse_event_dict():
    event = create_sse_event({"key": "value"})
    assert event == 'data: {"key": "value"}\n\n'
