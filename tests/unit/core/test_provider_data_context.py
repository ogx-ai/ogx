# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import json
from contextlib import contextmanager
from contextvars import ContextVar

from ogx.core.utils.context import preserve_contexts_async_generator

# Define provider data context variable and context manager locally
PROVIDER_DATA_VAR = ContextVar("provider_data", default=None)


@contextmanager
def request_provider_data_context(headers):
    val = headers.get("X-OGX-Provider-Data")
    provider_data = json.loads(val) if val else {}
    token = PROVIDER_DATA_VAR.set(provider_data)
    try:
        yield
    finally:
        PROVIDER_DATA_VAR.reset(token)


def create_sse_event(data):
    return f"data: {json.dumps(data)}\n\n"


async def sse_generator(event_gen_coroutine):
    event_gen = await event_gen_coroutine
    async for item in event_gen:
        yield create_sse_event(item)
        await asyncio.sleep(0)


async def async_event_gen():
    async def event_gen():
        yield PROVIDER_DATA_VAR.get()

    return event_gen()


async def test_provider_data_context_cleared_between_sse_requests():
    headers = {"X-OGX-Provider-Data": json.dumps({"api_key": "abc"})}
    with request_provider_data_context(headers):
        gen1 = preserve_contexts_async_generator(sse_generator(async_event_gen()), [PROVIDER_DATA_VAR])

    events1 = [event async for event in gen1]
    assert events1 == [create_sse_event({"api_key": "abc"})]
    assert PROVIDER_DATA_VAR.get() is None

    gen2 = preserve_contexts_async_generator(sse_generator(async_event_gen()), [PROVIDER_DATA_VAR])
    events2 = [event async for event in gen2]
    assert events2 == [create_sse_event(None)]
    assert PROVIDER_DATA_VAR.get() is None
