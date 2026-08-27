# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Shared utility functions for the OGX API."""

import asyncio
import contextvars
import json
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from typing import Any

from pydantic import BaseModel


def _preserve_context_for_sse(event_gen: AsyncGenerator[str, None]) -> AsyncGenerator[str, None]:
    """Preserve request context for SSE streaming.

    StreamingResponse runs in a different task, losing request contextvars.
    This wrapper captures and restores the context.
    """
    context = contextvars.copy_context()

    async def wrapper() -> AsyncGenerator[str, None]:
        try:
            while True:
                try:
                    task: asyncio.Task[str] = context.run(asyncio.create_task, event_gen.__anext__())
                    item = await task
                except StopAsyncIteration:
                    break
                yield item
        except (asyncio.CancelledError, GeneratorExit):
            await event_gen.aclose()
            raise

    return wrapper()


def _serialize_sse_data(data: Any) -> str:
    if isinstance(data, BaseModel):
        return data.model_dump_json()
    return json.dumps(data)


def create_sse_event(data: Any) -> str:
    """Create a Server-Sent Event string: data: <json>\\n\\n."""
    return f"data: {_serialize_sse_data(data)}\n\n"


def create_sse_event_with_type(event_type: str, data: Any) -> str:
    """Create a named Server-Sent Event string: event: <type>\\ndata: <json>\\n\\n."""
    return f"event: {event_type}\ndata: {_serialize_sse_data(data)}\n\n"


async def sse_stream(
    event_gen: AsyncIterator[Any],
    format_event: Callable[[Any], str],
    format_error_event: Callable[[Exception], str],
) -> AsyncGenerator[str, None]:
    """Yield SSE events from an async generator.

    Each item is serialized with ``format_event``. Cancellation closes the
    underlying generator. Any other exception is reported as the final event
    via ``format_error_event``, which should also log the exception.
    """
    try:
        async for item in event_gen:
            yield format_event(item)
    except asyncio.CancelledError:
        if hasattr(event_gen, "aclose"):
            await event_gen.aclose()
        raise
    except Exception as e:
        yield format_error_event(e)
