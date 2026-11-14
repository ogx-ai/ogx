# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Shared utility functions for the OGX API."""

import asyncio
import json
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from typing import Any

from pydantic import BaseModel

from ogx_api.router_utils import try_translate_to_http_exception


def get_sse_error_message(exc: Exception) -> str:
    """Preserve client error details while hiding server exception messages."""
    http_exc = try_translate_to_http_exception(exc)
    if http_exc is not None and 400 <= http_exc.status_code < 500:
        return str(http_exc.detail)
    return "Internal server error: An unexpected error occurred."


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

    Each item is serialized with ``format_event``. Cancellation or
    abandonment closes the underlying generator. Any other exception is
    reported as the final event via ``format_error_event``, which should
    also log the exception.
    """
    try:
        async for item in event_gen:
            yield format_event(item)
    except (asyncio.CancelledError, GeneratorExit):
        if hasattr(event_gen, "aclose"):
            await event_gen.aclose()
        raise
    except Exception as e:
        yield format_error_event(e)
