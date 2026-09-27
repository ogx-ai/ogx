# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import contextvars
from collections.abc import AsyncGenerator


def preserve_context_for_sse(event_gen: AsyncGenerator[str, None]) -> AsyncGenerator[str, None]:
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
