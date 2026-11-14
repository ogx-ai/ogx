# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import contextlib
import inspect
from collections.abc import AsyncIterator
from typing import Any

from ogx.log import get_logger
from ogx_api import OpenAIChatCompletionChunk, OpenAIChatCompletionChunkWithReasoning

log = get_logger(name=__name__, category="providers::utils")


async def close_async_stream(stream: Any) -> None:
    """Best-effort close of a stream, accepting both aclose() and close().

    Closing is what releases the underlying HTTP response back to its pool,
    so every wrapper that iterates another stream must run this when it
    finishes, fails, or is abandoned by its consumer.
    """
    close = getattr(stream, "aclose", None)
    if close is None:
        close = getattr(stream, "close", None)
    if close is None:
        return
    with contextlib.suppress(Exception):
        result = close()
        if inspect.isawaitable(result):
            await result


async def wrap_async_stream[T](stream: AsyncIterator[T]) -> AsyncIterator[T]:
    """
    Wrap an async stream to ensure it returns a proper AsyncIterator.
    """
    try:
        async for item in stream:
            yield item
    except Exception as e:
        log.error("Failed to iterate wrapped async stream", error=str(e))
        raise
    finally:
        await close_async_stream(stream)


async def wrap_reasoning_chunks(
    stream: AsyncIterator[OpenAIChatCompletionChunk],
) -> AsyncIterator[OpenAIChatCompletionChunkWithReasoning]:
    """Extract reasoning content from OpenAI chat chunks and close the stream.

    Shared by the vllm, bedrock and ollama providers, which previously each
    defined their own copy of this wrapper. Closes ``stream`` on normal
    completion, upstream error, and consumer abandon.
    """
    try:
        async for chunk in stream:
            reasoning = None
            for choice in chunk.choices or []:
                reasoning = getattr(choice.delta, "reasoning", None) or getattr(choice.delta, "reasoning_content", None)
            yield OpenAIChatCompletionChunkWithReasoning(
                chunk=chunk,
                reasoning_content=reasoning,
            )
    finally:
        await close_async_stream(stream)
