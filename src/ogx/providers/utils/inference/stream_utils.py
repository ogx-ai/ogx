# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncIterator

from ogx.log import get_logger

log = get_logger(name=__name__, category="providers::utils")


async def wrap_async_stream[T](stream: AsyncIterator[T]) -> AsyncIterator[T]:
    """
    Wrap an async stream to ensure it returns a proper AsyncIterator.
    """
    try:
        async for item in stream:
            yield item
    except Exception as e:
        log.error(f"Error in wrapped async stream: {e}")
        raise
