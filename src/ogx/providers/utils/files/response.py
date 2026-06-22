# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from fastapi import Response
from fastapi.responses import StreamingResponse


async def response_body_bytes(response: Response | Any) -> bytes:
    """Read bytes from regular or streaming FastAPI responses."""
    body = getattr(response, "body", None)
    if body is not None:
        return bytes(body)

    if isinstance(response, StreamingResponse):
        chunks: list[bytes] = []
        async for chunk in response.body_iterator:
            if isinstance(chunk, str):
                chunks.append(chunk.encode("utf-8"))
            elif isinstance(chunk, memoryview):
                chunks.append(bytes(chunk))
            else:
                chunks.append(chunk)
        return b"".join(chunks)

    raise ValueError("Failed to read response body bytes")
