# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Upload safety utilities for enforcing file size limits.

Provides a bounded-read helper that reads UploadFile objects in chunks,
aborting early when the configured maximum size is exceeded. This avoids
buffering arbitrarily large uploads into memory before rejecting them.
"""

import io

from fastapi import UploadFile
from starlette.datastructures import Headers

from ogx_api.common.errors import FileTooLargeError

# Default maximum upload size: 100 MiB
DEFAULT_MAX_UPLOAD_SIZE_BYTES: int = 100 * 1024 * 1024

# Chunk size for incremental reads (1 MiB)
_READ_CHUNK_SIZE: int = 1 * 1024 * 1024


async def read_upload_with_size_limit(
    file: UploadFile,
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_SIZE_BYTES,
) -> bytes:
    """Read an uploaded file with a size limit, aborting early if exceeded.

    Performs two levels of checking:
    1. Pre-check: if ``file.size`` is populated (from Content-Length), reject
       immediately without reading any bytes.
    2. Chunked read: reads the file in fixed-size chunks, tracking total bytes.
       Stops as soon as the cumulative size exceeds *max_upload_bytes*.

    Args:
        file: The FastAPI UploadFile to read.
        max_upload_bytes: Maximum allowed file size in bytes.

    Returns:
        The full file content as bytes.

    Raises:
        FileTooLargeError: If the file exceeds *max_upload_bytes*.
    """
    # Fast path: reject based on declared size when available.
    # Use getattr because not all file-like objects have .size (e.g. LibraryClientUploadFile).
    declared_size = getattr(file, "size", None)
    if declared_size is not None and declared_size > max_upload_bytes:
        raise FileTooLargeError(file_size=declared_size, max_size=max_upload_bytes)

    # Read in bounded chunks to avoid buffering unbounded data.
    # Some file-like objects (e.g. LibraryClientUploadFile) don't accept a size
    # argument to read(), so fall back to a single read() if chunked read fails.
    chunks: list[bytes] = []
    total_bytes = 0

    try:
        while True:
            chunk = await file.read(_READ_CHUNK_SIZE)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > max_upload_bytes:
                raise FileTooLargeError(file_size=total_bytes, max_size=max_upload_bytes)
            chunks.append(chunk)
    except TypeError:
        # file.read() doesn't accept a size argument — read all at once
        content: bytes = await file.read()
        if len(content) > max_upload_bytes:
            raise FileTooLargeError(file_size=len(content), max_size=max_upload_bytes) from None
        return content

    return b"".join(chunks)


class PreReadUploadFile(UploadFile):
    """UploadFile subclass backed by already-read bytes.

    Providers call ``await file.read()`` to get content. After the route handler
    has already performed a bounded read via :func:`read_upload_with_size_limit`,
    this wrapper lets providers consume the pre-read bytes without a second disk/
    network read.
    """

    def __init__(self, content: bytes, *, filename: str | None = None, content_type: str | None = None):
        super().__init__(
            file=io.BytesIO(content),
            filename=filename or "",
            size=len(content),
            headers=Headers({"content-type": content_type}) if content_type else None,
        )
