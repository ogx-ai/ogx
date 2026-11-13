# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Upload limit utilities for enforcing file size limits.

An upload's size is measured by seeking to the end of its file, so it is checked
against the limit without reading it and without trusting the declared
``UploadFile.size``.
"""

import io

from fastapi import UploadFile
from starlette.datastructures import Headers

from ogx_api.common.errors import FileTooLargeError

# Default maximum upload size: 100 MiB
DEFAULT_MAX_UPLOAD_SIZE_BYTES: int = 100 * 1024 * 1024


def enforce_upload_size_limit(file: UploadFile, max_upload_bytes: int = DEFAULT_MAX_UPLOAD_SIZE_BYTES) -> None:
    """Reject an upload larger than *max_upload_bytes* without reading it.

    The size is the number of bytes actually received, not the declared
    ``file.size``. The file is left at its start, ready to be streamed.

    Args:
        file: The FastAPI UploadFile to check.
        max_upload_bytes: Maximum allowed file size in bytes.

    Raises:
        FileTooLargeError: If the file exceeds *max_upload_bytes*.
    """
    size = file.file.seek(0, io.SEEK_END)
    file.file.seek(0)
    if size > max_upload_bytes:
        raise FileTooLargeError(file_size=size, max_size=max_upload_bytes)


async def read_upload_with_size_limit(
    file: UploadFile,
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_SIZE_BYTES,
) -> bytes:
    """Read an uploaded file, rejecting it before any read if it exceeds *max_upload_bytes*.

    Args:
        file: The FastAPI UploadFile to read.
        max_upload_bytes: Maximum allowed file size in bytes.

    Returns:
        The full file content as bytes.

    Raises:
        FileTooLargeError: If the file exceeds *max_upload_bytes*.
    """
    enforce_upload_size_limit(file, max_upload_bytes)
    content: bytes = await file.read()
    return content


class PreReadUploadFile(UploadFile):
    """UploadFile backed by bytes that are already in memory.

    Used wherever an upload is built in process rather than parsed from a
    multipart request: after :func:`read_upload_with_size_limit`, by the library
    client, and for files that providers create themselves.
    """

    def __init__(self, content: bytes, *, filename: str | None = None, content_type: str | None = None):
        super().__init__(
            file=io.BytesIO(content),
            filename=filename or "",
            size=len(content),
            headers=Headers({"content-type": content_type}) if content_type else None,
        )
