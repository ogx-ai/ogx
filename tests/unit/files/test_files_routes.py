# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import tempfile
import tracemalloc
from typing import Any

from fastapi import FastAPI, UploadFile
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from ogx.core.library_client import LibraryClientUploadFile
from ogx.core.server.server import global_exception_handler
from ogx_api.files.fastapi_routes import create_router
from ogx_api.files.models import OpenAIFileObject, OpenAIFilePurpose, OpenAIFileUploadPurpose, UploadFileRequest

_MIB = 1024 * 1024


class _StreamingFiles:
    """Files impl that consumes the upload in 1 MiB chunks, like a streaming provider."""

    def __init__(self) -> None:
        self.received = bytearray()
        self.calls = 0

    async def openai_upload_file(self, request: UploadFileRequest, file: UploadFile) -> OpenAIFileObject:
        self.calls += 1
        size = 0
        while chunk := await file.read(_MIB):
            size += len(chunk)
            if size <= _MIB:
                self.received += chunk
        return OpenAIFileObject(
            id="file-1",
            bytes=size,
            created_at=0,
            expires_at=None,
            filename=file.filename or "",
            purpose=OpenAIFilePurpose(request.purpose.value),
            status="processed",
            status_details="",
        )


def _client(impl: _StreamingFiles, max_upload_size_bytes: int) -> TestClient:
    app = FastAPI()
    app.include_router(create_router(impl, max_upload_size_bytes=max_upload_size_bytes))  # type: ignore[arg-type]  # minimal stand-in that only implements uploads
    app.exception_handler(Exception)(global_exception_handler)
    return TestClient(app, raise_server_exceptions=False)


def _upload_endpoint(impl: _StreamingFiles, max_upload_size_bytes: int) -> Any:
    router = create_router(impl, max_upload_size_bytes=max_upload_size_bytes)  # type: ignore[arg-type]  # minimal stand-in that only implements uploads
    return next(
        route.endpoint
        for route in router.routes
        if isinstance(route, APIRoute) and route.path == "/v1/files" and "POST" in route.methods
    )


def test_upload_within_limit_reaches_provider() -> None:
    impl = _StreamingFiles()

    response = _client(impl, max_upload_size_bytes=10).post(
        "/v1/files", files={"file": ("a.txt", b"0123456789")}, data={"purpose": "assistants"}
    )

    assert response.status_code == 200
    assert response.json()["bytes"] == 10
    assert bytes(impl.received) == b"0123456789"


def test_upload_over_limit_is_rejected_with_413_before_provider() -> None:
    impl = _StreamingFiles()

    response = _client(impl, max_upload_size_bytes=10).post(
        "/v1/files", files={"file": ("a.txt", b"0123456789A")}, data={"purpose": "assistants"}
    )

    assert response.status_code == 413
    assert response.json()["error"]["message"].startswith(
        "File size 11 bytes exceeds the maximum allowed upload size of 10 bytes"
    )
    assert impl.calls == 0


async def test_library_client_upload_reaches_provider() -> None:
    """The library client calls the endpoint directly with an in-memory upload instead of a multipart-parsed one."""
    impl = _StreamingFiles()
    endpoint = _upload_endpoint(impl, max_upload_size_bytes=10)

    result = await endpoint(
        file=LibraryClientUploadFile("a.txt", b"0123456789"),
        purpose=OpenAIFileUploadPurpose.ASSISTANTS,
        expires_after=None,
    )

    assert result.bytes == 10
    assert bytes(impl.received) == b"0123456789"


async def test_upload_is_streamed_to_provider_without_buffering() -> None:
    """Router memory must not grow with the upload size: the provider reads the spooled request file directly."""
    size = 32 * _MIB
    # Starlette's multipart parser hands the route a SpooledTemporaryFile that has rolled to disk
    spooled = tempfile.SpooledTemporaryFile(max_size=_MIB)
    for _ in range(size // _MIB):
        spooled.write(b"x" * _MIB)
    spooled.seek(0)
    upload = UploadFile(file=spooled, size=size, filename="big.bin")
    impl = _StreamingFiles()
    endpoint = _upload_endpoint(impl, max_upload_size_bytes=size)

    tracemalloc.start()
    try:
        result = await endpoint(file=upload, purpose=OpenAIFileUploadPurpose.ASSISTANTS, expires_after=None)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
        spooled.close()

    assert result.bytes == size
    assert peak < 8 * _MIB
