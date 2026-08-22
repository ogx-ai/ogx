# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from io import BytesIO

import pytest
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.testclient import TestClient

from ogx_api.file_processors import FileProcessors, ProcessFileResponse
from ogx_api.file_processors.fastapi_routes import _build_request_and_file, create_router
from ogx_api.vector_io import Chunk


class _LegacyFileProcessor:
    """Provider SDK implementation from before job lifecycle methods existed."""

    async def process_file(self, request, file=None):
        return ProcessFileResponse(
            chunks=[Chunk(content="ok", chunk_id="chunk-1", chunk_metadata={})],
            metadata={"provider": "legacy"},
        )


def test_legacy_provider_still_satisfies_file_processors_protocol() -> None:
    assert isinstance(_LegacyFileProcessor(), FileProcessors)


def test_legacy_provider_job_route_returns_not_implemented() -> None:
    app = FastAPI()
    app.include_router(create_router(_LegacyFileProcessor()))

    response = TestClient(app, raise_server_exceptions=False).post(
        "/v1alpha/file-processors/jobs",
        data={"file_id": "file-existing"},
    )

    assert response.status_code == 501
    assert "job execution" in response.json()["detail"].lower()


def test_legacy_provider_job_route_rejects_before_reading_upload() -> None:
    app = FastAPI()
    app.include_router(create_router(_LegacyFileProcessor(), max_upload_size_bytes=1))

    response = TestClient(app, raise_server_exceptions=False).post(
        "/v1alpha/file-processors/jobs",
        files={"file": ("input.txt", b"larger than the configured limit")},
    )

    assert response.status_code == 501


@pytest.mark.parametrize(
    ("file", "file_id"),
    [
        (None, None),
        (UploadFile(filename="input.txt", file=BytesIO(b"payload")), "file-existing"),
    ],
)
async def test_file_processor_routes_require_exactly_one_input(
    file: UploadFile | None,
    file_id: str | None,
) -> None:
    with pytest.raises(HTTPException, match="Exactly one") as exc_info:
        await _build_request_and_file(file, file_id, None, None, max_upload_size_bytes=1024)

    assert exc_info.value.status_code == 400
    if file is not None:
        assert file.file.tell() == 0
