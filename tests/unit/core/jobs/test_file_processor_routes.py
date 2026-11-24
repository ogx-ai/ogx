# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from io import BytesIO

import pytest
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.testclient import TestClient

from ogx_api.common.job_types import JobStatus
from ogx_api.file_processors import (
    FileProcessors,
    ListProcessFileJobsResponse,
    ProcessFileJob,
    ProcessFileRequest,
    ProcessFileResponse,
)
from ogx_api.file_processors.fastapi_routes import _build_request_and_file, create_router
from ogx_api.vector_io import Chunk


class _LegacyFileProcessor:
    """Provider SDK implementation from before job lifecycle methods existed."""

    async def process_file(self, request, file=None):
        return ProcessFileResponse(
            chunks=[Chunk(content="ok", chunk_id="chunk-1", chunk_metadata={})],
            metadata={"provider": "legacy", "options": request.options},
        )


class _JobFileProcessor:
    """Provider SDK implementation with job lifecycle methods, capturing the last request."""

    def __init__(self) -> None:
        self.last_request: ProcessFileRequest | None = None

    async def create_process_file_job(self, request, file=None):
        self.last_request = request
        return ProcessFileJob(job_id="job-1", status=JobStatus.scheduled, created_at=0)

    async def list_process_file_jobs(self, after=None, limit=100):
        return ListProcessFileJobsResponse(data=[])

    async def retrieve_process_file_job(self, job_id):
        return ProcessFileJob(job_id=job_id, status=JobStatus.scheduled, created_at=0)

    async def cancel_process_file_job(self, job_id):
        return ProcessFileJob(job_id=job_id, status=JobStatus.cancelled, created_at=0)


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


def test_file_processor_route_parses_options_json() -> None:
    provider = _JobFileProcessor()
    app = FastAPI()
    app.include_router(create_router(provider))

    response = TestClient(app).post(
        "/v1alpha/file-processors/jobs",
        files={"file": ("input.txt", b"payload")},
        data={"options": json.dumps({"use_markdown_tables": True})},
    )

    assert response.status_code == 200
    assert provider.last_request is not None
    assert provider.last_request.options == {"use_markdown_tables": True}


@pytest.mark.parametrize("options", ["not JSON", "[]", '"text"'])
def test_file_processor_route_rejects_invalid_options_json(options: str) -> None:
    app = FastAPI()
    app.include_router(create_router(_JobFileProcessor()))

    response = TestClient(app).post(
        "/v1alpha/file-processors/jobs",
        files={"file": ("input.txt", b"payload")},
        data={"options": options},
    )

    assert response.status_code == 422


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
