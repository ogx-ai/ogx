# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""FastAPI router for the File Processors API.

This module defines the FastAPI router for the File Processors API using standard
FastAPI route decorators. The router is defined in the API package to keep
all API-related code together.
"""

import json
from typing import Annotated, Any, Protocol, cast

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from pydantic import ValidationError

from ogx_api.common.upload_limits import (
    DEFAULT_MAX_UPLOAD_SIZE_BYTES,
    PreReadUploadFile,
    read_upload_with_size_limit,
)
from ogx_api.router_utils import standard_responses
from ogx_api.vector_io import (
    VectorStoreChunkingStrategy,
    VectorStoreChunkingStrategyAuto,
    VectorStoreChunkingStrategyStatic,
)
from ogx_api.version import OGX_API_V1ALPHA

from .api import FileProcessors
from .models import (
    ListProcessFileJobsResponse,
    ProcessFileJob,
    ProcessFileRequest,
    ProcessFileResponse,
)


def _parse_chunking_strategy(chunking_strategy: str | None) -> VectorStoreChunkingStrategy | None:
    """Parse and validate the multipart chunking_strategy JSON string."""
    if not chunking_strategy:
        return None
    try:
        chunking_data = json.loads(chunking_strategy)
        if not isinstance(chunking_data, dict):
            raise HTTPException(
                status_code=400,
                detail="chunking_strategy must be a JSON object, not a list, string, or other type",
            )
        if chunking_data.get("type") == "auto":
            return VectorStoreChunkingStrategyAuto.model_validate(chunking_data)
        if chunking_data.get("type") == "static":
            return VectorStoreChunkingStrategyStatic.model_validate(chunking_data)
        raise HTTPException(status_code=400, detail=f"Invalid chunking strategy type: {chunking_data.get('type')}")
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in chunking_strategy: {str(e)}") from e
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=f"Invalid chunking strategy: {str(e)}") from e


async def _build_request_and_file(
    file: UploadFile | None,
    file_id: str | None,
    options: dict[str, Any] | None,
    chunking_strategy: str | None,
    max_upload_size_bytes: int,
) -> tuple[ProcessFileRequest, UploadFile | None]:
    """Shared multipart handling for the blocking and job-based submit endpoints."""
    if (file is None) == (file_id is None):
        raise HTTPException(status_code=400, detail="Exactly one of file or file_id must be provided.")

    parsed_chunking_strategy = _parse_chunking_strategy(chunking_strategy)

    safe_file: UploadFile | None = None
    if file is not None:
        content = await read_upload_with_size_limit(file, max_upload_size_bytes)
        safe_file = PreReadUploadFile(content, filename=file.filename, content_type=file.content_type)

    request = ProcessFileRequest(file_id=file_id, options=options, chunking_strategy=parsed_chunking_strategy)
    return request, safe_file


class _FileProcessorJobs(Protocol):
    async def create_process_file_job(
        self, request: ProcessFileRequest, file: UploadFile | None = None
    ) -> ProcessFileJob: ...

    async def list_process_file_jobs(
        self, after: str | None = None, limit: int = 100
    ) -> ListProcessFileJobsResponse: ...

    async def retrieve_process_file_job(self, job_id: str) -> ProcessFileJob: ...

    async def cancel_process_file_job(self, job_id: str) -> ProcessFileJob: ...


def _job_impl(impl: FileProcessors) -> _FileProcessorJobs:
    """Return the optional server-side job capability without expanding the provider SDK contract."""
    methods = (
        "create_process_file_job",
        "list_process_file_jobs",
        "retrieve_process_file_job",
        "cancel_process_file_job",
    )
    if not all(callable(getattr(impl, name, None)) for name in methods):
        raise HTTPException(
            status_code=501,
            detail="File processor job execution is not enabled for the configured provider.",
        )
    return cast(_FileProcessorJobs, impl)


def create_router(impl: FileProcessors, max_upload_size_bytes: int = DEFAULT_MAX_UPLOAD_SIZE_BYTES) -> APIRouter:
    """Create a FastAPI router for the File Processors API.

    Args:
        impl: The FileProcessors implementation instance
        max_upload_size_bytes: Maximum allowed upload size in bytes for direct file uploads.

    Returns:
        APIRouter configured for the File Processors API
    """
    router = APIRouter(
        prefix=f"/{OGX_API_V1ALPHA}",
        tags=["File Processors"],
        responses=standard_responses,
    )

    @router.post(
        "/file-processors/process",
        response_model=ProcessFileResponse,
        summary="Process a file into chunks ready for vector database storage.",
        description=(
            "Deprecated. This endpoint blocks until processing completes, which can hold a connection open "
            "for large files. Use POST /file-processors/jobs to submit the file asynchronously and poll "
            "GET /file-processors/jobs/{job_id} for the result. "
            "Supports direct upload via multipart form or processing files already uploaded to file storage "
            "via file_id. Exactly one of file or file_id must be provided."
        ),
        deprecated=True,
        responses={
            200: {"description": "The processed file chunks."},
        },
    )
    async def process_file(
        file: Annotated[
            UploadFile | None,
            File(description="The File object to be uploaded and processed. Mutually exclusive with file_id."),
        ] = None,
        file_id: Annotated[
            str | None, Form(description="ID of file already uploaded to file storage. Mutually exclusive with file.")
        ] = None,
        options: Annotated[
            dict[str, Any] | None,
            Form(
                description="Optional processing options. Provider-specific parameters (e.g., OCR settings, output format)."
            ),
        ] = None,
        chunking_strategy: Annotated[
            str | None,
            Form(
                description="Optional chunking strategy for splitting content into chunks. Must be valid JSON string."
            ),
        ] = None,
    ) -> ProcessFileResponse:
        request, safe_file = await _build_request_and_file(
            file, file_id, options, chunking_strategy, max_upload_size_bytes
        )
        return await impl.process_file(request, safe_file)

    @router.post(
        "/file-processors/jobs",
        response_model=ProcessFileJob,
        summary="Submit a file for asynchronous processing.",
        description=(
            "Submit a file for processing and return immediately with a job handle. The work runs "
            "out-of-process so the server is not blocked. Poll GET /file-processors/jobs/{job_id} until the "
            "status is terminal. Supports direct upload via multipart form or a previously uploaded file_id."
        ),
        responses={
            200: {"description": "The created file-processing job."},
        },
    )
    async def create_process_file_job(
        file: Annotated[
            UploadFile | None,
            File(description="The File object to be uploaded and processed. Mutually exclusive with file_id."),
        ] = None,
        file_id: Annotated[
            str | None, Form(description="ID of file already uploaded to file storage. Mutually exclusive with file.")
        ] = None,
        options: Annotated[
            dict[str, Any] | None,
            Form(description="Optional processing options. Provider-specific parameters."),
        ] = None,
        chunking_strategy: Annotated[
            str | None,
            Form(
                description="Optional chunking strategy for splitting content into chunks. Must be valid JSON string."
            ),
        ] = None,
    ) -> ProcessFileJob:
        job_impl = _job_impl(impl)
        request, safe_file = await _build_request_and_file(
            file, file_id, options, chunking_strategy, max_upload_size_bytes
        )
        return await job_impl.create_process_file_job(request, safe_file)

    @router.get(
        "/file-processors/jobs",
        response_model=ListProcessFileJobsResponse,
        summary="List file-processing jobs.",
        description="List file-processing jobs, most recent first.",
    )
    async def list_process_file_jobs(
        after: Annotated[str | None, Query(description="Return jobs after this job ID.")] = None,
        limit: Annotated[int, Query(ge=1, le=100, description="Maximum jobs to return.")] = 100,
    ) -> ListProcessFileJobsResponse:
        return await _job_impl(impl).list_process_file_jobs(after=after, limit=limit)

    @router.get(
        "/file-processors/jobs/{job_id}",
        response_model=ProcessFileJob,
        summary="Retrieve a file-processing job.",
        description="Retrieve the current state of a file-processing job, including its result once completed.",
    )
    async def retrieve_process_file_job(job_id: str) -> ProcessFileJob:
        return await _job_impl(impl).retrieve_process_file_job(job_id)

    @router.post(
        "/file-processors/jobs/{job_id}/cancel",
        response_model=ProcessFileJob,
        summary="Cancel a file-processing job.",
        description="Cancel a scheduled or in-progress file-processing job.",
    )
    async def cancel_process_file_job(job_id: str) -> ProcessFileJob:
        return await _job_impl(impl).cancel_process_file_job(job_id)

    return router
