# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Pydantic models for File Processors API requests and responses.

This module defines the request and response models for the File Processors API
using Pydantic with Field descriptions for OpenAPI schema generation.

The ProcessFileRequest model wraps the serializable parameters for file processing.
The UploadFile parameter is kept separate (not serializable), following the same
pattern as UploadFileRequest in the Files API.
"""

from typing import Any

from pydantic import BaseModel, Field

from ogx_api.common.job_types import JobStatus
from ogx_api.schema_utils import json_schema_type
from ogx_api.vector_io import Chunk, VectorStoreChunkingStrategy


@json_schema_type
class ProcessFileResponse(BaseModel):
    """Response model for file processing operation.

    Returns a list of chunks ready for storage in vector databases.
    Each chunk contains the content and metadata.
    """

    chunks: list[Chunk] = Field(..., description="Processed chunks from the file. Always returns at least one chunk.")

    metadata: dict[str, Any] = Field(
        ...,
        description="Processing-run metadata such as processor name/version, processing_time_ms, page_count, extraction_method (e.g. docling/pypdf/ocr), confidence scores, plus provider-specific fields.",
    )


@json_schema_type
class ProcessFileRequest(BaseModel):
    """Request model for file processing operation.

    Wraps the serializable parameters for process_file. The UploadFile parameter
    is kept separate (not serializable), following the same pattern as
    UploadFileRequest in the Files API.
    """

    file_id: str | None = Field(
        default=None,
        description="ID of file already uploaded to file storage. Mutually exclusive with file.",
    )

    options: dict[str, Any] | None = Field(
        default=None,
        description="Optional processing options. Provider-specific parameters (e.g., OCR settings, output format).",
    )

    chunking_strategy: VectorStoreChunkingStrategy | None = Field(
        default=None,
        description="Optional chunking strategy for splitting content into chunks.",
    )


@json_schema_type
class ProcessFileJob(BaseModel):
    """An asynchronous file-processing job.

    Returned when a file is submitted for processing via the job-based API. Poll
    by job_id until ``status`` is terminal (completed/failed/cancelled). When
    completed, ``result`` holds the processed chunks; when failed, ``error``
    explains why.
    """

    job_id: str = Field(..., description="Unique identifier for the job.")
    status: JobStatus = Field(..., description="Current execution status of the job.")
    created_at: int = Field(..., description="Unix timestamp (seconds) for when the job was created.")
    result: ProcessFileResponse | None = Field(
        default=None, description="The processed file result. Present only once the job has completed successfully."
    )
    error: str | None = Field(default=None, description="Error message. Present only if the job failed.")


@json_schema_type
class ListProcessFileJobsResponse(BaseModel):
    """Response model listing file-processing jobs."""

    data: list[ProcessFileJob] = Field(..., description="The list of file-processing jobs.")
    has_more: bool = Field(default=False, description="Whether more jobs are available after this page.")


__all__ = [
    "ListProcessFileJobsResponse",
    "ProcessFileJob",
    "ProcessFileRequest",
    "ProcessFileResponse",
]
