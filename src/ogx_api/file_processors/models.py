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


__all__ = [
    "ProcessFileRequest",
    "ProcessFileResponse",
]
