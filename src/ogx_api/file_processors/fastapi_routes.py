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
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import ValidationError

from ogx_api.common.upload_safety import (
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
from .models import ProcessFileRequest, ProcessFileResponse


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
        description="Process a file into chunks ready for vector database storage. Supports direct upload via multipart form or processing files already uploaded to file storage via file_id. Exactly one of file or file_id must be provided.",
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
        # Parse chunking_strategy JSON string if provided
        parsed_chunking_strategy: VectorStoreChunkingStrategy | None = None
        if chunking_strategy:
            try:
                chunking_data = json.loads(chunking_strategy)

                # Validate that chunking_data is a JSON object (dict)
                if not isinstance(chunking_data, dict):
                    raise HTTPException(
                        status_code=400,
                        detail="chunking_strategy must be a JSON object, not a list, string, or other type",
                    )

                if chunking_data.get("type") == "auto":
                    parsed_chunking_strategy = VectorStoreChunkingStrategyAuto.model_validate(chunking_data)
                elif chunking_data.get("type") == "static":
                    parsed_chunking_strategy = VectorStoreChunkingStrategyStatic.model_validate(chunking_data)
                else:
                    raise HTTPException(
                        status_code=400, detail=f"Invalid chunking strategy type: {chunking_data.get('type')}"
                    )
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, detail=f"Invalid JSON in chunking_strategy: {str(e)}") from e
            except ValidationError as e:
                raise HTTPException(status_code=400, detail=f"Invalid chunking strategy: {str(e)}") from e

        # For direct uploads, enforce the upload size limit before passing to the provider
        safe_file = None
        if file is not None:
            content = await read_upload_with_size_limit(file, max_upload_size_bytes)
            safe_file = PreReadUploadFile(content, filename=file.filename, content_type=file.content_type)

        request = ProcessFileRequest(
            file_id=file_id,
            options=options,
            chunking_strategy=parsed_chunking_strategy,
        )
        return await impl.process_file(request, safe_file)

    return router
