# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Annotated

from fastapi import APIRouter, Depends, UploadFile
from fastapi.param_functions import File, Form
from fastapi.responses import Response

from ogx_api.common.upload_safety import (
    DEFAULT_MAX_UPLOAD_SIZE_BYTES,
    PreReadUploadFile,
    read_upload_with_size_limit,
)
from ogx_api.router_utils import create_path_dependency, create_query_dependency, standard_responses
from ogx_api.version import OGX_API_V1

from .api import Files
from .models import (
    DeleteFileRequest,
    ExpiresAfter,
    ListFilesRequest,
    ListOpenAIFileResponse,
    OpenAIFileDeleteResponse,
    OpenAIFileObject,
    OpenAIFilePurpose,
    RetrieveFileContentRequest,
    RetrieveFileRequest,
    UploadFileRequest,
)

# Automatically generate dependency functions from Pydantic models
# This ensures the models are the single source of truth for descriptions
get_list_files_request = create_query_dependency(ListFilesRequest)
get_get_files_request = create_path_dependency(RetrieveFileRequest)
get_delete_files_request = create_path_dependency(DeleteFileRequest)
get_retrieve_file_content_request = create_path_dependency(RetrieveFileContentRequest)


def create_router(impl: Files, max_upload_size_bytes: int = DEFAULT_MAX_UPLOAD_SIZE_BYTES) -> APIRouter:
    router = APIRouter(
        prefix=f"/{OGX_API_V1}",
        tags=["Files"],
        responses=standard_responses,
    )

    @router.get(
        "/files",
        response_model=ListOpenAIFileResponse,
        summary="List files",
        description="List files",
        responses={
            200: {"description": "The list of files."},
        },
    )
    async def list_files(
        request: Annotated[ListFilesRequest, Depends(get_list_files_request)],
    ) -> ListOpenAIFileResponse:
        return await impl.openai_list_files(request)

    @router.get(
        "/files/{file_id}",
        response_model=OpenAIFileObject,
        summary="Get file",
        description="Get file",
        responses={
            200: {"description": "The file."},
        },
    )
    async def get_file(
        request: Annotated[RetrieveFileRequest, Depends(get_get_files_request)],
    ) -> OpenAIFileObject:
        return await impl.openai_retrieve_file(request)

    @router.delete(
        "/files/{file_id}",
        response_model=OpenAIFileDeleteResponse,
        summary="Delete file",
        description="Delete file",
        responses={
            200: {"description": "The file was deleted."},
        },
    )
    async def delete_file(
        request: Annotated[DeleteFileRequest, Depends(get_delete_files_request)],
    ) -> OpenAIFileDeleteResponse:
        return await impl.openai_delete_file(request)

    @router.get(
        "/files/{file_id}/content",
        status_code=200,
        summary="Retrieve file content",
        description="Retrieve file content",
        responses={
            200: {
                "description": "The file content.",
                "content": {"application/json": {"schema": {"type": "string"}}},
            },
        },
    )
    async def retrieve_file_content(
        request: Annotated[RetrieveFileContentRequest, Depends(get_retrieve_file_content_request)],
    ) -> Response:
        return await impl.openai_retrieve_file_content(request)

    @router.post(
        "/files",
        response_model=OpenAIFileObject,
        summary="Upload file",
        description="Upload a file.",
        responses={
            200: {"description": "The uploaded file."},
        },
    )
    async def upload_file(
        file: Annotated[UploadFile, File(description="The file to upload.")],
        purpose: Annotated[OpenAIFilePurpose, Form(description="The intended purpose of the uploaded file.")],
        expires_after: Annotated[ExpiresAfter | None, Form(description="Optional expiration settings.")] = None,
    ) -> OpenAIFileObject:
        content = await read_upload_with_size_limit(file, max_upload_size_bytes)
        safe_file = PreReadUploadFile(content, filename=file.filename, content_type=file.content_type)
        request = UploadFileRequest(
            purpose=purpose,
            expires_after=expires_after,
        )
        return await impl.openai_upload_file(request, safe_file)

    return router
