# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""FastAPI routes for the Containers API.

Endpoints are mounted at ``/v1alpha/containers``. See
:class:`ogx_api.containers.api.Containers` for the protocol implementations
must satisfy.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, UploadFile
from fastapi.param_functions import File
from fastapi.responses import Response

from ogx_api.common.upload_limits import (
    DEFAULT_MAX_UPLOAD_SIZE_BYTES,
    PreReadUploadFile,
    read_upload_with_size_limit,
)
from ogx_api.router_utils import create_path_dependency, create_query_dependency, standard_responses
from ogx_api.version import OGX_API_V1ALPHA

from .api import Containers
from .models import (
    Container,
    ContainerCreateRequest,
    ContainerDeleteResponse,
    ContainerFile,
    ContainerFileDeleteResponse,
    DeleteContainerFileRequest,
    DeleteContainerRequest,
    GetContainerFileContentRequest,
    GetContainerFileRequest,
    GetContainerRequest,
    ListContainerFilesRequest,
    ListContainerFilesResponse,
    ListContainersRequest,
    ListContainersResponse,
    UploadContainerFileRequest,
)

get_list_containers_request = create_query_dependency(ListContainersRequest)
get_container_request = create_path_dependency(GetContainerRequest)
get_delete_container_request = create_path_dependency(DeleteContainerRequest)
get_upload_container_file_request = create_path_dependency(UploadContainerFileRequest)


# Multi-field path-parameter models cannot use create_path_dependency, which only
# supports single-field models. The helpers below combine path (+ query) params
# into the request model explicitly.


def _list_container_files_request(
    container_id: str,
    after: str | None = None,
    limit: int | None = 20,
    order: str | None = "desc",
) -> ListContainerFilesRequest:
    return ListContainerFilesRequest.model_validate(
        {"container_id": container_id, "after": after, "limit": limit, "order": order}
    )


def _get_container_file_request(container_id: str, file_id: str) -> GetContainerFileRequest:
    return GetContainerFileRequest(container_id=container_id, file_id=file_id)


def _get_container_file_content_request(container_id: str, file_id: str) -> GetContainerFileContentRequest:
    return GetContainerFileContentRequest(container_id=container_id, file_id=file_id)


def _delete_container_file_request(container_id: str, file_id: str) -> DeleteContainerFileRequest:
    return DeleteContainerFileRequest(container_id=container_id, file_id=file_id)


def create_router(impl: Containers, max_upload_size_bytes: int = DEFAULT_MAX_UPLOAD_SIZE_BYTES) -> APIRouter:
    router = APIRouter(
        prefix=f"/{OGX_API_V1ALPHA}",
        tags=["Containers"],
        responses=standard_responses,
    )

    @router.post(
        "/containers",
        response_model=Container,
        summary="Create container",
        description="Create a sandboxed container for shell/code execution.",
        responses={200: {"description": "The created container."}},
    )
    async def create_container(request: ContainerCreateRequest) -> Container:
        return await impl.create_container(request)

    @router.get(
        "/containers",
        response_model=ListContainersResponse,
        summary="List containers",
        description="List containers.",
        responses={200: {"description": "The list of containers."}},
    )
    async def list_containers(
        request: Annotated[ListContainersRequest, Depends(get_list_containers_request)],
    ) -> ListContainersResponse:
        return await impl.list_containers(request)

    @router.get(
        "/containers/{container_id}",
        response_model=Container,
        summary="Get container",
        description="Get a container by ID.",
        responses={200: {"description": "The container."}},
    )
    async def get_container(
        request: Annotated[GetContainerRequest, Depends(get_container_request)],
    ) -> Container:
        return await impl.get_container(request)

    @router.delete(
        "/containers/{container_id}",
        response_model=ContainerDeleteResponse,
        summary="Delete container",
        description="Stop and remove a container.",
        responses={200: {"description": "The container was deleted."}},
    )
    async def delete_container(
        request: Annotated[DeleteContainerRequest, Depends(get_delete_container_request)],
    ) -> ContainerDeleteResponse:
        return await impl.delete_container(request)

    @router.post(
        "/containers/{container_id}/files",
        response_model=ContainerFile,
        summary="Upload container file",
        description="Upload a file into a container's filesystem.",
        responses={200: {"description": "The uploaded container file."}},
    )
    async def upload_container_file(
        container_id: str,
        file: Annotated[UploadFile, File(description="The file to upload into the container.")],
    ) -> ContainerFile:
        content = await read_upload_with_size_limit(file, max_upload_size_bytes)
        safe_file = PreReadUploadFile(content, filename=file.filename, content_type=file.content_type)
        return await impl.upload_container_file(
            UploadContainerFileRequest(container_id=container_id),
            safe_file,
        )

    @router.get(
        "/containers/{container_id}/files",
        response_model=ListContainerFilesResponse,
        summary="List container files",
        description="List files inside a container.",
        responses={200: {"description": "The list of files."}},
    )
    async def list_container_files(
        request: Annotated[ListContainerFilesRequest, Depends(_list_container_files_request)],
    ) -> ListContainerFilesResponse:
        return await impl.list_container_files(request)

    @router.get(
        "/containers/{container_id}/files/{file_id}",
        response_model=ContainerFile,
        summary="Get container file",
        description="Get metadata for a file inside a container.",
        responses={200: {"description": "The container file metadata."}},
    )
    async def get_container_file(
        request: Annotated[GetContainerFileRequest, Depends(_get_container_file_request)],
    ) -> ContainerFile:
        return await impl.get_container_file(request)

    @router.get(
        "/containers/{container_id}/files/{file_id}/content",
        status_code=200,
        summary="Get container file content",
        description="Download the contents of a file inside a container.",
        responses={
            200: {
                "description": "The file content.",
                "content": {"application/octet-stream": {"schema": {"type": "string", "format": "binary"}}},
            },
        },
    )
    async def get_container_file_content(
        request: Annotated[GetContainerFileContentRequest, Depends(_get_container_file_content_request)],
    ) -> Response:
        return await impl.get_container_file_content(request)

    @router.delete(
        "/containers/{container_id}/files/{file_id}",
        response_model=ContainerFileDeleteResponse,
        summary="Delete container file",
        description="Remove a file from a container's filesystem.",
        responses={200: {"description": "The file was deleted."}},
    )
    async def delete_container_file(
        request: Annotated[DeleteContainerFileRequest, Depends(_delete_container_file_request)],
    ) -> ContainerFileDeleteResponse:
        return await impl.delete_container_file(request)

    return router
