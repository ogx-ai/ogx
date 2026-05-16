# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Annotated

from fastapi import APIRouter, Depends

from ogx_api.router_utils import create_path_dependency, create_query_dependency, standard_responses
from ogx_api.version import OGX_API_V1

from .api import Containers
from .models import (
    Container,
    CreateContainerRequest,
    DeleteContainerRequest,
    DeleteContainerResponse,
    ExecInContainerRequest,
    ExecInContainerResponse,
    ListContainersRequest,
    ListContainersResponse,
    RetrieveContainerRequest,
)

get_list_containers_request = create_query_dependency(ListContainersRequest)
get_retrieve_container_request = create_path_dependency(RetrieveContainerRequest)
get_delete_container_request = create_path_dependency(DeleteContainerRequest)


def create_router(impl: Containers) -> APIRouter:
    router = APIRouter(
        prefix=f"/{OGX_API_V1}",
        tags=["Containers"],
        responses=standard_responses,
    )

    @router.post(
        "/containers",
        response_model=Container,
        summary="Create container",
        description="Creates a sandboxed container environment.",
    )
    async def create_container(
        request: CreateContainerRequest,
    ) -> Container:
        return await impl.create_container(request)

    @router.get(
        "/containers",
        response_model=ListContainersResponse,
        summary="List containers",
        description="Lists containers.",
    )
    async def list_containers(
        request: Annotated[ListContainersRequest, Depends(get_list_containers_request)],
    ) -> ListContainersResponse:
        return await impl.list_containers(request)

    @router.get(
        "/containers/{container_id}",
        response_model=Container,
        summary="Retrieve container",
        description="Retrieves a container.",
    )
    async def retrieve_container(
        request: Annotated[RetrieveContainerRequest, Depends(get_retrieve_container_request)],
    ) -> Container:
        return await impl.retrieve_container(request)

    @router.delete(
        "/containers/{container_id}",
        response_model=DeleteContainerResponse,
        summary="Delete container",
        description="Deletes a container.",
    )
    async def delete_container(
        request: Annotated[DeleteContainerRequest, Depends(get_delete_container_request)],
    ) -> DeleteContainerResponse:
        return await impl.delete_container(request)

    @router.post(
        "/containers/{container_id}/exec",
        response_model=ExecInContainerResponse,
        summary="Execute in container",
        description="Executes shell commands in a container.",
    )
    async def exec_in_container(
        container_id: str,
        request: ExecInContainerRequest,
    ) -> ExecInContainerResponse:
        request.container_id = container_id
        return await impl.exec_in_container(request)

    return router
