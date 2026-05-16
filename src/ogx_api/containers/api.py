# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Protocol, runtime_checkable

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


@runtime_checkable
class Containers(Protocol):
    """Containers API for managing sandboxed execution environments."""

    async def create_container(
        self,
        request: CreateContainerRequest,
    ) -> Container: ...

    async def list_containers(
        self,
        request: ListContainersRequest,
    ) -> ListContainersResponse: ...

    async def retrieve_container(
        self,
        request: RetrieveContainerRequest,
    ) -> Container: ...

    async def delete_container(
        self,
        request: DeleteContainerRequest,
    ) -> DeleteContainerResponse: ...

    async def exec_in_container(
        self,
        request: ExecInContainerRequest,
    ) -> ExecInContainerResponse: ...
