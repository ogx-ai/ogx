# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Containers API protocol definitions.

This module contains two protocols:

* ``Containers`` — the HTTP-facing CRUD API for sandboxed execution
  environments.
* ``ContainerRuntime`` — the internal provider protocol that backends
  (Docker/Podman, Kubernetes, local) implement. The Containers provider
  delegates to a ``ContainerRuntime`` for the actual lifecycle, file, and
  shell-execution work.

Pydantic models are defined in ``models.py`` and FastAPI routes in
``fastapi_routes.py``.
"""

from typing import Protocol, runtime_checkable

from fastapi import Response, UploadFile

from .models import (
    Container,
    ContainerCreateRequest,
    ContainerDeleteResponse,
    ContainerFile,
    ContainerFileDeleteResponse,
    DeleteContainerFileRequest,
    DeleteContainerRequest,
    ExecuteShellRequest,
    GetContainerFileContentRequest,
    GetContainerFileRequest,
    GetContainerRequest,
    ListContainerFilesRequest,
    ListContainerFilesResponse,
    ListContainersRequest,
    ListContainersResponse,
    MountSkillsRequest,
    ShellCallOutput,
    UploadContainerFileRequest,
)


@runtime_checkable
class Containers(Protocol):
    """HTTP API for managing sandboxed execution containers.

    Implementations enforce policy layering (request-supplied
    ``NetworkPolicyExtended`` cannot expand operator defaults) and then
    delegate to a configured :class:`ContainerRuntime` for backend work.
    """

    async def create_container(
        self,
        request: ContainerCreateRequest,
    ) -> Container: ...

    async def list_containers(
        self,
        request: ListContainersRequest,
    ) -> ListContainersResponse: ...

    async def get_container(
        self,
        request: GetContainerRequest,
    ) -> Container: ...

    async def delete_container(
        self,
        request: DeleteContainerRequest,
    ) -> ContainerDeleteResponse: ...

    async def upload_container_file(
        self,
        request: UploadContainerFileRequest,
        file: UploadFile,
    ) -> ContainerFile: ...

    async def list_container_files(
        self,
        request: ListContainerFilesRequest,
    ) -> ListContainerFilesResponse: ...

    async def get_container_file(
        self,
        request: GetContainerFileRequest,
    ) -> ContainerFile: ...

    async def get_container_file_content(
        self,
        request: GetContainerFileContentRequest,
    ) -> Response: ...

    async def delete_container_file(
        self,
        request: DeleteContainerFileRequest,
    ) -> ContainerFileDeleteResponse: ...


@runtime_checkable
class ContainerRuntime(Protocol):
    """Internal provider protocol for container backends.

    A ``ContainerRuntime`` provider is the thin layer between the Containers
    API and a concrete backend (Docker/Podman socket, Kubernetes API, local
    process supervisor). The Responses ``shell`` tool also calls
    ``execute_shell`` directly when running in ``container_auto`` /
    ``container_reference`` modes.

    This protocol is not exposed over HTTP.
    """

    # --- lifecycle -------------------------------------------------------

    async def create_container(
        self,
        request: ContainerCreateRequest,
    ) -> Container:
        """Create and start a container. Raises if resource limits are exceeded."""
        ...

    async def get_container(self, request: GetContainerRequest) -> Container:
        """Fetch the current state of a container, including ``last_active_at``."""
        ...

    async def list_containers(
        self,
        request: ListContainersRequest,
    ) -> ListContainersResponse:
        """List containers known to this runtime."""
        ...

    async def delete_container(self, request: DeleteContainerRequest) -> ContainerDeleteResponse:
        """Stop and remove a container along with its filesystem."""
        ...

    # --- file management ------------------------------------------------

    async def upload_file(
        self,
        request: UploadContainerFileRequest,
        file: UploadFile,
    ) -> ContainerFile:
        """Copy an uploaded file into the container at the runtime-chosen path."""
        ...

    async def list_files(
        self,
        request: ListContainerFilesRequest,
    ) -> ListContainerFilesResponse:
        """List files tracked inside a container."""
        ...

    async def get_file(
        self,
        request: GetContainerFileRequest,
    ) -> ContainerFile:
        """Get metadata for a single file inside a container."""
        ...

    async def get_file_content(
        self,
        request: GetContainerFileContentRequest,
    ) -> Response:
        """Stream the bytes of a file inside a container."""
        ...

    async def delete_file(
        self,
        request: DeleteContainerFileRequest,
    ) -> ContainerFileDeleteResponse:
        """Remove a file from a container's filesystem."""
        ...

    # --- execution ------------------------------------------------------

    async def execute_shell(
        self,
        request: ExecuteShellRequest,
    ) -> ShellCallOutput:
        """Run a shell command inside the container and capture its output.

        ``request.command`` is passed as ``argv`` (no shell expansion) to keep
        the interface uniform across backends. Implementations are expected to
        update ``last_active_at`` on the container as a side effect.
        """
        ...

    # --- skill mounting -------------------------------------------------

    async def mount_skills(
        self,
        request: MountSkillsRequest,
    ) -> None:
        """Mount skill bundles into ``/mnt/skills/{skill_name}/`` inside the container.

        Each entry of ``request.skill_bundles`` is ``(skill_name, zip_bytes)``
        — the zip archive is extracted into the named directory. Calling this
        method multiple times with overlapping names overwrites the previous
        contents.
        """
        ...
