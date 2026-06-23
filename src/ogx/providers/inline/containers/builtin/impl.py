# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Built-in Containers provider.

Implements the public :class:`ogx_api.containers.Containers` HTTP API. The
provider is a thin policy-and-delegation layer: it enforces operator network
policy layering (a request may only narrow the operator default, never expand
it), seeds requested ``file_ids`` from the Files API into the new container, and
forwards all lifecycle, file, and shell-execution work to a configured
``container_runtime`` provider.
"""

import io

from fastapi import Response, UploadFile

from ogx.core.datatypes import AccessRule
from ogx.log import get_logger
from ogx_api import (
    Container,
    ContainerCreateRequest,
    ContainerDeleteResponse,
    ContainerFile,
    ContainerFileDeleteResponse,
    ContainerRuntime,
    Containers,
    DeleteContainerFileRequest,
    DeleteContainerRequest,
    Files,
    GetContainerFileContentRequest,
    GetContainerFileRequest,
    GetContainerRequest,
    InvalidParameterError,
    ListContainerFilesRequest,
    ListContainerFilesResponse,
    ListContainersRequest,
    ListContainersResponse,
    NetworkPolicy,
    NetworkPolicyExtended,
    NetworkPolicyMode,
    NetworkPolicyViolationError,
    RetrieveFileContentRequest,
    RetrieveFileRequest,
    UploadContainerFileRequest,
)

from .config import BuiltinContainersConfig

logger = get_logger(name=__name__, category="containers")

# Egress permissiveness ordering: a request may only move to an equal or more
# restrictive mode than the operator default.
_MODE_RANK = {
    NetworkPolicyMode.DENY: 0,
    NetworkPolicyMode.ALLOW_LIST: 1,
    NetworkPolicyMode.ALLOW_ALL: 2,
}


class BuiltinContainersImpl(Containers):
    """Containers API implementation that delegates to a ContainerRuntime."""

    def __init__(
        self,
        config: BuiltinContainersConfig,
        runtime: ContainerRuntime,
        files: Files,
        policy: list[AccessRule],
    ) -> None:
        self.config = config
        self.runtime = runtime
        self.files = files
        self.policy = policy

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    # --- policy + validation --------------------------------------------

    def _validate_image(self, image: str | None) -> None:
        if image is None or self.config.allowed_images is None:
            return
        if image not in self.config.allowed_images:
            raise InvalidParameterError(
                "image",
                image,
                f"Image is not permitted. Allowed images: {', '.join(self.config.allowed_images)}.",
            )

    def _effective_network_policy(self, requested: NetworkPolicyExtended | None) -> NetworkPolicyExtended:
        """Layer a request policy over the operator default, enforcing narrowing-only.

        Raises ``NetworkPolicyViolationError`` (HTTP 403) when the request tries to
        widen egress beyond the operator default.
        """
        operator: NetworkPolicy = self.config.default_network_policy
        if requested is None:
            return NetworkPolicyExtended(
                mode=operator.mode,
                allow_domains=list(operator.allow_domains),
                deny_domains=list(operator.deny_domains),
            )

        if _MODE_RANK[requested.mode] > _MODE_RANK[operator.mode]:
            raise NetworkPolicyViolationError(
                f"Failed to apply network policy: requested mode '{requested.mode}' is less restrictive "
                f"than the operator default '{operator.mode}'."
            )

        if operator.mode == NetworkPolicyMode.DENY and requested.allow_domains:
            raise NetworkPolicyViolationError(
                "Failed to apply network policy: operator default denies all egress, "
                "so no allow_domains may be requested."
            )
        if operator.mode == NetworkPolicyMode.ALLOW_LIST:
            extra = set(requested.allow_domains) - set(operator.allow_domains)
            if extra:
                raise NetworkPolicyViolationError(
                    "Failed to apply network policy: allow_domains "
                    f"{sorted(extra)} are not within the operator allow list."
                )

        # deny_domains can only grow (further narrowing), so union operator + request.
        merged_deny = sorted(set(operator.deny_domains) | set(requested.deny_domains))
        return NetworkPolicyExtended(
            mode=requested.mode,
            allow_domains=list(requested.allow_domains),
            deny_domains=merged_deny,
            domain_credentials=list(requested.domain_credentials),
        )

    # --- lifecycle -------------------------------------------------------

    async def create_container(self, request: ContainerCreateRequest) -> Container:
        self._validate_image(request.image)
        effective_policy = self._effective_network_policy(request.network_policy)

        runtime_request = request.model_copy(update={"network_policy": effective_policy})
        container = await self.runtime.create_container(runtime_request)

        for file_id in request.file_ids:
            await self._seed_file(container.id, file_id)
        return container

    async def _seed_file(self, container_id: str, file_id: str) -> None:
        """Copy a Files-API file into the new container's /mnt/data."""
        meta = await self.files.openai_retrieve_file(RetrieveFileRequest(file_id=file_id))
        content = await self.files.openai_retrieve_file_content(RetrieveFileContentRequest(file_id=file_id))
        data = content.body if isinstance(content, Response) else bytes(content)
        upload = UploadFile(file=io.BytesIO(data), filename=meta.filename)
        await self.runtime.upload_file(UploadContainerFileRequest(container_id=container_id), upload)

    async def list_containers(self, request: ListContainersRequest) -> ListContainersResponse:
        return await self.runtime.list_containers(request)

    async def get_container(self, request: GetContainerRequest) -> Container:
        return await self.runtime.get_container(request)

    async def delete_container(self, request: DeleteContainerRequest) -> ContainerDeleteResponse:
        return await self.runtime.delete_container(request)

    # --- files (delegated) ----------------------------------------------

    async def upload_container_file(self, request: UploadContainerFileRequest, file: UploadFile) -> ContainerFile:
        return await self.runtime.upload_file(request, file)

    async def list_container_files(self, request: ListContainerFilesRequest) -> ListContainerFilesResponse:
        return await self.runtime.list_files(request)

    async def get_container_file(self, request: GetContainerFileRequest) -> ContainerFile:
        return await self.runtime.get_file(request)

    async def get_container_file_content(self, request: GetContainerFileContentRequest) -> Response:
        return await self.runtime.get_file_content(request)

    async def delete_container_file(self, request: DeleteContainerFileRequest) -> ContainerFileDeleteResponse:
        return await self.runtime.delete_file(request)
