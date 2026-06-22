# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Containers API module.

Provides CRUD over sandboxed execution environments used by the Responses
``shell`` and ``code_interpreter`` tools. Protocol definitions are in
``api.py``, Pydantic models in ``models.py``, and FastAPI routes in
``fastapi_routes.py``.
"""

from . import fastapi_routes
from .api import ContainerRuntime, Containers
from .models import (
    Container,
    ContainerCreateRequest,
    ContainerDeleteResponse,
    ContainerExpiresAfter,
    ContainerFile,
    ContainerFileDeleteResponse,
    ContainerFileSource,
    ContainerStatus,
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
    NetworkCredential,
    NetworkDomainCredential,
    NetworkPolicy,
    NetworkPolicyExtended,
    NetworkPolicyMode,
    ShellCallOutput,
    ShellEnvironment,
    ShellEnvironmentContainerAuto,
    ShellEnvironmentContainerReference,
    ShellEnvironmentLocal,
    ShellOutcome,
    ShellOutcomeFailure,
    ShellOutcomeSuccess,
    ShellOutcomeTimeout,
    UploadContainerFileRequest,
)

__all__ = [
    "Container",
    "ContainerCreateRequest",
    "ContainerDeleteResponse",
    "ContainerExpiresAfter",
    "ContainerFile",
    "ContainerFileDeleteResponse",
    "ContainerFileSource",
    "ContainerRuntime",
    "ContainerStatus",
    "Containers",
    "DeleteContainerFileRequest",
    "DeleteContainerRequest",
    "ExecuteShellRequest",
    "GetContainerFileContentRequest",
    "GetContainerFileRequest",
    "GetContainerRequest",
    "ListContainerFilesRequest",
    "ListContainerFilesResponse",
    "ListContainersRequest",
    "ListContainersResponse",
    "MountSkillsRequest",
    "NetworkCredential",
    "NetworkDomainCredential",
    "NetworkPolicy",
    "NetworkPolicyExtended",
    "NetworkPolicyMode",
    "ShellCallOutput",
    "ShellEnvironment",
    "ShellEnvironmentContainerAuto",
    "ShellEnvironmentContainerReference",
    "ShellEnvironmentLocal",
    "ShellOutcome",
    "ShellOutcomeFailure",
    "ShellOutcomeSuccess",
    "ShellOutcomeTimeout",
    "UploadContainerFileRequest",
    "fastapi_routes",
]
