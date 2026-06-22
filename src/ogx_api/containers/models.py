# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Pydantic models for the Containers API.

The Containers API provides CRUD over sandboxed execution environments used
by the Responses ``shell`` and ``code_interpreter`` tools. The models here
cover three concerns:

* container lifecycle (``Container``, ``ContainerCreateRequest``)
* container file management (``ContainerFile`` and request models)
* network policy layering (``NetworkPolicy`` / ``NetworkPolicyExtended``)
* shell execution shapes consumed by the Responses provider
  (``ShellEnvironment``, ``ShellCallOutput``, ``ShellOutcome``)
"""

from enum import StrEnum
from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, Field, SecretStr

from ogx_api.common.responses import Order
from ogx_api.schema_utils import json_schema_type

# ---------------------------------------------------------------------------
# Expiration
# ---------------------------------------------------------------------------


@json_schema_type
class ContainerExpiresAfter(BaseModel):
    """Control expiration of a container.

    Anchored on ``last_active_at`` (each shell execution or file operation
    refreshes the anchor). Operator-set bounds protect the host from
    long-lived sandboxes.
    """

    MIN: ClassVar[int] = 60  # 1 minute
    MAX: ClassVar[int] = 86400  # 24 hours

    anchor: Literal["last_active_at"] = Field(
        default="last_active_at",
        description="The anchor point for expiration. Must be 'last_active_at'.",
    )
    minutes: int = Field(
        ...,
        ge=1,
        le=1440,
        description="Minutes of inactivity after the anchor before the container expires.",
    )


# ---------------------------------------------------------------------------
# Network policy
# ---------------------------------------------------------------------------


class NetworkPolicyMode(StrEnum):
    """Egress policy mode applied to a container's outbound network."""

    DENY = "deny"
    ALLOW_LIST = "allow_list"
    ALLOW_ALL = "allow_all"


@json_schema_type
class NetworkCredential(BaseModel):
    """A named credential available to outbound network calls.

    The ``value`` should be a secret reference (e.g. ``${env.MY_SECRET}``)
    in operator-supplied configuration, never a raw secret in a request body.
    """

    name: str = Field(..., description="Logical name used by the container to look up the credential.")
    value: SecretStr = Field(..., description="Secret reference or literal value to be injected into the container.")


@json_schema_type
class NetworkDomainCredential(BaseModel):
    """Bind a ``NetworkCredential`` to a specific outbound domain."""

    domain: str = Field(..., description="Fully-qualified domain name to which the credential applies.")
    credential: NetworkCredential = Field(..., description="Credential injected on outbound calls to this domain.")


@json_schema_type
class NetworkPolicy(BaseModel):
    """Operator-set egress policy for a container.

    A NetworkPolicy is the *upper bound* — request-supplied
    ``NetworkPolicyExtended`` values may only narrow this policy.
    """

    mode: NetworkPolicyMode = Field(
        default=NetworkPolicyMode.DENY,
        description="Default egress disposition. 'deny' blocks all egress except entries in 'allow_domains'.",
    )
    allow_domains: list[str] = Field(
        default_factory=list,
        description="Domains permitted for outbound traffic. Used when mode is 'allow_list'.",
    )
    deny_domains: list[str] = Field(
        default_factory=list,
        description="Domains explicitly blocked. Takes precedence over 'allow_domains'.",
    )


@json_schema_type
class NetworkPolicyExtended(NetworkPolicy):
    """Request-layer extension of an operator NetworkPolicy.

    The request may add domain credentials and narrow allow/deny lists, but
    cannot expand the operator default — enforcement is performed at the API
    layer; see issue #5892 task 8.
    """

    domain_credentials: list[NetworkDomainCredential] = Field(
        default_factory=list,
        description="Per-domain credentials injected on outbound calls from this container.",
    )


# ---------------------------------------------------------------------------
# Container resource
# ---------------------------------------------------------------------------


class ContainerStatus(StrEnum):
    """Lifecycle status of a container."""

    ACTIVE = "active"
    EXPIRED = "expired"


@json_schema_type
class Container(BaseModel):
    """A sandboxed execution environment.

    Mirrors the OpenAI Containers API resource with OGX-specific extensions
    for network policy and image selection.
    """

    id: str = Field(..., description="Identifier for the container.")
    object: Literal["container"] = Field(
        default="container", description="The object type, which is always 'container'."
    )
    created_at: int = Field(..., description="Unix timestamp (in seconds) for when the container was created.")
    status: ContainerStatus = Field(..., description="Current lifecycle status.")
    last_active_at: int = Field(
        ..., description="Unix timestamp (in seconds) of the last operation performed against this container."
    )
    name: str | None = Field(default=None, description="Human-readable name for the container.")
    expires_after: ContainerExpiresAfter | None = Field(
        default=None, description="Inactivity-based expiration settings."
    )
    image: str | None = Field(
        default=None,
        description="Container image used to run the sandbox. May be operator-locked.",
    )
    network_policy: NetworkPolicy | None = Field(
        default=None,
        description="Effective network policy after layering operator defaults with request extensions.",
    )


@json_schema_type
class ContainerCreateRequest(BaseModel):
    """Request body for ``POST /containers``."""

    name: str | None = Field(default=None, description="Human-readable name for the container.")
    file_ids: list[str] = Field(
        default_factory=list,
        description="Files (from the Files API) to seed into the container at /mnt/data/.",
    )
    expires_after: ContainerExpiresAfter | None = Field(
        default=None, description="Inactivity-based expiration settings."
    )
    image: str | None = Field(
        default=None,
        description="Requested container image. The operator policy may pin or reject this value.",
    )
    network_policy: NetworkPolicyExtended | None = Field(
        default=None,
        description="Request-supplied network policy extension. Must be a subset of the operator default.",
    )


@json_schema_type
class ListContainersRequest(BaseModel):
    """Query parameters for ``GET /containers``."""

    after: str | None = Field(default=None, description="Cursor for pagination. Returns containers after this ID.")
    limit: int | None = Field(default=20, ge=1, le=100, description="Maximum number of containers to return (1-100).")
    order: Order | None = Field(default=Order.desc, description="Sort order by created_at timestamp ('asc' or 'desc').")


@json_schema_type
class ListContainersResponse(BaseModel):
    """Response for ``GET /containers``."""

    object: Literal["list"] = Field(default="list", description="The object type, which is always 'list'.")
    data: list[Container] = Field(..., description="The list of containers.")
    first_id: str | None = Field(default=None, description="ID of the first container in the page.")
    last_id: str | None = Field(default=None, description="ID of the last container in the page.")
    has_more: bool = Field(..., description="Whether more containers exist beyond this page.")


@json_schema_type
class GetContainerRequest(BaseModel):
    container_id: str = Field(..., description="The ID of the container to retrieve.")


@json_schema_type
class DeleteContainerRequest(BaseModel):
    container_id: str = Field(..., description="The ID of the container to delete.")


@json_schema_type
class ContainerDeleteResponse(BaseModel):
    """Response for ``DELETE /containers/{container_id}``."""

    id: str = Field(..., description="The container identifier that was deleted.")
    object: Literal["container"] = Field(
        default="container", description="The object type, which is always 'container'."
    )
    deleted: bool = Field(..., description="Whether the container was successfully deleted.")


# ---------------------------------------------------------------------------
# Container files
# ---------------------------------------------------------------------------


class ContainerFileSource(StrEnum):
    """Origin of a file inside a container."""

    USER = "user"
    ASSISTANT = "assistant"


@json_schema_type
class ContainerFile(BaseModel):
    """A file present inside a container's filesystem."""

    id: str = Field(..., description="Identifier of the container file.")
    object: Literal["container.file"] = Field(
        default="container.file", description="The object type, which is always 'container.file'."
    )
    container_id: str = Field(..., description="ID of the container holding the file.")
    created_at: int = Field(..., description="Unix timestamp (in seconds) when the file was created.")
    bytes: int = Field(..., description="Size of the file in bytes.")
    path: str = Field(..., description="Absolute path to the file inside the container.")
    source: ContainerFileSource = Field(
        ..., description="Whether the file was supplied by the user or written by the model."
    )


@json_schema_type
class UploadContainerFileRequest(BaseModel):
    """Path parameters for ``POST /containers/{container_id}/files``.

    The file content itself is supplied as a multipart upload and not part of
    this Pydantic body; see ``fastapi_routes.py``.
    """

    container_id: str = Field(..., description="The ID of the container to upload into.")


@json_schema_type
class ListContainerFilesRequest(BaseModel):
    container_id: str = Field(..., description="The ID of the container whose files should be listed.")
    after: str | None = Field(default=None, description="Cursor for pagination.")
    limit: int | None = Field(default=20, ge=1, le=100, description="Maximum number of files to return (1-100).")
    order: Order | None = Field(default=Order.desc, description="Sort order by created_at timestamp.")


@json_schema_type
class ListContainerFilesResponse(BaseModel):
    object: Literal["list"] = Field(default="list", description="The object type, which is always 'list'.")
    data: list[ContainerFile] = Field(..., description="The list of files in the container.")
    first_id: str | None = Field(default=None, description="ID of the first file in the page.")
    last_id: str | None = Field(default=None, description="ID of the last file in the page.")
    has_more: bool = Field(..., description="Whether more files exist beyond this page.")


@json_schema_type
class GetContainerFileRequest(BaseModel):
    container_id: str = Field(..., description="The ID of the container holding the file.")
    file_id: str = Field(..., description="The ID of the container file to retrieve.")


@json_schema_type
class GetContainerFileContentRequest(BaseModel):
    container_id: str = Field(..., description="The ID of the container holding the file.")
    file_id: str = Field(..., description="The ID of the container file to download.")


@json_schema_type
class DeleteContainerFileRequest(BaseModel):
    container_id: str = Field(..., description="The ID of the container holding the file.")
    file_id: str = Field(..., description="The ID of the container file to delete.")


@json_schema_type
class ContainerFileDeleteResponse(BaseModel):
    id: str = Field(..., description="The container file identifier that was deleted.")
    object: Literal["container.file"] = Field(
        default="container.file", description="The object type, which is always 'container.file'."
    )
    deleted: bool = Field(..., description="Whether the file was successfully deleted.")


# ---------------------------------------------------------------------------
# Shell execution shapes
# ---------------------------------------------------------------------------


@json_schema_type
class ShellEnvironmentContainerAuto(BaseModel):
    """Provider-managed container environment.

    The provider lazily creates and reuses a container for the calling
    response chain. Useful when the caller does not need to persist or
    reference the container across responses.
    """

    type: Literal["container_auto"] = Field(default="container_auto", description="Discriminator.")
    image: str | None = Field(default=None, description="Optional preferred container image.")
    expires_after: ContainerExpiresAfter | None = Field(
        default=None, description="Inactivity-based expiration for the auto-created container."
    )


@json_schema_type
class ShellEnvironmentContainerReference(BaseModel):
    """Reference an existing container by ID."""

    type: Literal["container_reference"] = Field(default="container_reference", description="Discriminator.")
    container_id: str = Field(..., description="The ID of an existing container to execute inside.")


@json_schema_type
class ShellEnvironmentLocal(BaseModel):
    """Local (non-container) execution mode.

    Only available when the operator has explicitly enabled local mode in
    the ContainerRuntime provider configuration.
    """

    type: Literal["local"] = Field(default="local", description="Discriminator.")
    working_directory: str | None = Field(default=None, description="Optional working directory for local execution.")


ShellEnvironment = Annotated[
    ShellEnvironmentContainerAuto | ShellEnvironmentContainerReference | ShellEnvironmentLocal,
    Field(discriminator="type"),
]
"""Discriminated union of the three shell execution environments."""


@json_schema_type
class ShellOutcomeSuccess(BaseModel):
    """Process exited cleanly with status 0."""

    type: Literal["success"] = Field(default="success", description="Discriminator.")
    exit_code: Literal[0] = Field(default=0, description="Process exit code (always 0 for success).")


@json_schema_type
class ShellOutcomeFailure(BaseModel):
    """Process exited with a non-zero status."""

    type: Literal["failure"] = Field(default="failure", description="Discriminator.")
    exit_code: int = Field(..., description="Process exit code.")
    reason: str | None = Field(default=None, description="Human-readable failure reason, if known.")


@json_schema_type
class ShellOutcomeTimeout(BaseModel):
    """Process was terminated for exceeding its time budget."""

    type: Literal["timeout"] = Field(default="timeout", description="Discriminator.")
    elapsed_seconds: float = Field(..., description="Wall-clock seconds elapsed before termination.")


ShellOutcome = Annotated[
    ShellOutcomeSuccess | ShellOutcomeFailure | ShellOutcomeTimeout,
    Field(discriminator="type"),
]
"""Discriminated union describing how a shell command terminated."""


@json_schema_type
class ShellCallOutput(BaseModel):
    """Captured output of a single shell execution.

    Consumed by the Responses provider to construct ``ShellCallOutputItem``
    entries on the output stream.
    """

    stdout: str = Field(..., description="UTF-8 decoded standard output (truncated by the runtime if oversized).")
    stderr: str = Field(..., description="UTF-8 decoded standard error (truncated by the runtime if oversized).")
    outcome: ShellOutcome = Field(..., description="How the shell process terminated.")
    duration_ms: int = Field(..., ge=0, description="Wall-clock duration of the shell call in milliseconds.")
    container_id: str | None = Field(
        default=None,
        description="ID of the container the call executed in, when applicable. Null for local mode.",
    )


# ---------------------------------------------------------------------------
# ContainerRuntime request models
#
# These describe the internal ContainerRuntime call surface, which is not
# exposed over HTTP. They are deliberately NOT decorated with
# ``@json_schema_type`` so they stay out of the public OpenAPI spec.
# ---------------------------------------------------------------------------


class ExecuteShellRequest(BaseModel):
    """Internal request to run a shell command inside a container."""

    container_id: str = Field(..., description="The ID of the container to execute inside.")
    command: list[str] = Field(..., description="Command argv to execute. Passed without shell expansion.")
    timeout_seconds: float | None = Field(
        default=None, description="Optional wall-clock timeout for the command in seconds."
    )


class MountSkillsRequest(BaseModel):
    """Internal request to mount skill bundles into a container."""

    container_id: str = Field(..., description="The ID of the container to mount skills into.")
    skill_bundles: list[tuple[str, bytes]] = Field(
        ...,
        description=(
            "Skill bundles as (skill_name, zip_bytes) pairs. Each archive is "
            "extracted into /mnt/skills/{skill_name}/ inside the container."
        ),
    )
