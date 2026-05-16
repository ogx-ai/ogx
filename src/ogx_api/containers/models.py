# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Annotated, Literal

from pydantic import BaseModel, Field, WithJsonSchema

from ogx_api.common.responses import Order
from ogx_api.schema_utils import json_schema_type


@json_schema_type
class ContainerExpiresAfter(BaseModel):
    """Expiration settings for a container."""

    anchor: Literal["last_active_at"] = Field(
        default="last_active_at",
        description="The reference point for the expiration. Currently only 'last_active_at' is supported.",
    )
    minutes: int = Field(..., description="The number of minutes after the anchor before the container expires.")


@json_schema_type
class NetworkPolicyDisabled(BaseModel):
    """Disabled network policy. No outbound network access."""

    type: Literal["disabled"] = Field(default="disabled", description="Disable outbound network access.")


@json_schema_type
class NetworkPolicyAllowlist(BaseModel):
    """Allowlist network policy. Only specified domains are reachable."""

    type: Literal["allowlist"] = Field(default="allowlist", description="Allow outbound access to specified domains.")
    allowed_domains: list[str] = Field(..., description="A list of allowed outbound domains.")


ContainerNetworkPolicy = Annotated[
    NetworkPolicyDisabled | NetworkPolicyAllowlist,
    Field(discriminator="type"),
]


@json_schema_type
class Container(BaseModel):
    """A sandboxed container environment."""

    id: str = Field(..., description="Unique identifier for the container.")
    object: Literal["container"] = Field(default="container", description="The object type, always 'container'.")
    name: str = Field(..., description="Name of the container.")
    created_at: int = Field(..., description="Unix timestamp (in seconds) when the container was created.")
    status: str = Field(..., description="Status of the container (e.g., running, stopped, deleted).")
    last_active_at: Annotated[int | None, WithJsonSchema({"type": "integer"})] = Field(
        default=None, description="Unix timestamp (in seconds) when the container was last active."
    )
    expires_after: ContainerExpiresAfter | None = Field(
        default=None, description="The container expiration policy."
    )
    memory_limit: Literal["1g", "4g", "16g", "64g"] | None = Field(
        default=None, description="The memory limit configured for the container."
    )
    network_policy: NetworkPolicyDisabled | NetworkPolicyAllowlist | None = Field(
        default=None, description="Network access policy for the container."
    )


@json_schema_type
class CreateContainerRequest(BaseModel):
    """Request model for creating a container."""

    name: str = Field(..., description="Name of the container to create.")
    expires_after: ContainerExpiresAfter | None = Field(
        default=None, description="Container expiration policy."
    )
    memory_limit: Literal["1g", "4g", "16g", "64g"] | None = Field(
        default=None, description="Memory limit for the container. Defaults to '1g'."
    )
    network_policy: NetworkPolicyDisabled | NetworkPolicyAllowlist | None = Field(
        default=None, description="Network access policy for the container."
    )


@json_schema_type
class ListContainersRequest(BaseModel):
    """Request model for listing containers."""

    limit: int | None = Field(default=20, description="Maximum number of containers to return (1-100).")
    order: Order | None = Field(default=Order.desc, description="Sort order by created_at timestamp.")
    after: str | None = Field(default=None, description="Cursor for pagination. Returns containers after this ID.")
    name: str | None = Field(default=None, description="Filter results by container name.")


@json_schema_type
class ListContainersResponse(BaseModel):
    """Response for listing containers."""

    data: list[Container] = Field(..., description="The list of containers.")
    has_more: bool = Field(..., description="Whether there are more containers available.")
    first_id: str = Field(..., description="The ID of the first container in the list.")
    last_id: str = Field(..., description="The ID of the last container in the list.")
    object: Literal["list"] = Field(default="list", description="The object type, always 'list'.")


@json_schema_type
class RetrieveContainerRequest(BaseModel):
    """Request model for retrieving a container."""

    container_id: str = Field(..., description="The ID of the container to retrieve.")


@json_schema_type
class DeleteContainerRequest(BaseModel):
    """Request model for deleting a container."""

    container_id: str = Field(..., description="The ID of the container to delete.")


@json_schema_type
class DeleteContainerResponse(BaseModel):
    """Response for deleting a container."""

    id: str = Field(..., description="The container identifier that was deleted.")
    object: Literal["container.deleted"] = Field(
        default="container.deleted", description="The object type, always 'container.deleted'."
    )
    deleted: bool = Field(default=True, description="Whether the container was successfully deleted.")
