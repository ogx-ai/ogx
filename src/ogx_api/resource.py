# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import StrEnum

from pydantic import BaseModel, Field


class ResourceType(StrEnum):
    """Enumeration of all resource types managed by OGX."""

    model = "model"
    shield = "shield"
    vector_store = "vector_store"
    tool = "tool"
    tool_group = "tool_group"
    prompt = "prompt"


class Resource(BaseModel):
    """Base class for all OGX resources"""

    identifier: str = Field(description="Unique identifier for this resource in ogx")

    provider_resource_id: str | None = Field(
        default=None,
        description="Unique identifier for this resource in the provider",
    )

    provider_id: str = Field(description="ID of the provider that owns this resource")

    type: ResourceType = Field(description="Type of resource (e.g. 'model', 'shield', 'vector_store', etc.)")
