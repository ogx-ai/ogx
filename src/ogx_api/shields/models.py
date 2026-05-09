# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Pydantic models for Shields API requests and responses.

This module defines the request and response models for the Shields API
using Pydantic with Field descriptions for OpenAPI schema generation.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

from ogx_api.resource import Resource, ResourceType
from ogx_api.schema_utils import json_schema_type


class CommonShieldFields(BaseModel):
    """Common fields shared across shield creation and retrieval."""

    params: dict[str, Any] | None = None


@json_schema_type
class Shield(CommonShieldFields, Resource):
    """A safety shield resource that can be used to check content."""

    type: Literal[ResourceType.shield] = ResourceType.shield

    @property
    def shield_id(self) -> str:
        return self.identifier

    @property
    def provider_shield_id(self) -> str | None:
        return self.provider_resource_id


class ShieldInput(CommonShieldFields):
    """Input model for registering a new shield."""

    shield_id: str
    provider_id: str | None = None
    provider_shield_id: str | None = None


@json_schema_type
class ListShieldsResponse(BaseModel):
    """Response containing a list of all shields."""

    data: list[Shield] = Field(..., description="List of shield objects")


@json_schema_type
class GetShieldRequest(BaseModel):
    """Request model for getting a shield by identifier."""

    identifier: str = Field(..., description="The identifier of the shield to get.")


@json_schema_type
class RegisterShieldRequest(BaseModel):
    """Request model for registering a shield."""

    shield_id: str = Field(..., description="The identifier of the shield to register.")
    provider_shield_id: str | None = Field(None, description="The identifier of the shield in the provider.")
    provider_id: str | None = Field(None, description="The identifier of the provider.")
    params: dict[str, Any] | None = Field(None, description="The parameters of the shield.")


@json_schema_type
class UnregisterShieldRequest(BaseModel):
    """Request model for unregistering a shield."""

    identifier: str = Field(..., description="The identifier of the shield to unregister.")
