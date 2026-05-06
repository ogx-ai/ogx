# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Connectors API models.

This module contains the Pydantic models for the Connectors API.
"""

from enum import StrEnum

from pydantic import BaseModel, Field

from ogx_api.schema_utils import json_schema_type
from ogx_api.tools import ToolDef


@json_schema_type
class ConnectorType(StrEnum):
    """Type of connector."""

    MCP = "mcp"


class CommonConnectorFields(BaseModel):
    """Common fields for all connectors"""

    connector_type: ConnectorType = Field(default=ConnectorType.MCP)
    connector_id: str = Field(..., description="Identifier for the connector")
    url: str = Field(..., description="URL of the connector")
    server_label: str | None = Field(default=None, description="Label of the server")


@json_schema_type
class Connector(CommonConnectorFields):
    """A connector registered in OGX"""

    model_config = {"populate_by_name": True}
    server_name: str | None = Field(default=None, description="Name of the server")
    server_description: str | None = Field(default=None, description="Description of the server")
    server_version: str | None = Field(default=None, description="Version of the server")


@json_schema_type
class ConnectorInput(CommonConnectorFields):
    """Input for creating a connector"""


# Path parameter models (single field for create_path_dependency)


@json_schema_type
class GetConnectorRequest(BaseModel):
    """Request model for getting a connector by ID."""

    connector_id: str = Field(..., description="Identifier for the connector")


@json_schema_type
class ListConnectorToolsRequest(BaseModel):
    """Request model for listing tools from a connector."""

    connector_id: str = Field(..., description="Identifier for the connector")


@json_schema_type
class GetConnectorToolRequest(BaseModel):
    """Request model for getting a tool from a connector."""

    connector_id: str = Field(..., description="Identifier for the connector")
    tool_name: str = Field(..., description="Name of the tool")


# Response models


@json_schema_type
class ListConnectorsResponse(BaseModel):
    """Response containing a list of configured connectors"""

    data: list[Connector]


@json_schema_type
class ListToolsResponse(BaseModel):
    """Response containing a list of tools"""

    data: list[ToolDef]


__all__ = [
    "ConnectorType",
    "CommonConnectorFields",
    "Connector",
    "ConnectorInput",
    "GetConnectorRequest",
    "ListConnectorsResponse",
    "ListConnectorToolsRequest",
    "ListToolsResponse",
    "GetConnectorToolRequest",
]
