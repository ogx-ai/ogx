# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Pydantic models for Tools API requests and responses.

This module defines the request and response models for the Tools API
using Pydantic with Field descriptions for OpenAPI schema generation.
"""

from enum import Enum
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from ogx_api.common.content_types import URL, InterleavedContent
from ogx_api.resource import Resource, ResourceType
from ogx_api.schema_utils import json_schema_type


@json_schema_type
class ToolDef(BaseModel):
    """Tool definition used in runtime contexts."""

    toolgroup_id: str | None = Field(default=None, description="The ID of the tool group this tool belongs to.")
    name: str = Field(..., description="The name of the tool.")
    description: str | None = Field(default=None, description="A human-readable description of what the tool does.")
    input_schema: dict[str, Any] | None = Field(
        default=None, description="JSON Schema describing the tool's input parameters."
    )
    output_schema: dict[str, Any] | None = Field(
        default=None, description="JSON Schema describing the tool's output format."
    )
    metadata: dict[str, Any] | None = Field(default=None, description="Additional metadata associated with the tool.")


@json_schema_type
class ToolGroupInput(BaseModel):
    """Input data for registering a tool group."""

    toolgroup_id: str = Field(..., description="The unique identifier for the tool group.")
    provider_id: str = Field(..., description="The ID of the provider that serves this tool group.")
    args: dict[str, Any] | None = Field(default=None, description="Additional arguments to pass to the provider.")
    mcp_endpoint: URL | None = Field(
        default=None, description="Model Context Protocol endpoint URL for remote tool groups."
    )


@json_schema_type
class ToolGroup(Resource):
    """A group of related tools managed together."""

    type: Literal[ResourceType.tool_group] = ResourceType.tool_group
    mcp_endpoint: URL | None = Field(
        default=None, description="Model Context Protocol endpoint URL for remote tool groups."
    )
    args: dict[str, Any] | None = Field(default=None, description="Additional arguments for the tool group.")


@json_schema_type
class ToolInvocationResult(BaseModel):
    """Result of a tool invocation."""

    content: InterleavedContent | None = Field(default=None, description="The content returned by the tool.")
    error_message: str | None = Field(default=None, description="Error message if the tool invocation failed.")
    error_code: int | None = Field(default=None, description="Error code if the tool invocation failed.")
    metadata: dict[str, Any] | None = Field(
        default=None, description="Additional metadata returned by the tool invocation."
    )


class ToolStore(Protocol):
    """Protocol for accessing tool and tool group definitions."""

    async def get_tool(self, tool_name: str) -> ToolDef: ...
    async def get_tool_group(self, toolgroup_id: str) -> ToolGroup: ...


@json_schema_type
class ListToolGroupsResponse(BaseModel):
    """Response containing a list of tool groups."""

    data: list[ToolGroup] = Field(..., description="List of tool groups.")


@json_schema_type
class ListToolDefsResponse(BaseModel):
    """Response containing a list of tool definitions."""

    data: list[ToolDef] = Field(..., description="List of tool definitions.")


@json_schema_type
class ListToolsRequest(BaseModel):
    """Request model for listing tools."""

    toolgroup_id: str | None = Field(default=None, description="The ID of the tool group to filter tools by.")


class SpecialToolGroup(Enum):
    """Special tool groups with predefined functionality.

    :cvar rag_tool: Retrieval-Augmented Generation tool group for document search and retrieval
    """

    rag_tool = "rag_tool"


__all__ = [
    "ListToolDefsResponse",
    "ListToolGroupsResponse",
    "ListToolsRequest",
    "SpecialToolGroup",
    "ToolDef",
    "ToolGroup",
    "ToolGroupInput",
    "ToolInvocationResult",
    "ToolStore",
]
