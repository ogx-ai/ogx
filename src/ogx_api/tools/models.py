# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Pydantic models for tool definitions and invocation results.

Tool discovery for MCP connectors is served by the Connectors API; the
Responses API serves MCP tool execution directly. These models are the
shared data types both use.
"""

from typing import Any

from pydantic import BaseModel, Field

from ogx_api.common.content_types import URL, InterleavedContent
from ogx_api.schema_utils import json_schema_type


@json_schema_type
class ToolDef(BaseModel):
    """Tool definition used in runtime contexts."""

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
class ToolInvocationResult(BaseModel):
    """Result of a tool invocation."""

    content: InterleavedContent | None = Field(default=None, description="The content returned by the tool.")
    error_message: str | None = Field(default=None, description="Error message if the tool invocation failed.")
    error_code: int | None = Field(default=None, description="Error code if the tool invocation failed.")
    metadata: dict[str, Any] | None = Field(
        default=None, description="Additional metadata returned by the tool invocation."
    )


@json_schema_type
class ListToolDefsResponse(BaseModel):
    """Response containing a list of tool definitions."""

    data: list[ToolDef] = Field(..., description="List of tool definitions.")


__all__ = [
    "ListToolDefsResponse",
    "ToolDef",
    "ToolGroupInput",
    "ToolInvocationResult",
]
