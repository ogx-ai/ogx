# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Tools API protocol definitions.

This module contains the ToolGroups and ToolRuntime protocol definitions.
Pydantic models are defined in ogx_api.tools.models.
The FastAPI router is defined in ogx_api.tools.fastapi_routes.
"""

from typing import Any, Protocol

from ogx_api.common.content_types import URL

from .models import (
    ListToolDefsResponse,
    ListToolGroupsResponse,
    ListToolsRequest,
    ToolDef,
    ToolGroup,
    ToolInvocationResult,
    ToolStore,
)


class ToolGroups(Protocol):
    """Protocol for tool group management and tool discovery.

    Tool groups are auto-registered from configured tool_runtime providers.
    Management methods (register, unregister) are internal. Read-only
    discovery endpoints (list_tools, get_tool) are exposed via HTTP so
    clients can discover which built-in tools are available.
    """

    async def register_tool_group(
        self,
        toolgroup_id: str,
        provider_id: str,
        mcp_endpoint: URL | None = None,
        args: dict[str, Any] | None = None,
    ) -> None:
        """Register a tool group."""
        ...

    async def get_tool_group(
        self,
        toolgroup_id: str,
    ) -> ToolGroup:
        """Get a tool group by its ID."""
        ...

    async def list_tool_groups(self) -> ListToolGroupsResponse:
        """List tool groups."""
        ...

    async def list_tools(self, request: ListToolsRequest) -> ListToolDefsResponse:
        """List tools with optional tool group filter."""
        ...

    async def get_tool(
        self,
        tool_name: str,
    ) -> ToolDef:
        """Get a tool by its name."""
        ...

    async def unregister_toolgroup(
        self,
        toolgroup_id: str,
    ) -> None:
        """Unregister a tool group."""
        ...


class ToolRuntime(Protocol):
    """Internal protocol for listing and invoking tools from registered tool groups.

    This protocol is used internally by agents and is not exposed as an HTTP API.
    """

    tool_store: ToolStore | None = None

    async def list_runtime_tools(
        self,
        tool_group_id: str | None = None,
        mcp_endpoint: URL | None = None,
        authorization: str | None = None,
    ) -> ListToolDefsResponse:
        """List all tools in the runtime."""
        ...

    async def invoke_tool(
        self,
        tool_name: str,
        kwargs: dict[str, Any],
        authorization: str | None = None,
    ) -> ToolInvocationResult:
        """Run a tool with the given arguments."""
        ...
