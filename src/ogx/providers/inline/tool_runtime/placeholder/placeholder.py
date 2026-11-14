# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from ogx_api import (
    URL,
    ListToolDefsResponse,
    ToolDef,
    ToolGroup,
    ToolGroupsProtocolPrivate,
    ToolInvocationResult,
    ToolRuntime,
)

from .config import PlaceholderWebSearchToolConfig


class PlaceholderWebSearchToolRuntimeImpl(ToolGroupsProtocolPrivate, ToolRuntime):
    """Tool runtime that hosts web_search without a search backend.

    Invocations return a notice that web search is not configured instead of
    failing, so clients that send the web_search tool unconditionally (e.g.
    Codex) degrade gracefully. The stack auto-registers this provider when no
    real web search provider is configured.
    """

    NOT_AVAILABLE_MESSAGE = (
        "Web search is not available on this server. No results were retrieved. "
        "Configure a web search provider to enable it. Answer using your existing "
        "knowledge, or tell the user that web search is not configured."
    )

    def __init__(self, config: PlaceholderWebSearchToolConfig):
        self.config = config

    async def initialize(self):
        pass

    async def register_toolgroup(self, toolgroup: ToolGroup) -> None:
        pass

    async def unregister_toolgroup(self, toolgroup_id: str) -> None:
        return

    async def list_runtime_tools(
        self,
        tool_group_id: str | None = None,
        mcp_endpoint: URL | None = None,
        authorization: str | None = None,
    ) -> ListToolDefsResponse:
        return ListToolDefsResponse(
            data=[
                ToolDef(
                    name="web_search",
                    description=(
                        "Search the web for information. Note: web search is not configured "
                        "on this server; calling this tool returns a notice instead of results."
                    ),
                    input_schema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The query to search for",
                            }
                        },
                        "required": ["query"],
                    },
                )
            ]
        )

    async def invoke_tool(
        self, tool_name: str, kwargs: dict[str, Any], authorization: str | None = None
    ) -> ToolInvocationResult:
        return ToolInvocationResult(
            content=self.NOT_AVAILABLE_MESSAGE,
            metadata={"query": kwargs.get("query"), "sources": []},
        )
