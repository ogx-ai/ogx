# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any
from urllib.parse import urlparse

from ogx.core.request_headers import NeedsRequestProviderData
from ogx.log import get_logger
from ogx.providers.utils.tools.mcp import invoke_mcp_tool, list_mcp_tools
from ogx_api import (
    URL,
    Api,
    ListToolDefsResponse,
    ToolGroup,
    ToolGroupsProtocolPrivate,
    ToolInvocationResult,
    ToolRuntime,
)

from .config import MCPProviderConfig

logger = get_logger(__name__, category="tools")


class ModelContextProtocolToolRuntimeImpl(ToolGroupsProtocolPrivate, ToolRuntime, NeedsRequestProviderData):
    """Tool runtime for discovering and invoking tools via the Model Context Protocol."""

    def __init__(self, config: MCPProviderConfig, _deps: dict[Api, Any]):
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
        # this endpoint should be retrieved by getting the tool group right?
        if mcp_endpoint is None:
            raise ValueError("mcp_endpoint is required")

        # Get other headers from provider data (but NOT authorization)
        provider_headers = await self.get_headers_from_request(mcp_endpoint.uri)

        return await list_mcp_tools(endpoint=mcp_endpoint.uri, headers=provider_headers, authorization=authorization)

    async def invoke_tool(
        self, tool_name: str, kwargs: dict[str, Any], authorization: str | None = None
    ) -> ToolInvocationResult:
        tool = await self.tool_store.get_tool(tool_name)
        if tool.metadata is None or tool.metadata.get("endpoint") is None:
            raise ValueError(f"Tool {tool_name} does not have metadata")
        endpoint = tool.metadata.get("endpoint")
        if urlparse(endpoint).scheme not in ("http", "https"):
            raise ValueError(f"Endpoint {endpoint} is not a valid HTTP(S) URL")

        # Get other headers from provider data (but NOT authorization)
        provider_headers = await self.get_headers_from_request(endpoint)

        return await invoke_mcp_tool(
            endpoint=endpoint,
            tool_name=tool_name,
            kwargs=kwargs,
            headers=provider_headers,
            authorization=authorization,
        )

    async def get_headers_from_request(self, mcp_endpoint_uri: str) -> dict[str, str]:
        """
        Extract headers from request provider data, excluding authorization.

        Authorization must be provided via the dedicated authorization parameter.
        If Authorization is found in mcp_headers, raise an error to guide users to the correct approach.

        Args:
            mcp_endpoint_uri: The MCP endpoint URI to match against provider data

        Returns:
            dict[str, str]: Headers dictionary (without Authorization)

        Raises:
            ValueError: If Authorization header is found in mcp_headers
        """

        def canonicalize_uri(uri: str) -> str:
            return f"{urlparse(uri).netloc or ''}/{urlparse(uri).path or ''}"

        headers = {}

        provider_data = self.get_request_provider_data()
        if provider_data and hasattr(provider_data, "mcp_headers") and provider_data.mcp_headers:
            for uri, values in provider_data.mcp_headers.items():
                if canonicalize_uri(uri) != canonicalize_uri(mcp_endpoint_uri):
                    continue

                # Reject Authorization in mcp_headers - must use authorization parameter
                for key in values.keys():
                    if key.lower() == "authorization":
                        raise ValueError(
                            "Authorization cannot be provided via mcp_headers in provider_data. "
                            "Please use the dedicated 'authorization' parameter instead. "
                            "Example: tool_runtime.invoke_tool(..., authorization='your-token')"
                        )
                    headers[key] = values[key]

        return headers
