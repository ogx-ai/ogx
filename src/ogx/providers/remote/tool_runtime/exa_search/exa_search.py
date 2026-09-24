# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from typing import Any

import httpx

from ogx.core.request_headers import NeedsRequestProviderData
from ogx_api import (
    URL,
    ListToolDefsResponse,
    ToolDef,
    ToolGroup,
    ToolGroupsProtocolPrivate,
    ToolInvocationResult,
    ToolRuntime,
)

from .config import ExaSearchToolConfig


class ExaSearchToolRuntimeImpl(ToolGroupsProtocolPrivate, ToolRuntime, NeedsRequestProviderData):
    """Tool runtime for performing agent-optimized web searches using the Exa API."""

    _CONTEXT_SIZE_TO_COUNT = {"low": 3, "medium": 5, "high": 10}
    _RESULT_FIELDS = ("title", "url", "publishedDate", "author", "highlights")

    def __init__(self, config: ExaSearchToolConfig):
        self.config = config
        self._client: httpx.AsyncClient | None = None

    async def initialize(self):
        self._client = httpx.AsyncClient(timeout=self.config.to_httpx_timeout())

    async def shutdown(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def register_toolgroup(self, toolgroup: ToolGroup) -> None:
        pass

    async def unregister_toolgroup(self, toolgroup_id: str) -> None:
        return

    def _get_api_key(self) -> str | None:
        api_key = self.config.api_key.get_secret_value() if self.config.api_key else None

        provider_data = self.get_request_provider_data()
        if provider_data and provider_data.exa_search_api_key:
            api_key = provider_data.exa_search_api_key.get_secret_value()

        return api_key

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
                    description="Search the web for information",
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
        api_key = self._get_api_key()
        request_body: dict[str, Any] = {
            "query": kwargs["query"],
            "type": "auto",
            "contents": {"highlights": True},
            "numResults": self.config.max_results,
        }
        headers: dict[str, str] = {}
        if api_key:
            headers["x-api-key"] = api_key

        allowed_domains = kwargs.get("allowed_domains")
        if allowed_domains:
            request_body["includeDomains"] = allowed_domains

        search_context_size = kwargs.get("search_context_size")
        if search_context_size and search_context_size in self._CONTEXT_SIZE_TO_COUNT:
            request_body["numResults"] = self._CONTEXT_SIZE_TO_COUNT[search_context_size]

        user_location = kwargs.get("user_location")
        if user_location and user_location.get("country"):
            request_body["userLocation"] = user_location["country"]

        if self._client is None:
            raise RuntimeError("Failed to invoke tool: provider not initialized")
        response = await self._client.post(
            "https://api.exa.ai/search",
            headers=headers,
            json=request_body,
        )
        response.raise_for_status()

        response_json = response.json()
        top_k = [
            {field: r[field] for field in self._RESULT_FIELDS if field in r} for r in response_json.get("results", [])
        ]
        sources = [{"url": r["url"]} for r in top_k if "url" in r]
        return ToolInvocationResult(
            content=json.dumps({"query": kwargs["query"], "top_k": top_k}),
            metadata={"query": kwargs["query"], "sources": sources},
        )
