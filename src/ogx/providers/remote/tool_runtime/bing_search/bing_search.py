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

from .config import BingSearchToolConfig


class BingSearchToolRuntimeImpl(ToolGroupsProtocolPrivate, ToolRuntime, NeedsRequestProviderData):
    """Tool runtime for performing web searches using the Bing Search API."""

    _CONTEXT_SIZE_TO_COUNT = {"low": 3, "medium": 5, "high": 10}

    def __init__(self, config: BingSearchToolConfig):
        self.config = config
        self.url = "https://api.bing.microsoft.com/v7.0/search"
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

    def _get_api_key(self) -> str:
        if self.config.api_key:
            return self.config.api_key

        provider_data = self.get_request_provider_data()
        if provider_data is None or not provider_data.bing_search_api_key:
            raise ValueError(
                'Pass Bing Search API Key in the header X-OGX-Provider-Data as { "bing_search_api_key": <your api key>}'
            )
        return provider_data.bing_search_api_key.get_secret_value()

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
                    description="Search the web using Bing Search API",
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
        headers = {
            "Ocp-Apim-Subscription-Key": api_key,
        }

        query = kwargs["query"]

        allowed_domains = kwargs.get("allowed_domains")
        if allowed_domains:
            site_filter = " OR ".join(f"site:{domain}" for domain in allowed_domains)
            query = f"{query} ({site_filter})"

        params: dict[str, Any] = {
            "count": self.config.top_k,
            "textDecorations": True,
            "textFormat": "HTML",
            "q": query,
        }

        user_location = kwargs.get("user_location")
        if user_location and user_location.get("country"):
            params["cc"] = user_location["country"]

        search_context_size = kwargs.get("search_context_size")
        if search_context_size and search_context_size in self._CONTEXT_SIZE_TO_COUNT:
            params["count"] = self._CONTEXT_SIZE_TO_COUNT[search_context_size]

        if self._client is None:
            raise RuntimeError("Failed to invoke tool: provider not initialized")
        response = await self._client.get(
            url=self.url,
            params=params,
            headers=headers,
        )
        response.raise_for_status()

        response_json = response.json()
        sources = []
        if "webPages" in response_json:
            for page in response_json["webPages"]["value"]:
                if "url" in page:
                    sources.append({"url": page["url"]})
        return ToolInvocationResult(
            content=json.dumps(self._clean_response(response_json)),
            metadata={"query": kwargs["query"], "sources": sources},
        )

    def _clean_response(self, search_response):
        clean_response = []
        query = search_response["queryContext"]["originalQuery"]
        if "webPages" in search_response:
            pages = search_response["webPages"]["value"]
            for p in pages:
                selected_keys = {"name", "url", "snippet"}
                clean_response.append({k: v for k, v in p.items() if k in selected_keys})
        if "news" in search_response:
            clean_news = []
            news = search_response["news"]["value"]
            for n in news:
                selected_keys = {"name", "url", "description"}
                clean_news.append({k: v for k, v in n.items() if k in selected_keys})

            clean_response.append(clean_news)

        return {"query": query, "top_k": clean_response}
