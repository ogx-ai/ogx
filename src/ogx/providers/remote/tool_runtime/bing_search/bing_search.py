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

    def __init__(self, config: BingSearchToolConfig):
        self.config = config
        self.url = "https://api.bing.microsoft.com/v7.0/search"

    async def initialize(self):
        pass

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
        params = {
            "count": self.config.top_k,
            "textDecorations": True,
            "textFormat": "HTML",
            "q": kwargs["query"],
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(
                url=self.url,
                params=params,
                headers=headers,
            )
            response.raise_for_status()

        return ToolInvocationResult(content=json.dumps(self._clean_response(response.json())))

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
