# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

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

from .config import BraveSearchToolConfig


class BraveSearchToolRuntimeImpl(ToolGroupsProtocolPrivate, ToolRuntime, NeedsRequestProviderData):
    """Tool runtime for performing web searches using the Brave Search API."""

    def __init__(self, config: BraveSearchToolConfig):
        self.config = config

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
        if provider_data is None or not provider_data.brave_search_api_key:
            raise ValueError(
                'Pass Search provider\'s API Key in the header X-OGX-Provider-Data as { "brave_search_api_key": <your api key>}'
            )
        return provider_data.brave_search_api_key.get_secret_value()

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
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "X-Subscription-Token": api_key,
            "Accept-Encoding": "gzip",
            "Accept": "application/json",
        }
        payload = {"q": kwargs["query"]}
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url=url,
                params=payload,
                headers=headers,
            )
            response.raise_for_status()
        results = self._clean_brave_response(response.json())
        content_items = "\n".join([str(result) for result in results])
        return ToolInvocationResult(
            content=content_items,
        )

    def _clean_brave_response(self, search_response):
        clean_response = []
        if "mixed" in search_response:
            mixed_results = search_response["mixed"]
            for m in mixed_results["main"][: self.config.max_results]:
                r_type = m["type"]
                results = search_response[r_type]["results"]
                cleaned = self._clean_result_by_type(r_type, results, m.get("index"))
                clean_response.append(cleaned)

        return clean_response

    def _clean_result_by_type(self, r_type, results, idx=None):
        type_cleaners = {
            "web": (
                ["type", "title", "url", "description", "date", "extra_snippets"],
                lambda x: x[idx],
            ),
            "faq": (["type", "question", "answer", "title", "url"], lambda x: x),
            "infobox": (
                ["type", "title", "url", "description", "long_desc"],
                lambda x: x[idx],
            ),
            "videos": (["type", "url", "title", "description", "date"], lambda x: x),
            "locations": (
                [
                    "type",
                    "title",
                    "url",
                    "description",
                    "coordinates",
                    "postal_address",
                    "contact",
                    "rating",
                    "distance",
                    "zoom_level",
                ],
                lambda x: x,
            ),
            "news": (["type", "title", "url", "description"], lambda x: x),
        }

        if r_type not in type_cleaners:
            return ""

        selected_keys, result_selector = type_cleaners[r_type]
        results = result_selector(results)

        if isinstance(results, list):
            cleaned = [{k: v for k, v in item.items() if k in selected_keys} for item in results]
        else:
            cleaned = {k: v for k, v in results.items() if k in selected_keys}

        return str(cleaned)
