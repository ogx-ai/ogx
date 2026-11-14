# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from typing import Any

import httpx2

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

from .config import SearchApiSearchToolConfig


class SearchApiSearchToolRuntimeImpl(ToolGroupsProtocolPrivate, ToolRuntime, NeedsRequestProviderData):
    """Tool runtime for performing web searches using the SearchApi Google Search API."""

    _SEARCH_URL = "https://www.searchapi.io/api/v1/search"
    _CONTEXT_SIZE_TO_COUNT = {"low": 3, "medium": 5, "high": 10}
    # Google phased out the `num` parameter in September 2025, so one request returns
    # one results page of at most 10 organic rows. `num` is not sent; the page is
    # trimmed to the requested count client-side instead.
    _MAX_RESULTS = 10

    def __init__(self, config: SearchApiSearchToolConfig):
        self.config = config
        self._client: httpx2.AsyncClient | None = None

    async def initialize(self) -> None:
        self._client = httpx2.AsyncClient(timeout=self.config.to_httpx_timeout())

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
        if provider_data and provider_data.searchapi_search_api_key:
            api_key = str(provider_data.searchapi_search_api_key.get_secret_value())

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
        query = kwargs["query"]

        # SearchApi has no dedicated domain filter, so the restriction goes into the
        # query itself as Google `site:` operators.
        allowed_domains = kwargs.get("allowed_domains")
        if allowed_domains:
            site_filter = " OR ".join(f"site:{domain}" for domain in allowed_domains)
            query = f"{query} ({site_filter})"

        result_limit = self.config.max_results
        search_context_size = kwargs.get("search_context_size")
        if search_context_size and search_context_size in self._CONTEXT_SIZE_TO_COUNT:
            result_limit = self._CONTEXT_SIZE_TO_COUNT[search_context_size]
        result_limit = max(1, min(result_limit, self._MAX_RESULTS))

        params: dict[str, Any] = {"engine": "google", "q": query}

        # Geo-targeting is per-request user-supplied context, not server config.
        # `gl` is the documented country parameter of the Google engine.
        user_location = kwargs.get("user_location")
        if user_location and user_location.get("country"):
            params["gl"] = str(user_location["country"]).lower()

        if self._client is None:
            raise RuntimeError("Failed to invoke tool: provider not initialized")
        headers = {"Accept": "application/json", "User-Agent": "ogx"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        response = await self._client.get(
            self._SEARCH_URL,
            params=params,
            headers=headers,
        )
        if response.status_code == 401:
            # A missing or invalid key. Return the reason as tool content so the
            # executor forwards it to the model rather than a generic failure.
            message = "Failed to query SearchApi Search: the API key is missing or invalid."
            return ToolInvocationResult(
                content=json.dumps({"error": message}),
                error_code=401,
                error_message=message,
                metadata={"query": kwargs["query"], "sources": []},
            )
        response.raise_for_status()

        response_json = response.json()
        results = []
        sources = []
        for r in response_json.get("organic_results", [])[:result_limit]:
            url = r.get("link")
            results.append(
                {
                    "title": r.get("title", ""),
                    "url": url,
                    "content": r.get("snippet", ""),
                }
            )
            if url:
                sources.append({"url": url})

        return ToolInvocationResult(
            content=json.dumps({"query": kwargs["query"], "results": results}),
            metadata={"query": kwargs["query"], "sources": sources},
        )
