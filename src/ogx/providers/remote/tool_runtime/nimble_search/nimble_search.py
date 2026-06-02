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

from .config import NimbleSearchToolConfig


class NimbleSearchToolRuntimeImpl(ToolGroupsProtocolPrivate, ToolRuntime, NeedsRequestProviderData):
    """Tool runtime for performing web searches using the Nimble Search API."""

    _SEARCH_URL = "https://sdk.nimbleway.com/v1/search"
    _CONTEXT_SIZE_TO_COUNT = {"low": 3, "medium": 5, "high": 10}

    def __init__(self, config: NimbleSearchToolConfig):
        self.config = config
        self._client: httpx.AsyncClient | None = None

    async def initialize(self) -> None:
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
        if provider_data is None or not provider_data.nimble_search_api_key:
            raise ValueError(
                'Failed to get Nimble API key: pass it in the header X-OGX-Provider-Data as { "nimble_search_api_key": <your api key>}'
            )
        return str(provider_data.nimble_search_api_key.get_secret_value())

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
        # Geo-targeting is configured statically via country/locale; the per-request
        # user_location kwarg is intentionally not forwarded.
        request_body: dict[str, Any] = {
            "query": kwargs["query"],
            "max_results": self.config.max_results,
            "search_depth": self.config.search_depth,
            "country": self.config.country,
            "locale": self.config.locale,
        }

        allowed_domains = kwargs.get("allowed_domains")
        if allowed_domains:
            request_body["include_domains"] = allowed_domains

        search_context_size = kwargs.get("search_context_size")
        if search_context_size and search_context_size in self._CONTEXT_SIZE_TO_COUNT:
            request_body["max_results"] = self._CONTEXT_SIZE_TO_COUNT[search_context_size]

        if self._client is None:
            raise RuntimeError("Failed to invoke tool: provider not initialized")
        response = await self._client.post(
            self._SEARCH_URL,
            json=request_body,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        if response.status_code == 403:
            # The account is not entitled to the requested capability (e.g. a higher
            # search_depth tier). Surface a clear tool error instead of raising.
            return ToolInvocationResult(
                error_code=403,
                error_message=(
                    f"Failed to query Nimble Search: the account is not entitled to "
                    f"search_depth={self.config.search_depth!r}. Use search_depth='lite' "
                    f"or contact Nimble to enable higher tiers."
                ),
                metadata={"query": kwargs["query"], "sources": []},
            )
        response.raise_for_status()

        response_json = response.json()
        results = []
        sources = []
        for r in response_json.get("results", []):
            url = r.get("url")
            # In 'lite' depth the API returns metadata only (empty content); the
            # description carries the result text, so fall back to it.
            results.append(
                {
                    "title": r.get("title", ""),
                    "url": url,
                    "content": r.get("content") or r.get("description", ""),
                }
            )
            if url:
                sources.append({"url": url})

        return ToolInvocationResult(
            content=json.dumps({"query": kwargs["query"], "results": results}),
            metadata={"query": kwargs["query"], "sources": sources},
        )
