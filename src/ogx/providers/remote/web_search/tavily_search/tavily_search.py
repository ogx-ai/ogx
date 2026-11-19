# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from typing import Any

import httpx2

from ogx.core.request_headers import NeedsRequestProviderData
from ogx.providers.utils.common.web_search import check_search_response
from ogx.web_search import WebSearch, WebSearchResult, WebSearchSource
from ogx_api import ToolDef

from .config import TavilySearchConfig


class TavilySearchWebSearchImpl(WebSearch, NeedsRequestProviderData):
    """Web search backend using the Tavily API."""

    _CONTEXT_SIZE_TO_COUNT = {"low": 3, "medium": 5, "high": 10}

    # automatically set by the resolver when instantiating the provider
    __provider_id__: str

    def __init__(self, config: TavilySearchConfig):
        self.config = config
        self._client: httpx2.AsyncClient | None = None

    async def initialize(self) -> None:
        self._client = httpx2.AsyncClient(timeout=self.config.to_httpx_timeout())

    async def shutdown(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def available_providers(self) -> list[str]:
        return [self.__provider_id__]

    async def is_available(self) -> bool:
        return self._get_api_key() is not None

    def _get_api_key(self) -> str | None:
        api_key = self.config.api_key.get_secret_value() if self.config.api_key else None

        provider_data = self.get_request_provider_data(quiet=True)
        if provider_data and provider_data.tavily_search_api_key:
            api_key = provider_data.tavily_search_api_key.get_secret_value()

        return api_key

    async def tool_def(self, provider_id: str | None = None) -> ToolDef:
        return ToolDef(
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

    async def search(
        self,
        provider_id: str | None,
        query: str,
        *,
        search_context_size: str | None = None,
        allowed_domains: list[str] | None = None,
        user_location: dict[str, Any] | None = None,
    ) -> WebSearchResult:
        api_key = self._get_api_key()
        request_body: dict[str, Any] = {
            "query": query,
        }
        if api_key:
            request_body["api_key"] = api_key

        if allowed_domains:
            request_body["include_domains"] = allowed_domains

        if search_context_size and search_context_size in self._CONTEXT_SIZE_TO_COUNT:
            request_body["max_results"] = self._CONTEXT_SIZE_TO_COUNT[search_context_size]

        if self._client is None:
            raise RuntimeError("Failed to invoke web search: provider not initialized")
        response = await self._client.post(
            "https://api.tavily.com/search",
            json=request_body,
        )
        check_search_response(response, "Tavily Search")

        response_json = response.json()
        sources = [WebSearchSource(url=r["url"]) for r in response_json.get("results", []) if "url" in r]
        return WebSearchResult(
            content=json.dumps(self._clean_tavily_response(response_json)),
            query=query,
            sources=sources,
        )

    def _clean_tavily_response(self, search_response, top_k=3):
        return {"query": search_response["query"], "top_k": search_response["results"]}
