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

from .config import SearchApiSearchConfig


class SearchApiSearchWebSearchImpl(WebSearch, NeedsRequestProviderData):
    """Web search backend using the SearchApi Google Search API."""

    _SEARCH_URL = "https://www.searchapi.io/api/v1/search"
    _CONTEXT_SIZE_TO_COUNT = {"low": 3, "medium": 5, "high": 10}
    # Google phased out the `num` parameter in September 2025, so one request returns
    # one results page of at most 10 organic rows. `num` is not sent; the page is
    # trimmed to the requested count client-side instead.
    _MAX_RESULTS = 10

    # automatically set by the resolver when instantiating the provider
    __provider_id__: str

    def __init__(self, config: SearchApiSearchConfig):
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
        if provider_data and provider_data.searchapi_search_api_key:
            api_key = str(provider_data.searchapi_search_api_key.get_secret_value())

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

        # SearchApi has no dedicated domain filter, so the restriction goes into the
        # query itself as Google `site:` operators.
        if allowed_domains:
            site_filter = " OR ".join(f"site:{domain}" for domain in allowed_domains)
            query = f"{query} ({site_filter})"

        result_limit = self.config.max_results
        if search_context_size and search_context_size in self._CONTEXT_SIZE_TO_COUNT:
            result_limit = self._CONTEXT_SIZE_TO_COUNT[search_context_size]
        result_limit = max(1, min(result_limit, self._MAX_RESULTS))

        params: dict[str, Any] = {"engine": "google", "q": query}

        # Geo-targeting is per-request user-supplied context, not server config.
        # `gl` is the documented country parameter of the Google engine.
        if user_location and user_location.get("country"):
            params["gl"] = str(user_location["country"]).lower()

        if self._client is None:
            raise RuntimeError("Failed to invoke web search: provider not initialized")
        headers = {"Accept": "application/json", "User-Agent": "ogx"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        response = await self._client.get(
            self._SEARCH_URL,
            params=params,
            headers=headers,
        )
        check_search_response(response, "SearchApi Search")

        response_json = response.json()
        results = []
        sources: list[WebSearchSource] = []
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
                sources.append(WebSearchSource(url=url))

        return WebSearchResult(
            content=json.dumps({"query": query, "results": results}),
            query=query,
            sources=sources,
        )
