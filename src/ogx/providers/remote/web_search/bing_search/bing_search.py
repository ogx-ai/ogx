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

from .config import BingSearchConfig


class BingSearchWebSearchImpl(WebSearch, NeedsRequestProviderData):
    """Web search backend using the Bing Search API."""

    _CONTEXT_SIZE_TO_COUNT = {"low": 3, "medium": 5, "high": 10}
    _SEARCH_URL = "https://api.bing.microsoft.com/v7.0/search"

    # automatically set by the resolver when instantiating the provider
    __provider_id__: str

    def __init__(self, config: BingSearchConfig):
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
        if provider_data and provider_data.bing_search_api_key:
            api_key = provider_data.bing_search_api_key.get_secret_value()

        return api_key

    async def tool_def(self, provider_id: str | None = None) -> ToolDef:
        return ToolDef(
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
        headers: dict[str, str] = {}
        if api_key:
            headers["Ocp-Apim-Subscription-Key"] = api_key

        if allowed_domains:
            site_filter = " OR ".join(f"site:{domain}" for domain in allowed_domains)
            query = f"{query} ({site_filter})"

        params: dict[str, Any] = {
            "count": self.config.top_k,
            "textDecorations": True,
            "textFormat": "HTML",
            "q": query,
        }

        if user_location and user_location.get("country"):
            params["cc"] = user_location["country"]

        if search_context_size and search_context_size in self._CONTEXT_SIZE_TO_COUNT:
            params["count"] = self._CONTEXT_SIZE_TO_COUNT[search_context_size]

        if self._client is None:
            raise RuntimeError("Failed to invoke web search: provider not initialized")
        response = await self._client.get(
            url=self._SEARCH_URL,
            params=params,
            headers=headers,
        )
        check_search_response(response, "Bing Search")

        response_json = response.json()
        sources: list[str] = []
        if "webPages" in response_json:
            for page in response_json["webPages"]["value"]:
                if "url" in page:
                    sources.append(page["url"])
        return WebSearchResult(
            content=json.dumps(self._clean_response(response_json)),
            query=query,
            sources=[WebSearchSource(url=s) for s in sources],
        )

    def _clean_response(self, search_response: dict[str, Any]) -> dict[str, Any]:
        clean_response: list[Any] = []
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
