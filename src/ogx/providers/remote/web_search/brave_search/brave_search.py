# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

import httpx2

from ogx.core.request_headers import NeedsRequestProviderData
from ogx.providers.utils.common.web_search import check_search_response
from ogx.web_search import WebSearch, WebSearchResult, WebSearchSource
from ogx_api import ToolDef

from .config import BraveSearchConfig


class BraveSearchWebSearchImpl(WebSearch, NeedsRequestProviderData):
    """Web search backend using the Brave Search API.

    Each backend owns its model-facing tool definition: the description and
    input schema returned by ``tool_def`` are what the model sees when this
    backend is selected.
    """

    _CONTEXT_SIZE_TO_COUNT = {"low": 3, "medium": 5, "high": 10}

    # automatically set by the resolver when instantiating the provider
    __provider_id__: str

    def __init__(self, config: BraveSearchConfig):
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
        if provider_data and provider_data.brave_search_api_key:
            api_key = provider_data.brave_search_api_key.get_secret_value()

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
        url = f"{str(self.config.base_url).rstrip('/')}/res/v1/web/search"
        headers: dict[str, str] = {
            "Accept-Encoding": "gzip",
            "Accept": "application/json",
        }
        if api_key:
            headers["X-Subscription-Token"] = api_key

        if allowed_domains:
            site_filter = " OR ".join(f"site:{domain}" for domain in allowed_domains)
            query = f"{query} ({site_filter})"

        payload: dict[str, Any] = {"q": query}

        if user_location and user_location.get("country"):
            payload["country"] = user_location["country"]

        result_limit = self.config.max_results
        if search_context_size and search_context_size in self._CONTEXT_SIZE_TO_COUNT:
            result_limit = self._CONTEXT_SIZE_TO_COUNT[search_context_size]
            payload["count"] = result_limit

        if self._client is None:
            raise RuntimeError("Failed to invoke web search: provider not initialized")
        response = await self._client.get(
            url=url,
            params=payload,
            headers=headers,
        )
        check_search_response(response, "Brave Search")
        response_json = response.json()
        results = self._clean_brave_response(response_json, result_limit)
        content = "\n".join([str(result) for result in results])
        sources = [WebSearchSource(url=s["url"]) for s in self._extract_sources(response_json, result_limit)]
        return WebSearchResult(content=content, query=query, sources=sources)

    def _extract_sources(self, search_response: dict[str, Any], max_results: int) -> list[dict[str, str]]:
        sources = []
        if "mixed" in search_response:
            for m in search_response["mixed"]["main"][:max_results]:
                r_type = m["type"]
                if r_type in search_response:
                    results = search_response[r_type].get("results", [])
                    idx = m.get("index")
                    if idx is not None and idx < len(results):
                        item = results[idx]
                    elif isinstance(results, list) and results:
                        item = results[0]
                    else:
                        continue
                    if isinstance(item, dict) and "url" in item:
                        sources.append({"url": item["url"]})
        return sources

    def _clean_brave_response(self, search_response: dict[str, Any], max_results: int) -> list[str]:
        clean_response = []
        if "mixed" in search_response:
            mixed_results = search_response["mixed"]
            for m in mixed_results["main"][:max_results]:
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
