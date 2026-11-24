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

from .config import NimbleSearchConfig


class NimbleSearchWebSearchImpl(WebSearch, NeedsRequestProviderData):
    """Web search backend using the Nimble Search API."""

    _SEARCH_URL = "https://sdk.nimbleway.com/v1/search"
    _CONTEXT_SIZE_TO_COUNT = {"low": 3, "medium": 5, "high": 10}

    # automatically set by the resolver when instantiating the provider
    __provider_id__: str

    def __init__(self, config: NimbleSearchConfig):
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
        if provider_data and provider_data.nimble_search_api_key:
            api_key = str(provider_data.nimble_search_api_key.get_secret_value())

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
            "max_results": self.config.max_results,
            "search_depth": self.config.search_depth,
        }

        if allowed_domains:
            request_body["include_domains"] = allowed_domains

        # Geo-targeting is per-request user-supplied context, not server config.
        if user_location and user_location.get("country"):
            request_body["country"] = user_location["country"]

        if search_context_size and search_context_size in self._CONTEXT_SIZE_TO_COUNT:
            request_body["max_results"] = self._CONTEXT_SIZE_TO_COUNT[search_context_size]

        if self._client is None:
            raise RuntimeError("Failed to invoke web search: provider not initialized")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        response = await self._client.post(
            self._SEARCH_URL,
            json=request_body,
            headers=headers,
        )
        # Only a non-default search_depth can be tier-gated; a 403 under the
        # default 'lite' depth is a generic entitlement failure, so the
        # vendor-specific reason applies only to the higher tiers.
        forbidden_detail = None
        if self.config.search_depth != "lite":
            forbidden_detail = (
                f"the account is not entitled to search_depth={self.config.search_depth!r}; "
                "use search_depth='lite' or contact Nimble to enable higher tiers"
            )
        check_search_response(response, "Nimble Search", forbidden_detail=forbidden_detail)

        response_json = response.json()
        results = []
        sources: list[WebSearchSource] = []
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
                sources.append(WebSearchSource(url=url))

        return WebSearchResult(
            content=json.dumps({"query": query, "results": results}),
            query=query,
            sources=sources,
        )
