# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from ogx.web_search import WebSearch, WebSearchResult
from ogx_api import ToolDef

from .config import PlaceholderWebSearchConfig


class PlaceholderWebSearchImpl(WebSearch):
    """Web search backend without a real search engine.

    Invocations return a notice that web search is not configured instead of
    failing, so clients that send the web_search tool unconditionally (e.g.
    Codex) degrade gracefully. Distributions include this provider as the
    fallback backend when no real web search provider is configured.
    """

    NOT_AVAILABLE_MESSAGE = (
        "Web search is not available on this server. No results were retrieved. "
        "Configure a web search provider to enable it. Answer using your existing "
        "knowledge, or tell the user that web search is not configured."
    )

    # automatically set by the resolver when instantiating the provider
    __provider_id__: str

    def __init__(self, config: PlaceholderWebSearchConfig):
        self.config = config

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    def available_providers(self) -> list[str]:
        return [self.__provider_id__]

    async def is_available(self) -> bool:
        # The placeholder always answers (with a "not configured" notice), so it
        # is always available, even though it never has an API key.
        return True

    async def tool_def(self, provider_id: str | None = None) -> ToolDef:
        return ToolDef(
            name="web_search",
            description=(
                "Search the web for information. Note: web search is not configured "
                "on this server; calling this tool returns a notice instead of results."
            ),
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
        return WebSearchResult(content=self.NOT_AVAILABLE_MESSAGE, query=query)
