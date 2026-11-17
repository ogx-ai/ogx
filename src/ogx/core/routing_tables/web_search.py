# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, cast

from ogx.web_search import WebSearch, WebSearchResult
from ogx_api import ToolDef

from .common import CommonRoutingTableImpl


class WebSearchRoutingTable(CommonRoutingTableImpl, WebSearch):
    """Routing table for the web_search API.

    Unlike the models / vector_stores / tool_groups tables, web-search
    backends are static config, not registered resources: the routing key is
    the provider_id itself. Lookups are therefore a direct
    ``impls_by_provider_id`` access with no ``dist_registry`` entry
    (cf. ``InferenceRouter._get_provider_by_fallback``).

    Default resolution (client omits provider_id): the first available
    provider in configuration order (keyed backends are available when a key
    is present for the request, the placeholder always is), else the first
    configured provider.
    """

    def _web_search_impls(self) -> dict[str, WebSearch]:
        # The base class keys impls by the routed-protocol union; this table
        # only ever holds web_search backends.
        return cast("dict[str, WebSearch]", self.impls_by_provider_id)

    async def get_provider_impl(self, routing_key: str, provider_id: str | None = None) -> WebSearch:
        impls = self._web_search_impls()
        if routing_key not in impls:
            available = ", ".join(sorted(impls)) or "<none>"
            raise ValueError(f"Failed to find web_search provider `{routing_key}`. Available providers: {available}")
        return impls[routing_key]

    def available_providers(self) -> list[str]:
        return list(self._web_search_impls())

    async def is_available(self) -> bool:
        for impl in self._web_search_impls().values():
            if await impl.is_available():
                return True
        return False

    async def resolve_provider_id(self, provider_id: str | None) -> str:
        if provider_id is not None:
            await self.get_provider_impl(provider_id)  # validates + raises if unknown
            return provider_id
        impls = self._web_search_impls()
        for pid, impl in impls.items():
            if await impl.is_available():
                return pid
        if not impls:
            raise ValueError("Failed to find web_search provider: no web_search providers are configured")
        return next(iter(impls))

    async def tool_def(self, provider_id: str | None = None) -> ToolDef:
        pid = await self.resolve_provider_id(provider_id)
        provider = await self.get_provider_impl(pid)
        return await provider.tool_def(pid)

    async def search(
        self,
        provider_id: str | None,
        query: str,
        *,
        search_context_size: str | None = None,
        allowed_domains: list[str] | None = None,
        user_location: dict[str, Any] | None = None,
    ) -> WebSearchResult:
        pid = await self.resolve_provider_id(provider_id)
        provider = await self.get_provider_impl(pid)
        return await provider.search(
            pid,
            query,
            search_context_size=search_context_size,
            allowed_domains=allowed_domains,
            user_location=user_location,
        )
