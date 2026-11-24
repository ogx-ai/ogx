# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import time
from typing import Any

from ogx.log import get_logger
from ogx.telemetry.tool_metrics import tool_duration, tool_invocations_total
from ogx.web_search import WebSearch, WebSearchResult
from ogx_api import ToolDef

from ..routing_tables.web_search import WebSearchRoutingTable

log = get_logger(name=__name__, category="core")


class WebSearchRouter(WebSearch):
    """Router for the web_search API.

    Dispatches each request to the selected backend via the routing table
    (routing key: provider_id) and records tool-invocation metrics.
    """

    def __init__(self, routing_table: WebSearchRoutingTable) -> None:
        self.routing_table = routing_table

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        await self.routing_table.shutdown()

    def available_providers(self) -> list[str]:
        return self.routing_table.available_providers()

    async def is_available(self) -> bool:
        return await self.routing_table.is_available()

    async def tool_def(self, provider_id: str | None = None) -> ToolDef:
        return await self.routing_table.tool_def(provider_id)

    async def search(
        self,
        provider_id: str | None,
        query: str,
        *,
        search_context_size: str | None = None,
        allowed_domains: list[str] | None = None,
        user_location: dict[str, Any] | None = None,
    ) -> WebSearchResult:
        log.debug("WebSearchRouter.search", provider_id=provider_id)
        start_time = time.perf_counter()
        resolved: str | None = None
        try:
            resolved = await self.routing_table.resolve_provider_id(provider_id)
            provider = await self.routing_table.get_provider_impl(resolved)
            result = await provider.search(
                resolved,
                query,
                search_context_size=search_context_size,
                allowed_domains=allowed_domains,
                user_location=user_location,
            )
            tool_invocations_total.add(1, {"tool_name": "web_search", "provider": resolved, "status": "success"})
            return result
        except Exception as e:
            # The caller (the responses tool executor) feeds this error back to the
            # model as a tool message but never logs it, so without this a failing
            # backend leaves no trace in the server log beyond a bare 'failed' status.
            log.error(
                "Failed to perform web search",
                provider=resolved or str(provider_id),
                query=query,
                error=str(e),
            )
            tool_invocations_total.add(
                1, {"tool_name": "web_search", "provider": resolved or str(provider_id), "status": "error"}
            )
            raise
        finally:
            tool_duration.record(
                time.perf_counter() - start_time, {"tool_name": "web_search", "provider": resolved or str(provider_id)}
            )
