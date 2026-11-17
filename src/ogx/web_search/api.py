# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Protocol

from ogx_api.tools.models import ToolDef

from .models import WebSearchResult


class WebSearch(Protocol):
    """Web search API, routed by provider_id.

    Each configured search backend (brave, tavily, ...) is a first-class
    provider of this API. The routing key is the provider_id. When it is
    omitted, the API resolves a default: the first available provider in
    configuration order.

    Both the concrete providers and the auto-routed API impl satisfy this
    protocol (the Inference precedent): concrete providers ignore
    `provider_id`; the router uses it to dispatch.
    """

    async def initialize(self) -> None: ...

    async def shutdown(self) -> None: ...

    def available_providers(self) -> list[str]:
        """Provider ids this impl can serve (itself, or all backends for the router)."""
        ...

    async def is_available(self) -> bool:
        """Whether this provider can serve web search for the current request.

        Keyed backends are available when a key is present (config or
        request header); the placeholder is always available. Implementations
        may combine other readiness signals or perform a live check (e.g.
        validating the key or pinging the endpoint), but this runs on the
        default-resolution path for every web_search request, so keep it cheap.
        """
        ...

    async def tool_def(self, provider_id: str | None = None) -> ToolDef:
        """The model-facing tool definition, owned by the selected backend.

        `provider_id=None` resolves the default provider.
        """
        ...

    async def search(
        self,
        provider_id: str | None,
        query: str,
        *,
        search_context_size: str | None = None,
        allowed_domains: list[str] | None = None,
        user_location: dict[str, Any] | None = None,
    ) -> WebSearchResult:
        """Run a web search on the selected backend.

        `provider_id=None` resolves the default provider. Auth-class
        failures (401/403) raise a ``RuntimeError`` with an actionable
        message (see ``ogx.providers.utils.common.web_search.check_search_response``);
        other non-2xx responses raise via ``raise_for_status()``.
        """
        ...
