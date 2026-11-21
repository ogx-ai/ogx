# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from ogx_api import (
    Api,
    InlineProviderSpec,
    ProviderSpec,
    RemoteProviderSpec,
)


def available_providers() -> list[ProviderSpec]:
    """Return the list of available web search provider specifications.

    Each backend is a first-class provider of the web_search API; the
    framework routes to one per request by provider_id.

    Returns:
        List of ProviderSpec objects describing available providers
    """
    return [
        RemoteProviderSpec(
            api=Api.web_search,
            adapter_type="brave-search",
            provider_type="remote::brave-search",
            module="ogx.providers.remote.web_search.brave_search",
            config_class="ogx.providers.remote.web_search.brave_search.config.BraveSearchConfig",
            pip_packages=[],
            provider_data_validator="ogx.providers.remote.web_search.brave_search.BraveSearchProviderDataValidator",
            description="Brave Search backend for the web_search API.",
        ),
        RemoteProviderSpec(
            api=Api.web_search,
            adapter_type="tavily-search",
            provider_type="remote::tavily-search",
            module="ogx.providers.remote.web_search.tavily_search",
            config_class="ogx.providers.remote.web_search.tavily_search.config.TavilySearchConfig",
            pip_packages=[],
            provider_data_validator="ogx.providers.remote.web_search.tavily_search.TavilySearchProviderDataValidator",
            description="Tavily Search backend for the web_search API.",
        ),
        RemoteProviderSpec(
            api=Api.web_search,
            adapter_type="bing-search",
            provider_type="remote::bing-search",
            module="ogx.providers.remote.web_search.bing_search",
            config_class="ogx.providers.remote.web_search.bing_search.config.BingSearchConfig",
            pip_packages=[],
            provider_data_validator="ogx.providers.remote.web_search.bing_search.BingSearchProviderDataValidator",
            description="Bing Search backend for the web_search API.",
        ),
        RemoteProviderSpec(
            api=Api.web_search,
            adapter_type="exa-search",
            provider_type="remote::exa-search",
            module="ogx.providers.remote.web_search.exa_search",
            config_class="ogx.providers.remote.web_search.exa_search.config.ExaSearchConfig",
            pip_packages=[],
            provider_data_validator="ogx.providers.remote.web_search.exa_search.ExaSearchProviderDataValidator",
            description="Exa Search backend for the web_search API.",
        ),
        RemoteProviderSpec(
            api=Api.web_search,
            adapter_type="nimble-search",
            provider_type="remote::nimble-search",
            module="ogx.providers.remote.web_search.nimble_search",
            config_class="ogx.providers.remote.web_search.nimble_search.config.NimbleSearchConfig",
            pip_packages=[],
            provider_data_validator="ogx.providers.remote.web_search.nimble_search.NimbleSearchProviderDataValidator",
            description="Nimble Search backend for the web_search API.",
        ),
        RemoteProviderSpec(
            api=Api.web_search,
            adapter_type="serply-search",
            provider_type="remote::serply-search",
            module="ogx.providers.remote.web_search.serply_search",
            config_class="ogx.providers.remote.web_search.serply_search.config.SerplySearchConfig",
            pip_packages=[],
            provider_data_validator="ogx.providers.remote.web_search.serply_search.SerplySearchProviderDataValidator",
            description="Serply Search backend for the web_search API.",
        ),
        RemoteProviderSpec(
            api=Api.web_search,
            adapter_type="searchapi-search",
            provider_type="remote::searchapi-search",
            module="ogx.providers.remote.web_search.searchapi_search",
            config_class="ogx.providers.remote.web_search.searchapi_search.config.SearchApiSearchConfig",
            pip_packages=[],
            provider_data_validator="ogx.providers.remote.web_search.searchapi_search.SearchApiSearchProviderDataValidator",
            description="SearchApi Search backend for the web_search API.",
        ),
        InlineProviderSpec(
            api=Api.web_search,
            provider_type="inline::placeholder",
            module="ogx.providers.inline.web_search.placeholder",
            config_class="ogx.providers.inline.web_search.placeholder.config.PlaceholderWebSearchConfig",
            description="Placeholder web search backend. Performs no search; invocations return a notice that web search is not configured, so clients that send the web_search tool unconditionally degrade gracefully.",
        ),
    ]
