# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import logging  # allow-direct-logging

import pytest

from ogx.core.routing_tables.web_search import WebSearchRoutingTable
from ogx.providers.inline.web_search.placeholder import PlaceholderWebSearchConfig
from ogx.providers.inline.web_search.placeholder.placeholder import PlaceholderWebSearchImpl
from ogx.web_search import WebSearchResult
from ogx_api import ToolDef


class FakeWebSearchProvider:
    def __init__(self, provider_id: str, available: bool = False) -> None:
        self.provider_id = provider_id
        self.available = available
        self.search_calls: list[tuple[str, str]] = []

    def available_providers(self) -> list[str]:
        return [self.provider_id]

    async def is_available(self) -> bool:
        return self.available

    async def tool_def(self, provider_id: str | None = None) -> ToolDef:
        return ToolDef(name="web_search", description=f"tool def from {self.provider_id}", input_schema={})

    async def search(
        self,
        provider_id: str | None,
        query: str,
        **kwargs,
    ) -> WebSearchResult:
        self.search_calls.append((provider_id, query))
        return WebSearchResult(content=f"{self.provider_id}: {query}", query=query)


def make_table(impls: dict) -> WebSearchRoutingTable:
    return WebSearchRoutingTable(impls_by_provider_id=impls, dist_registry=None, policy=[])


async def test_search_routes_by_explicit_provider_id():
    brave = FakeWebSearchProvider("brave", available=True)
    tavily = FakeWebSearchProvider("tavily")
    table = make_table({"brave": brave, "tavily": tavily})

    result = await table.search("tavily", "query", search_context_size="high")

    assert result.content == "tavily: query"
    assert tavily.search_calls == [("tavily", "query")]
    assert brave.search_calls == []


async def test_tool_def_owned_by_selected_provider():
    brave = FakeWebSearchProvider("brave")
    tavily = FakeWebSearchProvider("tavily")
    table = make_table({"brave": brave, "tavily": tavily})

    assert (await table.tool_def("brave")).description == "tool def from brave"
    assert (await table.tool_def("tavily")).description == "tool def from tavily"


async def test_unknown_provider_raises():
    table = make_table({"brave": FakeWebSearchProvider("brave")})

    with pytest.raises(ValueError, match="Failed to find web_search provider"):
        await table.search("bing", "query")
    with pytest.raises(ValueError, match="Failed to find web_search provider"):
        await table.tool_def("bing")


async def test_default_resolution_prefers_keyed_provider():
    brave = FakeWebSearchProvider("brave")
    tavily = FakeWebSearchProvider("tavily", available=True)
    table = make_table({"brave": brave, "tavily": tavily})

    assert await table.resolve_provider_id(None) == "tavily"
    result = await table.search(None, "query")
    assert result.content == "tavily: query"
    assert brave.search_calls == []


async def test_default_resolution_falls_back_to_placeholder():
    # The placeholder is modeled as always-available, like the real backend.
    brave = FakeWebSearchProvider("brave")
    placeholder = FakeWebSearchProvider("placeholder", available=True)
    table = make_table({"brave": brave, "placeholder": placeholder})

    assert await table.resolve_provider_id(None) == "placeholder"


async def test_default_resolution_falls_back_to_first_configured():
    brave = FakeWebSearchProvider("brave")
    tavily = FakeWebSearchProvider("tavily")
    table = make_table({"brave": brave, "tavily": tavily})

    assert await table.resolve_provider_id(None) == "brave"


async def test_default_resolution_follows_configuration_order():
    """Availability, not backend kind, decides: the first available provider in config order wins."""
    placeholder = FakeWebSearchProvider("placeholder", available=True)
    brave = FakeWebSearchProvider("brave", available=True)

    assert await make_table({"placeholder": placeholder, "brave": brave}).resolve_provider_id(None) == "placeholder"
    assert await make_table({"brave": brave, "placeholder": placeholder}).resolve_provider_id(None) == "brave"


async def test_default_resolution_with_no_providers_raises():
    table = make_table({})

    with pytest.raises(ValueError, match="no web_search providers are configured"):
        await table.resolve_provider_id(None)


async def test_placeholder_backend_degrades_gracefully():
    placeholder = PlaceholderWebSearchImpl(PlaceholderWebSearchConfig())
    placeholder.__provider_id__ = "placeholder"
    table = make_table({"placeholder": placeholder})

    # The placeholder never has an API key but is always available.
    assert await placeholder.is_available() is True
    tool_def = await table.tool_def(None)
    assert "not configured" in tool_def.description
    result = await table.search(None, "query")
    assert result.content == PlaceholderWebSearchImpl.NOT_AVAILABLE_MESSAGE


async def test_real_backends_route_and_default_resolves_to_placeholder():
    """End-to-end through the real router with the real backend impls (no keys, no network)."""
    from ogx.core.routers.web_search import WebSearchRouter
    from ogx.providers.registry.web_search import available_providers as ws_providers
    from ogx.providers.remote.web_search.brave_search import BraveSearchConfig
    from ogx.providers.remote.web_search.brave_search.brave_search import BraveSearchWebSearchImpl
    from ogx.providers.remote.web_search.tavily_search import TavilySearchConfig
    from ogx.providers.remote.web_search.tavily_search.tavily_search import TavilySearchWebSearchImpl

    specs = {spec.provider_type: spec for spec in ws_providers()}

    brave = BraveSearchWebSearchImpl(BraveSearchConfig())
    brave.__provider_id__ = "brave"
    brave.__provider_spec__ = specs["remote::brave-search"]
    tavily = TavilySearchWebSearchImpl(TavilySearchConfig())
    tavily.__provider_id__ = "tavily"
    tavily.__provider_spec__ = specs["remote::tavily-search"]
    placeholder = PlaceholderWebSearchImpl(PlaceholderWebSearchConfig())
    placeholder.__provider_id__ = "placeholder"

    table = make_table({"brave": brave, "tavily": tavily, "placeholder": placeholder})
    router = WebSearchRouter(table)

    assert router.available_providers() == ["brave", "tavily", "placeholder"]
    # No keyed backend, but the placeholder keeps the router available.
    assert await router.is_available() is True

    # No backend has a key: default resolution must land on the placeholder
    tool_def = await router.tool_def(None)
    assert "not configured" in tool_def.description
    result = await router.search(None, "query")
    assert result.content == PlaceholderWebSearchImpl.NOT_AVAILABLE_MESSAGE

    # An explicit backend selection still reaches that backend's tool def
    brave_def = await router.tool_def("brave")
    assert brave_def.name == "web_search"
    assert brave_def.description == "Search the web for information"


async def test_router_logs_backend_failure(caplog):
    """A failing backend's error is logged, not just fed back to the model.

    The responses tool executor forwards a search error to the model as a tool
    message and marks the web_search_call 'failed' without logging anything, so
    a broken backend (bad key, no egress) would leave no trace in the server log.
    """
    from ogx.core.routers.web_search import WebSearchRouter

    class FailingProvider(FakeWebSearchProvider):
        async def search(self, provider_id, query, **kwargs):
            raise RuntimeError("Failed to query Fake Search: the API key is missing or invalid.")

    failing = FailingProvider("fake", available=True)
    router = WebSearchRouter(make_table({"fake": failing}))

    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError, match="Failed to query Fake Search"):
            await router.search(None, "query")

    assert "Failed to perform web search" in caplog.text
    assert "fake" in caplog.text
    assert "the API key is missing or invalid" in caplog.text


async def test_web_search_missing_per_request_key_is_quiet(caplog):
    """A web search backend without a per-request key falls back to its config key
    without logging an error: requests without provider data are the normal case.
    The non-quiet path (other provider families) still logs."""
    from ogx.core.request_headers import RequestProviderDataContext
    from ogx.providers.registry.web_search import available_providers as ws_providers
    from ogx.providers.remote.web_search.tavily_search import TavilySearchConfig
    from ogx.providers.remote.web_search.tavily_search.tavily_search import TavilySearchWebSearchImpl

    specs = {spec.provider_type: spec for spec in ws_providers()}
    tavily = TavilySearchWebSearchImpl(TavilySearchConfig())
    tavily.__provider_id__ = "tavily"
    tavily.__provider_spec__ = specs["remote::tavily-search"]

    with RequestProviderDataContext({"__test_id": "some-test::tavily-search"}):
        with caplog.at_level(logging.ERROR):
            # No config key either: the quiet lookup returns None and logs nothing
            assert tavily._get_api_key() is None
            assert "Error parsing provider data" not in caplog.text

            # The same lookup without quiet still surfaces the validation failure
            assert tavily.get_request_provider_data() is None
            assert "Error parsing provider data" in caplog.text


def test_legacy_tool_runtime_web_search_providers_translate_to_web_search():
    """Web search providers listed under tool_runtime in an old config move to web_search at read time."""
    from ogx.core.datatypes import Provider, StackConfig

    with pytest.warns(DeprecationWarning, match="tool_runtime"):
        config = StackConfig(
            distro_name="test",
            apis=["tool_runtime", "responses"],
            providers={
                "tool_runtime": [
                    Provider(provider_id="brave", provider_type="remote::brave-search"),
                    Provider(provider_id="mcp", provider_type="remote::model-context-protocol"),
                    Provider(provider_id="searchapi", provider_type="remote::searchapi-search"),
                    Provider(provider_id="placeholder", provider_type="inline::placeholder"),
                ],
            },
        )

    assert [(p.provider_id, p.provider_type) for p in config.providers["web_search"]] == [
        ("brave", "remote::brave-search"),
        ("searchapi", "remote::searchapi-search"),
        ("placeholder", "inline::placeholder"),
    ]
    # Non-web-search tool_runtime providers stay put
    assert [(p.provider_id, p.provider_type) for p in config.providers["tool_runtime"]] == [
        ("mcp", "remote::model-context-protocol"),
    ]
    # The translated providers must be reachable over the HTTP surface
    assert "web_search" in config.apis


def test_legacy_config_without_apis_list_is_untouched_by_translation():
    """An apis: null (serve everything) config keeps its open API list; only the providers move."""
    from ogx.core.datatypes import Provider, StackConfig

    with pytest.warns(DeprecationWarning):
        config = StackConfig(
            distro_name="test",
            providers={
                "tool_runtime": [Provider(provider_id="tavily", provider_type="remote::tavily-search")],
            },
        )

    assert config.apis is None
    assert [p.provider_type for p in config.providers["web_search"]] == ["remote::tavily-search"]


def test_removed_file_search_provider_is_dropped_from_tool_runtime():
    """A config still listing the removed file-search provider loads, with the entry dropped."""
    from ogx.core.datatypes import Provider, StackConfig

    with pytest.warns(DeprecationWarning, match="inline::file-search"):
        config = StackConfig(
            distro_name="test",
            providers={
                "tool_runtime": [
                    Provider(provider_id="file-search", provider_type="inline::file-search"),
                    Provider(provider_id="mcp", provider_type="remote::model-context-protocol"),
                ],
            },
        )

    # The removed provider is dropped; other tool_runtime providers stay put
    assert [p.provider_type for p in config.providers["tool_runtime"]] == ["remote::model-context-protocol"]
    # Nothing is moved to another API section
    assert "web_search" not in config.providers


def test_config_without_file_search_provider_is_untouched():
    from ogx.core.datatypes import Provider, StackConfig

    config = StackConfig(
        distro_name="test",
        providers={
            "tool_runtime": [Provider(provider_id="mcp", provider_type="remote::model-context-protocol")],
        },
    )

    assert [p.provider_type for p in config.providers["tool_runtime"]] == ["remote::model-context-protocol"]


def test_generated_starter_config_keeps_legacy_shape_and_translates_on_load():
    """The bundled distro config keeps the legacy tool_runtime spelling, exercising the translation.

    The distro templates deliberately do not spell a web_search section: loading
    the generated starter config must surface the DeprecationWarning and produce
    a translated web_search section with web_search added to the served APIs.
    """
    from ogx.core.stack import get_stack_run_config_from_distro

    with pytest.warns(DeprecationWarning, match="tool_runtime"):
        config = get_stack_run_config_from_distro("starter")

    # Config order is the resolution priority order
    assert [p.provider_type for p in config.providers["web_search"]] == [
        "remote::exa-search",
        "remote::brave-search",
        "remote::tavily-search",
        "remote::nimble-search",
        "remote::serply-search",
        "remote::searchapi-search",
    ]
    # Non-web-search tool_runtime providers stay put
    assert [p.provider_type for p in config.providers["tool_runtime"]] == ["remote::model-context-protocol"]
    assert "web_search" in config.apis


async def test_legacy_tool_runtime_config_resolves_web_search_router():
    """A config that still lists web search under tool_runtime resolves to the web_search router.

    The translated provider is what the router dispatches to: with only the legacy
    placeholder configured, a default search lands on the placeholder backend.
    """
    from unittest.mock import MagicMock

    from ogx.core.datatypes import Api, Provider, StackConfig
    from ogx.core.resolver import resolve_impls
    from ogx.core.routers.web_search import WebSearchRouter
    from ogx.providers.registry.web_search import available_providers as ws_providers

    provider_registry = {Api.web_search: {spec.provider_type: spec for spec in ws_providers()}}
    dist_registry = MagicMock()

    with pytest.warns(DeprecationWarning):
        run_config = StackConfig(
            distro_name="test",
            apis=["web_search"],
            providers={
                "tool_runtime": [Provider(provider_id="placeholder", provider_type="inline::placeholder")],
            },
        )

    impls = await resolve_impls(run_config, provider_registry, dist_registry, policy={})

    router = impls[Api.web_search]
    assert isinstance(router, WebSearchRouter)
    assert "placeholder" in router.routing_table.impls_by_provider_id

    result = await router.search(None, "query")
    assert result.content == PlaceholderWebSearchImpl.NOT_AVAILABLE_MESSAGE
