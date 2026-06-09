# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import SecretStr

from ogx.providers.remote.tool_runtime.brave_search.brave_search import BraveSearchToolRuntimeImpl
from ogx.providers.remote.tool_runtime.brave_search.config import BraveSearchToolConfig


@pytest.fixture
def brave_search():
    return BraveSearchToolRuntimeImpl(BraveSearchToolConfig(api_key="config-key", max_results=3))


@pytest.fixture
def mock_brave_response():
    return httpx.Response(
        200,
        json={
            "mixed": {
                "main": [
                    {"type": "web", "index": 0},
                ]
            },
            "web": {
                "results": [
                    {
                        "type": "web",
                        "title": "Test Result",
                        "url": "https://example.com",
                        "description": "A test result",
                        "date": "2025-01-01",
                        "extra_snippets": ["snippet1"],
                    }
                ]
            },
        },
        request=httpx.Request("GET", "https://api.search.brave.com/res/v1/web/search"),
    )


async def test_invoke_with_allowed_domains(brave_search, mock_brave_response):
    with patch.object(brave_search, "get_request_provider_data", return_value=None):
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_brave_response) as mock_get:
            await brave_search.invoke_tool(
                "web_search",
                {
                    "query": "test query",
                    "allowed_domains": ["example.com", "docs.example.com"],
                },
            )
            call_kwargs = mock_get.call_args
            query_param = call_kwargs.kwargs["params"]["q"]
            assert "site:example.com" in query_param
            assert "site:docs.example.com" in query_param
            assert query_param == "test query (site:example.com OR site:docs.example.com)"


async def test_invoke_with_user_location_country(brave_search, mock_brave_response):
    with patch.object(brave_search, "get_request_provider_data", return_value=None):
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_brave_response) as mock_get:
            await brave_search.invoke_tool(
                "web_search",
                {
                    "query": "test query",
                    "user_location": {"country": "US", "city": "San Francisco"},
                },
            )
            call_kwargs = mock_get.call_args
            assert call_kwargs.kwargs["params"]["country"] == "US"


async def test_invoke_with_user_location_no_country(brave_search, mock_brave_response):
    with patch.object(brave_search, "get_request_provider_data", return_value=None):
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_brave_response) as mock_get:
            await brave_search.invoke_tool(
                "web_search",
                {
                    "query": "test query",
                    "user_location": {"city": "San Francisco"},
                },
            )
            call_kwargs = mock_get.call_args
            assert "country" not in call_kwargs.kwargs["params"]


async def test_invoke_with_search_context_size(brave_search, mock_brave_response):
    with patch.object(brave_search, "get_request_provider_data", return_value=None):
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_brave_response) as mock_get:
            await brave_search.invoke_tool(
                "web_search",
                {
                    "query": "test query",
                    "search_context_size": "high",
                },
            )
            call_kwargs = mock_get.call_args
            assert call_kwargs.kwargs["params"]["count"] == 10


async def test_invoke_with_search_context_size_updates_result_limit(brave_search):
    multi_result_response = httpx.Response(
        200,
        json={
            "mixed": {
                "main": [
                    {"type": "web", "index": 0},
                    {"type": "web", "index": 1},
                    {"type": "web", "index": 2},
                    {"type": "web", "index": 3},
                    {"type": "web", "index": 4},
                ]
            },
            "web": {
                "results": [
                    {
                        "type": "web",
                        "title": "Result 0",
                        "url": "https://example0.com",
                        "description": "A test result",
                        "date": "2025-01-01",
                        "extra_snippets": ["snippet0"],
                    },
                    {
                        "type": "web",
                        "title": "Result 1",
                        "url": "https://example1.com",
                        "description": "A test result",
                        "date": "2025-01-01",
                        "extra_snippets": ["snippet1"],
                    },
                    {
                        "type": "web",
                        "title": "Result 2",
                        "url": "https://example2.com",
                        "description": "A test result",
                        "date": "2025-01-01",
                        "extra_snippets": ["snippet2"],
                    },
                    {
                        "type": "web",
                        "title": "Result 3",
                        "url": "https://example3.com",
                        "description": "A test result",
                        "date": "2025-01-01",
                        "extra_snippets": ["snippet3"],
                    },
                    {
                        "type": "web",
                        "title": "Result 4",
                        "url": "https://example4.com",
                        "description": "A test result",
                        "date": "2025-01-01",
                        "extra_snippets": ["snippet4"],
                    },
                ]
            },
        },
        request=httpx.Request("GET", "https://api.search.brave.com/res/v1/web/search"),
    )
    with patch.object(brave_search, "get_request_provider_data", return_value=None):
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=multi_result_response) as mock_get:
            result = await brave_search.invoke_tool(
                "web_search",
                {
                    "query": "test query",
                    "search_context_size": "medium",
                },
            )
            call_kwargs = mock_get.call_args
            assert call_kwargs.kwargs["params"]["count"] == 5
            assert result.metadata is not None
            assert len(result.metadata["sources"]) == 5
            source_urls = {s["url"] for s in result.metadata["sources"]}
            expected_urls = {f"https://example{i}.com" for i in range(5)}
            assert source_urls == expected_urls


async def test_invoke_without_extra_params(brave_search, mock_brave_response):
    with patch.object(brave_search, "get_request_provider_data", return_value=None):
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_brave_response) as mock_get:
            await brave_search.invoke_tool(
                "web_search",
                {"query": "test query"},
            )
            call_kwargs = mock_get.call_args
            params = call_kwargs.kwargs["params"]
            assert params["q"] == "test query"
            assert "country" not in params
            assert "count" not in params


async def test_invoke_with_empty_allowed_domains(brave_search, mock_brave_response):
    with patch.object(brave_search, "get_request_provider_data", return_value=None):
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_brave_response) as mock_get:
            await brave_search.invoke_tool(
                "web_search",
                {
                    "query": "test query",
                    "allowed_domains": [],
                },
            )
            call_kwargs = mock_get.call_args
            assert call_kwargs.kwargs["params"]["q"] == "test query"


async def test_invoke_returns_source_metadata(brave_search, mock_brave_response):
    """Test that invoke_tool returns source URLs in metadata."""
    with patch.object(brave_search, "get_request_provider_data", return_value=None):
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_brave_response):
            result = await brave_search.invoke_tool(tool_name="web_search", kwargs={"query": "test query"})
            assert result.metadata is not None
            assert "sources" in result.metadata
            assert len(result.metadata["sources"]) == 1
            assert result.metadata["sources"][0]["url"] == "https://example.com"
            assert result.metadata["query"] == "test query"


class TestProviderDataApiKeyOverride:
    async def test_provider_data_api_key_overrides_config_api_key(self, brave_search, mock_brave_response):
        """Provider data API key should override the config API key."""
        with patch.object(
            brave_search,
            "get_request_provider_data",
            return_value=MagicMock(brave_search_api_key=SecretStr("provider-data-key")),
        ):
            with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_brave_response) as mock_get:
                await brave_search.invoke_tool(
                    "web_search",
                    {"query": "test query"},
                )
                headers = mock_get.call_args.kwargs["headers"]
                assert headers["X-Subscription-Token"] == "provider-data-key"

    async def test_config_api_key_used_when_no_provider_data(self, brave_search, mock_brave_response):
        """Config API key should be used when no provider data is provided."""
        with patch.object(brave_search, "get_request_provider_data", return_value=None):
            with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_brave_response) as mock_get:
                await brave_search.invoke_tool(
                    "web_search",
                    {"query": "test query"},
                )
                headers = mock_get.call_args.kwargs["headers"]
                assert headers["X-Subscription-Token"] == "config-key"

    async def test_config_api_key_used_when_provider_data_key_is_none(self, brave_search, mock_brave_response):
        """Config API key should be used when provider data key is None."""
        with patch.object(
            brave_search,
            "get_request_provider_data",
            return_value=MagicMock(brave_search_api_key=None),
        ):
            with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_brave_response) as mock_get:
                await brave_search.invoke_tool(
                    "web_search",
                    {"query": "test query"},
                )
                headers = mock_get.call_args.kwargs["headers"]
                assert headers["X-Subscription-Token"] == "config-key"

    async def test_returned_api_key_is_none_when_no_keys(self):
        """_get_api_key should return None when both config and provider data keys are absent."""
        impl = BraveSearchToolRuntimeImpl(BraveSearchToolConfig(max_results=3))
        with patch.object(impl, "get_request_provider_data", return_value=None):
            assert impl._get_api_key() is None

    async def test_api_key_header_omitted_when_both_keys_null(self, mock_brave_response):
        """Request should not include X-Subscription-Token header when no key is available."""
        impl = BraveSearchToolRuntimeImpl(BraveSearchToolConfig(max_results=3))
        with patch.object(impl, "get_request_provider_data", return_value=None):
            with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_brave_response) as mock_get:
                await impl.invoke_tool("web_search", {"query": "test query"})
                headers = mock_get.call_args.kwargs["headers"]
                assert "X-Subscription-Token" not in headers
