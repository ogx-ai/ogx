# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from ogx.providers.remote.tool_runtime.bing_search.bing_search import BingSearchToolRuntimeImpl
from ogx.providers.remote.tool_runtime.bing_search.config import BingSearchToolConfig


@pytest.fixture
def bing_search():
    impl = BingSearchToolRuntimeImpl(BingSearchToolConfig(api_key="test-key", top_k=3))
    impl._client = MagicMock(spec=httpx.AsyncClient)
    return impl


@pytest.fixture
def mock_bing_response():
    return httpx.Response(
        200,
        json={
            "queryContext": {"originalQuery": "test query"},
            "webPages": {
                "value": [
                    {
                        "name": "Test Result",
                        "url": "https://example.com",
                        "snippet": "A test result snippet",
                    }
                ]
            },
        },
        request=httpx.Request("GET", "https://api.bing.microsoft.com/v7.0/search"),
    )


async def test_invoke_with_allowed_domains(bing_search, mock_bing_response):
    bing_search._client.get = AsyncMock(return_value=mock_bing_response)
    await bing_search.invoke_tool(
        "web_search",
        {
            "query": "test query",
            "allowed_domains": ["example.com", "docs.example.com"],
        },
    )
    call_kwargs = bing_search._client.get.call_args
    query_param = call_kwargs.kwargs["params"]["q"]
    assert "site:example.com" in query_param
    assert "site:docs.example.com" in query_param
    assert query_param == "test query (site:example.com OR site:docs.example.com)"


async def test_invoke_with_user_location_country(bing_search, mock_bing_response):
    bing_search._client.get = AsyncMock(return_value=mock_bing_response)
    await bing_search.invoke_tool(
        "web_search",
        {
            "query": "test query",
            "user_location": {"country": "US", "city": "San Francisco"},
        },
    )
    call_kwargs = bing_search._client.get.call_args
    assert call_kwargs.kwargs["params"]["cc"] == "US"


async def test_invoke_with_search_context_size(bing_search, mock_bing_response):
    bing_search._client.get = AsyncMock(return_value=mock_bing_response)
    await bing_search.invoke_tool(
        "web_search",
        {
            "query": "test query",
            "search_context_size": "high",
        },
    )
    call_kwargs = bing_search._client.get.call_args
    assert call_kwargs.kwargs["params"]["count"] == 10


async def test_invoke_without_extra_params(bing_search, mock_bing_response):
    bing_search._client.get = AsyncMock(return_value=mock_bing_response)
    await bing_search.invoke_tool(
        "web_search",
        {"query": "test query"},
    )
    call_kwargs = bing_search._client.get.call_args
    params = call_kwargs.kwargs["params"]
    assert params["q"] == "test query"
    assert params["count"] == 3
    assert "cc" not in params


async def test_invoke_with_empty_allowed_domains(bing_search, mock_bing_response):
    bing_search._client.get = AsyncMock(return_value=mock_bing_response)
    await bing_search.invoke_tool(
        "web_search",
        {
            "query": "test query",
            "allowed_domains": [],
        },
    )
    call_kwargs = bing_search._client.get.call_args
    assert call_kwargs.kwargs["params"]["q"] == "test query"


async def test_invoke_returns_source_metadata(bing_search, mock_bing_response):
    """Test that invoke_tool returns source URLs in metadata."""
    bing_search._client.get = AsyncMock(return_value=mock_bing_response)
    result = await bing_search.invoke_tool(tool_name="web_search", kwargs={"query": "test query"})
    assert result.metadata is not None
    assert "sources" in result.metadata
    assert len(result.metadata["sources"]) == 1
    assert result.metadata["sources"][0]["url"] == "https://example.com"
    assert result.metadata["query"] == "test query"
