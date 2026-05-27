# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from ogx.providers.remote.tool_runtime.tavily_search.config import TavilySearchToolConfig
from ogx.providers.remote.tool_runtime.tavily_search.tavily_search import TavilySearchToolRuntimeImpl


@pytest.fixture
def tavily_search():
    impl = TavilySearchToolRuntimeImpl(TavilySearchToolConfig(api_key="test-key", max_results=3))
    impl._client = MagicMock(spec=httpx.AsyncClient)
    return impl


@pytest.fixture
def mock_tavily_response():
    return httpx.Response(
        200,
        json={
            "query": "test query",
            "results": [
                {
                    "title": "Test Result",
                    "url": "https://example.com",
                    "content": "A test result content",
                    "score": 0.95,
                }
            ],
        },
        request=httpx.Request("POST", "https://api.tavily.com/search"),
    )


async def test_invoke_with_allowed_domains(tavily_search, mock_tavily_response):
    tavily_search._client.post = AsyncMock(return_value=mock_tavily_response)
    await tavily_search.invoke_tool(
        "web_search",
        {
            "query": "test query",
            "allowed_domains": ["example.com", "docs.example.com"],
        },
    )
    call_kwargs = tavily_search._client.post.call_args
    request_body = call_kwargs.kwargs["json"]
    assert request_body["include_domains"] == ["example.com", "docs.example.com"]


async def test_invoke_with_search_context_size(tavily_search, mock_tavily_response):
    tavily_search._client.post = AsyncMock(return_value=mock_tavily_response)
    await tavily_search.invoke_tool(
        "web_search",
        {
            "query": "test query",
            "search_context_size": "high",
        },
    )
    call_kwargs = tavily_search._client.post.call_args
    request_body = call_kwargs.kwargs["json"]
    assert request_body["max_results"] == 10


async def test_invoke_without_extra_params(tavily_search, mock_tavily_response):
    tavily_search._client.post = AsyncMock(return_value=mock_tavily_response)
    await tavily_search.invoke_tool(
        "web_search",
        {"query": "test query"},
    )
    call_kwargs = tavily_search._client.post.call_args
    request_body = call_kwargs.kwargs["json"]
    assert request_body["query"] == "test query"
    assert request_body["api_key"] == "test-key"
    assert "include_domains" not in request_body
    assert "max_results" not in request_body


async def test_invoke_with_user_location_ignored(tavily_search, mock_tavily_response):
    tavily_search._client.post = AsyncMock(return_value=mock_tavily_response)
    await tavily_search.invoke_tool(
        "web_search",
        {
            "query": "test query",
            "user_location": {"country": "US", "city": "San Francisco"},
        },
    )
    call_kwargs = tavily_search._client.post.call_args
    request_body = call_kwargs.kwargs["json"]
    assert "user_location" not in request_body
    assert "country" not in request_body
    assert "location" not in request_body


async def test_invoke_with_empty_allowed_domains(tavily_search, mock_tavily_response):
    tavily_search._client.post = AsyncMock(return_value=mock_tavily_response)
    await tavily_search.invoke_tool(
        "web_search",
        {
            "query": "test query",
            "allowed_domains": [],
        },
    )
    call_kwargs = tavily_search._client.post.call_args
    request_body = call_kwargs.kwargs["json"]
    assert "include_domains" not in request_body


async def test_invoke_returns_source_metadata(tavily_search, mock_tavily_response):
    """Test that invoke_tool returns source URLs in metadata."""
    tavily_search._client.post = AsyncMock(return_value=mock_tavily_response)
    result = await tavily_search.invoke_tool(tool_name="web_search", kwargs={"query": "test query"})
    assert result.metadata is not None
    assert "sources" in result.metadata
    assert len(result.metadata["sources"]) == 1
    assert result.metadata["sources"][0]["url"] == "https://example.com"
    assert result.metadata["query"] == "test query"
