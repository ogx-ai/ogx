# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from ogx.providers.remote.tool_runtime.brave_search.brave_search import BraveSearchToolRuntimeImpl
from ogx.providers.remote.tool_runtime.brave_search.config import BraveSearchToolConfig


@pytest.fixture
def brave_search():
    return BraveSearchToolRuntimeImpl(BraveSearchToolConfig(api_key="test-key", max_results=3))


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


async def test_invoke_without_extra_params(brave_search, mock_brave_response):
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
