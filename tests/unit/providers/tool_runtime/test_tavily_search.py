# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import SecretStr

from ogx.providers.remote.tool_runtime.tavily_search.config import TavilySearchToolConfig
from ogx.providers.remote.tool_runtime.tavily_search.tavily_search import TavilySearchToolRuntimeImpl


@pytest.fixture
def tavily_search():
    return TavilySearchToolRuntimeImpl(TavilySearchToolConfig(api_key="test-key", max_results=3))


@pytest.fixture
def tavily_search_client(tavily_search):
    tavily_search._client = MagicMock(spec=httpx.AsyncClient)
    return tavily_search


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


async def test_invoke_with_allowed_domains(tavily_search_client, mock_tavily_response):
    with patch.object(tavily_search_client, "get_request_provider_data", return_value=None):
        tavily_search_client._client.post = AsyncMock(return_value=mock_tavily_response)
        await tavily_search_client.invoke_tool(
            "web_search",
            {
                "query": "test query",
                "allowed_domains": ["example.com", "docs.example.com"],
            },
        )
    call_kwargs = tavily_search_client._client.post.call_args
    request_body = call_kwargs.kwargs["json"]
    assert request_body["include_domains"] == ["example.com", "docs.example.com"]


async def test_invoke_with_search_context_size(tavily_search_client, mock_tavily_response):
    with patch.object(tavily_search_client, "get_request_provider_data", return_value=None):
        tavily_search_client._client.post = AsyncMock(return_value=mock_tavily_response)
        await tavily_search_client.invoke_tool(
            "web_search",
            {
                "query": "test query",
                "search_context_size": "high",
            },
        )
    call_kwargs = tavily_search_client._client.post.call_args
    request_body = call_kwargs.kwargs["json"]
    assert request_body["max_results"] == 10


async def test_invoke_without_extra_params(tavily_search_client, mock_tavily_response):
    with patch.object(tavily_search_client, "get_request_provider_data", return_value=None):
        tavily_search_client._client.post = AsyncMock(return_value=mock_tavily_response)
        await tavily_search_client.invoke_tool(
            "web_search",
            {"query": "test query"},
        )
    call_kwargs = tavily_search_client._client.post.call_args
    request_body = call_kwargs.kwargs["json"]
    assert request_body["query"] == "test query"
    assert request_body["api_key"] == "test-key"
    assert "include_domains" not in request_body
    assert "max_results" not in request_body


async def test_invoke_with_user_location_ignored(tavily_search_client, mock_tavily_response):
    with patch.object(tavily_search_client, "get_request_provider_data", return_value=None):
        tavily_search_client._client.post = AsyncMock(return_value=mock_tavily_response)
        await tavily_search_client.invoke_tool(
            "web_search",
            {
                "query": "test query",
                "user_location": {"country": "US", "city": "San Francisco"},
            },
        )
    call_kwargs = tavily_search_client._client.post.call_args
    request_body = call_kwargs.kwargs["json"]
    assert "user_location" not in request_body
    assert "country" not in request_body
    assert "location" not in request_body


async def test_invoke_with_empty_allowed_domains(tavily_search_client, mock_tavily_response):
    with patch.object(tavily_search_client, "get_request_provider_data", return_value=None):
        tavily_search_client._client.post = AsyncMock(return_value=mock_tavily_response)
        await tavily_search_client.invoke_tool(
            "web_search",
            {
                "query": "test query",
                "allowed_domains": [],
            },
        )
    call_kwargs = tavily_search_client._client.post.call_args
    request_body = call_kwargs.kwargs["json"]
    assert "include_domains" not in request_body


async def test_invoke_returns_source_metadata(tavily_search_client, mock_tavily_response):
    """Test that invoke_tool returns source URLs in metadata."""
    with patch.object(tavily_search_client, "get_request_provider_data", return_value=None):
        tavily_search_client._client.post = AsyncMock(return_value=mock_tavily_response)
        result = await tavily_search_client.invoke_tool(
            tool_name="web_search",
            kwargs={"query": "test query"},
        )
    assert result.metadata is not None
    assert "sources" in result.metadata
    assert len(result.metadata["sources"]) == 1
    assert result.metadata["sources"][0]["url"] == "https://example.com"
    assert result.metadata["query"] == "test query"


class TestProviderDataApiKeyOverride:
    async def test_provider_data_api_key_overrides_config_api_key(self, tavily_search_client, mock_tavily_response):
        """Provider data API key should override the config API key."""
        with patch.object(
            tavily_search_client,
            "get_request_provider_data",
            return_value=MagicMock(tavily_search_api_key=SecretStr("provider-data-key")),
        ):
            tavily_search_client._client.post = AsyncMock(return_value=mock_tavily_response)
            await tavily_search_client.invoke_tool(
                "web_search",
                {"query": "test query"},
            )
        request_body = tavily_search_client._client.post.call_args.kwargs["json"]
        assert request_body["api_key"] == "provider-data-key"

    async def test_config_api_key_used_when_no_provider_data(self, tavily_search_client, mock_tavily_response):
        """Config API key should be used when no provider data is provided."""
        with patch.object(tavily_search_client, "get_request_provider_data", return_value=None):
            tavily_search_client._client.post = AsyncMock(return_value=mock_tavily_response)
            await tavily_search_client.invoke_tool(
                "web_search",
                {"query": "test query"},
            )
        request_body = tavily_search_client._client.post.call_args.kwargs["json"]
        assert request_body["api_key"] == "test-key"

    async def test_config_api_key_used_when_provider_data_key_is_none(self, tavily_search_client, mock_tavily_response):
        """Config API key should be used when provider data key is None."""
        with patch.object(
            tavily_search_client,
            "get_request_provider_data",
            return_value=MagicMock(tavily_search_api_key=None),
        ):
            tavily_search_client._client.post = AsyncMock(return_value=mock_tavily_response)
            await tavily_search_client.invoke_tool(
                "web_search",
                {"query": "test query"},
            )
        request_body = tavily_search_client._client.post.call_args.kwargs["json"]
        assert request_body["api_key"] == "test-key"

    async def test_returned_api_key_is_none_when_no_keys(self):
        """_get_api_key should return None when both config and provider data keys are absent."""
        impl = TavilySearchToolRuntimeImpl(TavilySearchToolConfig(max_results=3))
        with patch.object(impl, "get_request_provider_data", return_value=None):
            assert impl._get_api_key() is None

    async def test_api_key_omitted_from_body_when_both_keys_null(self):
        """Request body should not include api_key field when no key is available."""
        mock_response = httpx.Response(
            200,
            json={
                "query": "test query",
                "results": [],
            },
            request=httpx.Request("POST", "https://api.tavily.com/search"),
        )
        impl = TavilySearchToolRuntimeImpl(TavilySearchToolConfig(max_results=3))
        impl._client = MagicMock(spec=httpx.AsyncClient)
        impl._client.post = AsyncMock(return_value=mock_response)
        with patch.object(impl, "get_request_provider_data", return_value=None):
            await impl.invoke_tool("web_search", {"query": "test query"})
        request_body = impl._client.post.call_args.kwargs["json"]
        assert "api_key" not in request_body
