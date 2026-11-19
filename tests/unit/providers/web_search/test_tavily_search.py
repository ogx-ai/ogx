# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock, MagicMock, patch

import httpx2
import pytest
from pydantic import SecretStr

from ogx.providers.remote.web_search.tavily_search.config import TavilySearchConfig
from ogx.providers.remote.web_search.tavily_search.tavily_search import TavilySearchWebSearchImpl


@pytest.fixture
def tavily_search():
    return TavilySearchWebSearchImpl(TavilySearchConfig(api_key="test-key", max_results=3))


@pytest.fixture
def tavily_search_client(tavily_search):
    tavily_search._client = MagicMock(spec=httpx2.AsyncClient)
    return tavily_search


@pytest.fixture
def mock_tavily_response():
    return httpx2.Response(
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
        request=httpx2.Request("POST", "https://api.tavily.com/search"),
    )


async def test_search_with_allowed_domains(tavily_search_client, mock_tavily_response):
    with patch.object(tavily_search_client, "get_request_provider_data", return_value=None):
        tavily_search_client._client.post = AsyncMock(return_value=mock_tavily_response)
        await tavily_search_client.search(
            None,
            "test query",
            allowed_domains=["example.com", "docs.example.com"],
        )
    call_kwargs = tavily_search_client._client.post.call_args
    request_body = call_kwargs.kwargs["json"]
    assert request_body["include_domains"] == ["example.com", "docs.example.com"]


async def test_search_with_search_context_size(tavily_search_client, mock_tavily_response):
    with patch.object(tavily_search_client, "get_request_provider_data", return_value=None):
        tavily_search_client._client.post = AsyncMock(return_value=mock_tavily_response)
        await tavily_search_client.search(
            None,
            "test query",
            search_context_size="high",
        )
    call_kwargs = tavily_search_client._client.post.call_args
    request_body = call_kwargs.kwargs["json"]
    assert request_body["max_results"] == 10


async def test_search_without_extra_params(tavily_search_client, mock_tavily_response):
    with patch.object(tavily_search_client, "get_request_provider_data", return_value=None):
        tavily_search_client._client.post = AsyncMock(return_value=mock_tavily_response)
        await tavily_search_client.search(None, "test query")
    call_kwargs = tavily_search_client._client.post.call_args
    request_body = call_kwargs.kwargs["json"]
    assert request_body["query"] == "test query"
    assert request_body["api_key"] == "test-key"
    assert "include_domains" not in request_body
    assert "max_results" not in request_body


async def test_search_with_user_location_ignored(tavily_search_client, mock_tavily_response):
    with patch.object(tavily_search_client, "get_request_provider_data", return_value=None):
        tavily_search_client._client.post = AsyncMock(return_value=mock_tavily_response)
        await tavily_search_client.search(
            None,
            "test query",
            user_location={"country": "US", "city": "San Francisco"},
        )
    call_kwargs = tavily_search_client._client.post.call_args
    request_body = call_kwargs.kwargs["json"]
    assert "user_location" not in request_body
    assert "country" not in request_body
    assert "location" not in request_body


async def test_search_with_empty_allowed_domains(tavily_search_client, mock_tavily_response):
    with patch.object(tavily_search_client, "get_request_provider_data", return_value=None):
        tavily_search_client._client.post = AsyncMock(return_value=mock_tavily_response)
        await tavily_search_client.search(
            None,
            "test query",
            allowed_domains=[],
        )
    call_kwargs = tavily_search_client._client.post.call_args
    request_body = call_kwargs.kwargs["json"]
    assert "include_domains" not in request_body


async def test_search_returns_sources(tavily_search_client, mock_tavily_response):
    """Test that search returns source URLs and the searched query."""
    with patch.object(tavily_search_client, "get_request_provider_data", return_value=None):
        tavily_search_client._client.post = AsyncMock(return_value=mock_tavily_response)
        result = await tavily_search_client.search(None, "test query")
    assert len(result.sources) == 1
    assert result.sources[0].url == "https://example.com"
    assert result.query == "test query"


class TestProviderDataApiKeyOverride:
    async def test_provider_data_api_key_overrides_config_api_key(self, tavily_search_client, mock_tavily_response):
        """Provider data API key should override the config API key."""
        with patch.object(
            tavily_search_client,
            "get_request_provider_data",
            return_value=MagicMock(tavily_search_api_key=SecretStr("provider-data-key")),
        ):
            tavily_search_client._client.post = AsyncMock(return_value=mock_tavily_response)
            await tavily_search_client.search(None, "test query")
        request_body = tavily_search_client._client.post.call_args.kwargs["json"]
        assert request_body["api_key"] == "provider-data-key"

    async def test_config_api_key_used_when_no_provider_data(self, tavily_search_client, mock_tavily_response):
        """Config API key should be used when no provider data is provided."""
        with patch.object(tavily_search_client, "get_request_provider_data", return_value=None):
            tavily_search_client._client.post = AsyncMock(return_value=mock_tavily_response)
            await tavily_search_client.search(None, "test query")
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
            await tavily_search_client.search(None, "test query")
        request_body = tavily_search_client._client.post.call_args.kwargs["json"]
        assert request_body["api_key"] == "test-key"

    async def test_returned_api_key_is_none_when_no_keys(self):
        """_get_api_key should return None when both config and provider data keys are absent."""
        impl = TavilySearchWebSearchImpl(TavilySearchConfig(max_results=3))
        with patch.object(impl, "get_request_provider_data", return_value=None):
            assert impl._get_api_key() is None

    async def test_api_key_omitted_from_body_when_both_keys_null(self):
        """Request body should not include api_key field when no key is available."""
        mock_response = httpx2.Response(
            200,
            json={
                "query": "test query",
                "results": [],
            },
            request=httpx2.Request("POST", "https://api.tavily.com/search"),
        )
        impl = TavilySearchWebSearchImpl(TavilySearchConfig(max_results=3))
        impl._client = MagicMock(spec=httpx2.AsyncClient)
        impl._client.post = AsyncMock(return_value=mock_response)
        with patch.object(impl, "get_request_provider_data", return_value=None):
            await impl.search(None, "test query")
        request_body = impl._client.post.call_args.kwargs["json"]
        assert "api_key" not in request_body


async def test_401_raises_with_actionable_error(tavily_search_client):
    unauthorized = httpx2.Response(
        401,
        json={"detail": "Unauthorized"},
        request=httpx2.Request("POST", "https://api.tavily.com/search"),
    )
    tavily_search_client._client.post = AsyncMock(return_value=unauthorized)
    with patch.object(tavily_search_client, "get_request_provider_data", return_value=None):
        with pytest.raises(RuntimeError, match="Tavily Search: the API key is missing or invalid"):
            await tavily_search_client.search(None, "test query")


async def test_403_raises_with_actionable_error(tavily_search_client):
    forbidden = httpx2.Response(
        403,
        request=httpx2.Request("POST", "https://api.tavily.com/search"),
    )
    tavily_search_client._client.post = AsyncMock(return_value=forbidden)
    with patch.object(tavily_search_client, "get_request_provider_data", return_value=None):
        with pytest.raises(RuntimeError, match="entitlement"):
            await tavily_search_client.search(None, "test query")
