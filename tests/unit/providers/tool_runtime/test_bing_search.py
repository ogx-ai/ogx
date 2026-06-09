# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import SecretStr

from ogx.providers.remote.tool_runtime.bing_search.bing_search import BingSearchToolRuntimeImpl
from ogx.providers.remote.tool_runtime.bing_search.config import BingSearchToolConfig


@pytest.fixture
def bing_search():
    return BingSearchToolRuntimeImpl(BingSearchToolConfig(api_key="config-key", top_k=3))


@pytest.fixture
def bing_search_client(bing_search):
    bing_search._client = MagicMock(spec=httpx.AsyncClient)
    return bing_search


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


async def test_invoke_with_allowed_domains(bing_search_client, mock_bing_response):
    with patch.object(bing_search_client, "get_request_provider_data", return_value=None):
        bing_search_client._client.get = AsyncMock(return_value=mock_bing_response)
        await bing_search_client.invoke_tool(
            "web_search",
            {
                "query": "test query",
                "allowed_domains": ["example.com", "docs.example.com"],
            },
        )
    call_kwargs = bing_search_client._client.get.call_args
    query_param = call_kwargs.kwargs["params"]["q"]
    assert "site:example.com" in query_param
    assert "site:docs.example.com" in query_param
    assert query_param == "test query (site:example.com OR site:docs.example.com)"


async def test_invoke_with_user_location_country(bing_search_client, mock_bing_response):
    with patch.object(bing_search_client, "get_request_provider_data", return_value=None):
        bing_search_client._client.get = AsyncMock(return_value=mock_bing_response)
        await bing_search_client.invoke_tool(
            "web_search",
            {
                "query": "test query",
                "user_location": {"country": "US", "city": "San Francisco"},
            },
        )
    call_kwargs = bing_search_client._client.get.call_args
    assert call_kwargs.kwargs["params"]["cc"] == "US"


async def test_invoke_with_search_context_size(bing_search_client, mock_bing_response):
    with patch.object(bing_search_client, "get_request_provider_data", return_value=None):
        bing_search_client._client.get = AsyncMock(return_value=mock_bing_response)
        await bing_search_client.invoke_tool(
            "web_search",
            {
                "query": "test query",
                "search_context_size": "high",
            },
        )
    call_kwargs = bing_search_client._client.get.call_args
    assert call_kwargs.kwargs["params"]["count"] == 10


async def test_invoke_without_extra_params(bing_search_client, mock_bing_response):
    with patch.object(bing_search_client, "get_request_provider_data", return_value=None):
        bing_search_client._client.get = AsyncMock(return_value=mock_bing_response)
        await bing_search_client.invoke_tool(
            "web_search",
            {"query": "test query"},
        )
    call_kwargs = bing_search_client._client.get.call_args
    params = call_kwargs.kwargs["params"]
    assert params["q"] == "test query"
    assert params["count"] == 3
    assert "cc" not in params


async def test_invoke_with_empty_allowed_domains(bing_search_client, mock_bing_response):
    with patch.object(bing_search_client, "get_request_provider_data", return_value=None):
        bing_search_client._client.get = AsyncMock(return_value=mock_bing_response)
        await bing_search_client.invoke_tool(
            "web_search",
            {
                "query": "test query",
                "allowed_domains": [],
            },
        )
    call_kwargs = bing_search_client._client.get.call_args
    assert call_kwargs.kwargs["params"]["q"] == "test query"


async def test_invoke_returns_source_metadata(bing_search_client, mock_bing_response):
    """Test that invoke_tool returns source URLs in metadata."""
    with patch.object(bing_search_client, "get_request_provider_data", return_value=None):
        bing_search_client._client.get = AsyncMock(return_value=mock_bing_response)
        result = await bing_search_client.invoke_tool(
            tool_name="web_search",
            kwargs={"query": "test query"},
        )
    assert result.metadata is not None
    assert "sources" in result.metadata
    assert len(result.metadata["sources"]) == 1
    assert result.metadata["sources"][0]["url"] == "https://example.com"
    assert result.metadata["query"] == "test query"


class TestProviderDataApiKeyOverride:
    async def test_provider_data_api_key_overrides_config_api_key(self, bing_search_client, mock_bing_response):
        """Provider data API key should override the config API key."""
        with patch.object(
            bing_search_client,
            "get_request_provider_data",
            return_value=MagicMock(bing_search_api_key=SecretStr("provider-data-key")),
        ):
            bing_search_client._client.get = AsyncMock(return_value=mock_bing_response)
            await bing_search_client.invoke_tool(
                "web_search",
                {"query": "test query"},
            )
        headers = bing_search_client._client.get.call_args.kwargs["headers"]
        assert headers["Ocp-Apim-Subscription-Key"] == "provider-data-key"

    async def test_config_api_key_used_when_no_provider_data(self, bing_search_client, mock_bing_response):
        """Config API key should be used when no provider data is provided."""
        with patch.object(bing_search_client, "get_request_provider_data", return_value=None):
            bing_search_client._client.get = AsyncMock(return_value=mock_bing_response)
            await bing_search_client.invoke_tool(
                "web_search",
                {"query": "test query"},
            )
        headers = bing_search_client._client.get.call_args.kwargs["headers"]
        assert headers["Ocp-Apim-Subscription-Key"] == "config-key"

    async def test_config_api_key_used_when_provider_data_key_is_none(self, bing_search_client, mock_bing_response):
        """Config API key should be used when provider data key is None."""
        with patch.object(
            bing_search_client,
            "get_request_provider_data",
            return_value=MagicMock(bing_search_api_key=None),
        ):
            bing_search_client._client.get = AsyncMock(return_value=mock_bing_response)
            await bing_search_client.invoke_tool(
                "web_search",
                {"query": "test query"},
            )
        headers = bing_search_client._client.get.call_args.kwargs["headers"]
        assert headers["Ocp-Apim-Subscription-Key"] == "config-key"

    def test_returned_api_key_is_none_when_both_keys_null(self):
        """_get_api_key should return None when both config and provider data keys are None."""
        impl = BingSearchToolRuntimeImpl(BingSearchToolConfig(top_k=3))
        with patch.object(impl, "get_request_provider_data", return_value=None):
            assert impl._get_api_key() is None

    async def test_api_key_header_omitted_when_no_key_available(self, mock_bing_response):
        """Request should not include Ocp-Apim-Subscription-Key header when no key is available."""
        impl = BingSearchToolRuntimeImpl(BingSearchToolConfig(top_k=3))
        impl._client = MagicMock(spec=httpx.AsyncClient)
        impl._client.get = AsyncMock(return_value=mock_bing_response)
        with patch.object(impl, "get_request_provider_data", return_value=None):
            await impl.invoke_tool("web_search", {"query": "test query"})
        headers = impl._client.get.call_args.kwargs["headers"]
        assert "Ocp-Apim-Subscription-Key" not in headers
