# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock, MagicMock, patch

import httpx2
import pytest
from pydantic import SecretStr

from ogx.providers.remote.web_search.bing_search.bing_search import BingSearchWebSearchImpl
from ogx.providers.remote.web_search.bing_search.config import BingSearchConfig


@pytest.fixture
def bing_search():
    return BingSearchWebSearchImpl(BingSearchConfig(api_key="config-key", top_k=3))


@pytest.fixture
def bing_search_client(bing_search):
    bing_search._client = MagicMock(spec=httpx2.AsyncClient)
    return bing_search


@pytest.fixture
def mock_bing_response():
    return httpx2.Response(
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
        request=httpx2.Request("GET", "https://api.bing.microsoft.com/v7.0/search"),
    )


async def test_search_with_allowed_domains(bing_search_client, mock_bing_response):
    with patch.object(bing_search_client, "get_request_provider_data", return_value=None):
        bing_search_client._client.get = AsyncMock(return_value=mock_bing_response)
        await bing_search_client.search(
            None,
            "test query",
            allowed_domains=["example.com", "docs.example.com"],
        )
    call_kwargs = bing_search_client._client.get.call_args
    query_param = call_kwargs.kwargs["params"]["q"]
    assert "site:example.com" in query_param
    assert "site:docs.example.com" in query_param
    assert query_param == "test query (site:example.com OR site:docs.example.com)"


async def test_search_with_user_location_country(bing_search_client, mock_bing_response):
    with patch.object(bing_search_client, "get_request_provider_data", return_value=None):
        bing_search_client._client.get = AsyncMock(return_value=mock_bing_response)
        await bing_search_client.search(
            None,
            "test query",
            user_location={"country": "US", "city": "San Francisco"},
        )
    call_kwargs = bing_search_client._client.get.call_args
    assert call_kwargs.kwargs["params"]["cc"] == "US"


async def test_search_with_search_context_size(bing_search_client, mock_bing_response):
    with patch.object(bing_search_client, "get_request_provider_data", return_value=None):
        bing_search_client._client.get = AsyncMock(return_value=mock_bing_response)
        await bing_search_client.search(
            None,
            "test query",
            search_context_size="high",
        )
    call_kwargs = bing_search_client._client.get.call_args
    assert call_kwargs.kwargs["params"]["count"] == 10


async def test_search_without_extra_params(bing_search_client, mock_bing_response):
    with patch.object(bing_search_client, "get_request_provider_data", return_value=None):
        bing_search_client._client.get = AsyncMock(return_value=mock_bing_response)
        await bing_search_client.search(None, "test query")
    call_kwargs = bing_search_client._client.get.call_args
    params = call_kwargs.kwargs["params"]
    assert params["q"] == "test query"
    assert params["count"] == 3
    assert "cc" not in params


async def test_search_with_empty_allowed_domains(bing_search_client, mock_bing_response):
    with patch.object(bing_search_client, "get_request_provider_data", return_value=None):
        bing_search_client._client.get = AsyncMock(return_value=mock_bing_response)
        await bing_search_client.search(
            None,
            "test query",
            allowed_domains=[],
        )
    call_kwargs = bing_search_client._client.get.call_args
    assert call_kwargs.kwargs["params"]["q"] == "test query"


async def test_search_returns_sources(bing_search_client, mock_bing_response):
    """Test that search returns source URLs and the searched query."""
    with patch.object(bing_search_client, "get_request_provider_data", return_value=None):
        bing_search_client._client.get = AsyncMock(return_value=mock_bing_response)
        result = await bing_search_client.search(None, "test query")
    assert len(result.sources) == 1
    assert result.sources[0].url == "https://example.com"
    assert result.query == "test query"


class TestProviderDataApiKeyOverride:
    async def test_provider_data_api_key_overrides_config_api_key(self, bing_search_client, mock_bing_response):
        """Provider data API key should override the config API key."""
        with patch.object(
            bing_search_client,
            "get_request_provider_data",
            return_value=MagicMock(bing_search_api_key=SecretStr("provider-data-key")),
        ):
            bing_search_client._client.get = AsyncMock(return_value=mock_bing_response)
            await bing_search_client.search(None, "test query")
        headers = bing_search_client._client.get.call_args.kwargs["headers"]
        assert headers["Ocp-Apim-Subscription-Key"] == "provider-data-key"

    async def test_config_api_key_used_when_no_provider_data(self, bing_search_client, mock_bing_response):
        """Config API key should be used when no provider data is provided."""
        with patch.object(bing_search_client, "get_request_provider_data", return_value=None):
            bing_search_client._client.get = AsyncMock(return_value=mock_bing_response)
            await bing_search_client.search(None, "test query")
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
            await bing_search_client.search(None, "test query")
        headers = bing_search_client._client.get.call_args.kwargs["headers"]
        assert headers["Ocp-Apim-Subscription-Key"] == "config-key"

    def test_returned_api_key_is_none_when_both_keys_null(self):
        """_get_api_key should return None when both config and provider data keys are None."""
        impl = BingSearchWebSearchImpl(BingSearchConfig(top_k=3))
        with patch.object(impl, "get_request_provider_data", return_value=None):
            assert impl._get_api_key() is None

    async def test_api_key_header_omitted_when_no_key_available(self, mock_bing_response):
        """Request should not include Ocp-Apim-Subscription-Key header when no key is available."""
        impl = BingSearchWebSearchImpl(BingSearchConfig(top_k=3))
        impl._client = MagicMock(spec=httpx2.AsyncClient)
        impl._client.get = AsyncMock(return_value=mock_bing_response)
        with patch.object(impl, "get_request_provider_data", return_value=None):
            await impl.search(None, "test query")
        headers = impl._client.get.call_args.kwargs["headers"]
        assert "Ocp-Apim-Subscription-Key" not in headers


async def test_401_raises_with_actionable_error(bing_search_client):
    unauthorized = httpx2.Response(
        401,
        json={"statusCode": 401},
        request=httpx2.Request("GET", "https://api.bing.microsoft.com/v7.0/search"),
    )
    bing_search_client._client.get = AsyncMock(return_value=unauthorized)
    with patch.object(bing_search_client, "get_request_provider_data", return_value=None):
        with pytest.raises(RuntimeError, match="Bing Search: the API key is missing or invalid"):
            await bing_search_client.search(None, "test query")


async def test_403_raises_with_actionable_error(bing_search_client):
    forbidden = httpx2.Response(
        403,
        request=httpx2.Request("GET", "https://api.bing.microsoft.com/v7.0/search"),
    )
    bing_search_client._client.get = AsyncMock(return_value=forbidden)
    with patch.object(bing_search_client, "get_request_provider_data", return_value=None):
        with pytest.raises(RuntimeError, match="entitlement"):
            await bing_search_client.search(None, "test query")


async def test_other_http_errors_raise(bing_search_client):
    server_error = httpx2.Response(
        500,
        request=httpx2.Request("GET", "https://api.bing.microsoft.com/v7.0/search"),
    )
    bing_search_client._client.get = AsyncMock(return_value=server_error)
    with patch.object(bing_search_client, "get_request_provider_data", return_value=None):
        with pytest.raises(httpx2.HTTPStatusError):
            await bing_search_client.search(None, "test query")
