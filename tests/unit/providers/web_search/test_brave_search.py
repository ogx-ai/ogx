# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx2
import pytest
from pydantic import HttpUrl, SecretStr, ValidationError

from ogx.core.stack import replace_env_vars
from ogx.providers.remote.web_search.brave_search.brave_search import BraveSearchWebSearchImpl
from ogx.providers.remote.web_search.brave_search.config import BraveSearchConfig


@pytest.fixture
def brave_search():
    impl = BraveSearchWebSearchImpl(BraveSearchConfig(api_key="config-key", max_results=3))
    # No per-request provider data by default; _get_api_key always consults it.
    impl.get_request_provider_data = MagicMock(return_value=None)
    return impl


@pytest.fixture
def brave_search_client(brave_search):
    brave_search._client = MagicMock(spec=httpx2.AsyncClient)
    return brave_search


@pytest.fixture
def mock_brave_response():
    return httpx2.Response(
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
        request=httpx2.Request("GET", "https://api.search.brave.com/res/v1/web/search"),
    )


async def test_search_with_allowed_domains(brave_search_client, mock_brave_response):
    brave_search_client._client.get = AsyncMock(return_value=mock_brave_response)
    await brave_search_client.search(
        None,
        "test query",
        allowed_domains=["example.com", "docs.example.com"],
    )
    call_kwargs = brave_search_client._client.get.call_args
    query_param = call_kwargs.kwargs["params"]["q"]
    assert "site:example.com" in query_param
    assert "site:docs.example.com" in query_param
    assert query_param == "test query (site:example.com OR site:docs.example.com)"


async def test_search_with_user_location_country(brave_search_client, mock_brave_response):
    brave_search_client._client.get = AsyncMock(return_value=mock_brave_response)
    await brave_search_client.search(
        None,
        "test query",
        user_location={"country": "US", "city": "San Francisco"},
    )
    call_kwargs = brave_search_client._client.get.call_args
    assert call_kwargs.kwargs["params"]["country"] == "US"


async def test_search_with_user_location_no_country(brave_search_client, mock_brave_response):
    brave_search_client._client.get = AsyncMock(return_value=mock_brave_response)
    await brave_search_client.search(
        None,
        "test query",
        user_location={"city": "San Francisco"},
    )
    call_kwargs = brave_search_client._client.get.call_args
    assert "country" not in call_kwargs.kwargs["params"]


async def test_search_with_search_context_size(brave_search_client, mock_brave_response):
    brave_search_client._client.get = AsyncMock(return_value=mock_brave_response)
    await brave_search_client.search(
        None,
        "test query",
        search_context_size="high",
    )
    call_kwargs = brave_search_client._client.get.call_args
    assert call_kwargs.kwargs["params"]["count"] == 10


async def test_search_with_search_context_size_updates_result_limit(brave_search_client):
    multi_result_response = httpx2.Response(
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
                        "title": f"Result {i}",
                        "url": f"https://example{i}.com",
                        "description": "A test result",
                        "date": "2025-01-01",
                        "extra_snippets": [f"snippet{i}"],
                    }
                    for i in range(5)
                ]
            },
        },
        request=httpx2.Request("GET", "https://api.search.brave.com/res/v1/web/search"),
    )
    brave_search_client._client.get = AsyncMock(return_value=multi_result_response)
    result = await brave_search_client.search(
        None,
        "test query",
        search_context_size="medium",
    )
    call_kwargs = brave_search_client._client.get.call_args
    assert call_kwargs.kwargs["params"]["count"] == 5
    assert len(result.sources) == 5
    source_urls = {s.url for s in result.sources}
    expected_urls = {f"https://example{i}.com" for i in range(5)}
    assert source_urls == expected_urls


async def test_search_without_extra_params(brave_search_client, mock_brave_response):
    brave_search_client._client.get = AsyncMock(return_value=mock_brave_response)
    await brave_search_client.search(None, "test query")
    call_kwargs = brave_search_client._client.get.call_args
    params = call_kwargs.kwargs["params"]
    assert params["q"] == "test query"
    assert "country" not in params
    assert "count" not in params


async def test_search_with_empty_allowed_domains(brave_search_client, mock_brave_response):
    brave_search_client._client.get = AsyncMock(return_value=mock_brave_response)
    await brave_search_client.search(
        None,
        "test query",
        allowed_domains=[],
    )
    call_kwargs = brave_search_client._client.get.call_args
    assert call_kwargs.kwargs["params"]["q"] == "test query"


async def test_search_returns_sources(brave_search_client, mock_brave_response):
    """Test that search returns source URLs and the searched query."""
    brave_search_client._client.get = AsyncMock(return_value=mock_brave_response)
    result = await brave_search_client.search(None, "test query")
    assert len(result.sources) == 1
    assert result.sources[0].url == "https://example.com"
    assert result.query == "test query"


class TestProviderDataApiKeyOverride:
    async def test_provider_data_api_key_overrides_config_api_key(self, brave_search_client, mock_brave_response):
        """Provider data API key should override the config API key."""
        with patch.object(
            brave_search_client,
            "get_request_provider_data",
            return_value=MagicMock(brave_search_api_key=SecretStr("provider-data-key")),
        ):
            brave_search_client._client.get = AsyncMock(return_value=mock_brave_response)
            await brave_search_client.search(None, "test query")
            headers = brave_search_client._client.get.call_args.kwargs["headers"]
            assert headers["X-Subscription-Token"] == "provider-data-key"

    async def test_config_api_key_used_when_no_provider_data(self, brave_search_client, mock_brave_response):
        """Config API key should be used when no provider data is provided."""
        brave_search_client._client.get = AsyncMock(return_value=mock_brave_response)
        await brave_search_client.search(None, "test query")
        headers = brave_search_client._client.get.call_args.kwargs["headers"]
        assert headers["X-Subscription-Token"] == "config-key"

    async def test_config_api_key_used_when_provider_data_key_is_none(self, brave_search_client, mock_brave_response):
        """Config API key should be used when provider data key is None."""
        with patch.object(
            brave_search_client,
            "get_request_provider_data",
            return_value=MagicMock(brave_search_api_key=None),
        ):
            brave_search_client._client.get = AsyncMock(return_value=mock_brave_response)
            await brave_search_client.search(None, "test query")
            headers = brave_search_client._client.get.call_args.kwargs["headers"]
            assert headers["X-Subscription-Token"] == "config-key"

    async def test_returned_api_key_is_none_when_no_keys(self):
        """_get_api_key should return None when both config and provider data keys are absent."""
        impl = BraveSearchWebSearchImpl(BraveSearchConfig(max_results=3))
        impl.get_request_provider_data = MagicMock(return_value=None)
        assert impl._get_api_key() is None

    async def test_empty_api_key_treated_as_missing(self):
        """An empty key (as produced by ${env.X:=} in keyless configs) must count
        as no key at all: _get_api_key returns None and the backend reports
        itself unavailable. This pins pydantic's SecretStr('') falsiness, which
        the keyless-distro availability contract relies on."""
        impl = BraveSearchWebSearchImpl(BraveSearchConfig(api_key=""))
        impl.get_request_provider_data = MagicMock(return_value=None)
        assert impl._get_api_key() is None
        assert await impl.is_available() is False

    async def test_api_key_header_omitted_when_both_keys_null(self, mock_brave_response):
        """Request should not include X-Subscription-Token header when no key is available."""
        impl = BraveSearchWebSearchImpl(BraveSearchConfig(max_results=3))
        impl._client = MagicMock(spec=httpx2.AsyncClient)
        impl._client.get = AsyncMock(return_value=mock_brave_response)
        impl.get_request_provider_data = MagicMock(return_value=None)
        await impl.search(None, "test query")
        headers = impl._client.get.call_args.kwargs["headers"]
        assert "X-Subscription-Token" not in headers


class TestBaseUrl:
    async def test_default_base_url(self, brave_search_client, mock_brave_response):
        """Requests go to the public Brave endpoint when no base URL is configured."""
        brave_search_client._client.get = AsyncMock(return_value=mock_brave_response)
        await brave_search_client.search(None, "test query")
        assert (
            brave_search_client._client.get.call_args.kwargs["url"] == "https://api.search.brave.com/res/v1/web/search"
        )

    @pytest.mark.parametrize("base_url", ["http://brave-proxy:8080/brave", "http://brave-proxy:8080/brave/"])
    async def test_custom_base_url(self, base_url, mock_brave_response):
        """A configured base URL, with or without a trailing slash, keeps its path prefix."""
        impl = BraveSearchWebSearchImpl(BraveSearchConfig(base_url=base_url))
        impl._client = MagicMock(spec=httpx2.AsyncClient)
        impl._client.get = AsyncMock(return_value=mock_brave_response)
        impl.get_request_provider_data = MagicMock(return_value=None)
        await impl.search(None, "test query")
        assert impl._client.get.call_args.kwargs["url"] == "http://brave-proxy:8080/brave/res/v1/web/search"

    @patch.dict(os.environ, {"BRAVE_SEARCH_BASE_URL": "http://brave-proxy:8080"})
    def test_sample_run_config_reads_base_url_from_environment(self):
        config_data = replace_env_vars(BraveSearchConfig.sample_run_config(__distro_dir__=""))
        assert BraveSearchConfig.model_validate(config_data).base_url == HttpUrl("http://brave-proxy:8080")

    def test_sample_run_config_defaults_to_public_endpoint(self):
        with patch.dict(os.environ):
            os.environ.pop("BRAVE_SEARCH_BASE_URL", None)
            config_data = replace_env_vars(BraveSearchConfig.sample_run_config(__distro_dir__=""))
        assert BraveSearchConfig.model_validate(config_data).base_url == HttpUrl("https://api.search.brave.com")

    @pytest.mark.parametrize("base_url", ["", "not a url", "brave-proxy:8080"])
    def test_invalid_base_url_fails_at_config_load(self, base_url):
        """A bad BRAVE_SEARCH_BASE_URL is rejected up front, not as an opaque error at request time."""
        with pytest.raises(ValidationError):
            BraveSearchConfig(base_url=base_url)


async def test_search_before_initialize_raises(brave_search):
    with pytest.raises(RuntimeError, match="not initialized"):
        await brave_search.search(None, "test query")


async def test_shutdown_closes_client():
    impl = BraveSearchWebSearchImpl(BraveSearchConfig(api_key="config-key"))
    await impl.initialize()
    assert isinstance(impl._client, httpx2.AsyncClient)
    await impl.shutdown()
    assert impl._client is None


async def test_401_raises_with_actionable_error(brave_search_client):
    unauthorized = httpx2.Response(
        401,
        json={"detail": "Unauthorized"},
        request=httpx2.Request("GET", "https://api.search.brave.com/res/v1/web/search"),
    )
    brave_search_client._client.get = AsyncMock(return_value=unauthorized)
    with pytest.raises(RuntimeError, match="Brave Search: the API key is missing or invalid"):
        await brave_search_client.search(None, "test query")


async def test_403_raises_with_actionable_error(brave_search_client):
    forbidden = httpx2.Response(
        403,
        request=httpx2.Request("GET", "https://api.search.brave.com/res/v1/web/search"),
    )
    brave_search_client._client.get = AsyncMock(return_value=forbidden)
    with pytest.raises(RuntimeError, match="entitlement"):
        await brave_search_client.search(None, "test query")


async def test_other_http_errors_raise(brave_search_client):
    server_error = httpx2.Response(
        500,
        request=httpx2.Request("GET", "https://api.search.brave.com/res/v1/web/search"),
    )
    brave_search_client._client.get = AsyncMock(return_value=server_error)
    with pytest.raises(httpx2.HTTPStatusError):
        await brave_search_client.search(None, "test query")
