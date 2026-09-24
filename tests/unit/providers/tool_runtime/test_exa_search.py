# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import SecretStr

from ogx.providers.remote.tool_runtime.exa_search.config import ExaSearchToolConfig
from ogx.providers.remote.tool_runtime.exa_search.exa_search import ExaSearchToolRuntimeImpl


@pytest.fixture
def exa_search():
    return ExaSearchToolRuntimeImpl(ExaSearchToolConfig(api_key="test-key", max_results=3))


@pytest.fixture
def exa_search_client(exa_search):
    exa_search._client = MagicMock(spec=httpx.AsyncClient)
    return exa_search


@pytest.fixture
def mock_exa_response():
    return httpx.Response(
        200,
        json={
            "requestId": "req-123",
            "results": [
                {
                    "title": "Test Result",
                    "url": "https://example.com",
                    "publishedDate": "2024-01-01",
                    "author": "Test Author",
                    "highlights": ["A test highlight"],
                    "score": 0.95,
                }
            ],
        },
        request=httpx.Request("POST", "https://api.exa.ai/search"),
    )


async def test_invoke_with_allowed_domains(exa_search_client, mock_exa_response):
    with patch.object(exa_search_client, "get_request_provider_data", return_value=None):
        exa_search_client._client.post = AsyncMock(return_value=mock_exa_response)
        await exa_search_client.invoke_tool(
            "web_search",
            {
                "query": "test query",
                "allowed_domains": ["example.com", "docs.example.com"],
            },
        )
    call_kwargs = exa_search_client._client.post.call_args
    request_body = call_kwargs.kwargs["json"]
    assert request_body["includeDomains"] == ["example.com", "docs.example.com"]


async def test_invoke_with_search_context_size(exa_search_client, mock_exa_response):
    with patch.object(exa_search_client, "get_request_provider_data", return_value=None):
        exa_search_client._client.post = AsyncMock(return_value=mock_exa_response)
        await exa_search_client.invoke_tool(
            "web_search",
            {
                "query": "test query",
                "search_context_size": "high",
            },
        )
    call_kwargs = exa_search_client._client.post.call_args
    request_body = call_kwargs.kwargs["json"]
    assert request_body["numResults"] == 10


async def test_invoke_without_extra_params(exa_search_client, mock_exa_response):
    with patch.object(exa_search_client, "get_request_provider_data", return_value=None):
        exa_search_client._client.post = AsyncMock(return_value=mock_exa_response)
        await exa_search_client.invoke_tool(
            "web_search",
            {"query": "test query"},
        )
    call_kwargs = exa_search_client._client.post.call_args
    request_body = call_kwargs.kwargs["json"]
    assert request_body == {
        "query": "test query",
        "type": "auto",
        "contents": {"highlights": True},
        "numResults": 3,
    }
    headers = call_kwargs.kwargs["headers"]
    assert headers["x-api-key"] == "test-key"


async def test_invoke_with_user_location_country(exa_search_client, mock_exa_response):
    with patch.object(exa_search_client, "get_request_provider_data", return_value=None):
        exa_search_client._client.post = AsyncMock(return_value=mock_exa_response)
        await exa_search_client.invoke_tool(
            "web_search",
            {
                "query": "test query",
                "user_location": {"country": "US", "city": "San Francisco"},
            },
        )
    call_kwargs = exa_search_client._client.post.call_args
    assert call_kwargs.kwargs["json"]["userLocation"] == "US"


async def test_invoke_with_empty_allowed_domains(exa_search_client, mock_exa_response):
    with patch.object(exa_search_client, "get_request_provider_data", return_value=None):
        exa_search_client._client.post = AsyncMock(return_value=mock_exa_response)
        await exa_search_client.invoke_tool(
            "web_search",
            {
                "query": "test query",
                "allowed_domains": [],
            },
        )
    call_kwargs = exa_search_client._client.post.call_args
    request_body = call_kwargs.kwargs["json"]
    assert "includeDomains" not in request_body


async def test_invoke_returns_source_metadata(exa_search_client, mock_exa_response):
    """Test that invoke_tool returns source URLs in metadata."""
    with patch.object(exa_search_client, "get_request_provider_data", return_value=None):
        exa_search_client._client.post = AsyncMock(return_value=mock_exa_response)
        result = await exa_search_client.invoke_tool(
            tool_name="web_search",
            kwargs={"query": "test query"},
        )
    assert result.metadata is not None
    assert "sources" in result.metadata
    assert len(result.metadata["sources"]) == 1
    assert result.metadata["sources"][0]["url"] == "https://example.com"
    assert result.metadata["query"] == "test query"


class TestProviderDataApiKeyOverride:
    async def test_provider_data_api_key_overrides_config_api_key(self, exa_search_client, mock_exa_response):
        """Provider data API key should override the config API key."""
        with patch.object(
            exa_search_client,
            "get_request_provider_data",
            return_value=MagicMock(exa_search_api_key=SecretStr("provider-data-key")),
        ):
            exa_search_client._client.post = AsyncMock(return_value=mock_exa_response)
            await exa_search_client.invoke_tool(
                "web_search",
                {"query": "test query"},
            )
        headers = exa_search_client._client.post.call_args.kwargs["headers"]
        assert headers["x-api-key"] == "provider-data-key"

    async def test_config_api_key_used_when_no_provider_data(self, exa_search_client, mock_exa_response):
        """Config API key should be used when no provider data is provided."""
        with patch.object(exa_search_client, "get_request_provider_data", return_value=None):
            exa_search_client._client.post = AsyncMock(return_value=mock_exa_response)
            await exa_search_client.invoke_tool(
                "web_search",
                {"query": "test query"},
            )
        headers = exa_search_client._client.post.call_args.kwargs["headers"]
        assert headers["x-api-key"] == "test-key"

    async def test_config_api_key_used_when_provider_data_key_is_none(self, exa_search_client, mock_exa_response):
        """Config API key should be used when provider data key is None."""
        with patch.object(
            exa_search_client,
            "get_request_provider_data",
            return_value=MagicMock(exa_search_api_key=None),
        ):
            exa_search_client._client.post = AsyncMock(return_value=mock_exa_response)
            await exa_search_client.invoke_tool(
                "web_search",
                {"query": "test query"},
            )
        headers = exa_search_client._client.post.call_args.kwargs["headers"]
        assert headers["x-api-key"] == "test-key"

    async def test_returned_api_key_is_none_when_no_keys(self):
        """_get_api_key should return None when both config and provider data keys are absent."""
        impl = ExaSearchToolRuntimeImpl(ExaSearchToolConfig(max_results=3))
        with patch.object(impl, "get_request_provider_data", return_value=None):
            assert impl._get_api_key() is None

    async def test_api_key_omitted_from_body_when_both_keys_null(self):
        """Request headers should not include x-api-key when no key is available."""
        mock_response = httpx.Response(
            200,
            json={
                "requestId": "req-123",
                "results": [],
            },
            request=httpx.Request("POST", "https://api.exa.ai/search"),
        )
        impl = ExaSearchToolRuntimeImpl(ExaSearchToolConfig(max_results=3))
        impl._client = MagicMock(spec=httpx.AsyncClient)
        impl._client.post = AsyncMock(return_value=mock_response)
        with patch.object(impl, "get_request_provider_data", return_value=None):
            await impl.invoke_tool("web_search", {"query": "test query"})
        headers = impl._client.post.call_args.kwargs["headers"]
        assert "x-api-key" not in headers
