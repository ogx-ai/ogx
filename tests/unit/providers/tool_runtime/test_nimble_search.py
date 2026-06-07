# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from pydantic import SecretStr

from ogx.providers.remote.tool_runtime.nimble_search.config import NimbleSearchToolConfig
from ogx.providers.remote.tool_runtime.nimble_search.nimble_search import NimbleSearchToolRuntimeImpl


@pytest.fixture
def nimble_search():
    impl = NimbleSearchToolRuntimeImpl(NimbleSearchToolConfig(api_key="test-key", max_results=3))
    impl._client = MagicMock(spec=httpx.AsyncClient)
    # No per-request provider data by default; _get_api_key now always consults it.
    impl.get_request_provider_data = MagicMock(return_value=None)
    return impl


@pytest.fixture
def mock_nimble_response():
    # Shape of POST /v1/search per nimble-python v0.18 SearchResponse: each result
    # carries title, url, content, description, and metadata. In 'lite' depth the
    # content is empty and the description carries the text.
    return httpx.Response(
        200,
        json={
            "request_id": "00000000-0000-0000-0000-000000000000",
            "total_results": 1,
            "answer": None,
            "results": [
                {
                    "title": "Test Result",
                    "url": "https://example.com",
                    "content": "",
                    "description": "A test result description",
                    "metadata": {
                        "country": "US",
                        "locale": "en",
                        "entity_type": "OrganicResult",
                        "position": 1,
                    },
                }
            ],
        },
        request=httpx.Request("POST", NimbleSearchToolRuntimeImpl._SEARCH_URL),
    )


async def test_invoke_with_allowed_domains(nimble_search, mock_nimble_response):
    nimble_search._client.post = AsyncMock(return_value=mock_nimble_response)
    await nimble_search.invoke_tool(
        "web_search",
        {"query": "test query", "allowed_domains": ["example.com", "docs.example.com"]},
    )
    request_body = nimble_search._client.post.call_args.kwargs["json"]
    assert request_body["include_domains"] == ["example.com", "docs.example.com"]


async def test_invoke_with_search_context_size(nimble_search, mock_nimble_response):
    nimble_search._client.post = AsyncMock(return_value=mock_nimble_response)
    await nimble_search.invoke_tool(
        "web_search",
        {"query": "test query", "search_context_size": "high"},
    )
    request_body = nimble_search._client.post.call_args.kwargs["json"]
    assert request_body["max_results"] == 10


async def test_invoke_without_extra_params_sends_config_defaults(nimble_search, mock_nimble_response):
    nimble_search._client.post = AsyncMock(return_value=mock_nimble_response)
    await nimble_search.invoke_tool("web_search", {"query": "test query"})
    request_body = nimble_search._client.post.call_args.kwargs["json"]
    assert request_body["query"] == "test query"
    assert request_body["max_results"] == 3
    assert request_body["search_depth"] == "lite"
    assert "country" not in request_body
    assert "locale" not in request_body
    assert "include_domains" not in request_body


async def test_api_key_sent_as_bearer_header_not_body(nimble_search, mock_nimble_response):
    nimble_search._client.post = AsyncMock(return_value=mock_nimble_response)
    await nimble_search.invoke_tool("web_search", {"query": "test query"})
    call = nimble_search._client.post.call_args
    assert call.kwargs["headers"]["Authorization"] == "Bearer test-key"
    # The key must never leak into the request body.
    assert "api_key" not in call.kwargs["json"]


async def test_enterprise_only_params_never_sent(nimble_search, mock_nimble_response):
    nimble_search._client.post = AsyncMock(return_value=mock_nimble_response)
    await nimble_search.invoke_tool("web_search", {"query": "test query"})
    request_body = nimble_search._client.post.call_args.kwargs["json"]
    assert "include_answer" not in request_body
    assert request_body["search_depth"] != "fast"


async def test_lite_body_falls_back_to_description(nimble_search, mock_nimble_response):
    nimble_search._client.post = AsyncMock(return_value=mock_nimble_response)
    result = await nimble_search.invoke_tool("web_search", {"query": "test query"})
    payload = json.loads(result.content)
    assert payload["results"][0]["content"] == "A test result description"


async def test_invoke_with_empty_allowed_domains(nimble_search, mock_nimble_response):
    nimble_search._client.post = AsyncMock(return_value=mock_nimble_response)
    await nimble_search.invoke_tool("web_search", {"query": "test query", "allowed_domains": []})
    request_body = nimble_search._client.post.call_args.kwargs["json"]
    assert "include_domains" not in request_body


async def test_invoke_returns_source_metadata(nimble_search, mock_nimble_response):
    nimble_search._client.post = AsyncMock(return_value=mock_nimble_response)
    result = await nimble_search.invoke_tool(tool_name="web_search", kwargs={"query": "test query"})
    assert result.metadata is not None
    assert result.metadata["query"] == "test query"
    assert result.metadata["sources"] == [{"url": "https://example.com"}]


@pytest.mark.parametrize("size,expected", [("low", 3), ("medium", 5), ("high", 10)])
async def test_search_context_size_mapping(nimble_search, mock_nimble_response, size, expected):
    nimble_search._client.post = AsyncMock(return_value=mock_nimble_response)
    await nimble_search.invoke_tool("web_search", {"query": "q", "search_context_size": size})
    request_body = nimble_search._client.post.call_args.kwargs["json"]
    assert request_body["max_results"] == expected


async def test_unknown_search_context_size_keeps_config_default(nimble_search, mock_nimble_response):
    nimble_search._client.post = AsyncMock(return_value=mock_nimble_response)
    await nimble_search.invoke_tool("web_search", {"query": "q", "search_context_size": "ultra"})
    request_body = nimble_search._client.post.call_args.kwargs["json"]
    assert request_body["max_results"] == 3


async def test_user_location_country_is_forwarded(nimble_search, mock_nimble_response):
    nimble_search._client.post = AsyncMock(return_value=mock_nimble_response)
    await nimble_search.invoke_tool(
        "web_search",
        {"query": "q", "user_location": {"country": "GB", "city": "London"}},
    )
    request_body = nimble_search._client.post.call_args.kwargs["json"]
    # Geo comes from the per-request user_location, not server config.
    assert request_body["country"] == "GB"
    assert "user_location" not in request_body


async def test_no_user_location_means_no_country(nimble_search, mock_nimble_response):
    nimble_search._client.post = AsyncMock(return_value=mock_nimble_response)
    await nimble_search.invoke_tool("web_search", {"query": "q"})
    request_body = nimble_search._client.post.call_args.kwargs["json"]
    assert "country" not in request_body
    assert "locale" not in request_body


async def test_missing_api_key_sends_no_auth_header(mock_nimble_response):
    impl = NimbleSearchToolRuntimeImpl(NimbleSearchToolConfig(api_key=None))
    impl._client = MagicMock(spec=httpx.AsyncClient)
    impl._client.post = AsyncMock(return_value=mock_nimble_response)
    # No config key and no per-request provider data -> no Authorization header; the API
    # rejects the request, matching the sibling search providers (no early raise).
    impl.get_request_provider_data = MagicMock(return_value=None)
    await impl.invoke_tool("web_search", {"query": "q"})
    assert "Authorization" not in impl._client.post.call_args.kwargs["headers"]


async def test_provider_data_overrides_config_api_key(nimble_search, mock_nimble_response):
    nimble_search._client.post = AsyncMock(return_value=mock_nimble_response)
    # Config carries "test-key"; a per-request provider-data key must take precedence.
    provider_data = MagicMock()
    provider_data.nimble_search_api_key = SecretStr("override-key")
    nimble_search.get_request_provider_data = MagicMock(return_value=provider_data)
    await nimble_search.invoke_tool("web_search", {"query": "q"})
    assert nimble_search._client.post.call_args.kwargs["headers"]["Authorization"] == "Bearer override-key"


async def test_403_returns_graceful_tool_error(nimble_search):
    forbidden = httpx.Response(
        403,
        json={"detail": "search_depth='fast' is not enabled for this account"},
        request=httpx.Request("POST", NimbleSearchToolRuntimeImpl._SEARCH_URL),
    )
    nimble_search._client.post = AsyncMock(return_value=forbidden)
    result = await nimble_search.invoke_tool("web_search", {"query": "q"})
    assert result.error_code == 403
    assert "Nimble Search" in result.error_message
    # The error must be forwarded to the model as content, not silently dropped.
    assert result.content
    assert "Nimble Search" in json.loads(result.content)["error"]
    assert result.metadata["sources"] == []
