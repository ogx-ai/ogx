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

from ogx.providers.remote.tool_runtime.serply_search.config import SerplySearchToolConfig
from ogx.providers.remote.tool_runtime.serply_search.serply_search import SerplySearchToolRuntimeImpl


@pytest.fixture
def serply_search():
    impl = SerplySearchToolRuntimeImpl(SerplySearchToolConfig(api_key="test-key", max_results=3))
    impl._client = MagicMock(spec=httpx.AsyncClient)
    impl.get_request_provider_data = MagicMock(return_value=None)
    return impl


def _serply_response(count: int = 1) -> httpx.Response:
    # Shape of GET /v1/search: each organic result carries title, link, and description.
    return httpx.Response(
        200,
        json={
            "results": [
                {
                    "title": f"Test Result {i}",
                    "link": f"https://example.com/{i}",
                    "description": f"A test result description {i}",
                }
                for i in range(count)
            ],
        },
        request=httpx.Request("GET", SerplySearchToolRuntimeImpl._SEARCH_URL),
    )


@pytest.fixture
def mock_serply_response():
    return _serply_response()


async def test_invoke_without_extra_params_sends_config_defaults(serply_search, mock_serply_response):
    serply_search._client.get = AsyncMock(return_value=mock_serply_response)
    await serply_search.invoke_tool("web_search", {"query": "test query"})
    params = serply_search._client.get.call_args.kwargs["params"]
    assert params == {"q": "test query", "num": 3}


async def test_invoke_with_allowed_domains(serply_search, mock_serply_response):
    serply_search._client.get = AsyncMock(return_value=mock_serply_response)
    await serply_search.invoke_tool(
        "web_search",
        {"query": "test query", "allowed_domains": ["example.com", "docs.example.com"]},
    )
    params = serply_search._client.get.call_args.kwargs["params"]
    assert params["q"] == "test query (site:example.com OR site:docs.example.com)"


async def test_invoke_with_empty_allowed_domains(serply_search, mock_serply_response):
    serply_search._client.get = AsyncMock(return_value=mock_serply_response)
    await serply_search.invoke_tool("web_search", {"query": "test query", "allowed_domains": []})
    params = serply_search._client.get.call_args.kwargs["params"]
    assert params["q"] == "test query"


@pytest.mark.parametrize("size,expected", [("low", 3), ("medium", 5), ("high", 10)])
async def test_search_context_size_mapping(serply_search, mock_serply_response, size, expected):
    serply_search._client.get = AsyncMock(return_value=mock_serply_response)
    await serply_search.invoke_tool("web_search", {"query": "q", "search_context_size": size})
    params = serply_search._client.get.call_args.kwargs["params"]
    assert params["num"] == expected


async def test_unknown_search_context_size_keeps_config_default(serply_search, mock_serply_response):
    serply_search._client.get = AsyncMock(return_value=mock_serply_response)
    await serply_search.invoke_tool("web_search", {"query": "q", "search_context_size": "ultra"})
    params = serply_search._client.get.call_args.kwargs["params"]
    assert params["num"] == 3


async def test_max_results_is_clamped_to_api_ceiling(mock_serply_response):
    impl = SerplySearchToolRuntimeImpl(SerplySearchToolConfig(api_key="test-key", max_results=50))
    impl._client = MagicMock(spec=httpx.AsyncClient)
    impl._client.get = AsyncMock(return_value=mock_serply_response)
    impl.get_request_provider_data = MagicMock(return_value=None)
    await impl.invoke_tool("web_search", {"query": "q"})
    assert impl._client.get.call_args.kwargs["params"]["num"] == 10


async def test_results_are_trimmed_to_the_limit(serply_search):
    # `num` is an upper bound on the API side, so extra rows are dropped client-side.
    serply_search._client.get = AsyncMock(return_value=_serply_response(count=8))
    result = await serply_search.invoke_tool("web_search", {"query": "q"})
    assert len(json.loads(result.content)["results"]) == 3
    assert len(result.metadata["sources"]) == 3


async def test_results_map_link_and_description(serply_search, mock_serply_response):
    serply_search._client.get = AsyncMock(return_value=mock_serply_response)
    result = await serply_search.invoke_tool("web_search", {"query": "test query"})
    payload = json.loads(result.content)
    assert payload["query"] == "test query"
    assert payload["results"] == [
        {
            "title": "Test Result 0",
            "url": "https://example.com/0",
            "content": "A test result description 0",
        }
    ]


async def test_invoke_returns_source_metadata(serply_search, mock_serply_response):
    serply_search._client.get = AsyncMock(return_value=mock_serply_response)
    result = await serply_search.invoke_tool(tool_name="web_search", kwargs={"query": "test query"})
    assert result.metadata is not None
    assert result.metadata["query"] == "test query"
    assert result.metadata["sources"] == [{"url": "https://example.com/0"}]


async def test_user_location_country_is_forwarded_as_gl(serply_search, mock_serply_response):
    serply_search._client.get = AsyncMock(return_value=mock_serply_response)
    await serply_search.invoke_tool(
        "web_search",
        {"query": "q", "user_location": {"country": "GB", "city": "London"}},
    )
    params = serply_search._client.get.call_args.kwargs["params"]
    assert params["gl"] == "gb"
    assert "user_location" not in params


async def test_api_key_sent_as_header_not_params(serply_search, mock_serply_response):
    serply_search._client.get = AsyncMock(return_value=mock_serply_response)
    await serply_search.invoke_tool("web_search", {"query": "test query"})
    call = serply_search._client.get.call_args
    assert call.kwargs["headers"]["X-Api-Key"] == "test-key"
    assert call.kwargs["headers"]["User-Agent"] == "ogx"
    assert "test-key" not in json.dumps(call.kwargs["params"])


async def test_missing_api_key_sends_no_auth_header(mock_serply_response):
    impl = SerplySearchToolRuntimeImpl(SerplySearchToolConfig(api_key=None))
    impl._client = MagicMock(spec=httpx.AsyncClient)
    impl._client.get = AsyncMock(return_value=mock_serply_response)
    impl.get_request_provider_data = MagicMock(return_value=None)
    await impl.invoke_tool("web_search", {"query": "q"})
    assert "X-Api-Key" not in impl._client.get.call_args.kwargs["headers"]


async def test_provider_data_overrides_config_api_key(serply_search, mock_serply_response):
    serply_search._client.get = AsyncMock(return_value=mock_serply_response)
    provider_data = MagicMock()
    provider_data.serply_search_api_key = SecretStr("override-key")
    serply_search.get_request_provider_data = MagicMock(return_value=provider_data)
    await serply_search.invoke_tool("web_search", {"query": "q"})
    assert serply_search._client.get.call_args.kwargs["headers"]["X-Api-Key"] == "override-key"


async def test_401_returns_graceful_tool_error(serply_search):
    unauthorized = httpx.Response(
        401,
        json={"detail": "Invalid API key"},
        request=httpx.Request("GET", SerplySearchToolRuntimeImpl._SEARCH_URL),
    )
    serply_search._client.get = AsyncMock(return_value=unauthorized)
    result = await serply_search.invoke_tool("web_search", {"query": "q"})
    assert result.error_code == 401
    assert "Serply Search" in result.error_message
    assert "Serply Search" in json.loads(result.content)["error"]
    assert result.metadata["sources"] == []


async def test_other_http_errors_raise(serply_search):
    server_error = httpx.Response(
        500,
        request=httpx.Request("GET", SerplySearchToolRuntimeImpl._SEARCH_URL),
    )
    serply_search._client.get = AsyncMock(return_value=server_error)
    with pytest.raises(httpx.HTTPStatusError):
        await serply_search.invoke_tool("web_search", {"query": "q"})
