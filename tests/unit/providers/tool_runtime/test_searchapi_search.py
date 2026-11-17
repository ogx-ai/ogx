# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from unittest.mock import AsyncMock, MagicMock

import httpx2
import pytest
from pydantic import SecretStr

from ogx.providers.remote.tool_runtime.searchapi_search.config import SearchApiSearchToolConfig
from ogx.providers.remote.tool_runtime.searchapi_search.searchapi_search import SearchApiSearchToolRuntimeImpl


@pytest.fixture
def searchapi_search():
    impl = SearchApiSearchToolRuntimeImpl(SearchApiSearchToolConfig(api_key="test-key", max_results=3))
    impl._client = MagicMock(spec=httpx2.AsyncClient)
    impl.get_request_provider_data = MagicMock(return_value=None)
    return impl


def _searchapi_response(count: int = 1) -> httpx2.Response:
    # Shape of GET /api/v1/search?engine=google: each organic result carries
    # position, title, link and snippet (plus fields this provider does not use).
    return httpx2.Response(
        200,
        json={
            "search_metadata": {"status": "Success"},
            "organic_results": [
                {
                    "position": i + 1,
                    "title": f"Test Result {i}",
                    "link": f"https://example.com/{i}",
                    "snippet": f"A test result snippet {i}",
                    "domain": "example.com",
                }
                for i in range(count)
            ],
        },
        request=httpx2.Request("GET", SearchApiSearchToolRuntimeImpl._SEARCH_URL),
    )


@pytest.fixture
def mock_searchapi_response():
    return _searchapi_response()


async def test_invoke_without_extra_params_sends_engine_and_query_only(searchapi_search, mock_searchapi_response):
    searchapi_search._client.get = AsyncMock(return_value=mock_searchapi_response)
    await searchapi_search.invoke_tool("web_search", {"query": "test query"})
    call = searchapi_search._client.get.call_args
    assert call.args[0] == "https://www.searchapi.io/api/v1/search"
    assert call.kwargs["params"] == {"engine": "google", "q": "test query"}


async def test_num_is_never_sent(searchapi_search, mock_searchapi_response):
    # Google phased out `num`; the provider trims the page locally instead.
    searchapi_search._client.get = AsyncMock(return_value=mock_searchapi_response)
    await searchapi_search.invoke_tool("web_search", {"query": "q", "search_context_size": "high"})
    assert "num" not in searchapi_search._client.get.call_args.kwargs["params"]


async def test_invoke_with_allowed_domains(searchapi_search, mock_searchapi_response):
    searchapi_search._client.get = AsyncMock(return_value=mock_searchapi_response)
    await searchapi_search.invoke_tool(
        "web_search",
        {"query": "test query", "allowed_domains": ["example.com", "docs.example.com"]},
    )
    params = searchapi_search._client.get.call_args.kwargs["params"]
    assert params["q"] == "test query (site:example.com OR site:docs.example.com)"


async def test_invoke_with_empty_allowed_domains(searchapi_search, mock_searchapi_response):
    searchapi_search._client.get = AsyncMock(return_value=mock_searchapi_response)
    await searchapi_search.invoke_tool("web_search", {"query": "test query", "allowed_domains": []})
    params = searchapi_search._client.get.call_args.kwargs["params"]
    assert params["q"] == "test query"


@pytest.mark.parametrize("size,expected", [("low", 3), ("medium", 5), ("high", 10)])
async def test_search_context_size_trims_the_page(searchapi_search, size, expected):
    searchapi_search._client.get = AsyncMock(return_value=_searchapi_response(count=10))
    result = await searchapi_search.invoke_tool("web_search", {"query": "q", "search_context_size": size})
    assert len(json.loads(result.content)["results"]) == expected
    assert len(result.metadata["sources"]) == expected


async def test_unknown_search_context_size_keeps_config_default(searchapi_search):
    searchapi_search._client.get = AsyncMock(return_value=_searchapi_response(count=10))
    result = await searchapi_search.invoke_tool("web_search", {"query": "q", "search_context_size": "ultra"})
    assert len(json.loads(result.content)["results"]) == 3


async def test_max_results_is_clamped_to_one_page():
    impl = SearchApiSearchToolRuntimeImpl(SearchApiSearchToolConfig(api_key="test-key", max_results=50))
    impl._client = MagicMock(spec=httpx2.AsyncClient)
    impl._client.get = AsyncMock(return_value=_searchapi_response(count=12))
    impl.get_request_provider_data = MagicMock(return_value=None)
    result = await impl.invoke_tool("web_search", {"query": "q"})
    assert len(json.loads(result.content)["results"]) == 10


async def test_fewer_results_than_the_limit_are_returned_as_is(searchapi_search):
    searchapi_search._client.get = AsyncMock(return_value=_searchapi_response(count=2))
    result = await searchapi_search.invoke_tool("web_search", {"query": "q", "search_context_size": "high"})
    assert len(json.loads(result.content)["results"]) == 2


async def test_missing_organic_results_yields_no_rows(searchapi_search):
    empty = httpx2.Response(
        200,
        json={"search_metadata": {"status": "Success"}},
        request=httpx2.Request("GET", SearchApiSearchToolRuntimeImpl._SEARCH_URL),
    )
    searchapi_search._client.get = AsyncMock(return_value=empty)
    result = await searchapi_search.invoke_tool("web_search", {"query": "q"})
    assert json.loads(result.content) == {"query": "q", "results": []}
    assert result.metadata["sources"] == []


async def test_results_map_link_and_snippet(searchapi_search, mock_searchapi_response):
    searchapi_search._client.get = AsyncMock(return_value=mock_searchapi_response)
    result = await searchapi_search.invoke_tool("web_search", {"query": "test query"})
    payload = json.loads(result.content)
    assert payload["query"] == "test query"
    assert payload["results"] == [
        {
            "title": "Test Result 0",
            "url": "https://example.com/0",
            "content": "A test result snippet 0",
        }
    ]


async def test_invoke_returns_source_metadata(searchapi_search, mock_searchapi_response):
    searchapi_search._client.get = AsyncMock(return_value=mock_searchapi_response)
    result = await searchapi_search.invoke_tool(tool_name="web_search", kwargs={"query": "test query"})
    assert result.metadata is not None
    assert result.metadata["query"] == "test query"
    assert result.metadata["sources"] == [{"url": "https://example.com/0"}]


async def test_user_location_country_is_forwarded_as_gl(searchapi_search, mock_searchapi_response):
    searchapi_search._client.get = AsyncMock(return_value=mock_searchapi_response)
    await searchapi_search.invoke_tool(
        "web_search",
        {"query": "q", "user_location": {"country": "GB", "city": "London"}},
    )
    params = searchapi_search._client.get.call_args.kwargs["params"]
    assert params["gl"] == "gb"
    assert "user_location" not in params


async def test_api_key_sent_as_bearer_header_not_params(searchapi_search, mock_searchapi_response):
    searchapi_search._client.get = AsyncMock(return_value=mock_searchapi_response)
    await searchapi_search.invoke_tool("web_search", {"query": "test query"})
    call = searchapi_search._client.get.call_args
    assert call.kwargs["headers"]["Authorization"] == "Bearer test-key"
    assert call.kwargs["headers"]["Accept"] == "application/json"
    assert call.kwargs["headers"]["User-Agent"] == "ogx"
    assert "api_key" not in call.kwargs["params"]
    assert "test-key" not in json.dumps(call.kwargs["params"])


async def test_missing_api_key_sends_no_auth_header(mock_searchapi_response):
    impl = SearchApiSearchToolRuntimeImpl(SearchApiSearchToolConfig(api_key=None))
    impl._client = MagicMock(spec=httpx2.AsyncClient)
    impl._client.get = AsyncMock(return_value=mock_searchapi_response)
    impl.get_request_provider_data = MagicMock(return_value=None)
    await impl.invoke_tool("web_search", {"query": "q"})
    assert "Authorization" not in impl._client.get.call_args.kwargs["headers"]


async def test_provider_data_overrides_config_api_key(searchapi_search, mock_searchapi_response):
    searchapi_search._client.get = AsyncMock(return_value=mock_searchapi_response)
    provider_data = MagicMock()
    provider_data.searchapi_search_api_key = SecretStr("override-key")
    searchapi_search.get_request_provider_data = MagicMock(return_value=provider_data)
    await searchapi_search.invoke_tool("web_search", {"query": "q"})
    assert searchapi_search._client.get.call_args.kwargs["headers"]["Authorization"] == "Bearer override-key"


async def test_401_returns_graceful_tool_error(searchapi_search):
    unauthorized = httpx2.Response(
        401,
        json={"error": "Invalid API key."},
        request=httpx2.Request("GET", SearchApiSearchToolRuntimeImpl._SEARCH_URL),
    )
    searchapi_search._client.get = AsyncMock(return_value=unauthorized)
    result = await searchapi_search.invoke_tool("web_search", {"query": "q"})
    assert result.error_code == 401
    assert "SearchApi Search" in result.error_message
    assert "SearchApi Search" in json.loads(result.content)["error"]
    assert result.metadata["sources"] == []


async def test_other_http_errors_raise(searchapi_search):
    server_error = httpx2.Response(
        500,
        request=httpx2.Request("GET", SearchApiSearchToolRuntimeImpl._SEARCH_URL),
    )
    searchapi_search._client.get = AsyncMock(return_value=server_error)
    with pytest.raises(httpx2.HTTPStatusError):
        await searchapi_search.invoke_tool("web_search", {"query": "q"})


async def test_invoke_before_initialize_raises():
    impl = SearchApiSearchToolRuntimeImpl(SearchApiSearchToolConfig(api_key="test-key"))
    impl.get_request_provider_data = MagicMock(return_value=None)
    with pytest.raises(RuntimeError, match="not initialized"):
        await impl.invoke_tool("web_search", {"query": "q"})


async def test_shutdown_closes_client():
    impl = SearchApiSearchToolRuntimeImpl(SearchApiSearchToolConfig(api_key="test-key"))
    await impl.initialize()
    assert isinstance(impl._client, httpx2.AsyncClient)
    await impl.shutdown()
    assert impl._client is None
