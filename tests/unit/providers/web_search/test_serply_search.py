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

from ogx.providers.remote.web_search.serply_search.config import SerplySearchConfig
from ogx.providers.remote.web_search.serply_search.serply_search import SerplySearchWebSearchImpl


@pytest.fixture
def serply_search():
    impl = SerplySearchWebSearchImpl(SerplySearchConfig(api_key="test-key", max_results=3))
    impl._client = MagicMock(spec=httpx2.AsyncClient)
    impl.get_request_provider_data = MagicMock(return_value=None)
    return impl


def _serply_response(count: int = 1) -> httpx2.Response:
    # Shape of GET /v1/search: each organic result carries title, link, and description.
    return httpx2.Response(
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
        request=httpx2.Request("GET", SerplySearchWebSearchImpl._SEARCH_URL),
    )


@pytest.fixture
def mock_serply_response():
    return _serply_response()


async def test_search_without_extra_params_sends_config_defaults(serply_search, mock_serply_response):
    serply_search._client.get = AsyncMock(return_value=mock_serply_response)
    await serply_search.search(None, "test query")
    params = serply_search._client.get.call_args.kwargs["params"]
    assert params == {"q": "test query", "num": 3}


async def test_search_with_allowed_domains(serply_search, mock_serply_response):
    serply_search._client.get = AsyncMock(return_value=mock_serply_response)
    await serply_search.search(
        None,
        "test query",
        allowed_domains=["example.com", "docs.example.com"],
    )
    params = serply_search._client.get.call_args.kwargs["params"]
    assert params["q"] == "test query (site:example.com OR site:docs.example.com)"


async def test_search_with_empty_allowed_domains(serply_search, mock_serply_response):
    serply_search._client.get = AsyncMock(return_value=mock_serply_response)
    await serply_search.search(None, "test query", allowed_domains=[])
    params = serply_search._client.get.call_args.kwargs["params"]
    assert params["q"] == "test query"


@pytest.mark.parametrize("size,expected", [("low", 3), ("medium", 5), ("high", 10)])
async def test_search_context_size_mapping(serply_search, mock_serply_response, size, expected):
    serply_search._client.get = AsyncMock(return_value=mock_serply_response)
    await serply_search.search(None, "q", search_context_size=size)
    params = serply_search._client.get.call_args.kwargs["params"]
    assert params["num"] == expected


async def test_unknown_search_context_size_keeps_config_default(serply_search, mock_serply_response):
    serply_search._client.get = AsyncMock(return_value=mock_serply_response)
    await serply_search.search(None, "q", search_context_size="ultra")
    params = serply_search._client.get.call_args.kwargs["params"]
    assert params["num"] == 3


async def test_max_results_is_clamped_to_api_ceiling(mock_serply_response):
    with pytest.warns(UserWarning, match="exceeds the Google engine ceiling"):
        config = SerplySearchConfig(api_key="test-key", max_results=50)
    assert config.max_results == 10
    impl = SerplySearchWebSearchImpl(config)
    impl._client = MagicMock(spec=httpx2.AsyncClient)
    impl._client.get = AsyncMock(return_value=mock_serply_response)
    impl.get_request_provider_data = MagicMock(return_value=None)
    await impl.search(None, "q")
    assert impl._client.get.call_args.kwargs["params"]["num"] == 10


async def test_results_are_trimmed_to_the_limit(serply_search):
    # `num` is an upper bound on the API side, so extra rows are dropped client-side.
    serply_search._client.get = AsyncMock(return_value=_serply_response(count=8))
    result = await serply_search.search(None, "q")
    assert len(json.loads(result.content)["results"]) == 3
    assert len(result.sources) == 3


async def test_results_map_link_and_description(serply_search, mock_serply_response):
    serply_search._client.get = AsyncMock(return_value=mock_serply_response)
    result = await serply_search.search(None, "test query")
    payload = json.loads(result.content)
    assert payload["query"] == "test query"
    assert payload["results"] == [
        {
            "title": "Test Result 0",
            "url": "https://example.com/0",
            "content": "A test result description 0",
        }
    ]


async def test_search_returns_sources(serply_search, mock_serply_response):
    serply_search._client.get = AsyncMock(return_value=mock_serply_response)
    result = await serply_search.search(None, "test query")
    assert result.query == "test query"
    assert [s.url for s in result.sources] == ["https://example.com/0"]


async def test_user_location_country_is_forwarded_as_gl(serply_search, mock_serply_response):
    serply_search._client.get = AsyncMock(return_value=mock_serply_response)
    await serply_search.search(
        None,
        "q",
        user_location={"country": "GB", "city": "London"},
    )
    params = serply_search._client.get.call_args.kwargs["params"]
    assert params["gl"] == "gb"
    assert "user_location" not in params


async def test_api_key_sent_as_header_not_params(serply_search, mock_serply_response):
    serply_search._client.get = AsyncMock(return_value=mock_serply_response)
    await serply_search.search(None, "test query")
    call = serply_search._client.get.call_args
    assert call.kwargs["headers"]["X-Api-Key"] == "test-key"
    assert call.kwargs["headers"]["User-Agent"] == "ogx"
    assert "test-key" not in json.dumps(call.kwargs["params"])


async def test_missing_api_key_sends_no_auth_header(mock_serply_response):
    impl = SerplySearchWebSearchImpl(SerplySearchConfig(api_key=None))
    impl._client = MagicMock(spec=httpx2.AsyncClient)
    impl._client.get = AsyncMock(return_value=mock_serply_response)
    impl.get_request_provider_data = MagicMock(return_value=None)
    await impl.search(None, "q")
    assert "X-Api-Key" not in impl._client.get.call_args.kwargs["headers"]


async def test_provider_data_overrides_config_api_key(serply_search, mock_serply_response):
    serply_search._client.get = AsyncMock(return_value=mock_serply_response)
    provider_data = MagicMock()
    provider_data.serply_search_api_key = SecretStr("override-key")
    serply_search.get_request_provider_data = MagicMock(return_value=provider_data)
    await serply_search.search(None, "q")
    assert serply_search._client.get.call_args.kwargs["headers"]["X-Api-Key"] == "override-key"


async def test_401_raises_with_the_reason(serply_search):
    unauthorized = httpx2.Response(
        401,
        json={"detail": "Invalid API key"},
        request=httpx2.Request("GET", SerplySearchWebSearchImpl._SEARCH_URL),
    )
    serply_search._client.get = AsyncMock(return_value=unauthorized)
    with pytest.raises(RuntimeError, match="Serply Search"):
        await serply_search.search(None, "q")


async def test_other_http_errors_raise(serply_search):
    server_error = httpx2.Response(
        500,
        request=httpx2.Request("GET", SerplySearchWebSearchImpl._SEARCH_URL),
    )
    serply_search._client.get = AsyncMock(return_value=server_error)
    with pytest.raises(httpx2.HTTPStatusError):
        await serply_search.search(None, "q")
