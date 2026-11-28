# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""The keyword-search fallback must actually search, and must not widen the search.

When BM25 search fails, MilvusIndex.query_keyword falls back to a plain text
query. That fallback has to match the query text, keep the caller's filters,
and a filter that cannot be translated has to raise instead of being dropped.
"""

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

if "pymilvus" not in sys.modules:
    pymilvus = ModuleType("pymilvus")
    pymilvus.AnnSearchRequest = object
    pymilvus.DataType = SimpleNamespace(
        VARCHAR="VARCHAR",
        FLOAT_VECTOR="FLOAT_VECTOR",
        JSON="JSON",
        SPARSE_FLOAT_VECTOR="SPARSE_FLOAT_VECTOR",
    )
    pymilvus.Function = object
    pymilvus.FunctionType = SimpleNamespace(BM25="BM25")
    pymilvus.AsyncMilvusClient = object
    pymilvus.RRFRanker = object
    pymilvus.WeightedRanker = object
    sys.modules["pymilvus"] = pymilvus

from ogx.providers.remote.vector_io.milvus.milvus import MilvusIndex
from ogx_api import ComparisonFilter


class _BM25UnavailableClient:
    def __init__(self) -> None:
        self.query_kwargs: dict[str, Any] | None = None

    async def search(self, **kwargs: Any) -> list[list[Any]]:
        raise RuntimeError("BM25 search is not available")

    async def query(self, **kwargs: Any) -> list[Any]:
        self.query_kwargs = kwargs
        return []


def _index(client: Any) -> MilvusIndex:
    index = MilvusIndex.__new__(MilvusIndex)
    index.client = client
    index.collection_name = "test_collection"
    return index


async def test_fallback_keeps_the_callers_filter():
    client = _BM25UnavailableClient()
    index = _index(client)
    team_filter = ComparisonFilter(key="team", type="eq", value="red")

    await index.query_keyword("fox", k=5, score_threshold=0.0, filters=team_filter)

    assert client.query_kwargs is not None
    assert index._translate_filters(team_filter) in client.query_kwargs["filter"]


async def test_filter_that_cannot_be_translated_raises():
    client = _BM25UnavailableClient()
    index = _index(client)

    with pytest.raises(ValueError, match="Unknown filter type"):
        await index.query_keyword("fox", k=5, score_threshold=0.0, filters=object())  # type: ignore[arg-type]

    assert client.query_kwargs is None


async def test_fallback_without_filters_matches_the_text():
    # Milvus only accepts a string literal after LIKE: a placeholder inside the quotes is
    # never substituted, so the query has to be written into the literal itself.
    client = _BM25UnavailableClient()
    index = _index(client)

    await index.query_keyword("fox", k=5, score_threshold=0.0)

    assert client.query_kwargs is not None
    assert client.query_kwargs["filter"] == 'content like "%fox%"'
    assert "filter_params" not in client.query_kwargs


async def test_fallback_escapes_quotes_and_backslashes():
    client = _BM25UnavailableClient()
    index = _index(client)

    await index.query_keyword('say "hi" \\ now', k=5, score_threshold=0.0)

    assert client.query_kwargs is not None
    assert client.query_kwargs["filter"] == 'content like "%say \\"hi\\" \\\\ now%"'
