# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

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
    pymilvus.MilvusClient = object
    pymilvus.RRFRanker = object
    pymilvus.WeightedRanker = object
    sys.modules["pymilvus"] = pymilvus

from ogx.providers.remote.vector_io.milvus import milvus as milvus_module
from ogx.providers.remote.vector_io.milvus.milvus import MilvusIndex


class _FakeEmbedding:
    def tolist(self) -> list[float]:
        return [0.1, 0.2, 0.3]


class _FakeClient:
    def __init__(self) -> None:
        self.hybrid_search_kwargs: dict[str, Any] | None = None

    def hybrid_search(self, **kwargs: Any) -> list[list[Any]]:
        self.hybrid_search_kwargs = kwargs
        return [[]]


class _FakeAnnSearchRequest:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


async def test_milvus_native_weighted_hybrid_uses_ranking_option_weights(monkeypatch):
    ranker_weights: list[tuple[float, float]] = []

    class _FakeWeightedRanker:
        def __init__(self, vector_weight: float, keyword_weight: float) -> None:
            ranker_weights.append((vector_weight, keyword_weight))

    monkeypatch.setattr(milvus_module, "AnnSearchRequest", _FakeAnnSearchRequest)
    monkeypatch.setattr(milvus_module, "WeightedRanker", _FakeWeightedRanker)

    client = _FakeClient()
    index = MilvusIndex(
        client=client,  # type: ignore[arg-type]
        vector_store=SimpleNamespace(identifier="test-store", embedding_dimension=3),  # type: ignore[arg-type]
        use_native_hybrid=True,
    )

    await index._query_hybrid_native(
        embedding=_FakeEmbedding(),  # type: ignore[arg-type]
        query_string="test query",
        k=5,
        score_threshold=0.0,
        reranker_type="weighted",
        reranker_params={"alpha": 0.5, "weights": {"vector": 0.2, "keyword": 0.8}},
    )

    assert ranker_weights == [(0.2, 0.8)]
    assert client.hybrid_search_kwargs is not None
    assert isinstance(client.hybrid_search_kwargs["ranker"], _FakeWeightedRanker)


async def test_milvus_native_rrf_with_weights_uses_in_memory_hybrid(monkeypatch):
    index = MilvusIndex(
        client=_FakeClient(),  # type: ignore[arg-type]
        vector_store=SimpleNamespace(identifier="test-store", embedding_dimension=3),  # type: ignore[arg-type]
        use_native_hybrid=True,
    )
    calls: list[str] = []

    async def fake_native(*args: Any, **kwargs: Any) -> str:
        calls.append("native")
        return "native"

    async def fake_in_memory(*args: Any, **kwargs: Any) -> str:
        calls.append("in_memory")
        return "in_memory"

    monkeypatch.setattr(index, "_query_hybrid_native", fake_native)
    monkeypatch.setattr(index, "_query_hybrid_in_memory", fake_in_memory)

    result = await index.query_hybrid(
        embedding=_FakeEmbedding(),  # type: ignore[arg-type]
        query_string="test query",
        k=5,
        score_threshold=0.0,
        reranker_type="rrf",
        reranker_params={"weights": {"vector": 1.0, "keyword": 0.0}},
    )

    assert result == "in_memory"
    assert calls == ["in_memory"]
