# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from ogx.core.storage.datatypes import KVStoreReference
from ogx.providers.remote.vector_io.neo4j.config import Neo4jVectorIOConfig
from ogx.providers.remote.vector_io.neo4j.neo4j import (
    Neo4jIndex,
    Neo4jVectorIOAdapter,
    _metadata_properties,
    _sanitize_identifier,
    _translate_filters,
)
from ogx_api import ChunkMetadata, EmbeddedChunk, QueryChunksResponse, VectorStore
from ogx_api.filters import ComparisonFilter, CompoundFilter

# These tests deliberately stay database-free. They cover provider-local
# translation, scoring, configuration, and lifecycle behavior; Neo4j query
# semantics are covered in tests/integration/vector_io/test_neo4j_graph_retrieval.py.


def _make_config(**overrides: object) -> Neo4jVectorIOConfig:
    values: dict[str, object] = {
        "uri": "bolt://localhost:7687",
        "user": "neo4j",
        "password": "ogxpassword",
        "persistence": KVStoreReference(backend="kv_default", namespace="vector_io::test"),
    }
    values.update(overrides)
    return Neo4jVectorIOConfig(**values)


def _make_vector_store() -> VectorStore:
    return VectorStore(
        identifier="test-store",
        provider_id="neo4j",
        embedding_model="test-embedding",
        embedding_dimension=3,
    )


def _make_chunk(chunk_id: str, content: str = "content") -> EmbeddedChunk:
    return EmbeddedChunk(
        content=content,
        chunk_id=chunk_id,
        metadata={"document_id": chunk_id},
        chunk_metadata=ChunkMetadata(document_id=chunk_id, chunk_id=chunk_id),
        embedding=[1.0, 0.0, 0.0],
        embedding_model="test-embedding",
        embedding_dimension=3,
    )


def test_neo4j_config_sample_run_config_uses_env_templates() -> None:
    config = Neo4jVectorIOConfig.sample_run_config("~/.ogx/distributions/starter")

    assert config["uri"] == "${env.NEO4J_URI:=bolt://localhost:7687}"
    assert config["user"] == "${env.NEO4J_USER:=neo4j}"
    assert config["password"] == "${env.NEO4J_PASSWORD:=}"
    assert config["database"] == "${env.NEO4J_DATABASE:=neo4j}"
    assert config["graph_retrieval_enabled"] is False
    assert config["graph_expansion_depth"] == 1
    assert config["graph_max_neighbors"] == 10
    assert config["graph_expansion_weight"] == 0.15
    assert config["persistence"]["namespace"] == "vector_io::neo4j"


def test_neo4j_config_accepts_persistence_reference() -> None:
    config = _make_config()

    assert config.uri == "bolt://localhost:7687"
    assert config.graph_relationship_types is None


def test_neo4j_identifiers_are_collision_safe() -> None:
    assert _sanitize_identifier("foo-bar") != _sanitize_identifier("foo_bar")


def test_neo4j_metadata_rejects_nested_values() -> None:
    with pytest.raises(ValueError, match="Failed to prepare Neo4j metadata"):
        _metadata_properties({"nested": {"value": "unsupported"}})


async def test_neo4j_adapter_initialize_creates_driver_and_loads_openai_stores() -> None:
    adapter = Neo4jVectorIOAdapter(_make_config(), inference_api=MagicMock(), files_api=None)
    mock_driver = AsyncMock()
    mock_driver.verify_connectivity = AsyncMock()

    with patch("ogx.providers.remote.vector_io.neo4j.neo4j.AsyncGraphDatabase.driver", return_value=mock_driver):
        with patch("ogx.providers.remote.vector_io.neo4j.neo4j.kvstore_impl", new_callable=AsyncMock) as kvstore_impl:
            kvstore = AsyncMock()
            kvstore.values_in_range.return_value = []
            kvstore_impl.return_value = kvstore
            with patch.object(adapter, "initialize_openai_vector_stores", new_callable=AsyncMock) as init_openai:
                await adapter.initialize()

    mock_driver.verify_connectivity.assert_awaited_once()
    init_openai.assert_awaited_once()
    assert adapter.kvstore is kvstore


async def test_neo4j_adapter_shutdown_closes_driver() -> None:
    adapter = Neo4jVectorIOAdapter(_make_config(), inference_api=MagicMock(), files_api=None)
    driver = AsyncMock()
    adapter.driver = driver

    await adapter.shutdown()

    driver.close.assert_awaited_once()
    assert adapter.driver is None


async def test_neo4j_adapter_recreates_driver_after_connectivity_failure() -> None:
    adapter = Neo4jVectorIOAdapter(_make_config(), inference_api=MagicMock(), files_api=None)
    stale_driver = AsyncMock()
    stale_driver.verify_connectivity.side_effect = RuntimeError("stale connection")
    fresh_driver = AsyncMock()
    adapter.driver = stale_driver

    with patch(
        "ogx.providers.remote.vector_io.neo4j.neo4j.AsyncGraphDatabase.driver", return_value=fresh_driver
    ) as create_driver:
        result = await adapter._ensure_driver()

    assert result is fresh_driver
    assert adapter.driver is fresh_driver
    stale_driver.close.assert_awaited_once()
    fresh_driver.verify_connectivity.assert_awaited_once()
    create_driver.assert_called_once()


def test_neo4j_translate_eq_filter() -> None:
    clause, params = _translate_filters(ComparisonFilter(type="eq", key="topic", value="programming"))

    assert clause == "node.`metadata_topic` = $filter_0"
    assert params == {"filter_0": "programming"}


def test_neo4j_translate_compound_filter() -> None:
    clause, params = _translate_filters(
        CompoundFilter(
            type="and",
            filters=[
                ComparisonFilter(type="eq", key="topic", value="programming"),
                ComparisonFilter(type="gte", key="priority", value=2),
            ],
        )
    )

    assert clause == "(node.`metadata_topic` = $filter_0) AND (node.`metadata_priority` >= $filter_1)"
    assert params == {"filter_0": "programming", "filter_1": 2}


def test_neo4j_translate_rejects_unsafe_metadata_key() -> None:
    with pytest.raises(ValueError, match="Failed to translate Neo4j metadata filter"):
        _translate_filters(ComparisonFilter(type="eq", key="topic.name", value="programming"))


def test_neo4j_translate_rejects_unsupported_filter_type() -> None:
    unsupported_filter = ComparisonFilter.model_construct(type="unsupported", key="topic", value="programming")

    with pytest.raises(ValueError, match="Failed to translate Neo4j metadata filter"):
        _translate_filters(unsupported_filter)


async def test_neo4j_graph_expansion_merges_related_chunks() -> None:
    index = Neo4jIndex(AsyncMock(), _make_config(graph_retrieval_enabled=True), _make_vector_store())
    index._query_graph_neighbors = AsyncMock(
        return_value=[
            ("chunk-ml", _make_chunk("chunk-ml", "Machine learning systems learn from data."), ["chunk-python"]),
        ]
    )
    response = QueryChunksResponse(
        chunks=[_make_chunk("chunk-python", "Python is readable and popular for data work.")],
        scores=[0.9],
    )

    expanded = await index.expand_graph(
        response,
        k=2,
        params={"graph_retrieval_enabled": True, "graph_expansion_weight": 0.2},
    )

    assert [chunk.chunk_id for chunk in expanded.chunks] == ["chunk-python", "chunk-ml"]
    assert expanded.scores == [0.9, pytest.approx(0.18)]


def test_neo4j_relationship_pattern_quotes_relationship_types() -> None:
    index = Neo4jIndex(AsyncMock(), _make_config(), _make_vector_store())

    assert index._relationship_pattern(2, ["MENTIONS", "OWNS"]) == "-[:`MENTIONS`|`OWNS`*1..2]-"


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"graph_max_neighbors": 0}, "graph_max_neighbors"),
        ({"graph_max_neighbors": 1001}, "graph_max_neighbors"),
        ({"graph_expansion_depth": 4}, "graph_expansion_depth"),
        ({"graph_expansion_weight": 1.1}, "graph_expansion_weight"),
        ({"graph_relationship_types": "MENTIONS"}, "graph_relationship_types"),
    ],
)
async def test_neo4j_graph_expansion_validates_request_params(params: dict[str, object], message: str) -> None:
    index = Neo4jIndex(AsyncMock(), _make_config(graph_retrieval_enabled=True), _make_vector_store())
    response = QueryChunksResponse(chunks=[_make_chunk("chunk-python")], scores=[0.9])

    with pytest.raises(ValueError, match=f"Failed to expand Neo4j graph: {message}"):
        await index.expand_graph(response, k=2, params={"graph_retrieval_enabled": True, **params})


@pytest.mark.parametrize("query_method", ["query_vector", "query_keyword"])
async def test_neo4j_filtered_queries_overfetch_candidates(query_method: str) -> None:
    session = MagicMock()
    result = AsyncMock()
    result.data.return_value = []
    session.run = AsyncMock(return_value=result)
    driver = MagicMock()
    driver.session.return_value.__aenter__ = AsyncMock(return_value=session)
    driver.session.return_value.__aexit__ = AsyncMock(return_value=None)
    index = Neo4jIndex(driver, _make_config(), _make_vector_store())
    filters = ComparisonFilter(type="eq", key="topic", value="programming")

    if query_method == "query_vector":
        await index.query_vector(np.array([1.0, 0.0, 0.0]), 2, 0.0, filters)
    else:
        await index.query_keyword("programming", 2, 0.0, filters)

    assert session.run.call_args.kwargs["candidate_limit"] > 2


@pytest.mark.parametrize("query_method", ["query_vector", "query_keyword"])
async def test_neo4j_unfiltered_queries_request_exact_k(query_method: str) -> None:
    session = MagicMock()
    result = AsyncMock()
    result.data.return_value = []
    session.run = AsyncMock(return_value=result)
    driver = MagicMock()
    driver.session.return_value.__aenter__ = AsyncMock(return_value=session)
    driver.session.return_value.__aexit__ = AsyncMock(return_value=None)
    index = Neo4jIndex(driver, _make_config(), _make_vector_store())

    if query_method == "query_vector":
        await index.query_vector(np.array([1.0, 0.0, 0.0]), 2, 0.0)
    else:
        await index.query_keyword("programming", 2, 0.0)

    assert session.run.call_args.kwargs["candidate_limit"] == 2
