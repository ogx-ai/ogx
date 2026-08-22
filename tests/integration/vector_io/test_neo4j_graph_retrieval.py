# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
import uuid
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from neo4j import AsyncGraphDatabase

from ogx.core.storage.datatypes import KVStoreReference
from ogx.providers.remote.vector_io.neo4j.config import Neo4jVectorIOConfig
from ogx.providers.remote.vector_io.neo4j.neo4j import Neo4jIndex, Neo4jVectorIOAdapter
from ogx_api import ChunkMetadata, EmbeddedChunk, InsertChunksRequest, QueryChunksRequest, VectorStore

pytestmark = pytest.mark.skipif(
    not os.environ.get("NEO4J_URI"),
    reason="Neo4j graph retrieval tests require NEO4J_URI",
)


def _chunk(chunk_id: str, content: str, embedding: list[float], **metadata: object) -> EmbeddedChunk:
    return EmbeddedChunk(
        content=content,
        chunk_id=chunk_id,
        metadata={"document_id": chunk_id, **metadata},
        chunk_metadata=ChunkMetadata(document_id=chunk_id, chunk_id=chunk_id),
        embedding=embedding,
        embedding_model="test-embedding",
        embedding_dimension=len(embedding),
    )


async def test_neo4j_graphrag_adds_related_chunk_beyond_standard_retrieval() -> None:
    password = os.environ.get("NEO4J_PASSWORD", "")
    driver = AsyncGraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ.get("NEO4J_USER", "neo4j"), password),
    )
    vector_store = VectorStore(
        identifier=f"neo4j-graph-test-{uuid.uuid4()}",
        provider_id="neo4j",
        embedding_model="test-embedding",
        embedding_dimension=3,
    )
    config = Neo4jVectorIOConfig(
        uri=os.environ["NEO4J_URI"],
        user=os.environ.get("NEO4J_USER", "neo4j"),
        password=password,
        database=os.environ.get("NEO4J_DATABASE", "neo4j"),
        graph_retrieval_enabled=True,
        persistence=KVStoreReference(backend="kv_default", namespace="vector_io::neo4j-test"),
    )
    adapter = Neo4jVectorIOAdapter(config, inference_api=MagicMock())
    adapter.driver = driver
    adapter.kvstore = AsyncMock()

    try:
        await driver.verify_connectivity()
        await adapter.register_vector_store(vector_store)
        index = adapter.cache[vector_store.identifier].index
        assert isinstance(index, Neo4jIndex)
        await adapter.insert_chunks(
            InsertChunksRequest(
                vector_store_id=vector_store.identifier,
                chunks=[
                    _chunk(
                        "chunk-python",
                        "Python is readable and popular for data work.",
                        [1.0, 0.0, 0.0],
                        source="original",
                    ),
                    _chunk("chunk-ml", "Machine learning systems learn from data.", [0.0, 1.0, 0.0]),
                    _chunk("chunk-java", "Java is common for enterprise services.", [0.0, 0.0, 1.0]),
                ],
            )
        )
        await adapter.insert_chunks(
            InsertChunksRequest(
                vector_store_id=vector_store.identifier,
                chunks=[
                    _chunk(
                        "chunk-python",
                        "Python is readable and popular for data work, revised.",
                        [1.0, 0.0, 0.0],
                        source="revised",
                    )
                ],
            )
        )
        async with driver.session(database=config.database) as session:
            count_result = await session.run(
                f"MATCH (node:`{index.chunk_label}` {{chunk_id: 'chunk-python'}}) RETURN count(node) AS count"
            )
            count_record = await count_result.single()
            assert count_record is not None
            assert count_record["count"] == 1
        async with driver.session(database=config.database) as session:
            await session.run(
                f"""
                MERGE (python:Entity {{name: 'python'}})
                MERGE (java:Entity {{name: 'java'}})
                WITH python, java
                MATCH (py:`{index.chunk_label}` {{chunk_id: 'chunk-python'}})
                MATCH (ml:`{index.chunk_label}` {{chunk_id: 'chunk-ml'}})
                MATCH (ja:`{index.chunk_label}` {{chunk_id: 'chunk-java'}})
                MERGE (py)-[:MENTIONS]->(python)
                MERGE (ml)-[:MENTIONS]->(python)
                MERGE (ja)-[:MENTIONS]->(java)
                """
            )

        vector_response = await index.query_vector(np.array([1.0, 0.0, 0.0], dtype=np.float32), 1, 0.0)
        assert [chunk.chunk_id for chunk in vector_response.chunks] == ["chunk-python"]
        revised_response = await index.query_vector(
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            1,
            0.0,
            {"type": "eq", "key": "source", "value": "revised"},
        )
        stale_response = await index.query_vector(
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            1,
            0.0,
            {"type": "eq", "key": "source", "value": "original"},
        )
        assert [chunk.chunk_id for chunk in revised_response.chunks] == ["chunk-python"]
        assert stale_response.chunks == []

        standard_response = await adapter.query_chunks(
            QueryChunksRequest(
                vector_store_id=vector_store.identifier,
                query="Python",
                params={
                    "mode": "keyword",
                    "max_chunks": 2,
                    "score_threshold": 0.0,
                    "graph_retrieval_enabled": False,
                },
            )
        )
        response = await adapter.query_chunks(
            QueryChunksRequest(
                vector_store_id=vector_store.identifier,
                query="Python",
                params={
                    "mode": "keyword",
                    "max_chunks": 2,
                    "score_threshold": 0.0,
                    "graph_retrieval_enabled": True,
                    "graph_expansion_depth": 2,
                    "graph_expansion_weight": 0.2,
                    "graph_max_neighbors": 2,
                },
            )
        )

        assert [chunk.chunk_id for chunk in standard_response.chunks] == ["chunk-python"]
        assert [chunk.chunk_id for chunk in response.chunks] == ["chunk-python", "chunk-ml"]
        assert "Machine learning systems learn from data." in response.chunks[1].content
    finally:
        await adapter.unregister_vector_store(vector_store.identifier)
        await driver.close()
