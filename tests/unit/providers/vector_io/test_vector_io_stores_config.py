# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from psycopg2 import sql

from ogx_api import (
    OpenAICreateVectorStoreRequestWithExtraBody,
    QueryChunksResponse,
    VectorStore,
)

# This test is a unit test for the inline VectorIO providers. This should only contain
# tests which are specific to this class. More general (API-level) tests should be placed in
# tests/integration/vector_io/
#
# How to run this test:
#
# pytest tests/unit/providers/vector_io/test_vector_io_openai_vector_stores.py \
# -v -s --tb=short --disable-warnings --asyncio-mode=auto


@pytest.fixture(autouse=True)
def mock_resume_file_batches(request):
    """Mock the resume functionality to prevent stale file batches from being processed during tests."""
    with patch(
        "ogx.providers.utils.memory.openai_vector_store_mixin.OpenAIVectorStoreMixin._resume_incomplete_batches",
        new_callable=AsyncMock,
    ):
        yield


async def test_embedding_config_from_metadata(vector_io_adapter):
    """Test that embedding configuration is correctly extracted from metadata."""

    # Set provider_id attribute for the adapter
    vector_io_adapter.__provider_id__ = "test_provider"

    # Test with embedding config in metadata
    params = OpenAICreateVectorStoreRequestWithExtraBody(
        name="test_store",
        metadata={
            "embedding_model": "test-embedding-model",
            "embedding_dimension": "512",
        },
        model_extra={},
    )

    result = await vector_io_adapter.openai_create_vector_store(params)

    # Verify the saved metadata contains the correct embedding config
    vector_store = vector_io_adapter.openai_vector_stores[result.id]
    assert vector_store["metadata"]["embedding_model"] == "test-embedding-model"
    assert vector_store["metadata"]["embedding_dimension"] == "512"


async def test_embedding_config_from_extra_body(vector_io_adapter):
    """Test that embedding configuration is correctly extracted from extra_body when metadata is empty."""

    # Set provider_id attribute for the adapter
    vector_io_adapter.__provider_id__ = "test_provider"

    # Test with embedding config in extra_body only (metadata has no embedding_model)
    params = OpenAICreateVectorStoreRequestWithExtraBody(
        name="test_store",
        metadata={},  # Empty metadata to ensure extra_body is used
        **{
            "embedding_model": "extra-body-model",
            "embedding_dimension": 1024,
        },
    )

    result = await vector_io_adapter.openai_create_vector_store(params)

    # Verify the saved metadata contains the correct embedding config
    vector_store = vector_io_adapter.openai_vector_stores[result.id]
    assert vector_store["metadata"]["embedding_model"] == "extra-body-model"
    assert vector_store["metadata"]["embedding_dimension"] == "1024"


async def test_embedding_config_consistency_check_passes(vector_io_adapter):
    """Test that consistent embedding config in both metadata and extra_body passes validation."""

    # Set provider_id attribute for the adapter
    vector_io_adapter.__provider_id__ = "test_provider"

    # Test with consistent embedding config in both metadata and extra_body
    params = OpenAICreateVectorStoreRequestWithExtraBody(
        name="test_store",
        metadata={
            "embedding_model": "consistent-model",
            "embedding_dimension": "768",
        },
        **{
            "embedding_model": "consistent-model",
            "embedding_dimension": 768,
        },
    )

    result = await vector_io_adapter.openai_create_vector_store(params)

    # Should not raise any error and use metadata config
    vector_store = vector_io_adapter.openai_vector_stores[result.id]
    assert vector_store["metadata"]["embedding_model"] == "consistent-model"
    assert vector_store["metadata"]["embedding_dimension"] == "768"


async def test_embedding_config_dimension_required(vector_io_adapter):
    """Test that embedding dimension is required when not provided."""

    # Set provider_id attribute for the adapter
    vector_io_adapter.__provider_id__ = "test_provider"

    # Test with only embedding model, no dimension (metadata empty to use extra_body)
    params = OpenAICreateVectorStoreRequestWithExtraBody(
        name="test_store",
        metadata={},  # Empty metadata to ensure extra_body is used
        **{
            "embedding_model": "model-without-dimension",
        },
    )

    # Should raise ValueError because embedding_dimension is not provided
    with pytest.raises(ValueError, match="Embedding dimension is required"):
        await vector_io_adapter.openai_create_vector_store(params)


async def test_embedding_config_required_model_missing(vector_io_adapter):
    """Test that missing embedding model raises error."""

    # Set provider_id attribute for the adapter
    vector_io_adapter.__provider_id__ = "test_provider"
    # Mock the default model lookup to return None (no default model available)
    vector_io_adapter._get_default_embedding_model_and_dimension = AsyncMock(return_value=None)

    # Test with no embedding model provided
    params = OpenAICreateVectorStoreRequestWithExtraBody(name="test_store", metadata={})

    with pytest.raises(ValueError, match="embedding_model is required"):
        await vector_io_adapter.openai_create_vector_store(params)


async def test_search_vector_store_ignores_rewrite_query(vector_io_adapter):
    """Test that the mixin ignores rewrite_query parameter since rewriting is done at router level."""

    # Create an OpenAI vector store for testing directly in the adapter's cache
    vector_store_id = "test_store_rewrite"
    openai_vector_store = {
        "id": vector_store_id,
        "name": "Test Store",
        "description": "A test OpenAI vector store",
        "vector_store_id": "test_db",
        "embedding_model": "test/embedding",
    }
    vector_io_adapter.openai_vector_stores[vector_store_id] = openai_vector_store

    # Mock query_chunks response from adapter
    mock_response = QueryChunksResponse(chunks=[], scores=[])

    async def mock_query_chunks(*args, **kwargs):
        return mock_response

    vector_io_adapter.query_chunks = mock_query_chunks

    # Test that rewrite_query=True doesn't cause an error (it's ignored at mixin level)
    # The mixin should process the search request without attempting to rewrite the query
    from ogx_api import OpenAISearchVectorStoreRequest

    request = OpenAISearchVectorStoreRequest(
        query="test query",
        max_num_results=5,
        rewrite_query=True,  # This should be ignored at mixin level
    )
    result = await vector_io_adapter.openai_search_vector_store(
        vector_store_id=vector_store_id,
        request=request,
    )

    # Search should succeed - the mixin ignores rewrite_query and just does the search
    assert result is not None
    assert result.search_query == ["test query"]  # Original query preserved


async def test_create_gin_index_executes_correct_sql():
    from unittest.mock import MagicMock

    from ogx.providers.remote.vector_io.pgvector.config import PGVectorHNSWVectorIndex
    from ogx.providers.remote.vector_io.pgvector.pgvector import PGVectorIndex

    connection = MagicMock()
    cursor = MagicMock()
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock()
    connection.cursor.return_value = cursor

    vector_store = VectorStore(
        identifier="test-vector-db",
        embedding_model="test-model",
        embedding_dimension=768,
        provider_id="pgvector",
    )

    with patch("ogx.providers.remote.vector_io.pgvector.pgvector.psycopg2"):
        index = PGVectorIndex(
            vector_store=vector_store,
            dimension=768,
            conn=connection,
            distance_metric="COSINE",
            vector_index=PGVectorHNSWVectorIndex(m=16, ef_construction=64),
        )
        index.table_name = "vs_test_table"
        index._table_sql = sql.Identifier("vs_test_table")

    await index.create_gin_index(cursor)

    cursor.execute.assert_called_once()
    executed_sql = repr(cursor.execute.call_args[0][0])
    assert "CREATE INDEX IF NOT EXISTS" in executed_sql
    assert "vs_test_table_content_gin_idx" in executed_sql
    assert "vs_test_table" in executed_sql
    assert "USING GIN(tokenized_content)" in executed_sql


async def test_create_gin_index_raises_runtime_error_on_db_error():
    from unittest.mock import MagicMock

    import psycopg2

    from ogx.providers.remote.vector_io.pgvector.config import PGVectorHNSWVectorIndex
    from ogx.providers.remote.vector_io.pgvector.pgvector import PGVectorIndex

    connection = MagicMock()
    cursor = MagicMock()
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock()
    cursor.execute.side_effect = psycopg2.Error("mock database error")
    connection.cursor.return_value = cursor

    vector_store = VectorStore(
        identifier="test-vector-db",
        embedding_model="test-model",
        embedding_dimension=768,
        provider_id="pgvector",
    )

    with patch("ogx.providers.remote.vector_io.pgvector.pgvector.psycopg2"):
        index = PGVectorIndex(
            vector_store=vector_store,
            dimension=768,
            conn=connection,
            distance_metric="COSINE",
            vector_index=PGVectorHNSWVectorIndex(m=16, ef_construction=64),
        )
        index.table_name = "vs_test_table"
        index._table_sql = sql.Identifier("vs_test_table")

    with pytest.raises(RuntimeError, match="Failed to create GIN index"):
        await index.create_gin_index(cursor)


async def test_gin_index_creation_in_initialize_call():
    from unittest.mock import MagicMock

    from ogx.providers.remote.vector_io.pgvector.config import PGVectorHNSWVectorIndex
    from ogx.providers.remote.vector_io.pgvector.pgvector import PGVectorIndex

    connection = MagicMock()
    cursor = MagicMock()
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock()
    connection.cursor.return_value = cursor

    vector_store = VectorStore(
        identifier="test-vector-db",
        embedding_model="test-model",
        embedding_dimension=768,
        provider_id="pgvector",
    )

    with patch("ogx.providers.remote.vector_io.pgvector.pgvector.psycopg2") as mock_psycopg2:
        mock_psycopg2.extras.DictCursor = MagicMock()

        index = PGVectorIndex(
            vector_store=vector_store,
            dimension=768,
            conn=connection,
            distance_metric="COSINE",
            vector_index=PGVectorHNSWVectorIndex(m=16, ef_construction=64),
        )

        with patch.object(index, "create_gin_index") as mock_gin:
            await index.initialize()
            mock_gin.assert_called_once()


async def test_set_ef_search_called_before_select_in_query_vector(mock_psycopg2_connection, embedding_dimension):
    from ogx.providers.remote.vector_io.pgvector.config import PGVectorHNSWVectorIndex
    from ogx.providers.remote.vector_io.pgvector.pgvector import PGVectorIndex

    connection, cursor = mock_psycopg2_connection
    cursor.fetchall.return_value = []

    index = PGVectorIndex(
        vector_store=VectorStore(
            identifier="test-vector-db",
            embedding_model="test-model",
            embedding_dimension=embedding_dimension,
            provider_id="pgvector",
        ),
        dimension=embedding_dimension,
        conn=connection,
        distance_metric="COSINE",
        vector_index=PGVectorHNSWVectorIndex(m=16, ef_construction=64, ef_search=50),
    )
    index.table_name = "test_table"
    index._table_sql = sql.Identifier("test_table")

    embedding = np.random.rand(embedding_dimension).astype(np.float32)
    await index.query_vector(embedding, k=5, score_threshold=0.5)

    calls = cursor.execute.call_args_list
    assert len(calls) == 2, f"Expected exactly 2 execute calls (SET + SELECT), got {len(calls)}"

    set_call_sql = str(calls[0])
    select_call_sql = repr(calls[1][0][0])
    assert f"SET hnsw.ef_search = {index.vector_index.ef_search}" in set_call_sql, (
        f"First call should be SET, got: {set_call_sql}"
    )
    assert "SELECT document" in select_call_sql, f"Second call should be SELECT, got: {select_call_sql}"


async def test_apply_default_ef_search_for_query_vector(mock_psycopg2_connection, embedding_dimension):
    from ogx.providers.remote.vector_io.pgvector.config import PGVectorHNSWVectorIndex
    from ogx.providers.remote.vector_io.pgvector.pgvector import PGVectorIndex

    connection, cursor = mock_psycopg2_connection
    cursor.fetchall.return_value = []

    index = PGVectorIndex(
        vector_store=VectorStore(
            identifier="test-vector-db",
            embedding_model="test-model",
            embedding_dimension=embedding_dimension,
            provider_id="pgvector",
        ),
        dimension=embedding_dimension,
        conn=connection,
        distance_metric="COSINE",
        vector_index=PGVectorHNSWVectorIndex(m=16, ef_construction=64),
    )
    index.table_name = "test_table"
    index._table_sql = sql.Identifier("test_table")

    embedding = np.random.rand(embedding_dimension).astype(np.float32)
    await index.query_vector(embedding, k=5, score_threshold=0.5)

    calls = cursor.execute.call_args_list
    set_call_sql = str(calls[0])
    assert f"SET hnsw.ef_search = {PGVectorHNSWVectorIndex().ef_search}" in set_call_sql, (
        f"Expected default 'SET hnsw.ef_search = {PGVectorHNSWVectorIndex().ef_search}' when ef_search is not explicitly configured, got: {set_call_sql}"
    )
