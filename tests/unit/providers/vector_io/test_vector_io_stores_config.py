# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import asyncpg
import numpy as np
import pytest

from ogx_api import (
    OpenAICreateVectorStoreRequestWithExtraBody,
    QueryChunksResponse,
    VectorStore,
)


def _make_mock_asyncpg_pool():
    """Create a mock asyncpg pool with acquire() as async context manager."""
    pool = MagicMock()
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock()
    mock_conn.executemany = AsyncMock()
    mock_conn.fetch = AsyncMock(return_value=[])
    mock_conn.fetchrow = AsyncMock(return_value=None)
    mock_conn.fetchval = AsyncMock(return_value=None)

    acm = AsyncMock()
    acm.__aenter__ = AsyncMock(return_value=mock_conn)
    acm.__aexit__ = AsyncMock(return_value=False)
    pool.acquire = MagicMock(return_value=acm)
    pool.close = AsyncMock()

    return pool, mock_conn


# This test is a unit test for the inline VectorIO providers. This should only contain
# tests which are specific to this class. More general (API-level) tests should be placed in
# tests/integration/vector_io/
#
# How to run this test:
#
# pytest tests/unit/providers/vector_io/test_vector_io_stores_config.py \
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
    from ogx.providers.remote.vector_io.pgvector.config import PGVectorHNSWVectorIndex
    from ogx.providers.remote.vector_io.pgvector.pgvector import PGVectorIndex

    pool, mock_conn = _make_mock_asyncpg_pool()

    vector_store = VectorStore(
        identifier="test-vector-db",
        embedding_model="test-model",
        embedding_dimension=768,
        provider_id="pgvector",
    )

    index = PGVectorIndex(
        vector_store=vector_store,
        dimension=768,
        pool=pool,
        distance_metric="COSINE",
        vector_index=PGVectorHNSWVectorIndex(m=16, ef_construction=64),
    )
    index.table_name = "vs_test_table"
    index._quoted_table = '"vs_test_table"'

    await index.create_gin_index(mock_conn)

    mock_conn.execute.assert_called_once()
    executed_sql = mock_conn.execute.call_args[0][0]
    assert "CREATE INDEX IF NOT EXISTS" in executed_sql
    assert "vs_test_table_content_gin_idx" in executed_sql
    assert "vs_test_table" in executed_sql
    assert "USING GIN(tokenized_content)" in executed_sql


async def test_create_gin_index_raises_runtime_error_on_db_error():
    from ogx.providers.remote.vector_io.pgvector.config import PGVectorHNSWVectorIndex
    from ogx.providers.remote.vector_io.pgvector.pgvector import PGVectorIndex

    pool, mock_conn = _make_mock_asyncpg_pool()
    mock_conn.execute = AsyncMock(side_effect=asyncpg.PostgresError("mock database error"))

    vector_store = VectorStore(
        identifier="test-vector-db",
        embedding_model="test-model",
        embedding_dimension=768,
        provider_id="pgvector",
    )

    index = PGVectorIndex(
        vector_store=vector_store,
        dimension=768,
        pool=pool,
        distance_metric="COSINE",
        vector_index=PGVectorHNSWVectorIndex(m=16, ef_construction=64),
    )
    index.table_name = "vs_test_table"
    index._quoted_table = '"vs_test_table"'

    with pytest.raises(RuntimeError, match="Failed to create GIN index"):
        await index.create_gin_index(mock_conn)


async def test_gin_index_creation_in_initialize_call():
    from ogx.providers.remote.vector_io.pgvector.config import PGVectorHNSWVectorIndex
    from ogx.providers.remote.vector_io.pgvector.pgvector import PGVectorIndex

    pool, mock_conn = _make_mock_asyncpg_pool()

    vector_store = VectorStore(
        identifier="test-vector-db",
        embedding_model="test-model",
        embedding_dimension=768,
        provider_id="pgvector",
    )

    index = PGVectorIndex(
        vector_store=vector_store,
        dimension=768,
        pool=pool,
        distance_metric="COSINE",
        vector_index=PGVectorHNSWVectorIndex(m=16, ef_construction=64),
    )

    with patch.object(index, "create_gin_index", new_callable=AsyncMock) as mock_gin:
        await index.initialize()
        mock_gin.assert_called_once()


async def test_set_ef_search_called_before_select_in_query_vector(mock_asyncpg_pool, embedding_dimension):
    from ogx.providers.remote.vector_io.pgvector.config import PGVectorHNSWVectorIndex
    from ogx.providers.remote.vector_io.pgvector.pgvector import PGVectorIndex

    pool, mock_conn = mock_asyncpg_pool
    mock_conn.fetch = AsyncMock(return_value=[])

    index = PGVectorIndex(
        vector_store=VectorStore(
            identifier="test-vector-db",
            embedding_model="test-model",
            embedding_dimension=embedding_dimension,
            provider_id="pgvector",
        ),
        dimension=embedding_dimension,
        pool=pool,
        distance_metric="COSINE",
        vector_index=PGVectorHNSWVectorIndex(m=16, ef_construction=64, ef_search=50),
    )
    index.table_name = "test_table"
    index._quoted_table = '"test_table"'

    embedding = np.random.rand(embedding_dimension).astype(np.float32)
    await index.query_vector(embedding, k=5, score_threshold=0.5)

    execute_calls = mock_conn.execute.call_args_list
    fetch_calls = mock_conn.fetch.call_args_list

    assert len(execute_calls) == 1, f"Expected 1 execute call (SET), got {len(execute_calls)}"
    assert len(fetch_calls) == 1, f"Expected 1 fetch call (SELECT), got {len(fetch_calls)}"

    set_call_sql = str(execute_calls[0])
    select_call_sql = str(fetch_calls[0])
    assert f"SET hnsw.ef_search = {index.vector_index.ef_search}" in set_call_sql, (
        f"First call should be SET, got: {set_call_sql}"
    )
    assert "SELECT document" in select_call_sql, f"Second call should be SELECT, got: {select_call_sql}"


async def test_apply_default_ef_search_for_query_vector(mock_asyncpg_pool, embedding_dimension):
    from ogx.providers.remote.vector_io.pgvector.config import PGVectorHNSWVectorIndex
    from ogx.providers.remote.vector_io.pgvector.pgvector import PGVectorIndex

    pool, mock_conn = mock_asyncpg_pool
    mock_conn.fetch = AsyncMock(return_value=[])

    index = PGVectorIndex(
        vector_store=VectorStore(
            identifier="test-vector-db",
            embedding_model="test-model",
            embedding_dimension=embedding_dimension,
            provider_id="pgvector",
        ),
        dimension=embedding_dimension,
        pool=pool,
        distance_metric="COSINE",
        vector_index=PGVectorHNSWVectorIndex(m=16, ef_construction=64),
    )
    index.table_name = "test_table"
    index._quoted_table = '"test_table"'

    embedding = np.random.rand(embedding_dimension).astype(np.float32)
    await index.query_vector(embedding, k=5, score_threshold=0.5)

    execute_calls = mock_conn.execute.call_args_list
    set_call_sql = str(execute_calls[0])
    assert f"SET hnsw.ef_search = {PGVectorHNSWVectorIndex().ef_search}" in set_call_sql, (
        f"Expected default 'SET hnsw.ef_search = {PGVectorHNSWVectorIndex().ef_search}' when ef_search is not explicitly configured, got: {set_call_sql}"
    )


def _make_pgvector_adapter():
    """Create a PGVectorVectorIOAdapter with mock dependencies for pool tests."""
    from ogx.providers.remote.vector_io.pgvector.config import PGVectorHNSWVectorIndex, PGVectorVectorIOConfig
    from ogx.providers.remote.vector_io.pgvector.pgvector import PGVectorVectorIOAdapter

    config = PGVectorVectorIOConfig(
        host="localhost",
        port=5432,
        db="test_db",
        user="test_user",
        password="test_password",
        distance_metric="COSINE",
        vector_index=PGVectorHNSWVectorIndex(m=16, ef_construction=64),
    )
    mock_inference = AsyncMock()
    return PGVectorVectorIOAdapter(config, mock_inference, None)


async def test_ensure_pool_concurrent_calls_create_single_pool():
    """Test that concurrent _ensure_pool() calls create only one pool."""
    adapter = _make_pgvector_adapter()
    pool, mock_conn = _make_mock_asyncpg_pool()
    call_count = 0

    async def mock_create_pool(**kwargs):
        nonlocal call_count
        call_count += 1
        return pool

    with patch(
        "ogx.providers.remote.vector_io.pgvector.pgvector.asyncpg.create_pool",
        side_effect=mock_create_pool,
    ):
        with patch(
            "ogx.providers.remote.vector_io.pgvector.pgvector.check_extension_version",
            new_callable=AsyncMock,
            return_value="0.5.1",
        ):
            results = await asyncio.gather(
                adapter._ensure_pool(),
                adapter._ensure_pool(),
                adapter._ensure_pool(),
            )

    assert call_count == 1, f"Expected 1 pool creation, got {call_count}"
    assert all(r is pool for r in results)


async def test_ensure_pool_closes_pool_on_init_failure():
    """Test that pool is closed if one-time initialization fails."""
    adapter = _make_pgvector_adapter()
    pool, mock_conn = _make_mock_asyncpg_pool()
    mock_conn.execute = AsyncMock(side_effect=asyncpg.PostgresError("init failure"))

    with patch(
        "ogx.providers.remote.vector_io.pgvector.pgvector.asyncpg.create_pool",
        new_callable=AsyncMock,
        return_value=pool,
    ):
        with patch(
            "ogx.providers.remote.vector_io.pgvector.pgvector.check_extension_version",
            new_callable=AsyncMock,
            return_value=None,
        ):
            with patch(
                "ogx.providers.remote.vector_io.pgvector.pgvector.create_vector_extension",
                new_callable=AsyncMock,
            ):
                with pytest.raises(asyncpg.PostgresError, match="init failure"):
                    await adapter._ensure_pool()

    pool.close.assert_awaited_once()
    assert adapter.pool is None
    assert adapter._pool_initialized is False


async def test_ensure_pool_recreates_on_stale_event_loop():
    """Test that stale pool is closed and recreated when health check fails."""
    adapter = _make_pgvector_adapter()
    stale_pool, stale_conn = _make_mock_asyncpg_pool()
    stale_conn.fetchval = AsyncMock(side_effect=RuntimeError("wrong event loop"))

    new_pool, new_conn = _make_mock_asyncpg_pool()

    adapter.pool = stale_pool
    adapter._pool_initialized = True

    with patch(
        "ogx.providers.remote.vector_io.pgvector.pgvector.asyncpg.create_pool",
        new_callable=AsyncMock,
        return_value=new_pool,
    ):
        with patch(
            "ogx.providers.remote.vector_io.pgvector.pgvector.check_extension_version",
            new_callable=AsyncMock,
            return_value="0.5.1",
        ):
            result = await adapter._ensure_pool()

    assert result is new_pool
    stale_pool.close.assert_awaited_once()
    assert adapter.pool is new_pool


async def test_adapter_initialize_cleans_up_pool_on_index_failure():
    """Test that adapter.initialize() closes pool if PGVectorIndex.initialize() fails."""
    adapter = _make_pgvector_adapter()
    pool, mock_conn = _make_mock_asyncpg_pool()

    with patch(
        "ogx.providers.remote.vector_io.pgvector.pgvector.kvstore_impl", new_callable=AsyncMock
    ) as mock_kvstore_impl:
        mock_kvstore = AsyncMock()
        mock_kvstore.values_in_range = AsyncMock(
            return_value=['{"identifier":"vs1","embedding_model":"m","embedding_dimension":768,"provider_id":"p"}']
        )
        mock_kvstore_impl.return_value = mock_kvstore

        with patch(
            "ogx.providers.remote.vector_io.pgvector.pgvector.asyncpg.create_pool",
            new_callable=AsyncMock,
            return_value=pool,
        ):
            with patch(
                "ogx.providers.remote.vector_io.pgvector.pgvector.check_extension_version",
                new_callable=AsyncMock,
                return_value="0.5.1",
            ):
                with patch.object(adapter, "initialize_openai_vector_stores", new_callable=AsyncMock):
                    with patch(
                        "ogx.providers.remote.vector_io.pgvector.pgvector.PGVectorIndex.initialize",
                        new_callable=AsyncMock,
                        side_effect=RuntimeError("index init failed"),
                    ):
                        with pytest.raises(RuntimeError, match="index init failed"):
                            await adapter.initialize()

    pool.close.assert_awaited_once()
