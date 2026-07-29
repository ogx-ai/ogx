# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from ogx.providers.remote.vector_io.pgvector.config import PGVectorHNSWVectorIndex
from ogx.providers.remote.vector_io.pgvector.pgvector import PGVectorIndex
from ogx_api import VectorStore


@pytest.fixture
def mock_vector_store():
    store = MagicMock(spec=VectorStore)
    store.identifier = "test-store-123"
    store.embedding_dimension = 128
    return store


@pytest.fixture
def mock_pool():
    pool = MagicMock()
    conn = AsyncMock()
    conn.execute = AsyncMock()
    conn.executemany = AsyncMock()
    conn.fetch = AsyncMock(return_value=[])
    conn.fetchrow = AsyncMock(return_value=None)
    conn.fetchval = AsyncMock(return_value=0)

    acm = AsyncMock()
    acm.__aenter__ = AsyncMock(return_value=conn)
    acm.__aexit__ = AsyncMock(return_value=False)
    pool.acquire = MagicMock(return_value=acm)

    txn_acm = AsyncMock()
    txn_acm.__aenter__ = AsyncMock()
    txn_acm.__aexit__ = AsyncMock(return_value=False)
    conn.transaction = MagicMock(return_value=txn_acm)

    return pool, conn


@pytest.fixture
def pool_factory(mock_pool):
    pool, _ = mock_pool
    factory = AsyncMock(return_value=pool)
    return factory


@pytest.fixture
def pgvector_index(mock_vector_store, pool_factory):
    return PGVectorIndex(
        vector_store=mock_vector_store,
        dimension=128,
        pool_factory=pool_factory,
        distance_metric="COSINE",
        vector_index=PGVectorHNSWVectorIndex(),
    )


async def test_get_pool_calls_factory(pgvector_index, pool_factory, mock_pool):
    """_get_pool delegates to the pool_factory callable."""
    pool, _ = mock_pool
    result = await pgvector_index._get_pool()
    pool_factory.assert_awaited_once()
    assert result is pool


async def test_initialize_uses_pool_factory(pgvector_index, pool_factory):
    """initialize() acquires pool via factory, not a stored reference."""
    await pgvector_index.initialize()
    pool_factory.assert_awaited()


async def test_add_chunks_uses_pool_factory(pgvector_index, pool_factory):
    """add_chunks() acquires pool via factory."""
    chunk = MagicMock()
    chunk.chunk_id = "chunk-1"
    chunk.content = "test content"
    chunk.embedding = [0.1] * 128
    chunk.model_dump = MagicMock(return_value={"chunk_id": "chunk-1", "content": "test"})

    with patch(
        "ogx.providers.remote.vector_io.pgvector.pgvector.interleaved_content_as_str",
        return_value="test content",
    ):
        await pgvector_index.initialize()
        pool_factory.reset_mock()
        await pgvector_index.add_chunks([chunk])

    pool_factory.assert_awaited()


async def test_query_vector_uses_pool_factory(pgvector_index, pool_factory):
    """query_vector() acquires pool via factory."""
    await pgvector_index.initialize()
    pool_factory.reset_mock()

    embedding = np.zeros(128)
    result = await pgvector_index.query_vector(embedding, k=5, score_threshold=0.0)

    pool_factory.assert_awaited()
    assert result.chunks == []


async def test_query_keyword_uses_pool_factory(pgvector_index, pool_factory):
    """query_keyword() acquires pool via factory."""
    await pgvector_index.initialize()
    pool_factory.reset_mock()

    result = await pgvector_index.query_keyword("test", k=5, score_threshold=0.0)

    pool_factory.assert_awaited()
    assert result.chunks == []


async def test_delete_uses_pool_factory(pgvector_index, pool_factory):
    """delete() acquires pool via factory."""
    await pgvector_index.initialize()
    pool_factory.reset_mock()

    await pgvector_index.delete()
    pool_factory.assert_awaited()


async def test_delete_chunks_uses_pool_factory(pgvector_index, pool_factory):
    """delete_chunks() acquires pool via factory."""
    await pgvector_index.initialize()
    pool_factory.reset_mock()

    chunk = MagicMock()
    chunk.chunk_id = "chunk-1"
    await pgvector_index.delete_chunks([chunk])
    pool_factory.assert_awaited()


async def test_pool_factory_called_each_operation(pgvector_index, pool_factory):
    """Each operation calls factory independently — stale pools get refreshed."""
    await pgvector_index.initialize()
    pool_factory.reset_mock()

    embedding = np.zeros(128)
    await pgvector_index.query_vector(embedding, k=5, score_threshold=0.0)
    await pgvector_index.query_keyword("test", k=5, score_threshold=0.0)
    await pgvector_index.delete()

    assert pool_factory.await_count == 3


async def test_pool_factory_recreation_on_new_pool(mock_vector_store):
    """When factory returns a different pool (after recreation), operations use new pool."""
    pool_a = MagicMock()
    pool_b = MagicMock()

    for pool in (pool_a, pool_b):
        conn = AsyncMock()
        conn.execute = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=None)
        conn.fetchval = AsyncMock(return_value=0)

        acm = AsyncMock()
        acm.__aenter__ = AsyncMock(return_value=conn)
        acm.__aexit__ = AsyncMock(return_value=False)
        pool.acquire = MagicMock(return_value=acm)

    call_count = 0

    async def rotating_factory():
        nonlocal call_count
        call_count += 1
        return pool_a if call_count <= 1 else pool_b

    index = PGVectorIndex(
        vector_store=mock_vector_store,
        dimension=128,
        pool_factory=rotating_factory,
        distance_metric="COSINE",
        vector_index=PGVectorHNSWVectorIndex(),
    )

    await index.initialize()
    pool_a.acquire.assert_called()

    await index.delete()
    pool_b.acquire.assert_called()
