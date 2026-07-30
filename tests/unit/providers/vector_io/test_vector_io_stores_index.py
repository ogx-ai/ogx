# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import time
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from ogx.providers.inline.vector_io.sqlite_vec.sqlite_vec import VECTOR_DBS_PREFIX
from ogx_api import (
    Chunk,
    EmbeddedChunk,
    InsertChunksRequest,
    QueryChunksRequest,
    QueryChunksResponse,
    VectorStore,
    VectorStoreNotFoundError,
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


async def test_initialize_index(vector_index):
    await vector_index.initialize()


async def test_add_chunks_query_vector(vector_index, sample_chunks, sample_embeddings):
    vector_index.delete()
    vector_index.initialize()
    # Create EmbeddedChunk objects using inheritance pattern
    embedded_chunks = [
        EmbeddedChunk(
            content=chunk.content,
            chunk_id=chunk.chunk_id,
            metadata=chunk.metadata,
            chunk_metadata=chunk.chunk_metadata,
            embedding=embedding.tolist(),
            embedding_model="test-embedding-model",
            embedding_dimension=len(embedding),
        )
        for chunk, embedding in zip(sample_chunks, sample_embeddings, strict=False)
    ]
    await vector_index.add_chunks(embedded_chunks)
    resp = await vector_index.query_vector(sample_embeddings[0], k=1, score_threshold=-1)
    assert resp.chunks[0].content == sample_chunks[0].content
    vector_index.delete()


async def test_chunk_id_conflict(vector_index, sample_chunks, embedding_dimension):
    embeddings = np.random.rand(len(sample_chunks), embedding_dimension).astype(np.float32)
    # Create EmbeddedChunk objects using inheritance pattern
    embedded_chunks = [
        EmbeddedChunk(
            content=chunk.content,
            chunk_id=chunk.chunk_id,
            metadata=chunk.metadata,
            chunk_metadata=chunk.chunk_metadata,
            embedding=embedding.tolist(),
            embedding_model="test-embedding-model",
            embedding_dimension=len(embedding),
        )
        for chunk, embedding in zip(sample_chunks, embeddings, strict=False)
    ]
    await vector_index.add_chunks(embedded_chunks)
    resp = await vector_index.query_vector(
        np.random.rand(embedding_dimension).astype(np.float32),
        k=len(sample_chunks),
        score_threshold=-1,
    )

    contents = [embedded_chunk.content for embedded_chunk in resp.chunks]
    assert len(contents) == len(set(contents))


async def test_initialize_adapter_with_existing_kvstore(vector_io_adapter):
    key = f"{VECTOR_DBS_PREFIX}db1"
    dummy = VectorStore(
        identifier="foo_db", provider_id="test_provider", embedding_model="test_model", embedding_dimension=128
    )
    await vector_io_adapter.kvstore.set(key=key, value=json.dumps(dummy.model_dump()))

    await vector_io_adapter.initialize()


async def test_persistence_across_adapter_restarts(vector_io_adapter):
    await vector_io_adapter.initialize()
    dummy = VectorStore(
        identifier="foo_db", provider_id="test_provider", embedding_model="test_model", embedding_dimension=128
    )
    await vector_io_adapter.register_vector_store(dummy)
    await vector_io_adapter.shutdown()

    await vector_io_adapter.initialize()
    assert "foo_db" in vector_io_adapter.cache
    await vector_io_adapter.shutdown()


async def test_vector_store_lazy_loading_from_kvstore(vector_io_adapter):
    """
    Test that vector stores can be lazy-loaded from KV store when not in cache.

    Verifies that clearing the cache doesn't break vector store access - they
    can be loaded on-demand from persistent storage.
    """
    await vector_io_adapter.initialize()

    vector_store_id = f"lazy_load_test_{np.random.randint(1e6)}"
    vector_store = VectorStore(
        identifier=vector_store_id,
        provider_id="test_provider",
        embedding_model="test_model",
        embedding_dimension=128,
    )
    await vector_io_adapter.register_vector_store(vector_store)
    assert vector_store_id in vector_io_adapter.cache

    vector_io_adapter.cache.clear()
    assert vector_store_id not in vector_io_adapter.cache

    loaded_index = await vector_io_adapter._get_and_cache_vector_store_index(vector_store_id)
    assert loaded_index is not None
    assert loaded_index.vector_store.identifier == vector_store_id
    assert vector_store_id in vector_io_adapter.cache

    cached_index = await vector_io_adapter._get_and_cache_vector_store_index(vector_store_id)
    assert cached_index is loaded_index

    await vector_io_adapter.shutdown()


async def test_vector_store_preloading_on_initialization(vector_io_adapter):
    """
    Test that vector stores are preloaded from KV store during initialization.

    Verifies that after restart, all vector stores are automatically loaded into
    cache and immediately accessible without requiring lazy loading.
    """
    await vector_io_adapter.initialize()

    vector_store_ids = [f"preload_test_{i}_{np.random.randint(1e6)}" for i in range(3)]
    for vs_id in vector_store_ids:
        vector_store = VectorStore(
            identifier=vs_id,
            provider_id="test_provider",
            embedding_model="test_model",
            embedding_dimension=128,
        )
        await vector_io_adapter.register_vector_store(vector_store)

    for vs_id in vector_store_ids:
        assert vs_id in vector_io_adapter.cache

    await vector_io_adapter.shutdown()
    await vector_io_adapter.initialize()

    for vs_id in vector_store_ids:
        assert vs_id in vector_io_adapter.cache

    for vs_id in vector_store_ids:
        loaded_index = await vector_io_adapter._get_and_cache_vector_store_index(vs_id)
        assert loaded_index is not None
        assert loaded_index.vector_store.identifier == vs_id

    await vector_io_adapter.shutdown()


async def test_kvstore_none_raises_runtime_error(vector_io_adapter):
    """
    Test that accessing vector stores with uninitialized kvstore raises RuntimeError.

    Verifies proper RuntimeError is raised instead of assertions when kvstore is None.
    """
    await vector_io_adapter.initialize()

    vector_store_id = f"kvstore_none_test_{np.random.randint(1e6)}"
    vector_store = VectorStore(
        identifier=vector_store_id,
        provider_id="test_provider",
        embedding_model="test_model",
        embedding_dimension=128,
    )
    await vector_io_adapter.register_vector_store(vector_store)

    vector_io_adapter.cache.clear()
    vector_io_adapter.kvstore = None

    with pytest.raises(RuntimeError, match="KVStore not initialized"):
        await vector_io_adapter._get_and_cache_vector_store_index(vector_store_id)


async def test_register_and_unregister_vector_store(vector_io_adapter):
    unique_id = f"foo_db_{np.random.randint(1e6)}"
    dummy = VectorStore(
        identifier=unique_id, provider_id="test_provider", embedding_model="test_model", embedding_dimension=128
    )

    await vector_io_adapter.register_vector_store(dummy)
    assert dummy.identifier in vector_io_adapter.cache
    await vector_io_adapter.unregister_vector_store(dummy.identifier)
    assert dummy.identifier not in vector_io_adapter.cache


async def test_query_unregistered_raises(vector_io_adapter, vector_provider):
    request = QueryChunksRequest(vector_store_id="no_such_db", query="test query")
    with pytest.raises(VectorStoreNotFoundError):
        await vector_io_adapter.query_chunks(request)


async def test_insert_chunks_calls_underlying_index(vector_io_adapter, sample_chunks):
    import numpy as np

    from ogx_api import EmbeddedChunk

    fake_index = AsyncMock()
    vector_io_adapter.cache["db1"] = fake_index

    # Convert Chunk objects to EmbeddedChunk objects
    embedded_chunks = [
        EmbeddedChunk(
            **chunk.model_dump(),
            embedding=np.random.rand(768).astype(np.float32).tolist(),  # Add mock embedding
            embedding_model="test-model",  # Add required field
            embedding_dimension=768,  # Add required field
        )
        for chunk in sample_chunks[:2]  # Take first 2 chunks from fixture
    ]
    request = InsertChunksRequest(vector_store_id="db1", chunks=embedded_chunks)
    await vector_io_adapter.insert_chunks(request)

    fake_index.insert_chunks.assert_awaited_once_with(request)


async def test_insert_chunks_missing_db_raises(vector_io_adapter):
    vector_io_adapter._get_and_cache_vector_store_index = AsyncMock(return_value=None)

    request = InsertChunksRequest(vector_store_id="db_not_exist", chunks=[])
    with pytest.raises(VectorStoreNotFoundError):
        await vector_io_adapter.insert_chunks(request)


async def test_insert_chunks_with_missing_document_id(vector_io_adapter):
    """Ensure no KeyError when document_id is missing or in different places."""
    from ogx_api import Chunk, ChunkMetadata

    fake_index = AsyncMock()
    vector_io_adapter.cache["db1"] = fake_index

    # Various document_id scenarios that shouldn't crash
    from ogx.providers.utils.vector_io.vector_utils import generate_chunk_id

    chunks = [
        Chunk(
            content="has doc_id in metadata",
            chunk_id=generate_chunk_id("doc-1", "has doc_id in metadata"),
            metadata={"document_id": "doc-1"},
            embedding=[],
            chunk_metadata=ChunkMetadata(
                document_id="doc-1",
                chunk_id=generate_chunk_id("doc-1", "has doc_id in metadata"),
                created_timestamp=int(time.time()),
                updated_timestamp=int(time.time()),
                chunk_embedding_model="test-model",
                chunk_embedding_dimension=768,
                content_token_count=5,
            ),
        ),
        Chunk(
            content="no doc_id anywhere",
            chunk_id=generate_chunk_id("unknown", "no doc_id anywhere"),
            metadata={"source": "test"},
            embedding=[],
            chunk_metadata=ChunkMetadata(
                document_id=None,
                chunk_id=generate_chunk_id("unknown", "no doc_id anywhere"),
                created_timestamp=int(time.time()),
                updated_timestamp=int(time.time()),
                chunk_embedding_model="test-model",
                chunk_embedding_dimension=768,
                content_token_count=4,
            ),
        ),
        Chunk(
            content="doc_id in chunk_metadata",
            chunk_id=generate_chunk_id("doc-3", "doc_id in chunk_metadata"),
            metadata={},
            embedding=[],
            chunk_metadata=ChunkMetadata(
                document_id="doc-3",
                chunk_id=generate_chunk_id("doc-3", "doc_id in chunk_metadata"),
                created_timestamp=int(time.time()),
                updated_timestamp=int(time.time()),
                chunk_embedding_model="test-model",
                chunk_embedding_dimension=768,
                content_token_count=5,
            ),
        ),
    ]

    # Convert Chunk objects to EmbeddedChunk objects
    import numpy as np

    from ogx_api import EmbeddedChunk

    embedded_chunks = [
        EmbeddedChunk(
            **chunk.model_dump(),
            embedding=np.random.rand(768).astype(np.float32).tolist(),  # Add mock embedding
            embedding_model="test-model",  # Add required field
            embedding_dimension=768,  # Add required field
        )
        for chunk in chunks
    ]

    # Should work without KeyError
    request = InsertChunksRequest(vector_store_id="db1", chunks=embedded_chunks)
    await vector_io_adapter.insert_chunks(request)
    fake_index.insert_chunks.assert_awaited_once()


async def test_document_id_with_invalid_type_raises_error():
    """Ensure TypeError is raised when document_id is not a string."""
    # Integer document_id should raise TypeError
    from ogx.providers.utils.vector_io.vector_utils import generate_chunk_id
    from ogx_api import Chunk, ChunkMetadata

    chunk_id = generate_chunk_id("test", "test")
    chunk = Chunk(
        content="test",
        chunk_id=chunk_id,
        metadata={"document_id": 12345},
        embedding=[],
        chunk_metadata=ChunkMetadata(
            document_id=None,
            chunk_id=chunk_id,
            created_timestamp=int(time.time()),
            updated_timestamp=int(time.time()),
            chunk_embedding_model="test-model",
            chunk_embedding_dimension=768,
            content_token_count=1,
        ),
    )
    with pytest.raises(TypeError) as exc_info:
        _ = chunk.document_id
    assert "metadata['document_id'] must be a string" in str(exc_info.value)
    assert "got int" in str(exc_info.value)


async def test_query_chunks_calls_underlying_index_and_returns(vector_io_adapter):
    from ogx.providers.utils.vector_io.vector_utils import generate_chunk_id
    from ogx_api import ChunkMetadata

    chunk_id = generate_chunk_id("test", "c1")
    chunk = Chunk(
        content="c1",
        chunk_id=chunk_id,
        metadata={},
        chunk_metadata=ChunkMetadata(
            document_id="test",
            chunk_id=chunk_id,
            created_timestamp=int(time.time()),
            updated_timestamp=int(time.time()),
            content_token_count=1,
        ),
    )

    embedded_chunk = EmbeddedChunk(
        content=chunk.content,
        chunk_id=chunk.chunk_id,
        metadata=chunk.metadata,
        chunk_metadata=chunk.chunk_metadata,
        embedding=[0.1, 0.2, 0.3],
        embedding_model="test-model",
        embedding_dimension=3,
    )
    expected = QueryChunksResponse(chunks=[embedded_chunk], scores=[0.1])
    fake_index = AsyncMock(query_chunks=AsyncMock(return_value=expected))
    vector_io_adapter.cache["db1"] = fake_index

    request = QueryChunksRequest(vector_store_id="db1", query="my_query", params={"param": 1})
    response = await vector_io_adapter.query_chunks(request)

    # Verify query_chunks was called with the expected request object
    fake_index.query_chunks.assert_awaited_once_with(request)
    assert response is expected


async def test_query_chunks_missing_db_raises(vector_io_adapter):
    vector_io_adapter._get_and_cache_vector_store_index = AsyncMock(return_value=None)

    request = QueryChunksRequest(vector_store_id="db_missing", query="q", params=None)
    with pytest.raises(VectorStoreNotFoundError):
        await vector_io_adapter.query_chunks(request)
