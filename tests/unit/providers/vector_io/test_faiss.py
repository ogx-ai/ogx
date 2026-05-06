# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ogx.providers.inline.vector_io.faiss.config import FaissVectorIOConfig
from ogx.providers.inline.vector_io.faiss.faiss import (
    FaissIndex,
    FaissVectorIOAdapter,
)
from ogx_api import Chunk, ChunkMetadata, EmbeddedChunk, Files, HealthStatus, QueryChunksResponse, VectorStore

# This test is a unit test for the FaissVectorIOAdapter class. This should only contain
# tests which are specific to this class. More general (API-level) tests should be placed in
# tests/integration/vector_io/
#
# How to run this test:
#
# pytest tests/unit/providers/vector_io/test_faiss.py \
# -v -s --tb=short --disable-warnings --asyncio-mode=auto

FAISS_PROVIDER = "faiss"


@pytest.fixture(scope="session")
def loop():
    return asyncio.new_event_loop()


@pytest.fixture
def embedding_dimension():
    return 768


@pytest.fixture
def vector_store_id():
    return "test_vector_store"


@pytest.fixture
def sample_chunks():
    import time

    from ogx.providers.utils.vector_io.vector_utils import generate_chunk_id

    chunk_id_1 = generate_chunk_id("mock-doc-1", "MOCK text content 1")
    chunk_id_2 = generate_chunk_id("mock-doc-2", "MOCK text content 1")

    return [
        Chunk(
            content="MOCK text content 1",
            chunk_id=chunk_id_1,
            metadata={"document_id": "mock-doc-1"},
            chunk_metadata=ChunkMetadata(
                chunk_id=chunk_id_1,
                document_id="mock-doc-1",
                created_timestamp=int(time.time()),
                updated_timestamp=int(time.time()),
                content_token_count=4,
            ),
        ),
        Chunk(
            content="MOCK text content 1",
            chunk_id=chunk_id_2,
            metadata={"document_id": "mock-doc-2"},
            chunk_metadata=ChunkMetadata(
                chunk_id=chunk_id_2,
                document_id="mock-doc-2",
                created_timestamp=int(time.time()),
                updated_timestamp=int(time.time()),
                content_token_count=4,
            ),
        ),
    ]


@pytest.fixture
def sample_embeddings(embedding_dimension):
    return np.random.rand(2, embedding_dimension).astype(np.float32)


@pytest.fixture
def mock_vector_store(vector_store_id, embedding_dimension) -> MagicMock:
    mock_vector_store = MagicMock(spec=VectorStore)
    mock_vector_store.embedding_model = "mock_embedding_model"
    mock_vector_store.identifier = vector_store_id
    mock_vector_store.embedding_dimension = embedding_dimension
    return mock_vector_store


@pytest.fixture
def mock_files_api():
    mock_api = MagicMock(spec=Files)
    return mock_api


@pytest.fixture
def faiss_config():
    config = MagicMock(spec=FaissVectorIOConfig)
    config.kvstore = None
    return config


@pytest.fixture
async def faiss_index(embedding_dimension):
    index = await FaissIndex.create(dimension=embedding_dimension)
    yield index


async def test_faiss_query_vector_returns_infinity_when_query_and_embedding_are_identical(
    faiss_index, sample_chunks, sample_embeddings, embedding_dimension
):
    # Create EmbeddedChunk objects from chunks and embeddings
    embedded_chunks = [
        EmbeddedChunk(
            content=chunk.content,
            chunk_id=chunk.chunk_id,
            metadata=chunk.metadata,
            chunk_metadata=chunk.chunk_metadata,
            embedding=embedding.tolist(),
            embedding_model="test-embedding-model",
            embedding_dimension=embedding_dimension,
        )
        for chunk, embedding in zip(sample_chunks, sample_embeddings, strict=False)
    ]
    await faiss_index.add_chunks(embedded_chunks)
    query_embedding = np.random.rand(embedding_dimension).astype(np.float32)

    with patch.object(faiss_index.index, "search") as mock_search:
        mock_search.return_value = (np.array([[0.0, 0.1]]), np.array([[0, 1]]))

        response = await faiss_index.query_vector(embedding=query_embedding, k=2, score_threshold=0.0)

        assert isinstance(response, QueryChunksResponse)
        assert len(response.chunks) == 2
        assert len(response.scores) == 2

        assert response.scores[0] == float("inf")  # infinity (1.0 / 0.0)
        assert response.scores[1] == 10.0  # (1.0 / 0.1 = 10.0)

        assert response.chunks[0] == embedded_chunks[0]
        assert response.chunks[1] == embedded_chunks[1]


async def test_meta_index_populated_on_add_chunks(faiss_index, sample_chunks, sample_embeddings, embedding_dimension):
    """_meta_index should reflect every (key, value) pair from added chunk metadata."""
    embedded_chunks = [
        EmbeddedChunk(
            content=chunk.content,
            chunk_id=chunk.chunk_id,
            metadata=chunk.metadata,
            chunk_metadata=chunk.chunk_metadata,
            embedding=embedding.tolist(),
            embedding_model="test-embedding-model",
            embedding_dimension=embedding_dimension,
        )
        for chunk, embedding in zip(sample_chunks, sample_embeddings, strict=False)
    ]
    await faiss_index.add_chunks(embedded_chunks)

    # Both chunks have "document_id" key with distinct values
    assert "document_id" in faiss_index._meta_index
    assert "mock-doc-1" in faiss_index._meta_index["document_id"]
    assert "mock-doc-2" in faiss_index._meta_index["document_id"]
    assert faiss_index._meta_index["document_id"]["mock-doc-1"] == {0}
    assert faiss_index._meta_index["document_id"]["mock-doc-2"] == {1}


async def test_meta_index_updated_on_delete_chunks(faiss_index, sample_chunks, sample_embeddings, embedding_dimension):
    """After deleting a chunk, its position should be removed and remaining positions shifted."""
    from ogx.providers.utils.memory.vector_store import ChunkForDeletion

    embedded_chunks = [
        EmbeddedChunk(
            content=chunk.content,
            chunk_id=chunk.chunk_id,
            metadata=chunk.metadata,
            chunk_metadata=chunk.chunk_metadata,
            embedding=embedding.tolist(),
            embedding_model="test-embedding-model",
            embedding_dimension=embedding_dimension,
        )
        for chunk, embedding in zip(sample_chunks, sample_embeddings, strict=False)
    ]
    await faiss_index.add_chunks(embedded_chunks)

    # Delete the first chunk (position 0); the second chunk should shift to position 0
    await faiss_index.delete_chunks([ChunkForDeletion(chunk_id=sample_chunks[0].chunk_id, document_id="mock-doc-1")])

    assert "mock-doc-1" not in faiss_index._meta_index.get("document_id", {})
    # mock-doc-2 was at position 1, should have shifted to 0
    assert faiss_index._meta_index["document_id"]["mock-doc-2"] == {0}


async def test_resolve_filter_positions_eq(faiss_index, sample_chunks, sample_embeddings, embedding_dimension):
    """eq filter should return only positions whose metadata value matches exactly."""
    embedded_chunks = [
        EmbeddedChunk(
            content=chunk.content,
            chunk_id=chunk.chunk_id,
            metadata=chunk.metadata,
            chunk_metadata=chunk.chunk_metadata,
            embedding=embedding.tolist(),
            embedding_model="test-embedding-model",
            embedding_dimension=embedding_dimension,
        )
        for chunk, embedding in zip(sample_chunks, sample_embeddings, strict=False)
    ]
    await faiss_index.add_chunks(embedded_chunks)

    from ogx.providers.utils.vector_io.filters import ComparisonFilter

    result = faiss_index._resolve_filter_positions(ComparisonFilter(key="document_id", value="mock-doc-1", type="eq"))
    assert result == {0}

    result = faiss_index._resolve_filter_positions(ComparisonFilter(key="document_id", value="missing-doc", type="eq"))
    assert result == set()


async def test_resolve_filter_positions_in_nin(faiss_index, sample_chunks, sample_embeddings, embedding_dimension):
    """in/nin filters should include/exclude the listed values."""
    embedded_chunks = [
        EmbeddedChunk(
            content=chunk.content,
            chunk_id=chunk.chunk_id,
            metadata=chunk.metadata,
            chunk_metadata=chunk.chunk_metadata,
            embedding=embedding.tolist(),
            embedding_model="test-embedding-model",
            embedding_dimension=embedding_dimension,
        )
        for chunk, embedding in zip(sample_chunks, sample_embeddings, strict=False)
    ]
    await faiss_index.add_chunks(embedded_chunks)

    from ogx.providers.utils.vector_io.filters import ComparisonFilter

    result = faiss_index._resolve_filter_positions(
        ComparisonFilter(key="document_id", value=["mock-doc-1", "mock-doc-2"], type="in")
    )
    assert result == {0, 1}

    result = faiss_index._resolve_filter_positions(
        ComparisonFilter(key="document_id", value=["mock-doc-1"], type="nin")
    )
    assert result == {1}


async def test_resolve_filter_positions_compound(faiss_index, sample_chunks, sample_embeddings, embedding_dimension):
    """CompoundFilter and/or should correctly combine sub-filter position sets."""
    embedded_chunks = [
        EmbeddedChunk(
            content=chunk.content,
            chunk_id=chunk.chunk_id,
            metadata=chunk.metadata,
            chunk_metadata=chunk.chunk_metadata,
            embedding=embedding.tolist(),
            embedding_model="test-embedding-model",
            embedding_dimension=embedding_dimension,
        )
        for chunk, embedding in zip(sample_chunks, sample_embeddings, strict=False)
    ]
    await faiss_index.add_chunks(embedded_chunks)

    from ogx.providers.utils.vector_io.filters import ComparisonFilter, CompoundFilter

    # AND: only positions matching both — no chunk has both doc ids, so empty
    result = faiss_index._resolve_filter_positions(
        CompoundFilter(
            type="and",
            filters=[
                ComparisonFilter(key="document_id", value="mock-doc-1", type="eq"),
                ComparisonFilter(key="document_id", value="mock-doc-2", type="eq"),
            ],
        )
    )
    assert result == set()

    # OR: positions matching either
    result = faiss_index._resolve_filter_positions(
        CompoundFilter(
            type="or",
            filters=[
                ComparisonFilter(key="document_id", value="mock-doc-1", type="eq"),
                ComparisonFilter(key="document_id", value="mock-doc-2", type="eq"),
            ],
        )
    )
    assert result == {0, 1}


async def test_query_vector_with_filter_returns_correct_chunks(embedding_dimension):
    """Filtered query_vector should only return chunks matching the filter."""
    import time

    from ogx.providers.utils.vector_io.filters import ComparisonFilter
    from ogx.providers.utils.vector_io.vector_utils import generate_chunk_id

    idx = await FaissIndex.create(dimension=embedding_dimension)
    rng = np.random.default_rng(0)

    chunks = []
    for i, topic in enumerate(["ai", "ml", "ai", "cv"]):
        cid = generate_chunk_id(f"doc{i}", f"content {i}")
        chunks.append(
            EmbeddedChunk(
                content=f"content {i}",
                chunk_id=cid,
                metadata={"topic": topic},
                chunk_metadata=ChunkMetadata(
                    chunk_id=cid,
                    document_id=f"doc{i}",
                    created_timestamp=int(time.time()),
                    updated_timestamp=int(time.time()),
                    content_token_count=2,
                ),
                embedding=rng.random(embedding_dimension, dtype=np.float32).tolist(),
                embedding_model="test-model",
                embedding_dimension=embedding_dimension,
            )
        )
    await idx.add_chunks(chunks)

    f = ComparisonFilter(key="topic", value="ai", type="eq")
    response = await idx.query_vector(
        embedding=rng.random(embedding_dimension, dtype=np.float32),
        k=10,
        score_threshold=0.0,
        filters=f,
    )

    assert len(response.chunks) == 2
    assert all(c.metadata["topic"] == "ai" for c in response.chunks)


async def test_health_success():
    """Test that the health check returns OK status when faiss is working correctly."""
    # Create a fresh instance of FaissVectorIOAdapter for testing
    config = MagicMock()
    inference_api = MagicMock()
    files_api = MagicMock()

    mock_faiss = MagicMock()
    mock_faiss.IndexFlatL2.return_value = MagicMock()

    with patch("ogx.providers.inline.vector_io.faiss.faiss._get_faiss", return_value=mock_faiss):
        adapter = FaissVectorIOAdapter(config=config, inference_api=inference_api, files_api=files_api)

        # Calling the health method directly
        response = await adapter.health()

        # Verifying the response
        assert isinstance(response, dict)
        assert response["status"] == HealthStatus.OK
        assert "message" not in response

        # Verifying that IndexFlatL2 was called with the correct dimension
        mock_faiss.IndexFlatL2.assert_called_once_with(128)  # VECTOR_DIMENSION is 128


async def test_health_failure():
    """Test that the health check returns ERROR status when faiss encounters an error."""
    # Create a fresh instance of FaissVectorIOAdapter for testing
    config = MagicMock()
    inference_api = MagicMock()
    files_api = MagicMock()

    mock_faiss = MagicMock()
    mock_faiss.IndexFlatL2.side_effect = Exception("Test error")

    with patch("ogx.providers.inline.vector_io.faiss.faiss._get_faiss", return_value=mock_faiss):
        adapter = FaissVectorIOAdapter(config=config, inference_api=inference_api, files_api=files_api)

        # Calling the health method directly
        response = await adapter.health()

        # Verifying the response
        assert isinstance(response, dict)
        assert response["status"] == HealthStatus.ERROR
        assert response["message"] == "Health check failed: Test error"
