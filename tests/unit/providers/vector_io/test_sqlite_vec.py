# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import time

import numpy as np
import pytest

from ogx.providers.inline.vector_io.sqlite_vec.sqlite_vec import (
    SQLiteVecIndex,
    SQLiteVecVectorIOAdapter,
    _create_sqlite_connection,
)
from ogx_api import Chunk, ChunkMetadata, EmbeddedChunk, QueryChunksResponse

# This test is a unit test for the SQLiteVecVectorIOAdapter class. This should only contain
# tests which are specific to this class. More general (API-level) tests should be placed in
# tests/integration/vector_io/
#
# How to run this test:
#
# pytest tests/unit/providers/vector_io/test_sqlite_vec.py \
# -v -s --tb=short --disable-warnings --asyncio-mode=auto

SQLITE_VEC_PROVIDER = "sqlite_vec"


@pytest.fixture(scope="session")
def loop():
    return asyncio.new_event_loop()


@pytest.fixture
async def sqlite_vec_index(embedding_dimension, tmp_path_factory):
    temp_dir = tmp_path_factory.getbasetemp()
    db_path = str(temp_dir / "test_sqlite.db")
    index = await SQLiteVecIndex.create(dimension=embedding_dimension, db_path=db_path, bank_id="test_bank.123")
    yield index
    await index.delete()


async def test_query_chunk_metadata(sqlite_vec_index, sample_chunks_with_metadata, sample_embeddings_with_metadata):
    # Create EmbeddedChunk objects
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
        for chunk, embedding in zip(sample_chunks_with_metadata, sample_embeddings_with_metadata, strict=False)
    ]
    await sqlite_vec_index.add_chunks(embedded_chunks)
    response = await sqlite_vec_index.query_vector(sample_embeddings_with_metadata[-1], k=2, score_threshold=0.0)
    assert response.chunks[0].chunk_metadata == sample_chunks_with_metadata[-1].chunk_metadata


async def test_query_chunks_full_text_search(sqlite_vec_index, sample_chunks, sample_embeddings):
    # Create EmbeddedChunk objects
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
    await sqlite_vec_index.add_chunks(embedded_chunks)
    query_string = "Sentence 5"
    response = await sqlite_vec_index.query_keyword(k=3, score_threshold=0.0, query_string=query_string)

    assert isinstance(response, QueryChunksResponse)
    assert len(response.chunks) == 3, f"Expected three chunks, but got {len(response.chunks)}"

    non_existent_query_str = "blablabla"
    response_no_results = await sqlite_vec_index.query_keyword(
        query_string=non_existent_query_str, k=1, score_threshold=0.0
    )

    assert isinstance(response_no_results, QueryChunksResponse)
    assert len(response_no_results.chunks) == 0, f"Expected 0 results, but got {len(response_no_results.chunks)}"


async def test_query_keyword_scores_are_positive(sqlite_vec_index, sample_chunks, sample_embeddings):
    """Test that keyword search returns positive scores (higher = more relevant).

    FTS5 BM25 internally returns negative scores (more negative = more relevant),
    but query_keyword must negate them so downstream consumers like RRF can sort
    descending (higher = better) without inverting the ranking.
    """
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
    await sqlite_vec_index.add_chunks(embedded_chunks)

    response = await sqlite_vec_index.query_keyword(k=5, score_threshold=0.0, query_string="Sentence 5")
    assert len(response.chunks) > 0, "Expected at least one result"
    for score in response.scores:
        assert score > 0, f"Keyword score should be positive (higher = more relevant), got {score}"
    # Scores should be in descending order (most relevant first)
    assert all(response.scores[i] >= response.scores[i + 1] for i in range(len(response.scores) - 1)), (
        "Keyword scores should be in descending order"
    )


async def test_query_chunks_hybrid(sqlite_vec_index, sample_chunks, sample_embeddings):
    # Create EmbeddedChunk objects
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
    await sqlite_vec_index.add_chunks(embedded_chunks)

    # Create a query embedding that's similar to the first chunk
    query_embedding = sample_embeddings[0]
    query_string = "Sentence 5"

    response = await sqlite_vec_index.query_hybrid(
        embedding=query_embedding,
        query_string=query_string,
        k=3,
        score_threshold=0.0,
        reranker_type="rrf",
        reranker_params={"impact_factor": 60.0},
    )

    assert len(response.chunks) == 3, f"Expected 3 results, got {len(response.chunks)}"
    # Verify scores are in descending order (higher is better)
    assert all(response.scores[i] >= response.scores[i + 1] for i in range(len(response.scores) - 1))


async def test_query_chunks_full_text_search_k_greater_than_results(sqlite_vec_index, sample_chunks, sample_embeddings):
    # Create EmbeddedChunk objects
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
    await sqlite_vec_index.add_chunks(embedded_chunks)

    query_str = "Sentence 1 from document 0"  # Should match only one chunk
    response = await sqlite_vec_index.query_keyword(k=5, score_threshold=0.0, query_string=query_str)

    assert isinstance(response, QueryChunksResponse)
    assert 0 < len(response.chunks) < 5, f"Expected results between [1, 4], got {len(response.chunks)}"
    assert any("Sentence 1 from document 0" in embedded_chunk.content for embedded_chunk in response.chunks), (
        "Expected chunk not found"
    )


async def test_chunk_id_conflict(sqlite_vec_index, sample_chunks, embedding_dimension):
    """Test that chunk IDs do not conflict across batches when inserting chunks."""
    # Reduce batch size to force multiple batches for same document
    # since there are 10 chunks per document and batch size is 2
    batch_size = 2
    sample_embeddings = np.random.rand(len(sample_chunks), embedding_dimension).astype(np.float32)

    # Create EmbeddedChunk objects
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
    await sqlite_vec_index.add_chunks(embedded_chunks, batch_size=batch_size)
    connection = _create_sqlite_connection(sqlite_vec_index.db_path)
    cur = connection.cursor()

    # Retrieve all chunk IDs to check for duplicates
    cur.execute(f"SELECT id FROM [{sqlite_vec_index.metadata_table}]")
    chunk_ids = [row[0] for row in cur.fetchall()]
    cur.close()
    connection.close()

    # Ensure all chunk IDs are unique
    assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunk IDs detected across batches!"


@pytest.fixture(scope="session")
async def sqlite_vec_adapter(sqlite_connection):
    config = type("Config", (object,), {"db_path": ":memory:"})  # Mock config with in-memory database
    adapter = SQLiteVecVectorIOAdapter(config=config, inference_api=None)
    await adapter.initialize()
    yield adapter
    await adapter.shutdown()


async def test_query_chunks_hybrid_no_keyword_matches(sqlite_vec_index, sample_chunks, sample_embeddings):
    """Test hybrid search when keyword search returns no matches - should still return vector results."""
    # Create EmbeddedChunk objects
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
    await sqlite_vec_index.add_chunks(embedded_chunks)

    # Use a non-existent keyword but a valid vector query
    query_embedding = sample_embeddings[0]
    query_string = "Sentence 499"

    # First verify keyword search returns no results
    keyword_response = await sqlite_vec_index.query_keyword(query_string, k=5, score_threshold=0.0)
    assert len(keyword_response.chunks) == 0, "Keyword search should return no results"

    # Get hybrid results
    response = await sqlite_vec_index.query_hybrid(
        embedding=query_embedding,
        query_string=query_string,
        k=3,
        score_threshold=0.0,
        reranker_type="rrf",
        reranker_params={"impact_factor": 60.0},
    )

    # Should still get results from vector search
    assert len(response.chunks) > 0, "Should get results from vector search even with no keyword matches"
    # Verify scores are in descending order
    assert all(response.scores[i] >= response.scores[i + 1] for i in range(len(response.scores) - 1))


async def test_query_chunks_hybrid_score_threshold(sqlite_vec_index, sample_chunks, sample_embeddings):
    """Test hybrid search with a high score threshold."""
    # Create EmbeddedChunk objects
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
    await sqlite_vec_index.add_chunks(embedded_chunks)

    # Use a very high score threshold that no results will meet
    query_embedding = sample_embeddings[0]
    query_string = "Sentence 5"

    response = await sqlite_vec_index.query_hybrid(
        embedding=query_embedding,
        query_string=query_string,
        k=3,
        score_threshold=1000.0,  # Very high threshold
        reranker_type="rrf",
        reranker_params={"impact_factor": 60.0},
    )

    # Should return no results due to high threshold
    assert len(response.chunks) == 0


async def test_query_chunks_hybrid_different_embedding(
    sqlite_vec_index, sample_chunks, sample_embeddings, embedding_dimension
):
    """Test hybrid search with a different embedding than the stored ones."""
    # Create EmbeddedChunk objects
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
    await sqlite_vec_index.add_chunks(embedded_chunks)

    # Create a random embedding that's different from stored ones
    query_embedding = np.random.rand(embedding_dimension).astype(np.float32)
    query_string = "Sentence 5"

    response = await sqlite_vec_index.query_hybrid(
        embedding=query_embedding,
        query_string=query_string,
        k=3,
        score_threshold=0.0,
        reranker_type="rrf",
        reranker_params={"impact_factor": 60.0},
    )

    # Should still get results if keyword matches exist
    assert len(response.chunks) > 0
    # Verify scores are in descending order
    assert all(response.scores[i] >= response.scores[i + 1] for i in range(len(response.scores) - 1))


async def test_query_chunks_hybrid_rrf_ranking(sqlite_vec_index, sample_chunks, sample_embeddings):
    """Test that RRF properly combines rankings when documents appear in both search methods."""
    # Create EmbeddedChunk objects
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
    await sqlite_vec_index.add_chunks(embedded_chunks)

    # Create a query embedding that's similar to the first chunk
    query_embedding = sample_embeddings[0]
    # Use a keyword that appears in multiple documents
    query_string = "Sentence 5"

    response = await sqlite_vec_index.query_hybrid(
        embedding=query_embedding,
        query_string=query_string,
        k=5,
        score_threshold=0.0,
        reranker_type="rrf",
        reranker_params={"impact_factor": 60.0},
    )

    # Verify we get results from both search methods
    assert len(response.chunks) > 0
    # Verify scores are in descending order (RRF should maintain this)
    assert all(response.scores[i] >= response.scores[i + 1] for i in range(len(response.scores) - 1))


async def test_query_chunks_hybrid_score_selection(sqlite_vec_index, sample_chunks, sample_embeddings):
    # Create EmbeddedChunk objects
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
    await sqlite_vec_index.add_chunks(embedded_chunks)

    # Create a query embedding that's similar to the first chunk
    query_embedding = sample_embeddings[0]
    # Use a keyword that appears in the first document
    query_string = "Sentence 0 from document 0"

    # Test weighted re-ranking
    response = await sqlite_vec_index.query_hybrid(
        embedding=query_embedding,
        query_string=query_string,
        k=1,
        score_threshold=0.0,
        reranker_type="weighted",
        reranker_params={"alpha": 0.5},
    )
    assert len(response.chunks) == 1
    # Score should be weighted average of normalized keyword score and vector score
    assert response.scores[0] > 0.5  # Both scores should be high

    # Test RRF re-ranking
    response = await sqlite_vec_index.query_hybrid(
        embedding=query_embedding,
        query_string=query_string,
        k=1,
        score_threshold=0.0,
        reranker_type="rrf",
        reranker_params={"impact_factor": 60.0},
    )
    assert len(response.chunks) == 1
    # RRF score should be sum of reciprocal ranks
    assert response.scores[0] == pytest.approx(2.0 / 61.0, rel=1e-6)  # 1/(60+1) + 1/(60+1)

    # Test default re-ranking (should be RRF)
    response = await sqlite_vec_index.query_hybrid(
        embedding=query_embedding,
        query_string=query_string,
        k=1,
        score_threshold=0.0,
        reranker_type="rrf",
        reranker_params={"impact_factor": 60.0},
    )
    assert len(response.chunks) == 1
    assert response.scores[0] == pytest.approx(2.0 / 61.0, rel=1e-6)  # Should behave like RRF


async def test_query_chunks_hybrid_mixed_results(sqlite_vec_index, sample_chunks, sample_embeddings):
    """Test hybrid search with documents that appear in only one search method."""
    # Create EmbeddedChunk objects
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
    await sqlite_vec_index.add_chunks(embedded_chunks)

    # Create a query embedding that's similar to the first chunk
    query_embedding = sample_embeddings[0]
    # Use a keyword that appears in a different document
    query_string = "Sentence 9 from document 2"

    response = await sqlite_vec_index.query_hybrid(
        embedding=query_embedding,
        query_string=query_string,
        k=3,
        score_threshold=0.0,
        reranker_type="rrf",
        reranker_params={"impact_factor": 60.0},
    )

    # Should get results from both search methods
    assert len(response.chunks) > 0
    # Verify scores are in descending order
    assert all(response.scores[i] >= response.scores[i + 1] for i in range(len(response.scores) - 1))
    # Verify we get results from both the vector-similar document and keyword-matched document
    doc_ids = {
        embedded_chunk.metadata.get("document_id") or embedded_chunk.chunk_metadata.document_id
        for embedded_chunk in response.chunks
    }
    assert "document-0" in doc_ids  # From vector search
    assert "document-2" in doc_ids  # From keyword search


async def test_query_chunks_hybrid_weighted_reranker_parametrization(
    sqlite_vec_index, sample_chunks, sample_embeddings
):
    """Test WeightedReRanker with different alpha values."""
    # Re-add data before each search to ensure test isolation
    # Create EmbeddedChunk objects
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
    await sqlite_vec_index.add_chunks(embedded_chunks)
    query_embedding = sample_embeddings[0]
    query_string = "Sentence 0 from document 0"

    # alpha=1.0 (should behave like pure keyword)
    response = await sqlite_vec_index.query_hybrid(
        embedding=query_embedding,
        query_string=query_string,
        k=1,
        score_threshold=0.0,
        reranker_type="weighted",
        reranker_params={"alpha": 1.0},
    )
    assert len(response.chunks) > 0  # Should get at least one result
    assert any(
        "document-0"
        in (
            embedded_chunk.metadata.get("document_id")
            or (embedded_chunk.chunk_metadata.document_id if embedded_chunk.chunk_metadata else "")
        )
        for embedded_chunk in response.chunks
    )

    # alpha=0.0 (should behave like pure vector)
    response = await sqlite_vec_index.query_hybrid(
        embedding=query_embedding,
        query_string=query_string,
        k=1,
        score_threshold=0.0,
        reranker_type="weighted",
        reranker_params={"alpha": 0.0},
    )
    assert len(response.chunks) > 0  # Should get at least one result
    assert any("document-0" in embedded_chunk.metadata["document_id"] for embedded_chunk in response.chunks)

    # Create EmbeddedChunk objects
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
    await sqlite_vec_index.add_chunks(embedded_chunks)
    # alpha=0.7 (should be a mix)
    response = await sqlite_vec_index.query_hybrid(
        embedding=query_embedding,
        query_string=query_string,
        k=1,
        score_threshold=0.0,
        reranker_type="weighted",
        reranker_params={"alpha": 0.7},
    )
    assert len(response.chunks) > 0  # Should get at least one result
    assert any(
        "document-0"
        in (
            embedded_chunk.metadata.get("document_id")
            or (embedded_chunk.chunk_metadata.document_id if embedded_chunk.chunk_metadata else "")
        )
        for embedded_chunk in response.chunks
    )


async def test_query_chunks_hybrid_rrf_impact_factor(sqlite_vec_index, sample_chunks, sample_embeddings):
    """Test RRFReRanker with different impact factors."""
    # Create EmbeddedChunk objects
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
    await sqlite_vec_index.add_chunks(embedded_chunks)
    query_embedding = sample_embeddings[0]
    query_string = "Sentence 0 from document 0"

    # impact_factor=10
    response = await sqlite_vec_index.query_hybrid(
        embedding=query_embedding,
        query_string=query_string,
        k=1,
        score_threshold=0.0,
        reranker_type="rrf",
        reranker_params={"impact_factor": 10.0},
    )
    assert len(response.chunks) == 1
    assert response.scores[0] == pytest.approx(2.0 / 11.0, rel=1e-6)

    # impact_factor=100
    response = await sqlite_vec_index.query_hybrid(
        embedding=query_embedding,
        query_string=query_string,
        k=1,
        score_threshold=0.0,
        reranker_type="rrf",
        reranker_params={"impact_factor": 100.0},
    )
    assert len(response.chunks) == 1
    assert response.scores[0] == pytest.approx(2.0 / 101.0, rel=1e-6)


async def test_query_chunks_hybrid_edge_cases(sqlite_vec_index, sample_chunks, sample_embeddings):
    # Create EmbeddedChunk objects
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
    await sqlite_vec_index.add_chunks(embedded_chunks)

    # No results from either search - use a completely different embedding and a nonzero threshold
    query_embedding = np.ones_like(sample_embeddings[0]) * -1  # Very different from sample embeddings
    query_string = "no_such_keyword_that_will_never_match"
    response = await sqlite_vec_index.query_hybrid(
        embedding=query_embedding,
        query_string=query_string,
        k=3,
        score_threshold=0.1,  # Nonzero threshold to filter out low-similarity matches
        reranker_type="rrf",
        reranker_params={"impact_factor": 60.0},
    )
    assert len(response.chunks) == 0

    # All results below threshold
    query_embedding = sample_embeddings[0]
    query_string = "Sentence 0 from document 0"
    response = await sqlite_vec_index.query_hybrid(
        embedding=query_embedding,
        query_string=query_string,
        k=3,
        score_threshold=1000.0,
        reranker_type="rrf",
        reranker_params={"impact_factor": 60.0},
    )
    assert len(response.chunks) == 0

    # Large k value
    response = await sqlite_vec_index.query_hybrid(
        embedding=query_embedding,
        query_string=query_string,
        k=100,
        score_threshold=0.0,
        reranker_type="rrf",
        reranker_params={"impact_factor": 60.0},
    )
    # Should not error, should return all available results
    assert len(response.chunks) > 0
    assert len(response.chunks) <= 100


async def test_query_chunks_hybrid_tie_breaking(
    sqlite_vec_index, sample_embeddings, embedding_dimension, tmp_path_factory
):
    """Test tie-breaking and determinism when scores are equal."""
    from ogx.providers.utils.vector_io.vector_utils import generate_chunk_id

    # Create two chunks with the same content and embedding
    chunk1_id = generate_chunk_id("docA", "identical")
    chunk2_id = generate_chunk_id("docB", "identical")

    chunk1 = Chunk(
        content="identical",
        chunk_id=chunk1_id,
        metadata={"document_id": "docA"},
        chunk_metadata=ChunkMetadata(
            document_id="docA",
            chunk_id=chunk1_id,
            created_timestamp=int(time.time()),
            updated_timestamp=int(time.time()),
            content_token_count=1,
        ),
    )
    chunk2 = Chunk(
        content="identical",
        chunk_id=chunk2_id,
        metadata={"document_id": "docB"},
        chunk_metadata=ChunkMetadata(
            document_id="docB",
            chunk_id=chunk2_id,
            created_timestamp=int(time.time()),
            updated_timestamp=int(time.time()),
            content_token_count=1,
        ),
    )
    # Use the same embedding for both chunks to ensure equal scores
    same_embedding = sample_embeddings[0]

    # Create EmbeddedChunk objects
    embedded_chunks = [
        EmbeddedChunk(
            content=chunk1.content,
            chunk_id=chunk1.chunk_id,
            metadata=chunk1.metadata,
            chunk_metadata=chunk1.chunk_metadata,
            embedding=same_embedding.tolist(),
            embedding_model="test-embedding-model",
            embedding_dimension=len(same_embedding),
        ),
        EmbeddedChunk(
            content=chunk2.content,
            chunk_id=chunk2.chunk_id,
            metadata=chunk2.metadata,
            chunk_metadata=chunk2.chunk_metadata,
            embedding=same_embedding.tolist(),
            embedding_model="test-embedding-model",
            embedding_dimension=len(same_embedding),
        ),
    ]

    # Clear existing data and recreate index
    await sqlite_vec_index.delete()
    temp_dir = tmp_path_factory.getbasetemp()
    db_path = str(temp_dir / "test_sqlite.db")
    sqlite_vec_index = await SQLiteVecIndex.create(dimension=embedding_dimension, db_path=db_path, bank_id="test_bank")
    await sqlite_vec_index.add_chunks(embedded_chunks)

    # Query with the same embedding and content to ensure equal scores
    query_embedding = same_embedding
    query_string = "identical"

    # Run multiple queries to verify determinism
    responses = []
    for _ in range(3):
        response = await sqlite_vec_index.query_hybrid(
            embedding=query_embedding,
            query_string=query_string,
            k=2,
            score_threshold=0.0,
            reranker_type="rrf",
            reranker_params={"impact_factor": 60.0},
        )
        responses.append(response)

    # Verify all responses are identical
    first_response = responses[0]
    for response in responses[1:]:
        assert response.chunks == first_response.chunks
        assert response.scores == first_response.scores

    # Verify both chunks are returned with equal scores
    assert len(first_response.chunks) == 2
    assert first_response.scores[0] == first_response.scores[1]
    assert {embedded_chunk.metadata["document_id"] for embedded_chunk in first_response.chunks} == {
        "docA",
        "docB",
    }
