# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import base64
import mimetypes
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from tiktoken import get_encoding

from ogx.core.datatypes import RerankerModel, VectorStoresConfig
from ogx.providers.utils.memory.vector_store import (
    VectorStoreWithIndex,
    _validate_embedding,
    make_overlapped_chunks,
)
from ogx.providers.utils.vector_io.vector_utils import generate_chunk_id
from ogx_api import (
    Chunk,
    ChunkMetadata,
    EmbeddedChunk,
    InsertChunksRequest,
    QueryChunksRequest,
    QueryChunksResponse,
    RerankData,
    RerankResponse,
)

DUMMY_PDF_PATH = Path(os.path.abspath(__file__)).parent / "fixtures" / "dummy.pdf"
# Depending on the machine, this can get parsed a couple of ways
DUMMY_PDF_TEXT_CHOICES = ["Dummy PDF file", "Dumm y PDF file"]


def read_file(file_path: str) -> bytes:
    with open(file_path, "rb") as file:
        return file.read()


def data_url_from_file(file_path: str) -> str:
    with open(file_path, "rb") as file:
        file_content = file.read()

    base64_content = base64.b64encode(file_content).decode("utf-8")
    mime_type, _ = mimetypes.guess_type(file_path)

    data_url = f"data:{mime_type};base64,{base64_content}"

    return data_url


class TestChunk:
    def test_chunk(self):
        chunk = Chunk(
            content="Example chunk content",
            chunk_id=generate_chunk_id("test-doc", "Example chunk content"),
            metadata={"key": "value"},
            chunk_metadata=ChunkMetadata(
                document_id="test-doc",
                chunk_id=generate_chunk_id("test-doc", "Example chunk content"),
                created_timestamp=1234567890,
                updated_timestamp=1234567890,
                content_token_count=3,
            ),
        )

        assert chunk.content == "Example chunk content"
        assert chunk.metadata == {"key": "value"}

    def test_embedded_chunk(self):
        chunk = Chunk(
            content="Example chunk content",
            chunk_id=generate_chunk_id("test-doc", "Example chunk content"),
            metadata={"key": "value"},
            chunk_metadata=ChunkMetadata(
                document_id="test-doc",
                chunk_id=generate_chunk_id("test-doc", "Example chunk content"),
                created_timestamp=1234567890,
                updated_timestamp=1234567890,
                content_token_count=3,
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

        assert embedded_chunk.content == "Example chunk content"
        assert embedded_chunk.metadata == {"key": "value"}
        assert embedded_chunk.embedding == [0.1, 0.2, 0.3]
        assert embedded_chunk.embedding_model == "test-model"
        assert embedded_chunk.embedding_dimension == 3


class TestValidateEmbedding:
    def test_valid_list_embeddings(self):
        _validate_embedding([0.1, 0.2, 0.3], 0, 3)
        _validate_embedding([1, 2, 3], 1, 3)
        _validate_embedding([0.1, 2, 3.5], 2, 3)

    def test_valid_numpy_embeddings(self):
        _validate_embedding(np.array([0.1, 0.2, 0.3], dtype=np.float32), 0, 3)
        _validate_embedding(np.array([0.1, 0.2, 0.3], dtype=np.float64), 1, 3)
        _validate_embedding(np.array([1, 2, 3], dtype=np.int32), 2, 3)
        _validate_embedding(np.array([1, 2, 3], dtype=np.int64), 3, 3)

    def test_invalid_embedding_type(self):
        error_msg = "must be a list or numpy array"

        with pytest.raises(ValueError, match=error_msg):
            _validate_embedding("not a list", 0, 3)

        with pytest.raises(ValueError, match=error_msg):
            _validate_embedding(None, 1, 3)

        with pytest.raises(ValueError, match=error_msg):
            _validate_embedding(42, 2, 3)

    def test_non_numeric_values(self):
        error_msg = "contains non-numeric values"

        with pytest.raises(ValueError, match=error_msg):
            _validate_embedding([0.1, "string", 0.3], 0, 3)

        with pytest.raises(ValueError, match=error_msg):
            _validate_embedding([0.1, None, 0.3], 1, 3)

        with pytest.raises(ValueError, match=error_msg):
            _validate_embedding([1, {}, 3], 2, 3)

    def test_wrong_dimension(self):
        with pytest.raises(ValueError, match="has dimension 4, expected 3"):
            _validate_embedding([0.1, 0.2, 0.3, 0.4], 0, 3)

        with pytest.raises(ValueError, match="has dimension 2, expected 3"):
            _validate_embedding([0.1, 0.2], 1, 3)

        with pytest.raises(ValueError, match="has dimension 0, expected 3"):
            _validate_embedding([], 2, 3)


class TestVectorStore:
    @pytest.mark.parametrize(
        "window_len, overlap_len, expected_chunks",
        [
            (5, 2, 4),  # Create 4 chunks with window of 5 and overlap of 2
            (4, 1, 4),  # Create 4 chunks with window of 4 and overlap of 1
        ],
    )
    def test_make_overlapped_chunks(self, window_len, overlap_len, expected_chunks):
        document_id = "test_doc_123"
        text = "This is a sample document for testing the chunking behavior"
        original_metadata = {"source": "test", "date": "2023-01-01", "author": "llama"}
        encoding = get_encoding("cl100k_base")
        len_metadata_tokens = len(encoding.encode(str(original_metadata)))

        expected_tokenizer_name = "tiktoken:cl100k_base"
        chunks = make_overlapped_chunks(document_id, text, window_len, overlap_len, original_metadata)

        assert len(chunks) == expected_chunks

        # Check that each chunk has the right metadata
        for chunk in chunks:
            # Original metadata should be preserved
            assert chunk.metadata["source"] == "test"
            assert chunk.metadata["date"] == "2023-01-01"
            assert chunk.metadata["author"] == "llama"

            # New metadata should be added
            assert chunk.metadata["document_id"] == document_id
            assert "token_count" in chunk.metadata
            assert isinstance(chunk.metadata["token_count"], int)
            assert chunk.metadata["token_count"] > 0
            assert chunk.metadata["metadata_token_count"] == len_metadata_tokens
            assert chunk.chunk_metadata.chunk_tokenizer == expected_tokenizer_name

    def test_raise_overlapped_chunks_metadata_serialization_error(self):
        document_id = "test_doc_ex"
        text = "Some text"
        window_len = 5
        overlap_len = 2

        class BadMetadata:
            def __repr__(self):
                raise TypeError("Cannot convert to string")

        problematic_metadata = {"bad_metadata_example": BadMetadata()}

        with pytest.raises(ValueError) as excinfo:
            make_overlapped_chunks(document_id, text, window_len, overlap_len, problematic_metadata)

        assert str(excinfo.value) == "Failed to serialize metadata to string"
        assert isinstance(excinfo.value.__cause__, TypeError)
        assert str(excinfo.value.__cause__) == "Cannot convert to string"


class TestVectorStoreWithIndex:
    async def test_insert_chunks_with_embedded_chunks(self):
        """Test that VectorStoreWithIndex.insert_chunks() works with EmbeddedChunk objects."""
        mock_vector_store = MagicMock()
        mock_vector_store.embedding_model = "test-embedding-model"
        mock_vector_store.embedding_dimension = 3
        mock_index = AsyncMock()
        mock_inference_api = AsyncMock()

        vector_store_with_index = VectorStoreWithIndex(
            vector_store=mock_vector_store, index=mock_index, inference_api=mock_inference_api
        )

        chunk = Chunk(
            content="Test 1",
            chunk_id=generate_chunk_id("test-doc", "Test 1"),
            metadata={},
            chunk_metadata=ChunkMetadata(
                document_id="test-doc",
                chunk_id=generate_chunk_id("test-doc", "Test 1"),
                created_timestamp=1234567890,
                updated_timestamp=1234567890,
                content_token_count=2,
            ),
        )

        embedded_chunks = [
            EmbeddedChunk(
                content=chunk.content,
                chunk_id=chunk.chunk_id,
                metadata=chunk.metadata,
                chunk_metadata=chunk.chunk_metadata,
                embedding=[0.1, 0.2, 0.3],
                embedding_model="test-embedding-model",
                embedding_dimension=3,
            )
        ]

        await vector_store_with_index.insert_chunks(
            InsertChunksRequest(
                vector_store_id="mock_vs_id",
                chunks=embedded_chunks,
            )
        )

        # Verify inference API was NOT called since we already have embeddings
        mock_inference_api.openai_embeddings.assert_not_called()
        # Verify index was called with the EmbeddedChunk objects we provided
        mock_index.add_chunks.assert_called_once_with(embedded_chunks)

    async def test_insert_chunks_with_multiple_embedded_chunks(self):
        """Test that VectorStoreWithIndex.insert_chunks() works with multiple EmbeddedChunk objects."""
        mock_vector_store = MagicMock()
        mock_vector_store.embedding_model = "test-embedding-model"
        mock_vector_store.embedding_dimension = 3
        mock_index = AsyncMock()
        mock_inference_api = AsyncMock()

        vector_store_with_index = VectorStoreWithIndex(
            vector_store=mock_vector_store, index=mock_index, inference_api=mock_inference_api
        )

        chunks = [
            Chunk(
                content="Test 1",
                chunk_id=generate_chunk_id("test-doc", "Test 1"),
                metadata={},
                chunk_metadata=ChunkMetadata(
                    document_id="test-doc",
                    chunk_id=generate_chunk_id("test-doc", "Test 1"),
                    created_timestamp=1234567890,
                    updated_timestamp=1234567890,
                    content_token_count=2,
                ),
            ),
            Chunk(
                content="Test 2",
                chunk_id=generate_chunk_id("test-doc", "Test 2"),
                metadata={},
                chunk_metadata=ChunkMetadata(
                    document_id="test-doc",
                    chunk_id=generate_chunk_id("test-doc", "Test 2"),
                    created_timestamp=1234567890,
                    updated_timestamp=1234567890,
                    content_token_count=2,
                ),
            ),
        ]

        embedded_chunks = [
            EmbeddedChunk(
                content=chunks[0].content,
                chunk_id=chunks[0].chunk_id,
                metadata=chunks[0].metadata,
                chunk_metadata=chunks[0].chunk_metadata,
                embedding=[0.1, 0.2, 0.3],
                embedding_model="test-embedding-model",
                embedding_dimension=3,
            ),
            EmbeddedChunk(
                content=chunks[1].content,
                chunk_id=chunks[1].chunk_id,
                metadata=chunks[1].metadata,
                chunk_metadata=chunks[1].chunk_metadata,
                embedding=[0.4, 0.5, 0.6],
                embedding_model="test-embedding-model",
                embedding_dimension=3,
            ),
        ]

        await vector_store_with_index.insert_chunks(
            InsertChunksRequest(
                vector_store_id="mock_vs_id",
                chunks=embedded_chunks,
            )
        )

        # Verify inference API was NOT called since we already have embeddings
        mock_inference_api.openai_embeddings.assert_not_called()
        # Verify index was called with the EmbeddedChunk objects we provided
        mock_index.add_chunks.assert_called_once_with(embedded_chunks)


class TestNeuralRerank:
    def _make_vector_store_with_index(self, vector_stores_config=None):
        mock_vector_store = MagicMock()
        mock_vector_store.embedding_model = "test-embedding-model"
        mock_vector_store.embedding_dimension = 3
        mock_index = AsyncMock()
        mock_inference_api = AsyncMock()
        mock_embedding_response = MagicMock()
        mock_embedding_data = MagicMock()
        mock_embedding_data.embedding = [0.1, 0.2, 0.3]
        mock_embedding_response.data = [mock_embedding_data]
        mock_inference_api.openai_embeddings.return_value = mock_embedding_response

        return VectorStoreWithIndex(
            vector_store=mock_vector_store,
            index=mock_index,
            inference_api=mock_inference_api,
            vector_stores_config=vector_stores_config,
        )

    def _make_query_chunks_response(self, chunks_data):
        chunks = []
        scores = []
        for content, score, doc_id in chunks_data:
            chunk_id = generate_chunk_id(doc_id, content)
            chunks.append(
                EmbeddedChunk(
                    content=content,
                    chunk_id=chunk_id,
                    metadata={"document_id": doc_id},
                    chunk_metadata=ChunkMetadata(
                        document_id=doc_id,
                        chunk_id=chunk_id,
                        created_timestamp=1234567890,
                        updated_timestamp=1234567890,
                        content_token_count=len(content.split()),
                    ),
                    embedding=[0.1, 0.2, 0.3],
                    embedding_model="test-embedding-model",
                    embedding_dimension=3,
                )
            )
            scores.append(score)
        return QueryChunksResponse(chunks=chunks, scores=scores)

    async def test_neural_rerank_reorders_chunks(self):
        config = VectorStoresConfig(
            default_reranker_model=RerankerModel(
                provider_id="transformers",
                model_id="Qwen/Qwen3-Reranker-0.6B",
            ),
        )
        store = self._make_vector_store_with_index(vector_stores_config=config)

        # Initial retrieval returns chunks ordered by vector similarity
        initial_response = self._make_query_chunks_response(
            [
                ("Python is a programming language", 2.5, "doc-1"),
                ("The capital of Ireland is Dublin", 2.0, "doc-2"),
                ("Machine learning is a subset of AI", 1.5, "doc-3"),
            ]
        )
        store.index.query_vector.return_value = initial_response

        # Mock rerank to return doc-2 as most relevant (reordering)
        store.inference_api.rerank.return_value = RerankResponse(
            data=[
                RerankData(index=1, relevance_score=0.95),
                RerankData(index=0, relevance_score=0.30),
                RerankData(index=2, relevance_score=0.05),
            ]
        )

        request = QueryChunksRequest(
            vector_store_id="test-store",
            query="What is the capital of Ireland?",
            params={
                "reranker_type": "neural",
                "reranker_params": {},
            },
        )

        result = await store.query_chunks(request)

        # Verify rerank was called
        store.inference_api.rerank.assert_called_once()

        # Verify chunks are reordered by neural relevance
        assert len(result.chunks) == 3
        assert result.chunks[0].metadata["document_id"] == "doc-2"
        assert result.chunks[1].metadata["document_id"] == "doc-1"
        assert result.chunks[2].metadata["document_id"] == "doc-3"

        # Verify scores are neural relevance scores
        assert result.scores[0] == pytest.approx(0.95)
        assert result.scores[1] == pytest.approx(0.30)
        assert result.scores[2] == pytest.approx(0.05)

    async def test_neural_rerank_uses_default_model_from_config(self):
        config = VectorStoresConfig(
            default_reranker_model=RerankerModel(
                provider_id="transformers",
                model_id="Qwen/Qwen3-Reranker-0.6B",
            ),
        )
        store = self._make_vector_store_with_index(vector_stores_config=config)

        initial_response = self._make_query_chunks_response(
            [
                ("Test chunk", 1.0, "doc-1"),
            ]
        )
        store.index.query_vector.return_value = initial_response
        store.inference_api.rerank.return_value = RerankResponse(
            data=[
                RerankData(index=0, relevance_score=0.9),
            ]
        )

        request = QueryChunksRequest(
            vector_store_id="test-store",
            query="test query",
            params={
                "reranker_type": "neural",
                "reranker_params": {},
            },
        )

        await store.query_chunks(request)

        # Verify rerank was called with the fully qualified model ID
        rerank_request = store.inference_api.rerank.call_args.args[0]
        assert rerank_request.model == "transformers/Qwen/Qwen3-Reranker-0.6B"

    async def test_neural_rerank_uses_model_from_params(self):
        config = VectorStoresConfig(
            default_reranker_model=RerankerModel(
                provider_id="transformers",
                model_id="Qwen/Qwen3-Reranker-0.6B",
            ),
        )
        store = self._make_vector_store_with_index(vector_stores_config=config)

        initial_response = self._make_query_chunks_response(
            [
                ("Test chunk", 1.0, "doc-1"),
            ]
        )
        store.index.query_vector.return_value = initial_response
        store.inference_api.rerank.return_value = RerankResponse(
            data=[
                RerankData(index=0, relevance_score=0.9),
            ]
        )

        request = QueryChunksRequest(
            vector_store_id="test-store",
            query="test query",
            params={
                "reranker_type": "neural",
                "reranker_params": {"model": "custom/reranker-model"},
            },
        )

        await store.query_chunks(request)

        # Verify rerank was called with the param model, not the config default
        rerank_request = store.inference_api.rerank.call_args.args[0]
        assert rerank_request.model == "custom/reranker-model"

    async def test_neural_rerank_respects_max_chunks_limit(self):
        config = VectorStoresConfig(
            default_reranker_model=RerankerModel(
                provider_id="transformers",
                model_id="Qwen/Qwen3-Reranker-0.6B",
            ),
        )
        store = self._make_vector_store_with_index(vector_stores_config=config)

        initial_response = self._make_query_chunks_response(
            [
                ("Chunk A", 3.0, "doc-1"),
                ("Chunk B", 2.0, "doc-2"),
                ("Chunk C", 1.0, "doc-3"),
            ]
        )
        store.index.query_vector.return_value = initial_response

        # Rerank returns only top 2 (max_num_results=2 passed to rerank)
        store.inference_api.rerank.return_value = RerankResponse(
            data=[
                RerankData(index=2, relevance_score=0.99),
                RerankData(index=0, relevance_score=0.50),
            ]
        )

        request = QueryChunksRequest(
            vector_store_id="test-store",
            query="test query",
            params={
                "max_chunks": 2,
                "reranker_type": "neural",
                "reranker_params": {},
            },
        )

        result = await store.query_chunks(request)

        # Should only return top 2 after reranking
        assert len(result.chunks) == 2
        assert result.chunks[0].metadata["document_id"] == "doc-3"
        assert result.chunks[1].metadata["document_id"] == "doc-1"

        # Verify max_num_results was passed to rerank
        rerank_request = store.inference_api.rerank.call_args.args[0]
        assert rerank_request.max_num_results == 2
