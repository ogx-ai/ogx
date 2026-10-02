# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Tests for the classifier reranker type in the search pipeline."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ogx.providers.utils.memory.vector_store import (
    RERANKER_TYPE_CLASSIFIER,
    VectorStoreWithIndex,
)
from ogx_api import ChunkMetadata, EmbeddedChunk, QueryChunksResponse


def _make_chunk(content: str, chunk_id: str = "c1") -> EmbeddedChunk:
    return EmbeddedChunk(
        content=content,
        chunk_id=chunk_id,
        metadata={"document_id": chunk_id},
        chunk_metadata=ChunkMetadata(document_id=chunk_id, chunk_id=chunk_id),
        embedding=[0.1, 0.2, 0.3],
        embedding_model="test",
        embedding_dimension=3,
    )


class TestClassifierRerankerConstant:
    def test_classifier_constant_defined(self):
        assert RERANKER_TYPE_CLASSIFIER == "classifier"


class TestApplyClassifierRerank:
    @pytest.fixture
    def mock_store(self):
        store = MagicMock(spec=VectorStoreWithIndex)
        store.inference_api = AsyncMock()
        store.vector_stores_config = None
        store.apply_classifier_rerank = VectorStoreWithIndex.apply_classifier_rerank.__get__(store)
        store._extract_chunk_texts = VectorStoreWithIndex._extract_chunk_texts
        return store

    async def test_filters_below_confidence_threshold(self, mock_store):
        chunks = [_make_chunk("high quality", "c1"), _make_chunk("low quality", "c2")]
        response = QueryChunksResponse(chunks=chunks, scores=[0.9, 0.3])

        rerank_data = MagicMock()
        rerank_data.data = [
            MagicMock(index=0, relevance_score=0.85),
            MagicMock(index=1, relevance_score=0.2),
        ]
        mock_store.inference_api.rerank.return_value = rerank_data

        result = await mock_store.apply_classifier_rerank(
            "test query", response, 5, {"model": "test-classifier", "confidence_threshold": 0.5}
        )

        assert len(result.chunks) == 1
        assert result.chunks[0].content == "high quality"
        assert result.scores[0] == 0.85

    async def test_keeps_all_above_threshold(self, mock_store):
        chunks = [_make_chunk("a", "c1"), _make_chunk("b", "c2")]
        response = QueryChunksResponse(chunks=chunks, scores=[0.9, 0.8])

        rerank_data = MagicMock()
        rerank_data.data = [
            MagicMock(index=0, relevance_score=0.9),
            MagicMock(index=1, relevance_score=0.8),
        ]
        mock_store.inference_api.rerank.return_value = rerank_data

        result = await mock_store.apply_classifier_rerank(
            "test query", response, 5, {"model": "test-classifier", "confidence_threshold": 0.1}
        )

        assert len(result.chunks) == 2

    async def test_zero_threshold_keeps_all(self, mock_store):
        chunks = [_make_chunk("a", "c1"), _make_chunk("b", "c2")]
        response = QueryChunksResponse(chunks=chunks, scores=[0.5, 0.1])

        rerank_data = MagicMock()
        rerank_data.data = [
            MagicMock(index=0, relevance_score=0.5),
            MagicMock(index=1, relevance_score=0.1),
        ]
        mock_store.inference_api.rerank.return_value = rerank_data

        result = await mock_store.apply_classifier_rerank(
            "test query", response, 5, {"model": "test-classifier", "confidence_threshold": 0.0}
        )

        assert len(result.chunks) == 2

    async def test_no_model_returns_original(self, mock_store):
        chunks = [_make_chunk("a", "c1")]
        response = QueryChunksResponse(chunks=chunks, scores=[0.5])

        result = await mock_store.apply_classifier_rerank("test query", response, 5, {})

        assert len(result.chunks) == 1
        mock_store.inference_api.rerank.assert_not_called()

    async def test_inference_error_returns_original(self, mock_store):
        chunks = [_make_chunk("a", "c1")]
        response = QueryChunksResponse(chunks=chunks, scores=[0.5])
        mock_store.inference_api.rerank.side_effect = RuntimeError("model unavailable")

        result = await mock_store.apply_classifier_rerank("test query", response, 5, {"model": "bad-model"})

        assert len(result.chunks) == 1
        assert result.scores[0] == 0.5

    async def test_calls_rerank_with_correct_model(self, mock_store):
        chunks = [_make_chunk("content", "c1")]
        response = QueryChunksResponse(chunks=chunks, scores=[0.5])

        rerank_data = MagicMock()
        rerank_data.data = [MagicMock(index=0, relevance_score=0.9)]
        mock_store.inference_api.rerank.return_value = rerank_data

        await mock_store.apply_classifier_rerank("my query", response, 5, {"model": "my-org/quality-classifier"})

        call_args = mock_store.inference_api.rerank.call_args[0][0]
        assert call_args.model == "my-org/quality-classifier"
        assert call_args.query == "my query"

    async def test_out_of_bounds_index_ignored(self, mock_store):
        chunks = [_make_chunk("only one", "c1")]
        response = QueryChunksResponse(chunks=chunks, scores=[0.5])

        rerank_data = MagicMock()
        rerank_data.data = [
            MagicMock(index=0, relevance_score=0.9),
            MagicMock(index=99, relevance_score=0.8),
        ]
        mock_store.inference_api.rerank.return_value = rerank_data

        result = await mock_store.apply_classifier_rerank("query", response, 5, {"model": "test-model"})

        assert len(result.chunks) == 1
        assert result.scores[0] == 0.9

    async def test_all_filtered_returns_empty(self, mock_store):
        chunks = [_make_chunk("low", "c1"), _make_chunk("also low", "c2")]
        response = QueryChunksResponse(chunks=chunks, scores=[0.3, 0.2])

        rerank_data = MagicMock()
        rerank_data.data = [
            MagicMock(index=0, relevance_score=0.1),
            MagicMock(index=1, relevance_score=0.05),
        ]
        mock_store.inference_api.rerank.return_value = rerank_data

        result = await mock_store.apply_classifier_rerank(
            "query", response, 5, {"model": "test-model", "confidence_threshold": 0.5}
        )

        assert len(result.chunks) == 0
        assert len(result.scores) == 0
