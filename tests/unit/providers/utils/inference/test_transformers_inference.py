# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from ogx.providers.inline.inference.transformers.transformers import (
    DEFAULT_RERANKER_INSTRUCTION,
    TransformersInferenceImpl,
)
from ogx_api import (
    OpenAIChatCompletionContentPartImageParam,
    OpenAIChatCompletionContentPartTextParam,
)
from ogx_api.inference import RerankRequest


class ConcreteRerankerImpl(TransformersInferenceImpl):
    def __init__(self):
        self.model_store = MagicMock()


class TestExtractText:
    def test_extract_text_from_string(self):
        reranker = ConcreteRerankerImpl()
        assert reranker.extract_text("hello world") == "hello world"

    def test_extract_text_from_text_content_part(self):
        reranker = ConcreteRerankerImpl()
        text_part = OpenAIChatCompletionContentPartTextParam(text="test text", type="text")
        assert reranker.extract_text(text_part) == "test text"

    def test_extract_text_from_image_raises(self):
        reranker = ConcreteRerankerImpl()
        image_part = MagicMock(spec=OpenAIChatCompletionContentPartImageParam)
        with pytest.raises(ValueError, match="Unsupported content type for reranking"):
            reranker.extract_text(image_part)


class TestFormatInstruction:
    def test_format_instruction_basic(self):
        reranker = ConcreteRerankerImpl()
        result = reranker.format_instruction("Retrieve relevant passages", "What is Python?", "Python is a language.")
        assert "<Instruct>: Retrieve relevant passages" in result
        assert "<Query>: What is Python?" in result
        assert "<Document>: Python is a language." in result

    def test_format_instruction_default_instruction(self):
        reranker = ConcreteRerankerImpl()
        result = reranker.format_instruction(DEFAULT_RERANKER_INSTRUCTION, "query", "doc")
        assert "<Instruct>: Given the search query, retrieve relevant passages that answer the query" in result


class TestRerank:
    def test_rerank_empty_items_rejected_by_validation(self):
        with pytest.raises(ValidationError, match="items"):
            RerankRequest(model="test-model", query="test query", items=[])

    @pytest.mark.parametrize("max_num_results", [0, -1])
    def test_rerank_invalid_max_num_results_rejected_by_validation(self, max_num_results):
        with pytest.raises(ValidationError, match="max_num_results"):
            RerankRequest(model="test-model", query="test query", items=["some item"], max_num_results=max_num_results)

    @patch.object(TransformersInferenceImpl, "load_reranker_model")
    @patch.object(TransformersInferenceImpl, "compute_reranked_scores")
    async def test_rerank_returns_sorted_results(self, mock_compute, mock_load):
        reranker = ConcreteRerankerImpl()

        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_load.return_value = (mock_tokenizer, mock_model)
        mock_compute.return_value = [0.3, 0.1, 0.9]

        request = RerankRequest(
            model="test-model",
            query="test query",
            items=[
                "Python is a programming language",
                "The capital of Ireland is Dublin",
                "Machine learning is a subset of AI",
            ],
        )

        result = await reranker.rerank(request)

        # Results should be sorted by score descending
        assert len(result.data) == 3
        assert result.data[0].index == 2
        assert result.data[0].relevance_score == pytest.approx(0.9)
        assert result.data[1].index == 0
        assert result.data[1].relevance_score == pytest.approx(0.3)
        assert result.data[2].index == 1
        assert result.data[2].relevance_score == pytest.approx(0.1)

    @patch.object(TransformersInferenceImpl, "load_reranker_model")
    @patch.object(TransformersInferenceImpl, "compute_reranked_scores")
    async def test_rerank_respects_max_num_results(self, mock_compute, mock_load):
        reranker = ConcreteRerankerImpl()

        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_load.return_value = (mock_tokenizer, mock_model)
        mock_compute.return_value = [0.5, 0.9, 0.1, 0.7]

        request = RerankRequest(
            model="test-model",
            query="test query",
            items=["a", "b", "c", "d"],
            max_num_results=2,
        )

        result = await reranker.rerank(request)

        # Should only return top 2
        assert len(result.data) == 2
        assert result.data[0].index == 1  # score 0.9
        assert result.data[1].index == 3  # score 0.7

    @patch.object(TransformersInferenceImpl, "load_reranker_model")
    @patch.object(TransformersInferenceImpl, "compute_reranked_scores")
    async def test_rerank_builds_correct_pairs(self, mock_compute, mock_load):
        reranker = ConcreteRerankerImpl()

        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_load.return_value = (mock_tokenizer, mock_model)
        mock_compute.return_value = [0.5, 0.5]

        request = RerankRequest(
            model="test-model",
            query="What is AI?",
            items=["AI is intelligence", "Cars are fast"],
        )

        await reranker.rerank(request)

        call_args = mock_compute.call_args[0]
        pairs = call_args[2]
        assert len(pairs) == 2
        assert "<Query>: What is AI?" in pairs[0]
        assert "<Document>: AI is intelligence" in pairs[0]
        assert "<Document>: Cars are fast" in pairs[1]
