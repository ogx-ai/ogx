# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from ogx.providers.utils.memory.vector_store import RERANKER_TYPE_RRF, RERANKER_TYPE_WEIGHTED
from ogx.providers.utils.vector_io.vector_utils import WeightedInMemoryAggregator


class TestNormalizeScores:
    """Test cases for score normalization."""

    def test_normalize_scores_basic(self):
        """Test basic score normalization."""
        scores = {"doc1": 10.0, "doc2": 5.0, "doc3": 0.0}
        normalized = WeightedInMemoryAggregator._normalize_scores(scores)

        assert normalized["doc1"] == 1.0  # Max score
        assert normalized["doc3"] == 0.0  # Min score
        assert normalized["doc2"] == 0.5  # Middle score
        assert all(0 <= score <= 1 for score in normalized.values())

    def test_normalize_scores_identical(self):
        """Test normalization when all scores are identical."""
        scores = {"doc1": 5.0, "doc2": 5.0, "doc3": 5.0}
        normalized = WeightedInMemoryAggregator._normalize_scores(scores)

        # All scores should be 1.0 when identical
        assert all(score == 1.0 for score in normalized.values())

    def test_normalize_scores_empty(self):
        """Test normalization with empty scores."""
        scores = {}
        normalized = WeightedInMemoryAggregator._normalize_scores(scores)

        assert normalized == {}

    def test_normalize_scores_single(self):
        """Test normalization with single score."""
        scores = {"doc1": 7.5}
        normalized = WeightedInMemoryAggregator._normalize_scores(scores)

        assert normalized["doc1"] == 1.0


class TestWeightedRerank:
    """Test cases for weighted reranking."""

    def test_weighted_rerank_basic(self):
        """Test basic weighted reranking."""
        vector_scores = {"doc1": 0.9, "doc2": 0.7, "doc3": 0.5}
        keyword_scores = {"doc1": 0.6, "doc2": 0.8, "doc4": 0.9}

        combined = WeightedInMemoryAggregator.weighted_rerank(vector_scores, keyword_scores, alpha=0.5)

        # Should include all documents
        expected_docs = {"doc1", "doc2", "doc3", "doc4"}
        assert set(combined.keys()) == expected_docs

        # All scores should be between 0 and 1
        assert all(0 <= score <= 1 for score in combined.values())

        # doc1 appears in both searches, should have higher combined score
        assert combined["doc1"] > 0

    def test_weighted_rerank_alpha_zero(self):
        """Test weighted reranking with alpha=0 (keyword only)."""
        vector_scores = {"doc1": 0.9, "doc2": 0.7, "doc3": 0.5}  # All docs present in vector
        keyword_scores = {"doc1": 0.1, "doc2": 0.3, "doc3": 0.9}  # All docs present in keyword

        combined = WeightedInMemoryAggregator.weighted_rerank(vector_scores, keyword_scores, alpha=0.0)

        # Alpha=0 means vector scores are ignored, keyword scores dominate
        # doc3 should score highest since it has highest keyword score
        assert combined["doc3"] > combined["doc2"] > combined["doc1"]

    def test_weighted_rerank_alpha_one(self):
        """Test weighted reranking with alpha=1 (vector only)."""
        vector_scores = {"doc1": 0.9, "doc2": 0.7, "doc3": 0.5}  # All docs present in vector
        keyword_scores = {"doc1": 0.1, "doc2": 0.3, "doc3": 0.9}  # All docs present in keyword

        combined = WeightedInMemoryAggregator.weighted_rerank(vector_scores, keyword_scores, alpha=1.0)

        # Alpha=1 means keyword scores are ignored, vector scores dominate
        # doc1 should score highest since it has highest vector score
        assert combined["doc1"] > combined["doc2"] > combined["doc3"]

    def test_weighted_rerank_no_overlap(self):
        """Test weighted reranking with no overlapping documents."""
        vector_scores = {"doc1": 0.9, "doc2": 0.7}
        keyword_scores = {"doc3": 0.8, "doc4": 0.6}

        combined = WeightedInMemoryAggregator.weighted_rerank(vector_scores, keyword_scores, alpha=0.5)

        assert len(combined) == 4
        # With min-max normalization, lowest scoring docs in each group get 0.0
        # but highest scoring docs should get positive scores
        assert all(score >= 0 for score in combined.values())
        assert combined["doc1"] > 0  # highest vector score
        assert combined["doc3"] > 0  # highest keyword score


class TestRRFRerank:
    """Test cases for RRF (Reciprocal Rank Fusion) reranking."""

    def test_rrf_rerank_basic(self):
        """Test basic RRF reranking."""
        vector_scores = {"doc1": 0.9, "doc2": 0.7, "doc3": 0.5}
        keyword_scores = {"doc1": 0.6, "doc2": 0.8, "doc4": 0.9}

        combined = WeightedInMemoryAggregator.rrf_rerank(vector_scores, keyword_scores, impact_factor=60.0)

        # Should include all documents
        expected_docs = {"doc1", "doc2", "doc3", "doc4"}
        assert set(combined.keys()) == expected_docs

        # All scores should be positive
        assert all(score > 0 for score in combined.values())

        # Documents appearing in both searches should have higher scores
        # doc1 and doc2 appear in both, doc3 and doc4 appear in only one
        assert combined["doc1"] > combined["doc3"]
        assert combined["doc2"] > combined["doc4"]

    def test_rrf_rerank_rank_calculation(self):
        """Test that RRF correctly calculates ranks."""
        # Create clear ranking order
        vector_scores = {"doc1": 1.0, "doc2": 0.8, "doc3": 0.6}  # Ranks: 1, 2, 3
        keyword_scores = {"doc1": 0.5, "doc2": 1.0, "doc3": 0.7}  # Ranks: 3, 1, 2

        combined = WeightedInMemoryAggregator.rrf_rerank(vector_scores, keyword_scores, impact_factor=60.0)

        # doc1: rank 1 in vector, rank 3 in keyword
        # doc2: rank 2 in vector, rank 1 in keyword
        # doc3: rank 3 in vector, rank 2 in keyword

        # doc2 should have the highest combined score (ranks 2+1=3)
        # followed by doc1 (ranks 1+3=4) and doc3 (ranks 3+2=5)
        # Remember: lower rank sum = higher RRF score
        assert combined["doc2"] > combined["doc1"] > combined["doc3"]

    def test_rrf_rerank_impact_factor(self):
        """Test that impact factor affects RRF scores."""
        vector_scores = {"doc1": 0.9, "doc2": 0.7}
        keyword_scores = {"doc1": 0.8, "doc2": 0.6}

        combined_low = WeightedInMemoryAggregator.rrf_rerank(vector_scores, keyword_scores, impact_factor=10.0)
        combined_high = WeightedInMemoryAggregator.rrf_rerank(vector_scores, keyword_scores, impact_factor=100.0)

        # Higher impact factor should generally result in lower scores
        # (because 1/(k+r) decreases as k increases)
        assert combined_low["doc1"] > combined_high["doc1"]
        assert combined_low["doc2"] > combined_high["doc2"]

    def test_rrf_rerank_missing_documents(self):
        """Test RRF handling of documents missing from one search."""
        vector_scores = {"doc1": 0.9, "doc2": 0.7}
        keyword_scores = {"doc1": 0.8, "doc3": 0.6}

        combined = WeightedInMemoryAggregator.rrf_rerank(vector_scores, keyword_scores, impact_factor=60.0)

        # Should include all documents
        assert len(combined) == 3

        # doc1 appears in both searches, should have highest score
        assert combined["doc1"] > combined["doc2"]
        assert combined["doc1"] > combined["doc3"]


class TestCombineSearchResults:
    """Test cases for the main combine_search_results function."""

    def test_combine_search_results_rrf_default(self):
        """Test combining with RRF as default."""
        vector_scores = {"doc1": 0.9, "doc2": 0.7}
        keyword_scores = {"doc1": 0.6, "doc3": 0.8}

        combined = WeightedInMemoryAggregator.combine_search_results(vector_scores, keyword_scores)

        # Should default to RRF
        assert len(combined) == 3
        assert all(score > 0 for score in combined.values())

    def test_combine_search_results_rrf_explicit(self):
        """Test combining with explicit RRF."""
        vector_scores = {"doc1": 0.9, "doc2": 0.7}
        keyword_scores = {"doc1": 0.6, "doc3": 0.8}

        combined = WeightedInMemoryAggregator.combine_search_results(
            vector_scores, keyword_scores, reranker_type=RERANKER_TYPE_RRF, reranker_params={"impact_factor": 50.0}
        )

        assert len(combined) == 3
        assert all(score > 0 for score in combined.values())

    def test_combine_search_results_weighted(self):
        """Test combining with weighted reranking."""
        vector_scores = {"doc1": 0.9, "doc2": 0.7}
        keyword_scores = {"doc1": 0.6, "doc3": 0.8}

        combined = WeightedInMemoryAggregator.combine_search_results(
            vector_scores, keyword_scores, reranker_type=RERANKER_TYPE_WEIGHTED, reranker_params={"alpha": 0.3}
        )

        assert len(combined) == 3
        assert all(0 <= score <= 1 for score in combined.values())

    def test_combine_search_results_unknown_type(self):
        """Test combining with unknown reranker type defaults to RRF."""
        vector_scores = {"doc1": 0.9}
        keyword_scores = {"doc2": 0.8}

        combined = WeightedInMemoryAggregator.combine_search_results(
            vector_scores, keyword_scores, reranker_type="unknown_type"
        )

        # Should fall back to RRF
        assert len(combined) == 2
        assert all(score > 0 for score in combined.values())

    def test_combine_search_results_empty_params(self):
        """Test combining with empty parameters."""
        vector_scores = {"doc1": 0.9}
        keyword_scores = {"doc2": 0.8}

        combined = WeightedInMemoryAggregator.combine_search_results(vector_scores, keyword_scores, reranker_params={})

        # Should use default parameters
        assert len(combined) == 2
        assert all(score > 0 for score in combined.values())

    def test_combine_search_results_empty_scores(self):
        """Test combining with empty score dictionaries."""
        # Test with empty vector scores
        combined = WeightedInMemoryAggregator.combine_search_results({}, {"doc1": 0.8})
        assert len(combined) == 1
        assert combined["doc1"] > 0

        # Test with empty keyword scores
        combined = WeightedInMemoryAggregator.combine_search_results({"doc1": 0.9}, {})
        assert len(combined) == 1
        assert combined["doc1"] > 0

        # Test with both empty
        combined = WeightedInMemoryAggregator.combine_search_results({}, {})
        assert len(combined) == 0
