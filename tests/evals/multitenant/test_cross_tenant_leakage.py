# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Cross-tenant retrieval leakage evaluation.

Measures the fraction of retrieved chunks that belong to a tenant other than the
querying tenant, under three configurations:

  1. **Ungated** (relevance-only): no metadata filter applied.
  2. **Chunk-level gated**: metadata filter on ``tenant_id`` applied post-retrieval.
  3. **Per-tenant index**: each tenant has a physically separate vector store.

Expected results for the paper:
  - Ungated leakage rate > 0  (demonstrates the relevance-authorization gap)
  - Chunk-level gated leakage rate == 0
  - Per-tenant index leakage rate == 0

Run::

    uv run pytest tests/evals/multitenant/test_cross_tenant_leakage.py -v
"""

import pytest

from .conftest import (
    QUERY_EMBEDDINGS,
    TOPICS,
    matches_filters,
)

# Number of results to retrieve per query
TOP_K = 5
SCORE_THRESHOLD = 0.0


def _leakage_rate(chunks, querying_tenant: str) -> float:
    """Fraction of returned chunks that do NOT belong to querying_tenant."""
    if not chunks:
        return 0.0
    unauthorized = sum(1 for c in chunks if c.metadata.get("tenant_id") != querying_tenant)
    return unauthorized / len(chunks)


def _leakage_details(chunks, querying_tenant: str) -> dict:
    """Per-chunk breakdown for debugging and paper tables."""
    details = []
    for c in chunks:
        details.append(
            {
                "chunk_tenant": c.metadata.get("tenant_id"),
                "topic": c.metadata.get("topic"),
                "authorized": c.metadata.get("tenant_id") == querying_tenant,
            }
        )
    return details


# ---------------------------------------------------------------------------
# 1. Ungated retrieval (relevance-only) -- demonstrates the problem
# ---------------------------------------------------------------------------


class TestUngatedRetrieval:
    """Retrieval without tenant filtering: demonstrates cross-tenant leakage."""

    @pytest.mark.parametrize("topic", TOPICS)
    async def test_ungated_retrieval_leaks_cross_tenant_data(self, shared_vector_index, topic):
        """When no tenant filter is applied, cross-tenant documents appear in results
        because they share semantic similarity on the same topic."""
        query_emb = QUERY_EMBEDDINGS[topic]
        result = await shared_vector_index.query_vector(embedding=query_emb, k=TOP_K, score_threshold=SCORE_THRESHOLD)

        # With 5 topics per tenant (10 total) and k=5, same-topic cross-tenant
        # docs will rank high due to the controlled embedding similarity
        tenants_in_results = {c.metadata.get("tenant_id") for c in result.chunks}

        # The key assertion: ungated retrieval returns data from BOTH tenants
        assert len(tenants_in_results) > 1, (
            f"Expected cross-tenant leakage for topic '{topic}' but all results "
            f"came from a single tenant. This means the embedding design did not "
            f"produce the expected cross-tenant similarity."
        )

    async def test_ungated_aggregate_leakage_rate(self, shared_vector_index):
        """Aggregate leakage rate across all topics when querying as tenant-a."""
        total_chunks = 0
        leaked_chunks = 0

        for topic in TOPICS:
            query_emb = QUERY_EMBEDDINGS[topic]
            result = await shared_vector_index.query_vector(
                embedding=query_emb, k=TOP_K, score_threshold=SCORE_THRESHOLD
            )
            for c in result.chunks:
                total_chunks += 1
                if c.metadata.get("tenant_id") != "tenant-a":
                    leaked_chunks += 1

        leakage = leaked_chunks / total_chunks if total_chunks > 0 else 0.0

        # Leakage MUST be > 0 for the experiment to be meaningful
        assert leakage > 0, (
            "Expected non-zero leakage rate without filtering. "
            "The synthetic embeddings should produce cross-tenant similarity."
        )


# ---------------------------------------------------------------------------
# 2. Chunk-level gated retrieval -- demonstrates the fix
# ---------------------------------------------------------------------------


class TestChunkLevelGatedRetrieval:
    """Retrieval with metadata filter on tenant_id: blocks cross-tenant leakage."""

    @pytest.mark.parametrize("topic", TOPICS)
    async def test_gated_retrieval_blocks_cross_tenant_data(self, shared_vector_index, topic):
        """With a tenant_id metadata filter, zero cross-tenant chunks are returned."""
        query_emb = QUERY_EMBEDDINGS[topic]
        result = await shared_vector_index.query_vector(embedding=query_emb, k=TOP_K, score_threshold=SCORE_THRESHOLD)

        # Apply chunk-level metadata filter (simulating what file_search does)
        tenant_filter = {"type": "eq", "key": "tenant_id", "value": "tenant-a"}
        filtered = [c for c in result.chunks if matches_filters(c.metadata, tenant_filter)]

        leakage = _leakage_rate(filtered, "tenant-a")
        assert leakage == 0.0, (
            f"Topic '{topic}': expected 0% leakage with tenant filter, "
            f"got {leakage:.2%}. Details: {_leakage_details(filtered, 'tenant-a')}"
        )

    async def test_gated_aggregate_leakage_rate(self, shared_vector_index):
        """Aggregate leakage rate with chunk-level gating is exactly zero."""
        total_chunks = 0
        leaked_chunks = 0
        tenant_filter = {"type": "eq", "key": "tenant_id", "value": "tenant-a"}

        for topic in TOPICS:
            query_emb = QUERY_EMBEDDINGS[topic]
            result = await shared_vector_index.query_vector(
                embedding=query_emb, k=TOP_K, score_threshold=SCORE_THRESHOLD
            )
            filtered = [c for c in result.chunks if matches_filters(c.metadata, tenant_filter)]
            for c in filtered:
                total_chunks += 1
                if c.metadata.get("tenant_id") != "tenant-a":
                    leaked_chunks += 1

        leakage = leaked_chunks / total_chunks if total_chunks > 0 else 0.0
        assert leakage == 0.0
        assert total_chunks > 0, "Filter should still return authorized chunks"

    @pytest.mark.parametrize(
        "querying_tenant",
        ["tenant-a", "tenant-b"],
        ids=["tenant_a_queries", "tenant_b_queries"],
    )
    async def test_gated_retrieval_symmetric(self, shared_vector_index, querying_tenant):
        """Gating works symmetrically for both tenants."""
        tenant_filter = {"type": "eq", "key": "tenant_id", "value": querying_tenant}

        for topic in TOPICS:
            query_emb = QUERY_EMBEDDINGS[topic]
            result = await shared_vector_index.query_vector(
                embedding=query_emb, k=TOP_K, score_threshold=SCORE_THRESHOLD
            )
            filtered = [c for c in result.chunks if matches_filters(c.metadata, tenant_filter)]
            leakage = _leakage_rate(filtered, querying_tenant)
            assert leakage == 0.0


# ---------------------------------------------------------------------------
# 3. Per-tenant index isolation -- demonstrates physical separation
# ---------------------------------------------------------------------------


class TestPerTenantIndexIsolation:
    """Each tenant has a separate vector index. No cross-tenant data exists."""

    @pytest.mark.parametrize("topic", TOPICS)
    async def test_per_tenant_index_zero_leakage(self, tenant_a_vector_index, topic):
        """Querying a per-tenant index returns only that tenant's data."""
        query_emb = QUERY_EMBEDDINGS[topic]
        result = await tenant_a_vector_index.query_vector(embedding=query_emb, k=TOP_K, score_threshold=SCORE_THRESHOLD)

        for c in result.chunks:
            assert c.metadata.get("tenant_id") == "tenant-a", (
                f"Per-tenant index returned chunk from wrong tenant: {c.metadata.get('tenant_id')}"
            )

    async def test_per_tenant_aggregate_leakage_rate(self, tenant_a_vector_index, tenant_b_vector_index):
        """Both per-tenant indexes have zero leakage."""
        for index, tenant_id in [
            (tenant_a_vector_index, "tenant-a"),
            (tenant_b_vector_index, "tenant-b"),
        ]:
            total = 0
            leaked = 0
            for topic in TOPICS:
                query_emb = QUERY_EMBEDDINGS[topic]
                result = await index.query_vector(embedding=query_emb, k=TOP_K, score_threshold=SCORE_THRESHOLD)
                for c in result.chunks:
                    total += 1
                    if c.metadata.get("tenant_id") != tenant_id:
                        leaked += 1

            leakage = leaked / total if total > 0 else 0.0
            assert leakage == 0.0


# ---------------------------------------------------------------------------
# 4. Comparative summary (all three configurations)
# ---------------------------------------------------------------------------


class TestLeakageComparison:
    """Side-by-side comparison of ungated, gated, and per-tenant configurations."""

    async def test_leakage_comparison_table(self, shared_vector_index, tenant_a_vector_index):
        """Produces a comparison table suitable for the paper's evaluation section."""
        tenant_filter = {"type": "eq", "key": "tenant_id", "value": "tenant-a"}
        results = {}

        for config_name, index, apply_filter in [
            ("ungated", shared_vector_index, False),
            ("chunk_gated", shared_vector_index, True),
            ("per_tenant", tenant_a_vector_index, False),
        ]:
            total = 0
            leaked = 0
            for topic in TOPICS:
                query_emb = QUERY_EMBEDDINGS[topic]
                result = await index.query_vector(embedding=query_emb, k=TOP_K, score_threshold=SCORE_THRESHOLD)
                chunks = result.chunks
                if apply_filter:
                    chunks = [c for c in chunks if matches_filters(c.metadata, tenant_filter)]
                for c in chunks:
                    total += 1
                    if c.metadata.get("tenant_id") != "tenant-a":
                        leaked += 1

            rate = leaked / total if total > 0 else 0.0
            results[config_name] = {
                "leakage_rate": rate,
                "leaked": leaked,
                "total": total,
            }

        # Ungated must leak, gated configurations must not
        assert results["ungated"]["leakage_rate"] > 0
        assert results["chunk_gated"]["leakage_rate"] == 0.0
        assert results["per_tenant"]["leakage_rate"] == 0.0
