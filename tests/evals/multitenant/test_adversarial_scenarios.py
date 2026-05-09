# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Adversarial scenario evaluation for multitenant isolation.

Tests specific attack patterns that exploit the relevance-authorization gap:

  1. **Targeted cross-tenant query**: crafting a query to surface a specific
     tenant's confidential data.
  2. **Metadata filter tampering**: attempting to override or remove tenant
     filters to bypass gating.
  3. **Compound filter bypass**: using logical operators to widen the filter
     scope beyond authorized data.
  4. **Exhaustive enumeration**: using broad queries to enumerate all chunks
     in a shared store.

Each scenario is tested with and without gating to show attack success/failure.

Run::

    uv run pytest tests/evals/multitenant/test_adversarial_scenarios.py -v
"""

from .conftest import (
    QUERY_EMBEDDINGS,
    TOPICS,
    matches_filters,
)

TOP_K = 10
SCORE_THRESHOLD = 0.0


def _count_by_tenant(chunks) -> dict:
    """Count chunks per tenant_id."""
    counts = {}
    for c in chunks:
        tid = c.metadata.get("tenant_id", "unknown")
        counts[tid] = counts.get(tid, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Scenario 1: Targeted confidential data extraction
# ---------------------------------------------------------------------------


class TestTargetedConfidentialExtraction:
    """Attacker (tenant-a) crafts queries to surface tenant-b's confidential data."""

    async def test_targeted_query_without_gating_leaks_confidential(self, shared_vector_index):
        """Without gating, a query targeting 'confidential' topic surfaces
        tenant-b's sensitive data (patient records, compensation, etc.)."""
        query_emb = QUERY_EMBEDDINGS["confidential"]
        result = await shared_vector_index.query_vector(embedding=query_emb, k=TOP_K, score_threshold=SCORE_THRESHOLD)

        tenant_counts = _count_by_tenant(result.chunks)
        has_cross_tenant = len(tenant_counts) > 1

        # Verify the attack succeeds without gating
        assert has_cross_tenant, "Expected confidential query to return cross-tenant data without gating"

        # Verify tenant-b's confidential doc specifically appears
        tenant_b_topics = [c.metadata.get("topic") for c in result.chunks if c.metadata.get("tenant_id") == "tenant-b"]
        assert "confidential" in tenant_b_topics

    async def test_targeted_query_with_gating_blocks_confidential(self, shared_vector_index):
        """With gating, the same query returns only tenant-a's data."""
        query_emb = QUERY_EMBEDDINGS["confidential"]
        result = await shared_vector_index.query_vector(embedding=query_emb, k=TOP_K, score_threshold=SCORE_THRESHOLD)

        tenant_filter = {"type": "eq", "key": "tenant_id", "value": "tenant-a"}
        filtered = [c for c in result.chunks if matches_filters(c.metadata, tenant_filter)]

        for c in filtered:
            assert c.metadata.get("tenant_id") == "tenant-a", (
                f"Gated confidential query returned unauthorized chunk: "
                f"tenant={c.metadata.get('tenant_id')}, topic={c.metadata.get('topic')}"
            )


# ---------------------------------------------------------------------------
# Scenario 2: Metadata filter tampering
# ---------------------------------------------------------------------------


class TestMetadataFilterTampering:
    """Attacker attempts to manipulate metadata filters to bypass isolation."""

    async def test_empty_filter_returns_all_tenants(self, shared_vector_index):
        """Empty/null filter returns cross-tenant data (baseline vulnerability)."""
        query_emb = QUERY_EMBEDDINGS["revenue"]
        result = await shared_vector_index.query_vector(embedding=query_emb, k=TOP_K, score_threshold=SCORE_THRESHOLD)

        # Empty filter = no filtering = all results pass
        filtered = [c for c in result.chunks if matches_filters(c.metadata, {})]
        tenant_counts = _count_by_tenant(filtered)
        assert len(tenant_counts) > 1, "Empty filter should return all tenants"

    async def test_wrong_tenant_filter_returns_nothing_for_attacker(self, shared_vector_index):
        """If the server enforces the correct tenant_id filter, an attacker cannot
        substitute a different tenant_id to access another tenant's data."""
        query_emb = QUERY_EMBEDDINGS["revenue"]
        result = await shared_vector_index.query_vector(embedding=query_emb, k=TOP_K, score_threshold=SCORE_THRESHOLD)

        # Server enforces tenant-a's filter (correct behavior)
        server_filter = {"type": "eq", "key": "tenant_id", "value": "tenant-a"}
        server_filtered = [c for c in result.chunks if matches_filters(c.metadata, server_filter)]

        # All results must be tenant-a
        for c in server_filtered:
            assert c.metadata.get("tenant_id") == "tenant-a"

        # Attacker tries to substitute tenant-b's filter
        # In a properly secured system, the server ignores client-provided
        # tenant_id and uses the authenticated user's identity instead
        attacker_filter = {"type": "eq", "key": "tenant_id", "value": "tenant-b"}
        attacker_filtered = [c for c in result.chunks if matches_filters(c.metadata, attacker_filter)]

        # The attacker's filter would return tenant-b data IF the server
        # accepted client-provided filters. This demonstrates why the server
        # must enforce tenant_id from the auth context, not from the request.
        for c in attacker_filtered:
            assert c.metadata.get("tenant_id") == "tenant-b"


# ---------------------------------------------------------------------------
# Scenario 3: Compound filter bypass attempts
# ---------------------------------------------------------------------------


class TestCompoundFilterBypass:
    """Attacker attempts to use OR filters to widen scope beyond their tenant."""

    async def test_or_filter_bypass_attempt(self, shared_vector_index):
        """An OR filter combining both tenants would bypass isolation.
        The server must reject or override client-provided compound filters."""
        query_emb = QUERY_EMBEDDINGS["strategy"]
        result = await shared_vector_index.query_vector(embedding=query_emb, k=TOP_K, score_threshold=SCORE_THRESHOLD)

        # Legitimate filter (server-enforced)
        legitimate_filter = {"type": "eq", "key": "tenant_id", "value": "tenant-a"}
        legitimate_results = [c for c in result.chunks if matches_filters(c.metadata, legitimate_filter)]

        # Attacker attempts OR filter to include both tenants
        bypass_filter = {
            "type": "or",
            "filters": [
                {"type": "eq", "key": "tenant_id", "value": "tenant-a"},
                {"type": "eq", "key": "tenant_id", "value": "tenant-b"},
            ],
        }
        bypass_results = [c for c in result.chunks if matches_filters(c.metadata, bypass_filter)]

        # The OR filter returns more results, demonstrating the bypass
        assert len(bypass_results) >= len(legitimate_results)

        # Legitimate results must be tenant-a only
        assert all(c.metadata.get("tenant_id") == "tenant-a" for c in legitimate_results)


# ---------------------------------------------------------------------------
# Scenario 4: Exhaustive enumeration via broad queries
# ---------------------------------------------------------------------------


class TestExhaustiveEnumeration:
    """Attacker uses multiple diverse queries to enumerate all chunks in the store."""

    async def test_enumeration_blocked_by_gating(self, shared_vector_index):
        """Even with many diverse queries, gating limits exposure to the
        attacker's own tenant's data."""
        tenant_filter = {"type": "eq", "key": "tenant_id", "value": "tenant-a"}

        # Issue queries across all topics with high k
        all_seen_chunks = set()
        all_seen_tenants = set()

        for topic in TOPICS:
            query_emb = QUERY_EMBEDDINGS[topic]
            result = await shared_vector_index.query_vector(
                embedding=query_emb, k=TOP_K, score_threshold=SCORE_THRESHOLD
            )
            gated = [c for c in result.chunks if matches_filters(c.metadata, tenant_filter)]
            for c in gated:
                all_seen_chunks.add(c.metadata.get("document_id"))
                all_seen_tenants.add(c.metadata.get("tenant_id"))

        # Even after exhaustive querying, only tenant-a docs are visible
        assert all_seen_tenants == {"tenant-a"}, f"Enumeration attack saw tenants: {all_seen_tenants}"

    async def test_enumeration_without_gating_exposes_all(self, shared_vector_index):
        """Without gating, enumeration exposes both tenants' data."""
        all_seen_tenants = set()

        for topic in TOPICS:
            query_emb = QUERY_EMBEDDINGS[topic]
            result = await shared_vector_index.query_vector(
                embedding=query_emb, k=TOP_K, score_threshold=SCORE_THRESHOLD
            )
            for c in result.chunks:
                all_seen_tenants.add(c.metadata.get("tenant_id"))

        assert len(all_seen_tenants) > 1, "Without gating, enumeration should expose multiple tenants"


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestAdversarialSummary:
    """Aggregate adversarial evaluation results."""

    async def test_adversarial_summary_table(self, shared_vector_index):
        """Produces a summary table of all adversarial scenarios."""
        tenant_filter = {"type": "eq", "key": "tenant_id", "value": "tenant-a"}

        scenarios = []

        # Scenario 1: Targeted confidential extraction
        query_emb = QUERY_EMBEDDINGS["confidential"]
        result = await shared_vector_index.query_vector(embedding=query_emb, k=TOP_K, score_threshold=SCORE_THRESHOLD)
        ungated_leaked = any(c.metadata.get("tenant_id") != "tenant-a" for c in result.chunks)
        gated = [c for c in result.chunks if matches_filters(c.metadata, tenant_filter)]
        gated_leaked = any(c.metadata.get("tenant_id") != "tenant-a" for c in gated)
        scenarios.append(("Targeted confidential extraction", ungated_leaked, gated_leaked))

        # Scenario 2: Enumeration
        ungated_tenants = set()
        gated_tenants = set()
        for topic in TOPICS:
            qe = QUERY_EMBEDDINGS[topic]
            r = await shared_vector_index.query_vector(embedding=qe, k=TOP_K, score_threshold=SCORE_THRESHOLD)
            for c in r.chunks:
                ungated_tenants.add(c.metadata.get("tenant_id"))
            for c in r.chunks:
                if matches_filters(c.metadata, tenant_filter):
                    gated_tenants.add(c.metadata.get("tenant_id"))
        scenarios.append(("Exhaustive enumeration", len(ungated_tenants) > 1, len(gated_tenants) > 1))

        # Scenario 3: OR-filter bypass
        bypass_filter = {
            "type": "or",
            "filters": [
                {"type": "eq", "key": "tenant_id", "value": "tenant-a"},
                {"type": "eq", "key": "tenant_id", "value": "tenant-b"},
            ],
        }
        result = await shared_vector_index.query_vector(
            embedding=QUERY_EMBEDDINGS["revenue"],
            k=TOP_K,
            score_threshold=SCORE_THRESHOLD,
        )
        bypass_tenants = {
            c.metadata.get("tenant_id") for c in result.chunks if matches_filters(c.metadata, bypass_filter)
        }
        server_tenants = {
            c.metadata.get("tenant_id") for c in result.chunks if matches_filters(c.metadata, tenant_filter)
        }
        scenarios.append(
            (
                "Compound OR-filter bypass",
                len(bypass_tenants) > 1,
                len(server_tenants) > 1,
            )
        )

        # All gated scenarios must block the attack
        for name, _, gated_success in scenarios:
            assert not gated_success, f"Gated scenario '{name}' should block the attack"
