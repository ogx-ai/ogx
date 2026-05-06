#!/usr/bin/env python3
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Predicate pushdown vs post-retrieval filtering benchmark.

Measures how post-retrieval metadata filtering scales with corpus size
on sqlite-vec (which does NOT support predicate pushdown). At each
corpus size, we compare:
  - Ungated search (no filter)
  - Gated search (tenant_id metadata filter applied post-retrieval)
  - Recall@k: does the 5x over-fetch still find all relevant docs?

This answers reviewer question: "How does the system behave when the
vector backend does not support predicate pushdown?"

Usage:
    uv run python tests/evals/multitenant/bench_predicate_pushdown.py
"""

import asyncio
import csv
import statistics
import time

import numpy as np

from ogx.providers.inline.vector_io.sqlite_vec.sqlite_vec import SQLiteVecIndex
from ogx_api import ChunkMetadata, EmbeddedChunk

EMBEDDING_DIM = 128
NUM_TENANTS = 2
DOCS_PER_TENANT_RATIOS = [0.5, 0.5]  # 50/50 split
CORPUS_SIZES = [100, 1_000, 10_000, 50_000]
TOP_K = 5
CHUNK_MULTIPLIERS = [1, 5, 10, 20]  # Compare different over-fetch ratios
N_QUERIES = 50
N_TOPICS = 10  # number of distinct topic clusters


def _make_corpus(num_chunks: int, dim: int, n_topics: int, seed: int = 42):
    """Build a corpus with controlled topic structure across two tenants.

    Each topic gets an orthogonal basis vector. Documents on the same topic
    across tenants share this basis (high similarity), creating conditions
    where post-retrieval filtering matters.
    """
    rng = np.random.RandomState(seed)

    # Create orthogonal topic bases
    raw = rng.randn(n_topics, dim).astype(np.float32)
    bases = np.zeros_like(raw)
    for i in range(n_topics):
        v = raw[i].copy()
        for j in range(i):
            v -= np.dot(v, bases[j]) * bases[j]
        norm = np.linalg.norm(v)
        bases[i] = v / norm if norm > 0 else v

    chunks = []
    per_tenant = num_chunks // NUM_TENANTS

    for t in range(NUM_TENANTS):
        tenant_id = f"tenant-{t}"
        for d in range(per_tenant):
            topic_idx = d % n_topics
            noise = rng.randn(dim).astype(np.float32) * 0.05
            emb = bases[topic_idx] + noise
            emb = emb / np.linalg.norm(emb)

            chunk_id = f"{tenant_id}-doc-{d}"
            chunks.append(
                EmbeddedChunk(
                    content=f"Document {d} for {tenant_id} on topic {topic_idx}",
                    chunk_id=chunk_id,
                    metadata={
                        "tenant_id": tenant_id,
                        "topic": str(topic_idx),
                        "document_id": chunk_id,
                    },
                    chunk_metadata=ChunkMetadata(
                        document_id=chunk_id,
                        chunk_id=chunk_id,
                    ),
                    embedding=emb.tolist(),
                    embedding_model="synthetic",
                    embedding_dimension=dim,
                )
            )

    return chunks, bases


def _matches_filters(metadata: dict, filters: dict) -> bool:
    if not filters:
        return True
    ft = filters.get("type")
    if ft == "eq":
        return metadata.get(filters["key"]) == filters["value"]
    if ft == "and":
        return all(_matches_filters(metadata, f) for f in filters.get("filters", []))
    if ft == "or":
        return any(_matches_filters(metadata, f) for f in filters.get("filters", []))
    return False


def _recall_at_k(chunks, relevant_ids: set) -> float:
    if not relevant_ids:
        return 0.0
    retrieved = {c.metadata.get("document_id") for c in chunks}
    return len(retrieved & relevant_ids) / len(relevant_ids)


def _percentile(data, p):
    s = sorted(data)
    return s[min(int(len(s) * p / 100), len(s) - 1)]


async def bench_corpus_size(num_chunks: int, multiplier: int, tmp_dir: str):
    """Run benchmark for a single corpus size and multiplier."""
    chunks, bases = _make_corpus(num_chunks, EMBEDDING_DIM, N_TOPICS)

    db_path = f"{tmp_dir}/bench_{num_chunks}_m{multiplier}.db"
    index = SQLiteVecIndex(EMBEDDING_DIM, db_path, f"bench_{num_chunks}_m{multiplier}")
    await index.initialize()
    await index.add_chunks(chunks)

    rng = np.random.RandomState(999)
    tenant_filter = {"type": "eq", "key": "tenant_id", "value": "tenant-0"}

    ungated_latencies = []
    gated_latencies = []
    gated_recalls = []

    for q in range(N_QUERIES):
        topic_idx = q % N_TOPICS
        noise = rng.randn(EMBEDDING_DIM).astype(np.float32) * 0.02
        query_emb = bases[topic_idx] + noise
        query_emb = query_emb / np.linalg.norm(query_emb)

        # Relevant docs: tenant-0 docs on this topic
        per_tenant = num_chunks // NUM_TENANTS
        relevant_ids = {f"tenant-0-doc-{d}" for d in range(per_tenant) if d % N_TOPICS == topic_idx}

        # Ungated (retrieve exactly k)
        start = time.perf_counter()
        await index.query_vector(embedding=query_emb, k=TOP_K, score_threshold=0.0)
        ungated_ms = (time.perf_counter() - start) * 1000
        ungated_latencies.append(ungated_ms)

        # Gated (over-fetch by multiplier, then filter to k)
        start = time.perf_counter()
        result = await index.query_vector(embedding=query_emb, k=TOP_K * multiplier, score_threshold=0.0)
        filtered = [c for c in result.chunks if _matches_filters(c.metadata, tenant_filter)][:TOP_K]
        gated_ms = (time.perf_counter() - start) * 1000
        gated_latencies.append(gated_ms)

        gated_recalls.append(_recall_at_k(filtered, relevant_ids))

    await index.delete()

    ungated_med = statistics.median(ungated_latencies)
    gated_med = statistics.median(gated_latencies)
    overhead = gated_med - ungated_med

    return {
        "corpus_size": num_chunks,
        "multiplier": multiplier,
        "ungated_median_ms": ungated_med,
        "gated_median_ms": gated_med,
        "filter_overhead_ms": overhead,
        "filter_overhead_pct": (overhead / ungated_med * 100) if ungated_med > 0 else 0,
        "gated_recall_at_k": statistics.mean(gated_recalls),
    }


async def main():
    import tempfile

    tmp_dir = tempfile.mkdtemp()
    results = []

    print("Predicate pushdown benchmark (sqlite-vec, post-retrieval filtering)")
    print(f"Embedding dim: {EMBEDDING_DIM}, Top-k: {TOP_K}, Queries/size: {N_QUERIES}")
    print(f"Topics: {N_TOPICS}, Tenants: {NUM_TENANTS} (50/50 split)")
    print(f"Multipliers: {CHUNK_MULTIPLIERS}")
    print()

    for size in CORPUS_SIZES:
        for mult in CHUNK_MULTIPLIERS:
            label = f"{size:>6} chunks, {mult}x"
            print(f"  {label} ...", end=" ", flush=True)
            r = await bench_corpus_size(size, mult, tmp_dir)
            results.append(r)
            print(
                f"ungated={r['ungated_median_ms']:.2f}ms  "
                f"gated={r['gated_median_ms']:.2f}ms  "
                f"overhead={r['filter_overhead_ms']:.2f}ms  "
                f"recall={r['gated_recall_at_k']:.3f}"
            )

    print()
    print("=" * 100)
    print("POST-RETRIEVAL FILTERING: LATENCY vs RECALL TRADE-OFF (sqlite-vec)")
    print("=" * 100)
    print(f"{'Corpus':>8} {'Mult':>5} {'Fetch k':>8} {'Ungated':>10} {'Gated':>10} {'Overhead':>10} {'Recall@5':>9}")
    print("-" * 100)
    for r in results:
        print(
            f"{r['corpus_size']:>8,} "
            f"{r['multiplier']:>4}x "
            f"{TOP_K * r['multiplier']:>7} "
            f"{r['ungated_median_ms']:>9.2f}ms "
            f"{r['gated_median_ms']:>9.2f}ms "
            f"{r['filter_overhead_ms']:>9.2f}ms "
            f"{r['gated_recall_at_k']:>8.3f}"
        )
    print("=" * 100)

    # Write CSV
    csv_path = "/tmp/predicate_pushdown_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCSV: {csv_path}")


if __name__ == "__main__":
    asyncio.run(main())
