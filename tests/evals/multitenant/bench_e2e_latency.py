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
End-to-end latency benchmark for OGX on OpenShift.

Measures real request latency through the full stack:
  1. Direct vLLM inference (baseline, no security layers)
  2. OGX inference (with routing, ABAC, provider dispatch)
  3. OGX vector store search (with tenant metadata filtering)

Usage:
    python tests/evals/multitenant/bench_e2e_latency.py \
        --ogx-url http://localhost:8321 \
        --vllm-url http://localhost:8000 \
        --num-requests 50
"""

import argparse
import csv
import json
import random
import statistics
import sys
import time
import urllib.request


def _percentile(data: list[float], p: float) -> float:
    s = sorted(data)
    idx = int(len(s) * p / 100)
    return s[min(idx, len(s) - 1)]


def _stats(latencies_ms: list[float]) -> dict:
    return {
        "n": len(latencies_ms),
        "mean": statistics.mean(latencies_ms),
        "median": statistics.median(latencies_ms),
        "p95": _percentile(latencies_ms, 95),
        "p99": _percentile(latencies_ms, 99),
        "min": min(latencies_ms),
        "max": max(latencies_ms),
        "stdev": statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0,
    }


def _post(url: str, body: dict, timeout: int = 120) -> tuple[float, dict]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read())
    elapsed_ms = (time.perf_counter() - start) * 1000
    return elapsed_ms, result


def _get(url: str, timeout: int = 30) -> tuple[float, dict]:
    req = urllib.request.Request(url, headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read())
    elapsed_ms = (time.perf_counter() - start) * 1000
    return elapsed_ms, result


PROMPTS = [
    "What is the capital of France?",
    "Explain photosynthesis in one sentence.",
    "What is 2 + 2?",
    "Name three programming languages.",
    "What color is the sky?",
    "Define machine learning briefly.",
    "Who wrote Romeo and Juliet?",
    "What is the boiling point of water?",
    "Name the largest ocean.",
    "What is an API?",
]

# Tenant-scoped documents for retrieval benchmark
TENANT_A_DOCS = [
    "Acme Financial Q1 2026 revenue reached $10.2M with 15% year-over-year growth driven by enterprise contracts.",
    "Acme Financial expanded engineering headcount to 500 employees with 50 new hires in AI division.",
    "Acme Financial completed SOC 2 Type II audit with zero findings across all jurisdictions.",
    "Acme Financial board approved $5M investment in AI infrastructure for FY2026 product roadmap.",
    "Acme Financial CEO compensation: base salary $450K, stock options $2.1M, performance bonus $180K.",
]

TENANT_B_DOCS = [
    "Beta Healthcare Q1 2026 revenue was $8.7M with 12% growth from telehealth platform expansion.",
    "Beta Healthcare reduced workforce to 300 employees after automating claims processing with AI.",
    "Beta Healthcare HIPAA compliance review identified three findings requiring remediation before Q3.",
    "Beta Healthcare strategic pivot to AI-first diagnostics with $3.2M R&D budget for clinical tools.",
    "Beta Healthcare patient cohort: 15,000 diabetes patients, average treatment cost $12,400.",
]

SEARCH_QUERIES = [
    "What was the quarterly revenue and growth rate?",
    "How many employees does the company have?",
    "What is the compliance audit status?",
    "What is the AI investment strategy?",
    "What are the confidential compensation details?",
]


def run_inference_benchmark(url: str, model: str, num_requests: int, label: str, max_tokens: int = 20) -> dict:
    latencies = []
    errors = 0
    for _ in range(3):
        try:
            _post(url, {"model": model, "messages": [{"role": "user", "content": "warmup"}], "max_tokens": 5})
        except Exception:
            pass

    for i in range(num_requests):
        try:
            ms, _ = _post(
                url,
                {
                    "model": model,
                    "messages": [{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}],
                    "max_tokens": max_tokens,
                },
            )
            latencies.append(ms)
        except Exception as e:
            errors += 1
            print(f"  [{label}] request {i} failed: {e}", file=sys.stderr)

    if not latencies:
        return {"label": label, "error": f"all {errors} requests failed"}
    s = _stats(latencies)
    s["label"] = label
    s["errors"] = errors
    return s


def setup_vector_store(ls_url: str, emb_model: str, emb_dim: int) -> str | None:
    """Create a vector store with tenant-scoped documents."""
    try:
        _, resp = _post(
            f"{ls_url}/v1/vector_stores",
            {
                "name": f"bench-multitenant-{random.randint(1000, 9999)}",
                "embedding_model": emb_model,
                "embedding_dimension": emb_dim,
            },
        )
        vs_id = resp.get("id")
        print(f"  Created vector store: {vs_id}")
    except Exception as e:
        print(f"  Failed to create vector store: {e}", file=sys.stderr)
        return None

    # Insert tenant-A documents with metadata
    for i, doc in enumerate(TENANT_A_DOCS):
        try:
            _post(
                f"{ls_url}/v1/vector_stores/{vs_id}/files",
                {
                    "content": doc,
                    "metadata": {"tenant_id": "tenant-a", "document_id": f"tenant-a-doc-{i}"},
                },
                timeout=120,
            )
        except Exception as e:
            print(f"  Failed to insert tenant-a doc {i}: {e}", file=sys.stderr)

    # Insert tenant-B documents with metadata
    for i, doc in enumerate(TENANT_B_DOCS):
        try:
            _post(
                f"{ls_url}/v1/vector_stores/{vs_id}/files",
                {
                    "content": doc,
                    "metadata": {"tenant_id": "tenant-b", "document_id": f"tenant-b-doc-{i}"},
                },
                timeout=120,
            )
        except Exception as e:
            print(f"  Failed to insert tenant-b doc {i}: {e}", file=sys.stderr)

    print(f"  Inserted {len(TENANT_A_DOCS) + len(TENANT_B_DOCS)} documents")
    return vs_id


def _warmup_search(ls_url: str, vs_id: str):
    """Warmup the embedding model by issuing throwaway search requests."""
    for _ in range(3):
        try:
            _post(
                f"{ls_url}/v1/vector_stores/{vs_id}/search",
                {
                    "query": "warmup query",
                    "max_num_results": 1,
                },
            )
        except Exception:
            pass


def run_search_benchmark(ls_url: str, vs_id: str, num_requests: int) -> dict:
    """Run vector store search with tenant filtering."""
    latencies = []
    errors = 0

    for i in range(num_requests):
        query = SEARCH_QUERIES[i % len(SEARCH_QUERIES)]
        try:
            ms, _ = _post(
                f"{ls_url}/v1/vector_stores/{vs_id}/search",
                {
                    "query": query,
                    "max_num_results": 5,
                    "filters": {"type": "eq", "key": "tenant_id", "value": "tenant-a"},
                },
            )
            latencies.append(ms)
        except Exception as e:
            errors += 1
            print(f"  [Search] request {i} failed: {e}", file=sys.stderr)

    if not latencies:
        return {"label": "Vector Search (gated)", "error": f"all {errors} requests failed"}
    s = _stats(latencies)
    s["label"] = "Vector Search (gated)"
    s["errors"] = errors
    return s


def run_search_ungated_benchmark(ls_url: str, vs_id: str, num_requests: int) -> dict:
    """Run vector store search without tenant filtering."""
    latencies = []
    errors = 0

    for i in range(num_requests):
        query = SEARCH_QUERIES[i % len(SEARCH_QUERIES)]
        try:
            ms, _ = _post(
                f"{ls_url}/v1/vector_stores/{vs_id}/search",
                {
                    "query": query,
                    "max_num_results": 5,
                },
            )
            latencies.append(ms)
        except Exception as e:
            errors += 1
            print(f"  [Search ungated] request {i} failed: {e}", file=sys.stderr)

    if not latencies:
        return {"label": "Vector Search (ungated)", "error": f"all {errors} requests failed"}
    s = _stats(latencies)
    s["label"] = "Vector Search (ungated)"
    s["errors"] = errors
    return s


def main():
    parser = argparse.ArgumentParser(description="E2E latency benchmark")
    parser.add_argument("--ogx-url", default="http://localhost:8321")
    parser.add_argument("--vllm-url", default="http://localhost:8000")
    parser.add_argument("--ls-model", default="vllm/meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--vllm-model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--emb-model", default="sentence-transformers/nomic-ai/nomic-embed-text-v1.5")
    parser.add_argument("--emb-dim", type=int, default=768)
    parser.add_argument("--num-requests", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=20)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    print("Benchmark config:")
    print(f"  Chat model (LS):  {args.ls_model}")
    print(f"  Chat model (vLLM): {args.vllm_model}")
    print(f"  Embedding model:  {args.emb_model}")
    print(f"  Requests/config:  {args.num_requests}")
    print(f"  Max tokens:       {args.max_tokens}")
    print()

    results = []

    # 1. Direct vLLM
    print("1. Direct vLLM inference (baseline)...")
    r = run_inference_benchmark(
        f"{args.vllm_url}/v1/chat/completions", args.vllm_model, args.num_requests, "vLLM Direct", args.max_tokens
    )
    results.append(r)
    print(f"   Median: {r['median']:.1f}ms, P95: {r['p95']:.1f}ms")

    # 2. OGX inference
    print("2. OGX inference (routing + ABAC + dispatch)...")
    r = run_inference_benchmark(
        f"{args.ogx_url}/v1/chat/completions", args.ls_model, args.num_requests, "LS Inference", args.max_tokens
    )
    results.append(r)
    print(f"   Median: {r['median']:.1f}ms, P95: {r['p95']:.1f}ms")

    # 3. Vector store setup + search
    print("3. Setting up vector store...")
    vs_id = setup_vector_store(args.ogx_url, args.emb_model, args.emb_dim)
    if vs_id:
        print("   Warming up embedding model...")
        _warmup_search(args.ogx_url, vs_id)
        print("4. Vector search (ungated)...")
        r = run_search_ungated_benchmark(args.ogx_url, vs_id, args.num_requests)
        results.append(r)
        print(f"   Median: {r['median']:.1f}ms, P95: {r['p95']:.1f}ms")

        print("5. Vector search (gated with tenant filter)...")
        r = run_search_benchmark(args.ogx_url, vs_id, args.num_requests)
        results.append(r)
        print(f"   Median: {r['median']:.1f}ms, P95: {r['p95']:.1f}ms")
    else:
        print("   Skipping search benchmarks (vector store setup failed)")

    # Compute overheads
    vllm_median = results[0]["median"]
    ls_median = results[1]["median"]
    inference_overhead_ms = ls_median - vllm_median
    inference_overhead_pct = (inference_overhead_ms / vllm_median) * 100 if vllm_median > 0 else 0

    print()
    print("=" * 85)
    print("END-TO-END LATENCY BENCHMARK RESULTS")
    print(f"Model: {args.vllm_model} | Embedding: {args.emb_model} | N={args.num_requests}")
    print("=" * 85)
    print(f"{'Configuration':<30} {'Median':>8} {'Mean':>8} {'P95':>8} {'P99':>8} {'StdDev':>8} {'N':>5}")
    print("-" * 85)
    for r in results:
        if "error" in r:
            print(f"{r['label']:<30} ERROR: {r['error']}")
            continue
        print(
            f"{r['label']:<30} "
            f"{r['median']:>7.1f}ms "
            f"{r['mean']:>7.1f}ms "
            f"{r['p95']:>7.1f}ms "
            f"{r['p99']:>7.1f}ms "
            f"{r['stdev']:>7.1f}ms "
            f"{r['n']:>5}"
        )
    print("-" * 85)
    print(
        f"{'Inference overhead':<30} {inference_overhead_ms:>7.1f}ms ({inference_overhead_pct:.1f}% of vLLM baseline)"
    )
    if vs_id and len(results) >= 4:
        search_ungated = results[2]["median"]
        search_gated = results[3]["median"]
        filter_overhead = search_gated - search_ungated
        print(f"{'Filter overhead':<30} {filter_overhead:>7.1f}ms (gated - ungated search)")
    print("=" * 85)

    if args.output_csv:
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["label", "n", "mean", "median", "p95", "p99", "min", "max", "stdev", "errors"]
            )
            writer.writeheader()
            for r in results:
                if "error" not in r:
                    writer.writerow(r)
        print(f"\nCSV written to {args.output_csv}")


if __name__ == "__main__":
    main()
