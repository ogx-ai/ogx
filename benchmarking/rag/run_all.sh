#!/usr/bin/env bash
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
OPENAI_URL="https://api.openai.com/v1"
OGX_URL="${OGX_URL:-http://localhost:8321/v1}"
MODEL="${MODEL:-gpt-4.1}"

BEIR_DATASETS="nfcorpus scifact arguana fiqa trec-covid"

echo "============================================"
echo "RAG Benchmark Suite"
echo "Model: $MODEL"
echo "OpenAI URL: $OPENAI_URL"
echo "OGX URL: $OGX_URL"
echo "============================================"

# --- BEIR benchmarks ---
for dataset in $BEIR_DATASETS; do
    echo ""
    echo ">>> BEIR/$dataset — OpenAI SaaS"
    python "$SCRIPT_DIR/run_benchmark.py" \
        --benchmark beir --dataset "$dataset" \
        --base-url "$OPENAI_URL" \
        --model "$MODEL" --resume

    echo ""
    echo ">>> BEIR/$dataset — OGX (vector)"
    python "$SCRIPT_DIR/run_benchmark.py" \
        --benchmark beir --dataset "$dataset" \
        --base-url "$OGX_URL" \
        --search-mode vector \
        --model "$MODEL" --resume

    echo ""
    echo ">>> BEIR/$dataset — OGX (hybrid)"
    python "$SCRIPT_DIR/run_benchmark.py" \
        --benchmark beir --dataset "$dataset" \
        --base-url "$OGX_URL" \
        --search-mode hybrid \
        --model "$MODEL" --resume
done

# --- MultiHOP RAG ---
echo ""
echo ">>> MultiHOP RAG — OpenAI SaaS"
python "$SCRIPT_DIR/run_benchmark.py" \
    --benchmark multihop \
    --base-url "$OPENAI_URL" \
    --model "$MODEL" --resume

echo ""
echo ">>> MultiHOP RAG — OGX (vector)"
python "$SCRIPT_DIR/run_benchmark.py" \
    --benchmark multihop \
    --base-url "$OGX_URL" \
    --search-mode vector \
    --model "$MODEL" --resume

echo ""
echo ">>> MultiHOP RAG — OGX (hybrid)"
python "$SCRIPT_DIR/run_benchmark.py" \
    --benchmark multihop \
    --base-url "$OGX_URL" \
    --search-mode hybrid \
    --model "$MODEL" --resume

# --- Doc2Dial ---
echo ""
echo ">>> Doc2Dial — OpenAI SaaS"
python "$SCRIPT_DIR/run_benchmark.py" \
    --benchmark doc2dial \
    --base-url "$OPENAI_URL" \
    --model "$MODEL" --resume

echo ""
echo ">>> Doc2Dial — OGX (vector)"
python "$SCRIPT_DIR/run_benchmark.py" \
    --benchmark doc2dial \
    --base-url "$OGX_URL" \
    --search-mode vector \
    --model "$MODEL" --resume

echo ""
echo ">>> Doc2Dial — OGX (hybrid)"
python "$SCRIPT_DIR/run_benchmark.py" \
    --benchmark doc2dial \
    --base-url "$OGX_URL" \
    --search-mode hybrid \
    --model "$MODEL" --resume

# --- QReCC ---
echo ""
echo ">>> QReCC — OpenAI SaaS"
python "$SCRIPT_DIR/run_benchmark.py" \
    --benchmark qrecc \
    --base-url "$OPENAI_URL" \
    --model "$MODEL" --resume

echo ""
echo ">>> QReCC — OGX (vector)"
python "$SCRIPT_DIR/run_benchmark.py" \
    --benchmark qrecc \
    --base-url "$OGX_URL" \
    --search-mode vector \
    --model "$MODEL" --resume

echo ""
echo ">>> QReCC — OGX (hybrid)"
python "$SCRIPT_DIR/run_benchmark.py" \
    --benchmark qrecc \
    --base-url "$OGX_URL" \
    --search-mode hybrid \
    --model "$MODEL" --resume

echo ""
echo "============================================"
echo "All benchmarks complete. Run compare_results.py to generate comparison table."
echo "============================================"
