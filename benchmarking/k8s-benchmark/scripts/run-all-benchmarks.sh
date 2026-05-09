#!/usr/bin/env bash

# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Define benchmark configurations: (target, stack_replicas, vllm_replicas, stack_workers)
configs=(
    "stack 1 1 1"
    "stack 1 1 2"
    "stack 1 1 4"
    "vllm 1 1 -"
)

set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Running comprehensive GuideLL benchmark suite..."
echo "Start time: $(date)"

# Default deployment names
STACK_DEPLOYMENT="ogx-benchmark-server"
VLLM_DEPLOYMENT="vllm-server"

# Scaling function
scale_deployments() {
    local stack_replicas=$1
    local vllm_replicas=$2
    local workers=$3

    echo "Scaling deployments..."

    if [[ "$vllm_replicas" != "-" ]]; then
        echo "Scaling $VLLM_DEPLOYMENT to $vllm_replicas replicas..."
        kubectl scale deployment $VLLM_DEPLOYMENT --replicas=$vllm_replicas
        kubectl rollout status deployment $VLLM_DEPLOYMENT --timeout=600s
    fi

    if [[ "$target" == "stack" ]]; then
        if [[ "$stack_replicas" != "-" ]]; then
            echo "Scaling $STACK_DEPLOYMENT to $stack_replicas replicas..."
            kubectl scale deployment $STACK_DEPLOYMENT --replicas=$stack_replicas
            kubectl rollout status deployment $STACK_DEPLOYMENT --timeout=600s
        fi

        if [[ "$workers" != "-" ]]; then
            echo "Updating $STACK_DEPLOYMENT to use $workers workers..."
            kubectl set env deployment/$STACK_DEPLOYMENT OGX_WORKERS=$workers
            kubectl rollout status deployment $STACK_DEPLOYMENT --timeout=600s
        fi
    fi

    echo "All scaling operations completed. Waiting additional 30s for services to stabilize..."
    sleep 30
}


for config in "${configs[@]}"; do
    read -r target stack_replicas vllm_replicas workers <<< "$config"

    echo ""
    echo "=========================================="
    if [[ "$workers" != "-" ]]; then
        echo "Running benchmark: $target (stack=$stack_replicas, vllm=$vllm_replicas, workers=$workers)"
    else
        echo "Running benchmark: $target (stack=$stack_replicas, vllm=$vllm_replicas)"
    fi
    echo "Start: $(date)"
    echo "=========================================="

    # Scale deployments before running benchmark
    scale_deployments "$stack_replicas" "$vllm_replicas" "$workers"

    # Generate output filename with setup info
    TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    if [[ "$target" == "stack" ]]; then
        OUTPUT_FILE="results/guidellm-benchmark-${target}-s${stack_replicas}-sw${workers}-v${vllm_replicas}-${TIMESTAMP}.txt"
    else
        OUTPUT_FILE="results/guidellm-benchmark-${target}-v${vllm_replicas}-${TIMESTAMP}.txt"
    fi

    # Run the benchmark with the cluster as configured
    "$SCRIPT_DIR/run-guidellm-benchmark.sh" \
        --target "$target" \
        --output-file "$OUTPUT_FILE"

    echo "Completed: $(date)"
    echo "Waiting 30 seconds before next benchmark..."
    sleep 30
done

echo ""
echo "=========================================="
echo "All benchmarks completed!"
echo "End time: $(date)"
echo "=========================================="
echo ""
echo "Results files generated:"
ls -la results/guidellm-*.txt results/guidellm-*.json 2>/dev/null || echo "No result files found"
