#!/bin/bash

# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIDS_FILE="$SCRIPT_DIR/.pids"

MOCK_PORT=8080
STACK_PORT=8321
WORKERS=1
LOCUST_ARGS=()
RESULTS_DIR="$SCRIPT_DIR/results"
USERS=1
SPAWN_RATE=10
RUN_TIME=60

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Runs complete vertical scaling benchmark: starts services, runs load test, cleans up.

Server Options:
    --workers NUMBER         Number of uvicorn workers (default: 1)
    --mock-port NUMBER       Port for mock server (default: 8080)
    --stack-port NUMBER      Port for OGX server (default: 8321)

Benchmark Options:
    --users NUMBER          Number of concurrent users (default: 1)
    --run-time SECONDS      Duration of test in seconds (default: 60)

Common Options:
    --help                  Show this help message

Examples:
    # Run with defaults (1 worker, 1 user, 60s)
    $0

    # Test vertical scaling with 8 workers
    $0 --workers 8

    # Longer test with more users
    $0 --workers 16 --users 200 --run-time 120
EOF
}

cleanup() {
    echo ""
    echo "=== Cleaning up ==="

    if [[ -f "$PIDS_FILE" ]]; then
        while read -r pid; do
            if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
                echo "Stopping process: $pid"
                kill "$pid" 2>/dev/null || true
            fi
        done < "$PIDS_FILE"
        rm -f "$PIDS_FILE"
    fi

    echo "✅ Cleanup complete"
}

trap cleanup EXIT ERR INT TERM

check_port_available() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 1
    fi
    return 0
}

wait_for_url() {
    local url=$1
    local max_attempts=30
    local attempt=0

    echo "Waiting for $url to be ready..."
    while [ $attempt -lt $max_attempts ]; do
        if curl -s "$url" >/dev/null 2>&1; then
            echo "✅ Service ready"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 1
    done

    echo "❌ Service failed to start after $max_attempts seconds"
    return 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --mock-port)
            MOCK_PORT="$2"
            shift 2
            ;;
        --stack-port)
            STACK_PORT="$2"
            shift 2
            ;;
        --users)
            USERS="$2"
            shift 2
            ;;
        --spawn-rate)
            SPAWN_RATE="$2"
            shift 2
            ;;
        --run-time)
            RUN_TIME="$2"
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

echo "==================================================================="
echo "  OGX Vertical Scaling Benchmark"
echo "==================================================================="
echo ""

if ! check_port_available $MOCK_PORT; then
    echo "❌ Port $MOCK_PORT is already in use"
    exit 1
fi

if ! check_port_available $STACK_PORT; then
    echo "❌ Port $STACK_PORT is already in use"
    exit 1
fi

echo "✅ Ports $MOCK_PORT and $STACK_PORT are available"
echo ""

echo "Step 1/3: Starting mock server..."
echo "Port: $MOCK_PORT"
echo ""

python "$SCRIPT_DIR/mock-server.py" $MOCK_PORT > /dev/null 2>&1 &
MOCK_PID=$!
echo "$MOCK_PID" > "$PIDS_FILE"
echo "Mock Server PID: $MOCK_PID"

MOCK_URL="http://localhost:$MOCK_PORT"
if ! wait_for_url "$MOCK_URL/v1/health"; then
    echo "Failed to start mock server"
    exit 1
fi

echo ""
echo "Step 2/3: Starting stack server..."
echo "Port: $STACK_PORT"
echo "Workers: $WORKERS"
echo "Backend: $MOCK_URL"
echo ""

OPENAI_BASE_URL="$MOCK_URL/v1" OPENAI_API_KEY="fake-token" uv run ogx run --providers inference=remote::openai --port $STACK_PORT > "$SCRIPT_DIR/stack.log" 2>&1 &
STACK_PID=$!
echo "$STACK_PID" >> "$PIDS_FILE"
echo "Stack Server PID: $STACK_PID"

STACK_URL="http://localhost:$STACK_PORT"
if ! wait_for_url "$STACK_URL/v1/health"; then
    echo "Failed to start stack server. Check $SCRIPT_DIR/stack.log"
    exit 1
fi

echo ""
echo "Step 3/3: Running benchmarks..."
echo ""

mkdir -p "$RESULTS_DIR"

# Baseline benchmark - mock server
echo "Benchmarking mock server (baseline)..."
echo "Host: $MOCK_URL"
echo "Users: $USERS"
echo "Spawn rate: $SPAWN_RATE users/sec"
echo "Run time: ${RUN_TIME}s"
echo ""

uv run --with locust locust \
    --locustfile "$SCRIPT_DIR/locustfile.py" \
    --host "$MOCK_URL" \
    --users $USERS \
    --spawn-rate $SPAWN_RATE \
    --run-time ${RUN_TIME}s \
    --headless \
    --html "$RESULTS_DIR/baseline.html" \
    --csv "$RESULTS_DIR/baseline" \
    --only-summary

if [ $? -ne 0 ]; then
    echo "❌ Baseline benchmark failed"
    exit 1
fi

echo ""
echo "Benchmarking stack server..."
echo "Host: $STACK_URL"
echo "Users: $USERS"
echo "Spawn rate: $SPAWN_RATE users/sec"
echo "Run time: ${RUN_TIME}s"
echo ""

uv run --with locust locust \
    --locustfile "$SCRIPT_DIR/locustfile.py" \
    --host "$STACK_URL" \
    --users $USERS \
    --spawn-rate $SPAWN_RATE \
    --run-time ${RUN_TIME}s \
    --headless \
    --html "$RESULTS_DIR/stack.html" \
    --csv "$RESULTS_DIR/stack" \
    --only-summary

if [ $? -ne 0 ]; then
    echo "❌ Stack benchmark failed"
    exit 1
fi

echo ""
echo "==================================================================="
echo "  Benchmark Complete"
echo "==================================================================="
echo ""
echo "Results:"
echo "  Baseline (Mock): $RESULTS_DIR/baseline.html"
echo "  Stack Server:    $RESULTS_DIR/stack.html"
echo ""
echo "Open reports:"
echo "  open $RESULTS_DIR/baseline.html  # macOS"
echo "  open $RESULTS_DIR/stack.html  # macOS"
echo "  xdg-open $RESULTS_DIR/baseline.html  # Linux"
echo "  xdg-open $RESULTS_DIR/stack.html  # Linux"
echo ""
