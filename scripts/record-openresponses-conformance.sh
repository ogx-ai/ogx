#!/bin/bash
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

set -euo pipefail

# Records OpenResponses conformance test interactions against a local ogx
# server so that CI can replay them without a live API key.
#
# Run this script whenever you add new compliance tests or the openresponses
# test suite changes, then commit the resulting recordings:
#
#   git add tests/integration/openresponses/recordings/
#   git commit -m "chore: update OpenResponses conformance recordings"
#
# Requirements:
#   - OPENAI_API_KEY must be set
#   - uv must be available (https://github.com/astral-sh/uv)
#   - bun will be installed automatically if missing

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RECORDING_DIR="$REPO_ROOT/tests/integration/openresponses"
PORT="${PORT:-8321}"
INFERENCE_MODEL="${INFERENCE_MODEL:-openai/gpt-4o-mini}"
OPENRESPONSES_DIR="${OPENRESPONSES_DIR:-/tmp/openresponses}"
LOG_FILE="/tmp/openresponses-server.log"

# ── Cleanup ────────────────────────────────────────────────────────────────────
SERVER_PID=""
cleanup() {
    if [[ -n "$SERVER_PID" ]]; then
        echo ""
        echo "Stopping ogx server (PID $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# ── Preflight checks ───────────────────────────────────────────────────────────
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "Error: OPENAI_API_KEY must be set to record conformance test interactions."
    exit 1
fi

cd "$REPO_ROOT"

# ── Bun ────────────────────────────────────────────────────────────────────────
if ! command -v bun &>/dev/null; then
    echo "=== Installing bun ==="
    curl -fsSL https://bun.sh/install | bash
    export PATH="$HOME/.bun/bin:$PATH"
fi

# ── OpenResponses CLI ──────────────────────────────────────────────────────────
if [[ -d "$OPENRESPONSES_DIR/.git" ]]; then
    echo "=== Updating openresponses ==="
    git -C "$OPENRESPONSES_DIR" pull --ff-only
else
    echo "=== Cloning openresponses ==="
    git clone --depth=1 https://github.com/openresponses/openresponses.git "$OPENRESPONSES_DIR"
fi
echo "=== Installing openresponses dependencies ==="
(cd "$OPENRESPONSES_DIR" && bun install)

# ── ogx provider dependencies ─────────────────────────────────────────
echo "=== Installing ci-tests distro dependencies ==="
ogx stack list-deps ci-tests --format uv | sh

# ── Start server ───────────────────────────────────────────────────────────────
echo "=== Starting ogx server (record-if-missing) ==="
mkdir -p "$(dirname "$LOG_FILE")"

OGX_TEST_INFERENCE_MODE=record-if-missing \
OGX_TEST_RECORDING_DIR="$RECORDING_DIR" \
OGX_LOG_WIDTH=200 \
nohup ogx stack run ci-tests --port "$PORT" \
    > "$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# ── Wait for health ────────────────────────────────────────────────────────────
echo "Waiting for ogx server to be ready..."
for i in {1..60}; do
    if curl -sf "http://localhost:$PORT/v1/health" 2>/dev/null | grep -q "OK"; then
        echo "Server is ready!"
        break
    fi
    if [[ $i -eq 60 ]]; then
        echo "Server failed to start within 120 seconds. Log:"
        cat "$LOG_FILE"
        exit 1
    fi
    sleep 2
done

# ── Run compliance tests ───────────────────────────────────────────────────────
echo ""
echo "=== Running OpenResponses compliance tests ==="
echo ""
(
    cd "$OPENRESPONSES_DIR"
    bun run bin/compliance-test.ts \
        --base-url "http://localhost:$PORT/v1" \
        --api-key "ogx" \
        --model "$INFERENCE_MODEL" \
        --verbose
) || true   # continue-on-error: failures here are expected while the implementation has gaps

# ── Summary ────────────────────────────────────────────────────────────────────
echo ""
echo "=== Recordings written to: $RECORDING_DIR/recordings/ ==="
RECORDING_COUNT=$(find "$RECORDING_DIR/recordings" -name "*.json" 2>/dev/null | wc -l | tr -d ' ')
echo "Total recording files: $RECORDING_COUNT"
echo ""
if [[ "$RECORDING_COUNT" -gt 0 ]]; then
    echo "Commit the recordings to include them in CI:"
    echo ""
    echo "  git add tests/integration/openresponses/recordings/"
    echo "  git commit -m 'chore: add OpenResponses conformance recordings'"
else
    echo "No recordings were created. Check $LOG_FILE for server errors."
fi
