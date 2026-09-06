#!/usr/bin/env bash
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Smoke test a built distribution image: start the container, wait for the
# server to report healthy, and exercise a couple of read-only endpoints. This
# does not call a real inference backend; it only verifies that the image boots
# and serves the API. Provider API keys are stubbed so distros that interpolate
# them can start without real credentials.
set -euo pipefail

IMAGE="${1:-}"
if [ -z "$IMAGE" ]; then
    echo "Usage: $0 IMAGE_NAME" >&2
    exit 1
fi

PORT="${SMOKE_PORT:-8321}"
CONTAINER_NAME="ogx-smoke-$$"
HEALTH_URL="http://localhost:${PORT}/v1/health"
RETRIES="${SMOKE_RETRIES:-30}"
SLEEP_SECONDS="${SMOKE_SLEEP_SECONDS:-5}"

cleanup() {
    echo "Collecting container logs for ${CONTAINER_NAME}:"
    docker logs "$CONTAINER_NAME" 2>&1 | tail -n 200 || true
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Starting container from image: ${IMAGE}"
docker run -d --name "$CONTAINER_NAME" \
    -p "${PORT}:8321" \
    -e OPENAI_API_KEY="smoke-test-dummy-key" \
    -e FIREWORKS_API_KEY="smoke-test-dummy-key" \
    -e TOGETHER_API_KEY="smoke-test-dummy-key" \
    -e GEMINI_API_KEY="smoke-test-dummy-key" \
    -e ANTHROPIC_API_KEY="smoke-test-dummy-key" \
    "$IMAGE" >/dev/null

echo "Waiting for ${HEALTH_URL} (up to $((RETRIES * SLEEP_SECONDS))s)..."
for i in $(seq 1 "$RETRIES"); do
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "::error::Container exited before becoming healthy"
        exit 1
    fi
    if curl -sf "$HEALTH_URL" 2>/dev/null | grep -q "OK"; then
        echo "Server is healthy after $((i * SLEEP_SECONDS))s"
        break
    fi
    if [ "$i" -eq "$RETRIES" ]; then
        echo "::error::Server did not become healthy within $((RETRIES * SLEEP_SECONDS))s"
        exit 1
    fi
    sleep "$SLEEP_SECONDS"
done

# Read-only sanity checks. /v1/models must respond with valid JSON; the models
# list may be empty when no provider credentials are present, which is fine.
echo "Checking /v1/models endpoint..."
if ! curl -sf "http://localhost:${PORT}/v1/models" | python3 -c "import sys, json; json.load(sys.stdin)"; then
    echo "::error::/v1/models did not return valid JSON"
    exit 1
fi

echo "Smoke test passed for ${IMAGE}"
