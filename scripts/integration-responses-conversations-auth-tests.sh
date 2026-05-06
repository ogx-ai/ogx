#!/bin/bash
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

set -euo pipefail

# Integration auth tests for responses and conversations in OGX
# This script tests authentication and authorization (ABAC) functionality
# Expects token files to be created before running (e.g., by CI workflow or manual setup)

echo ""
echo "Running conversations isolation tests..."

# Set tokens for pytest-based access control tests
export ALICE_TOKEN=$(cat ogx-user1-token || echo "alice-token")
export BOB_TOKEN=$(cat ogx-user2-token || echo "bob-token")

# Use same port as server when run as post-command (e.g. from run-and-record-tests)
OGX_SERVER_URL="http://127.0.0.1:${OGX_PORT:-8321}"

# Run conversations access control tests using pytest
# These tests verify that users cannot access each other's conversations
uv run pytest tests/integration/conversations/test_openai_conversations.py \
    -k "TestConversationAccessControl" \
    --stack-config="$OGX_SERVER_URL" \
    -v -s \
    --color=yes || exit 1

echo ""
echo "✓ Conversations isolation tests completed successfully!"

# Run responses access control tests if INFERENCE_MODEL is set
# Uses record-if-missing mode: replays from recordings if available, records if API key is set
if [ -n "${INFERENCE_MODEL:-}" ]; then
    echo ""
    echo "Running responses isolation tests..."
    echo "  Mode: ${OGX_TEST_INFERENCE_MODE:-replay}"
    echo "  Recording dir: ${OGX_TEST_RECORDING_DIR:-default}"

    uv run pytest tests/integration/responses/test_responses_access_control.py \
        -k "TestResponsesAccessControl" \
        --stack-config="$OGX_SERVER_URL" \
        --text-model="$INFERENCE_MODEL" \
        --inference-mode="${OGX_TEST_INFERENCE_MODE:-replay}" \
        -v -s \
        --color=yes || exit 1

    echo ""
    echo "✓ Responses isolation tests completed successfully!"
else
    echo ""
    echo "⚠ Skipping responses isolation tests (INFERENCE_MODEL not set)"
fi
