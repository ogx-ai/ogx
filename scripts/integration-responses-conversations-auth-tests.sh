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
echo "Running auth isolation tests (suite=auth, setup=auth)..."

# Set tokens for pytest-based access control tests
export ALICE_TOKEN=$(cat ogx-user1-token || echo "alice-token")
export BOB_TOKEN=$(cat ogx-user2-token || echo "bob-token")

# Use same port as server when run as post-command (e.g. from run-and-record-tests)
OGX_SERVER_URL="http://127.0.0.1:${OGX_PORT:-8321}"

# The auth suite (tests/integration/suites.py) selects exactly the access-control
# test classes, and the auth setup supplies the text model — the same
# definitions the dedicated auth workflow uses for its server config. Recording
# follows the standard modes: replay by default, record/record-if-missing when
# OGX_TEST_INFERENCE_MODE and a provider API key are set.
uv run pytest tests/integration \
    --suite=auth \
    --setup=auth \
    --stack-config="$OGX_SERVER_URL" \
    --inference-mode="${OGX_TEST_INFERENCE_MODE:-replay}" \
    -v -s \
    --color=yes || exit 1

echo ""
echo "Auth isolation tests completed successfully!"
