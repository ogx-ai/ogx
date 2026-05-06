#!/usr/bin/env bash
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="$SCRIPT_DIR/config.yaml"

echo "Starting OGX with config: $CONFIG"
echo "Milvus URI: ${MILVUS_URI:-http://localhost:19530}"
echo "Port: 8321"

ogx run "$CONFIG" --port 8321
