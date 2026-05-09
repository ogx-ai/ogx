#!/usr/bin/env bash

# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

PYTHONPATH=${PYTHONPATH:-}
THIS_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"

set -euo pipefail


stack_dir=$(dirname "$THIS_DIR")
PYTHONPATH=$PYTHONPATH:$stack_dir \
  python3 -m scripts.openapi_generator "$stack_dir"/docs/static

cp "$stack_dir"/docs/static/stainless-ogx-spec.yaml "$stack_dir"/client-sdks/stainless/openapi.yml
PYTHONPATH=$PYTHONPATH:$stack_dir \
  python3 -m scripts.openapi_generator.stainless_config.generate_config
