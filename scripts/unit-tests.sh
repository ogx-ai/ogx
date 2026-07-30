#!/bin/sh

# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

PYTHON_VERSION=${PYTHON_VERSION:-3.12}

set -e

# Always run this at the end, even if something fails
cleanup() {
    echo "Generating coverage report..."
    uv run --python "$PYTHON_VERSION" coverage html -d htmlcov-$PYTHON_VERSION
}
trap cleanup EXIT

command -v uv >/dev/null 2>&1 || { echo >&2 "uv is required but it's not installed. Exiting."; exit 1; }

uv python find "$PYTHON_VERSION"
FOUND_PYTHON=$?
if [ $FOUND_PYTHON -ne 0 ]; then
     uv python install "$PYTHON_VERSION"
fi

# Run unit tests with coverage
uv run --python "$PYTHON_VERSION" --with-editable . --group unit \
    coverage run --source=src/ogx -m pytest -s -v tests/unit/ "$@"
