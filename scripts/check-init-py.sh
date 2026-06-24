#!/usr/bin/env bash
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
#
# Check for missing __init__.py files in Python packages
# This script finds directories that contain Python files but are missing __init__.py

set -euo pipefail

# Use mapfile to get a faster way to iterate over directories
if (( BASH_VERSINFO[0] < 4 )); then
    echo "This script requires Bash 4.0 or higher for mapfile support."
    exit 1
fi

PACKAGE_DIR="${1:-src/ogx}"

if [ ! -d "$PACKAGE_DIR" ]; then
    echo "ERROR: Package directory '$PACKAGE_DIR' does not exist"
    exit 1
fi

# Get all directories with Python files (excluding __init__.py)
mapfile -t py_dirs < <(
    find "$PACKAGE_DIR" \
        -type f \
        -name "*.py" ! -name "__init__.py" \
        ! -path "*/.venv/*" \
        ! -path "*/node_modules/*" \
        -exec dirname {} \; | sort -u
)

missing_init_files=0

for dir in "${py_dirs[@]}"; do
    if [ ! -f "$dir/__init__.py" ]; then
        echo "ERROR: Missing __init__.py in directory: $dir"
        echo "This directory contains Python files but no __init__.py, which may cause packaging issues."
        missing_init_files=1
    fi
done

exit $missing_init_files
