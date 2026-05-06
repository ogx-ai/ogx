#!/usr/bin/env bash
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
#
# Fails if any GitHub Actions workflow uses an external action without a full SHA pin.

set -euo pipefail

failed=0

# Find all workflow YAML files

# Use GitHub Actions error format
# ::error file={name},line={line},col={col}::{message}

for file in $(find .github/workflows/ -type f \( -name "*.yml" -o -name "*.yaml" \)); do
    IFS=$'\n'
    # Get line numbers for each 'uses:'
    while IFS= read -r match; do
        line_num=$(echo "$match" | cut -d: -f1)
        line=$(echo "$match" | cut -d: -f2-)
        ref=$(echo "$line" | sed -E 's/.*@([A-Za-z0-9._-]+).*/\1/')
        if ! [[ $ref =~ ^[0-9a-fA-F]{40}$ ]]; then
            # Output in GitHub Actions annotation format
            echo "::error file=$file,line=$line_num::uses non-SHA action ref: $line"
            failed=1
        fi
    done < <(grep -n -E '^.*uses:[^@]+@[^ ]+' "$file")
done

exit $failed
