#!/bin/bash
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Check API spec for breaking changes using oasdiff
# Compares staged/working API spec against HEAD to detect breaking changes
# before they are committed. This provides early feedback for API stability.
#
# The check is skipped if the commit message indicates an intentional breaking
# change (contains "!:" like "feat!:" or has a "BREAKING CHANGE:" footer).
#
# Note: OpenAI compatibility is tracked separately via the openai-coverage
# pre-commit hook (see scripts/openai_coverage.py).

set -euo pipefail

# Check if breaking changes are intended (skip if so)
# Look at staged commit message or recent commits for breaking change indicators
check_breaking_change_intended() {
    # Check if there's a commit message file (during commit)
    if [ -f ".git/COMMIT_EDITMSG" ]; then
        if grep -qE '!:|BREAKING CHANGE:' .git/COMMIT_EDITMSG 2>/dev/null; then
            return 0
        fi
    fi

    # Check recent commits for breaking change indicators
    # This handles the case where changes are already committed
    if git log -1 --format="%B" 2>/dev/null | grep -qE '!:|BREAKING CHANGE:'; then
        return 0
    fi

    return 1
}

if check_breaking_change_intended; then
    echo "Skipping API conformance check: breaking change intended"
    exit 0
fi

# Determine which spec to check (prefer stable, fall back to monolithic)
if [ -f "docs/static/stable-ogx-spec.yaml" ]; then
    SPEC="docs/static/stable-ogx-spec.yaml"
elif [ -f "docs/static/ogx-spec.yaml" ]; then
    SPEC="docs/static/ogx-spec.yaml"
else
    echo "No API spec found"
    exit 0
fi

# Get the HEAD version for comparison
BASE_SPEC=$(mktemp)
trap "rm -f $BASE_SPEC" EXIT
if ! git show HEAD:"$SPEC" > "$BASE_SPEC" 2>/dev/null; then
    echo "No previous version of $SPEC in HEAD - skipping comparison"
    exit 0
fi

# Check for breaking changes against HEAD
echo "Checking for breaking changes in $SPEC..."
oasdiff breaking --fail-on ERR "$BASE_SPEC" "$SPEC" --match-path '^/v1/'
echo "No breaking changes detected."
