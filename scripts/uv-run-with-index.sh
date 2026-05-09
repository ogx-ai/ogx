#!/bin/bash
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

set -euo pipefail

# Add pre-commit golang environment to PATH if it exists
# This allows Python scripts to access tools installed by golang hooks (e.g., oasdiff)
if [[ -n "${HOME:-}" ]]; then
  GOLANG_BIN=$(find "$HOME/.cache/pre-commit" -path "*/golangenv-default/bin" -type d 2>/dev/null | head -1)
  if [[ -n "$GOLANG_BIN" ]]; then
    export PATH="$GOLANG_BIN:$PATH"
  fi
fi

# Detect current branch and target branch
# In GitHub Actions, use GITHUB_REF/GITHUB_BASE_REF
if [[ -n "${GITHUB_REF:-}" ]]; then
  BRANCH="${GITHUB_REF#refs/heads/}"
else
  BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")
fi

# For PRs, check the target branch
if [[ -n "${GITHUB_BASE_REF:-}" ]]; then
  TARGET_BRANCH="${GITHUB_BASE_REF}"
else
  TARGET_BRANCH=$(git rev-parse --abbrev-ref HEAD@{upstream} 2>/dev/null | sed 's|origin/||' || echo "")
fi

# Check if on a release branch or targeting one, or OGX_RELEASE_MODE is set
IS_RELEASE=false
if [[ "$BRANCH" =~ ^release-[0-9]+\.[0-9]+\.x$ ]]; then
  IS_RELEASE=true
elif [[ "$TARGET_BRANCH" =~ ^release-[0-9]+\.[0-9]+\.x$ ]]; then
  IS_RELEASE=true
elif [[ "${OGX_RELEASE_MODE:-}" == "true" ]]; then
  IS_RELEASE=true
fi

# On release branches, use test.pypi as extra index for RC versions
if [[ "$IS_RELEASE" == "true" ]]; then
  export UV_EXTRA_INDEX_URL="https://test.pypi.org/simple/"
  export UV_INDEX_STRATEGY="unsafe-best-match"
fi

# Run uv with all arguments passed through
exec uv "$@"
