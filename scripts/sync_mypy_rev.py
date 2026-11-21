#!/usr/bin/env python3
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Sync the mirrors-mypy pre-commit hook rev with the mypy pin in pyproject.toml.

Dependabot bumps the mypy pin in pyproject.toml (and uv.lock) but never the
mirrors-mypy rev in .pre-commit-config.yaml. When the two diverge, the
pre-commit hook and the project venv run different mypy versions, which
invalidates each other's .mypy_cache entries and can report divergent
errors. This script rewrites the mirrors-mypy rev to match the pin, leaving
the working tree dirty for the caller to commit.

Exit codes:
    0 - rev already matched, or it was synced
    1 - the mypy pin or the mirrors-mypy entry could not be found
"""

import re
import sys

PYPROJECT_PATH = "pyproject.toml"
PRE_COMMIT_CONFIG_PATH = ".pre-commit-config.yaml"
MIRRORS_MYPY_REPO = "https://github.com/pre-commit/mirrors-mypy"


def get_pinned_mypy_version(pyproject: str) -> str | None:
    match = re.search(r'^\s*"mypy==([0-9][0-9.]*)"', pyproject, re.M)
    return match.group(1) if match else None


def get_mirrors_mypy_rev_index(lines: list[str]) -> int | None:
    repo_idx = next(
        (i for i, line in enumerate(lines) if MIRRORS_MYPY_REPO in line),
        None,
    )
    if repo_idx is None:
        return None
    return next(
        (i for i in range(repo_idx + 1, len(lines)) if lines[i].lstrip().startswith("rev:")),
        None,
    )


def main() -> None:
    pyproject = open(PYPROJECT_PATH).read()
    version = get_pinned_mypy_version(pyproject)
    if version is None:
        sys.exit(f"Failed to find a mypy==<version> pin in {PYPROJECT_PATH}")

    with open(PRE_COMMIT_CONFIG_PATH) as f:
        lines = f.readlines()
    rev_idx = get_mirrors_mypy_rev_index(lines)
    if rev_idx is None:
        sys.exit(f"Failed to find the mirrors-mypy rev in {PRE_COMMIT_CONFIG_PATH}")

    expected = f"    rev: v{version}\n"
    if lines[rev_idx] == expected:
        print(f"mirrors-mypy rev already matches the mypy pin (v{version})")
        return

    old = lines[rev_idx].strip()
    lines[rev_idx] = expected
    with open(PRE_COMMIT_CONFIG_PATH, "w") as f:
        f.writelines(lines)
    print(f"Synced mirrors-mypy rev from {old} to v{version} to match the mypy pin in {PYPROJECT_PATH}")


if __name__ == "__main__":
    main()
