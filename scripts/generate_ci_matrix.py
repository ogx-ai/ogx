#!/usr/bin/env python3
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Generate CI test matrix from ci_matrix.json with schedule/input overrides.

This script is used by .github/workflows/integration-tests.yml to generate
the test matrix dynamically based on the CI_MATRIX definition.
"""

import json
import sys
from pathlib import Path

CI_MATRIX_FILE = Path(__file__).parent.parent / "tests/integration/ci_matrix.json"

with open(CI_MATRIX_FILE) as f:
    matrix_config = json.load(f)

DEFAULT_MATRIX = matrix_config["default"]
SCHEDULE_MATRICES: dict[str, list[dict[str, str]]] = matrix_config.get("schedules", {})

# Maps path prefixes to the test setup names they should trigger.
# Order matters: more specific paths should come before broader ones.
PROVIDER_PATH_TO_SETUPS: list[tuple[str, list[str]]] = [
    ("src/ogx/providers/remote/inference/ollama/", ["ollama", "ollama-vision", "ollama-postgres", "ollama-reasoning"]),
    ("src/ogx/providers/remote/inference/openai/", ["gpt", "gpt-reasoning"]),
    ("src/ogx/providers/remote/inference/azure/", ["azure"]),
    ("src/ogx/providers/remote/inference/bedrock/", ["bedrock"]),
    ("src/ogx/providers/remote/inference/vllm/", ["vllm"]),
    ("src/ogx/providers/remote/inference/watsonx/", ["watsonx"]),
    ("src/ogx/providers/remote/inference/vertexai/", ["vertexai"]),
    ("src/ogx/providers/remote/inference/gemini/", ["gemini"]),
]

# Changes to these paths trigger the full matrix (core/shared code).
CORE_PATHS = [
    "src/ogx/core/",
    "src/ogx/providers/utils/",
    "src/ogx/providers/inline/responses/",
    "src/ogx/providers/inline/agents/",
    "src/ogx_api/",
    "tests/integration/conftest",
]


def _setups_from_changed_files(changed_files: list[str]) -> set[str] | None:
    """Determine which test setups are needed based on changed files.

    Returns None if core code changed (meaning: run everything).
    Returns a set of setup names if only provider-specific code changed.
    """
    setups: set[str] = set()

    for filepath in changed_files:
        # Core paths → run everything
        if any(filepath.startswith(core) for core in CORE_PATHS):
            return None

        matched = False
        for prefix, setup_names in PROVIDER_PATH_TO_SETUPS:
            if filepath.startswith(prefix):
                setups.update(setup_names)
                matched = True
                break

        if not matched and filepath.startswith("src/ogx/providers/remote/inference/"):
            return None

    return setups


def _filter_matrix_by_setups(matrix: list[dict], setups: set[str]) -> list[dict]:
    """Filter matrix configs to only those matching the given setups."""
    # Always include ollama/base as a baseline smoke test
    setups.add("ollama")
    return [config for config in matrix if config.get("setup") in setups]


def generate_matrix(schedule="", test_setup="", matrix_key="default", changed_files: list[str] | None = None):
    """
    Generate test matrix based on schedule, manual input, or matrix key.

    Args:
        schedule: GitHub cron schedule string (e.g., "1 0 * * 0" for weekly)
        test_setup: Manual test setup input (e.g., "ollama-vision")
        matrix_key: Matrix configuration key from ci_matrix.json (e.g., "default", "stainless")
        changed_files: List of changed file paths for targeted PR testing

    Returns:
        Matrix configuration as JSON string
    """
    # Weekly scheduled test matrices (highest priority)
    if schedule and schedule in SCHEDULE_MATRICES:
        matrix = SCHEDULE_MATRICES[schedule]
    # Manual input for specific setup
    elif test_setup == "ollama-vision":
        matrix = [{"suite": "vision", "setup": "ollama-vision"}]
    # Use specified matrix key from ci_matrix.json
    elif matrix_key:
        if matrix_key not in matrix_config:
            raise ValueError(f"Invalid matrix_key '{matrix_key}'. Available keys: {list(matrix_config.keys())}")
        matrix = matrix_config[matrix_key]
    # Default: use JSON-defined default matrix
    else:
        matrix = DEFAULT_MATRIX

    # For PRs with changed files, filter to only relevant provider tests
    if changed_files is not None:
        setups = _setups_from_changed_files(changed_files)
        if setups is not None:
            filtered = _filter_matrix_by_setups(matrix, setups)
            if filtered:
                matrix = filtered
                print(
                    f"Filtered matrix to {len(matrix)} config(s) based on changed files: {', '.join(sorted(setups))}",
                    file=sys.stderr,
                )
            else:
                print("No matching configs after filtering, using full matrix", file=sys.stderr)

    # GitHub Actions expects {"include": [...]} format
    return json.dumps({"include": matrix})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate CI test matrix")
    parser.add_argument("--schedule", default="", help="GitHub schedule cron string")
    parser.add_argument("--test-setup", default="", help="Manual test setup input")
    parser.add_argument("--matrix-key", default="default", help="Matrix configuration key from ci_matrix.json")
    parser.add_argument("--changed-files", default="", help="Newline-separated list of changed file paths")

    args = parser.parse_args()

    changed = [f for f in args.changed_files.strip().split("\n") if f] if args.changed_files.strip() else None
    print(generate_matrix(args.schedule, args.test_setup, args.matrix_key, changed))
