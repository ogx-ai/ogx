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
from pathlib import Path

CI_MATRIX_FILE = Path(__file__).parent.parent / "tests/integration/ci_matrix.json"

with open(CI_MATRIX_FILE) as f:
    matrix_config = json.load(f)

DEFAULT_MATRIX = matrix_config["default"]
SCHEDULE_MATRICES: dict[str, list[dict[str, str]]] = matrix_config.get("schedules", {})


def generate_matrix(schedule="", test_setup="", matrix_key="default"):
    """
    Generate test matrix based on schedule, manual input, or matrix key.

    Args:
        schedule: GitHub cron schedule string (e.g., "1 0 * * 0" for weekly)
        test_setup: Manual test setup input (e.g., "ollama-vision")
        matrix_key: Matrix configuration key from ci_matrix.json (e.g., "default", "stainless")

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

    # GitHub Actions expects {"include": [...]} format
    return json.dumps({"include": matrix})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate CI test matrix")
    parser.add_argument("--schedule", default="", help="GitHub schedule cron string")
    parser.add_argument("--test-setup", default="", help="Manual test setup input")
    parser.add_argument("--matrix-key", default="default", help="Matrix configuration key from ci_matrix.json")

    args = parser.parse_args()

    print(generate_matrix(args.schedule, args.test_setup, args.matrix_key))
