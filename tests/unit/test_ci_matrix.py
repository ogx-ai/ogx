# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Guards on tests/integration/ci_matrix.json.

ci_matrix.json is a static file: nothing generates it from suites.py, so these
tests are what keep the two in agreement. In particular, setups flagged
matrix_excluded (dedicated-workflow lanes like the k8s auth lane) must stay out
of the matrix, and every one of them needs a suite that defaults to it so
cleanup_recordings.py counts its recordings as referenced.
"""

import json
from pathlib import Path

from tests.integration.suites import SETUP_DEFINITIONS, SUITE_DEFINITIONS

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_MATRIX_FILE = REPO_ROOT / "tests" / "integration" / "ci_matrix.json"


def _all_matrix_entries() -> list[dict[str, str]]:
    matrix = json.loads(CI_MATRIX_FILE.read_text())
    entries = list(matrix["default"])
    for configs in matrix.get("schedules", {}).values():
        entries.extend(configs)
    return entries


def test_matrix_lanes_reference_known_suites_and_setups():
    for entry in _all_matrix_entries():
        assert entry["suite"] in SUITE_DEFINITIONS, f"Unknown suite in ci_matrix.json: {entry['suite']}"
        assert entry["setup"] in SETUP_DEFINITIONS, f"Unknown setup in ci_matrix.json: {entry['setup']}"


def test_matrix_excluded_setups_are_absent_from_ci_matrix():
    excluded = {name for name, setup in SETUP_DEFINITIONS.items() if setup.matrix_excluded}
    in_matrix = {entry["setup"] for entry in _all_matrix_entries()}
    offenders = excluded & in_matrix
    assert not offenders, (
        f"matrix_excluded setups must not appear in ci_matrix.json (they run only under "
        f"their dedicated workflow, which the standard runner cannot serve): {sorted(offenders)}"
    )


def test_matrix_excluded_setups_have_a_defaulting_suite():
    for name, setup in SETUP_DEFINITIONS.items():
        if not setup.matrix_excluded:
            continue
        defaulting = [suite.name for suite in SUITE_DEFINITIONS.values() if suite.default_setup == name]
        assert defaulting, (
            f"matrix_excluded setup '{name}' needs a suite with default_setup='{name}' so "
            f"cleanup_recordings.py collects its test IDs and keeps its recordings"
        )
