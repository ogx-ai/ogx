# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# compare_results.py lives outside the Python package tree and does `from lib.answer_match
# import ...`, so it needs benchmarking/rag itself on sys.path (not just the one module) to
# resolve that import.
import json
import pathlib
import sys

import pytest
from click.testing import CliRunner

_rag_dir = pathlib.Path(__file__).resolve().parents[2] / "benchmarking" / "rag"
if str(_rag_dir) not in sys.path:
    sys.path.insert(0, str(_rag_dir))

import compare_results  # noqa: E402


def _write_metrics(results_dir, backend: str, benchmark: str, metrics: dict) -> None:
    bench_dir = results_dir / backend / benchmark
    bench_dir.mkdir(parents=True, exist_ok=True)
    (bench_dir / "metrics.json").write_text(json.dumps(metrics))


@pytest.fixture
def results_dir(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    monkeypatch.setattr(compare_results, "RESULTS_DIR", results_dir)
    return results_dir


BELOW_BASELINE_METRICS = {
    "dataset": "multihop",
    "containment": 0.70,
    "constant_baseline": 0.8337,
    "constant_reply": "Yes No",
    "search_mode": "default",
}
ABOVE_BASELINE_METRICS = {
    "dataset": "multihop",
    "containment": 0.90,
    "constant_baseline": 0.8337,
    "constant_reply": "Yes No",
    "search_mode": "default",
}


class TestCompareResultsBaselineMarker:
    """Regression tests for the review finding that the Comparison Summary pivot only ever
    carries the Score column through, so a row that doesn't clear the constant-reply baseline
    looked identical there to one that did -- the warning lived only in a separate list.
    """

    def test_row_below_baseline_is_marked_in_the_main_table(self, tmp_path, results_dir):
        _write_metrics(results_dir, "openai", "multihop", BELOW_BASELINE_METRICS)

        result = CliRunner().invoke(compare_results.main, [])

        assert result.exit_code == 0
        assert "0.7000 !" in result.output

    def test_row_at_or_above_baseline_is_not_marked(self, tmp_path, results_dir):
        _write_metrics(results_dir, "ogx", "multihop", ABOVE_BASELINE_METRICS)

        result = CliRunner().invoke(compare_results.main, [])

        assert result.exit_code == 0
        assert "!" not in result.output
        assert "0.9" in result.output

    def test_pivot_comparison_also_carries_the_marker(self, tmp_path, results_dir):
        _write_metrics(results_dir, "openai", "multihop", BELOW_BASELINE_METRICS)
        _write_metrics(results_dir, "ogx", "multihop", ABOVE_BASELINE_METRICS)

        result = CliRunner().invoke(compare_results.main, [])

        assert result.exit_code == 0
        summary = result.output.split("--- Comparison Summary ---")[1]
        assert "0.7000 !" in summary
        assert "0.9" in summary

    def test_marker_appears_in_json_output_and_json_stays_valid(self, tmp_path, results_dir):
        _write_metrics(results_dir, "openai", "multihop", BELOW_BASELINE_METRICS)
        output_path = tmp_path / "out.json"

        result = CliRunner().invoke(compare_results.main, ["--format", "json", "--output", str(output_path)])

        assert result.exit_code == 0
        data = json.loads(output_path.read_text())  # raises if the marker broke JSON parsing
        assert data[0]["Score"] == "0.7000 !"

    def test_marker_appears_in_csv_output_and_csv_stays_valid(self, tmp_path, results_dir):
        _write_metrics(results_dir, "openai", "multihop", BELOW_BASELINE_METRICS)
        output_path = tmp_path / "out.csv"

        result = CliRunner().invoke(compare_results.main, ["--format", "csv", "--output", str(output_path)])

        assert result.exit_code == 0
        import csv

        with output_path.open() as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["Score"] == "0.7000 !"


class TestCompareResultsWarningsFile:
    """Regression tests for the review finding that --output only ever wrote the table/csv/json
    content; the warnings printed to stderr were never captured anywhere in the file(s) written
    to disk.
    """

    @pytest.mark.parametrize("fmt", ["table", "csv", "json"])
    def test_warnings_are_written_to_a_companion_file(self, tmp_path, results_dir, fmt):
        _write_metrics(results_dir, "openai", "multihop", BELOW_BASELINE_METRICS)
        output_path = tmp_path / f"out.{fmt}"

        result = CliRunner().invoke(compare_results.main, ["--format", fmt, "--output", str(output_path)])

        assert result.exit_code == 0
        warnings_path = output_path.with_name(output_path.name + ".warnings.txt")
        assert warnings_path.exists()
        assert "does not beat the constant-reply baseline" in warnings_path.read_text()

    def test_no_companion_file_when_nothing_to_warn_about(self, tmp_path, results_dir):
        _write_metrics(results_dir, "ogx", "multihop", ABOVE_BASELINE_METRICS)
        output_path = tmp_path / "out.json"

        result = CliRunner().invoke(compare_results.main, ["--format", "json", "--output", str(output_path)])

        assert result.exit_code == 0
        assert not output_path.with_name(output_path.name + ".warnings.txt").exists()

    def test_companion_file_is_not_written_without_output(self, tmp_path, results_dir):
        _write_metrics(results_dir, "openai", "multihop", BELOW_BASELINE_METRICS)

        result = CliRunner().invoke(compare_results.main, [])

        assert result.exit_code == 0
        assert "WARNING multihop" in result.output
