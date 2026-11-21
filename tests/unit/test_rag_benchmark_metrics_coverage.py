# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Coverage reporting for the RAG benchmark metrics.

The benchmark runners skip a conversation when a query or ingestion raises, so
those query ids never reach the metric functions. The averages are taken over
the queries that did arrive, which means a partial run and a complete run look
the same in the output unless the counts are reported alongside.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("pytrec_eval")
pytest.importorskip("evaluate")

# `benchmarking/` is not a package and is not on the path by default.
_BENCHMARKING_ROOT = Path(__file__).resolve().parents[2]
if str(_BENCHMARKING_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCHMARKING_ROOT))

from benchmarking.rag.lib.metrics import retrieval_metrics  # noqa: E402


def _qrels(n: int) -> dict[str, dict[str, int]]:
    return {f"q{i}": {f"d{i}": 1} for i in range(n)}


def _results(qids: list[str]) -> dict[str, dict[str, float]]:
    return {qid: {f"d{qid[1:]}": 1.0} for qid in qids}


def test_reports_counts_when_all_queries_present():
    metrics = retrieval_metrics(_qrels(3), _results(["q0", "q1", "q2"]), k_values=[5])
    assert metrics["num_scored_queries"] == 3
    assert metrics["num_missing_queries"] == 0


def test_missing_queries_are_counted_not_hidden():
    # Two of five queries errored during the run, so they never reached the metrics.
    metrics = retrieval_metrics(_qrels(5), _results(["q0", "q1", "q2"]), k_values=[5])
    assert metrics["num_scored_queries"] == 3
    assert metrics["num_missing_queries"] == 2


def test_partial_run_is_distinguishable_from_complete_run():
    complete = retrieval_metrics(_qrels(4), _results(["q0", "q1", "q2", "q3"]), k_values=[5])
    partial = retrieval_metrics(_qrels(4), _results(["q0", "q1"]), k_values=[5])
    # Both score 1.0 on the queries they saw; only the counts tell them apart.
    assert complete["ndcg_cut_5"] == partial["ndcg_cut_5"]
    assert (complete["num_scored_queries"], complete["num_missing_queries"]) != (
        partial["num_scored_queries"],
        partial["num_missing_queries"],
    )


def test_no_results_at_all_reports_every_query_missing():
    metrics = retrieval_metrics(_qrels(3), {}, k_values=[5])
    assert metrics["num_scored_queries"] == 0
    assert metrics["num_missing_queries"] == 3
    assert metrics["ndcg_cut_5"] == 0.0
