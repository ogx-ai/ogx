# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import statistics
from typing import Any


def aggregate_categorical_count(
    scoring_results: list[dict[str, bool | float | str | list[object] | object | None]],
) -> dict[str, Any]:
    scores = [str(r["score"]) for r in scoring_results]
    unique_scores = sorted(set(scores))
    return {"categorical_count": {s: scores.count(s) for s in unique_scores}}


def aggregate_average(
    scoring_results: list[dict[str, bool | float | str | list[object] | object | None]],
) -> dict[str, Any]:
    return {
        "average": sum(result["score"] for result in scoring_results if result["score"] is not None)
        / len([_ for _ in scoring_results if _["score"] is not None]),
    }


def aggregate_weighted_average(
    scoring_results: list[dict[str, bool | float | str | list[object] | object | None]],
) -> dict[str, Any]:
    return {
        "weighted_average": sum(
            result["score"] * result["weight"]
            for result in scoring_results
            if result["score"] is not None and result["weight"] is not None
        )
        / sum(result["weight"] for result in scoring_results if result["weight"] is not None),
    }


def aggregate_median(
    scoring_results: list[dict[str, bool | float | str | list[object] | object | None]],
) -> dict[str, Any]:
    scores = [r["score"] for r in scoring_results if r["score"] is not None]
    median = statistics.median(scores) if scores else None
    return {"median": median}


def aggregate_accuracy(
    scoring_results: list[dict[str, bool | float | str | list[object] | object | None]],
) -> dict[str, Any]:
    num_correct = sum(result["score"] for result in scoring_results)
    avg_score = num_correct / len(scoring_results)

    return {
        "accuracy": avg_score,
        "num_correct": num_correct,
        "num_total": len(scoring_results),
    }
