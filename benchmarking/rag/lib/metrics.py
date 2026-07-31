# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Retrieval and answer quality metrics — delegates to pytrec_eval, evaluate, rouge-score."""

from __future__ import annotations

import evaluate
import pytrec_eval
from rouge_score import rouge_scorer

# ---------------------------------------------------------------------------
# Retrieval metrics via pytrec_eval
# ---------------------------------------------------------------------------


def retrieval_metrics(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Compute retrieval metrics using pytrec_eval.

    Args:
        qrels: {query_id: {doc_id: relevance}} — ground truth.
        results: {query_id: {doc_id: score}} — system output.
        k_values: Cutoff values for nDCG, recall, MAP. Default [5, 10].

    Returns:
        Aggregated metrics dict, e.g. {"ndcg_cut_10": 0.45, "recall_10": 0.78, ...}
    """
    if k_values is None:
        k_values = [5, 10]

    metrics_set = set()
    for k in k_values:
        metrics_set.add(f"ndcg_cut_{k}")
        metrics_set.add(f"recall_{k}")
        metrics_set.add(f"map_cut_{k}")

    # Filter to queries present in both qrels and results
    common_qids = set(qrels.keys()) & set(results.keys())
    if not common_qids:
        return dict.fromkeys(metrics_set, 0.0)

    filtered_qrels = {qid: qrels[qid] for qid in common_qids}
    filtered_results = {qid: results[qid] for qid in common_qids}

    evaluator = pytrec_eval.RelevanceEvaluator(filtered_qrels, metrics_set)
    per_query = evaluator.evaluate(filtered_results)

    # Aggregate: mean across queries
    aggregated = {}
    for metric in metrics_set:
        values = [per_query[qid][metric] for qid in per_query if metric in per_query[qid]]
        aggregated[metric] = sum(values) / len(values) if values else 0.0

    return aggregated


def per_query_retrieval_metrics(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    k: int = 10,
) -> dict[str, dict[str, float]]:
    """Return per-query retrieval metrics at a single cutoff."""
    metrics_set = {f"ndcg_cut_{k}", f"recall_{k}", f"map_cut_{k}"}
    common_qids = set(qrels.keys()) & set(results.keys())
    if not common_qids:
        return {}

    filtered_qrels = {qid: qrels[qid] for qid in common_qids}
    filtered_results = {qid: results[qid] for qid in common_qids}

    evaluator = pytrec_eval.RelevanceEvaluator(filtered_qrels, metrics_set)
    return evaluator.evaluate(filtered_results)


# ---------------------------------------------------------------------------
# Answer quality metrics via HuggingFace evaluate + rouge-score
# ---------------------------------------------------------------------------

# Load the SQuAD metric once (handles EM and token-level F1)
_squad_metric = evaluate.load("squad")
_rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def answer_metrics(
    predictions: dict[str, str],
    ground_truths: dict[str, str | list[str]],
) -> dict[str, float]:
    """Compute aggregated answer quality metrics.

    Uses HuggingFace `evaluate` (SQuAD metric) for EM/F1 and `rouge-score` for ROUGE-L.

    Args:
        predictions: {query_id: predicted_answer}
        ground_truths: {query_id: ground_truth_answer} or {query_id: [answer1, answer2, ...]}

    Returns:
        {"exact_match": ..., "f1": ..., "rouge_l": ...}
    """
    common_qids = sorted(set(predictions.keys()) & set(ground_truths.keys()))
    if not common_qids:
        return {"exact_match": 0.0, "f1": 0.0, "rouge_l": 0.0, "num_queries": 0}

    # Format for HuggingFace squad metric
    hf_predictions = []
    hf_references = []
    for qid in common_qids:
        pred = predictions[qid]
        gt = ground_truths[qid]
        answers = gt if isinstance(gt, list) else [gt]

        hf_predictions.append({"id": qid, "prediction_text": pred})
        hf_references.append(
            {
                "id": qid,
                "answers": {"text": answers, "answer_start": [0] * len(answers)},
            }
        )

    squad_results = _squad_metric.compute(predictions=hf_predictions, references=hf_references)

    # ROUGE-L (take max across multiple references)
    rouge_scores = []
    for qid in common_qids:
        pred = predictions[qid]
        gt = ground_truths[qid]
        answers = gt if isinstance(gt, list) else [gt]
        best_rouge = max(_rouge_scorer.score(ref, pred)["rougeL"].fmeasure for ref in answers)
        rouge_scores.append(best_rouge)

    return {
        "exact_match": squad_results["exact_match"] / 100.0,  # squad returns 0-100
        "f1": squad_results["f1"] / 100.0,
        "rouge_l": sum(rouge_scores) / len(rouge_scores),
        "num_queries": len(common_qids),
    }
