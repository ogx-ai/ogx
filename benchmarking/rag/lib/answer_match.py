# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Containment scoring for short gold answers, plus the constant-reply baseline.

MultiHOP gold answers are short (median 1 word: Yes, No, an entity name), while the
file_search template makes replies several sentences long. SQuAD token-F1 then measures
reply length rather than correctness, so MultiHOP is scored by whether the normalized gold
answer appears, as whole tokens, in the normalized reply.

Containment accepts hedged replies (a reply of "Yes No" contains both), and most gold answers
come from a few values, so a query-independent reply that lists the most common answers scores
high; listing every distinct answer scores 1.0. Both sides of that problem are the same fix:
containment is only checked within a fixed word budget (CONTAINMENT_WORD_BUDGET) of the
reply, on both a scored prediction and the constant-reply baseline, so enumerating more
answers than fit in the budget helps neither. The baseline is the highest-scoring reply a 0/1
knapsack (answer -> (frequency, word width)) can build within that same budget, assuming each
answer's credit is independent of what else is packed alongside it; see
constant_reply_baseline for why that assumption, and the "nothing budget-bounded beats this"
property it gives, hold on MultiHopRAG but aren't a general guarantee. Every score is
reported next to the baseline; containment is only informative above it.

The budget is a tradeoff, not a free fix: a reply that states its answer after more than
CONTAINMENT_WORD_BUDGET words of framing scores 0 on that query even if the answer itself is
right. file_search replies are typically multi-sentence with citations, so this can matter;
how often it does hasn't been measured (see README.md).
"""

from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter

_CITATION_RE = re.compile(r"<\|[^|>]*\|>")
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")
_ASCII_PUNCTUATION = frozenset(string.punctuation)

# Word budget (normalized tokens) that containment checks within, on both a scored prediction
# and the constant-reply baseline (see contains_answer and constant_reply_baseline). 20 words
# comfortably fits a direct one-sentence answer with light framing (e.g. "Based on the
# retrieved articles, the company most closely associated with this event is Google." is 15
# words through the answer), while still bounding what a query-independent reply can list:
# on MultiHopRAG it caps the baseline at 0.9202 rather than letting it drift toward 1.0.
CONTAINMENT_WORD_BUDGET = 20


def _is_punctuation(ch: str) -> bool:
    """ASCII punctuation, plus Unicode punctuation such as em dashes, curly quotes and ellipses."""
    return ch in _ASCII_PUNCTUATION or unicodedata.category(ch).startswith("P")


def normalize_answer(text: str) -> list[str]:
    """SQuAD-style normalization (lowercase, no punctuation or articles), as tokens.

    Unlike SQuAD, punctuation (ASCII or Unicode) becomes a space rather than being deleted, so
    "Bankman-Fried" and "Bankman Fried" match and "Yes\u2014both" splits into two tokens. Citation markers such as ``<|file-abc|>`` are dropped first so
    file ids cannot match.
    """
    text = _CITATION_RE.sub(" ", text).lower()
    text = "".join(" " if _is_punctuation(ch) else ch for ch in text)
    return _ARTICLES_RE.sub(" ", text).split()


def contains_answer(prediction: str, gold: str | list[str], word_budget: int = CONTAINMENT_WORD_BUDGET) -> bool:
    """Whether any gold answer appears, as a contiguous run of whole tokens, within the first
    ``word_budget`` normalized words of the prediction.

    The budget applies here, not just in constant_reply_baseline: without it, a prediction
    could enumerate arbitrarily many candidate answers and contain one by chance, with no
    corresponding cost the baseline also pays, making "beats the baseline" meaningless (see
    the module docstring). Callers that build a constant_reply_baseline with a non-default
    word_budget must pass the same value here (containment_accuracy does), or the two sides
    of that comparison stop being bounded by the same constraint.
    """
    golds = gold if isinstance(gold, list) else [gold]
    pred_tokens = normalize_answer(prediction)[:word_budget]
    for answer in golds:
        gold_tokens = normalize_answer(answer)
        if not gold_tokens:
            continue
        width = len(gold_tokens)
        if any(pred_tokens[i : i + width] == gold_tokens for i in range(len(pred_tokens) - width + 1)):
            return True
    return False


def containment_accuracy(
    predictions: dict[str, str],
    ground_truths: dict[str, str | list[str]],
    word_budget: int = CONTAINMENT_WORD_BUDGET,
) -> float:
    """Fraction of queries, among those present in both dicts, whose reply contains the gold
    answer within the first ``word_budget`` normalized words (see contains_answer)."""
    common_qids = set(predictions) & set(ground_truths)
    if not common_qids:
        return 0.0
    hits = sum(contains_answer(predictions[qid], ground_truths[qid], word_budget) for qid in common_qids)
    return hits / len(common_qids)


def constant_reply_baseline(
    ground_truths: dict[str, str | list[str]], word_budget: int = CONTAINMENT_WORD_BUDGET
) -> tuple[str, float]:
    """The highest-scoring query-independent reply buildable within ``word_budget`` normalized
    words, and its containment score. It uses no retrieval and no generation.

    Each distinct gold answer is worth its frequency count and costs its normalized word
    width; picking answers by frequency alone (ignoring width) is not optimal, since e.g. a
    less frequent one-word answer can be worth more per word than a more frequent multi-word
    one. This is a 0/1 knapsack over that trade-off, treating each distinct answer's value as
    independent of every other one it packs alongside.

    That independence assumption isn't quite true: contains_answer does substring matching, so
    one gold answer's tokens can be a contiguous run inside another's (MultiHopRAG has a few
    such pairs, e.g. "Taylor Swift" inside "Taylor Swift and Travis Kelce"), and packing the
    longer one for free credits the shorter one's queries too -- the knapsack doesn't see that
    bonus, so in principle a reply exploiting it could beat this one. It doesn't on
    MultiHopRAG (checked by forcing each overlapping answer in as a bundle with the answers
    inside it, which scores no higher), but that's true of this dataset, not a general
    guarantee the knapsack enforces for arbitrary ground_truths.
    """
    counts: Counter[str] = Counter()
    representative: dict[str, str] = {}
    for gold in ground_truths.values():
        answer = gold[0] if isinstance(gold, list) else gold
        key = " ".join(normalize_answer(answer))
        if not key:
            continue
        counts[key] += 1
        representative.setdefault(key, answer)
    if not counts:
        return "", 0.0

    # dp[b] / chosen[b]: best score and the answer keys achieving it, using <= b words.
    # Standard 0/1 knapsack: iterating each item's capacity slots from high to low ensures it
    # is considered at most once (dp[b - width] still reflects only earlier items).
    dp = [0] * (word_budget + 1)
    chosen: list[list[str]] = [[] for _ in range(word_budget + 1)]
    for key, count in counts.items():
        width = len(key.split())
        if width > word_budget:
            continue
        for b in range(word_budget, width - 1, -1):
            candidate_score = dp[b - width] + count
            if candidate_score > dp[b]:
                dp[b] = candidate_score
                chosen[b] = [*chosen[b - width], key]

    best_budget_used = max(range(word_budget + 1), key=lambda b: dp[b])
    reply = " ".join(representative[key] for key in chosen[best_budget_used])
    return reply, containment_accuracy(dict.fromkeys(ground_truths, reply), ground_truths, word_budget)


def baseline_warning(metrics: dict) -> str | None:
    """Warning for a MultiHOP run that does not clear the constant-reply baseline, else None."""
    if "containment" not in metrics:
        return (
            "scored with SQuAD token-F1 only, which is dominated by reply length on MultiHOP "
            "and cannot rank systems; re-run to get containment and the constant-reply baseline"
        )
    baseline = metrics.get("constant_baseline")
    if baseline is None:
        return "no constant-reply baseline recorded; re-run to get one"
    if metrics["containment"] <= baseline:
        return (
            f"containment {metrics['containment']:.4f} does not beat the constant-reply baseline "
            f"{baseline:.4f} (always replying {metrics.get('constant_reply', '?')!r})"
        )
    return None
