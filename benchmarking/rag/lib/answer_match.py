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
high. Every score is therefore reported next to that constant-reply baseline, which needs no
retrieval and no generation. Containment is only informative above it.
"""

from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter

_CITATION_RE = re.compile(r"<\|[^|>]*\|>")
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")
_ASCII_PUNCTUATION = frozenset(string.punctuation)

# How many of the most common gold answers the constant-reply baseline lists. Listing every
# distinct answer would score 1.0, so the reply is capped; five covers 83% of MultiHOP.
CONSTANT_REPLY_MAX_ANSWERS = 5


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


def contains_answer(prediction: str, gold: str | list[str]) -> bool:
    """Whether any gold answer appears in the prediction as a contiguous run of whole tokens."""
    golds = gold if isinstance(gold, list) else [gold]
    pred_tokens = normalize_answer(prediction)
    for answer in golds:
        gold_tokens = normalize_answer(answer)
        if not gold_tokens:
            continue
        width = len(gold_tokens)
        if any(pred_tokens[i : i + width] == gold_tokens for i in range(len(pred_tokens) - width + 1)):
            return True
    return False


def containment_accuracy(predictions: dict[str, str], ground_truths: dict[str, str | list[str]]) -> float:
    """Fraction of queries, among those present in both dicts, whose reply contains the gold answer."""
    common_qids = set(predictions) & set(ground_truths)
    if not common_qids:
        return 0.0
    hits = sum(contains_answer(predictions[qid], ground_truths[qid]) for qid in common_qids)
    return hits / len(common_qids)


def constant_reply_baseline(
    ground_truths: dict[str, str | list[str]], max_answers: int = CONSTANT_REPLY_MAX_ANSWERS
) -> tuple[str, float]:
    """Best query-independent reply and its containment score.

    The reply lists the ``max_answers`` most common gold answers, so it catches the hedge that
    a single "always Yes" reply misses. It uses no retrieval and no generation.
    """
    counts: Counter[str] = Counter()
    representative: dict[str, str] = {}
    for gold in ground_truths.values():
        answer = gold[0] if isinstance(gold, list) else gold
        key = " ".join(normalize_answer(answer))
        counts[key] += 1
        representative.setdefault(key, answer)
    if not counts:
        return "", 0.0
    reply = " ".join(representative[key] for key, _ in counts.most_common(max_answers))
    return reply, containment_accuracy(dict.fromkeys(ground_truths, reply), ground_truths)


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
