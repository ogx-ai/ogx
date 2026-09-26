# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# The benchmark harness lives outside the Python package tree, so import the
# dependency-free scorer module by path.
import importlib.util
import pathlib

import pytest

_script_path = pathlib.Path(__file__).resolve().parents[2] / "benchmarking" / "rag" / "lib" / "answer_match.py"
_spec = importlib.util.spec_from_file_location("answer_match", _script_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

normalize_answer = _mod.normalize_answer
contains_answer = _mod.contains_answer
containment_accuracy = _mod.containment_accuracy
constant_reply_baseline = _mod.constant_reply_baseline
baseline_warning = _mod.baseline_warning

# Shaped like MultiHopRAG: mostly one-word gold answers, dominated by Yes/No.
GOLD = {
    "1": "Yes",
    "2": "Yes",
    "3": "No",
    "4": "Insufficient information.",
    "5": "Sam Bankman-Fried",
    "6": "Google",
}


def _pad(answer: str, tokens: int) -> str:
    return answer + " " + " ".join(["lorem"] * tokens)


class TestNormalizeAnswer:
    def test_lowercases_and_drops_punctuation_and_articles(self):
        assert normalize_answer("The Answer, is: A Cat!") == ["answer", "is", "cat"]

    def test_hyphenated_names_split_into_tokens(self):
        assert normalize_answer("Sam Bankman-Fried") == normalize_answer("Sam Bankman Fried")

    def test_citation_markers_are_dropped(self):
        assert normalize_answer("Yes <|file-abc123|>.") == ["yes"]

    @pytest.mark.parametrize(
        "text",
        ["Yes\u2014both agree", "\u201cYes\u201d both agree", "Yes\u2026 both agree", "Yes\u2013both agree"],
    )
    def test_unicode_punctuation_separates_tokens(self, text):
        assert normalize_answer(text) == ["yes", "both", "agree"]

    def test_ascii_symbols_are_still_separators(self):
        assert normalize_answer("$5+6|7") == ["5", "6", "7"]


class TestContainsAnswer:
    def test_answer_inside_a_long_cited_reply(self):
        reply = "Based on the retrieved articles, the answer is Google <|file-1|>. It was reported twice."
        assert contains_answer(reply, "Google")

    def test_whole_tokens_only(self):
        assert not contains_answer("I know nothing", "no")
        assert not contains_answer("Googled it", "Google")

    def test_multi_word_answer_must_be_contiguous_and_in_order(self):
        assert contains_answer("there is insufficient information here", "Insufficient information.")
        assert not contains_answer("information is insufficient", "Insufficient information.")

    def test_any_of_several_gold_answers_counts(self):
        assert contains_answer("It was Paris", ["London", "Paris"])

    def test_empty_gold_never_matches(self):
        assert not contains_answer("anything at all", "")
        assert not contains_answer("anything at all", "The.")

    def test_empty_reply_never_matches(self):
        assert not contains_answer("", "Yes")

    @pytest.mark.parametrize(
        "reply",
        ["Yes\u2014both reports agree.", "The answer is \u201cYes\u201d.", "Yes\u2026 they agree", "\u2018Yes\u2019"],
    )
    def test_answer_next_to_unicode_punctuation_is_found(self, reply):
        assert contains_answer(reply, "Yes")


class TestContainmentAccuracy:
    """The controls from the issue: wrong answers must fail and correct ones must pass at any length."""

    def test_empty_replies_fail(self):
        assert containment_accuracy(dict.fromkeys(GOLD, ""), GOLD) == 0.0

    def test_gold_copied_verbatim_passes(self):
        assert containment_accuracy(dict(GOLD), GOLD) == 1.0

    def test_wrong_answers_fail_however_long(self):
        assert containment_accuracy(dict.fromkeys(GOLD, "zzz"), GOLD) == 0.0
        assert containment_accuracy(dict.fromkeys(GOLD, _pad("zzz", 250)), GOLD) == 0.0

    @pytest.mark.parametrize("tokens", [2, 10, 60, 120, 250])
    def test_correct_answers_score_the_same_at_every_length(self, tokens):
        replies = {qid: _pad(answer, tokens) for qid, answer in GOLD.items()}
        assert containment_accuracy(replies, GOLD) == 1.0

    def test_verbose_correct_system_beats_terse_partially_correct_one(self):
        """The inversion SQuAD F1 produced: fewer correct answers scored higher because they were shorter."""
        verbose_all_correct = {qid: _pad(answer, 120) for qid, answer in GOLD.items()}
        terse_half_correct = {qid: (answer if int(qid) <= 3 else "zzz") for qid, answer in GOLD.items()}
        assert containment_accuracy(verbose_all_correct, GOLD) > containment_accuracy(terse_half_correct, GOLD)

    def test_only_queries_present_in_both_are_scored(self):
        assert containment_accuracy({"1": "Yes", "99": "Yes"}, GOLD) == 1.0

    def test_no_overlap_scores_zero(self):
        assert containment_accuracy({"99": "Yes"}, GOLD) == 0.0


# 40 queries: a few categorical answers plus a long tail of one-off entities, like MultiHopRAG.
LONG_TAIL = {
    **{f"y{i}": "Yes" for i in range(10)},
    **{f"n{i}": "No" for i in range(8)},
    **{f"i{i}": "Insufficient information." for i in range(5)},
    **{f"s{i}": "Sam Bankman-Fried" for i in range(4)},
    **{f"g{i}": "Google" for i in range(3)},
    **{f"t{i}": f"Entity {i}" for i in range(10)},
}


class TestConstantReplyBaseline:
    def test_reply_lists_the_most_common_answers_and_scores_their_coverage(self):
        reply, score = constant_reply_baseline(LONG_TAIL)
        assert reply == "Yes No Insufficient information. Sam Bankman-Fried Google"
        assert score == pytest.approx(30 / 40)

    def test_cap_is_configurable(self):
        reply, score = constant_reply_baseline(LONG_TAIL, max_answers=1)
        assert (reply, score) == ("Yes", pytest.approx(10 / 40))

    def test_case_and_punctuation_variants_count_as_one_answer(self):
        reply, score = constant_reply_baseline({"1": "Yes", "2": "yes.", "3": "No"}, max_answers=1)
        assert reply == "Yes"
        assert score == pytest.approx(2 / 3)

    def test_empty_ground_truths(self):
        assert constant_reply_baseline({}) == ("", 0.0)

    def test_a_query_independent_hedge_of_common_answers_does_not_clear_it(self):
        """A constant listing the frequent answers is the best a no-retrieval system can do."""
        reply, baseline = constant_reply_baseline(LONG_TAIL)
        hedge = containment_accuracy(dict.fromkeys(LONG_TAIL, reply), LONG_TAIL)
        metrics = {"containment": hedge, "constant_baseline": baseline, "constant_reply": reply}
        assert "does not beat the constant-reply baseline" in baseline_warning(metrics)

    def test_a_partial_hedge_also_fails_to_clear_it(self):
        _, baseline = constant_reply_baseline(LONG_TAIL)
        hedge = containment_accuracy(dict.fromkeys(LONG_TAIL, "Yes No"), LONG_TAIL)
        assert hedge < baseline

    def test_a_correct_verbose_system_clears_it(self):
        _, baseline = constant_reply_baseline(LONG_TAIL)
        verbose = {qid: _pad(answer, 120) for qid, answer in LONG_TAIL.items()}
        assert containment_accuracy(verbose, LONG_TAIL) > baseline


class TestBaselineWarning:
    def test_run_that_clears_the_baseline_is_quiet(self):
        assert baseline_warning({"containment": 0.5, "constant_baseline": 0.3}) is None

    @pytest.mark.parametrize("containment", [0.3, 0.1])
    def test_run_at_or_below_the_baseline_is_flagged(self, containment):
        warning = baseline_warning({"containment": containment, "constant_baseline": 0.3, "constant_reply": "Yes"})
        assert "does not beat the constant-reply baseline" in warning
        assert "'Yes'" in warning

    def test_legacy_f1_only_results_are_flagged_as_not_rankable(self):
        assert "token-F1" in baseline_warning({"f1": 0.0141, "exact_match": 0.0})

    def test_missing_baseline_is_flagged(self):
        assert "no constant-reply baseline" in baseline_warning({"containment": 0.9})
