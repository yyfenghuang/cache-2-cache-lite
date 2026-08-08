# SPDX-License-Identifier: Apache-2.0

"""Accuracy exists, with an interval and a paired test, over records that
were read before the aggregate was trusted.

Silent on success, exit 0.

The load-bearing test here recomputes every accuracy from the per-sample
records and requires them to equal the reported aggregate. That is what "read
before the aggregate was trusted" means when a machine does the reading: an
aggregate that cannot be rebuilt from the rows it summarises is an aggregate
nobody checked. The same applies to the discordant counts behind every paired
test, and to the answer distribution, which is the figure that separates a low
score from a collapse.

Which conditions ran is read from the run file rather than named here. The
comparison has grown from two conditions to four, and a test that hardcoded
the set would need editing every time the set moved, which is the same as not
checking it.

This gate does not assert which condition scores higher. Two predictions on
record say a replacing build lands below the unassisted Receiver, and a gate
that failed when that happened would be a gate designed to be wrong.
"""

from __future__ import annotations

import json
import sys
from math import comb
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = REPO_ROOT / "results"

# Conditions that read a trained module have to say which one. The Receiver
# alone reads nothing, so it is exempt.
UNTRAINED_CONDITIONS = ("baseline",)

TOLERANCE = 1e-12


def load():
    runs = sorted(RESULTS_DIR.glob("run_*_n*.json"))
    if not runs:
        raise SystemExit(
            "no results/run_<date>_n<N>.json; run scripts/analysis.py first"
        )
    return json.loads(runs[-1].read_text(encoding="utf-8")), runs[-1].name


def conditions(run):
    named = run["conditions"]
    assert named, "the run scored no conditions"
    assert "baseline" in named, "the Receiver alone was not scored"
    return named


def test_every_sample_has_a_complete_record():
    run, name = load()
    records = run["per_sample"]
    named = conditions(run)
    assert len(records) == run["corpus"]["n_sampled"], name
    for record in records:
        assert record["answer"] in (0, 1, 2, 3), record["index"]
        for field in ("choice", "correct", "logits"):
            assert sorted(record[field]) == sorted(named), (
                f"record {record['index']} is missing conditions in {field}"
            )
        for condition in named:
            assert record["choice"][condition] in (0, 1, 2, 3)
            assert len(record["logits"][condition]) == 4


def test_every_choice_is_the_argmax_of_the_logits_beside_it():
    """The recorded choice has to come from the recorded logits.

    A choice written from anywhere else would agree with the accuracy, agree
    with the discordant counts, and still be a different measurement than the
    one this file claims to describe.
    """
    run, _ = load()
    for record in run["per_sample"]:
        for condition, logits in record["logits"].items():
            best = max(range(4), key=lambda i: logits[i])
            assert record["choice"][condition] == best, (
                f"record {record['index']}, condition {condition}"
            )
            assert record["correct"][condition] == (best == record["answer"])


def test_every_aggregate_can_be_rebuilt_from_the_records():
    run, _ = load()
    records = run["per_sample"]
    for condition in conditions(run):
        rebuilt = sum(1 for r in records if r["correct"][condition]) / len(records)
        assert abs(run["accuracy"][condition] - rebuilt) < TOLERANCE, condition


def test_the_answer_distribution_agrees_with_the_records():
    """The figure that separates a low score from a collapse.

    An accuracy near the base rate of one letter and an accuracy earned across
    four are the same number, and only this tells them apart.
    """
    run, _ = load()
    records = run["per_sample"]
    for condition, counts in run["answer_distribution"].items():
        assert sum(counts) == len(records), condition
        for letter in range(4):
            rebuilt = sum(1 for r in records if r["choice"][condition] == letter)
            assert counts[letter] == rebuilt, (condition, letter)


def test_every_condition_answered_the_same_questions():
    run, _ = load()
    named = conditions(run)
    for record in run["per_sample"]:
        assert set(record["choice"]) == set(named), record["index"]


def test_an_interval_and_a_paired_test_are_reported_for_every_comparison():
    run, _ = load()
    named = conditions(run)
    assert run["comparisons"], "no comparison was reported"
    for label, comparison in run["comparisons"].items():
        assert comparison["a"] in named and comparison["b"] in named, label
        interval = comparison["bootstrap"]
        assert interval["low"] <= interval["difference"] <= interval["high"], label
        assert 0.0 <= comparison["mcnemar"]["p_value"] <= 1.0, label


def test_every_bootstrap_difference_matches_the_accuracies():
    run, _ = load()
    for label, comparison in run["comparisons"].items():
        stated = run["accuracy"][comparison["b"]] - run["accuracy"][comparison["a"]]
        assert abs(comparison["bootstrap"]["difference"] - stated) < 1e-9, label


def test_every_mcnemar_count_agrees_with_the_records():
    run, _ = load()
    records = run["per_sample"]
    for label, comparison in run["comparisons"].items():
        a, b = comparison["a"], comparison["b"]
        mcnemar = comparison["mcnemar"]
        a_only = sum(1 for r in records if r["correct"][a] and not r["correct"][b])
        b_only = sum(1 for r in records if r["correct"][b] and not r["correct"][a])
        assert mcnemar["a_only"] == a_only, label
        assert mcnemar["b_only"] == b_only, label
        assert mcnemar["n_discordant"] == a_only + b_only, label

        n = a_only + b_only
        if n:
            tail = min(a_only, b_only)
            expected = min(
                1.0, sum(comb(n, k) for k in range(tail + 1)) / (2 ** n) * 2
            )
            assert abs(mcnemar["p_value"] - expected) < 1e-12, label


def test_every_trained_condition_names_what_it_loaded():
    """A score with no checkpoint behind it cannot be reproduced or refuted."""
    run, _ = load()
    provenance = run.get("provenance", {})
    for condition in conditions(run):
        if condition in UNTRAINED_CONDITIONS:
            continue
        assert condition in provenance, (
            f"{condition} was scored and the run does not say from which weights"
        )
        assert provenance[condition].get("checkpoint"), condition


def test_broken_questions_were_removed_and_the_removal_recorded():
    """MMLU-Redux labels questions with no correct answer, several correct
    answers, or a wrong ground truth. Leaving them in adds noise that falls on
    every condition and shrinks the difference being measured."""
    run, _ = load()
    corpus = run["corpus"]
    assert corpus["kept_error_type"] == "ok"
    assert "dropped_by_error_type" in corpus
    assert corpus["pool_size"] >= corpus["n_sampled"]


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()


if __name__ == "__main__":
    main()