"""Accuracy exists, with an interval and a paired test, over records that
were read before the aggregate was trusted.

Silent on success, exit 0.

The load-bearing test here recomputes both accuracies from the per-sample
records and requires them to equal the reported aggregate. That is what
"read before the aggregate was trusted" means when a machine does the
reading: an aggregate that cannot be rebuilt from the rows it summarises is
an aggregate nobody checked.

This gate does not assert which condition scores higher. The prediction on
record is that the projection-only build scores below the unassisted
Receiver, and a gate that failed when that happened would be a gate designed
to be wrong.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = REPO_ROOT / "results"
CONDITIONS = ("baseline", "injected")


def load():
    runs = sorted(RESULTS_DIR.glob("run_*_n*.json"))
    if not runs:
        raise SystemExit(
            "no results/run_<date>_n<N>.json; run scripts/analysis.py first"
        )
    return json.loads(runs[-1].read_text(encoding="utf-8")), runs[-1].name


def test_every_sample_has_a_complete_record():
    run, name = load()
    records = run["per_sample"]
    assert len(records) == run["corpus"]["n_sampled"], name
    for record in records:
        assert record["answer"] in (0, 1, 2, 3), record["index"]
        for condition in CONDITIONS:
            choice = record[f"{condition}_choice"]
            assert choice in (0, 1, 2, 3), (record["index"], condition)
            assert len(record[f"{condition}_logits"]) == 4
            assert record[f"{condition}_correct"] == (choice == record["answer"])


def test_the_aggregate_can_be_rebuilt_from_the_records():
    """An aggregate that cannot be rebuilt from its rows is one nobody read."""
    run, name = load()
    records = run["per_sample"]
    for condition in CONDITIONS:
        rebuilt = sum(r[f"{condition}_correct"] for r in records) / len(records)
        reported = run["accuracy"][condition]
        assert abs(rebuilt - reported) < 1e-12, (
            f"{name}: {condition} reports {reported} and its records give {rebuilt}"
        )


def test_the_two_conditions_answered_the_same_questions():
    """Otherwise the pairing the statistics assume does not exist."""
    run, _ = load()
    for record in run["per_sample"]:
        assert "baseline_choice" in record and "injected_choice" in record
        assert record["n_prompt_tokens"] > 0


def test_an_interval_and_a_paired_test_are_both_reported():
    run, _ = load()
    bootstrap, mcnemar = run["bootstrap"], run["mcnemar"]
    assert bootstrap["n_resamples"] > 0
    assert bootstrap["low"] <= bootstrap["difference"] <= bootstrap["high"]
    assert bootstrap["interval"] == 0.95
    assert "p_value" in mcnemar and 0.0 <= mcnemar["p_value"] <= 1.0


def test_mcnemar_counts_agree_with_the_records():
    """The discordant counts are the whole of the test, so they are checked
    against the rows rather than taken on trust."""
    run, _ = load()
    records = run["per_sample"]
    mcnemar = run["mcnemar"]
    if mcnemar["n_discordant"] == 0:
        return
    b = sum(1 for r in records if r["baseline_correct"] and not r["injected_correct"])
    c = sum(1 for r in records if r["injected_correct"] and not r["baseline_correct"])
    assert mcnemar["b_baseline_only"] == b
    assert mcnemar["c_injected_only"] == c
    assert mcnemar["n_discordant"] == b + c


def test_broken_questions_were_removed_and_the_removal_recorded():
    """MMLU-Redux labels questions with no correct answer, several correct
    answers, or a wrong ground truth. Leaving them in adds noise to both
    conditions and shrinks the difference being measured."""
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