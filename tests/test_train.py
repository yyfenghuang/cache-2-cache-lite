"""A projection reaches below the geometric null, on data it did not see.

Silent on success, exit 0. Refuses to start when results/train_log.json is
absent.

The criterion is the held out relative loss, per paired layer, per kind. It
is not the in sample loss and it is not the shape of the curve. Measured on
uninformative input, an affine map from 128 to 1024 channels reaches an in
sample relative loss of 0.668 at 387 positions and 0.996 at 32000, both
below one. The held out relative loss on the same input is 1.497 and 1.004,
above one in both cases. Only the second comparison can fail when nothing was
learned, which is the only property that makes it a gate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

LOG_PATH = REPO_ROOT / "results" / "train_log.json"
CONTRACTS_PATH = REPO_ROOT / "results" / "contracts.json"

MAX_RELATIVE_HELD_OUT = 1.0
MIN_POSITIONS_PER_PARAMETER = 50.0
KINDS = ("keys", "values")
SELECTION_SPLIT = "validation"
GRADED_SPLIT = "held_out"


def load(path):
    if not path.exists():
        raise SystemExit(
            f"missing {path.relative_to(REPO_ROOT)}; the gate below it has "
            "not been closed"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def test_every_paired_layer_was_trained_for_both_kinds():
    log = load(LOG_PATH)
    contract = load(CONTRACTS_PATH)
    expected = min(contract["sharer"]["n_layers"], contract["receiver"]["n_layers"])
    for kind in KINDS:
        assert log["summary"][kind]["n_layers"] == expected, (
            f"{kind}: {log['summary'][kind]['n_layers']} layers trained, "
            f"{expected} pairs exist"
        )


def test_keys_and_values_were_trained_under_identical_conditions():
    """The prediction on record compares their floors, so a difference in
    configuration would answer a different question."""
    log = load(LOG_PATH)
    by_layer = {}
    for record in log["per_layer"]:
        by_layer.setdefault(record["target_layer"], {})[record["kind"]] = record
    for target, pair in by_layer.items():
        assert set(pair) == set(KINDS), target
        assert (
            pair["keys"]["parameters_per_output_channel"]
            == pair["values"]["parameters_per_output_channel"]
        ), target
        assert (
            pair["keys"]["n_train_positions"]
            == pair["values"]["n_train_positions"]
        ), target


def test_the_projection_beat_a_constant_on_data_it_did_not_see():
    log = load(LOG_PATH)
    failures = [
        f"{r['kind']} layer {r['target_layer']}: "
        f"held out relative {r['relative_held_out']:.4f}"
        for r in log["per_layer"]
        if r["relative_held_out"] >= MAX_RELATIVE_HELD_OUT
    ]
    if failures:
        raise AssertionError(
            "these projections did not beat a constant predictor:\n  "
            + "\n  ".join(failures)
        )


def test_the_fit_had_enough_data_to_mean_anything():
    """A held out loss below the null is valid at any ratio. A held out loss
    above it, at a starved ratio, does not separate "no signal" from "not
    enough data to find it", so the ratio is recorded and gated."""
    log = load(LOG_PATH)
    starved = [
        f"{r['kind']} layer {r['target_layer']}: "
        f"{r['positions_per_parameter']:.1f} positions per parameter"
        for r in log["per_layer"]
        if r["positions_per_parameter"] < MIN_POSITIONS_PER_PARAMETER
    ]
    if starved:
        raise AssertionError(
            f"below {MIN_POSITIONS_PER_PARAMETER:.0f} positions per parameter "
            "the result cannot separate absence of signal from absence of "
            "data:\n  " + "\n  ".join(starved)
        )


def test_the_epoch_was_chosen_without_reading_the_graded_split():
    """Structural, not a matter of care.

    Choosing the epoch by held out loss and then reporting that loss grades
    the projection on data it was tuned against. The run of 2026-08-02 shows
    the choice is not cosmetic: four value projections sat near 0.70 at their
    tenth epoch and above 1.0 at their two hundredth, while training loss
    fell throughout.
    """
    log = load(LOG_PATH)
    assert log["config"]["selection_split"] == SELECTION_SPLIT
    assert log["config"]["graded_split"] == GRADED_SPLIT
    assert SELECTION_SPLIT != GRADED_SPLIT
    for record in log["per_layer"]:
        assert record["selection_split"] == SELECTION_SPLIT, record["target_layer"]
        assert record["selected_epoch"] >= 0
        assert record["selected_epoch"] <= record["stopped_at_epoch"]
        assert record["n_validation_positions"] > 0


def test_the_in_sample_number_is_recorded_but_not_the_criterion():
    """Three numbers are written so the gaps between them are visible. A
    large gap is overfitting made legible rather than hidden."""
    log = load(LOG_PATH)
    for record in log["per_layer"]:
        for field in ("relative_train", "relative_validation",
                      "relative_held_out", "relative_best_held_out"):
            assert field in record, field
        assert record["n_held_out_positions"] > 0


def test_an_injected_projected_cache_moves_the_output():
    """Gate two, in the units Tier 1 fixed.

    Bit identical output is what a correctly accepted cache produces and also
    what an ignored one produces, so the no-op alone proves nothing. The
    corrupted case is repeated here from Tier 1 so the threshold is read
    against a control measured on the same prefix rather than against a
    number quoted from another run.
    """
    log = load(LOG_PATH)
    injection = log["injection"]
    ratios = injection["ratios"]

    assert injection["conditions"]["noop"] == 0.0, (
        f"substituting the receiver's own tensors moved the logits by "
        f"{injection['conditions']['noop']}, so the injection path is not "
        "returning what it was given"
    )
    assert ratios["corrupted"] > injection["threshold"], (
        "the negative control did not degrade, so this prefix cannot "
        "distinguish a cache being read from one being ignored"
    )
    assert ratios["projected"] > injection["threshold"], (
        f"the projected cache moved the logits by {ratios['projected']:.4g} of "
        f"their scale, below the {injection['threshold']} Tier 1 fixed"
    )
    assert injection["replaced_target_layers"], "nothing was replaced"


def test_the_null_was_run_as_a_condition_and_not_only_quoted():
    """The constant predictor is what a projection that learned nothing would
    emit. Running it through the same injection path is what makes the
    projected number comparable to something rather than only to zero."""
    log = load(LOG_PATH)
    injection = log["injection"]
    assert "constant" in injection["conditions"]
    assert injection["projected_over_constant"] is not None
    assert set(injection["conditions"]) == {
        "noop", "constant", "projected", "corrupted"
    }


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()


if __name__ == "__main__":
    main()