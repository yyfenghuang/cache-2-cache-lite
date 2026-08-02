"""The receiver reads an injected cache, and reading it has consequences.

Silent on success, exit 0. Refuses to start when results/substrate.json is
absent.

Two kinds of comparison appear below and they cannot share a criterion.

A same-shape comparison runs the identical computation twice and differs only
in what was handed to it. The reduction is split across threads the same way
both times, so the result is bit-identical for a structural reason. Those are
gated at exactly 0.0, and a tolerance there would hide the failure this gate
exists to catch: an injection path that drops a tensor and lets the framework
recompute it also lands close to zero.

A different-shape comparison computes the same quantity through matmuls of
different M. BLAS partitions the reduction by shape, so the summation order
differs and the results disagree by a few float32 steps. How many depends on
the thread count, which is a property of the machine and not of the
mechanism. Those are gated in ulps, with a separation requirement doing the
real work.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

SUBSTRATE_PATH = REPO_ROOT / "results" / "substrate.json"
CONTRACTS_PATH = REPO_ROOT / "results" / "contracts.json"

# Same-shape comparisons. Structurally bit-identical.
EXACT_CONDITIONS = ("baseline_vs_itself", "noop_injection")

# Different-shape comparison. Bounded in float32 steps at the logit scale.
# Measured at 8 ulps on the receiver and up to 116 on a synthetic model at
# eight threads. The budget leaves room for a machine that splits reductions
# more aggressively and still sits far below anything a wrong rotation
# produces.
BOUNDED_CONDITION = "position_absolute"
MAX_ULP = 1024.0

# Conditions that must move the logits, as a fraction of their scale.
DEGRADED_CONDITIONS = ("corrupted_injection", "position_relative")
MIN_DEGRADATION_RATIO = 0.02

# The criterion with teeth: a wrong rotation must be orders of magnitude
# louder than floating point noise, not merely larger than it.
MIN_SEPARATION = 1000.0


def load(path):
    if not path.exists():
        raise SystemExit(
            f"missing {path.relative_to(REPO_ROOT)}; the gate below it has "
            "not been closed"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def test_every_condition_was_measured():
    result = load(SUBSTRATE_PATH)
    names = EXACT_CONDITIONS + DEGRADED_CONDITIONS + (BOUNDED_CONDITION,)
    for section in ("conditions", "ratios", "ulps"):
        assert section in result, f"the run recorded no {section}"
        for name in names:
            assert name in result[section], f"{name} missing from {section}"
    assert result["logit_ulp"] > 0
    assert "torch_num_threads" in result, (
        "the thread count determines the size of every different-shape "
        "disagreement below and has to travel with the numbers"
    )


def test_an_accepted_cache_changes_nothing():
    """Same-shape, so bit-identical or the injection path is not wired."""
    result = load(SUBSTRATE_PATH)
    for name in EXACT_CONDITIONS:
        got = result["conditions"][name]
        assert got == 0.0, f"{name} is {got}, expected exactly 0.0"


def test_a_corrupted_cache_changes_the_output():
    """The control that makes the zeros above mean something.

    Bit-identical output is what a correctly accepted cache produces. It is
    also what a completely ignored cache produces. Only this assertion
    separates them.
    """
    result = load(SUBSTRATE_PATH)
    ratio = result["ratios"]["corrupted_injection"]
    assert ratio > MIN_DEGRADATION_RATIO, (
        f"a norm-matched corrupted cache moved the logits by {ratio:.4g} of "
        f"their scale. Below {MIN_DEGRADATION_RATIO} the cache is not being "
        "read, and every zero above this line means nothing."
    )


def test_absolute_position_ids_reproduce_the_reference():
    """Bounded in float32 steps, because the shapes differ.

    The full forward attends over the whole sequence at once; the
    continuation attends over the same keys with fewer query rows. Same
    mathematics, different matmul, different summation order.
    """
    result = load(SUBSTRATE_PATH)
    ulps = result["ulps"][BOUNDED_CONDITION]
    assert ulps <= MAX_ULP, (
        f"{BOUNDED_CONDITION} is {ulps:.1f} float32 steps at the logit scale, "
        f"above the {MAX_ULP:.0f} budget. At {result['torch_num_threads']} "
        "threads this is too large to be reduction order alone."
    )


def test_relative_position_ids_actually_fail():
    """Both directions, or the test detects nothing.

    A regression test that only exercises the correct path cannot catch the
    bug it guards. This bug is structurally silent: rotation preserves norm,
    so a wrongly rotated cache passes every shape and NaN check downstream.
    """
    result = load(SUBSTRATE_PATH)
    ratio = result["ratios"]["position_relative"]
    assert ratio > MIN_DEGRADATION_RATIO, (
        f"slice-relative position ids moved the logits by only {ratio:.4g} of "
        "their scale, so this run cannot distinguish the bug from its absence"
    )

    wrong = result["conditions"]["position_relative"]
    right = result["conditions"][BOUNDED_CONDITION]
    separation = wrong / right if right > 0 else float("inf")
    assert separation > MIN_SEPARATION, (
        f"a wrong rotation is only {separation:.4g} times louder than "
        f"floating point noise. Below {MIN_SEPARATION:.0f} the two are not "
        "distinguishable and this gate reports nothing."
    )


def test_the_corruption_was_norm_matched():
    """Otherwise the degradation has two causes and separates neither."""
    result = load(SUBSTRATE_PATH)
    check = result["corruption_check"]
    assert check["max_norm_deviation"] < 1e-4, (
        f"corruption changed vector lengths by {check['max_norm_deviation']}, "
        "so it is not a matched control"
    )
    assert check["max_abs_cosine"] < 0.95, (
        f"corruption left directions nearly intact, cosine "
        f"{check['max_abs_cosine']}"
    )


def test_the_cache_matches_the_contract():
    """Tier 1 is read against Tier 0, not against literals typed here."""
    result = load(SUBSTRATE_PATH)
    contract = load(CONTRACTS_PATH)
    receiver = contract["receiver"]

    batch, n_kv_heads, seq, head_dim = result["cache_shape"]
    assert n_kv_heads == receiver["n_kv_heads"], (
        f"cache holds {n_kv_heads} key-value heads, contract says "
        f"{receiver['n_kv_heads']}"
    )
    assert head_dim == receiver["head_dim"], (
        f"cache head_dim {head_dim}, contract says {receiver['head_dim']}"
    )
    assert result["n_layers"] == receiver["n_layers"]
    assert seq == result["sequence"]["n_prefix"]
    assert result["model_id"] == receiver["model_id"]


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()


if __name__ == "__main__":
    main()