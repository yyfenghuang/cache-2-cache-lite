"""A shut fuser is a no-op, and the loss reaches it. Both, or neither counts.

Silent on success, exit 0. Refuses to start when
results/fuser_substrate.json is absent.

The two halves fail in opposite directions and each is the other's control.

The identity half is gated at exactly 0.0 rather than at a tolerance. Both
conditions run the same computation on the same shapes and differ only in
what was handed to them, so bit-identical is a structural consequence and not
a coincidence to be hedged against. A tolerance here would pass a gate that
is merely small, which would mean the fused system cannot reproduce the
baseline it is graded against, which would mean the floor this tier claims
does not exist.

The gradient half is gated on a census rather than an aggregate. A mean
gradient norm above zero is satisfied by one parameter carrying everything
while the rest carry nothing, and the parameters most likely to be silently
disconnected are the ones furthest from the loss.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

SUBSTRATE_PATH = REPO_ROOT / "results" / "fuser_substrate.json"
CONTRACTS_PATH = REPO_ROOT / "results" / "contracts.json"

# Same-shape comparison, therefore structurally bit-identical.
EXACT_CONDITION = "shut_gate_identity"

# The control that gives the zero above its meaning, as a fraction of the
# logit scale. Same threshold as the Tier 1 substrate gate, because it is the
# same question asked of a different injection path.
CHANGE_CONDITION = "open_gate_change"
MIN_CHANGE_RATIO = 0.02

# The gradient census must be taken where training actually starts. At the
# annealing endpoint the relaxed gate is saturated and its gradient underflows
# to zero by design, so a census taken there would report the exact failure
# this gate exists to detect.
EXPECTED_TEMPERATURE = 1.0


def load(path):
    if not path.exists():
        raise SystemExit(
            f"missing {path.relative_to(REPO_ROOT)}; the gate below it has "
            "not been closed"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def test_both_halves_were_measured():
    result = load(SUBSTRATE_PATH)
    for section in ("conditions", "ratios"):
        assert section in result, f"the run recorded no {section}"
        for name in (EXACT_CONDITION, CHANGE_CONDITION):
            assert name in result[section], f"{name} missing from {section}"
    assert "gradient" in result, "the run recorded no gradient census"
    assert result["logit_scale"] > 0


def test_a_shut_gate_changes_nothing():
    """Same-shape, so bit-identical or the fused cache is not being read."""
    result = load(SUBSTRATE_PATH)
    got = result["conditions"][EXACT_CONDITION]
    assert got == 0.0, f"{EXACT_CONDITION} is {got}, expected exactly 0.0"


def test_the_shut_correction_was_zero_at_the_source():
    """Asserted on the correction itself, not only on the logits.

    Two large tensors whose difference cancels to zero also produce identical
    logits. Reading the correction before it is added separates a gate that
    emitted nothing from a gate whose output happened to cancel.
    """
    result = load(SUBSTRATE_PATH)
    got = result["shut_gate_max_abs_correction"]
    assert got == 0.0, (
        f"a shut fuser emitted a correction of {got}. The inference gate is "
        "not landing on exactly zero, and the floor this tier rests on does "
        "not hold"
    )


def test_an_open_gate_changes_the_output():
    """The control that makes the zeros above mean something.

    Bit-identical output is what a correctly accepted fused cache produces
    when the gate is shut. It is also what a fused cache the framework
    discarded produces, at every gate setting. Only this assertion separates
    them.
    """
    result = load(SUBSTRATE_PATH)
    ratio = result["ratios"][CHANGE_CONDITION]
    assert ratio > MIN_CHANGE_RATIO, (
        f"an open fuser moved the logits by {ratio:.4g} of their scale. "
        f"Below {MIN_CHANGE_RATIO} the fused cache is not being read, and "
        "every zero above this line means nothing."
    )


def test_the_census_was_taken_where_training_starts():
    result = load(SUBSTRATE_PATH)
    got = result["temperature"]
    assert got == EXPECTED_TEMPERATURE, (
        f"the gradient census was taken at temperature {got}. At anything "
        "near the annealing endpoint the gate is saturated and reports zero "
        "gradient by design, which is indistinguishable from the failure "
        "this gate detects"
    )
    assert result["bank"]["gate_activation_ratio_at_init"] == 0.0, (
        "the gates were not shut at initialisation, so the run did not start "
        "at the Receiver's baseline and the floor was not established"
    )


def test_every_fuser_parameter_received_a_gradient():
    """The census, parameter by parameter.

    A detached cache produces exactly this failure while every shape check
    passes and every value stays finite. The loss falls, because the model is
    being conditioned on a constant, and the module being trained never
    hears about it.
    """
    result = load(SUBSTRATE_PATH)
    census = result["gradient"]
    assert census["n_parameters"] > 0, "the bank has no parameters"
    assert census["n_missing"] == 0, (
        f"{census['n_missing']} parameters received no gradient at all, "
        f"including {census['missing']}. The loss does not reach them."
    )
    assert census["n_zero"] == 0, (
        f"{census['n_zero']} parameters received an identically zero "
        f"gradient, including {census['zero']}"
    )
    assert census["n_nonfinite"] == 0, (
        f"{census['n_nonfinite']} parameters received a non-finite gradient, "
        f"including {census['nonfinite']}"
    )
    assert census["min_norm"] > 0.0, (
        f"the smallest gradient norm is {census['min_norm']}"
    )


def test_the_bank_matches_the_contract():
    """Tier 5 is read against Tier 0, not against literals typed here."""
    result = load(SUBSTRATE_PATH)
    contract = load(CONTRACTS_PATH)
    receiver, sharer = contract["receiver"], contract["sharer"]

    _, r_heads, _, r_head_dim = result["receiver_cache_shape"]
    _, s_heads, _, s_head_dim = result["sharer_cache_shape"]
    assert r_heads == receiver["n_kv_heads"]
    assert r_head_dim == receiver["head_dim"]
    assert s_heads == sharer["n_kv_heads"]
    assert s_head_dim == sharer["head_dim"]

    bank = result["bank"]
    assert bank["n_receiver_layers"] == receiver["n_layers"]
    assert bank["n_sharer_layers"] == sharer["n_layers"]
    assert bank["paired_layers"] == min(
        receiver["n_layers"], sharer["n_layers"]
    )
    assert len(bank["unpaired_layers"]) == (
        receiver["n_layers"] - bank["paired_layers"]
    ), (
        "terminal alignment leaves the Receiver's shallowest layers without a "
        "partner, and they have to arrive unfused rather than be dropped"
    )
    assert result["receiver_id"] == receiver["model_id"]
    assert result["sharer_id"] == sharer["model_id"]


def test_the_two_caches_cover_the_same_positions():
    """The fuser concatenates position by position, so a length mismatch is a
    token alignment failure and not a shape to broadcast around."""
    result = load(SUBSTRATE_PATH)
    assert result["receiver_cache_shape"][2] == result["sharer_cache_shape"][2]
    assert result["receiver_cache_shape"][2] == result["sequence"]["n_prefix"]


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()


if __name__ == "__main__":
    main()