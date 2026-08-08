# SPDX-License-Identifier: Apache-2.0

"""The fuser and its gate, with no model and no checkpoint.

Silent on success, exit 0.

Two claims here are load-bearing and the rest are guardrails.

A shut gate at inference returns the Receiver's cache bit for bit. Not close
to it. The whole floor argument for this tier rests on that being exact, and
exactness is what separates it from a gate that is merely small.

Every parameter receives a gradient on the first backward pass. This file can
only prove it for the module in isolation, driven by a synthetic loss. Whether
the gradient survives the trip through a real Receiver's attention and cache
handling is a different question with a different failure mode, and it is
answered by `scripts/capture_fuser_substrate.py` against real weights. The
split is the same one `capture_substrate.py` already uses: everything that can
be checked in a second without a checkpoint is checked here, and what remains
is the part that genuinely needs one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from c2c.alignment import align_layers
from c2c.fuser import KINDS, CacheFuser, FuserBank
from c2c.gate import (
    TEMPERATURE_END,
    TEMPERATURE_START,
    ScalarGate,
    annealed_temperature,
    close_gate,
    open_gate,
)

# The real pair, from results/contracts.json. Literals only because this file
# reads no results file; the assertions that must track the contract live in
# tests/test_geometry.py.
RECEIVER_LAYERS, SHARER_LAYERS = 28, 24
RECEIVER_HEADS, RECEIVER_HEAD_DIM = 8, 128
SHARER_HEADS, SHARER_HEAD_DIM = 2, 64


def _fuser(**kwargs) -> CacheFuser:
    return CacheFuser(
        RECEIVER_HEADS, RECEIVER_HEAD_DIM, SHARER_HEADS, SHARER_HEAD_DIM,
        **kwargs,
    )


def _bank(**kwargs) -> FuserBank:
    return FuserBank(
        RECEIVER_LAYERS, SHARER_LAYERS,
        RECEIVER_HEADS, RECEIVER_HEAD_DIM, SHARER_HEADS, SHARER_HEAD_DIM,
        **kwargs,
    )


def _caches(batch=2, positions=6, scale=1.0):
    receiver = torch.randn(batch, RECEIVER_HEADS, positions, RECEIVER_HEAD_DIM)
    sharer = torch.randn(batch, SHARER_HEADS, positions, SHARER_HEAD_DIM) * scale
    return receiver, sharer


def test_output_carries_the_receiver_geometry():
    torch.manual_seed(0)
    fuser = _fuser()
    receiver, sharer = _caches()
    assert fuser(receiver, sharer).shape == receiver.shape


def test_a_shut_gate_returns_the_receiver_cache_exactly():
    """The floor, made executable.

    `torch.equal` and not `allclose`. A correction that is small rather than
    zero would pass a tolerance and would mean the fused system cannot
    reproduce the baseline it is graded against.
    """
    torch.manual_seed(0)
    fuser = _fuser().eval()
    close_gate(fuser)
    receiver, sharer = _caches()
    correction = fuser.correction(receiver, sharer)
    assert torch.equal(correction, torch.zeros_like(correction))
    assert torch.equal(fuser(receiver, sharer), receiver)


def test_a_shut_gate_holds_against_an_enormous_sharer_cache():
    """Zero times large is zero. Small times large is not.

    This is the test that fails if the inference-time gate is ever changed
    from a threshold to a sigmoid, which would look like a simplification and
    would remove the property the tier depends on.
    """
    torch.manual_seed(0)
    fuser = _fuser().eval()
    close_gate(fuser)
    receiver, sharer = _caches(scale=1e6)
    assert torch.equal(fuser(receiver, sharer), receiver)


def test_an_open_gate_changes_the_cache():
    torch.manual_seed(0)
    fuser = _fuser().eval()
    open_gate(fuser)
    receiver, sharer = _caches()
    assert not torch.equal(fuser(receiver, sharer), receiver)


def test_the_gate_is_binary_at_inference_and_continuous_in_training():
    torch.manual_seed(0)
    gate = ScalarGate()

    gate.eval()
    close_gate(gate)
    assert float(gate.decision()) == 0.0
    open_gate(gate)
    assert float(gate.decision()) == 1.0

    gate.train()
    gate.logit.data.fill_(0.0)
    draws = [float(gate.decision().detach()) for _ in range(64)]
    assert all(0.0 < d < 1.0 for d in draws), "a training draw hit an endpoint"
    assert len(set(draws)) > 1, "the relaxed sample is not sampling"


def test_every_parameter_receives_a_gradient_at_the_starting_temperature():
    """Run at the temperature training starts from, and at the shut
    initialisation, because both are the conditions of the first real step.

    A gate initialised closed still has to pass gradient. If it did not, the
    floor would have been bought by making the module untrainable.
    """
    torch.manual_seed(0)
    fuser = _fuser().train()
    fuser.gate.set_temperature(TEMPERATURE_START)
    receiver, sharer = _caches()

    out = fuser(receiver, sharer)
    (out * torch.randn_like(out)).sum().backward()

    for name, parameter in fuser.named_parameters():
        assert parameter.grad is not None, f"{name} has no gradient"
        assert torch.isfinite(parameter.grad).all(), f"{name} gradient is not finite"
        assert float(parameter.grad.abs().max()) > 0.0, (
            f"{name} gradient is identically zero"
        )


def test_the_correction_is_position_wise():
    """Position p's correction reads position p and nothing else.

    The fuser sees one vector per position and has no mixing across the
    position axis. If a future edit reshapes the wrong way, corrections would
    leak between tokens and the failure would be invisible in every shape
    check.
    """
    torch.manual_seed(0)
    fuser = _fuser().eval()
    open_gate(fuser)
    receiver, sharer = _caches(batch=1, positions=5)

    before = fuser.correction(receiver, sharer)
    disturbed = sharer.clone()
    disturbed[0, :, 2, :] += 10.0
    after = fuser.correction(receiver, disturbed)

    changed = [
        p for p in range(5)
        if not torch.equal(before[:, :, p, :], after[:, :, p, :])
    ]
    assert changed == [2], f"positions {changed} moved, expected only [2]"


def test_the_dynamic_weights_are_one_per_head_per_token():
    torch.manual_seed(0)
    fuser = _fuser()
    assert fuser.weighting.out_features == RECEIVER_HEADS
    assert fuser.weighting.in_features == fuser.joint_width


def test_geometry_errors_are_refused():
    fuser = _fuser()
    receiver, sharer = _caches(batch=2, positions=6)
    for bad_receiver, bad_sharer in (
        (receiver[:, :, :, :64], sharer),
        (receiver, sharer[:, :1, :, :]),
        (receiver, sharer[:, :, :3, :]),
        (receiver[:1], sharer),
        (receiver[0], sharer[0]),
    ):
        try:
            fuser(bad_receiver, bad_sharer)
        except ValueError:
            continue
        raise AssertionError("accepted caches of the wrong geometry")


def test_configuration_errors_are_refused():
    for kwargs in ({"hidden": 0}, {"activation": "tanh"}):
        try:
            _fuser(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"accepted {kwargs}")
    for call in (
        lambda: close_gate(_fuser(), logit=1.0),
        lambda: open_gate(_fuser(), logit=-1.0),
        lambda: ScalarGate().set_temperature(0.0),
    ):
        try:
            call()
        except ValueError:
            continue
        raise AssertionError("accepted a nonsensical gate setting")


def test_keys_and_values_get_independent_parameters():
    torch.manual_seed(0)
    bank = _bank()
    target = bank.paired_layers[0]
    keys = bank.fusers_for("keys")[target]
    values = bank.fusers_for("values")[target]
    assert not torch.equal(keys.fuse.weight, values.fuse.weight)
    with torch.no_grad():
        keys.gate.logit.fill_(5.0)
    assert float(values.gate.logit.detach()) != 5.0


def test_the_bank_follows_terminal_alignment():
    bank = _bank()
    assert bank.mapping == align_layers(SHARER_LAYERS, RECEIVER_LAYERS)
    assert bank.unpaired_layers == [0, 1, 2, 3]
    assert len(bank.paired_layers) == SHARER_LAYERS


def test_unpaired_layers_pass_through_untouched():
    """Not a degenerate case. Four Receiver layers have no partner, and their
    caches must arrive at the model as the same objects they left it as.
    """
    torch.manual_seed(0)
    bank = _bank().eval()
    open_gate(bank)
    receiver_pairs = [
        (torch.randn(1, RECEIVER_HEADS, 4, RECEIVER_HEAD_DIM),
         torch.randn(1, RECEIVER_HEADS, 4, RECEIVER_HEAD_DIM))
        for _ in range(RECEIVER_LAYERS)
    ]
    sharer_pairs = [
        (torch.randn(1, SHARER_HEADS, 4, SHARER_HEAD_DIM),
         torch.randn(1, SHARER_HEADS, 4, SHARER_HEAD_DIM))
        for _ in range(SHARER_LAYERS)
    ]
    out = bank(receiver_pairs, sharer_pairs)

    assert len(out) == RECEIVER_LAYERS
    for layer in bank.unpaired_layers:
        for index in range(len(KINDS)):
            assert out[layer][index] is receiver_pairs[layer][index]
    for layer in bank.paired_layers:
        for index in range(len(KINDS)):
            assert not torch.equal(out[layer][index], receiver_pairs[layer][index])


def test_the_bank_checks_the_layer_counts_it_was_handed():
    bank = _bank()
    short = [
        (torch.randn(1, RECEIVER_HEADS, 4, RECEIVER_HEAD_DIM),) * 2
        for _ in range(RECEIVER_LAYERS - 1)
    ]
    full_sharer = [
        (torch.randn(1, SHARER_HEADS, 4, SHARER_HEAD_DIM),) * 2
        for _ in range(SHARER_LAYERS)
    ]
    try:
        bank(short, full_sharer)
    except ValueError:
        return
    raise AssertionError("accepted a cache with the wrong number of layers")


def test_a_shut_bank_is_the_identity_on_every_layer():
    torch.manual_seed(0)
    bank = _bank().eval()
    close_gate(bank)
    receiver_pairs = [
        (torch.randn(1, RECEIVER_HEADS, 4, RECEIVER_HEAD_DIM),
         torch.randn(1, RECEIVER_HEADS, 4, RECEIVER_HEAD_DIM))
        for _ in range(RECEIVER_LAYERS)
    ]
    sharer_pairs = [
        (torch.randn(1, SHARER_HEADS, 4, SHARER_HEAD_DIM),
         torch.randn(1, SHARER_HEADS, 4, SHARER_HEAD_DIM))
        for _ in range(SHARER_LAYERS)
    ]
    out = bank(receiver_pairs, sharer_pairs)
    for layer in range(RECEIVER_LAYERS):
        for index in range(len(KINDS)):
            assert torch.equal(out[layer][index], receiver_pairs[layer][index])


def test_the_gate_activation_ratio_counts_what_a_4_2_counts():
    bank = _bank()
    close_gate(bank)
    assert bank.gate_activation_ratio() == 0.0
    open_gate(bank)
    assert bank.gate_activation_ratio() == 1.0

    gates = [g for g in bank.modules() if isinstance(g, ScalarGate)]
    assert len(gates) == SHARER_LAYERS * len(KINDS)
    with torch.no_grad():
        for gate in gates[: len(gates) // 2]:
            gate.logit.fill_(-1.0)
    assert abs(bank.gate_activation_ratio() - 0.5) < 1e-9


def test_the_temperature_anneals_across_the_whole_run():
    assert annealed_temperature(0, 100) == TEMPERATURE_START
    assert abs(annealed_temperature(99, 100) - TEMPERATURE_END) < 1e-12
    assert annealed_temperature(0, 1) == TEMPERATURE_END
    midpoint = annealed_temperature(50, 101)
    assert TEMPERATURE_END < midpoint < TEMPERATURE_START

    for call in (
        lambda: annealed_temperature(0, 0),
        lambda: annealed_temperature(100, 100),
        lambda: annealed_temperature(-1, 10),
    ):
        try:
            call()
        except ValueError:
            continue
        raise AssertionError("accepted a step outside the run")


def test_the_temperature_is_carried_in_the_state_dict():
    """The gate's behaviour depends on it, so a checkpoint that omits it
    would reload into a different module than the one that was saved."""
    bank = _bank()
    bank.set_temperature(0.25)
    state = bank.state_dict()
    keys = [k for k in state if k.endswith("temperature")]
    assert len(keys) == SHARER_LAYERS * len(KINDS)
    assert all(float(state[k]) == 0.25 for k in keys)


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()


if __name__ == "__main__":
    main()