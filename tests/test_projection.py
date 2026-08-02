"""The projection, and the floor it is graded against, in the same units.

Silent on success, exit 0. No checkpoint, no results file, no model.

The load-bearing test here is that a projection put into its constant
predictor state reproduces the null exactly, to the last bits. That is what
makes "the loss went below the null" a statement about the input rather than
about two numbers that happen to be computed by different code.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from c2c.cache_ops import flatten_heads, unflatten_heads
from c2c.projection import CacheProjection, constant_predictor_state

# The real pair, from results/contracts.json. Written here as literals only
# because this file never reads a results file; the assertions that must
# track the contract live in tests/test_geometry.py.
SHARER_HEADS, SHARER_HEAD_DIM = 2, 64
RECEIVER_HEADS, RECEIVER_HEAD_DIM = 8, 128


def _projection(**kwargs):
    return CacheProjection(
        SHARER_HEADS, SHARER_HEAD_DIM, RECEIVER_HEADS, RECEIVER_HEAD_DIM,
        **kwargs,
    )


def test_head_flattening_round_trips():
    torch.manual_seed(0)
    tensor = torch.randn(2, 8, 5, 128)
    flat = flatten_heads(tensor)
    assert flat.shape == (10, 1024)
    back = unflatten_heads(flat, 2, 8, 128)
    assert torch.equal(back, tensor)


def test_flattening_puts_one_position_per_row():
    """Row order is positions within batch, not heads within position."""
    tensor = torch.zeros(1, 2, 3, 2)
    tensor[0, 0, 1, :] = torch.tensor([1.0, 2.0])
    tensor[0, 1, 1, :] = torch.tensor([3.0, 4.0])
    flat = flatten_heads(tensor)
    assert flat[1].tolist() == [1.0, 2.0, 3.0, 4.0]


def test_flattening_rejects_wrong_ranks():
    for call in (
        lambda: flatten_heads(torch.randn(2, 3, 4)),
        lambda: unflatten_heads(torch.randn(2, 3, 4), 1, 2, 2),
        lambda: unflatten_heads(torch.randn(6, 7), 1, 2, 2),
        lambda: unflatten_heads(torch.randn(5, 4), 2, 2, 2),
    ):
        try:
            call()
        except ValueError:
            continue
        raise AssertionError("accepted a shape it should not have")


def test_output_carries_the_receiver_geometry():
    projection = _projection()
    out = projection(torch.randn(3, SHARER_HEADS, 7, SHARER_HEAD_DIM))
    assert out.shape == (3, RECEIVER_HEADS, 7, RECEIVER_HEAD_DIM)


def test_input_geometry_is_checked():
    projection = _projection()
    for bad in (
        torch.randn(1, RECEIVER_HEADS, 5, SHARER_HEAD_DIM),
        torch.randn(1, SHARER_HEADS, 5, RECEIVER_HEAD_DIM),
    ):
        try:
            projection(bad)
        except ValueError:
            continue
        raise AssertionError("accepted a cache of the wrong geometry")


def test_the_constant_predictor_state_reproduces_the_null_exactly():
    """The floor, made executable, at every depth that will be used.

    A projection with zero weights and the per channel mean in its output
    bias is the best constant predictor. Its mean squared error must equal
    the null the geometry file records, or the two numbers are not
    comparable and "below the null" means nothing.

    Depth is covered because the state has to survive a hidden layer: zero
    weights make the hidden activations zero, the activation function has to
    map zero to zero, and only then does the output bias reach the target
    untouched. An activation with a nonzero value at the origin would break
    this silently and the floor would stop being reachable.
    """
    torch.manual_seed(0)
    target = torch.randn(4, RECEIVER_HEADS, 60, RECEIVER_HEAD_DIM) * 3.0 + 5.0
    flat_target = flatten_heads(target)
    mean = flat_target.mean(dim=0)
    null = flat_target.var(dim=0, unbiased=False).mean()

    for kwargs in ({}, {"depth": 2, "hidden": 256},
                   {"depth": 3, "hidden": 64, "activation": "silu"}):
        projection = _projection(**kwargs)
        constant_predictor_state(projection, mean)
        with torch.no_grad():
            predicted = projection(
                torch.randn(4, SHARER_HEADS, 60, SHARER_HEAD_DIM)
            )
        achieved = ((predicted - target) ** 2).mean()
        assert torch.allclose(achieved, null, rtol=0, atol=1e-5), (
            f"{kwargs or 'affine'}: constant predictor reached "
            f"{float(achieved)}, null is {float(null)}"
        )


def test_the_constant_predictor_ignores_its_input():
    torch.manual_seed(0)
    projection = _projection()
    constant_predictor_state(projection, torch.randn(1024))
    with torch.no_grad():
        a = projection(torch.randn(1, SHARER_HEADS, 4, SHARER_HEAD_DIM))
        b = projection(torch.randn(1, SHARER_HEADS, 4, SHARER_HEAD_DIM) * 100)
    assert torch.equal(a, b)


def test_constant_predictor_state_checks_the_width():
    projection = _projection()
    try:
        constant_predictor_state(projection, torch.zeros(128))
    except ValueError:
        return
    raise AssertionError("accepted a mean of the wrong width")


def test_degrees_of_freedom_are_reported():
    """The ratio of training positions to this number decides whether a loss
    below the null is a finding or an artefact of the fit."""
    affine = _projection()
    assert affine.n_parameters() == 128 * 1024 + 1024
    assert abs(affine.n_parameters_per_output_channel() - 129.0) < 1e-9

    deep = _projection(depth=2, hidden=256)
    assert deep.n_parameters() > affine.n_parameters()
    assert deep.n_parameters_per_output_channel() > 129.0


def test_configuration_errors_are_refused():
    for kwargs in (
        {"depth": 0},
        {"depth": 2},
        {"depth": 1, "hidden": 256},
        {"depth": 2, "hidden": 256, "activation": "tanh"},
    ):
        try:
            _projection(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"accepted {kwargs}")


def test_keys_and_values_get_independent_parameters():
    torch.manual_seed(0)
    keys = _projection()
    values = _projection()
    assert not torch.equal(
        keys.output_layer.weight, values.output_layer.weight
    ), "two constructions produced identical weights"
    with torch.no_grad():
        keys.output_layer.bias.fill_(1.0)
        assert float(values.output_layer.bias.abs().max()) != 1.0


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()


if __name__ == "__main__":
    main()