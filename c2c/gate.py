# SPDX-License-Identifier: Apache-2.0
"""The gate that decides whether a Sharer's contribution is injected at all.

No I/O, no model loading.

The gate is the reason this tier has a floor. A closed gate at inference
multiplies the whole Sharer contribution by exactly zero, so the fused cache
is the Receiver's cache, bit for bit, and the fused system scores exactly
what the Receiver scores alone. Retreating to the baseline is therefore
always available to the optimiser, which is the property the projection-only
build did not have and the reason its 0.198 could not be read as anything.

That exactness is a claim about arithmetic and it is easy to lose. It holds
because the inference-time gate is drawn from {0.0, 1.0} rather than from a
sigmoid that is merely small: `target + 0.0 * x` is `target` for any finite
`x`, whereas `target + 1e-9 * x` is not. Every design decision below that
looks fussy is protecting that one line.

Two learned quantities, not one
-------------------------------
`logit` decides whether, and `magnitude` decides how much. The reference form
recorded in `TODO.md` carries both:

    output = target + gate * sigmoid(scalar) * projected

They are not redundant. The gate hardens to a binary decision as the
temperature anneals, at which point it can no longer express "a little", and
`magnitude` is what remains to express it. Collapsing them into one parameter
would mean that by the end of training every open layer contributes at full
strength.

Training and inference disagree on purpose
------------------------------------------
Under `train()` the gate is a relaxed Bernoulli sample: sigmoid of the logit
plus logistic noise, divided by a temperature. It is continuous, strictly
inside (0, 1), and differentiable, so gradient reaches the projection behind
it. Under `eval()` it is a threshold on the logit, with no noise, and lands on
0.0 or 1.0 exactly.

The temperature is annealed from 1.0 to 0.001 across the run, which is what
closes the distance between those two behaviours by the end. Note the cost
this imposes and do not mistake it for a bug: at a temperature of 0.001 the
relaxed sample is saturated, its gradient underflows to zero, and the gate
stops learning. That is the intended end state, not a failure, but it means a
gradient check run at the final temperature would report exactly the failure
the check exists to detect. Gradient checks run at temperature 1.0.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = [
    "ScalarGate",
    "annealed_temperature",
    "close_gate",
    "open_gate",
]

# The reference anneals across the whole run, linearly, between these two.
TEMPERATURE_START = 1.0
TEMPERATURE_END = 0.001

# Closed at initialisation. Chosen here rather than inherited: the reference
# recipe's initial gate value is not recorded in this repository's notes, so
# this is a decision and is documented as one.
#
# Closed init buys the floor at step zero, since the graded system runs under
# eval() where a negative logit is exactly 0.0. It costs early gradient
# magnitude, because the relaxed sample sits near 0.12 rather than 0.5 and
# scales everything behind it down by that factor. The trade is taken because
# a floor that holds by construction is worth more here than faster early
# progress, and because the alternative reintroduces the one thing that made
# the previous tier unreadable: a build with no lower bound.
INIT_LOGIT = -2.0


def annealed_temperature(
    step: int,
    total_steps: int,
    start: float = TEMPERATURE_START,
    end: float = TEMPERATURE_END,
) -> float:
    """Linear interpolation from `start` at step 0 to `end` at the last step.

    Defined as a pure function of the step rather than as internal state on
    the gate, so that the value written into the training log and the value
    used in the forward pass cannot drift apart.
    """
    if total_steps < 1:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    if not 0 <= step < total_steps:
        raise ValueError(f"step {step} is outside [0, {total_steps})")
    if total_steps == 1:
        return float(end)
    fraction = step / (total_steps - 1)
    return float(start + (end - start) * fraction)


class ScalarGate(nn.Module):
    """One learned gate for one projector.

    Per-layer and per-tensor-kind: keys and values at the same layer get
    separate instances, because the reference gates them separately and
    because the geometry file already shows the two behave differently with
    depth.
    """

    def __init__(
        self,
        init_logit: float = INIT_LOGIT,
        init_magnitude: float = 0.0,
    ):
        super().__init__()
        self.logit = nn.Parameter(torch.tensor(float(init_logit)))
        self.magnitude = nn.Parameter(torch.tensor(float(init_magnitude)))
        self.register_buffer(
            "temperature", torch.tensor(TEMPERATURE_START), persistent=True
        )

    def set_temperature(self, temperature: float) -> None:
        if not temperature > 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        with torch.no_grad():
            self.temperature.fill_(float(temperature))

    @property
    def is_open(self) -> bool:
        """What the gate decides at inference, as a plain bool.

        This is what the activation ratio in A.4.2 counts, and reading it
        does not require a forward pass.
        """
        return bool(self.logit.item() > 0.0)

    def decision(self, generator: torch.Generator | None = None) -> torch.Tensor:
        """The `gate` term alone, without the magnitude.

        Under training this is a relaxed Bernoulli sample in the open
        interval (0, 1). Under evaluation it is a threshold, in {0.0, 1.0}
        exactly.
        """
        if not self.training:
            return (self.logit > 0).to(self.logit.dtype)

        uniform = torch.rand(
            (),
            generator=generator,
            dtype=self.logit.dtype,
            device=self.logit.device,
        )
        # Clamped away from both ends: log(0) is not a number the gradient
        # survives, and a single unlucky draw would poison the whole step.
        tiny = torch.finfo(self.logit.dtype).tiny
        uniform = uniform.clamp(tiny, 1.0 - torch.finfo(self.logit.dtype).eps)
        logistic_noise = torch.log(uniform) - torch.log1p(-uniform)
        return torch.sigmoid((self.logit + logistic_noise) / self.temperature)

    def value(self, generator: torch.Generator | None = None) -> torch.Tensor:
        """The full scalar the Sharer contribution is multiplied by."""
        return self.decision(generator) * torch.sigmoid(self.magnitude)

    def forward(
        self, contribution: torch.Tensor, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        return self.value(generator) * contribution

    def extra_repr(self) -> str:
        return (
            f"logit={self.logit.item():.4f}, "
            f"magnitude={torch.sigmoid(self.magnitude).item():.4f}, "
            f"temperature={self.temperature.item():.4g}"
        )


@torch.no_grad()
def close_gate(module: nn.Module, logit: float = -10.0) -> None:
    """Force every gate under `module` shut.

    Used by the substrate gate to produce a fused cache that must equal the
    Receiver's own exactly. The logit only has to be negative; the value is
    generous so that a later change to the threshold convention does not
    quietly turn this into a no-op.
    """
    if logit >= 0:
        raise ValueError(f"a closing logit must be negative, got {logit}")
    for gate in module.modules():
        if isinstance(gate, ScalarGate):
            gate.logit.fill_(float(logit))


@torch.no_grad()
def open_gate(module: nn.Module, logit: float = 10.0) -> None:
    """Force every gate under `module` open, magnitude untouched."""
    if logit <= 0:
        raise ValueError(f"an opening logit must be positive, got {logit}")
    for gate in module.modules():
        if isinstance(gate, ScalarGate):
            gate.logit.fill_(float(logit))