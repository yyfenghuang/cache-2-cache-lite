"""The fuser, which adds where the projection replaced.

No I/O, no model loading.

The tier below this one trained a map from the Sharer's cache to the
Receiver's and then handed the result to the Receiver in place of its own.
That build is not the mechanism the paper claims and was never claimed to
work; Table 8 prices the difference between them at 24.18 points. The
mechanism is this:

    fused = receiver + gate * sigmoid(scalar) * f(concat(receiver, sharer))

Two consequences follow from the shape of that line and neither is optional.

The Receiver's own cache is on both sides. Under mean squared error against
that cache the loss is minimised at `f = 0`, because the target is already
present before `f` contributes anything. There is no ground truth for a
correction. The objective is therefore next-token prediction on the
Receiver's response with both models frozen, and the change of objective is
entailed by the residual rather than chosen alongside it.

The Receiver's cache is also an input to `f`, not merely something added
afterwards. That is what makes this a fuser rather than a projector with a
skip connection: the correction is allowed to depend on what the Receiver
already understood, so the same Sharer cache can produce different
corrections depending on where the Receiver is uncertain.

Why the output layer is not zero-initialised
--------------------------------------------
There is a cheaper-looking way to start this system at the Receiver's
baseline: initialise the last linear layer to zero, so the correction begins
at zero and grows. It is rejected. A zero output layer means every gradient
reaching the layers behind it is multiplied by zero, so the first backward
pass reports exactly zero gradient throughout the projection. That is
indistinguishable, from inside the training loop, from the failure the
substrate gate exists to detect: a cache that arrived detached and never
connected the loss to this module at all. The floor is provided by the gate
instead, which multiplies the finished contribution rather than sitting
inside the path the gradient has to travel.
"""

from __future__ import annotations

import torch
from torch import nn

from c2c.alignment import align_layers
from c2c.cache_ops import flatten_heads, unflatten_heads
from c2c.gate import ScalarGate

__all__ = ["CacheFuser", "FuserBank", "KINDS"]

KINDS = ("keys", "values")

ACTIVATIONS = {"gelu": nn.GELU, "silu": nn.SiLU, "relu": nn.ReLU}

# Matches the hidden width the projection tier was trained at. Held equal on
# purpose: the rung that prices the residual against replacement compares a
# projection-only build with this one, and a capacity difference between them
# would sit inside the comparison as a second explanation for any gap.
HIDDEN = 256


class CacheFuser(nn.Module):
    """One target layer, one tensor kind.

    Keys and values get separate instances at every layer, for the same
    reason the projections did: they are different tensors with different
    scales, and `results/geometric_null.json` already shows their difficulty
    diverging with depth.

    Three parts, named after the reference's three modules rather than
    collapsed into one `Sequential`, so that an ablation can remove one
    without the removal having to be reconstructed from a slice index.

    `project` and `fuse` are the projection and feature fusion layers. They
    read the two caches concatenated along the channel axis and emit one
    correction at the Receiver's width.

    `weighting` is the dynamic weighting module: one scalar per head per
    token, read from the same concatenation, passed through a sigmoid, and
    multiplied into the correction. Per head and per token rather than per
    layer, which is what lets the same fused layer contribute strongly at one
    position and not at another.

    `gate` is the learned per-layer decision. See `c2c/gate.py`.
    """

    def __init__(
        self,
        receiver_heads: int,
        receiver_head_dim: int,
        sharer_heads: int,
        sharer_head_dim: int,
        *,
        hidden: int = HIDDEN,
        activation: str = "gelu",
    ):
        super().__init__()
        for name, count in (
            ("receiver_heads", receiver_heads),
            ("receiver_head_dim", receiver_head_dim),
            ("sharer_heads", sharer_heads),
            ("sharer_head_dim", sharer_head_dim),
            ("hidden", hidden),
        ):
            if count < 1:
                raise ValueError(f"{name} must be positive, got {count}")
        if activation not in ACTIVATIONS:
            raise ValueError(
                f"unknown activation {activation!r}, expected any of "
                f"{tuple(ACTIVATIONS)}"
            )

        self.receiver_heads = receiver_heads
        self.receiver_head_dim = receiver_head_dim
        self.sharer_heads = sharer_heads
        self.sharer_head_dim = sharer_head_dim
        self.receiver_width = receiver_heads * receiver_head_dim
        self.sharer_width = sharer_heads * sharer_head_dim
        self.joint_width = self.receiver_width + self.sharer_width
        self.hidden = hidden

        self.project = nn.Linear(self.joint_width, hidden, bias=True)
        self.activation = ACTIVATIONS[activation]()
        self.fuse = nn.Linear(hidden, self.receiver_width, bias=True)
        self.weighting = nn.Linear(self.joint_width, receiver_heads, bias=True)
        self.gate = ScalarGate()

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _check(self, receiver: torch.Tensor, sharer: torch.Tensor) -> None:
        if receiver.dim() != 4 or sharer.dim() != 4:
            raise ValueError(
                f"expected two 4-dimensional caches, got "
                f"{tuple(receiver.shape)} and {tuple(sharer.shape)}"
            )
        expected_r = (self.receiver_heads, self.receiver_head_dim)
        expected_s = (self.sharer_heads, self.sharer_head_dim)
        if (receiver.shape[1], receiver.shape[3]) != expected_r:
            raise ValueError(
                f"receiver cache is {tuple(receiver.shape)}, expected heads "
                f"{expected_r[0]} and head_dim {expected_r[1]}"
            )
        if (sharer.shape[1], sharer.shape[3]) != expected_s:
            raise ValueError(
                f"sharer cache is {tuple(sharer.shape)}, expected heads "
                f"{expected_s[0]} and head_dim {expected_s[1]}"
            )
        if receiver.shape[0] != sharer.shape[0]:
            raise ValueError(
                f"batch sizes differ: {receiver.shape[0]} against "
                f"{sharer.shape[0]}"
            )
        if receiver.shape[2] != sharer.shape[2]:
            raise ValueError(
                f"position counts differ: {receiver.shape[2]} against "
                f"{sharer.shape[2]}. The two caches are concatenated position "
                "by position, so a mismatch here is a token alignment "
                "failure and not a shape to broadcast around"
            )

    def correction(
        self,
        receiver: torch.Tensor,
        sharer: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Everything that is added to the Receiver's cache, gate included.

        Exposed separately from `forward` because the substrate gate has to
        assert that this is exactly zero when the gate is shut, and asserting
        it on the difference of two large tensors would test float32
        cancellation rather than the gate.
        """
        self._check(receiver, sharer)
        batch, _, positions, _ = receiver.shape

        joint = torch.cat(
            [flatten_heads(receiver), flatten_heads(sharer)], dim=-1
        )
        fused = self.fuse(self.activation(self.project(joint)))
        weights = torch.sigmoid(self.weighting(joint))

        modulated = fused.view(-1, self.receiver_heads, self.receiver_head_dim)
        modulated = modulated * weights.unsqueeze(-1)
        modulated = modulated.reshape(-1, self.receiver_width)

        return self.gate(
            unflatten_heads(
                modulated, batch, self.receiver_heads, self.receiver_head_dim
            ),
            generator,
        )

    def forward(
        self,
        receiver: torch.Tensor,
        sharer: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return receiver + self.correction(receiver, sharer, generator)

    def extra_repr(self) -> str:
        return (
            f"{self.receiver_width} + {self.sharer_width} -> {self.hidden} "
            f"-> {self.receiver_width}"
        )


class FuserBank(nn.Module):
    """One fuser per aligned target layer per tensor kind.

    Target layers with no Sharer partner are not a special case to be
    tolerated. Terminal alignment leaves the Receiver's four shallowest
    layers unpaired, and their caches pass through this module untouched and
    unfused. That is the correct behaviour and it is the reason
    `align_layers` returns a list indexed by target with `None` in it: a
    caller working from a list of pairs would produce a bank of 24 entries
    and then silently misalign it against a 28 layer cache.
    """

    def __init__(
        self,
        n_receiver_layers: int,
        n_sharer_layers: int,
        receiver_heads: int,
        receiver_head_dim: int,
        sharer_heads: int,
        sharer_head_dim: int,
        *,
        hidden: int = HIDDEN,
        activation: str = "gelu",
        strategy: str = "terminal",
    ):
        super().__init__()
        self.mapping = align_layers(
            n_sharer_layers, n_receiver_layers, strategy=strategy
        )
        self.n_receiver_layers = n_receiver_layers
        self.n_sharer_layers = n_sharer_layers

        def make() -> nn.ModuleList:
            return nn.ModuleList(
                CacheFuser(
                    receiver_heads,
                    receiver_head_dim,
                    sharer_heads,
                    sharer_head_dim,
                    hidden=hidden,
                    activation=activation,
                )
                if source is not None
                else nn.Identity()
                for source in self.mapping
            )

        # Two named lists rather than one `nn.ModuleDict` keyed by kind.
        # `ModuleDict` registers its keys as attributes, and both of the
        # names this project uses for the two tensor kinds are already
        # methods on it, so that construction raises. Keeping them as
        # separate attributes also means a key-only or value-only ablation is
        # a change of one line in `forward`.
        self.key_fusers = make()
        self.value_fusers = make()

    def fusers_for(self, kind: str) -> nn.ModuleList:
        if kind not in KINDS:
            raise ValueError(f"unknown kind {kind!r}, expected any of {KINDS}")
        return self.key_fusers if kind == "keys" else self.value_fusers

    @property
    def paired_layers(self) -> list[int]:
        return [t for t, s in enumerate(self.mapping) if s is not None]

    @property
    def unpaired_layers(self) -> list[int]:
        return [t for t, s in enumerate(self.mapping) if s is None]

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def gate_activation_ratio(self) -> float:
        """The fraction of gates that are open, as A.4.2 counts them.

        Reported against 52.67 percent under task-specific training and
        98.21 percent under general-purpose training.
        """
        gates = [g for g in self.modules() if isinstance(g, ScalarGate)]
        if not gates:
            return float("nan")
        return sum(1.0 for g in gates if g.is_open) / len(gates)

    def set_temperature(self, temperature: float) -> None:
        for gate in self.modules():
            if isinstance(gate, ScalarGate):
                gate.set_temperature(temperature)

    def forward(
        self,
        receiver_pairs: list[tuple[torch.Tensor, torch.Tensor]],
        sharer_pairs: list[tuple[torch.Tensor, torch.Tensor]],
        generator: torch.Generator | None = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Fuse layer by layer, returning tensors in the Receiver's geometry.

        Takes and returns plain tensors rather than a cache object. The cache
        class belongs to the framework and appending to one has side effects;
        keeping it out of this module is what lets the whole bank be
        exercised without a model.
        """
        if len(receiver_pairs) != self.n_receiver_layers:
            raise ValueError(
                f"expected {self.n_receiver_layers} receiver layers, got "
                f"{len(receiver_pairs)}"
            )
        if len(sharer_pairs) != self.n_sharer_layers:
            raise ValueError(
                f"expected {self.n_sharer_layers} sharer layers, got "
                f"{len(sharer_pairs)}"
            )

        out: list[tuple[torch.Tensor, torch.Tensor]] = []
        for target, source in enumerate(self.mapping):
            if source is None:
                out.append(receiver_pairs[target])
                continue
            fused = tuple(
                self.fusers_for(kind)[target](
                    receiver_pairs[target][index],
                    sharer_pairs[source][index],
                    generator,
                )
                for index, kind in enumerate(KINDS)
            )
            out.append(fused)
        return out