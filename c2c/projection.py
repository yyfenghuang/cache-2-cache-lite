"""A projection from the Sharer's cache to the Receiver's, written from scratch.

No I/O, no model loading, no pretrained weights. Checkpoints exist upstream
for exactly this model pair; using them would answer whether the mechanism
works and not whether it is understood, which is the question this repository
was built to ask.

The bias is load-bearing and not an implementation detail. The number this
projection is graded against is the error of the best constant predictor, and
a constant predictor is a bias with the weights at zero. Without a bias the
projection would be unable to reach the floor it is being asked to clear, and
a failure to clear it would say nothing about the input.

Keys and values get separate instances. They are separate tensors with
different scales and, as the geometric null shows, different difficulty
profiles across depth. Sharing a projection between them would make the
key against value comparison meaningless in the same step that made it
cheaper.
"""

from __future__ import annotations

import torch
from torch import nn

from c2c.cache_ops import flatten_heads, unflatten_heads

__all__ = ["CacheProjection", "constant_predictor_state"]

ACTIVATIONS = {"gelu": nn.GELU, "silu": nn.SiLU, "relu": nn.ReLU}


class CacheProjection(nn.Module):
    """Map one cache tensor to another cache tensor's shape.

    Input and output are both [batch, heads, positions, head_dim], with the
    head count and head dimension of their own model. Internally the head
    axis is folded into the channel axis, so the projection sees one vector
    per position at the source `kv_width` and produces one at the target
    `kv_width`.

    `depth` is the number of linear layers. One is an affine map, which is
    the smallest thing that can clear the constant-predictor floor and
    therefore the honest baseline. Two or more insert hidden layers with an
    activation between, which is what the reference implementation uses.
    """

    def __init__(
        self,
        source_heads: int,
        source_head_dim: int,
        target_heads: int,
        target_head_dim: int,
        *,
        depth: int = 1,
        hidden: int | None = None,
        activation: str = "gelu",
    ):
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be at least 1, got {depth}")
        if depth > 1 and hidden is None:
            raise ValueError("depth above 1 needs a hidden width")
        if depth == 1 and hidden is not None:
            raise ValueError("a single layer has no hidden width to set")
        if activation not in ACTIVATIONS:
            raise ValueError(
                f"unknown activation {activation!r}, expected any of "
                f"{tuple(ACTIVATIONS)}"
            )

        self.source_heads = source_heads
        self.source_head_dim = source_head_dim
        self.target_heads = target_heads
        self.target_head_dim = target_head_dim
        self.source_width = source_heads * source_head_dim
        self.target_width = target_heads * target_head_dim
        self.depth = depth
        self.hidden = hidden
        self.activation = activation

        widths = [self.source_width] + [hidden] * (depth - 1) + [self.target_width]
        layers: list[nn.Module] = []
        for index in range(depth):
            layers.append(nn.Linear(widths[index], widths[index + 1], bias=True))
            if index < depth - 1:
                layers.append(ACTIVATIONS[activation]())
        self.net = nn.Sequential(*layers)

    @property
    def output_layer(self) -> nn.Linear:
        """The last linear layer, whose bias is the constant predictor."""
        return self.net[-1]

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def n_parameters_per_output_channel(self) -> float:
        """How many degrees of freedom each predicted channel costs.

        The ratio of training positions to this number is what decides
        whether a loss below the null means the projection learned something
        or merely consumed the data. At three the fit explains a third of the
        target from an input that carries nothing.
        """
        return self.n_parameters() / self.target_width

    def forward(self, cache_tensor: torch.Tensor) -> torch.Tensor:
        batch, heads, positions, head_dim = cache_tensor.shape
        if heads != self.source_heads or head_dim != self.source_head_dim:
            raise ValueError(
                f"expected [{batch}, {self.source_heads}, {positions}, "
                f"{self.source_head_dim}], got {tuple(cache_tensor.shape)}"
            )
        flat = flatten_heads(cache_tensor)
        projected = self.net(flat)
        return unflatten_heads(
            projected, batch, self.target_heads, self.target_head_dim
        )

    def describe(self) -> dict:
        return {
            "source_heads": self.source_heads,
            "source_head_dim": self.source_head_dim,
            "source_width": self.source_width,
            "target_heads": self.target_heads,
            "target_head_dim": self.target_head_dim,
            "target_width": self.target_width,
            "depth": self.depth,
            "hidden": self.hidden,
            "activation": self.activation if self.depth > 1 else None,
            "n_parameters": self.n_parameters(),
            "n_parameters_per_output_channel": self.n_parameters_per_output_channel(),
        }


@torch.no_grad()
def constant_predictor_state(
    projection: CacheProjection, target_mean: torch.Tensor
) -> None:
    """Set the projection to ignore its input and emit `target_mean`.

    This is the null made executable rather than described. Every weight goes
    to zero and the output bias takes the per channel mean, so the projection
    reproduces exactly the error the null records. A projection that cannot be
    put into this state is not being graded against the number it is compared
    to.
    """
    if target_mean.shape != (projection.target_width,):
        raise ValueError(
            f"expected a mean of width {projection.target_width}, got "
            f"{tuple(target_mean.shape)}"
        )
    for module in projection.net:
        if isinstance(module, nn.Linear):
            module.weight.zero_()
            module.bias.zero_()
    projection.output_layer.bias.copy_(target_mean)