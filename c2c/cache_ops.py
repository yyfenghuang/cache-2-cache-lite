"""Operations on a key-value cache.

Pure with respect to its arguments: every function here returns a new cache
and leaves the one it was given untouched. That matters more than it looks.
A forward pass appends to whatever cache it is handed, so a cache reused
across two conditions is a different object by the time the second one runs,
and the resulting comparison silently measures nothing.
"""

from __future__ import annotations

import torch
from transformers.cache_utils import DynamicCache

__all__ = [
    "cache_tensors",
    "build_cache",
    "clone_cache",
    "slice_cache",
    "corrupt_norm_matched",
    "absolute_position_ids",
    "relative_position_ids",
    "max_abs_difference",
]

KINDS = ("keys", "values")


def cache_tensors(cache: DynamicCache) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Read the cache out as plain tensors, one (keys, values) pair per layer.

    Shape per tensor is [batch, n_kv_heads, seq, head_dim]: the heads are the
    key-value heads, before expansion to query heads. That is the shape the
    contract records and the shape a projector has to accept.
    """
    return [(layer.keys, layer.values) for layer in cache.layers]


def build_cache(pairs) -> DynamicCache:
    """Assemble a cache from (keys, values) pairs, one per layer, in order."""
    cache = DynamicCache()
    for layer_idx, (keys, values) in enumerate(pairs):
        cache.update(keys, values, layer_idx)
    return cache


def clone_cache(cache: DynamicCache) -> DynamicCache:
    """A detached copy that shares no storage with the original."""
    return build_cache(
        (k.detach().clone(), v.detach().clone()) for k, v in cache_tensors(cache)
    )


def slice_cache(cache: DynamicCache, start: int, end: int) -> DynamicCache:
    """Keep positions [start, end) of every layer.

    The slice carries no record of where it came from. Its keys were rotated
    at their original absolute positions and still are; nothing in the tensor
    says so. Whatever is forwarded on top of this slice has to supply that
    information itself, which is the entire content of the position id gate.
    """
    length = cache.get_seq_length()
    if not 0 <= start < end <= length:
        raise ValueError(
            f"slice [{start}, {end}) is not inside a cache of length {length}"
        )
    return build_cache(
        (k[:, :, start:end, :].detach().clone(),
         v[:, :, start:end, :].detach().clone())
        for k, v in cache_tensors(cache)
    )


def corrupt_norm_matched(
    cache: DynamicCache,
    generator: torch.Generator,
    kinds: tuple[str, ...] = KINDS,
) -> DynamicCache:
    """Replace each vector's direction, keeping each vector's length.

    Matching the norm per vector rather than per tensor is what makes this a
    control rather than a demonstration. A corruption that also changes
    magnitude degrades the output for two reasons at once, and the result
    cannot separate "the cache was read" from "the cache was read and it was
    the wrong size". Here every vector along head_dim keeps its own norm
    exactly, so the only thing that changed is where it points.

    Zero vectors stay zero: there is no direction to randomize.
    """
    for kind in kinds:
        if kind not in KINDS:
            raise ValueError(f"unknown kind {kind!r}, expected any of {KINDS}")

    def scramble(tensor: torch.Tensor) -> torch.Tensor:
        norms = tensor.norm(dim=-1, keepdim=True)
        noise = torch.randn(
            tensor.shape, generator=generator, dtype=tensor.dtype,
            device=tensor.device,
        )
        noise_norms = noise.norm(dim=-1, keepdim=True)
        noise = noise / noise_norms.clamp_min(torch.finfo(tensor.dtype).tiny)
        return noise * norms

    return build_cache(
        (
            scramble(k) if "keys" in kinds else k.detach().clone(),
            scramble(v) if "values" in kinds else v.detach().clone(),
        )
        for k, v in cache_tensors(cache)
    )


def absolute_position_ids(
    past_length: int, new_length: int, device=None
) -> torch.Tensor:
    """Positions the new tokens actually occupy in the original sequence."""
    if past_length < 0 or new_length < 1:
        raise ValueError(
            f"past_length={past_length}, new_length={new_length} is not a "
            "forward step"
        )
    return torch.arange(
        past_length, past_length + new_length, device=device
    ).unsqueeze(0)


def relative_position_ids(new_length: int, device=None) -> torch.Tensor:
    """Positions counted from the start of the chunk being forwarded.

    This is the wrong answer. It exists so that a test can assert it is
    wrong, because a test that only exercises the correct path cannot detect
    the bug it is supposed to guard against.
    """
    if new_length < 1:
        raise ValueError(f"new_length={new_length} is not a forward step")
    return torch.arange(0, new_length, device=device).unsqueeze(0)


def max_abs_difference(a: torch.Tensor, b: torch.Tensor) -> float:
    """The one number every condition in the substrate gate reports."""
    if a.shape != b.shape:
        raise ValueError(f"shapes differ: {tuple(a.shape)} against {tuple(b.shape)}")
    return float((a - b).abs().max())