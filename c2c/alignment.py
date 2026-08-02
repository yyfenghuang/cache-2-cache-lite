"""Layer and token alignment between a Sharer and a Receiver.
"""

from __future__ import annotations

__all__ = ["align_layers", "aligned_pairs", "align_tokens"]

_STRATEGIES = ("terminal",)


def align_layers(
    n_source: int,
    n_target: int,
    strategy: str = "terminal",
) -> list[int | None]:
    """Map every target layer to the source layer that feeds it.

    Returns a list of length ``n_target``. Entry ``t`` is the source layer
    index feeding target layer ``t``, or ``None`` when target layer ``t`` has
    no partner. The return is indexed by target rather than by source because
    the fuser is instantiated per target layer, and a target layer with no
    partner is a case the caller must handle rather than a case that can be
    dropped from a list of pairs without anyone noticing.

    Terminal alignment anchors the last source layer to the last target layer
    and counts backwards. With 24 source and 28 target layers:

        target 27 <- source 23
        target 26 <- source 22
        ...
        target  4 <- source  0
        target  3 <- None

    The same formula handles the reverse case. With 28 source and 24 target
    layers the shallow source layers 0 through 3 are dropped and every target
    layer has a partner.

    >>> align_layers(4, 8)
    [None, None, None, None, 0, 1, 2, 3]
    >>> align_layers(8, 4)
    [4, 5, 6, 7]
    """
    if strategy not in _STRATEGIES:
        raise ValueError(
            f"unknown strategy {strategy!r}, expected one of {_STRATEGIES}"
        )
    if n_source < 1 or n_target < 1:
        raise ValueError(
            f"layer counts must be positive, got "
            f"n_source={n_source}, n_target={n_target}"
        )

    offset = n_target - n_source
    mapping: list[int | None] = []
    for t in range(n_target):
        s = t - offset
        mapping.append(s if 0 <= s < n_source else None)
    return mapping


def aligned_pairs(mapping: list[int | None]) -> list[tuple[int, int]]:
    """Drop the unpaired target layers, keeping ``(target, source)`` pairs."""
    return [(t, s) for t, s in enumerate(mapping) if s is not None]


def align_tokens(*args, **kwargs):
    """Not implemented, and deliberately so.

    The reference recipe ``recipe/train_recipe/C2C_0.6+0.5.json`` sets
    ``is_do_alignment: false`` for the Qwen3-0.6B / Qwen2.5-0.5B-Instruct
    pair, because the two tokenizers produce the same ids for the same
    string. The token-alignment machinery in the paper exists for
    cross-family pairs, which this is not.

    This raises rather than returning identity so that a caller who reaches
    it has to confront the assumption instead of inheriting it. If the
    tokenizer probe in ``results/contracts.json`` ever reports differing ids,
    that is the signal to implement this, and the probe result is the
    evidence that it is needed.
    """
    raise NotImplementedError(
        "token alignment is out of scope for this pair; see "
        "results/contracts.json -> tokenizers.probe_ids_identical"
    )