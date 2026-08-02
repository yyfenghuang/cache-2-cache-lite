"""C2C-Lite: mechanism only. Nothing in this package performs I/O."""

from c2c.alignment import align_layers, aligned_pairs, align_tokens
from c2c.contracts import (
    build_contracts,
    compare_rope,
    model_contract,
    resolve_rope,
    rope_inv_freq,
)

__all__ = [
    "align_layers",
    "aligned_pairs",
    "align_tokens",
    "build_contracts",
    "compare_rope",
    "model_contract",
    "resolve_rope",
    "rope_inv_freq",
]