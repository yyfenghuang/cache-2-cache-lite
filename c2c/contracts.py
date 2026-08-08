# SPDX-License-Identifier: Apache-2.0
"""Config contract for a Sharer/Receiver pair.

Everything here is a pure function of configuration objects. A "config" is anything
exposing the attribute names a HuggingFace ``PretrainedConfig`` exposes; the
functions duck-type rather than import, which is what lets this tier run
before a checkpoint is on disk.

Two facts are recorded, because deriving them from a
config is not possible for this model family:

- ``has_q_norm`` and ``has_k_norm`` live in the modeling code, not the config.
  They are required keyword arguments so that a caller cannot silently omit
  the probe and leave the field guessed.
- The rotary inverse frequency ladder is recomputed here from the same
  formula the library uses. The caller is expected to cross-check the result
  against the library's own initializer and record whether the cross-check
  ran.

Load-bearing fields have no fallback, and presence is not accepted as
evidence of meaning. The rotary base is resolved through ``resolve_rope``
rather than read off an attribute, because in transformers v5 the deprecated
``config.rope_theta`` is still present and still readable while the value the
model uses lives in ``config.rope_parameters``. Fields that carry a
legitimate convention default record whether the value was declared, so that
the distinction survives into the file.
"""

from __future__ import annotations

__all__ = [
    "rope_inv_freq",
    "resolve_rope",
    "model_contract",
    "compare_rope",
    "build_contracts",
]


def rope_inv_freq(
    head_dim: int,
    rope_theta: float,
    partial_rotary_factor: float = 1.0,
) -> list[float]:
    """Return the rotary inverse frequency ladder for one model.

    Mirrors ``_compute_default_rope_parameters`` in transformers:

        dim      = int(head_dim * partial_rotary_factor)
        inv_freq = 1 / theta ** (arange(0, dim, 2) / dim)

    The ladder has ``dim // 2`` entries. Entry 0 is always exactly 1.0 and the
    last entry is theta ** (-(dim - 2) / dim), so two ladders built from the
    same theta span the same range whatever their length. What differs is how
    finely that range is sampled, and which channel index each frequency lands
    on.
    """
    if head_dim <= 0:
        raise ValueError(f"head_dim must be positive, got {head_dim}")
    if rope_theta <= 0:
        raise ValueError(f"rope_theta must be positive, got {rope_theta}")

    dim = int(head_dim * partial_rotary_factor)
    if dim < 2:
        raise ValueError(f"rotary dim must be at least 2, got {dim}")

    return [1.0 / (rope_theta ** (i / dim)) for i in range(0, dim, 2)]


def resolve_rope(config, *, name: str) -> dict:
    """Find the rotary base the model actually uses, and record the other one.

    Transformers v5 moved the rotary parameters into ``config.rope_parameters``
    and deprecated the standalone ``config.rope_theta``. The deprecated
    attribute did not disappear: it remains present, remains readable, and on
    a v5 load it can hold a class default while the value from the checkpoint
    sits in the dictionary. ``hasattr`` therefore proves nothing here, and a
    contract that trusts it records a number no forward pass will ever use.

    The dictionary wins when it exists, because it is what
    ``ROPE_INIT_FUNCTIONS`` reads. The legacy attribute is recorded beside it
    rather than dropped, so that a later reader can see the discrepancy
    instead of having to rediscover it.
    """
    params = getattr(config, "rope_parameters", None)
    legacy = getattr(config, "rope_theta", None)
    legacy = None if legacy is None else float(legacy)

    if params is None:
        if legacy is None:
            raise ValueError(
                f"{name}: config exposes neither rope_parameters nor "
                "rope_theta; there is nothing to read"
            )
        resolved = {
            "rope_theta": legacy,
            "rope_type": "default",
            "source": "config.rope_theta",
            "rope_parameters": None,
        }
    elif "rope_theta" in params:
        resolved = {
            "rope_theta": float(params["rope_theta"]),
            "rope_type": params.get("rope_type", "default"),
            "source": "config.rope_parameters",
            "rope_parameters": dict(params),
        }
    else:
        # Nested form: one entry per layer type. Uniform is the only case a
        # single-ladder contract can honestly describe.
        nested = {k: v for k, v in params.items() if isinstance(v, dict)}
        if not nested:
            raise ValueError(
                f"{name}: rope_parameters has no rope_theta and no nested "
                f"per-layer-type entries: {params!r}"
            )
        thetas = {k: float(v["rope_theta"]) for k, v in nested.items()}
        if len(set(thetas.values())) != 1:
            raise ValueError(
                f"{name}: rotary base differs per layer type {thetas!r}; a "
                "single-ladder contract cannot describe this model"
            )
        first = next(iter(nested.values()))
        resolved = {
            "rope_theta": next(iter(thetas.values())),
            "rope_type": first.get("rope_type", "default"),
            "source": "config.rope_parameters (per layer type, uniform)",
            "rope_parameters": dict(params),
        }

    resolved["legacy_attr"] = legacy
    resolved["legacy_agrees"] = (
        None if legacy is None else legacy == resolved["rope_theta"]
    )
    return resolved


def model_contract(
    config,
    *,
    name: str,
    has_q_norm: bool,
    has_k_norm: bool,
) -> dict:
    """Record the KV geometry of one model as a flat dictionary.

    ``has_q_norm`` and ``has_k_norm`` are required because the config does not
    carry them. The caller determines them by inspecting the attention module
    and is expected to record how it did so.
    """
    hidden_size = int(config.hidden_size)
    n_q_heads = int(config.num_attention_heads)
    n_kv_heads = int(getattr(config, "num_key_value_heads", n_q_heads))
    n_layers = int(config.num_hidden_layers)

    declared_head_dim = getattr(config, "head_dim", None)
    head_dim_from_hidden = hidden_size // n_q_heads
    head_dim = (
        int(declared_head_dim)
        if declared_head_dim is not None
        else head_dim_from_hidden
    )

    if n_q_heads % n_kv_heads != 0:
        raise ValueError(
            f"{name}: {n_q_heads} query heads is not a multiple of "
            f"{n_kv_heads} key-value heads"
        )

    rope = resolve_rope(config, name=name)
    rope_theta = rope["rope_theta"]

    declared_prf = getattr(config, "partial_rotary_factor", None)
    partial_rotary_factor = 1.0 if declared_prf is None else float(declared_prf)

    inv_freq = rope_inv_freq(head_dim, rope_theta, partial_rotary_factor)

    return {
        "name": name,
        "model_type": getattr(config, "model_type", None),
        "n_layers": n_layers,
        "hidden_size": hidden_size,
        "n_q_heads": n_q_heads,
        "n_kv_heads": n_kv_heads,
        "n_kv_heads_declared_in_config": (
            getattr(config, "num_key_value_heads", None) is not None
        ),
        "gqa_group_size": n_q_heads // n_kv_heads,
        "head_dim": head_dim,
        "head_dim_declared_in_config": declared_head_dim is not None,
        "head_dim_from_hidden_size": head_dim_from_hidden,
        "head_dim_is_decoupled": head_dim != head_dim_from_hidden,
        "kv_width": n_kv_heads * head_dim,
        "cache_tensor_shape": "[batch, n_kv_heads, seq, head_dim]",
        "rope_theta": rope_theta,
        "rope_theta_source": rope["source"],
        "rope_theta_legacy_attr": rope["legacy_attr"],
        "rope_theta_legacy_agrees": rope["legacy_agrees"],
        "rope_type": rope["rope_type"],
        "rope_parameters": rope["rope_parameters"],
        "partial_rotary_factor": partial_rotary_factor,
        "partial_rotary_factor_declared_in_config": declared_prf is not None,
        "n_inv_freq": len(inv_freq),
        "inv_freq": inv_freq,
        "has_q_norm": bool(has_q_norm),
        "has_k_norm": bool(has_k_norm),
        "vocab_size": int(getattr(config, "vocab_size", 0)),
        "max_position_embeddings": int(
            getattr(config, "max_position_embeddings", 0)
        ),
    }


def compare_rope(sharer: dict, receiver: dict, *, rtol: float = 1e-9) -> dict:
    """Compare two rotary ladders element-wise, and test for nesting.

    Element-wise comparison is only defined when the two ladders have the same
    length. When they do not, the coarser ladder may still be a decimation of
    the finer one: if the lengths are in an integer ratio ``s`` and the thetas
    match, then ``short[i]`` and ``long[s * i]`` are the same frequency. That
    is a stronger statement than "the ladders differ" and a weaker one than
    "the ladders match", and it has a different consequence for projection
    design, so it is recorded separately rather than folded into a boolean.
    """
    a = sharer["inv_freq"]
    b = receiver["inv_freq"]

    theta_match = sharer["rope_theta"] == receiver["rope_theta"]
    length_match = len(a) == len(b)

    elementwise_match = None
    elementwise_max_rel_dev = None
    if length_match:
        devs = [abs(x - y) / max(abs(y), 1e-30) for x, y in zip(a, b)]
        elementwise_max_rel_dev = max(devs) if devs else 0.0
        elementwise_match = elementwise_max_rel_dev <= rtol

    short, long = (a, b) if len(a) <= len(b) else (b, a)
    short_name = sharer["name"] if len(a) <= len(b) else receiver["name"]
    long_name = receiver["name"] if len(a) <= len(b) else sharer["name"]

    nesting_holds = False
    nesting_stride = None
    nesting_max_rel_dev = None
    if len(short) > 0 and len(long) % len(short) == 0:
        nesting_stride = len(long) // len(short)
        devs = [
            abs(short[i] - long[nesting_stride * i])
            / max(abs(long[nesting_stride * i]), 1e-30)
            for i in range(len(short))
        ]
        nesting_max_rel_dev = max(devs) if devs else 0.0
        nesting_holds = nesting_max_rel_dev <= rtol

    return {
        "theta_match": theta_match,
        "length_match": length_match,
        "n_inv_freq": {sharer["name"]: len(a), receiver["name"]: len(b)},
        "elementwise_match": elementwise_match,
        "elementwise_max_rel_dev": elementwise_max_rel_dev,
        "nesting": {
            "coarse": short_name,
            "fine": long_name,
            "stride": nesting_stride,
            "holds": nesting_holds,
            "max_rel_dev": nesting_max_rel_dev,
        },
        "range_span": {
            sharer["name"]: [a[0], a[-1]] if a else None,
            receiver["name"]: [b[0], b[-1]] if b else None,
        },
    }


def build_contracts(
    sharer_config,
    receiver_config,
    *,
    sharer_name: str,
    receiver_name: str,
    sharer_qk_norm: tuple[bool, bool],
    receiver_qk_norm: tuple[bool, bool],
) -> dict:
    """The whole contract as a pure function of two configs.

    ``*_qk_norm`` is ``(has_q_norm, has_k_norm)``.
    """
    sharer = model_contract(
        sharer_config,
        name=sharer_name,
        has_q_norm=sharer_qk_norm[0],
        has_k_norm=sharer_qk_norm[1],
    )
    receiver = model_contract(
        receiver_config,
        name=receiver_name,
        has_q_norm=receiver_qk_norm[0],
        has_k_norm=receiver_qk_norm[1],
    )

    total_kv_width = sharer["kv_width"] + receiver["kv_width"]

    return {
        "sharer": sharer,
        "receiver": receiver,
        "rope_comparison": compare_rope(sharer, receiver),
        "concat_widths": {
            "sharer_kv_width": sharer["kv_width"],
            "receiver_kv_width": receiver["kv_width"],
            "concat_width": total_kv_width,
            "sharer_share": sharer["kv_width"] / total_kv_width,
            "receiver_share": receiver["kv_width"] / total_kv_width,
        },
        "layer_counts": {
            "n_source_layers": sharer["n_layers"],
            "n_target_layers": receiver["n_layers"],
        },
    }