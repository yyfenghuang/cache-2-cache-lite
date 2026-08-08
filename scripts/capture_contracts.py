# SPDX-License-Identifier: Apache-2.0
"""Read two live configs and write the contract.

Silent on success. Everything it learns goes into results/contracts.json and
into the marked block in README.md; printing it again would be a third copy
that can drift from the other two.

This is the only file in Tier 0 that imports transformers. No checkpoint
weights are downloaded: only configs, tokenizers, and a single attention
module instantiated on the meta device to see whether q_norm and k_norm
exist at all.
"""

from __future__ import annotations

import importlib
import inspect
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from c2c.alignment import align_layers, aligned_pairs  # noqa: E402
from c2c.contracts import build_contracts, rope_inv_freq  # noqa: E402

SHARER_ID = "Qwen/Qwen2.5-0.5B-Instruct"
RECEIVER_ID = "Qwen/Qwen3-0.6B"

SHARER_NAME = "sharer"
RECEIVER_NAME = "receiver"

PROBE_STRINGS = [
    "The capital of France is Paris.",
    "Which of the following is correct? A. 2500  B. 3700",
    "cache-to-cache",
]

RESULTS_PATH = REPO_ROOT / "results" / "contracts.json"
README_PATH = REPO_ROOT / "README.md"

BLOCK_START = "<!-- contracts:start -->"
BLOCK_END = "<!-- contracts:end -->"

# Our ladder is float64 Python; the library builds its own in the dtype its
# rotary module uses. Agreement is therefore stated in units of that dtype's
# eps rather than as a fixed relative tolerance. A disagreement about the
# formula would show up at order one, nowhere near a fraction of an ulp, so
# this stays a real gate rather than a rubber stamp.
ROPE_CROSSCHECK_ULP = 4.0


# --------------------------------------------------------------------------
# probes


def modeling_module(config):
    """The modeling file for this architecture."""
    model_type = config.model_type
    return importlib.import_module(
        f"transformers.models.{model_type}.modeling_{model_type}"
    )


def class_named(module, pattern: str):
    """The shortest class in `module` whose own name matches `pattern`.

    Restricted to classes defined in that module, so that names imported for
    convenience do not win over the architecture's own.
    """
    candidates = [
        obj
        for nm, obj in vars(module).items()
        if isinstance(obj, type)
        and re.fullmatch(pattern, nm)
        and obj.__module__ == module.__name__
    ]
    if not candidates:
        raise RuntimeError(f"no class matching {pattern} in {module.__name__}")
    return min(candidates, key=lambda c: len(c.__name__))


def probe_qk_norm(config) -> tuple[bool, bool, str]:
    """Return (has_q_norm, has_k_norm, method).

    q_norm and k_norm are properties of the modeling code, not the config, so
    there is nothing in the config to read. The attention class is
    instantiated on the meta device, which builds the module graph without
    allocating storage for any parameter.
    """
    module = modeling_module(config)
    cls = class_named(module, r"[A-Za-z0-9]+Attention")

    try:
        import torch

        with torch.device("meta"):
            attn = cls(config, layer_idx=0)
        return (
            getattr(attn, "q_norm", None) is not None,
            getattr(attn, "k_norm", None) is not None,
            f"meta-instantiated {cls.__name__}",
        )
    except Exception:
        src = inspect.getsource(cls.__init__)
        return (
            "self.q_norm" in src,
            "self.k_norm" in src,
            f"source inspection of {cls.__name__}.__init__",
        )


def _describe(tensor, source: str, attention_scaling: float) -> dict:
    """Carry the library's dtype along with its values.

    Without the dtype there is no way to say whether a small disagreement is
    a rounding artefact or a real one, and a tolerance chosen without knowing
    it is a guess.
    """
    import torch

    return {
        "inv_freq": [float(x) for x in tensor.tolist()],
        "attention_scaling": float(attention_scaling),
        "source": source,
        "dtype": str(tensor.dtype),
        "eps": float(torch.finfo(tensor.dtype).eps),
    }


def _ladder_from_rotary_module(config) -> dict:
    """Read inv_freq off the module the model actually instantiates.

    This is closer to ground truth than any registry lookup: it is the buffer
    the forward pass reads. It costs one tensor of 32 or 64 floats.
    """
    cls = class_named(modeling_module(config), r"[A-Za-z0-9]+RotaryEmbedding")

    errors = []
    mod = None
    for build in (
        lambda: cls(config),
        lambda: cls(config, device="cpu"),
        lambda: cls(config=config),
    ):
        try:
            mod = build()
            break
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {exc}")
    if mod is None:
        raise RuntimeError(f"{cls.__name__} would not instantiate: {errors}")

    inv = getattr(mod, "inv_freq", None)
    if inv is None:
        raise RuntimeError(f"{cls.__name__} exposes no inv_freq buffer")

    return _describe(inv, cls.__name__, getattr(mod, "attention_scaling", 1.0))


def _ladder_from_rope_registry(config, rope_type: str) -> dict:
    """Fallback: the shared registry, if this version still exposes one."""
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    fn = ROPE_INIT_FUNCTIONS[rope_type]
    errors = []
    out = None
    for call in (
        lambda: fn(config, device="cpu"),
        lambda: fn(config, "cpu"),
        lambda: fn(config=config, device="cpu"),
        lambda: fn(config),
    ):
        try:
            out = call()
            break
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {exc}")
    if out is None:
        raise RuntimeError(
            f"ROPE_INIT_FUNCTIONS[{rope_type!r}] rejected every call shape: "
            f"{errors}"
        )

    return _describe(out[0], f"ROPE_INIT_FUNCTIONS[{rope_type!r}]", out[1])


def crosscheck_inv_freq(config, ours: list[float], rope_type: str) -> dict:
    """Compare our recomputed ladder against the one the library builds.

    The point of recomputing the ladder in c2c_lite/contracts.py is to know
    the formula rather than to inherit it. The point of this check is that
    knowing a formula and matching the library are different claims, and only
    the second one keeps the contract usable.

    Ours is float64, theirs is whatever their rotary module uses. The
    comparison is therefore reported in ulps of their dtype: a rounding
    difference lands below one ulp, a formula difference lands at order one,
    and nothing in between is quietly waved through.

    Every route that fails records why. One run should show the whole
    picture, not the first obstacle.
    """
    attempts = []
    theirs = None
    for route, call in (
        ("rotary module", lambda: _ladder_from_rotary_module(config)),
        ("rope registry", lambda: _ladder_from_rope_registry(config, rope_type)),
    ):
        try:
            theirs = call()
            break
        except Exception as exc:  # noqa: BLE001
            attempts.append(f"{route}: {type(exc).__name__}: {exc}")

    if theirs is None:
        return {"ran": False, "attempts": attempts}

    values = theirs["inv_freq"]
    if len(values) != len(ours):
        return {
            "ran": True,
            "match": False,
            "source": theirs["source"],
            "attempts": attempts,
            "reason": f"length {len(values)} against {len(ours)}",
        }

    max_rel = max(
        abs(a - b) / max(abs(b), 1e-30) for a, b in zip(ours, values)
    )
    max_ulp = max_rel / theirs["eps"]

    return {
        "ran": True,
        "match": max_ulp <= ROPE_CROSSCHECK_ULP,
        "source": theirs["source"],
        "attempts": attempts,
        "their_dtype": theirs["dtype"],
        "their_eps": theirs["eps"],
        "max_rel_dev": max_rel,
        "max_dev_ulp": max_ulp,
        "ulp_budget": ROPE_CROSSCHECK_ULP,
        "rope_type": rope_type,
        "attention_scaling": theirs["attention_scaling"],
    }


PROVENANCE_KEYS = (
    "rope_theta",
    "rope_parameters",
    "rope_scaling",
    "head_dim",
    "partial_rotary_factor",
    "num_key_value_heads",
    "num_attention_heads",
    "num_hidden_layers",
    "hidden_size",
    "vocab_size",
)


def config_provenance(model_id: str) -> dict:
    """Record which fields the checkpoint's own config.json actually states.

    A ``PretrainedConfig`` attribute is present whether or not the checkpoint
    declared it, because the config class fills in its own defaults. Those two
    cases produce the same attribute and the same number, and only one of them
    is a property of this checkpoint. Reading the raw file is the only way to
    tell them apart, and the distinction matters because a class default can
    move under a library upgrade while the checkpoint sits still.
    """
    try:
        from transformers.utils import cached_file

        path = cached_file(model_id, "config.json")
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"ran": False, "reason": f"{type(exc).__name__}: {exc}"}

    return {
        "ran": True,
        "source": str(path),
        "present": {k: (k in raw) for k in PROVENANCE_KEYS},
        "raw_values": {k: raw[k] for k in PROVENANCE_KEYS if k in raw},
        "transformers_version_in_file": raw.get("transformers_version"),
    }


def probe_tokenizers(sharer_tok, receiver_tok) -> dict:
    per_model = {}
    for name, tok in ((SHARER_NAME, sharer_tok), (RECEIVER_NAME, receiver_tok)):
        per_model[name] = {
            "class": type(tok).__name__,
            "vocab_size": int(tok.vocab_size),
            "len_tokenizer": len(tok),
            "bos_token_id": tok.bos_token_id,
            "eos_token_id": tok.eos_token_id,
            "pad_token_id": tok.pad_token_id,
            "probe_ids": {
                s: tok(s, add_special_tokens=False)["input_ids"]
                for s in PROBE_STRINGS
            },
        }

    identical = all(
        per_model[SHARER_NAME]["probe_ids"][s]
        == per_model[RECEIVER_NAME]["probe_ids"][s]
        for s in PROBE_STRINGS
    )
    return {
        "probe_strings": PROBE_STRINGS,
        "per_model": per_model,
        "probe_ids_identical": identical,
        "vocab_size_identical": (
            per_model[SHARER_NAME]["vocab_size"]
            == per_model[RECEIVER_NAME]["vocab_size"]
        ),
        "align_tokens_in_scope": not identical,
    }


# --------------------------------------------------------------------------
# README rendering


def render_table(contract: dict) -> str:
    s, r = contract["sharer"], contract["receiver"]
    rows = [
        ("model", SHARER_ID, RECEIVER_ID),
        ("role", "Sharer", "Receiver"),
        ("n_layers", s["n_layers"], r["n_layers"]),
        ("hidden_size", s["hidden_size"], r["hidden_size"]),
        ("n_q_heads", s["n_q_heads"], r["n_q_heads"]),
        ("n_kv_heads", s["n_kv_heads"], r["n_kv_heads"]),
        ("gqa_group_size", s["gqa_group_size"], r["gqa_group_size"]),
        ("head_dim", s["head_dim"], r["head_dim"]),
        (
            "hidden_size / n_q_heads",
            s["head_dim_from_hidden_size"],
            r["head_dim_from_hidden_size"],
        ),
        (
            "head_dim decoupled",
            s["head_dim_is_decoupled"],
            r["head_dim_is_decoupled"],
        ),
        ("kv_width", s["kv_width"], r["kv_width"]),
        ("rope_theta", s["rope_theta"], r["rope_theta"]),
        ("rope_theta source", s["rope_theta_source"], r["rope_theta_source"]),
        (
            "deprecated rope_theta attr",
            s["rope_theta_legacy_attr"],
            r["rope_theta_legacy_attr"],
        ),
        ("len(inv_freq)", s["n_inv_freq"], r["n_inv_freq"]),
        ("q_norm", s["has_q_norm"], r["has_q_norm"]),
        ("k_norm", s["has_k_norm"], r["has_k_norm"]),
        ("vocab_size", s["vocab_size"], r["vocab_size"]),
    ]

    lines = ["| field | sharer | receiver |", "| --- | --- | --- |"]
    lines += [f"| {a} | {b} | {c} |" for a, b, c in rows]

    cw = contract["concat_widths"]
    rope = contract["rope_comparison"]
    nest = rope["nesting"]
    tok = contract["tokenizers"]
    mapping = contract["layer_alignment"]

    lines += [
        "",
        f"Concatenated key width entering the projection: "
        f"{cw['sharer_kv_width']} + {cw['receiver_kv_width']} = "
        f"{cw['concat_width']}. The Sharer contributes "
        f"{cw['sharer_share']:.1%} of the channels.",
        "",
        f"Rotary ladders: theta match {rope['theta_match']}, length match "
        f"{rope['length_match']}. Nesting at stride {nest['stride']}: "
        f"{nest['holds']}.",
        "",
        f"Layer alignment (terminal): {mapping['n_pairs']} paired target "
        f"layers, target {mapping['unpaired_targets']} unpaired.",
        "",
        f"Token alignment: probe ids identical {tok['probe_ids_identical']}, "
        f"therefore align_tokens is "
        f"{'in scope' if tok['align_tokens_in_scope'] else 'out of scope'}.",
    ]
    return "\n".join(lines)


def write_readme_block(table: str) -> None:
    block = f"{BLOCK_START}\n{table}\n{BLOCK_END}"
    if README_PATH.exists():
        text = README_PATH.read_text(encoding="utf-8")
    else:
        text = "# c2c-lite\n\n## Config contract\n\n"

    if BLOCK_START in text and BLOCK_END in text:
        head, rest = text.split(BLOCK_START, 1)
        _, tail = rest.split(BLOCK_END, 1)
        text = head + block + tail
    else:
        if not text.endswith("\n"):
            text += "\n"
        text += "\n## Config contract\n\n" + block + "\n"

    README_PATH.write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------


def main() -> None:
    from transformers import AutoConfig, AutoTokenizer

    sharer_cfg = AutoConfig.from_pretrained(SHARER_ID)
    receiver_cfg = AutoConfig.from_pretrained(RECEIVER_ID)

    sharer_q, sharer_k, sharer_method = probe_qk_norm(sharer_cfg)
    receiver_q, receiver_k, receiver_method = probe_qk_norm(receiver_cfg)

    contract = build_contracts(
        sharer_cfg,
        receiver_cfg,
        sharer_name=SHARER_NAME,
        receiver_name=RECEIVER_NAME,
        sharer_qk_norm=(sharer_q, sharer_k),
        receiver_qk_norm=(receiver_q, receiver_k),
    )

    contract["sharer"]["model_id"] = SHARER_ID
    contract["receiver"]["model_id"] = RECEIVER_ID
    contract["sharer"]["qk_norm_probe_method"] = sharer_method
    contract["receiver"]["qk_norm_probe_method"] = receiver_method

    for role, cfg in ((SHARER_NAME, sharer_cfg), (RECEIVER_NAME, receiver_cfg)):
        m = contract[role]
        m["inv_freq_crosscheck"] = crosscheck_inv_freq(
            cfg,
            rope_inv_freq(
                m["head_dim"], m["rope_theta"], m["partial_rotary_factor"]
            ),
            m["rope_type"],
        )

    mapping = align_layers(
        contract["sharer"]["n_layers"],
        contract["receiver"]["n_layers"],
        strategy="terminal",
    )
    contract["layer_alignment"] = {
        "strategy": "terminal",
        "target_to_source": mapping,
        "pairs": aligned_pairs(mapping),
        "n_pairs": sum(1 for s in mapping if s is not None),
        "unpaired_targets": [t for t, s in enumerate(mapping) if s is None],
    }

    contract["tokenizers"] = probe_tokenizers(
        AutoTokenizer.from_pretrained(SHARER_ID),
        AutoTokenizer.from_pretrained(RECEIVER_ID),
    )

    import transformers

    contract["config_provenance"] = {
        "transformers_version": transformers.__version__,
        SHARER_NAME: config_provenance(SHARER_ID),
        RECEIVER_NAME: config_provenance(RECEIVER_ID),
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(contract, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    write_readme_block(render_table(contract))


if __name__ == "__main__":
    main()