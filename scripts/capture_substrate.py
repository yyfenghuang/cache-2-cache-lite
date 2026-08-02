"""Five conditions on one model, no gradients.

Silent on success. Everything goes into results/substrate.json.

The measurement lives in `measure_substrate`, which takes a model rather than
loading one. That split is deliberate: it lets the whole measurement be
exercised against a small model built from a config in `tests/`, in about a
second, with no checkpoint on disk. What remains untested until this script
runs for real is one line that loads weights.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from c2c.cache_ops import (  # noqa: E402
    absolute_position_ids,
    build_cache,
    cache_tensors,
    clone_cache,
    corrupt_norm_matched,
    max_abs_difference,
    relative_position_ids,
    slice_cache,
)

RECEIVER_ID = "Qwen/Qwen3-0.6B"

PROMPT = (
    "A key-value cache stores, for every layer and every position, the key "
    "and value vectors that attention has already computed. Keys leave the "
    "projection with rotary position information folded in. Values leave it "
    "with none. When a cache is handed to a model that did not produce it, "
    "the positions its keys were rotated at do not travel with the tensor."
)

DTYPE = torch.float32
DEVICE = "cpu"
ATTN_IMPLEMENTATION = "eager"
CORRUPTION_SEED = 1234

RESULTS_PATH = REPO_ROOT / "results" / "substrate.json"


def float32_ulp_at(scale: float) -> float:
    """The size of one float32 step at a given magnitude.

    Conditions that compare two differently shaped matmuls disagree by a few
    of these, and the count depends on how the reduction was split across
    threads. Reporting the raw difference alone invites a threshold chosen by
    how big the number happened to look.
    """
    if not scale > 0 or not math.isfinite(scale):
        return float("nan")
    return 2.0 ** (math.floor(math.log2(scale)) - 23)


@torch.no_grad()
def _forward(model, input_ids, cache=None, position_ids=None):
    kwargs = {}
    if cache is not None:
        kwargs["past_key_values"] = cache
    if position_ids is not None:
        kwargs["position_ids"] = position_ids
    return model(input_ids, use_cache=True, **kwargs)


@torch.no_grad()
def measure_substrate(model, input_ids, n_prefix: int, seed: int) -> dict:
    """Run every condition and report one number each.

    Every condition receives its own cache. A forward pass appends to the
    cache it is handed, so a cache reused between two conditions has grown by
    the time the second one reads it, and the comparison stops meaning
    anything without raising.
    """
    total = int(input_ids.shape[1])
    if not 1 <= n_prefix < total:
        raise ValueError(f"n_prefix {n_prefix} does not split a length {total} sequence")
    n_new = total - n_prefix
    prefix, suffix = input_ids[:, :n_prefix], input_ids[:, n_prefix:]
    device = input_ids.device

    absolute = absolute_position_ids(n_prefix, n_new, device=device)
    relative = relative_position_ids(n_new, device=device)

    full = _forward(model, input_ids)
    logits_full = full.logits[:, n_prefix:, :]
    cache_full = full.past_key_values

    cache_prefix = _forward(model, prefix).past_key_values
    logits_reference = _forward(
        model, suffix, clone_cache(cache_prefix), absolute
    ).logits

    generator = torch.Generator(device=device).manual_seed(seed)

    conditions = {
        "baseline_vs_itself": max_abs_difference(
            _forward(model, suffix, clone_cache(cache_prefix), absolute).logits,
            logits_reference,
        ),
        "noop_injection": max_abs_difference(
            _forward(
                model, suffix,
                build_cache(cache_tensors(clone_cache(cache_prefix))),
                absolute,
            ).logits,
            logits_reference,
        ),
        "corrupted_injection": max_abs_difference(
            _forward(
                model, suffix,
                corrupt_norm_matched(clone_cache(cache_prefix), generator),
                absolute,
            ).logits,
            logits_reference,
        ),
        "position_absolute": max_abs_difference(
            _forward(model, suffix, slice_cache(cache_full, 0, n_prefix), absolute).logits,
            logits_full,
        ),
        "position_relative": max_abs_difference(
            _forward(model, suffix, slice_cache(cache_full, 0, n_prefix), relative).logits,
            logits_full,
        ),
    }

    logit_scale = float(logits_full.abs().max())
    ulp = float32_ulp_at(logit_scale)
    keys0, values0 = cache_tensors(cache_prefix)[0]
    corrupted0 = cache_tensors(
        corrupt_norm_matched(
            clone_cache(cache_prefix),
            torch.Generator(device=device).manual_seed(seed),
        )
    )[0]

    return {
        "conditions": conditions,
        "ratios": {k: v / logit_scale for k, v in conditions.items()},
        "ulps": {k: v / ulp for k, v in conditions.items()},
        "logit_scale": logit_scale,
        "logit_ulp": ulp,
        "torch_num_threads": torch.get_num_threads(),
        "sequence": {
            "n_total": total,
            "n_prefix": n_prefix,
            "n_new": n_new,
        },
        "corruption_check": {
            "max_norm_deviation": float(
                (keys0.norm(dim=-1) - corrupted0[0].norm(dim=-1)).abs().max()
            ),
            "max_abs_cosine": float(
                torch.nn.functional.cosine_similarity(
                    keys0, corrupted0[0], dim=-1
                ).abs().max()
            ),
        },
        "cache_shape": list(cache_tensors(cache_prefix)[0][0].shape),
        "n_layers": len(cache_tensors(cache_prefix)),
        "seed": seed,
    }


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(RECEIVER_ID)
    model = AutoModelForCausalLM.from_pretrained(
        RECEIVER_ID, dtype=DTYPE, attn_implementation=ATTN_IMPLEMENTATION
    ).to(DEVICE).eval()

    input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"].to(DEVICE)
    n_prefix = int(input_ids.shape[1]) // 2

    result = measure_substrate(model, input_ids, n_prefix, CORRUPTION_SEED)
    result["model_id"] = RECEIVER_ID
    result["dtype"] = str(DTYPE)
    result["device"] = DEVICE
    result["attn_implementation"] = ATTN_IMPLEMENTATION

    import transformers

    result["transformers_version"] = transformers.__version__
    result["torch_version"] = torch.__version__

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()