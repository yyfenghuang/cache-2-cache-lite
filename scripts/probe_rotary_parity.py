# SPDX-License-Identifier: Apache-2.0
"""Does the Receiver's key cache split along the Sharer's rotary ladder?

A probe, not a gate. It records a measurement and adjudicates a prediction by
being read, not by an exit code, because the prediction is a hypothesis that
may well be false and false is a legitimate outcome here.

The question
------------
Tier 0 measured that the two rotary ladders nest at stride two, to zero
relative deviation: Sharer pair i turns at exactly the frequency of Receiver
pair 2i. So half the Receiver's rotary pairs, the even indexed ones, turn at
frequencies that are physically present in the Sharer's key. The odd indexed
half turn at frequencies the Sharer never uses.

If the key projection's advantage over the value projection comes from
position, the even half should be easier to predict than the odd half:
copying a phase that is already in the input is a linear operation, and
synthesising a phase that is not is not. If the advantage comes from k_norm
bounding the Receiver's keys, or from how variance is spread across channels,
parity is invisible to both of those and there should be no split.

Why parity is a fair partition
------------------------------
The even and odd pairs interleave along the frequency ladder, so the two
halves hold almost the same distribution of rotation speeds. What differs
between them is not how fast they turn but whether that exact speed exists on
the other side.

Two controls
------------
Values are never rotated. The same parity arithmetic applied to the value
cache partitions nothing, so a gap there would mean parity marks something
other than rotation.

The permutation control is the stronger one. Parity is one partition of the
64 pairs into two halves; a thousand random partitions give the distribution
a gap of no particular meaning would have. Permuting whole pairs rather than
channels keeps the within pair correlation structure identical to parity's,
so the comparison is like for like.

Why least squares and not the trained MLP
-----------------------------------------
Closed form, no seed, no epoch, no early stopping, and nothing to tune. The
numbers will not match results/train_log.json and are not meant to. What is
under test is whether a gap exists, not how large the residual is.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from c2c.alignment import align_layers  # noqa: E402

CONTRACTS_PATH = REPO_ROOT / "results" / "contracts.json"
NULL_PATH = REPO_ROOT / "results" / "geometric_null.json"
CACHE_ROOT = REPO_ROOT / "results" / "caches"
RESULTS_PATH = REPO_ROOT / "results" / "rotary_parity.json"

FIT_SPLIT = "train"
EVAL_SPLIT = "held_out"
KINDS = ("keys", "values")

N_PERMUTATIONS = 1000
PERMUTATION_SEED = 20260802

# Fixed before the run. Adjudicated by reading results/rotary_parity.json.
#
#   1. For keys, the odd indexed rotary pairs are harder than the even ones,
#      so the gap is positive, in at least 18 of the 24 paired layers. Under
#      no parity effect each layer is a coin flip, and 18 or more of 24 has
#      probability 0.0113.
#   2. For keys, the parity gap sits above the 97.5th percentile of the
#      permutation gaps in the majority of layers.
#   3. For values, neither holds.
#
# If 1 and 2 hold and 3 does not, parity marks something that is not
# rotation and the hypothesis is dead rather than supported.
PREDICTION = {
    "min_layers_with_positive_key_gap": 18,
    "sign_test_p_value": 0.0113,
    "permutation_percentile": 97.5,
}


def load_json(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(
            f"missing {path.relative_to(REPO_ROOT)}; the gate below it has "
            "not been closed"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_layer(split: str, role: str, kind: str, layer: int) -> torch.Tensor:
    path = CACHE_ROOT / split / f"{role}_{kind}_layer{layer:02d}.pt"
    if not path.exists():
        raise SystemExit(
            f"missing {path.relative_to(REPO_ROOT)}; run "
            "scripts/capture_dual_cache.py first"
        )
    return torch.load(path, map_location="cpu").to(torch.float64)


def pair_index(n_heads: int, head_dim: int) -> torch.Tensor:
    """The rotary pair each flattened channel belongs to.

    Rotary turns dimension j together with j + head_dim / 2, and both read
    entry j mod (head_dim / 2) of the frequency ladder. Channels are laid out
    head major, so the pair index repeats identically under every head.
    """
    half = head_dim // 2
    within = torch.arange(head_dim) % half
    return within.repeat(n_heads)


def per_channel_relative_residual(x_fit, y_fit, x_eval, y_eval) -> torch.Tensor:
    """Least squares per output channel, read back on the evaluation split.

    Returns residual divided by that channel's own variance, so a channel
    contributes on its own scale rather than in proportion to how large it
    happens to be.
    """
    ones_fit = torch.ones(x_fit.shape[0], 1, dtype=x_fit.dtype)
    ones_eval = torch.ones(x_eval.shape[0], 1, dtype=x_eval.dtype)
    design = torch.cat([x_fit, ones_fit], dim=1)
    solution = torch.linalg.lstsq(design, y_fit).solution

    predicted = torch.cat([x_eval, ones_eval], dim=1) @ solution
    residual = ((predicted - y_eval) ** 2).mean(dim=0)
    variance = y_eval.var(dim=0, unbiased=False).clamp_min(1e-30)
    return residual / variance


def parity_gap(relative: torch.Tensor, pairs: torch.Tensor, odd_pairs) -> dict:
    """Mean relative residual on the odd side minus the even side.

    Both weightings are reported. Equal weight per channel answers "is a
    typical odd channel harder"; weighting by variance answers "is more of
    the total left unexplained on the odd side". They can disagree, and a
    disagreement is a fact about the tensor rather than a choice to hide.
    """
    odd_mask = torch.isin(pairs, odd_pairs)
    odd, even = relative[odd_mask], relative[~odd_mask]
    return {
        "gap_equal_weight": float(odd.mean() - even.mean()),
        "odd_mean": float(odd.mean()),
        "even_mean": float(even.mean()),
        "n_odd_channels": int(odd_mask.sum()),
        "n_even_channels": int((~odd_mask).sum()),
    }


def permutation_null(relative, pairs, n_pairs, generator) -> dict:
    """The distribution of the gap under partitions of no particular meaning.

    Whole pairs move, never single channels, so every draw has the same
    within pair structure the parity partition has.
    """
    gaps = []
    for _ in range(N_PERMUTATIONS):
        order = torch.randperm(n_pairs, generator=generator)
        chosen = order[: n_pairs // 2]
        mask = torch.isin(pairs, chosen)
        gaps.append(float(relative[mask].mean() - relative[~mask].mean()))
    ordered = sorted(gaps)
    return {
        "n_permutations": N_PERMUTATIONS,
        "mean": sum(gaps) / len(gaps),
        "percentile_97_5": ordered[int(0.975 * (len(ordered) - 1))],
        "percentile_2_5": ordered[int(0.025 * (len(ordered) - 1))],
        "max_abs": max(abs(g) for g in gaps),
        "_sorted": ordered,
    }


def main() -> None:
    contract = load_json(CONTRACTS_PATH)
    load_json(NULL_PATH)

    n_heads = contract["receiver"]["n_kv_heads"]
    head_dim = contract["receiver"]["head_dim"]
    n_pairs = head_dim // 2
    pairs = pair_index(n_heads, head_dim)
    odd_pairs = torch.arange(1, n_pairs, 2)

    mapping = align_layers(
        contract["sharer"]["n_layers"], contract["receiver"]["n_layers"],
        strategy="terminal",
    )
    generator = torch.Generator().manual_seed(PERMUTATION_SEED)

    records = []
    for target, source in enumerate(mapping):
        if source is None:
            continue
        for kind in KINDS:
            relative = per_channel_relative_residual(
                load_layer(FIT_SPLIT, "sharer", kind, source),
                load_layer(FIT_SPLIT, "receiver", kind, target),
                load_layer(EVAL_SPLIT, "sharer", kind, source),
                load_layer(EVAL_SPLIT, "receiver", kind, target),
            )
            record = parity_gap(relative, pairs, odd_pairs)
            null = permutation_null(relative, pairs, n_pairs, generator)
            gap = record["gap_equal_weight"]
            record.update(
                target_layer=target, source_layer=source, kind=kind,
                overall_relative=float(relative.mean()),
                permutation_mean=null["mean"],
                permutation_percentile_97_5=null["percentile_97_5"],
                permutation_percentile_2_5=null["percentile_2_5"],
                permutation_max_abs=null["max_abs"],
                above_permutation_97_5=bool(gap > null["percentile_97_5"]),
                permutation_rank=sum(1 for g in null["_sorted"] if g < gap)
                / null["n_permutations"],
            )
            records.append(record)

    verdict = {}
    for kind in KINDS:
        subset = [r for r in records if r["kind"] == kind]
        positive = [r["target_layer"] for r in subset if r["gap_equal_weight"] > 0]
        above = [r["target_layer"] for r in subset if r["above_permutation_97_5"]]
        verdict[kind] = {
            "n_layers": len(subset),
            "layers_with_positive_gap": positive,
            "n_positive": len(positive),
            "layers_above_permutation_97_5": above,
            "n_above_permutation": len(above),
            "median_gap": sorted(r["gap_equal_weight"] for r in subset)[
                len(subset) // 2
            ],
            "meets_sign_criterion": len(positive)
            >= PREDICTION["min_layers_with_positive_key_gap"],
            "meets_permutation_criterion": len(above) > len(subset) / 2,
        }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps({
        "question": "are the Receiver's even indexed rotary pairs, whose "
                    "frequencies exist in the Sharer, easier to predict than "
                    "the odd indexed ones, whose frequencies do not",
        "prediction": PREDICTION,
        "verdict": verdict,
        "per_layer": records,
        "method": {
            "estimator": "ordinary least squares with intercept, per output channel",
            "fit_split": FIT_SPLIT,
            "eval_split": EVAL_SPLIT,
            "permutations": N_PERMUTATIONS,
            "permutation_unit": "rotary pair",
            "permutation_seed": PERMUTATION_SEED,
            "n_rotary_pairs": n_pairs,
        },
        "torch_version": torch.__version__,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()