# SPDX-License-Identifier: Apache-2.0
"""Does the projection carry to the distribution it was graded on?

A probe, not a gate. It records a measurement and is adjudicated by being
read.

The question
------------
The projections were fitted on wikitext and their quality was measured on
held out wikitext. The accuracy run scored them on MMLU prompts, which are
questions followed by four labelled options rather than flowing prose. No
measurement so far touches the projection's quality on the distribution where
its behaviour was judged.

Two accounts produce the same collapse to a single option letter, and nothing
measured so far separates them. Either the projection works as measured and
replacing the whole cache destroys the Receiver's ability to condition on the
question, which is a statement about the mechanism. Or the projection does
not carry to MMLU at all, which is a statement about the corpus this build
chose.

The measurement
---------------
Relative loss per layer on MMLU prompts, in exactly the units the training
reported: mean squared error divided by the mean per channel variance of the
target. The null is recomputed on MMLU, because dividing an MMLU loss by a
wikitext null would compare two different things and call the ratio a
quality.

Three predictors, so the answer decomposes
------------------------------------------
The trained projection is one. The second is the per channel mean of the
Receiver's held out wikitext cache, which is what a projection that learned
only the training distribution's centre would emit; if that alone lands above
one on MMLU, the bias is miscalibrated before any weight is consulted. The
third is the MMLU mean itself, which is the null and is one by construction,
and is computed anyway because a number that must be one is a check on the
arithmetic that produced the other two.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from c2c.alignment import align_layers  # noqa: E402
from c2c.cache_ops import cache_tensors, flatten_heads  # noqa: E402

CONTRACTS_PATH = REPO_ROOT / "results" / "contracts.json"
TRAIN_LOG_PATH = REPO_ROOT / "results" / "train_log.json"
PROJECTIONS_PATH = REPO_ROOT / "results" / "projections.pt"
CACHE_ROOT = REPO_ROOT / "results" / "caches"
RESULTS_PATH = REPO_ROOT / "results" / "distribution_shift.json"

# The same prompts the accuracy run scored, so this measures the projection
# on the text where the collapse was seen rather than on a fresh draw.
N_PROMPTS = 50

DTYPE = torch.float32
DEVICE = "cpu"
ATTN_IMPLEMENTATION = "eager"
KINDS = ("keys", "values")


def load_json(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(
            f"missing {path.relative_to(REPO_ROOT)}; the gate below it has "
            "not been closed"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def analysis_module():
    path = REPO_ROOT / "scripts" / "analysis.py"
    spec = importlib.util.spec_from_file_location("_analysis", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_analysis"] = module
    spec.loader.exec_module(module)
    return module


class Accumulator:
    """Squared error against three predictors, and the target's own spread.

    Streaming, in float64. Nothing is held across prompts except a handful of
    vectors of width kv_width, so the prompt budget is free to grow.
    """

    def __init__(self, width: int, wikitext_mean: torch.Tensor):
        self.width = width
        self.wikitext_mean = wikitext_mean.to(torch.float64)
        self.count = 0
        self.total = torch.zeros(width, dtype=torch.float64)
        self.total_sq = torch.zeros(width, dtype=torch.float64)
        self.error_projected = 0.0
        self.error_wikitext_mean = 0.0

    def add(self, target: torch.Tensor, projected: torch.Tensor) -> None:
        wide = target.to(torch.float64)
        self.count += wide.shape[0]
        self.total += wide.sum(dim=0)
        self.total_sq += (wide * wide).sum(dim=0)
        self.error_projected += float(
            ((projected.to(torch.float64) - wide) ** 2).sum()
        )
        self.error_wikitext_mean += float(((self.wikitext_mean - wide) ** 2).sum())

    def report(self) -> dict:
        n, c = self.count, self.width
        mean = self.total / n
        null = float((self.total_sq / n - mean * mean).clamp_min(0.0).mean())
        elements = n * c
        return {
            "n_positions": n,
            "n_channels": c,
            "null_mmlu": null,
            "mse_projected": self.error_projected / elements,
            "mse_wikitext_mean": self.error_wikitext_mean / elements,
            "relative_projected": self.error_projected / elements / null,
            "relative_wikitext_mean": self.error_wikitext_mean / elements / null,
            "relative_mmlu_mean": 1.0,
            "wikitext_mean_offset": float(
                ((mean - self.wikitext_mean) ** 2).mean() / null
            ),
        }


@torch.no_grad()
def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    contract = load_json(CONTRACTS_PATH)
    train_log = load_json(TRAIN_LOG_PATH)
    if not PROJECTIONS_PATH.exists():
        raise SystemExit(
            f"missing {PROJECTIONS_PATH.relative_to(REPO_ROOT)}; run "
            "scripts/train_projection.py first"
        )

    analysis = analysis_module()
    from datasets import get_dataset_config_names, load_dataset

    subjects = sorted(get_dataset_config_names(analysis.DATASET))
    pool = []
    for subject in subjects:
        for row in load_dataset(analysis.DATASET, subject, split=analysis.DATASET_SPLIT):
            if row.get("error_type", analysis.KEEP_ERROR_TYPE) != analysis.KEEP_ERROR_TYPE:
                continue
            pool.append({
                "subject": subject, "question": row["question"],
                "choices": list(row["choices"]), "answer": row["answer"],
            })
    order = torch.randperm(
        len(pool), generator=torch.Generator().manual_seed(analysis.SAMPLE_SEED)
    )[:N_PROMPTS].tolist()
    rows = [pool[i] for i in order]

    tokenizer = AutoTokenizer.from_pretrained(analysis.RECEIVER_ID)
    prompts = [
        tokenizer(
            analysis.build_prompt(r["subject"], r["question"], r["choices"]),
            add_special_tokens=False,
        )["input_ids"]
        for r in rows
    ]

    mapping = align_layers(
        contract["sharer"]["n_layers"], contract["receiver"]["n_layers"],
        strategy="terminal",
    )
    projections = analysis.load_projections(contract)

    width = contract["receiver"]["kv_width"]
    accumulators = {}
    for target, source in enumerate(mapping):
        if source is None:
            continue
        for kind in KINDS:
            path = CACHE_ROOT / "held_out" / f"receiver_{kind}_layer{target:02d}.pt"
            if not path.exists():
                raise SystemExit(
                    f"missing {path.relative_to(REPO_ROOT)}; the wikitext mean "
                    "cannot be recovered without it"
                )
            stored = torch.load(path, map_location="cpu").to(torch.float64)
            accumulators[(target, kind)] = Accumulator(width, stored.mean(dim=0))
            del stored

    sharer = AutoModelForCausalLM.from_pretrained(
        analysis.SHARER_ID, dtype=DTYPE, attn_implementation=ATTN_IMPLEMENTATION
    ).to(DEVICE).eval()
    receiver = AutoModelForCausalLM.from_pretrained(
        analysis.RECEIVER_ID, dtype=DTYPE, attn_implementation=ATTN_IMPLEMENTATION
    ).to(DEVICE).eval()

    for ids in prompts:
        tensor = torch.tensor([ids], dtype=torch.long, device=DEVICE)
        sharer_cache = cache_tensors(sharer(tensor, use_cache=True).past_key_values)
        receiver_cache = cache_tensors(receiver(tensor, use_cache=True).past_key_values)
        for target, source in enumerate(mapping):
            if source is None:
                continue
            for slot, kind in enumerate(KINDS):
                source_flat = flatten_heads(sharer_cache[source][slot])
                target_flat = flatten_heads(receiver_cache[target][slot])
                predicted = projections[(target, kind)].net(source_flat)
                accumulators[(target, kind)].add(target_flat, predicted)
        del sharer_cache, receiver_cache

    wikitext = {}
    for record in train_log["per_layer"]:
        wikitext[(record["target_layer"], record["kind"])] = record["relative_held_out"]

    per_layer = []
    for (target, kind), accumulator in sorted(accumulators.items()):
        report = accumulator.report()
        report.update(
            target_layer=target, kind=kind,
            relative_on_wikitext=wikitext.get((target, kind)),
        )
        report["mmlu_over_wikitext"] = (
            report["relative_projected"] / report["relative_on_wikitext"]
            if report["relative_on_wikitext"] else None
        )
        per_layer.append(report)

    summary = {}
    for kind in KINDS:
        subset = [r for r in per_layer if r["kind"] == kind]
        summary[kind] = {
            "n_layers": len(subset),
            "mean_relative_on_mmlu": sum(r["relative_projected"] for r in subset) / len(subset),
            "mean_relative_on_wikitext": sum(r["relative_on_wikitext"] for r in subset) / len(subset),
            "mean_relative_wikitext_mean_predictor": sum(
                r["relative_wikitext_mean"] for r in subset
            ) / len(subset),
            "layers_at_or_above_null_on_mmlu": [
                r["target_layer"] for r in subset if r["relative_projected"] >= 1.0
            ],
        }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps({
        "question": "does a projection fitted on wikitext carry to the MMLU "
                    "prompts the accuracy run scored it on",
        "summary": summary,
        "per_layer": per_layer,
        "method": {
            "n_prompts": N_PROMPTS,
            "prompts": "the first N of the accuracy run's sample, same seed",
            "null": "recomputed on MMLU; dividing an MMLU loss by a wikitext "
                    "null would compare two different things",
            "predictors": ["trained projection", "wikitext held out per channel "
                           "mean", "MMLU per channel mean, one by construction"],
        },
        "torch_version": torch.__version__,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()