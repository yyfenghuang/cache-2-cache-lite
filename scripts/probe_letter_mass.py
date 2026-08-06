"""Where the loss went. A probe, not a gate.

Writes results/letter_mass_<run name>.json.

The fused run's held-out loss fell 0.518 below the Receiver's own while its
four-way accuracy moved by one question in sixty-four. Two readings fit that
and the log cannot separate them. Either the model became better at choosing
between the four letters, or it became more certain that the answer is a
letter at all without changing which letter wins.

The separation is not a heuristic. It is an identity, and it lives in
`c2c/prompting.py` beside the prompt whose answer it is reading, because the
training loop now records the same decomposition and two copies of it would
drift.

What this can and cannot settle
-------------------------------
It settles which term the training pressure went into, on this corpus, for
this checkpoint. It says nothing about MMLU-Redux, and nothing about whether
adding beats substituting, because the second arm has not been trained.

A note on the samples
---------------------
The comparison only means something if these are the same sixty-four
questions the training run held out. The selection is not restated here; it
is imported from `train_fuser`, along with every constant that shapes it, so
that changing one there cannot leave this behind.

That the reproduction actually worked is then checked, not assumed, and the
check is the Receiver's own loss on the sixty-four. It has to equal the number
the training log recorded at step zero. A walk that landed anywhere else would
have to score identically on a different set of questions to slip past. The
count of rows skipped along the way is recorded too, but it is not the check:
this script stops after sixty-four questions and the training run walked on to
two thousand more, so the two counts are not comparable and were never
supposed to be.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from c2c.fuser import FuserBank  # noqa: E402
from c2c.prompting import (  # noqa: E402
    IDENTITY_TOLERANCE, OPTIONS, decompose_loss, option_token_ids,
)

import train_fuser as T  # noqa: E402

# Which run to probe, named rather than guessed. Globbing for the newest
# checkpoint would make the answer depend on the state of a directory, and
# this file's whole job is attribution.
RUN_NAME = "fused_2026-08-06_n2000"

RESULTS_DIR = REPO_ROOT / "results"
LOG_PATH = RESULTS_DIR / f"fuser_{RUN_NAME}.json"
RESULTS_PATH = RESULTS_DIR / f"letter_mass_{RUN_NAME}.json"
CONTRACTS_PATH = RESULTS_DIR / "contracts.json"

@torch.no_grad()
def measure(receiver, sharer, bank, samples, letters) -> list[dict]:
    """One record per question per condition, paired by index.

    Per-sample records rather than an aggregate, because an aggregate cannot
    tell a model that improved everywhere from a model that improved on a few
    questions and collapsed on the rest, and this repository has already been
    caught by that once.
    """
    if bank is not None:
        bank.eval()
    records = []
    for index, sample in enumerate(samples):
        receiver_pairs = T._prefill(receiver, sample["prefix"])
        if bank is None:
            cache_pairs = receiver_pairs
        else:
            cache_pairs = bank(
                receiver_pairs, T._prefill(sharer, sample["prefix"])
            )
        logits = T._score(receiver, sample, cache_pairs)
        record = decompose_loss(logits, letters, sample["answer"])
        record.update(
            {
                "index": index,
                "answer": sample["answer"],
                "n_prompt_tokens": sample["n_prompt_tokens"],
            }
        )
        records.append(record)
    return records


def summarise(records: list[dict]) -> dict:
    n = len(records)
    return {
        "n": n,
        "full": sum(r["full"] for r in records) / n,
        "four_way": sum(r["four_way"] for r in records) / n,
        "letter_mass": sum(r["letter_mass"] for r in records) / n,
        "mass_on_letters": sum(r["mass_on_letters"] for r in records) / n,
        "accuracy": sum(r["correct"] for r in records) / n,
    }


def main() -> None:
    for path in (LOG_PATH, CONTRACTS_PATH):
        if not path.exists():
            raise SystemExit(
                f"missing {path.relative_to(REPO_ROOT)}; there is no run "
                f"named {RUN_NAME!r} to probe"
            )

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log = json.loads(LOG_PATH.read_text(encoding="utf-8"))
    contract = json.loads(CONTRACTS_PATH.read_text(encoding="utf-8"))

    # The weights this log describes, taken from the log rather than assembled
    # from the same parts a second time.
    mode = log["mode"]
    checkpoint_path = RESULTS_DIR / log["checkpoint"]
    if not checkpoint_path.exists():
        raise SystemExit(
            f"the log names {log['checkpoint']} as its checkpoint and it is "
            "not in results/"
        )

    # The held-out questions are rebuilt from train_fuser's live constants, so
    # a constant that has moved since the run would rebuild a different set.
    # The Receiver's own loss catches that afterwards; these catch it now.
    for name, recorded in (
        ("SEED", log["seed"]),
        ("N_VALIDATION", log["n_validation"]),
        ("MAX_PROMPT_TOKENS", log["max_prompt_tokens"]),
    ):
        current = getattr(T, name)
        if current != recorded:
            raise SystemExit(
                f"train_fuser.{name} is {current} now and was {recorded} "
                f"during {RUN_NAME}. The held-out set would not be the same."
            )
    receiver_contract, sharer_contract = contract["receiver"], contract["sharer"]

    tokenizer = AutoTokenizer.from_pretrained(T.RECEIVER_ID)
    letters = option_token_ids(tokenizer)
    if letters != log["option_token_ids"]:
        raise SystemExit(
            f"the option letters tokenize to {letters} now and to "
            f"{log['option_token_ids']} during training"
        )

    receiver = (
        AutoModelForCausalLM.from_pretrained(
            T.RECEIVER_ID, dtype=T.DTYPE, attn_implementation=T.ATTN_IMPLEMENTATION
        )
        .to(T.DEVICE)
        .eval()
    )
    sharer = (
        AutoModelForCausalLM.from_pretrained(
            T.SHARER_ID, dtype=T.DTYPE, attn_implementation=T.ATTN_IMPLEMENTATION
        )
        .to(T.DEVICE)
        .eval()
    )

    rows = load_dataset(
        T.DATASET, T.DATASET_CONFIG, split=T.DATASET_SPLIT
    ).shuffle(seed=T.SEED)

    samples, skipped, cursor = [], 0, 0
    while len(samples) < T.N_VALIDATION and cursor < len(rows):
        prepared = T.prepare(rows[cursor], tokenizer, letters)
        cursor += 1
        if prepared is None:
            skipped += 1
            continue
        samples.append(prepared)
    if len(samples) < T.N_VALIDATION:
        raise SystemExit("ran out of rows before filling the held-out set")

    for record in log["history"]:
        if record["step"] == 0:
            recorded_baseline = record["validation_loss"]
            break

    bank = FuserBank(
        receiver_contract["n_layers"],
        sharer_contract["n_layers"],
        receiver_contract["n_kv_heads"],
        receiver_contract["head_dim"],
        sharer_contract["n_kv_heads"],
        sharer_contract["head_dim"],
        residual=(mode == "fused"),
    )
    bank.load_state_dict(torch.load(checkpoint_path, weights_only=True))

    baseline_records = measure(receiver, sharer, None, samples, letters)
    fused_records = measure(receiver, sharer, bank, samples, letters)
    baseline, fused = summarise(baseline_records), summarise(fused_records)

    # The samples reproduced only if the walk landed in the same place and the
    # Receiver alone scores what it scored then. Both are checked, because the
    # first can agree by luck on a short walk and the second cannot.
    if abs(baseline["full"] - recorded_baseline) > IDENTITY_TOLERANCE:
        raise SystemExit(
            f"the Receiver alone scores {baseline['full']} on these questions "
            f"and scored {recorded_baseline} during training. These are not "
            "the same sixty-four questions, so nothing below would be paired."
        )

    delta = {
        key: fused[key] - baseline[key]
        for key in ("full", "four_way", "letter_mass", "mass_on_letters", "accuracy")
    }
    residue = delta["full"] - (delta["four_way"] + delta["letter_mass"])
    if abs(residue) > IDENTITY_TOLERANCE:
        raise SystemExit(f"the change does not decompose; residue {residue}")

    flipped = [
        {
            "index": b["index"],
            "answer": b["answer"],
            "baseline": OPTIONS[b["chosen"]],
            "fused": OPTIONS[f["chosen"]],
        }
        for b, f in zip(baseline_records, fused_records)
        if b["chosen"] != f["chosen"]
    ]

    result = {
        "run_name": RUN_NAME,
        "mode": mode,
        "checkpoint": log["checkpoint"],
        "baseline": baseline,
        "fused": fused,
        "delta": delta,
        "share_of_delta": {
            "four_way": delta["four_way"] / delta["full"] if delta["full"] else None,
            "letter_mass": (
                delta["letter_mass"] / delta["full"] if delta["full"] else None
            ),
        },
        "decomposition_residue": residue,
        "n_flipped": len(flipped),
        "flipped": flipped,
        "n_skipped": skipped,
        "n_skipped_during_training": log["n_skipped"],
        "recorded_baseline_loss": recorded_baseline,
        "gate_activation_ratio": log["history"][-1]["gate_activation_ratio"],
        "total_steps": log["total_steps"],
        "records": {"baseline": baseline_records, "fused": fused_records},
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()