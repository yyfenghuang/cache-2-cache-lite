"""Accuracy, once, with a confidence interval and a paired test.

Silent on success. Writes results/run_<date>_n<N>.json.

Two conditions on the same questions
------------------------------------
The Receiver answers with its own cache, and the Receiver answers with that
cache replaced by the projected Sharer cache. Every other thing is held
identical: same prompt, same tokens, same position ids, same scoring. The two
differ in exactly one respect, which is the only arrangement under which the
difference between them means anything.

Because both conditions answer the same questions, the samples are paired and
the statistics must be paired too. An unpaired interval on two accuracies
throws away the pairing and reports a wider interval than the design earns.

Scoring reads logits, not text
------------------------------
The answer is chosen by comparing the logits of the four option letters at
the position after "Answer:". No generation, no parsing, no regular
expression. A previous project in this line lost a result to an extractor
that mis-read correct outputs, and the cheapest defence against that is to
have nothing to extract. The script refuses to start if any option letter is
not a single token, because then the comparison would not be between four
comparable quantities.

Broken questions are removed before anything is measured
--------------------------------------------------------
MMLU-Redux exists because MMLU contains questions with no correct answer,
several correct answers, or a wrong ground truth, and it labels them. Leaving
them in adds noise that falls on both conditions and shrinks the very
difference being measured. Only rows annotated "ok" are used, and the counts
of what was dropped are recorded.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from c2c.alignment import align_layers  # noqa: E402
from c2c.cache_ops import (  # noqa: E402
    absolute_position_ids, build_cache, cache_tensors, clone_cache,
    flatten_heads, unflatten_heads,
)
from c2c.projection import CacheProjection  # noqa: E402
from c2c.prompting import (  # noqa: E402
    OPTIONS, build_prompt, option_token_ids,
)

CONTRACTS_PATH = REPO_ROOT / "results" / "contracts.json"
PROJECTIONS_PATH = REPO_ROOT / "results" / "projections.pt"
RESULTS_DIR = REPO_ROOT / "results"

SHARER_ID = "Qwen/Qwen2.5-0.5B-Instruct"
RECEIVER_ID = "Qwen/Qwen3-0.6B"

DATASET = "edinburgh-dawg/mmlu-redux-2.0"
DATASET_SPLIT = "test"
KEEP_ERROR_TYPE = "ok"

N_SAMPLES = 500
SAMPLE_SEED = 1234

BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20260802
INTERVAL = 0.95

DTYPE = torch.float32
DEVICE = "cpu"
ATTN_IMPLEMENTATION = "eager"


def load_json(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(
            f"missing {path.relative_to(REPO_ROOT)}; the gate below it has "
            "not been closed"
        )
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------- statistics


def paired_bootstrap(baseline, injected, generator) -> dict:
    """Resample questions, not conditions.

    The two conditions answered the same questions, so a resample draws a
    question and takes both of its outcomes. Drawing them independently would
    discard the pairing and widen the interval past what the design earns.
    """
    n = len(baseline)
    a = torch.tensor(baseline, dtype=torch.float64)
    b = torch.tensor(injected, dtype=torch.float64)
    draws = torch.randint(0, n, (BOOTSTRAP_RESAMPLES, n), generator=generator)
    differences = (b[draws].mean(dim=1) - a[draws].mean(dim=1)).sort().values
    low = int((1 - INTERVAL) / 2 * (BOOTSTRAP_RESAMPLES - 1))
    high = int((1 + INTERVAL) / 2 * (BOOTSTRAP_RESAMPLES - 1))
    return {
        "n_resamples": BOOTSTRAP_RESAMPLES,
        "interval": INTERVAL,
        "difference": float(b.mean() - a.mean()),
        "low": float(differences[low]),
        "high": float(differences[high]),
        "distribution": [float(x) for x in differences[::10]],
    }


def mcnemar(baseline, injected) -> dict:
    """Only the questions the two conditions disagree on carry information.

    Exact two sided binomial on the discordant pairs. The normal
    approximation is unreliable when the discordant count is small, and there
    is no reason to risk it for a test this cheap.
    """
    b = sum(1 for x, y in zip(baseline, injected) if x and not y)
    c = sum(1 for x, y in zip(baseline, injected) if y and not x)
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "n_discordant": 0, "p_value": 1.0,
                "note": "the two conditions never disagreed"}

    from math import comb
    tail = min(b, c)
    p = sum(comb(n, k) for k in range(tail + 1)) / (2 ** n) * 2
    return {
        "b_baseline_only": b,
        "c_injected_only": c,
        "n_discordant": n,
        "p_value": min(1.0, p),
        "test": "exact two sided binomial",
    }


# ------------------------------------------------------------------- model


@torch.no_grad()
def _forward(model, input_ids, cache=None, position_ids=None):
    kwargs = {}
    if cache is not None:
        kwargs["past_key_values"] = cache
    if position_ids is not None:
        kwargs["position_ids"] = position_ids
    return model(input_ids, use_cache=True, **kwargs)


def load_projections(contract) -> dict:
    saved = torch.load(PROJECTIONS_PATH, map_location="cpu")
    config = saved["config"]
    modules = {}
    for name, state in saved["state_dicts"].items():
        target, kind = name.split(":")
        module = CacheProjection(
            source_heads=1, source_head_dim=contract["sharer"]["kv_width"],
            target_heads=1, target_head_dim=contract["receiver"]["kv_width"],
            depth=config["depth"], hidden=config["hidden"],
            **({"activation": config["activation"]} if config["depth"] > 1 else {}),
        )
        module.load_state_dict(state)
        module.eval()
        modules[(int(target), kind)] = module
    return modules


def substitute_projected(receiver_cache, sharer_cache, mapping, projections,
                         heads, head_dim):
    """Receiver's own cache everywhere except the layers that have a partner."""
    sharer = cache_tensors(sharer_cache)
    pairs = []
    for layer, (keys, values) in enumerate(cache_tensors(receiver_cache)):
        source = mapping[layer]
        if source is None:
            pairs.append((keys.clone(), values.clone()))
            continue
        made = []
        for kind, tensor in (("keys", sharer[source][0]), ("values", sharer[source][1])):
            flat = projections[(layer, kind)].net(flatten_heads(tensor))
            made.append(unflatten_heads(flat, tensor.shape[0], heads, head_dim))
        pairs.append(tuple(made))
    return build_cache(pairs)


@torch.no_grad()
def score_dataset(sharer, receiver, tokenizer, rows, projections, mapping,
                  contract) -> list[dict]:
    letters = option_token_ids(tokenizer)
    heads = contract["receiver"]["n_kv_heads"]
    head_dim = contract["receiver"]["head_dim"]
    records = []

    for index, row in enumerate(rows):
        prompt = build_prompt(row["subject"], row["question"], row["choices"])
        ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        input_ids = torch.tensor([ids], dtype=torch.long, device=DEVICE)
        prefix, last = input_ids[:, :-1], input_ids[:, -1:]
        positions = absolute_position_ids(prefix.shape[1], 1, device=DEVICE)

        receiver_cache = clone_cache(_forward(receiver, prefix).past_key_values)
        sharer_cache = _forward(sharer, prefix).past_key_values

        baseline_logits = _forward(
            receiver, last, clone_cache(receiver_cache), positions
        ).logits[0, -1, letters]
        injected_logits = _forward(
            receiver, last,
            substitute_projected(receiver_cache, sharer_cache, mapping,
                                 projections, heads, head_dim),
            positions,
        ).logits[0, -1, letters]

        answer = int(row["answer"])
        baseline_choice = int(baseline_logits.argmax())
        injected_choice = int(injected_logits.argmax())
        records.append({
            "index": index,
            "subject": row["subject"],
            "n_prompt_tokens": len(ids),
            "answer": answer,
            "baseline_choice": baseline_choice,
            "injected_choice": injected_choice,
            "baseline_correct": baseline_choice == answer,
            "injected_correct": injected_choice == answer,
            "baseline_logits": [float(x) for x in baseline_logits],
            "injected_logits": [float(x) for x in injected_logits],
        })
        del receiver_cache, sharer_cache
    return records


def main() -> None:
    from datasets import get_dataset_config_names, load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    contract = load_json(CONTRACTS_PATH)
    if not PROJECTIONS_PATH.exists():
        raise SystemExit(
            f"missing {PROJECTIONS_PATH.relative_to(REPO_ROOT)}; run "
            "scripts/train_projection.py first"
        )

    subjects = sorted(get_dataset_config_names(DATASET))
    pool, dropped = [], {}
    for subject in subjects:
        for row in load_dataset(DATASET, subject, split=DATASET_SPLIT):
            error = row.get("error_type", KEEP_ERROR_TYPE)
            if error != KEEP_ERROR_TYPE:
                dropped[error] = dropped.get(error, 0) + 1
                continue
            pool.append({
                "subject": subject,
                "question": row["question"],
                "choices": list(row["choices"]),
                "answer": row["answer"],
            })

    if len(pool) < N_SAMPLES:
        raise SystemExit(
            f"{len(pool)} usable questions is fewer than the {N_SAMPLES} asked for"
        )
    order = torch.randperm(
        len(pool), generator=torch.Generator().manual_seed(SAMPLE_SEED)
    )[:N_SAMPLES].tolist()
    rows = [pool[i] for i in order]

    tokenizer = AutoTokenizer.from_pretrained(RECEIVER_ID)
    sharer = AutoModelForCausalLM.from_pretrained(
        SHARER_ID, dtype=DTYPE, attn_implementation=ATTN_IMPLEMENTATION
    ).to(DEVICE).eval()
    receiver = AutoModelForCausalLM.from_pretrained(
        RECEIVER_ID, dtype=DTYPE, attn_implementation=ATTN_IMPLEMENTATION
    ).to(DEVICE).eval()

    mapping = align_layers(
        contract["sharer"]["n_layers"], contract["receiver"]["n_layers"],
        strategy="terminal",
    )
    records = score_dataset(
        sharer, receiver, tokenizer, rows, load_projections(contract),
        mapping, contract,
    )

    baseline = [r["baseline_correct"] for r in records]
    injected = [r["injected_correct"] for r in records]
    generator = torch.Generator().manual_seed(BOOTSTRAP_SEED)

    out = RESULTS_DIR / f"run_{date.today().isoformat()}_n{len(records)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "conditions": ["baseline", "injected"],
        "accuracy": {
            "baseline": sum(baseline) / len(baseline),
            "injected": sum(injected) / len(injected),
        },
        "bootstrap": paired_bootstrap(baseline, injected, generator),
        "mcnemar": mcnemar(baseline, injected),
        "per_sample": records,
        "corpus": {
            "dataset": DATASET, "split": DATASET_SPLIT,
            "n_subjects": len(subjects), "kept_error_type": KEEP_ERROR_TYPE,
            "pool_size": len(pool), "dropped_by_error_type": dropped,
            "n_sampled": len(records), "sample_seed": SAMPLE_SEED,
        },
        "model_ids": {"sharer": SHARER_ID, "receiver": RECEIVER_ID},
        "option_letters": list(OPTIONS),
        "scoring": "argmax over the four option letter logits after 'Answer:'",
        "replaced_target_layers": [t for t, s in enumerate(mapping) if s is not None],
        "dtype": str(DTYPE), "device": DEVICE,
        "attn_implementation": ATTN_IMPLEMENTATION,
        "torch_version": torch.__version__,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()