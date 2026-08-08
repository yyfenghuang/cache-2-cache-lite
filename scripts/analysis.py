# SPDX-License-Identifier: Apache-2.0
"""Accuracy, once, with a confidence interval and a paired test.

Silent on success. Writes results/run_<date>_n<N>.json.

Several conditions on the same questions
----------------------------------------
The Receiver answers each question once per condition, and everything outside
the cache is held identical: same prompt, same tokens, same position ids, same
scoring. Conditions differ in what the Receiver reads, and nothing else, which
is the only arrangement under which a difference between them means anything.

Which conditions run is a constant at the top of this file, not a shape baked
into the code, because this comparison has grown from two to four and there is
no reason to think four is where it stops. Each condition names the checkpoint
it needs, and dropping one from the list drops its requirement with it.

The expensive part is shared. Both caches are built once per question and every
condition reads the same pair, so the fourth condition costs a one-token
forward rather than another pass over the corpus.

Because all conditions answer the same questions, the samples are paired and
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
import time
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
from c2c.fuser import FuserBank  # noqa: E402
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

# The Receiver alone, the wikitext projection that replaces, and the two arms
# of the fuser comparison. Ordered so that "baseline" is first; nothing else
# about the order matters.
CONDITIONS = ("baseline", "projection_mse", "fused", "replace")

# Which fuser run supplies each of its conditions. The checkpoint is read from
# the log rather than assembled from the same parts a second time, so a log and
# the weights it describes cannot come apart.
FUSER_RUNS = {
    "fused": "fused_2026-08-06_n2000",
    "replace": "replace_2026-08-07_n2000",
}

# Every pair worth a paired test. The first three price each condition against
# doing nothing. The fourth is the one variable the fuser tier was built to
# isolate, and it is the only comparison here that does not involve the
# Receiver alone.
COMPARISONS = (
    ("baseline", "fused"),
    ("baseline", "replace"),
    ("baseline", "projection_mse"),
    ("replace", "fused"),
)

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


def paired_bootstrap(a, b, generator) -> dict:
    """Resample questions, not conditions.

    The two conditions answered the same questions, so a resample draws a
    question and takes both of its outcomes. Drawing them independently would
    discard the pairing and widen the interval past what the design earns.
    """
    n = len(a)
    a = torch.tensor(a, dtype=torch.float64)
    b = torch.tensor(b, dtype=torch.float64)
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


def mcnemar(a, b) -> dict:
    """Only the questions the two conditions disagree on carry information.

    Exact two sided binomial on the discordant pairs. The normal
    approximation is unreliable when the discordant count is small, and there
    is no reason to risk it for a test this cheap.
    """
    a_only = sum(1 for x, y in zip(a, b) if x and not y)
    b_only = sum(1 for x, y in zip(a, b) if y and not x)
    n = a_only + b_only
    if n == 0:
        return {"a_only": 0, "b_only": 0, "n_discordant": 0, "p_value": 1.0,
                "test": "exact two sided binomial",
                "note": "the two conditions never disagreed"}

    from math import comb
    tail = min(a_only, b_only)
    p = sum(comb(n, k) for k in range(tail + 1)) / (2 ** n) * 2
    return {
        "a_only": a_only,
        "b_only": b_only,
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


def load_fuser(run_name: str, contract: dict, expected_mode: str):
    """The bank a training run produced, together with the log describing it.

    The checkpoint filename comes out of the log rather than being assembled
    from the same parts a second time, so a log and the weights it describes
    cannot come apart. The mode is checked against the log too: loading a
    replacement checkpoint into a residual bank would succeed, every shape
    would agree, and the result would be a system neither arm ever trained.
    """
    log = load_json(RESULTS_DIR / f"fuser_{run_name}.json")
    if log["mode"] != expected_mode:
        raise SystemExit(
            f"run {run_name!r} is a {log['mode']!r} run and is being loaded "
            f"as {expected_mode!r}"
        )
    checkpoint = RESULTS_DIR / log["checkpoint"]
    if not checkpoint.exists():
        raise SystemExit(
            f"run {run_name!r} names {log['checkpoint']} as its checkpoint "
            "and it is not in results/"
        )
    receiver, sharer = contract["receiver"], contract["sharer"]
    bank = FuserBank(
        receiver["n_layers"], sharer["n_layers"],
        receiver["n_kv_heads"], receiver["head_dim"],
        sharer["n_kv_heads"], sharer["head_dim"],
        residual=(expected_mode == "fused"),
    )
    bank.load_state_dict(torch.load(checkpoint, weights_only=True))
    bank.eval()
    return bank, log


def fuser_builder(bank: FuserBank):
    """A cache builder bound to one bank.

    Written as a factory rather than a closure inside a loop, because a
    closure over a loop variable would leave every condition holding the last
    bank and every shape check would still pass.
    """
    def build(receiver_cache, sharer_cache):
        return build_cache(
            bank(cache_tensors(receiver_cache), cache_tensors(sharer_cache))
        )
    return build


@torch.no_grad()
def score_dataset(sharer, receiver, tokenizer, rows, builders) -> list[dict]:
    """Every condition answers every question, from one pair of caches.

    The two prefills are the expensive part and they are computed once per
    question. A condition costs a cache construction and a one-token forward
    on top of that, which is why four conditions cost about what two did.

    Each condition receives its own cache object. A forward pass appends to
    the cache it is handed, so a cache reused between conditions has grown by
    the time the second one reads it, and the comparison stops meaning
    anything without raising.
    """
    letters = option_token_ids(tokenizer)
    records = []

    for index, row in enumerate(rows):
        prompt = build_prompt(row["subject"], row["question"], row["choices"])
        ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        input_ids = torch.tensor([ids], dtype=torch.long, device=DEVICE)
        prefix, last = input_ids[:, :-1], input_ids[:, -1:]
        positions = absolute_position_ids(prefix.shape[1], 1, device=DEVICE)

        receiver_cache = clone_cache(_forward(receiver, prefix).past_key_values)
        sharer_cache = _forward(sharer, prefix).past_key_values

        answer = int(row["answer"])
        record = {
            "index": index,
            "subject": row["subject"],
            "n_prompt_tokens": len(ids),
            "answer": answer,
            "choice": {},
            "correct": {},
            "logits": {},
        }
        for name in CONDITIONS:
            logits = _forward(
                receiver, last, builders[name](receiver_cache, sharer_cache),
                positions,
            ).logits[0, -1, letters]
            choice = int(logits.argmax())
            record["choice"][name] = choice
            record["correct"][name] = choice == answer
            record["logits"][name] = [float(x) for x in logits]
        records.append(record)
        del receiver_cache, sharer_cache
    return records


def main() -> None:
    from datasets import get_dataset_config_names, load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if "baseline" not in CONDITIONS:
        raise SystemExit("the Receiver alone is not among the conditions")
    for a, b in COMPARISONS:
        for name in (a, b):
            if name not in CONDITIONS:
                raise SystemExit(f"comparison names {name!r}, which is not scored")

    out = RESULTS_DIR / f"run_{date.today().isoformat()}_n{N_SAMPLES}.json"
    if out.exists():
        raise SystemExit(
            f"{out.relative_to(REPO_ROOT)} already exists. Move it aside; "
            "this script will not overwrite a graded run."
        )

    contract = load_json(CONTRACTS_PATH)
    mapping = align_layers(
        contract["sharer"]["n_layers"], contract["receiver"]["n_layers"],
        strategy="terminal",
    )

    builders = {"baseline": lambda receiver_cache, _: clone_cache(receiver_cache)}
    provenance = {}

    if "projection_mse" in CONDITIONS:
        if not PROJECTIONS_PATH.exists():
            raise SystemExit(
                f"missing {PROJECTIONS_PATH.relative_to(REPO_ROOT)}; run "
                "scripts/train_projection.py first"
            )
        projections = load_projections(contract)
        heads = contract["receiver"]["n_kv_heads"]
        head_dim = contract["receiver"]["head_dim"]
        builders["projection_mse"] = (
            lambda receiver_cache, sharer_cache: substitute_projected(
                receiver_cache, sharer_cache, mapping, projections,
                heads, head_dim,
            )
        )
        provenance["projection_mse"] = {"checkpoint": PROJECTIONS_PATH.name}

    for name in ("fused", "replace"):
        if name not in CONDITIONS:
            continue
        bank, log = load_fuser(FUSER_RUNS[name], contract, name)
        builders[name] = fuser_builder(bank)
        provenance[name] = {
            "run_name": log["run_name"],
            "checkpoint": log["checkpoint"],
            "total_steps": log["total_steps"],
            "n_train": log["n_train"],
            "dataset": log["dataset"],
            "dataset_split": log["dataset_split"],
            "n_trainable": log["n_trainable"],
            "gate_activation_ratio": log["history"][-1].get("gate_activation_ratio"),
        }

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

    started = time.perf_counter()
    records = score_dataset(sharer, receiver, tokenizer, rows, builders)
    elapsed = time.perf_counter() - started

    correct = {
        name: [r["correct"][name] for r in records] for name in CONDITIONS
    }
    generator = torch.Generator().manual_seed(BOOTSTRAP_SEED)
    comparisons = {}
    for a, b in COMPARISONS:
        comparisons[f"{a}_vs_{b}"] = {
            "a": a,
            "b": b,
            "bootstrap": paired_bootstrap(correct[a], correct[b], generator),
            "mcnemar": mcnemar(correct[a], correct[b]),
        }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "conditions": list(CONDITIONS),
        "accuracy": {
            name: sum(values) / len(values) for name, values in correct.items()
        },
        "answer_distribution": {
            name: [sum(1 for r in records if r["choice"][name] == i)
                   for i in range(len(OPTIONS))]
            for name in CONDITIONS
        },
        "comparisons": comparisons,
        "provenance": provenance,
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
        "seconds_total": elapsed,
        "seconds_per_question": elapsed / len(records),
        "dtype": str(DTYPE), "device": DEVICE,
        "attn_implementation": ATTN_IMPLEMENTATION,
        "torch_num_threads": torch.get_num_threads(),
        "torch_version": torch.__version__,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()