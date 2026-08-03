"""Train the fuser under the objective the residual form entails.

Silent on success. Writes results/fuser_log_<mode>.json and
results/fuser_<mode>.pt.

Two arms, one script
--------------------
`MODE` selects between adding and substituting. Both arms use the same module,
the same corpus, the same loss, the same step count and the same seed, and
they run through the same code path down to the single branch inside
`CacheFuser.forward`. That is the whole point: the run recorded in
`results/run_2026-08-03_n500.json` differed from a fused build in three
respects at once, and holding two of them fixed is what leaves one.

The replacement arm here is not the paper's Project row. That row also drops
the Receiver's cache from the module's input, so it changes two things at
once. This one changes exactly the addition, and is a stricter control for
the question actually being asked.

Why mean squared error is gone
------------------------------
The fused cache is `C_n + F(C_n, C_s)`. Under mean squared error against the
Receiver's own cache the target already stands on the left, and the loss is
minimised at `F = 0`. There is no ground truth for a correction. So the
objective is next-token prediction with both models frozen, which is also
what the reference uses, and the Receiver enters the training graph. The graph
is short: both caches are produced under `no_grad` and enter as constants, so
backpropagation spans one token of forward and nothing else.

The floor, which is this tier's geometric null
----------------------------------------------
One tier below, a falling loss curve was readable only because
`results/geometric_null.json` said where the constant predictor sat. The
equivalent here is the Receiver's own loss on the same held-out questions with
no fuser at all, measured before training starts and written into the log. A
loss that falls without reaching it is a loss that fell to somewhere worse
than doing nothing.

In the residual arm that floor is also checked rather than merely recorded.
The gates initialise shut, and a shut gate at inference multiplies by exactly
zero, so the first validation pass must reproduce the baseline to the last
bit. It is asserted. If it does not hold, the run stops, because everything
measured after that point would be measured against a baseline the system
cannot actually reach.

Where the cache is cut
----------------------
`scripts/analysis.py` scores by reading the logits of the four option letters
at the last prompt position, the one holding "Answer:". So the cache covers
every token but that one, the fuser acts on it, and the Receiver forwards the
final token alone on top of the fused cache. The logit that gets a gradient
here is therefore the same logit that gets graded there, at the same position,
under the same prompt. Nothing about the training arrangement has to be
translated into the grading arrangement afterwards.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from c2c.cache_ops import (  # noqa: E402
    absolute_position_ids,
    build_cache,
    cache_tensors,
)
from c2c.fuser import FuserBank  # noqa: E402
from c2c.gate import annealed_temperature  # noqa: E402
from c2c.prompting import build_prompt, option_token_ids  # noqa: E402

# ----------------------------------------------------------------- constants

# "fused" adds under a gate. "replace" substitutes. Run both; the second is
# what prices the first.
MODE = "fused"

SHARER_ID = "Qwen/Qwen2.5-0.5B-Instruct"
RECEIVER_ID = "Qwen/Qwen3-0.6B"

# MMLU auxiliary train is drawn from ARC, MC-TEST, OpenBookQA and RACE and
# holds no MMLU test items, so grading on MMLU-Redux is not contaminated by
# it. The split lives inside the "all" configuration under this name, which
# is not "train"; a copy of this dataset without it is published separately,
# so a loader that silently finds no split is the likely first failure here.
DATASET = "cais/mmlu"
DATASET_CONFIG = "all"
DATASET_SPLIT = "auxiliary_train"

# Deliberately small. The reference figure is 15,000 samples at 116 steps, and
# it was measured on datacentre hardware. Nothing in results/ records the cost
# of a step on this machine, so this starts at a size whose only job is to
# produce that number in the log. Raise it once `seconds_per_step` is known,
# and remember the ladder asks for two runs at whatever size is chosen.
N_TRAIN = 256
MAX_PROMPT_TOKENS = 512

ACCUMULATE = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.0
MAX_GRAD_NORM = 1.0

# These two set how much of the run is spent measuring rather than training,
# and the arithmetic is easy to get wrong by an order of magnitude. The unit
# of cost is a prefill, and a validation point costs two per sample.
#
#     training       N_TRAIN * 2
#     validation     (N_TRAIN / ACCUMULATE / VALIDATE_EVERY) * N_VALIDATION * 2
#
# At the values below both sides are near 512, which is the intended balance.
# Halving VALIDATE_EVERY doubles the second line and leaves the first alone.
N_VALIDATION = 64
VALIDATE_EVERY = 8

SEED = 42
DTYPE = torch.float32
DEVICE = "cpu"
ATTN_IMPLEMENTATION = "eager"

CONTRACTS_PATH = REPO_ROOT / "results" / "contracts.json"
SUBSTRATE_PATH = REPO_ROOT / "results" / "fuser_substrate.json"
RESULTS_DIR = REPO_ROOT / "results"


# ------------------------------------------------------------------- sample


def prepare(row, tokenizer, letters: list[int]) -> dict | None:
    """One question, tokenized, or None when it is too long to afford.

    Returns the prompt without its final token and that token separately,
    because the cache has to stop one short of the position being scored.
    """
    prompt = build_prompt(row["subject"], row["question"], row["choices"])
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if len(ids) < 2 or len(ids) > MAX_PROMPT_TOKENS:
        return None
    answer = int(row["answer"])
    if not 0 <= answer < len(letters):
        return None
    return {
        "prefix": torch.tensor([ids[:-1]], dtype=torch.long),
        "last": torch.tensor([ids[-1:]], dtype=torch.long),
        "answer": answer,
        "n_prompt_tokens": len(ids),
    }


# ------------------------------------------------------------------ forward


@torch.no_grad()
def _prefill(model, input_ids):
    return cache_tensors(model(input_ids, use_cache=True).past_key_values)


def _score(receiver, sample, cache_pairs):
    """The logits at the position `analysis.py` reads, and nowhere else."""
    n_prefix = int(sample["prefix"].shape[1])
    positions = absolute_position_ids(n_prefix, 1, device=sample["last"].device)
    out = receiver(
        sample["last"],
        past_key_values=build_cache(cache_pairs),
        position_ids=positions,
        use_cache=True,
    )
    return out.logits[0, -1, :]


def _loss_and_choice(logits, sample, letters):
    """Full-vocabulary next-token loss, plus the four-way pick as a readout.

    The loss is over the whole vocabulary because that is the objective the
    reference trains under, and narrowing it to four logits would be a change
    from the reference that no measurement here motivates. The four-way pick
    is recorded alongside it and optimises nothing; it exists so the log can
    be read in the units the result will eventually be stated in.
    """
    target = torch.tensor([letters[sample["answer"]]], device=logits.device)
    loss = F.cross_entropy(logits.unsqueeze(0), target)
    choice = int(torch.argmax(logits[letters]))
    return loss, choice == sample["answer"]


# --------------------------------------------------------------- evaluation


@torch.no_grad()
def evaluate(receiver, sharer, bank, samples, letters) -> dict:
    """Held-out loss and four-way accuracy, under inference gates.

    `bank` is put in eval mode, so the gates are the hard thresholds the
    graded system will use rather than the relaxed samples training uses.
    Passing `bank=None` measures the Receiver alone, which is the floor.
    """
    was_training = bank.training if bank is not None else False
    if bank is not None:
        bank.eval()

    total_loss, n_correct = 0.0, 0
    for sample in samples:
        receiver_pairs = _prefill(receiver, sample["prefix"])
        if bank is None:
            cache_pairs = receiver_pairs
        else:
            sharer_pairs = _prefill(sharer, sample["prefix"])
            cache_pairs = bank(receiver_pairs, sharer_pairs)
        loss, correct = _loss_and_choice(
            _score(receiver, sample, cache_pairs), sample, letters
        )
        total_loss += float(loss)
        n_correct += int(correct)

    if bank is not None and was_training:
        bank.train()
    return {
        "loss": total_loss / len(samples),
        "accuracy": n_correct / len(samples),
        "n": len(samples),
    }


# ----------------------------------------------------------------- training


def train(
    receiver,
    sharer,
    bank: FuserBank,
    train_samples: list[dict],
    validation_samples: list[dict],
    letters: list[int],
    seed: int = SEED,
) -> dict:
    """One pass over the training samples. Takes models rather than loading.

    Split this way so the whole loop can be exercised against two small models
    built from configs, in seconds, with no checkpoint on disk. What stays
    untested until this runs for real is the pair of lines that load weights
    and the one that loads the dataset.
    """
    for parameter in receiver.parameters():
        parameter.requires_grad_(False)
    for parameter in sharer.parameters():
        parameter.requires_grad_(False)

    total_steps = max(1, len(train_samples) // ACCUMULATE)
    optimizer = torch.optim.AdamW(
        bank.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    baseline = evaluate(receiver, sharer, None, validation_samples, letters)
    initial = evaluate(receiver, sharer, bank, validation_samples, letters)

    if bank.residual and initial["loss"] != baseline["loss"]:
        raise SystemExit(
            "the fused system at initialisation does not reproduce the "
            f"Receiver alone: {initial['loss']} against {baseline['loss']}. "
            "The gates initialise shut and a shut gate multiplies by exactly "
            "zero, so these must be equal. They are not, so the floor this "
            "tier is measured against is not reachable and nothing measured "
            "after this line would mean anything."
        )

    history = [
        {
            "step": 0,
            "temperature": annealed_temperature(0, total_steps),
            "train_loss": None,
            "gate_activation_ratio": bank.gate_activation_ratio(),
            "gate_logits": _gate_logits(bank),
            **{f"validation_{k}": v for k, v in initial.items()},
        }
    ]

    bank.train()
    started = time.perf_counter()
    step_times: list[float] = []
    index = 0

    for step in range(total_steps):
        step_started = time.perf_counter()
        bank.set_temperature(annealed_temperature(step, total_steps))
        optimizer.zero_grad(set_to_none=True)

        accumulated = 0.0
        for _ in range(ACCUMULATE):
            sample = train_samples[index]
            index += 1
            receiver_pairs = _prefill(receiver, sample["prefix"])
            sharer_pairs = _prefill(sharer, sample["prefix"])
            fused = bank(receiver_pairs, sharer_pairs, generator)
            loss, _ = _loss_and_choice(
                _score(receiver, sample, fused), sample, letters
            )
            (loss / ACCUMULATE).backward()
            accumulated += float(loss.detach()) / ACCUMULATE

        grad_norm = torch.nn.utils.clip_grad_norm_(
            bank.parameters(), MAX_GRAD_NORM
        )
        optimizer.step()
        step_times.append(time.perf_counter() - step_started)

        record = {
            "step": step + 1,
            "temperature": annealed_temperature(step, total_steps),
            "train_loss": accumulated,
            "grad_norm": float(grad_norm),
            "seconds": step_times[-1],
        }
        last = step == total_steps - 1
        if last or (step + 1) % VALIDATE_EVERY == 0:
            record.update(
                {
                    f"validation_{k}": v
                    for k, v in evaluate(
                        receiver, sharer, bank, validation_samples, letters
                    ).items()
                }
            )
            record["gate_activation_ratio"] = bank.gate_activation_ratio()
            record["gate_logits"] = _gate_logits(bank)
        history.append(record)

    return {
        "mode": "fused" if bank.residual else "replace",
        "baseline": baseline,
        "initial": initial,
        "final": history[-1],
        "history": history,
        "total_steps": total_steps,
        "accumulate": ACCUMULATE,
        "n_train": len(train_samples),
        "n_validation": len(validation_samples),
        "n_trainable": bank.n_parameters(),
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "max_grad_norm": MAX_GRAD_NORM,
        "seconds_total": time.perf_counter() - started,
        "seconds_per_step": sum(step_times) / len(step_times),
        "seconds_per_sample": sum(step_times) / (len(step_times) * ACCUMULATE),
        "seed": seed,
        "torch_num_threads": torch.get_num_threads(),
    }


def _gate_logits(bank: FuserBank) -> dict:
    """Every gate logit, for the animation and for A.4.2.

    Forty-eight numbers per validation point. Recorded as a distribution
    rather than a mean because the claim being checked is about how many
    gates are open, and a mean cannot distinguish half of them open from all
    of them half open.
    """
    with torch.no_grad():
        return {
            kind: [
                float(bank.fusers_for(kind)[target].gate.logit)
                for target in bank.paired_layers
            ]
            for kind in ("keys", "values")
        }


# --------------------------------------------------------------------- main


def main() -> None:
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    for path in (CONTRACTS_PATH, SUBSTRATE_PATH):
        if not path.exists():
            raise SystemExit(
                f"missing {path.relative_to(REPO_ROOT)}; the gate below this "
                "one has not been closed"
            )
    contract = json.loads(CONTRACTS_PATH.read_text(encoding="utf-8"))
    receiver_contract, sharer_contract = contract["receiver"], contract["sharer"]

    if MODE not in ("fused", "replace"):
        raise SystemExit(f"unknown MODE {MODE!r}")

    tokenizer = AutoTokenizer.from_pretrained(RECEIVER_ID)
    letters = option_token_ids(tokenizer)

    receiver = (
        AutoModelForCausalLM.from_pretrained(
            RECEIVER_ID, dtype=DTYPE, attn_implementation=ATTN_IMPLEMENTATION
        )
        .to(DEVICE)
        .eval()
    )
    sharer = (
        AutoModelForCausalLM.from_pretrained(
            SHARER_ID, dtype=DTYPE, attn_implementation=ATTN_IMPLEMENTATION
        )
        .to(DEVICE)
        .eval()
    )

    rows = load_dataset(DATASET, DATASET_CONFIG, split=DATASET_SPLIT)
    rows = rows.shuffle(seed=SEED)

    samples, skipped, cursor = [], 0, 0
    while len(samples) < N_TRAIN + N_VALIDATION and cursor < len(rows):
        prepared = prepare(rows[cursor], tokenizer, letters)
        cursor += 1
        if prepared is None:
            skipped += 1
            continue
        samples.append(prepared)
    if len(samples) < N_TRAIN + N_VALIDATION:
        raise SystemExit(
            f"only {len(samples)} usable questions in {len(rows)} rows"
        )

    # Validation is taken first and never seen by the optimiser.
    validation_samples = samples[:N_VALIDATION]
    train_samples = samples[N_VALIDATION:]

    bank = FuserBank(
        receiver_contract["n_layers"],
        sharer_contract["n_layers"],
        receiver_contract["n_kv_heads"],
        receiver_contract["head_dim"],
        sharer_contract["n_kv_heads"],
        sharer_contract["head_dim"],
        residual=(MODE == "fused"),
    )

    result = train(
        receiver, sharer, bank, train_samples, validation_samples, letters
    )
    result.update(
        {
            "receiver_id": RECEIVER_ID,
            "sharer_id": SHARER_ID,
            "dataset": DATASET,
            "dataset_config": DATASET_CONFIG,
            "dataset_split": DATASET_SPLIT,
            "max_prompt_tokens": MAX_PROMPT_TOKENS,
            "n_skipped": skipped,
            "option_token_ids": letters,
            "dtype": str(DTYPE),
            "device": DEVICE,
            "attn_implementation": ATTN_IMPLEMENTATION,
        }
    )

    import transformers

    result["transformers_version"] = transformers.__version__
    result["torch_version"] = torch.__version__

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"fuser_log_{MODE}.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    torch.save(bank.state_dict(), RESULTS_DIR / f"fuser_{MODE}.pt")


if __name__ == "__main__":
    main()