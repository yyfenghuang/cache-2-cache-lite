"""Train the fuser under the objective the residual form entails.

Silent on success. Writes results/fuser_<mode>_<date>_n<N>.json and the
checkpoint beside it under the same name.

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
from datetime import date
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
from c2c.gate import ScalarGate, annealed_temperature  # noqa: E402
from c2c.prompting import (  # noqa: E402
    build_prompt, decompose_loss, option_token_ids,
)

# ----------------------------------------------------------------- constants

# "fused" adds under a gate. "replace" substitutes. Run both; the second is
# what prices the first.
MODE = "replace"

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
# Measured on this machine at 1.92 seconds per sample, six threads, float32,
# eager, from the first recorded run of this script. Training
# only, before validation: 2,000 samples is about 1.1 hours per arm and the
# ladder asks for two arms. The reference figure of 15,000 is about 8 hours
# per arm and 23 hours for the pair once validation is counted.
N_TRAIN = 2000
MAX_PROMPT_TOKENS = 512

ACCUMULATE = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.0
MAX_GRAD_NORM = 1.0

# The gate gets its own step size, and this is the correction to the run that
# produced no movement at all. A scalar that has to travel from its shut
# initialisation across a threshold, and a weight matrix that has to be
# nudged, have no reason to share a step size. Under Adam a parameter
# moves at most about `lr` per step whatever its gradient says, so at 1e-4 a
# gate initialised at -2.0 needs 20,000 steps to reach the threshold. The
# first run had 32.
#
# Weight decay is zero here and not merely small. Decay pulls a logit toward
# zero, and zero is the threshold, so a decaying gate drifts toward the
# decision boundary for reasons that have nothing to do with the data.
GATE_LEARNING_RATE = 3e-2

# How many times over the gate must be able to cross its own initialisation
# within the step budget. Two reasons it is not 1. The temperature anneals
# toward saturation across the run, so the later steps pass less gradient to
# the gate than the earlier ones. And a gate that can only just reach the
# threshold can open but never close again, which is not a decision.
GATE_TRAVERSAL_MARGIN = 3.0

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

# Dated and sized, following results/run_<date>_n<N>.json. A run costs hours
# here, and a fixed filename means the second run destroys the first without
# asking. One log has already been lost that way. The date and the sample
# count are the two things that actually distinguish two runs of this script
# once MODE is in the name, so they are what the name carries.
RUN_NAME = f"{MODE}_{date.today().isoformat()}_n{N_TRAIN}"
LOG_PATH = RESULTS_DIR / f"fuser_{RUN_NAME}.json"
CHECKPOINT_PATH = RESULTS_DIR / f"fuser_{RUN_NAME}.pt"


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
def evaluate(
    receiver, sharer, bank, samples, letters, soft: bool = False, seed: int = SEED
) -> dict:
    """Held-out loss and four-way accuracy on the same questions every time.

    Three things this is asked to measure, and they are not interchangeable.

    `bank=None` measures the Receiver alone. That is the floor, and it is what
    the fused system must reproduce exactly while its gates are shut.

    `soft=False` puts the bank in eval mode, so the gates are the hard
    thresholds the graded system will use. This is the number that matters and
    it is also the number that says nothing at all while every gate is shut,
    because a shut gate pins the fused system to the floor by construction.
    The first run produced five of these and all five were the same number.

    All three report the loss split into the part the grader can see and the
    part it normalises away. On the first fused run the reported loss fell
    0.517 while the term that can change an answer accounted for 0.123 of it,
    with a confidence interval crossing zero. Recording only the total is
    therefore recording mostly the term that cannot matter, and the split
    costs nothing: the logits are already computed and it is arithmetic.

    `soft=True` keeps the bank in training mode, so the gates are relaxed
    samples and the fuser contributes at partial strength. This is the only
    view that moves before a gate opens, and it is what separates "the
    projection is learning nothing" from "the projection is learning and the
    gate has not let it through yet". The noise generator is reseeded to the
    same value at the start of every pass, so successive measurements are
    paired on the draws as well as on the questions and a difference between
    them cannot be a different roll.
    """
    was_training = bank.training if bank is not None else False
    if bank is not None:
        bank.train() if soft else bank.eval()
    generator = torch.Generator(device=DEVICE).manual_seed(seed) if soft else None

    terms = ("full", "four_way", "letter_mass", "mass_on_letters")
    totals = dict.fromkeys(terms, 0.0)
    n_correct = 0
    for sample in samples:
        receiver_pairs = _prefill(receiver, sample["prefix"])
        if bank is None:
            cache_pairs = receiver_pairs
        else:
            sharer_pairs = _prefill(sharer, sample["prefix"])
            cache_pairs = bank(receiver_pairs, sharer_pairs, generator)
        record = decompose_loss(
            _score(receiver, sample, cache_pairs), letters, sample["answer"]
        )
        for term in terms:
            totals[term] += record[term]
        n_correct += int(record["correct"])

    if bank is not None and was_training:
        bank.train()
    n = len(samples)
    return {
        # `loss` stays the full-vocabulary figure, because the floor
        # assertion and every existing log compare against it. What is new is
        # that it is no longer the only thing recorded.
        "loss": totals["full"] / n,
        "four_way": totals["four_way"] / n,
        "letter_mass": totals["letter_mass"] / n,
        "mass_on_letters": totals["mass_on_letters"] / n,
        "accuracy": n_correct / n,
        "n": n,
    }


def gate_reach(bank: FuserBank, total_steps: int) -> dict:
    """Can the gate get where it has to go before the run ends.

    Answered before a single forward pass, because the alternative is
    answering it afterwards from a log full of identical numbers.

    The distance is read from the gates themselves rather than from the
    constant they were initialised with, so that changing the initialisation
    moves this check with it instead of leaving it behind.
    """
    distance = max(
        abs(float(gate.logit.detach()))
        for gate in bank.modules()
        if isinstance(gate, ScalarGate)
    )
    reach = GATE_LEARNING_RATE * total_steps
    return {
        "distance": distance,
        "reach": reach,
        "crossings": reach / distance if distance > 0 else float("inf"),
        "required": GATE_TRAVERSAL_MARGIN,
        "minimum_learning_rate": distance * GATE_TRAVERSAL_MARGIN / total_steps,
    }


def build_optimizer(bank: FuserBank):
    """Two groups, because the two kinds of parameter are not alike.

    The split is asserted rather than assumed. A rename inside `c2c/gate.py`
    would silently empty the gate group and hand every gate back the step size
    that already failed once, and nothing downstream would raise.
    """
    gate_parameters, other_parameters = [], []
    for name, parameter in bank.named_parameters():
        (gate_parameters if ".gate." in name else other_parameters).append(parameter)
    if not gate_parameters or not other_parameters:
        raise SystemExit(
            f"the parameter split found {len(gate_parameters)} gate and "
            f"{len(other_parameters)} other parameters; one group is empty, so "
            "the two learning rates are not both in use"
        )
    return torch.optim.AdamW(
        [
            {
                "params": other_parameters,
                "lr": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
            },
            {"params": gate_parameters, "lr": GATE_LEARNING_RATE, "weight_decay": 0.0},
        ]
    ), gate_parameters, other_parameters


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
    optimizer, gate_parameters, other_parameters = build_optimizer(bank)
    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    reach = gate_reach(bank, total_steps)
    if bank.residual and reach["crossings"] < reach["required"]:
        raise SystemExit(
            f"the gate cannot reach its threshold in {total_steps} steps. It "
            f"starts {reach['distance']:.3f} away and can move about "
            f"{reach['reach']:.3f} in total, which is {reach['crossings']:.2f} "
            f"crossings against the {reach['required']:.1f} required. Under "
            "Adam a parameter moves at most about one learning rate per step "
            "whatever its gradient says, so this is arithmetic and not a "
            "prediction: the run would end with every gate where it started "
            "and every validation number equal to the baseline. Raise "
            f"GATE_LEARNING_RATE to at least {reach['minimum_learning_rate']:.4f}, "
            "or lengthen the run, or start the gates nearer the threshold."
        )

    baseline = evaluate(receiver, sharer, None, validation_samples, letters)
    initial = evaluate(receiver, sharer, bank, validation_samples, letters)
    initial_soft = evaluate(
        receiver, sharer, bank, validation_samples, letters, soft=True, seed=seed
    )

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
            **{f"soft_{k}": v for k, v in initial_soft.items()},
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

        # Clipped per group. A single global norm couples the gate's step to
        # the projection's gradient, which is a coupling nobody asked for and
        # which is invisible in the log unless the two are reported apart.
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(other_parameters, MAX_GRAD_NORM)
        )
        gate_grad_norm = float(
            torch.nn.utils.clip_grad_norm_(gate_parameters, MAX_GRAD_NORM)
        )
        optimizer.step()
        step_times.append(time.perf_counter() - step_started)

        record = {
            "step": step + 1,
            "temperature": annealed_temperature(step, total_steps),
            "train_loss": accumulated,
            "grad_norm": grad_norm,
            "gate_grad_norm": gate_grad_norm,
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
            record.update(
                {
                    f"soft_{k}": v
                    for k, v in evaluate(
                        receiver, sharer, bank, validation_samples, letters,
                        soft=True, seed=seed,
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
        "initial_soft": initial_soft,
        "gate_reach": reach,
        "final": history[-1],
        "history": history,
        "total_steps": total_steps,
        "accumulate": ACCUMULATE,
        "n_train": len(train_samples),
        "n_validation": len(validation_samples),
        "n_trainable": bank.n_parameters(),
        "learning_rate": LEARNING_RATE,
        "gate_learning_rate": GATE_LEARNING_RATE,
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
    # Checked before the imports, not merely before the weights. "First" is
    # cheap to make literal here, and a collision found at the end of the run
    # is found three hours too late to be a guard at all.
    for path in (LOG_PATH, CHECKPOINT_PATH):
        if path.exists():
            raise SystemExit(
                f"{path.relative_to(REPO_ROOT)} already exists. A run today "
                f"at MODE={MODE!r} and N_TRAIN={N_TRAIN} has been recorded "
                "already. Move it aside or change the configuration; this "
                "script will not overwrite hours of measurement."
            )

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

    # Seeded here and not only in `train`, because the two arms have to start
    # from the same weights. Same corpus, same loss, same steps and same seed
    # is the whole claim of the comparison, and an unseeded construction would
    # have quietly given the two arms different initial projections while
    # every other control held.
    torch.manual_seed(SEED)
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
            "run_name": RUN_NAME,
            # Recorded so the probe reads the weights this log describes and
            # not whichever checkpoint happens to sit beside it.
            "checkpoint": CHECKPOINT_PATH.name,
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
    torch.save(bank.state_dict(), CHECKPOINT_PATH)
    LOG_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()