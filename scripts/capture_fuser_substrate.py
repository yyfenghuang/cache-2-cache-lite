"""Two halves of one gate: a shut fuser is a no-op, and the loss reaches it.

Silent on success. Everything goes into results/fuser_substrate.json.

Neither half is meaningful alone, for the same reason the Tier 1 pair was not.

A fused cache that reproduces the baseline exactly is what a correctly wired
shut gate produces. It is also what a fuser whose output the framework
silently discards produces. The first half cannot tell those apart.

A gradient that reaches every fuser parameter is what a connected loss
produces. It says nothing about whether the fused tensors are the ones the
model actually attended over. The second half cannot tell that apart either.

Together they pin it down: the tensors this module emits are read, and the
loss can travel back to the module that emitted them.

What is being guarded against
-----------------------------
The failure this exists to catch is a cache that arrives detached. Every shape
check passes, every value is finite, the loss falls because the model is
merely being conditioned on a constant, and the fuser learns nothing. The
symptom appears at the end of a training run rather than at the start of one,
which is the most expensive place for it to appear on a machine without a GPU.

The measurement lives in `measure_fuser_substrate`, which takes models rather
than loading them, so the whole of it can be exercised against two small
models built from configs in about a second with no checkpoint on disk. What
remains untested until this script runs for real is the pair of lines that
load weights.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from c2c.cache_ops import (  # noqa: E402
    absolute_position_ids,
    build_cache,
    cache_tensors,
    max_abs_difference,
)
from c2c.fuser import FuserBank  # noqa: E402
from c2c.gate import TEMPERATURE_START, close_gate, open_gate  # noqa: E402

SHARER_ID = "Qwen/Qwen2.5-0.5B-Instruct"
RECEIVER_ID = "Qwen/Qwen3-0.6B"

PROMPT = (
    "A fuser reads two key-value caches and emits a correction to one of "
    "them. The correction is added rather than substituted, so the model "
    "that receives it keeps whatever it already understood and the learned "
    "part only has to supply what it did not. A gate multiplies that "
    "correction, and a shut gate multiplies it by zero exactly."
)

DTYPE = torch.float32
DEVICE = "cpu"
ATTN_IMPLEMENTATION = "eager"
SEED = 42

CONTRACTS_PATH = REPO_ROOT / "results" / "contracts.json"
RESULTS_PATH = REPO_ROOT / "results" / "fuser_substrate.json"


@torch.no_grad()
def _prefill(model, input_ids):
    return model(input_ids, use_cache=True).past_key_values


def _forward_suffix(model, suffix, cache, position_ids):
    return model(
        suffix, past_key_values=cache, position_ids=position_ids, use_cache=True
    ).logits


def _next_token_loss(logits, suffix):
    """Next-token prediction over the response, which is the real objective.

    The logit at response position i predicts position i+1, so the last one
    predicts a token this sequence does not contain and is dropped. Using the
    training objective here rather than a convenient scalar means the graph
    exercised by this gate is the graph training will use.
    """
    return F.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.shape[-1]),
        suffix[:, 1:].reshape(-1),
    )


def measure_fuser_substrate(
    receiver_model,
    sharer_model,
    bank: FuserBank,
    input_ids: torch.Tensor,
    n_prefix: int,
    seed: int = SEED,
) -> dict:
    """Run both halves and report one number each, plus the gradient census.

    Every condition receives its own cache. A forward pass appends to the
    cache it is handed, so a cache reused between conditions has grown by the
    time the second one reads it, and the comparison stops meaning anything
    without raising.
    """
    total = int(input_ids.shape[1])
    if not 1 <= n_prefix < total:
        raise ValueError(
            f"n_prefix {n_prefix} does not split a length {total} sequence"
        )
    n_new = total - n_prefix
    if n_new < 2:
        raise ValueError(
            f"the response is {n_new} tokens; next-token loss needs at least 2"
        )
    prefix, suffix = input_ids[:, :n_prefix], input_ids[:, n_prefix:]
    device = input_ids.device
    positions = absolute_position_ids(n_prefix, n_new, device=device)

    for parameter in receiver_model.parameters():
        parameter.requires_grad_(False)
    for parameter in sharer_model.parameters():
        parameter.requires_grad_(False)

    receiver_pairs = cache_tensors(_prefill(receiver_model, prefix))
    sharer_pairs = cache_tensors(_prefill(sharer_model, prefix))

    with torch.no_grad():
        reference = _forward_suffix(
            receiver_model, suffix, build_cache(receiver_pairs), positions
        )

    # First half. Shut, evaluating, and therefore exact.
    bank.eval()
    close_gate(bank)
    with torch.no_grad():
        shut = _forward_suffix(
            receiver_model,
            suffix,
            build_cache(bank(receiver_pairs, sharer_pairs)),
            positions,
        )
        shut_correction = max(
            float(
                bank.fusers_for(kind)[target]
                .correction(receiver_pairs[target][index], sharer_pairs[source][index])
                .abs()
                .max()
            )
            for target, source in enumerate(bank.mapping)
            if source is not None
            for index, kind in enumerate(("keys", "values"))
        )

    # Second half. Open, evaluating, and required to move the logits.
    open_gate(bank)
    with torch.no_grad():
        opened = _forward_suffix(
            receiver_model,
            suffix,
            build_cache(bank(receiver_pairs, sharer_pairs)),
            positions,
        )

    # Gradient census, under the conditions of the first real training step:
    # gates at their shut initialisation, temperature at the annealing start.
    fresh = FuserBank(
        bank.n_receiver_layers,
        bank.n_sharer_layers,
        bank.key_fusers[bank.paired_layers[0]].receiver_heads,
        bank.key_fusers[bank.paired_layers[0]].receiver_head_dim,
        bank.key_fusers[bank.paired_layers[0]].sharer_heads,
        bank.key_fusers[bank.paired_layers[0]].sharer_head_dim,
        hidden=bank.key_fusers[bank.paired_layers[0]].hidden,
    ).train()
    fresh.set_temperature(TEMPERATURE_START)
    generator = torch.Generator(device=device).manual_seed(seed)

    fused_logits = _forward_suffix(
        receiver_model,
        suffix,
        build_cache(fresh(receiver_pairs, sharer_pairs, generator)),
        positions,
    )
    loss = _next_token_loss(fused_logits, suffix)
    loss.backward()

    grads = {
        name: parameter.grad for name, parameter in fresh.named_parameters()
    }
    missing = sorted(name for name, grad in grads.items() if grad is None)
    norms = {
        name: float(grad.norm())
        for name, grad in grads.items()
        if grad is not None
    }
    nonfinite = sorted(
        name
        for name, grad in grads.items()
        if grad is not None and not bool(torch.isfinite(grad).all())
    )
    zero = sorted(name for name, norm in norms.items() if norm == 0.0)

    logit_scale = float(reference.abs().max())
    conditions = {
        "shut_gate_identity": max_abs_difference(shut, reference),
        "open_gate_change": max_abs_difference(opened, reference),
    }

    return {
        "conditions": conditions,
        "ratios": {k: v / logit_scale for k, v in conditions.items()},
        "logit_scale": logit_scale,
        "shut_gate_max_abs_correction": shut_correction,
        "gradient": {
            "loss": float(loss.detach()),
            "n_parameters": len(grads),
            "n_missing": len(missing),
            "n_zero": len(zero),
            "n_nonfinite": len(nonfinite),
            "missing": missing[:8],
            "zero": zero[:8],
            "nonfinite": nonfinite[:8],
            "min_norm": min(norms.values()) if norms else float("nan"),
            "max_norm": max(norms.values()) if norms else float("nan"),
        },
        "bank": {
            "n_receiver_layers": bank.n_receiver_layers,
            "n_sharer_layers": bank.n_sharer_layers,
            "paired_layers": len(bank.paired_layers),
            "unpaired_layers": bank.unpaired_layers,
            "n_trainable": fresh.n_parameters(),
            "gate_activation_ratio_at_init": fresh.gate_activation_ratio(),
        },
        "receiver_cache_shape": list(receiver_pairs[0][0].shape),
        "sharer_cache_shape": list(sharer_pairs[0][0].shape),
        "sequence": {"n_total": total, "n_prefix": n_prefix, "n_new": n_new},
        "temperature": TEMPERATURE_START,
        "seed": seed,
        "torch_num_threads": torch.get_num_threads(),
    }


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not CONTRACTS_PATH.exists():
        raise SystemExit(
            "missing results/contracts.json; the Tier 0 gate has not been "
            "closed and the bank has no geometry to be built from"
        )
    contract = json.loads(CONTRACTS_PATH.read_text(encoding="utf-8"))
    receiver_contract, sharer_contract = contract["receiver"], contract["sharer"]

    tokenizer = AutoTokenizer.from_pretrained(RECEIVER_ID)
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

    bank = FuserBank(
        receiver_contract["n_layers"],
        sharer_contract["n_layers"],
        receiver_contract["n_kv_heads"],
        receiver_contract["head_dim"],
        sharer_contract["n_kv_heads"],
        sharer_contract["head_dim"],
    )

    input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"].to(DEVICE)
    n_prefix = int(input_ids.shape[1]) // 2

    result = measure_fuser_substrate(receiver, sharer, bank, input_ids, n_prefix)
    result["receiver_id"] = RECEIVER_ID
    result["sharer_id"] = SHARER_ID
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