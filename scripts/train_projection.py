"""Train one projection per paired layer, per tensor kind, and grade it.

Silent on success. Everything goes into results/train_log.json.

Graded on held out data, never in sample
----------------------------------------
A projection costs degrees of freedom, and a fit read back on the data it was
fitted to spends them on the target rather than on the input. Measured on
pure noise with an affine map from 128 channels to 1024, which costs 129 per
output channel: at N = 387 the in sample relative loss is 0.668, at N = 32000
it is 0.996. Both are below one. In sample comparison therefore passes on
noise at every corpus size, and no corpus is large enough to repair that. The
two hidden layer configuration used here costs 289 per output channel, so it
spends more, not less.

The held out relative loss on the same noise is 1.497 and 1.004, above one in
both cases. It is the only comparison that carries the claim, so it is the
only one this gate reads.

Input contract
--------------
    results/geometric_null.json
        per split, per layer, per kind nulls, under the key "splits"

    results/caches/<split>/<role>_<kind>_layer<NN>.pt
        one tensor per layer, shape [positions, width], written by
        scripts/capture_dual_cache.py

One file per layer rather than one per model: training is independent per
layer, so streaming the layer axis keeps the resident set at one layer's
worth regardless of how large the corpus becomes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from c2c.alignment import align_layers  # noqa: E402
from c2c.cache_ops import (  # noqa: E402
    absolute_position_ids,
    build_cache,
    cache_tensors,
    clone_cache,
    corrupt_norm_matched,
    flatten_heads,
    max_abs_difference,
    unflatten_heads,
)
from c2c.projection import CacheProjection  # noqa: E402

NULL_PATH = REPO_ROOT / "results" / "geometric_null.json"
CONTRACTS_PATH = REPO_ROOT / "results" / "contracts.json"
CACHE_ROOT = REPO_ROOT / "results" / "caches"
LOG_PATH = REPO_ROOT / "results" / "train_log.json"
HELD_OUT_IDS_PATH = REPO_ROOT / "results" / "held_out_ids.json"

RECEIVER_ID = "Qwen/Qwen3-0.6B"
SHARER_ID = "Qwen/Qwen2.5-0.5B-Instruct"
INJECTION_DTYPE = torch.float32
INJECTION_DEVICE = "cpu"
ATTN_IMPLEMENTATION = "eager"

# Which held out chunk the injection is measured on, and where it is cut into
# a prefix that is replaced and a suffix that is forwarded on top.
INJECTION_CHUNK = 0
INJECTION_PREFIX_FRACTION = 0.5
CORRUPTION_SEED = 1234

# The same threshold Tier 1 fixed, in the same units: maximum absolute logit
# difference as a fraction of the logit scale.
MIN_DEGRADATION_RATIO = 0.02

SPLITS = ("train", "validation", "held_out")
KINDS = ("keys", "values")

# Identical for keys and values. The prediction on record compares their
# floors "under identical conditions", and a configuration that differs
# between them answers a different question than the one being asked.
DEPTH = 2
HIDDEN = 256
ACTIVATION = "gelu"

EPOCHS = 200
BATCH_SIZE = 512
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 0.0
SEED = 1234
CURVE_EVERY = 10

# Early stopping looks at the validation split and never at the one that
# grades. Without it the run of 2026-08-02 drove four value projections from
# a held out relative of about 0.70 at epoch 10 to above 1.0 at epoch 199
# while their training loss kept falling. Key projections showed none of it
# under the same configuration, which is a finding rather than a nuisance.
PATIENCE = 20

# Fixed before the first run. A held out relative loss at or above one means
# the projection did not beat a constant, whatever its curve did.
MAX_RELATIVE_HELD_OUT = 1.0

# Below this ratio of positions to degrees of freedom the held out number is
# still meaningful but the projection is unlikely to reach its own optimum,
# so the run records the shortfall rather than discovering it later.
MIN_POSITIONS_PER_PARAMETER = 50.0


def load_json(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(
            f"missing {path.relative_to(REPO_ROOT)}; the gate below it has "
            "not been closed"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def cache_path(split: str, role: str, kind: str, layer: int) -> Path:
    return CACHE_ROOT / split / f"{role}_{kind}_layer{layer:02d}.pt"


def load_layer(split: str, role: str, kind: str, layer: int) -> torch.Tensor:
    path = cache_path(split, role, kind, layer)
    if not path.exists():
        raise SystemExit(
            f"missing {path.relative_to(REPO_ROOT)}. The training reads cache "
            "tensors split into train and held out; scripts/capture_dual_cache.py "
            "must write them before this gate can run at all."
        )
    return torch.load(path, map_location="cpu").to(torch.float32)


def require_splits(null: dict) -> dict:
    """A null without a held out split cannot grade anything.

    Refusing here rather than proceeding is the whole point: a run that
    silently compared against an in sample null would produce a passing
    number and a false finding, and nothing downstream could tell.
    """
    splits = null.get("splits")
    if not isinstance(splits, dict) or set(SPLITS) - set(splits):
        raise SystemExit(
            "results/geometric_null.json has no train and held_out splits. "
            "In sample comparison passes on uninformative input at every "
            "corpus size, so this gate cannot be run against a single split "
            "null. Re-run scripts/capture_dual_cache.py with a document level "
            "split before training."
        )
    return splits


def train_one(x, y, nulls, generator) -> dict:
    """Fit one projection, stop on validation, report on held out.

    `x` and `y` are dicts keyed by split. The epoch is chosen by validation
    loss, the parameters at that epoch are restored, and every reported
    number comes from those parameters. Nothing that selects reads held out.
    """
    projection = CacheProjection(
        source_heads=1, source_head_dim=x["train"].shape[1],
        target_heads=1, target_head_dim=y["train"].shape[1],
        depth=DEPTH, hidden=HIDDEN,
        **({"activation": ACTIVATION} if DEPTH > 1 else {}),
    )
    optimiser = torch.optim.Adam(
        projection.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = torch.nn.MSELoss()
    n = x["train"].shape[0]
    curve = []

    def evaluate(split: str) -> float:
        with torch.no_grad():
            return float(loss_fn(projection.net(x[split]), y[split]))

    best_validation, best_epoch, best_state, waited = float("inf"), -1, None, 0
    stopped_at = EPOCHS - 1
    for epoch in range(EPOCHS):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH_SIZE):
            index = order[start:start + BATCH_SIZE]
            optimiser.zero_grad()
            loss = loss_fn(projection.net(x["train"][index]), y["train"][index])
            loss.backward()
            optimiser.step()

        validation = evaluate("validation")
        if validation < best_validation:
            best_validation, best_epoch, waited = validation, epoch, 0
            best_state = {
                name: tensor.detach().clone()
                for name, tensor in projection.state_dict().items()
            }
        else:
            waited += 1

        if epoch % CURVE_EVERY == 0 or epoch == EPOCHS - 1:
            curve.append({
                "epoch": epoch,
                "train": evaluate("train"),
                "validation": validation,
                # Diagnostic only. Recorded so that overfitting is legible.
                # Selection above never reads it.
                "held_out": evaluate("held_out"),
            })

        if waited >= PATIENCE:
            stopped_at = epoch
            break

    projection.load_state_dict(best_state)
    projection.eval()
    final_train = evaluate("train")
    final_validation = evaluate("validation")
    final_held = evaluate("held_out")
    best = min(curve, key=lambda point: point["held_out"])
    per_channel = projection.n_parameters_per_output_channel()

    return {
        "selection_split": "validation",
        "selected_epoch": best_epoch,
        "stopped_at_epoch": stopped_at,
        "final_train_mse": final_train,
        "final_validation_mse": final_validation,
        "final_held_out_mse": final_held,
        "null_train": nulls["train"],
        "null_validation": nulls["validation"],
        "null_held_out": nulls["held_out"],
        "relative_train": final_train / nulls["train"],
        "relative_validation": final_validation / nulls["validation"],
        "relative_held_out": final_held / nulls["held_out"],
        # What selecting on held out would have given. Never the criterion:
        # it grades the projection on the data it was tuned against. Recorded
        # so that the distance between it and the reported number shows how
        # much the honest procedure costs.
        "best_held_out_mse": best["held_out"],
        "best_held_out_epoch": best["epoch"],
        "relative_best_held_out": best["held_out"] / nulls["held_out"],
        "n_train_positions": n,
        "n_validation_positions": int(x["validation"].shape[0]),
        "n_held_out_positions": int(x["held_out"].shape[0]),
        "parameters_per_output_channel": per_channel,
        "positions_per_parameter": n / per_channel,
        "curve": curve,
    }, projection


# -------------------------------------------------------------------------
# gate two: does an injected projected cache move the output


@torch.no_grad()
def _forward(model, input_ids, cache=None, position_ids=None):
    kwargs = {}
    if cache is not None:
        kwargs["past_key_values"] = cache
    if position_ids is not None:
        kwargs["position_ids"] = position_ids
    return model(input_ids, use_cache=True, **kwargs)


def substitute(receiver_cache, replacements) -> object:
    """A cache holding the Receiver's own tensors except where told otherwise.

    Target layers with no Sharer partner keep the Receiver's own entries.
    Nothing is projected into them because nothing was trained for them, and
    zeroing them would measure the cost of blanking four layers rather than
    the effect of the projection.
    """
    pairs = []
    for layer, (keys, values) in enumerate(cache_tensors(receiver_cache)):
        new_keys, new_values = replacements.get(layer, (None, None))
        pairs.append((
            keys.clone() if new_keys is None else new_keys,
            values.clone() if new_values is None else new_values,
        ))
    return build_cache(pairs)


def project_cache(sharer_cache, mapping, projections, heads, head_dim) -> dict:
    """Run each trained projection on the Sharer's cache for its own layer."""
    sharer = cache_tensors(sharer_cache)
    out = {}
    for target, source in enumerate(mapping):
        if source is None:
            continue
        keys, values = sharer[source]
        batch = keys.shape[0]
        made = []
        for kind, tensor in (("keys", keys), ("values", values)):
            flat = projections[(target, kind)].net(flatten_heads(tensor))
            made.append(unflatten_heads(flat, batch, heads, head_dim))
        out[target] = tuple(made)
    return out


def constant_cache(receiver_cache, mapping, means, heads, head_dim) -> dict:
    """What a projection that learned nothing would emit.

    The per channel mean of the target is the best constant predictor, and it
    is exactly the value the geometric null prices. Putting it through the
    same injection path turns the null from a number the loss is compared
    against into a condition the Receiver is actually run on.
    """
    positions = cache_tensors(receiver_cache)[0][0].shape[2]
    out = {}
    for target, source in enumerate(mapping):
        if source is None:
            continue
        made = []
        for kind in KINDS:
            flat = means[(target, kind)].unsqueeze(0).expand(positions, -1)
            made.append(unflatten_heads(flat.contiguous(), 1, heads, head_dim))
        out[target] = tuple(made)
    return out


@torch.no_grad()
def measure_injection(projections, means, mapping, contract) -> dict:
    """Replace the Receiver's cache and see whether its output moves.

    Four conditions on one prefix. The no-op says the injection path is
    wired. The corrupted case is the Tier 1 control, repeated here so the
    threshold is read in the same units it was set in. The constant case is
    the null run as a condition rather than quoted as a number, and the
    distance between it and the projected case is the only thing here that
    says whether the training changed anything the Receiver can feel.
    """
    from transformers import AutoModelForCausalLM

    ids = json.loads(HELD_OUT_IDS_PATH.read_text(encoding="utf-8"))
    chunk = ids["chunks"][INJECTION_CHUNK]
    input_ids = torch.tensor([chunk], dtype=torch.long, device=INJECTION_DEVICE)
    n_prefix = int(len(chunk) * INJECTION_PREFIX_FRACTION)
    prefix, suffix = input_ids[:, :n_prefix], input_ids[:, n_prefix:]
    n_new = int(suffix.shape[1])
    positions = absolute_position_ids(n_prefix, n_new, device=INJECTION_DEVICE)

    sharer = AutoModelForCausalLM.from_pretrained(
        SHARER_ID, dtype=INJECTION_DTYPE,
        attn_implementation=ATTN_IMPLEMENTATION,
    ).to(INJECTION_DEVICE).eval()
    sharer_cache = clone_cache(_forward(sharer, prefix).past_key_values)
    del sharer

    receiver = AutoModelForCausalLM.from_pretrained(
        RECEIVER_ID, dtype=INJECTION_DTYPE,
        attn_implementation=ATTN_IMPLEMENTATION,
    ).to(INJECTION_DEVICE).eval()
    receiver_cache = clone_cache(_forward(receiver, prefix).past_key_values)

    heads = contract["receiver"]["n_kv_heads"]
    head_dim = contract["receiver"]["head_dim"]

    reference = _forward(
        receiver, suffix, clone_cache(receiver_cache), positions
    ).logits
    logit_scale = float(reference.abs().max())

    conditions = {}

    conditions["noop"] = max_abs_difference(
        _forward(
            receiver, suffix,
            substitute(receiver_cache, {}), positions,
        ).logits,
        reference,
    )
    conditions["constant"] = max_abs_difference(
        _forward(
            receiver, suffix,
            substitute(receiver_cache, constant_cache(
                receiver_cache, mapping, means, heads, head_dim)),
            positions,
        ).logits,
        reference,
    )
    conditions["projected"] = max_abs_difference(
        _forward(
            receiver, suffix,
            substitute(receiver_cache, project_cache(
                sharer_cache, mapping, projections, heads, head_dim)),
            positions,
        ).logits,
        reference,
    )
    conditions["corrupted"] = max_abs_difference(
        _forward(
            receiver, suffix,
            corrupt_norm_matched(
                clone_cache(receiver_cache),
                torch.Generator(device=INJECTION_DEVICE).manual_seed(CORRUPTION_SEED),
            ),
            positions,
        ).logits,
        reference,
    )
    del receiver

    ratios = {name: value / logit_scale for name, value in conditions.items()}
    return {
        "conditions": conditions,
        "ratios": ratios,
        "logit_scale": logit_scale,
        "threshold": MIN_DEGRADATION_RATIO,
        "clears_threshold": ratios["projected"] > MIN_DEGRADATION_RATIO,
        "projected_over_constant": (
            ratios["projected"] / ratios["constant"]
            if ratios["constant"] > 0 else None
        ),
        "projected_over_corrupted": (
            ratios["projected"] / ratios["corrupted"]
            if ratios["corrupted"] > 0 else None
        ),
        "sequence": {
            "chunk": INJECTION_CHUNK, "n_total": len(chunk),
            "n_prefix": n_prefix, "n_new": n_new,
        },
        "replaced_target_layers": [t for t, s in enumerate(mapping) if s is not None],
        "kept_target_layers": [t for t, s in enumerate(mapping) if s is None],
        "source": "results/held_out_ids.json",
    }


def main() -> None:
    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)

    null = load_json(NULL_PATH)
    contract = load_json(CONTRACTS_PATH)
    splits = require_splits(null)

    n_source = contract["sharer"]["n_layers"]
    n_target = contract["receiver"]["n_layers"]
    mapping = align_layers(n_source, n_target, strategy="terminal")

    records = []
    projections, means = {}, {}
    for target, source in enumerate(mapping):
        if source is None:
            continue
        for kind in KINDS:
            x = {s: load_layer(s, "sharer", kind, source) for s in SPLITS}
            y = {s: load_layer(s, "receiver", kind, target) for s in SPLITS}
            nulls = {
                s: splits[s]["receiver"][kind][target]["null_mse_per_channel_mean"]
                for s in SPLITS
            }
            record, projection = train_one(x, y, nulls, generator)
            record.update(target_layer=target, source_layer=source, kind=kind)
            records.append(record)
            projections[(target, kind)] = projection
            # The best constant predictor of this layer, taken from the split
            # that grades. It is the null, kept as a tensor so it can be run
            # through the injection path instead of only quoted.
            means[(target, kind)] = y["held_out"].mean(dim=0)
            del x, y

    summary = {}
    for kind in KINDS:
        subset = [r for r in records if r["kind"] == kind]
        summary[kind] = {
            "n_layers": len(subset),
            "worst_relative_held_out": max(r["relative_held_out"] for r in subset),
            "mean_relative_held_out": sum(
                r["relative_held_out"] for r in subset
            ) / len(subset),
            "layers_above_null": [
                r["target_layer"] for r in subset
                if r["relative_held_out"] >= MAX_RELATIVE_HELD_OUT
            ],
            "min_positions_per_parameter": min(
                r["positions_per_parameter"] for r in subset
            ),
            "median_selected_epoch": sorted(
                r["selected_epoch"] for r in subset
            )[len(subset) // 2],
        }

    injection = measure_injection(projections, means, mapping, contract)

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(
        json.dumps(
            {
                "config": {
                    "depth": DEPTH, "hidden": HIDDEN,
                    "activation": ACTIVATION if DEPTH > 1 else None,
                    "epochs": EPOCHS, "batch_size": BATCH_SIZE,
                    "learning_rate": LEARNING_RATE,
                    "weight_decay": WEIGHT_DECAY, "seed": SEED,
                    "max_relative_held_out": MAX_RELATIVE_HELD_OUT,
                    "min_positions_per_parameter": MIN_POSITIONS_PER_PARAMETER,
                    "patience": PATIENCE,
                    "selection_split": "validation",
                    "graded_split": "held_out",
                },
                "per_layer": records,
                "summary": summary,
                "injection": injection,
                "torch_version": torch.__version__,
                "torch_num_threads": torch.get_num_threads(),
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()