"""Both caches, split by article, and the number every training loss is read against.

Silent on success. Writes results/geometric_null.json and the cache tensors
under results/caches/, which are gitignored.

What the null is
----------------
The best constant predictor of the target, and the mean squared error it
achieves. A projection carries a bias, so a bias alone reaches the target's
per channel mean without learning anything about the input. Three constant
predictors are recorded, in increasing strictness: predicting zero,
predicting one global mean, and predicting a mean per channel. The last is
the null.

Why there are two splits
------------------------
Because the comparison the null exists for is only valid out of sample.
Measured on uninformative input, an affine map from 128 channels to 1024
reaches an in sample relative loss of 0.668 at 387 positions and 0.996 at
32000. Both are below one. No corpus is large enough to make an in sample
comparison mean anything, so the file carries a held out split and the gate
above reads that one.

The split is by article. Chunks from one article never straddle the boundary,
because two passages from the same article share topic and vocabulary and a
held out number computed across that boundary is optimistic for a reason
unrelated to the projection.

What the null is not
--------------------
A distance between the Sharer cache and the Receiver cache. Those live at
widths 128 and 1024, so any number claiming to be their separation has an
arbitrary embedding inside it, and that choice would grade every result in
the tier above. In its place this file records the per layer scale of both
caches directly.

What the target is
------------------
The Receiver's own cache, read from `past_key_values`. That is already the
post-norm post-rotary tensor: keys enter it after `k_norm` and after rotary
embedding. Targeting the right thing is a property of reading the cache
rather than something to be careful about.

Two phases
----------
Tensors are written first, and the statistics are computed from the files
that were written. Computing them from a separate pass over the same forward
would leave two numbers that can disagree without anything raising, and the
one the training reads would not be the one the gate checked.
"""

from __future__ import annotations

import json
import platform
import re
import shutil
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from c2c.alignment import align_layers  # noqa: E402
from c2c.cache_ops import cache_tensors, flatten_heads  # noqa: E402

SHARER_ID = "Qwen/Qwen2.5-0.5B-Instruct"
RECEIVER_ID = "Qwen/Qwen3-0.6B"

# The namespace is not optional. Recent huggingface_hub rejects a bare
# canonical name and requires namespace/name, and wikitext now lives under
# Salesforce. A dataset id without a namespace resolves to nothing rather
# than to something wrong, which is the better of the two failures.
DATASET = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
DATASET_SPLIT = "train"

# 16000 train positions is 55 per parameter for the two layer projection at
# hidden 256, which costs 289 degrees of freedom per output channel. Below
# roughly 50 a held out failure cannot be told apart from not having enough
# data to find the signal.
TRAIN_TOKEN_BUDGET = 16000
HELD_OUT_TOKEN_BUDGET = 6000

# Every chunk is prefilled on its own, so positions restart at zero and the
# models never see a position beyond this. Keys carry position, so a
# projection trained here has only been shown this range. That is a real
# limit on what the trained projection can be claimed to cover, and it is
# recorded rather than assumed away.
MAX_SEQUENCE_TOKENS = 256
MIN_SEQUENCE_TOKENS = 32

# float16 halves the several gigabytes this writes. At the smallest null in
# the file, 0.0097, the quantisation variance is around 3e-10, nine orders
# below it. The statistics are computed from these stored tensors, so the
# number the gate checks and the data the training reads are the same bits.
STORAGE_DTYPE = torch.float16

DTYPE = torch.float32
DEVICE = "cpu"
ATTN_IMPLEMENTATION = "eager"
SPLIT_SEED = 1234
FLUSH_EVERY_CHUNKS = 8

SPLITS = ("train", "held_out")
KINDS = ("keys", "values")
ROLES = ("sharer", "receiver")

RESULTS_PATH = REPO_ROOT / "results" / "geometric_null.json"
CACHE_ROOT = REPO_ROOT / "results" / "caches"

ARTICLE_TITLE = re.compile(r"^ = [^=].* = $")


# -------------------------------------------------------------------------
# corpus, pure


def iter_articles(lines):
    """Group wikitext lines into articles.

    A title line is ` = Something = `. A section heading has two or more
    equals signs on each side and stays inside its article.
    """
    current: list[str] = []
    for line in lines:
        if ARTICLE_TITLE.match(line.rstrip("\n")):
            if current:
                yield "".join(current)
            current = [line]
        else:
            current.append(line)
    if current:
        yield "".join(current)


def chunk_ids(ids, max_tokens: int, min_tokens: int) -> list[list[int]]:
    """Cut one article into prefill sized pieces, dropping the short tail.

    The tail is dropped rather than padded because a padded chunk would carry
    positions the model never assigned to real tokens, and those keys would
    enter the statistics as if they were content.
    """
    chunks = [
        ids[start:start + max_tokens]
        for start in range(0, len(ids), max_tokens)
    ]
    return [c for c in chunks if len(c) >= min_tokens]


def assign_splits(articles, budgets, seed: int) -> dict:
    """Fill each budget with whole articles, in a shuffled but fixed order.

    `articles` is a sequence of (index, chunks). Assignment is by article so
    that no chunk of a training article appears in the held out set.
    """
    order = torch.randperm(
        len(articles), generator=torch.Generator().manual_seed(seed)
    ).tolist()

    assigned = {name: {"articles": [], "chunks": [], "n_tokens": 0}
                for name in budgets}
    remaining = dict(budgets)
    for position in order:
        index, chunks = articles[position]
        target = next(
            (name for name in budgets if remaining[name] > 0), None
        )
        if target is None:
            break
        assigned[target]["articles"].append(index)
        assigned[target]["chunks"].extend(chunks)
        tokens = sum(len(c) for c in chunks)
        assigned[target]["n_tokens"] += tokens
        remaining[target] -= tokens

    short = [name for name, need in remaining.items() if need > 0]
    if short:
        raise ValueError(
            f"the dataset ran out before filling {short}; lower the budgets "
            "or use a larger configuration"
        )
    return assigned


def compare_tokenizations(sharer_ids, receiver_ids) -> dict:
    """Both sides must see the same tokens or the positions stop corresponding."""
    if len(sharer_ids) != len(receiver_ids):
        raise ValueError("the two tokenizers produced different passage counts")
    mismatches = [
        {"index": i, "sharer_len": len(a), "receiver_len": len(b)}
        for i, (a, b) in enumerate(zip(sharer_ids, receiver_ids))
        if list(a) != list(b)
    ]
    return {
        "n_passages": len(sharer_ids),
        "n_tokens": sum(len(x) for x in receiver_ids),
        "identical": not mismatches,
        "mismatches": mismatches[:8],
    }


# -------------------------------------------------------------------------
# statistics, pure


def channel_report(stacked: torch.Tensor) -> dict:
    """Per channel variance of one layer's stored tensor.

    Accumulated in float64. A sum of squares in the storage dtype loses the
    variance to cancellation once the mean is large relative to the spread,
    which for post-norm keys is exactly the regime.
    """
    if stacked.dim() != 2:
        raise ValueError(f"expected [positions, channels], got {tuple(stacked.shape)}")
    n, channels = stacked.shape
    if n < 2:
        raise ValueError(f"{n} positions is not enough for a variance")

    wide = stacked.to(torch.float64)
    mean = wide.mean(dim=0)
    per_channel_var = (wide.var(dim=0, unbiased=False)).clamp_min(0.0)
    mean_square = float((wide * wide).mean())
    global_mean = float(mean.mean())

    return {
        "n_positions": int(n),
        "n_channels": int(channels),
        "null_mse_per_channel_mean": float(per_channel_var.mean()),
        "null_mse_global_mean": max(mean_square - global_mean**2, 0.0),
        "null_mse_zero": mean_square,
        "rms": mean_square**0.5,
        "channel_var_min": float(per_channel_var.min()),
        "channel_var_max": float(per_channel_var.max()),
    }


# -------------------------------------------------------------------------
# capture


def part_path(split: str, role: str, kind: str, layer: int, part: int) -> Path:
    return (CACHE_ROOT / split / "_parts"
            / f"{role}_{kind}_layer{layer:02d}_part{part:04d}.pt")


def final_path(split: str, role: str, kind: str, layer: int) -> Path:
    return CACHE_ROOT / split / f"{role}_{kind}_layer{layer:02d}.pt"


@torch.no_grad()
def capture_role(model, role: str, split: str, chunks) -> list:
    """Prefill every chunk and write per layer part files.

    Parts exist so that the resident set stays at a handful of chunks rather
    than the whole split. They are concatenated and deleted in the next
    phase.
    """
    n_layers = model.config.num_hidden_layers
    buffers = {kind: [[] for _ in range(n_layers)] for kind in KINDS}
    part, shape = 0, None

    def flush(index: int) -> None:
        for kind in KINDS:
            for layer in range(n_layers):
                if not buffers[kind][layer]:
                    continue
                path = part_path(split, role, kind, layer, index)
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(torch.cat(buffers[kind][layer], dim=0), path)
                buffers[kind][layer] = []

    for position, ids in enumerate(chunks):
        tensor = torch.tensor([ids], dtype=torch.long, device=DEVICE)
        pairs = cache_tensors(model(tensor, use_cache=True).past_key_values)
        if shape is None:
            shape = [len(pairs), *list(pairs[0][0].shape)]
        for layer, (keys, values) in enumerate(pairs):
            buffers["keys"][layer].append(
                flatten_heads(keys).to(STORAGE_DTYPE))
            buffers["values"][layer].append(
                flatten_heads(values).to(STORAGE_DTYPE))
        del pairs
        if (position + 1) % FLUSH_EVERY_CHUNKS == 0:
            flush(part)
            part += 1
    flush(part)
    return shape


def finalise(split: str, role: str, n_layers: int) -> dict:
    """Concatenate the parts of each layer, then read the statistics off the
    file that was actually written."""
    reports = {kind: [] for kind in KINDS}
    for kind in KINDS:
        for layer in range(n_layers):
            parts = sorted(
                (CACHE_ROOT / split / "_parts").glob(
                    f"{role}_{kind}_layer{layer:02d}_part*.pt"
                )
            )
            if not parts:
                raise ValueError(f"no parts for {split}/{role}/{kind}/{layer}")
            stacked = torch.cat([torch.load(p, map_location="cpu") for p in parts])
            path = final_path(split, role, kind, layer)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(stacked, path)
            reports[kind].append(channel_report(stacked))
            del stacked
    return reports


def aggregate(reports, layers) -> dict:
    """Element weighted, so a layer does not count twice for being wider."""
    out = {}
    for kind in KINDS:
        entries = [reports[kind][t] for t in layers]
        weight = sum(e["n_positions"] * e["n_channels"] for e in entries)
        out[kind] = sum(
            e["null_mse_per_channel_mean"] * e["n_positions"] * e["n_channels"]
            for e in entries
        ) / weight
    return out


def main() -> None:
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sharer_tokenizer = AutoTokenizer.from_pretrained(SHARER_ID)
    receiver_tokenizer = AutoTokenizer.from_pretrained(RECEIVER_ID)

    dataset = load_dataset(DATASET, DATASET_CONFIG, split=DATASET_SPLIT)
    rows = dataset["text"]
    try:
        dataset_info = {
            "config_name": dataset.info.config_name,
            "version": str(dataset.info.version),
            "n_rows": dataset.num_rows,
        }
    except Exception as exc:  # noqa: BLE001
        dataset_info = {"recorded": False, "reason": f"{type(exc).__name__}: {exc}"}
    articles = []
    for index, text in enumerate(iter_articles(rows)):
        if not text.strip():
            continue
        ids = receiver_tokenizer(text, add_special_tokens=False)["input_ids"]
        chunks = chunk_ids(ids, MAX_SEQUENCE_TOKENS, MIN_SEQUENCE_TOKENS)
        if chunks:
            articles.append((index, chunks))

    assigned = assign_splits(
        articles,
        {"train": TRAIN_TOKEN_BUDGET, "held_out": HELD_OUT_TOKEN_BUDGET},
        SPLIT_SEED,
    )

    sample = [c for split in SPLITS for c in assigned[split]["chunks"][:4]]
    tokenization = compare_tokenizations(
        [sharer_tokenizer(receiver_tokenizer.decode(c),
                          add_special_tokens=False)["input_ids"] for c in sample],
        [receiver_tokenizer(receiver_tokenizer.decode(c),
                            add_special_tokens=False)["input_ids"] for c in sample],
    )

    shutil.rmtree(CACHE_ROOT, ignore_errors=True)
    splits, shapes = {}, {}
    for role, model_id in (("sharer", SHARER_ID), ("receiver", RECEIVER_ID)):
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=DTYPE, attn_implementation=ATTN_IMPLEMENTATION
        ).to(DEVICE).eval()
        n_layers = model.config.num_hidden_layers
        for split in SPLITS:
            shapes[role] = capture_role(
                model, role, split, assigned[split]["chunks"]
            )
            splits.setdefault(split, {})[role] = finalise(split, role, n_layers)
        del model
    for split in SPLITS:
        shutil.rmtree(CACHE_ROOT / split / "_parts", ignore_errors=True)
        splits[split]["n_positions"] = splits[split]["receiver"]["keys"][0]["n_positions"]
        splits[split]["n_chunks"] = len(assigned[split]["chunks"])
        splits[split]["n_articles"] = len(assigned[split]["articles"])

    n_source = shapes["sharer"][0]
    n_target = shapes["receiver"][0]
    mapping = align_layers(n_source, n_target, strategy="terminal")
    paired = [t for t, s in enumerate(mapping) if s is not None]

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps({
        "splits": splits,
        "null": {
            "definition": "mean squared error of the best constant predictor, "
                          "one mean per channel, channels are n_kv_heads x head_dim",
            "graded_on": "held_out",
            "aggregate_all_target_layers": aggregate(
                splits["held_out"]["receiver"], range(n_target)),
            "aggregate_paired_target_layers": aggregate(
                splits["held_out"]["receiver"], paired),
            "paired_target_layers": paired,
            "unpaired_target_layers": [t for t, s in enumerate(mapping) if s is None],
        },
        "cache_shapes": shapes,
        "layer_counts": {"sharer": n_source, "receiver": n_target},
        "tokenization": tokenization,
        "corpus": {
            "dataset": DATASET, "config": DATASET_CONFIG, "split": DATASET_SPLIT,
            "train_token_budget": TRAIN_TOKEN_BUDGET,
            "held_out_token_budget": HELD_OUT_TOKEN_BUDGET,
            "max_sequence_tokens": MAX_SEQUENCE_TOKENS,
            "min_sequence_tokens": MIN_SEQUENCE_TOKENS,
            "split_by": "article", "split_seed": SPLIT_SEED,
            "positions_seen": f"0 to {MAX_SEQUENCE_TOKENS - 1}",
            "dataset_info": dataset_info,
        },
        "storage_dtype": str(STORAGE_DTYPE),
        "model_ids": {"sharer": SHARER_ID, "receiver": RECEIVER_ID},
        "dtype": str(DTYPE), "device": DEVICE,
        "attn_implementation": ATTN_IMPLEMENTATION,
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        # The interpreter is recorded because the repository declares one and
        # the environment can be running another. A declared version that is
        # not the version in use is the same failure as a config attribute
        # that is present and no longer carries the value.
        "python_version": platform.python_version(),
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()