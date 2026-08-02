"""Both caches, and the number every training loss will be read against.

Silent on success. Everything goes into results/geometric_null.json.

What the null is
----------------
The best constant predictor of the target, and the mean squared error it
achieves. A projection carries a bias, so a bias alone can reach the target's
per-channel mean without learning anything about the input. That is the floor
a falling loss curve has to clear before it means something, and it is the
only quantity here that is commensurable with the training loss.

Three constant predictors are recorded, in increasing strictness: predicting
zero, predicting one global mean, and predicting a mean per channel. The last
is the null. The other two cost nothing to record and make the choice
auditable rather than asserted.

What the null is not
--------------------
A distance between the Sharer cache and the Receiver cache. Those live at
widths 128 and 1024, so any number claiming to be their separation has an
arbitrary embedding hidden inside it, and that arbitrary choice would end up
grading every result in the tier above. In its place this file records the
per-layer scale of both caches directly, which is a comparison that does not
require inventing a shared space to live in.

What the target is
------------------
The Receiver's own cache, read from `past_key_values`. That is already the
post-norm post-rotary tensor: keys enter it after `k_norm` and after rotary
embedding. Targeting the right thing is therefore a property of reading the
cache rather than something to be careful about, which is the opposite of
what a hook on `k_proj` would give.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from c2c.alignment import align_layers  # noqa: E402
from c2c.cache_ops import cache_tensors, flatten_heads  # noqa: E402

SHARER_ID = "Qwen/Qwen2.5-0.5B-Instruct"
RECEIVER_ID = "Qwen/Qwen3-0.6B"

DTYPE = torch.float32
DEVICE = "cpu"
ATTN_IMPLEMENTATION = "eager"

# Plain text only. The Tier 0 tokenizer probe found the two vocabularies
# agree on ordinary strings and differ in size by four added tokens, which
# are the Receiver's own. A chat template or a thinking tag reaching the
# Sharer would tokenize differently on the two sides and the positions would
# stop corresponding, without anything raising.
CORPUS = [
    "A key-value cache stores, for every layer and every position, the key "
    "and value vectors attention has already computed.",
    "Rotary position embedding rotates each pair of channels by an angle "
    "proportional to the position, so a key carries where it was computed.",
    "Values are never rotated. They leave the projection carrying no "
    "position information at all, and the cache holds them unchanged.",
    "Grouped query attention gives several query heads a single key-value "
    "head, so the cache is narrower than the attention that consumes it.",
    "The smallest model in a family is often the most revealing, because "
    "every mechanism is present and nothing is large enough to hide behind.",
    "A loss that falls tells you the optimizer is working. It does not tell "
    "you the model learned the thing you meant it to learn.",
    "Two measurements taken under different conditions are only comparable "
    "when the conditions differ in exactly one respect.",
    "Normalization applied to queries and keys before attention bounds the "
    "scale of the logits without changing which token attends to which.",
    "An absence drawn as a gap can be seen. An absence drawn as a shorter "
    "row looks like a drafting choice and disappears.",
    "Floating point addition is not associative, so two orderings of the "
    "same sum can disagree in the last bits and both be correct.",
    "The cheapest experiment that could change your mind is usually worth "
    "running before the expensive one that could only confirm it.",
    "A number without the conditions it was measured under is not a "
    "measurement, it is a rumour with a decimal point.",
    "Depth changes what a representation carries. Early layers hold the "
    "shape of the input, later ones hold what the model intends to say.",
    "Transferring a representation between two models assumes there is "
    "something in it that does not depend on the model that produced it.",
    "The point of a control is not to succeed. It is to fail in a way that "
    "proves the main condition could have failed too.",
    "Writing the prediction down before the run is what turns a result into "
    "evidence rather than a story assembled afterwards.",
]

RESULTS_PATH = REPO_ROOT / "results" / "geometric_null.json"
CONTRACTS_PATH = REPO_ROOT / "results" / "contracts.json"


class ChannelStats:
    """Streaming per-channel mean and variance, accumulated in float64.

    Streaming rather than storing: the whole corpus of caches at both models
    is several hundred megabytes, and none of it is needed twice. float64
    rather than the model dtype: a sum of squares in float32 loses the
    variance to cancellation once the mean is large relative to the spread,
    which for post-norm keys is exactly the regime.
    """

    def __init__(self, n_channels: int):
        self.n_channels = n_channels
        self.count = 0
        self.total = torch.zeros(n_channels, dtype=torch.float64)
        self.total_sq = torch.zeros(n_channels, dtype=torch.float64)

    def add(self, tensor: torch.Tensor) -> None:
        """Take one [batch, heads, positions, head_dim] tensor.

        The channel axis is heads times head_dim, which is the vector a
        projection has to produce for one position, and matches the
        `kv_width` the contract records.
        """
        flat = flatten_heads(tensor).to(torch.float64)
        if flat.shape[1] != self.n_channels:
            raise ValueError(
                f"expected {self.n_channels} channels, got {flat.shape[1]}"
            )
        self.count += flat.shape[0]
        self.total += flat.sum(dim=0)
        self.total_sq += (flat * flat).sum(dim=0)

    def report(self) -> dict:
        if self.count < 2:
            raise ValueError(f"{self.count} positions is not enough for a variance")
        n, c = self.count, self.n_channels
        mean = self.total / n
        second = self.total_sq / n
        per_channel_var = (second - mean * mean).clamp_min(0.0)

        global_mean = float(self.total.sum() / (n * c))
        mean_square = float(self.total_sq.sum() / (n * c))

        return {
            "n_positions": n,
            "n_channels": c,
            "null_mse_per_channel_mean": float(per_channel_var.mean()),
            "null_mse_global_mean": max(mean_square - global_mean**2, 0.0),
            "null_mse_zero": mean_square,
            "rms": mean_square**0.5,
            "channel_var_min": float(per_channel_var.min()),
            "channel_var_max": float(per_channel_var.max()),
        }


def compare_tokenizations(sharer_ids, receiver_ids) -> dict:
    """Both sides must see the same tokens or the positions stop corresponding.

    Pure, so that this check is testable without either tokenizer on disk.
    """
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
        "mismatches": mismatches,
    }


@torch.no_grad()
def measure_geometry(sharer_model, receiver_model, id_tensors) -> dict:
    """Prefill both models on the same tokens and accumulate cache statistics.

    Takes models rather than loading them, so the whole measurement can be
    rehearsed against models built from config objects in a second, with no
    checkpoint on disk.
    """
    if not id_tensors:
        raise ValueError("no passages to measure")

    stats: dict[str, dict[str, list[ChannelStats]]] = {}
    shapes: dict[str, list] = {}

    for role, model in (("sharer", sharer_model), ("receiver", receiver_model)):
        config = model.config
        head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        width = config.num_key_value_heads * head_dim
        stats[role] = {
            kind: [ChannelStats(width) for _ in range(config.num_hidden_layers)]
            for kind in ("keys", "values")
        }

    for ids in id_tensors:
        for role, model in (("sharer", sharer_model), ("receiver", receiver_model)):
            cache = model(ids, use_cache=True).past_key_values
            pairs = cache_tensors(cache)
            shapes[role] = [len(pairs), *list(pairs[0][0].shape)]
            for layer_idx, (keys, values) in enumerate(pairs):
                stats[role]["keys"][layer_idx].add(keys)
                stats[role]["values"][layer_idx].add(values)
            del cache, pairs

    per_layer = {
        role: {
            kind: [s.report() for s in series]
            for kind, series in kinds.items()
        }
        for role, kinds in stats.items()
    }

    n_source = sharer_model.config.num_hidden_layers
    n_target = receiver_model.config.num_hidden_layers
    mapping = align_layers(n_source, n_target, strategy="terminal")
    paired = [t for t, s in enumerate(mapping) if s is not None]

    def aggregate(layers):
        """Element-weighted mean, so a layer does not count twice for being
        wider than another."""
        out = {}
        for kind in ("keys", "values"):
            entries = [per_layer["receiver"][kind][t] for t in layers]
            weight = sum(e["n_positions"] * e["n_channels"] for e in entries)
            out[kind] = sum(
                e["null_mse_per_channel_mean"] * e["n_positions"] * e["n_channels"]
                for e in entries
            ) / weight
        return out

    return {
        "per_layer": per_layer,
        "null": {
            "definition": "mean squared error of the best constant predictor, "
                          "one mean per channel, channels are n_kv_heads x head_dim",
            "aggregate_all_target_layers": aggregate(range(n_target)),
            "aggregate_paired_target_layers": aggregate(paired),
            "paired_target_layers": paired,
            "unpaired_target_layers": [
                t for t, s in enumerate(mapping) if s is None
            ],
        },
        "cache_shapes": shapes,
        "layer_counts": {"sharer": n_source, "receiver": n_target},
    }


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sharer_tokenizer = AutoTokenizer.from_pretrained(SHARER_ID)
    receiver_tokenizer = AutoTokenizer.from_pretrained(RECEIVER_ID)

    sharer_ids = [
        sharer_tokenizer(text, add_special_tokens=False)["input_ids"]
        for text in CORPUS
    ]
    receiver_ids = [
        receiver_tokenizer(text, add_special_tokens=False)["input_ids"]
        for text in CORPUS
    ]
    tokenization = compare_tokenizations(sharer_ids, receiver_ids)

    id_tensors = [
        torch.tensor([ids], dtype=torch.long, device=DEVICE)
        for ids in receiver_ids
    ]

    sharer = AutoModelForCausalLM.from_pretrained(
        SHARER_ID, dtype=DTYPE, attn_implementation=ATTN_IMPLEMENTATION
    ).to(DEVICE).eval()
    receiver = AutoModelForCausalLM.from_pretrained(
        RECEIVER_ID, dtype=DTYPE, attn_implementation=ATTN_IMPLEMENTATION
    ).to(DEVICE).eval()

    result = measure_geometry(sharer, receiver, id_tensors)
    result["tokenization"] = tokenization
    result["corpus"] = {
        "n_passages": len(CORPUS),
        "n_tokens": tokenization["n_tokens"],
        "source": "scripts/capture_dual_cache.py CORPUS",
    }
    result["model_ids"] = {"sharer": SHARER_ID, "receiver": RECEIVER_ID}
    result["dtype"] = str(DTYPE)
    result["device"] = DEVICE
    result["attn_implementation"] = ATTN_IMPLEMENTATION

    import transformers

    result["transformers_version"] = transformers.__version__
    result["torch_version"] = torch.__version__
    result["torch_num_threads"] = torch.get_num_threads()

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()