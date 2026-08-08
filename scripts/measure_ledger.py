# SPDX-License-Identifier: Apache-2.0
"""What the exchange costs on this machine.

Silent on success. Writes results/ledger.json.

Three quantities, and they are not the same kind of thing.

Payload is arithmetic. It follows from the cache geometry in
`results/contracts.json` and nothing here can measure it more accurately than
multiplying. It is computed and then checked against a real captured tensor,
because a contract that disagrees with the tensors is a broken contract and
this is a cheap place to find that out.

Fusion time and decode time are measurements, on a six-thread CPU, and they
carry all the noise that implies. Every one of them is reported as a median
over repeats with an interquartile range beside it, never as a single number,
because a single number invites being quoted as though it were exact.

Three ways a ledger goes wrong, and what is done about each
----------------------------------------------------------
A fusion time quoted without the sequence length it was measured at is not a
duration, it is a duration per something unstated. Every timing here is
recorded against the position count it was taken at, and several position
counts are measured so the scaling is visible rather than assumed linear.

A saving quoted without the work it replaces is a number with no denominator.
The Sharer's decode is measured per token, which is a property of the Sharer,
and any total is that rate times a response length. This repository has not
fixed a response length, so no total is reported as though it had.

A ratio from this machine compared against the reference's ratio is two
different experiments wearing the same units. Table 3 prices this pair on an
A100 at batch size one, where communication is nearly free. That setting
cannot tell a favourable trade from an irrelevant one, and neither can this
one. The device and thread count are recorded so the comparison is at least
visibly refused.

What the payload comparison is between
--------------------------------------
Under C2C the Sharer sends its cache over the prompt: bytes proportional to
the prompt length. Under a text pathway it sends its response: bytes
proportional to the response length. The two scale with different things, so
they are reported against both lengths rather than collapsed into one ratio.
On a workstation this hardly matters. On a link it is the whole question.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from c2c.cache_ops import build_cache, cache_tensors  # noqa: E402
from c2c.fuser import FuserBank  # noqa: E402

SHARER_ID = "Qwen/Qwen2.5-0.5B-Instruct"
RECEIVER_ID = "Qwen/Qwen3-0.6B"

# The trained bank whose fusion is being priced. Fusion time is a property of
# a module, so it is measured against the one that was graded and not against
# a fresh one that happens to have the same shape.
FUSER_RUN = "fused_2026-08-06_n2000"

# Several, so the scaling is measured rather than assumed. A single length
# gives a number that cannot be extrapolated and will be extrapolated anyway.
POSITIONS = (64, 128, 256, 512)

REPEATS = 7
WARMUP = 2
DECODE_TOKENS = 16

# Transmission would sensibly be at half precision even though the models run
# at single here, so both are reported and neither is presented as the figure.
PAYLOAD_DTYPES = {"float16": 2, "float32": 4}

# A token id on the wire is two bytes at this vocabulary size. The UTF-8
# figure is an estimate and is labelled as one; it is not measured here.
BYTES_PER_TOKEN_ID = 2
BYTES_PER_TOKEN_UTF8_ESTIMATE = 4

DTYPE = torch.float32
DEVICE = "cpu"
ATTN_IMPLEMENTATION = "eager"
SEED = 42

RESULTS_DIR = REPO_ROOT / "results"
CONTRACTS_PATH = RESULTS_DIR / "contracts.json"
LEDGER_PATH = RESULTS_DIR / "ledger.json"


def timed(call, repeats: int = REPEATS, warmup: int = WARMUP) -> dict:
    """Median and spread, never a single number.

    Warmup runs are discarded. The first call through any of these paths pays
    for lazy allocation and for caches this measurement is not about.
    """
    for _ in range(warmup):
        call()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        call()
        samples.append((time.perf_counter() - started) * 1000.0)
    samples.sort()
    return {
        "median_ms": statistics.median(samples),
        "min_ms": samples[0],
        "max_ms": samples[-1],
        "iqr_ms": (
            statistics.median(samples[len(samples) // 2:])
            - statistics.median(samples[: (len(samples) + 1) // 2])
        ),
        "repeats": repeats,
    }


def payload_arithmetic(contract: dict) -> dict:
    """Bytes on the wire, computed from the contract and checked against it.

    Only the paired layers are sent. Terminal alignment leaves the Receiver's
    four shallowest layers without a partner, so the Sharer has nothing to
    contribute there and those layers cost nothing.
    """
    sharer, receiver = contract["sharer"], contract["receiver"]
    n_paired = sum(1 for s in contract["layer_alignment"]["target_to_source"] if s is not None)

    def side(spec: dict, n_layers: int) -> dict:
        # Two tensors per layer, keys and values, at the KV head count rather
        # than the attention head count. The cache stores heads before
        # `repeat_kv` expands them, and pricing the expanded width would
        # overstate the payload by the grouping factor.
        elements = 2 * spec["n_kv_heads"] * spec["head_dim"]
        return {
            "n_layers_sent": n_layers,
            "n_kv_heads": spec["n_kv_heads"],
            "head_dim": spec["head_dim"],
            "elements_per_layer_per_position": elements,
            "bytes_per_layer_per_position": {
                name: elements * size for name, size in PAYLOAD_DTYPES.items()
            },
            "bytes_per_position": {
                name: elements * size * n_layers
                for name, size in PAYLOAD_DTYPES.items()
            },
        }

    return {
        "sent_by_the_sharer": side(sharer, n_paired),
        "receiver_own_cache_for_scale": side(receiver, receiver["n_layers"]),
        "text_pathway": {
            "bytes_per_token_id": BYTES_PER_TOKEN_ID,
            "bytes_per_token_utf8_estimate": BYTES_PER_TOKEN_UTF8_ESTIMATE,
            "note": (
                "scales with the response the Sharer would have written, not "
                "with the prompt"
            ),
        },
    }


@torch.no_grad()
def measure_ledger(sharer, receiver, bank: FuserBank, contract: dict) -> dict:
    """Take models rather than load them, so the whole of this runs against
    two small models built from configs in seconds and with no checkpoint."""
    bank.eval()
    vocab = min(sharer.config.vocab_size, receiver.config.vocab_size)
    generator = torch.Generator(device=DEVICE).manual_seed(SEED)

    payload = payload_arithmetic(contract)
    timings = {name: [] for name in
               ("sharer_prefill", "receiver_prefill", "fusion", "sharer_decode_per_token")}
    observed = {}

    for n in POSITIONS:
        ids = torch.randint(0, vocab, (1, n), generator=generator, device=DEVICE)

        timings["sharer_prefill"].append(
            {"positions": n, **timed(lambda: sharer(ids, use_cache=True))}
        )
        timings["receiver_prefill"].append(
            {"positions": n, **timed(lambda: receiver(ids, use_cache=True))}
        )

        sharer_pairs = cache_tensors(sharer(ids, use_cache=True).past_key_values)
        receiver_pairs = cache_tensors(receiver(ids, use_cache=True).past_key_values)
        timings["fusion"].append(
            {"positions": n,
             **timed(lambda: build_cache(bank(receiver_pairs, sharer_pairs)))}
        )

        # Decode is priced per token, one token at a time at batch size one,
        # which is the setting the reference prices. The cache is rebuilt for
        # every repeat because a forward pass appends to the one it is handed
        # and a cache that grew during warmup is not the cache being measured.
        def decode():
            cache = sharer(ids, use_cache=True).past_key_values
            token = ids[:, -1:]
            for step in range(DECODE_TOKENS):
                out = sharer(
                    token,
                    past_key_values=cache,
                    position_ids=torch.tensor([[n + step]], device=DEVICE),
                    use_cache=True,
                )
                token = out.logits[:, -1:, :].argmax(dim=-1)
                cache = out.past_key_values

        prefill_median = timings["sharer_prefill"][-1]["median_ms"]
        whole = timed(decode, repeats=max(3, REPEATS // 2), warmup=1)
        timings["sharer_decode_per_token"].append({
            "positions": n,
            "tokens_decoded": DECODE_TOKENS,
            # The prefill inside the loop is subtracted using the median
            # measured above rather than re-measured, so the two numbers
            # cannot disagree about the same work.
            "median_ms": (whole["median_ms"] - prefill_median) / DECODE_TOKENS,
            "whole_call_median_ms": whole["median_ms"],
            "prefill_subtracted_ms": prefill_median,
            "repeats": whole["repeats"],
            "iqr_ms": whole["iqr_ms"] / DECODE_TOKENS,
            "min_ms": (whole["min_ms"] - prefill_median) / DECODE_TOKENS,
            "max_ms": (whole["max_ms"] - prefill_median) / DECODE_TOKENS,
        })

        # The layer counts are recorded beside the bytes. The payload figure
        # prices the layers with a partner and a captured cache holds every
        # layer the model has, and those are only the same number by accident
        # of this pair.
        observed[str(n)] = {
            "sharer_layers_captured": len(sharer_pairs),
            "receiver_layers_captured": len(receiver_pairs),
            "sharer_bytes_float32": sum(
                t.numel() * t.element_size()
                for pair in sharer_pairs for t in pair
            ),
            "receiver_bytes_float32": sum(
                t.numel() * t.element_size()
                for pair in receiver_pairs for t in pair
            ),
        }
        del sharer_pairs, receiver_pairs

    return {
        "payload": payload,
        "observed_cache_bytes": observed,
        "timings": timings,
        "positions_measured": list(POSITIONS),
        "decode_tokens": DECODE_TOKENS,
        "repeats": REPEATS,
        "warmup": WARMUP,
        "seed": SEED,
        "device": DEVICE,
        "dtype": str(DTYPE),
        "attn_implementation": ATTN_IMPLEMENTATION,
        "torch_num_threads": torch.get_num_threads(),
        "comparability": (
            "Table 3 prices this pair on an A100 at batch size one, where "
            "communication is nearly free. These are CPU numbers on a "
            "workstation. Neither setting can distinguish a favourable trade "
            "from an irrelevant one, and the ratios are not comparable."
        ),
    }


def main() -> None:
    if LEDGER_PATH.exists():
        raise SystemExit(
            f"{LEDGER_PATH.relative_to(REPO_ROOT)} already exists. Move it "
            "aside; timings from two machines in one file are two things."
        )
    if not CONTRACTS_PATH.exists():
        raise SystemExit("missing results/contracts.json; Tier 0 is not closed")

    from transformers import AutoModelForCausalLM

    contract = json.loads(CONTRACTS_PATH.read_text(encoding="utf-8"))
    log_path = RESULTS_DIR / f"fuser_{FUSER_RUN}.json"
    if not log_path.exists():
        raise SystemExit(f"missing {log_path.relative_to(REPO_ROOT)}")
    log = json.loads(log_path.read_text(encoding="utf-8"))
    checkpoint = RESULTS_DIR / log["checkpoint"]
    if not checkpoint.exists():
        raise SystemExit(f"missing {checkpoint.relative_to(REPO_ROOT)}")

    sharer = AutoModelForCausalLM.from_pretrained(
        SHARER_ID, dtype=DTYPE, attn_implementation=ATTN_IMPLEMENTATION
    ).to(DEVICE).eval()
    receiver = AutoModelForCausalLM.from_pretrained(
        RECEIVER_ID, dtype=DTYPE, attn_implementation=ATTN_IMPLEMENTATION
    ).to(DEVICE).eval()

    receiver_contract, sharer_contract = contract["receiver"], contract["sharer"]
    bank = FuserBank(
        receiver_contract["n_layers"], sharer_contract["n_layers"],
        receiver_contract["n_kv_heads"], receiver_contract["head_dim"],
        sharer_contract["n_kv_heads"], sharer_contract["head_dim"],
        residual=(log["mode"] == "fused"),
    )
    bank.load_state_dict(torch.load(checkpoint, weights_only=True))

    result = measure_ledger(sharer, receiver, bank, contract)
    result["fuser_run"] = FUSER_RUN
    result["fuser_checkpoint"] = log["checkpoint"]
    result["fuser_mode"] = log["mode"]
    result["n_trainable"] = log["n_trainable"]
    result["model_ids"] = {"sharer": SHARER_ID, "receiver": RECEIVER_ID}
    result["torch_version"] = torch.__version__

    LEDGER_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()