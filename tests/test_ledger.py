"""The ledger is complete, its arithmetic closes, and no timing stands alone.

Silent on success, exit 0. Refuses to start when results/ledger.json is absent.

This gate asserts no threshold on any duration. Nothing in this repository
says what fusion ought to cost on a six-thread CPU, and a gate that failed
when a number came out unwelcome would be a gate measuring the expectation
rather than the machine.

What it does assert is that the numbers are the kind of numbers that can be
quoted safely.

Payload is arithmetic, so it has to close: bytes must equal elements times the
width of the element, and the total must equal the per-layer figure times the
layers actually sent. The computed figure is then checked against a real
captured tensor, which is where a contract that disagrees with the tensors
would surface.

Every duration carries the position count it was taken at and a spread beside
its median. A median with no spread is a number that will be quoted as exact.
A duration with no length attached is not a duration.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

LEDGER_PATH = REPO_ROOT / "results" / "ledger.json"
CONTRACTS_PATH = REPO_ROOT / "results" / "contracts.json"

TIMINGS = (
    "sharer_prefill",
    "receiver_prefill",
    "fusion",
    "sharer_decode_per_token",
)

# Every timing entry must carry these, or it is a number without a question.
REQUIRED_FIELDS = ("positions", "median_ms", "iqr_ms", "min_ms", "max_ms", "repeats")


def load(path):
    if not path.exists():
        raise SystemExit(
            f"missing {path.relative_to(REPO_ROOT)}; run "
            "scripts/measure_ledger.py first"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_payload_arithmetic_closes():
    ledger = load(LEDGER_PATH)
    for side in ("sent_by_the_sharer", "receiver_own_cache_for_scale"):
        spec = ledger["payload"][side]
        expected = 2 * spec["n_kv_heads"] * spec["head_dim"]
        assert spec["elements_per_layer_per_position"] == expected, side
        for name, per_layer in spec["bytes_per_layer_per_position"].items():
            width = per_layer / expected
            assert width == int(width), (side, name)
            assert spec["bytes_per_position"][name] == (
                per_layer * spec["n_layers_sent"]
            ), (side, name)


def test_only_the_paired_layers_are_priced():
    """Terminal alignment leaves the Receiver's shallowest layers unpaired.
    The Sharer has nothing to send there, and counting them would overstate
    the payload by the layers it does not have."""
    ledger, contract = load(LEDGER_PATH), load(CONTRACTS_PATH)
    paired = sum(
        1 for s in contract["layer_alignment"]["target_to_source"] if s is not None
    )
    assert ledger["payload"]["sent_by_the_sharer"]["n_layers_sent"] == paired
    assert paired < contract["receiver"]["n_layers"]


def test_the_computed_payload_matches_a_real_cache():
    """Where a contract that disagrees with the tensors would surface.

    Compared per layer rather than per cache. The payload prices the layers
    with a partner and a captured cache holds every layer the model has, and
    those are the same number only by accident of this pair.
    """
    ledger = load(LEDGER_PATH)
    sides = (
        ("sharer", "sent_by_the_sharer"),
        ("receiver", "receiver_own_cache_for_scale"),
    )
    for positions, observed in ledger["observed_cache_bytes"].items():
        n = int(positions)
        for prefix, key in sides:
            per_layer = ledger["payload"][key]["bytes_per_layer_per_position"][
                "float32"
            ]
            layers = observed[f"{prefix}_layers_captured"]
            assert layers > 0, (positions, prefix)
            assert observed[f"{prefix}_bytes_float32"] == per_layer * layers * n, (
                positions, prefix
            )


def test_every_timing_carries_a_length_and_a_spread():
    ledger = load(LEDGER_PATH)
    measured = set(ledger["positions_measured"])
    for name in TIMINGS:
        entries = ledger["timings"][name]
        assert entries, name
        assert {e["positions"] for e in entries} == measured, name
        for entry in entries:
            for field in REQUIRED_FIELDS:
                assert field in entry, (name, entry.get("positions"), field)
            assert entry["repeats"] >= 3, (name, entry["positions"])
            assert entry["median_ms"] > 0, (name, entry["positions"])
            assert entry["min_ms"] <= entry["median_ms"] <= entry["max_ms"], (
                name, entry["positions"]
            )
            assert entry["iqr_ms"] >= 0, (name, entry["positions"])


def test_fusion_costs_more_over_a_longer_cache():
    """The one directional claim worth asserting.

    The fuser reads one vector per position, so its cost has to grow with the
    positions. If it does not, the module is not seeing what it is being
    handed, and the shortest and longest measured lengths differ by enough
    that noise cannot close the gap.
    """
    ledger = load(LEDGER_PATH)
    entries = sorted(ledger["timings"]["fusion"], key=lambda e: e["positions"])
    shortest, longest = entries[0], entries[-1]
    assert longest["positions"] >= 4 * shortest["positions"], (
        "the measured lengths are too close together for this to mean anything"
    )
    assert longest["median_ms"] > shortest["median_ms"], (
        f"fusion over {longest['positions']} positions "
        f"({longest['median_ms']:.2f} ms) is not slower than over "
        f"{shortest['positions']} ({shortest['median_ms']:.2f} ms)"
    )


def test_the_decode_rate_is_per_token_and_says_what_it_subtracted():
    """A saving with no denominator is not a saving.

    The rate is the primitive here. No total appears, because a total needs a
    response length and this repository has not fixed one.
    """
    ledger = load(LEDGER_PATH)
    for entry in ledger["timings"]["sharer_decode_per_token"]:
        assert entry["tokens_decoded"] >= 2, entry["positions"]
        assert entry["prefill_subtracted_ms"] > 0, entry["positions"]
        assert entry["whole_call_median_ms"] > entry["prefill_subtracted_ms"], (
            f"at {entry['positions']} positions the whole decode call is not "
            "longer than the prefill inside it"
        )
        rebuilt = (
            entry["whole_call_median_ms"] - entry["prefill_subtracted_ms"]
        ) / entry["tokens_decoded"]
        assert abs(entry["median_ms"] - rebuilt) < 1e-9, entry["positions"]


def test_the_measurement_says_where_it_was_taken():
    """Table 3 prices this pair on an A100. These are CPU numbers. A ledger
    that does not carry its device invites the two being divided."""
    ledger = load(LEDGER_PATH)
    for field in ("device", "torch_num_threads", "dtype", "attn_implementation",
                  "comparability", "fuser_checkpoint", "fuser_mode"):
        assert ledger.get(field) not in (None, ""), field
    assert ledger["fuser_mode"] == "fused"


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()


if __name__ == "__main__":
    main()