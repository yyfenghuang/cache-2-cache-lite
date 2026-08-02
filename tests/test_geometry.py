"""Both caches at the shapes the contract predicts, and a null worth reading.

Silent on success, exit 0.

The rehearsal builds a Qwen2 and a Qwen3 from config objects, at the layer
and head counts the real pair has, and runs the whole measurement on them.
Random weights change every magnitude and no part of the structure, so a
shape that is wrong here is wrong there, and a null that is negative here
would be negative there.

Thresholds fixed before the first run against real weights:

  the per channel null is the strictest of the three constant predictors,
  so it can never exceed the global mean null, which can never exceed the
  zero null. Those orderings are arithmetic and hold whatever the weights
  are. They are asserted rather than assumed because a variance computed by
  cancellation in float32 can come out negative and still look plausible.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

NULL_PATH = REPO_ROOT / "results" / "geometric_null.json"
CONTRACTS_PATH = REPO_ROOT / "results" / "contracts.json"


def _capture_module():
    path = REPO_ROOT / "scripts" / "capture_dual_cache.py"
    spec = importlib.util.spec_from_file_location("_capture_dual_cache", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_capture_dual_cache"] = module
    spec.loader.exec_module(module)
    return module


def load(path):
    if not path.exists():
        raise SystemExit(
            f"missing {path.relative_to(REPO_ROOT)}; the gate below it has "
            "not been closed"
        )
    return json.loads(path.read_text(encoding="utf-8"))


# -------------------------------------------------------------------------
# pure


def test_channel_stats_recovers_a_known_variance():
    stats = _capture_module().ChannelStats(4)
    torch.manual_seed(0)
    tensor = torch.randn(1, 2, 500, 2) * 3.0 + 7.0
    stats.add(tensor)
    report = stats.report()

    flat = tensor.permute(0, 2, 1, 3).reshape(500, 4).to(torch.float64)
    expected = flat.var(dim=0, unbiased=False).mean()
    assert abs(report["null_mse_per_channel_mean"] - float(expected)) < 1e-9
    assert report["n_positions"] == 500 and report["n_channels"] == 4


def test_constant_predictors_are_ordered_by_strictness():
    stats = _capture_module().ChannelStats(6)
    torch.manual_seed(1)
    stats.add(torch.randn(1, 3, 200, 2) * 2.0 + 5.0)
    r = stats.report()
    assert 0.0 <= r["null_mse_per_channel_mean"] <= r["null_mse_global_mean"] + 1e-9
    assert r["null_mse_global_mean"] <= r["null_mse_zero"] + 1e-9


def test_channel_stats_refuses_a_variance_of_one_sample():
    stats = _capture_module().ChannelStats(4)
    stats.add(torch.randn(1, 2, 1, 2))
    try:
        stats.report()
    except ValueError:
        return
    raise AssertionError("reported a variance from a single position")


def test_channel_stats_rejects_the_wrong_width():
    stats = _capture_module().ChannelStats(8)
    try:
        stats.add(torch.randn(1, 2, 5, 2))
    except ValueError:
        return
    raise AssertionError("accepted a tensor of the wrong channel count")


def test_tokenization_comparison_reports_both_outcomes():
    compare = _capture_module().compare_tokenizations
    same = compare([[1, 2, 3], [4, 5]], [[1, 2, 3], [4, 5]])
    assert same["identical"] is True and same["n_tokens"] == 5
    differ = compare([[1, 2, 3]], [[1, 2, 4]])
    assert differ["identical"] is False and differ["mismatches"][0]["index"] == 0


# -------------------------------------------------------------------------
# rehearsal


def test_geometry_rehearsal_on_models_with_no_checkpoint():
    from transformers import (
        Qwen2Config, Qwen2ForCausalLM, Qwen3Config, Qwen3ForCausalLM,
    )

    torch.manual_seed(0)
    rope = {"rope_type": "default", "rope_theta": 1000000.0}
    sharer_config = Qwen2Config(
        vocab_size=512, hidden_size=112, intermediate_size=224,
        num_hidden_layers=24, num_attention_heads=14, num_key_value_heads=2,
        max_position_embeddings=512, rope_parameters=rope,
    )
    receiver_config = Qwen3Config(
        vocab_size=512, hidden_size=128, intermediate_size=256,
        num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8,
        head_dim=32, max_position_embeddings=512, rope_parameters=rope,
    )
    for config in (sharer_config, receiver_config):
        config._attn_implementation = "eager"

    sharer = Qwen2ForCausalLM(sharer_config).eval()
    receiver = Qwen3ForCausalLM(receiver_config).eval()

    ids = [torch.randint(0, 512, (1, n)) for n in (23, 31, 19, 27)]
    result = _capture_module().measure_geometry(sharer, receiver, ids)

    assert result["cache_shapes"]["sharer"][0] == 24
    assert result["cache_shapes"]["sharer"][2] == 2
    assert result["cache_shapes"]["receiver"][0] == 28
    assert result["cache_shapes"]["receiver"][2] == 8
    assert result["cache_shapes"]["receiver"][4] == 32

    assert result["null"]["unpaired_target_layers"] == [0, 1, 2, 3]
    assert len(result["null"]["paired_target_layers"]) == 24

    for role in ("sharer", "receiver"):
        for kind in ("keys", "values"):
            series = result["per_layer"][role][kind]
            assert len(series) == result["layer_counts"][role]
            for entry in series:
                assert entry["null_mse_per_channel_mean"] >= 0.0
                assert (
                    entry["null_mse_per_channel_mean"]
                    <= entry["null_mse_global_mean"] + 1e-9
                )
                assert entry["null_mse_global_mean"] <= entry["null_mse_zero"] + 1e-9
                assert entry["n_positions"] == 100

    for kind in ("keys", "values"):
        assert result["null"]["aggregate_paired_target_layers"][kind] > 0.0


# -------------------------------------------------------------------------
# against the real run


def test_shapes_match_the_contract_not_literals():
    result = load(NULL_PATH)
    contract = load(CONTRACTS_PATH)

    for role in ("sharer", "receiver"):
        n_layers, batch, n_kv_heads, positions, head_dim = result["cache_shapes"][role]
        assert n_layers == contract[role]["n_layers"], role
        assert n_kv_heads == contract[role]["n_kv_heads"], (
            f"{role} cache holds {n_kv_heads} key-value heads, contract says "
            f"{contract[role]['n_kv_heads']}"
        )
        assert head_dim == contract[role]["head_dim"], role
        assert n_kv_heads * head_dim == contract[role]["kv_width"], role
        assert result["per_layer"][role]["keys"][0]["n_channels"] == (
            contract[role]["kv_width"]
        ), role


def test_both_tokenizers_saw_the_same_tokens():
    """The Tier 0 caveat, made operational.

    The two vocabularies agree on ordinary strings and differ by four tokens
    that belong to the Receiver. If one of those ever reaches the Sharer the
    two id streams diverge, the positions stop corresponding, and every
    number below is computed on a pair of caches that describe different
    text.
    """
    result = load(NULL_PATH)
    tokenization = result["tokenization"]
    assert tokenization["identical"] is True, tokenization["mismatches"]
    assert tokenization["n_tokens"] > 0


def test_the_null_is_ordered_and_positive():
    result = load(NULL_PATH)
    for role in ("sharer", "receiver"):
        for kind in ("keys", "values"):
            for index, entry in enumerate(result["per_layer"][role][kind]):
                where = f"{role}.{kind}.layer{index}"
                assert entry["null_mse_per_channel_mean"] > 0.0, where
                assert (
                    entry["null_mse_per_channel_mean"]
                    <= entry["null_mse_global_mean"] + 1e-9
                ), where
                assert entry["null_mse_global_mean"] <= entry["null_mse_zero"] + 1e-9, where


def test_the_aggregate_null_covers_only_layers_that_can_be_trained():
    """Unpaired target layers have no source, so nothing will be trained for
    them and they must not move the number the training is graded against."""
    result = load(NULL_PATH)
    contract = load(CONTRACTS_PATH)
    null = result["null"]

    n_source = contract["sharer"]["n_layers"]
    n_target = contract["receiver"]["n_layers"]
    assert len(null["paired_target_layers"]) == min(n_source, n_target)
    assert null["paired_target_layers"][-1] == n_target - 1
    assert set(null["paired_target_layers"]) & set(null["unpaired_target_layers"]) == set()

    for kind in ("keys", "values"):
        assert null["aggregate_paired_target_layers"][kind] > 0.0


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()


if __name__ == "__main__":
    main()