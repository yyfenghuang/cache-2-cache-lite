"""Both caches at the contract shapes, split by article, with a null per split.

Silent on success, exit 0.

The rehearsal builds a Qwen2 and a Qwen3 from config objects at the head
geometry the real pair has and runs the two phase capture on them. Random
weights change every magnitude and no part of the structure.

One gate here belongs to Tier 3 in spirit and sits here in practice: the
training split has to hold enough positions for the projection that will be
fitted on it. Discovering that after training has run wastes the run and,
worse, produces a held out failure that cannot be told apart from there
being no signal.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from c2c.projection import CacheProjection

NULL_PATH = REPO_ROOT / "results" / "geometric_null.json"
CONTRACTS_PATH = REPO_ROOT / "results" / "contracts.json"
CACHE_ROOT = REPO_ROOT / "results" / "caches"

SPLITS = ("train", "validation", "held_out")
KINDS = ("keys", "values")

# Must match scripts/train_projection.py. A mismatch would let this gate pass
# a corpus the training then starves on.
PROJECTION = {"depth": 2, "hidden": 256, "activation": "gelu"}
MIN_POSITIONS_PER_PARAMETER = 50.0


def _capture():
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


# ---------------------------------------------------------------- pure


def test_articles_are_split_at_titles_not_at_sections():
    lines = [" = Alpha = \n", "a\n", " = = Section = = \n", "still a\n",
             " = Beta = \n", "b\n"]
    articles = list(_capture().iter_articles(lines))
    assert len(articles) == 2
    assert "= = Section = =" in articles[0]


def test_chunking_drops_the_short_tail():
    chunk = _capture().chunk_ids
    assert [len(c) for c in chunk(list(range(300)), 128, 32)] == [128, 128, 44]
    assert [len(c) for c in chunk(list(range(150)), 128, 32)] == [128]
    assert chunk(list(range(20)), 128, 32) == []


def test_no_article_appears_in_both_splits():
    """A held out chunk from a training article is not held out."""
    assign = _capture().assign_splits
    articles = [(i, [list(range(64))] * 3) for i in range(40)]
    result = assign(articles, {"train": 400, "held_out": 200}, 1234)
    assert not set(result["train"]["articles"]) & set(result["held_out"]["articles"])
    assert result["train"]["n_tokens"] >= 400
    assert result["held_out"]["n_tokens"] >= 200


def test_split_refuses_when_the_dataset_runs_out():
    assign = _capture().assign_splits
    try:
        assign([(0, [list(range(64))])], {"train": 4000, "held_out": 2000}, 1)
    except ValueError:
        return
    raise AssertionError("filled a budget the dataset could not cover")


def test_channel_report_recovers_a_known_variance():
    torch.manual_seed(0)
    stacked = torch.randn(500, 4) * 3.0 + 7.0
    report = _capture().channel_report(stacked)
    expected = float(stacked.to(torch.float64).var(dim=0, unbiased=False).mean())
    assert abs(report["null_mse_per_channel_mean"] - expected) < 1e-9
    assert 0.0 <= report["null_mse_per_channel_mean"] <= report["null_mse_global_mean"] + 1e-9
    assert report["null_mse_global_mean"] <= report["null_mse_zero"] + 1e-9


def test_channel_report_refuses_a_single_position():
    for bad in (torch.randn(1, 4), torch.randn(3, 4, 5)):
        try:
            _capture().channel_report(bad)
        except ValueError:
            continue
        raise AssertionError("reported statistics it could not compute")


# ---------------------------------------------------------------- rehearsal


def test_capture_rehearsal_on_models_with_no_checkpoint():
    import shutil
    from transformers import (
        Qwen2Config, Qwen2ForCausalLM, Qwen3Config, Qwen3ForCausalLM,
    )

    capture = _capture()
    scratch = REPO_ROOT / "results" / "_rehearsal_caches"
    original_root, original_flush = capture.CACHE_ROOT, capture.FLUSH_EVERY_CHUNKS
    capture.CACHE_ROOT, capture.FLUSH_EVERY_CHUNKS = scratch, 3
    shutil.rmtree(scratch, ignore_errors=True)
    try:
        torch.manual_seed(0)
        rope = {"rope_type": "default", "rope_theta": 1000000.0}
        sharer = Qwen2ForCausalLM(Qwen2Config(
            vocab_size=512, hidden_size=128, intermediate_size=192,
            num_hidden_layers=24, num_attention_heads=14, num_key_value_heads=2,
            head_dim=64, max_position_embeddings=512, rope_parameters=rope,
            _attn_implementation="eager")).eval()
        receiver = Qwen3ForCausalLM(Qwen3Config(
            vocab_size=512, hidden_size=128, intermediate_size=192,
            num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8,
            head_dim=128, max_position_embeddings=512, rope_parameters=rope,
            _attn_implementation="eager")).eval()

        # One entry per split, derived from SPLITS rather than listed, so a
        # split added to the pipeline cannot be missed here again.
        chunks = {
            split: [torch.randint(0, 512, (96,)).tolist() for _ in range(n)]
            for split, n in zip(SPLITS, (6, 3, 3))
        }
        for role, model in (("sharer", sharer), ("receiver", receiver)):
            for split in SPLITS:
                shape = capture.capture_role(model, role, split, chunks[split])
                reports = capture.finalise(
                    split, role, model.config.num_hidden_layers
                )
                assert len(reports["keys"]) == model.config.num_hidden_layers
                stored = torch.load(
                    scratch / split / f"{role}_keys_layer00.pt",
                    map_location="cpu",
                )
                assert stored.shape[0] == reports["keys"][0]["n_positions"], (
                    "the statistics and the stored file disagree on how many "
                    "positions there are"
                )
                assert stored.shape[1] == reports["keys"][0]["n_channels"]
        assert shape[0] == 28 and shape[2] == 8 and shape[4] == 128
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        capture.CACHE_ROOT, capture.FLUSH_EVERY_CHUNKS = original_root, original_flush


# ---------------------------------------------------------- against the run


def test_shapes_match_the_contract_not_literals():
    result = load(NULL_PATH)
    contract = load(CONTRACTS_PATH)
    for role in ("sharer", "receiver"):
        n_layers, batch, n_kv_heads, positions, head_dim = result["cache_shapes"][role]
        assert n_layers == contract[role]["n_layers"], role
        assert n_kv_heads == contract[role]["n_kv_heads"], role
        assert head_dim == contract[role]["head_dim"], role
        assert n_kv_heads * head_dim == contract[role]["kv_width"], role
        for split in SPLITS:
            entry = result["splits"][split][role]["keys"][0]
            assert entry["n_channels"] == contract[role]["kv_width"], (role, split)


def test_both_splits_exist_and_are_disjoint_by_article():
    result = load(NULL_PATH)
    assert set(SPLITS) <= set(result["splits"])
    assert result["corpus"]["split_by"] == "article"
    for split in SPLITS:
        assert result["splits"][split]["n_articles"] > 0, split
        assert result["splits"][split]["n_positions"] > 0, split


def test_the_file_says_which_split_grades_and_which_one_selects():
    """In sample comparison passes on uninformative input at every corpus
    size, so the file has to say which split grades. And the epoch has to be
    chosen somewhere that is not the split doing the grading, or the number
    reported was tuned on the data it is reported against."""
    result = load(NULL_PATH)
    assert result["null"]["graded_on"] == "held_out"
    assert result["null"]["epoch_selected_on"] == "validation"
    assert result["null"]["graded_on"] != result["null"]["epoch_selected_on"]


def test_both_tokenizers_saw_the_same_tokens():
    result = load(NULL_PATH)
    tokenization = result["tokenization"]
    assert tokenization["identical"] is True, tokenization["mismatches"]


def test_the_null_is_ordered_and_positive_in_both_splits():
    result = load(NULL_PATH)
    for split in SPLITS:
        for role in ("sharer", "receiver"):
            for kind in KINDS:
                for index, entry in enumerate(result["splits"][split][role][kind]):
                    where = f"{split}.{role}.{kind}.layer{index}"
                    assert entry["null_mse_per_channel_mean"] > 0.0, where
                    assert (entry["null_mse_per_channel_mean"]
                            <= entry["null_mse_global_mean"] + 1e-9), where
                    assert (entry["null_mse_global_mean"]
                            <= entry["null_mse_zero"] + 1e-9), where


def test_the_aggregate_covers_only_layers_that_can_be_trained():
    result = load(NULL_PATH)
    contract = load(CONTRACTS_PATH)
    null = result["null"]
    assert len(null["paired_target_layers"]) == min(
        contract["sharer"]["n_layers"], contract["receiver"]["n_layers"]
    )
    assert null["paired_target_layers"][-1] == contract["receiver"]["n_layers"] - 1
    assert not set(null["paired_target_layers"]) & set(null["unpaired_target_layers"])


def test_the_cache_tensors_the_training_reads_exist_and_match():
    result = load(NULL_PATH)
    contract = load(CONTRACTS_PATH)
    for split in SPLITS:
        for role in ("sharer", "receiver"):
            n_layers = contract[role]["n_layers"]
            for kind in KINDS:
                for layer in range(n_layers):
                    path = CACHE_ROOT / split / f"{role}_{kind}_layer{layer:02d}.pt"
                    assert path.exists(), f"missing {path.relative_to(REPO_ROOT)}"
                entry = result["splits"][split][role][kind][0]
                stored = torch.load(
                    CACHE_ROOT / split / f"{role}_{kind}_layer00.pt",
                    map_location="cpu",
                )
                assert list(stored.shape) == [
                    entry["n_positions"], entry["n_channels"]
                ], f"{split}/{role}/{kind}: file and statistics disagree"


def test_the_training_split_can_support_the_projection_that_reads_it():
    """Sized here, not discovered after the training has run.

    Below this ratio a held out failure cannot be told apart from not having
    enough data to find the signal, and the run would produce a number that
    looks like a finding.
    """
    result = load(NULL_PATH)
    contract = load(CONTRACTS_PATH)
    projection = CacheProjection(
        contract["sharer"]["n_kv_heads"], contract["sharer"]["head_dim"],
        contract["receiver"]["n_kv_heads"], contract["receiver"]["head_dim"],
        **PROJECTION,
    )
    per_parameter = projection.n_parameters_per_output_channel()
    positions = result["splits"]["train"]["n_positions"]
    ratio = positions / per_parameter
    assert ratio >= MIN_POSITIONS_PER_PARAMETER, (
        f"{positions} training positions is {ratio:.1f} per parameter for a "
        f"projection costing {per_parameter:.1f} per output channel. Raise "
        f"TRAIN_TOKEN_BUDGET to at least "
        f"{int(MIN_POSITIONS_PER_PARAMETER * per_parameter)}."
    )


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()


if __name__ == "__main__":
    main()