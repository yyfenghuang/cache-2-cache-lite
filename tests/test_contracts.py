# SPDX-License-Identifier: Apache-2.0

"""The KV geometry of both models is known exactly.

Silent on success, exit 0. Refuses to start when results/contracts.json is
absent, because a test that quietly passes on a missing file closes the gate
without checking anything.

PREDICTED below is frozen. It was written from recollection before the first
run, which is the only arrangement under which this test can catch a wrong
recollection. It is never edited to agree with a run. When a field is
resolved against the raw config, the resolution goes in RESOLVED with the
date and the reason, and the prediction stays visible as it was made.

RESOLVED is what the regression assertion runs against. From the moment it is
filled it can no longer fail on the run that filled it, and that is fine: its
job from then on is to catch a library upgrade or a checkpoint swap moving the
geometry underneath the projection widths.

The prediction-versus-measurement diff is reported once, as a recorded fact,
and is not a pass criterion.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

CONTRACTS_PATH = REPO_ROOT / "results" / "contracts.json"

# Frozen. Written before the first run. Do not edit.
PREDICTED = {
    "sharer": {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "n_layers": 24,
        "hidden_size": 896,
        "n_q_heads": 14,
        "n_kv_heads": 2,
        "head_dim": 64,
        "kv_width": 128,
        "rope_theta": 1000000.0,
        "n_inv_freq": 32,
        "has_q_norm": False,
        "has_k_norm": False,
        "vocab_size": 151936,
    },
    "receiver": {
        "model_id": "Qwen/Qwen3-0.6B",
        "n_layers": 28,
        "hidden_size": 1024,
        "n_q_heads": 16,
        "n_kv_heads": 8,
        "head_dim": 128,
        "kv_width": 1024,
        "rope_theta": 1000000.0,
        "n_inv_freq": 64,
        "has_q_norm": True,
        "has_k_norm": True,
        "vocab_size": 151936,
    },
}

# Filled from the raw config.json, one field at a time, with a reason.
# Empty until the first run has been read.
RESOLVED: dict[str, dict] = {}

RESOLUTION_LOG: list[str] = [
    # "YYYY-MM-DD sharer.rope_theta 1000000.0 -> <read> : <how it was checked>"
]


def load():
    if not CONTRACTS_PATH.exists():
        raise SystemExit(
            f"missing {CONTRACTS_PATH.relative_to(REPO_ROOT)}; "
            "run scripts/capture_contracts.py first"
        )
    return json.loads(CONTRACTS_PATH.read_text(encoding="utf-8"))


def _diff(reference: dict, contract: dict) -> list[str]:
    out = []
    for role, fields in reference.items():
        for key, want in fields.items():
            got = contract[role].get(key, "<field absent>")
            if got != want:
                out.append(f"{role}.{key}: {want!r} against {got!r}")
    return out


def test_resolved_values_hold():
    """Regression. Silent while RESOLVED is empty, which is the honest state
    before the first run has been read."""
    if not RESOLVED:
        return
    c = load()
    bad = _diff(RESOLVED, c)
    if bad:
        raise AssertionError(
            "the live configs no longer match the resolved contract:\n  "
            + "\n  ".join(bad)
        )


def test_prediction_is_reported_in_full():
    """Every wrong recollection surfaces in one run, not one per run."""
    c = load()
    bad = _diff(PREDICTED, c)
    if bad and not RESOLVED:
        raise AssertionError(
            "prediction on record does not match the live configs. Resolve "
            "each against the raw config.json, then fill RESOLVED and "
            "RESOLUTION_LOG. Do not edit PREDICTED.\n  " + "\n  ".join(bad)
        )


def test_internal_consistency():
    """Relations that cannot be wrong by recollection, only by construction."""
    c = load()
    for role in ("sharer", "receiver"):
        m = c[role]
        assert m["kv_width"] == m["n_kv_heads"] * m["head_dim"], role
        assert m["n_q_heads"] % m["n_kv_heads"] == 0, role
        assert m["gqa_group_size"] == m["n_q_heads"] // m["n_kv_heads"], role
        assert m["n_inv_freq"] == len(m["inv_freq"]), role
        assert (
            m["n_inv_freq"]
            == int(m["head_dim"] * m["partial_rotary_factor"]) // 2
        ), role
        assert abs(m["inv_freq"][0] - 1.0) < 1e-12, role
        assert m["head_dim_is_decoupled"] == (
            m["head_dim"] != m["head_dim_from_hidden_size"]
        ), role


def test_rope_theta_agrees_with_the_checkpoint_file():
    """The recorded base must equal what the checkpoint states on disk.

    Attribute presence is not evidence of meaning. Under transformers v5 the
    deprecated config.rope_theta is present and readable while holding a
    class default, so the only check with teeth is against the raw file.
    """
    c = load()
    prov = c["config_provenance"]
    for role in ("sharer", "receiver"):
        p = prov[role]
        assert p["ran"] is True, f"{role}: {p.get('reason')}"
        raw = p["raw_values"].get("rope_theta")
        if raw is None:
            nested = p["raw_values"].get("rope_parameters") or {}
            raw = nested.get("rope_theta")
        assert raw is not None, (
            f"{role}: the checkpoint config states no rotary base anywhere; "
            f"the recorded value is a library convention under transformers "
            f"{prov['transformers_version']} and must be pinned explicitly"
        )
        assert float(raw) == c[role]["rope_theta"], (
            f"{role}: contract records {c[role]['rope_theta']} but "
            f"{p['source']} states {raw}"
        )


def test_deprecated_rope_attribute_disagreement_is_recorded():
    """Either outcome is legal. Silence is not.

    On a v5 load the deprecated attribute is expected to disagree. Recording
    it is what stops the next reader from reaching for the same wrong drawer.
    """
    c = load()
    for role in ("sharer", "receiver"):
        m = c[role]
        assert "rope_theta_source" in m, role
        assert "rope_theta_legacy_attr" in m, role
        assert m["rope_theta_legacy_agrees"] in (True, False, None), role


def test_head_dim_decoupling_is_recorded_not_assumed():
    """Qwen3-0.6B declares head_dim 128 while hidden_size / n_q_heads is 64.

    This is the getattr fallback in the modeling code doing load-bearing
    work. If the contract ever reports the derived value for the Receiver,
    every width downstream is halved and nothing raises.
    """
    c = load()
    assert c["receiver"]["head_dim_declared_in_config"] is True
    assert c["receiver"]["head_dim_is_decoupled"] is True
    assert c["receiver"]["head_dim_from_hidden_size"] == 64


def test_inv_freq_matches_the_library():
    """Knowing the formula and matching the library are different claims."""
    c = load()
    for role in ("sharer", "receiver"):
        check = c[role]["inv_freq_crosscheck"]
        if check["ran"] is not True:
            raise AssertionError(
                f"{role}: no route to the library ladder\n  "
                + "\n  ".join(check.get("attempts", ["<no attempts recorded>"]))
            )
        assert check["match"] is True, f"{role}: {json.dumps(check, indent=2)}"


def test_rope_comparison_is_recorded():
    """The comparison must exist. Either outcome is a legal result."""
    c = load()
    rope = c["rope_comparison"]
    for key in ("theta_match", "length_match", "n_inv_freq", "nesting"):
        assert key in rope, key
    assert rope["nesting"]["holds"] in (True, False)


def test_layer_alignment_matches_the_layer_counts():
    c = load()
    la = c["layer_alignment"]
    n_source = c["sharer"]["n_layers"]
    n_target = c["receiver"]["n_layers"]
    assert len(la["target_to_source"]) == n_target
    assert la["n_pairs"] == min(n_source, n_target)
    assert la["target_to_source"][-1] == n_source - 1


def test_tokenizer_probe_answers_the_scope_question():
    c = load()
    tok = c["tokenizers"]
    assert "probe_ids_identical" in tok
    assert tok["align_tokens_in_scope"] == (not tok["probe_ids_identical"])
    assert tok["probe_ids_identical"] is True, (
        "the reference recipe sets is_do_alignment false for this pair; "
        "differing probe ids contradict that and mean something else in the "
        "setup is wrong"
    )


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()


if __name__ == "__main__":
    main()