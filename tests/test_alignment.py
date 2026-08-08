# SPDX-License-Identifier: Apache-2.0

"""Layer alignment is a pure function.

SILENT ON SUCCESS by exit 0.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from c2c.alignment import align_layers, aligned_pairs, align_tokens


def test_terminal_24_to_28():
    m = align_layers(24, 28, strategy="terminal")
    assert len(m) == 28, len(m)
    assert m[27] == 23, m[27]
    assert m[4] == 0, m[4]
    assert m[:4] == [None, None, None, None], m[:4]
    used = [s for s in m if s is not None]
    assert used == list(range(24)), used


def test_identity_28_to_28():
    m = align_layers(28, 28, strategy="terminal")
    assert m == list(range(28)), m


def test_terminal_4_to_8():
    m = align_layers(4, 8, strategy="terminal")
    assert m == [None, None, None, None, 0, 1, 2, 3], m


def test_deeper_source_drops_shallow_source_layers():
    m = align_layers(8, 4, strategy="terminal")
    assert m == [4, 5, 6, 7], m
    assert None not in m


def test_terminal_anchor_holds_generally():
    for n_source in range(1, 33):
        for n_target in range(1, 33):
            m = align_layers(n_source, n_target, strategy="terminal")
            assert len(m) == n_target
            assert m[-1] == n_source - 1, (n_source, n_target, m[-1])
            used = [s for s in m if s is not None]
            assert used == sorted(set(used)), (n_source, n_target)
            assert len(used) == min(n_source, n_target), (n_source, n_target)


def test_pairs_helper():
    m = align_layers(4, 8, strategy="terminal")
    assert aligned_pairs(m) == [(4, 0), (5, 1), (6, 2), (7, 3)]


def test_rejects_unknown_strategy():
    try:
        align_layers(24, 28, strategy="depth_normalized")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown strategy was accepted")


def test_rejects_nonpositive_counts():
    for args in ((0, 28), (24, 0), (-1, 28)):
        try:
            align_layers(*args)
        except ValueError:
            continue
        raise AssertionError(f"accepted {args}")


def test_align_tokens_refuses_rather_than_returning_identity():
    try:
        align_tokens()
    except NotImplementedError:
        pass
    else:
        raise AssertionError("align_tokens returned instead of refusing")


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()


if __name__ == "__main__":
    main()