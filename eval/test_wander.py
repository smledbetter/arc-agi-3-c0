"""Tests for Layer 1 (wander policy)."""
from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from agent.layers.bce_frame_change import Layer0, GRID
from agent.layers.wander import wander, _sample_click, FILTER_THRESHOLD


def _frame() -> np.ndarray:
    return np.zeros((GRID, GRID), dtype=np.int8)


def test_filter_threshold_value() -> None:
    assert FILTER_THRESHOLD == 0.05, "pre-reg pins threshold at 0.05"
    print("  OK filter_threshold_value")


def test_returns_action_in_available() -> None:
    layer = Layer0(master_seed=42)
    rng = np.random.default_rng(0)
    available = [1, 2, 3, 4]
    for _ in range(20):
        a, data, audit = wander(layer, _frame(), available, rng)
        assert a in available
        assert data is None  # not ACTION6
        assert audit.n_available == 4
    print("  OK returns_action_in_available")


def test_action6_returns_xy() -> None:
    layer = Layer0(master_seed=42)
    rng = np.random.default_rng(0)
    available = [6]
    a, data, audit = wander(layer, _frame(), available, rng)
    assert a == 6
    assert data is not None and "x" in data and "y" in data
    assert 0 <= data["x"] < GRID and 0 <= data["y"] < GRID
    print("  OK action6_returns_xy")


def test_action7_always_kept() -> None:
    """ACTION7 (undo) has no Layer 0 signal → never filtered."""
    layer = Layer0(master_seed=42)
    rng = np.random.default_rng(0)
    a, data, audit = wander(layer, _frame(), [7], rng)
    assert a == 7
    assert data is None
    assert not audit.fallback_fired  # 7 was kept normally, not via fallback
    print("  OK action7_always_kept")


def test_filter_rejects_low_prob_actions() -> None:
    """Force Layer 0 to predict ~0 for ACTION1, ~1 for ACTION2 → only ACTION2 kept."""
    layer = Layer0(master_seed=42)
    rng = np.random.default_rng(0)
    # Train: ACTION1 NEVER changes, ACTION2 ALWAYS changes
    for i in range(100):
        f = np.random.default_rng(i).integers(0, 16, size=(GRID, GRID)).astype(np.int8)
        layer.observe_and_train(f, f"h_a1_{i}", 1, None, None, 0)
        layer.observe_and_train(f, f"h_a2_{i}", 2, None, None, 1)
    # Verify Layer 0 learned
    probs = layer.predict_action_probs(_frame())
    assert probs[1] < FILTER_THRESHOLD, f"P(A1)={probs[1]:.3f} should be < 0.05"
    assert probs[2] > FILTER_THRESHOLD, f"P(A2)={probs[2]:.3f} should be > 0.05"
    # Wander on [1, 2] → should always pick 2 (ACTION1 filtered out)
    picks = []
    for _ in range(20):
        a, _, audit = wander(layer, _frame(), [1, 2], rng)
        picks.append(a)
    assert all(p == 2 for p in picks), f"expected all ACTION2, got {picks}"
    # Audit should reflect rejection but NOT fallback
    _, _, audit = wander(layer, _frame(), [1, 2], rng)
    assert audit.rejected_any is True
    assert audit.fallback_fired is False
    assert audit.n_kept == 1
    print("  OK filter_rejects_low_prob_actions")


def test_fallback_when_all_filtered() -> None:
    """When ALL actions are below threshold → uniform fallback over original set."""
    layer = Layer0(master_seed=42)
    rng = np.random.default_rng(0)
    # Train so that both ACTION1 and ACTION2 predict ~0
    for i in range(100):
        f = np.random.default_rng(i).integers(0, 16, size=(GRID, GRID)).astype(np.int8)
        layer.observe_and_train(f, f"h_a1_{i}", 1, None, None, 0)
        layer.observe_and_train(f, f"h_a2_{i}", 2, None, None, 0)
    probs = layer.predict_action_probs(_frame())
    assert probs[1] < FILTER_THRESHOLD and probs[2] < FILTER_THRESHOLD
    a, _, audit = wander(layer, _frame(), [1, 2], rng)
    assert a in (1, 2)
    assert audit.fallback_fired is True
    assert audit.n_kept == 2  # both restored via fallback
    print("  OK fallback_when_all_filtered")


def test_sample_click_uses_high_prob_cells() -> None:
    """Click map with hot region in corner → sampled (x, y) lands in that region."""
    rng = np.random.default_rng(0)
    cm = np.zeros((GRID, GRID), dtype=np.float32)
    cm[10:20, 30:40] = 0.9  # hot region
    samples = [_sample_click(cm, rng, FILTER_THRESHOLD) for _ in range(50)]
    for x, y in samples:
        assert 30 <= x < 40 and 10 <= y < 20, f"({x},{y}) not in hot region"
    print("  OK sample_click_uses_high_prob_cells")


def test_sample_click_fallback_full_grid() -> None:
    """All cells below threshold → uniform over full grid."""
    rng = np.random.default_rng(0)
    cm = np.full((GRID, GRID), 0.01, dtype=np.float32)  # all below 0.05
    samples = [_sample_click(cm, rng, FILTER_THRESHOLD) for _ in range(100)]
    xs = [s[0] for s in samples]
    ys = [s[1] for s in samples]
    # Spans most of the grid (not collapsed to one corner)
    assert max(xs) - min(xs) > 30 and max(ys) - min(ys) > 30
    print("  OK sample_click_fallback_full_grid")


def main() -> int:
    print("Layer 1 (wander) tests:")
    test_filter_threshold_value()
    test_returns_action_in_available()
    test_action6_returns_xy()
    test_action7_always_kept()
    test_sample_click_uses_high_prob_cells()
    test_sample_click_fallback_full_grid()
    test_filter_rejects_low_prob_actions()
    test_fallback_when_all_filtered()
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
