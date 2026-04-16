"""Tests for Layer 3 (score_follow) + L2 frontier refactor."""
from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from agent.layers.score_follow import (
    Layer3, hash_to_embedding, action_to_one_hot,
    HASH_EMBED_DIM, NUM_ACTIONS, HORIZON, HUBER_DELTA, LR, BATCH_SIZE, HIDDEN, N_HIDDEN_LAYERS,
)
from agent.layers.state_graph import StateGraph


def _hash(s: str) -> str:
    """SHA256-formatted 64-byte (128 hex char) test hash."""
    import hashlib
    return hashlib.sha256(s.encode()).hexdigest()


def test_hyperparams_match_prereg() -> None:
    assert HORIZON == 50
    assert LR == 1e-3
    assert HUBER_DELTA == 1.0
    assert BATCH_SIZE == 32
    assert HIDDEN == 128
    assert N_HIDDEN_LAYERS == 3
    assert HASH_EMBED_DIM == 64
    assert NUM_ACTIONS == 7
    print("  OK hyperparams_match_prereg")


def test_hash_embedding_deterministic() -> None:
    h = _hash("foo")
    e1 = hash_to_embedding(h)
    e2 = hash_to_embedding(h)
    assert torch.equal(e1, e2)
    assert e1.shape == (64,)
    # All elements ±1
    assert set(e1.unique().tolist()) <= {-1.0, 1.0}
    # Different hashes → different embeddings
    e3 = hash_to_embedding(_hash("bar"))
    assert not torch.equal(e1, e3)
    print("  OK hash_embedding_deterministic")


def test_action_one_hot() -> None:
    for aid in range(1, 8):
        v = action_to_one_hot(aid)
        assert v.shape == (7,)
        assert v.sum().item() == 1.0
        assert v[aid - 1].item() == 1.0
    print("  OK action_one_hot")


def test_dormant_until_score_delta() -> None:
    layer = Layer3(master_seed=42)
    assert not layer.activated
    layer.record_step(_hash("h0"), 1)
    layer.record_step(_hash("h1"), 2)
    layer.on_score_delta(0)  # no delta
    assert not layer.activated
    layer.on_score_delta(1)  # positive!
    assert layer.activated
    assert len(layer.training_set) == 2
    print("  OK dormant_until_score_delta")


def test_picker_none_when_dormant() -> None:
    layer = Layer3(master_seed=42)
    picker = layer.make_frontier_picker()
    assert picker is None  # signals "use BFS default"
    print("  OK picker_none_when_dormant")


def test_back_label_distances_clipped_at_horizon() -> None:
    layer = Layer3(master_seed=42)
    # Record HORIZON + 5 steps
    for i in range(HORIZON + 5):
        layer.record_step(_hash(f"h{i}"), 1)
    # Path deque is capped at HORIZON+1, so only 51 entries kept
    assert len(layer.path) == HORIZON + 1
    layer.on_score_delta(1)
    # All distances must be <= HORIZON
    for ex in layer.training_set:
        assert ex.distance <= HORIZON
    print("  OK back_label_distances_clipped_at_horizon")


def test_path_clears_after_back_label() -> None:
    layer = Layer3(master_seed=42)
    layer.record_step(_hash("h0"), 1)
    layer.on_score_delta(1)
    assert len(layer.path) == 0
    print("  OK path_clears_after_back_label")


def test_reset_wipes_state() -> None:
    layer = Layer3(master_seed=42)
    layer.record_step(_hash("h0"), 1)
    layer.on_score_delta(1)
    assert layer.activated
    layer.reset()
    assert not layer.activated
    assert len(layer.training_set) == 0
    assert len(layer.path) == 0
    print("  OK reset_wipes_state")


def test_predict_distance_returns_float() -> None:
    layer = Layer3(master_seed=42)
    d = layer.predict_distance(_hash("h0"), 1)
    assert isinstance(d, float)
    print("  OK predict_distance_returns_float")


def test_picker_after_activation_picks_lowest_predicted() -> None:
    """After training, picker should pick the candidate with lowest predicted distance."""
    layer = Layer3(master_seed=42)
    # Synthetic training: hash_h_close + action 1 → distance 0; everything else → distance 50
    h_close = _hash("h_close")
    h_far = _hash("h_far")
    # Train by manual back-label: simulate path arriving at h_close
    layer.record_step(h_close, 1)
    layer.on_score_delta(1)
    # Now reinforce: lots of back-labels with h_far at distance 50 (no further milestones)
    for i in range(40):
        for j in range(50):
            layer.record_step(_hash(f"far{i}_{j}"), 2)
        # don't fire score_delta — these stay in path/never trained
    # Add concentrated training of h_far at distance 50
    for _ in range(50):
        for j in range(50):
            layer.record_step(h_far, 2)
        layer.on_score_delta(1)
    # Now picker should prefer h_close (distance ~0) over h_far (~50)
    picker = layer.make_frontier_picker()
    assert picker is not None  # activated
    candidates = [
        (h_close, 5, [(1, None, None)]),
        (h_far,   1, [(2, None, None)]),  # closer in BFS but should rerank
    ]
    chosen = picker(candidates)
    assert chosen[0] == h_close, f"picker should rerank to h_close; got {chosen[0]}"
    print("  OK picker_after_activation_picks_lowest_predicted")


def test_l2_find_all_frontiers() -> None:
    """L2 refactor: find_all_frontiers returns sorted list of (target, depth, path)."""
    g = StateGraph()
    # Build: h0 → h1 (via action 1), h0 → h2 (via action 2). Both h1 and h2 visited with untested.
    g.observe_state("h0", [1, 2])
    g.observe_transition("h0", 1, None, None, "h1")
    g.observe_state("h1", [3, 4])
    g.observe_transition("h0", 2, None, None, "h2")
    g.observe_state("h2", [5, 6])
    # h0 still has nothing untested? It started with [1,2], both got transitioned.
    # Actually after observe_transition, untested[h0] is reduced by the action_id. So {1,2} - {1} - {2} = {}.
    assert g.untested["h0"] == set()
    # h1 has untested {3, 4}, h2 has untested {5, 6}. Both depth 1 from h0.
    fronts = g.find_all_frontiers("h0")
    assert len(fronts) == 2, f"expected 2 frontiers, got {len(fronts)}: {fronts}"
    # sorted by depth then hash
    target_hashes = [f[0] for f in fronts]
    depths = [f[1] for f in fronts]
    assert depths == [1, 1]
    assert target_hashes == sorted(target_hashes)  # tie-break by hash
    print("  OK l2_find_all_frontiers")


def test_l2_select_action_with_layer3_picker() -> None:
    """Integration: L2.select_action accepts Layer3's picker callback."""
    g = StateGraph()
    g.observe_state("h0", [1, 2])
    g.observe_transition("h0", 1, None, None, "h1")
    g.observe_state("h1", [3])
    g.observe_transition("h0", 2, None, None, "h2")
    g.observe_state("h2", [4])
    layer3 = Layer3(master_seed=42)
    # Train Layer 3 to prefer h1 (low predicted distance)
    layer3.record_step("h1", 3)
    layer3.on_score_delta(1)
    picker = layer3.make_frontier_picker()
    # When at h0 with no untested locally (both 1, 2 used), L2 calls picker
    res = g.select_action("h0", [1, 2], lambda: (0, 0), frontier_picker=picker)
    assert res is not None
    aid, _ = res
    # Path to h1 = action 1; path to h2 = action 2. Picker prefers h1.
    # Note: predict_distance for (h0=h1, action=3) won't differentiate h1 vs h2 unless
    # training distinguishes them. This test just verifies the integration runs without error.
    assert aid in (1, 2)
    print("  OK l2_select_action_with_layer3_picker")


def main() -> int:
    print("Layer 3 (score_follow) tests:")
    test_hyperparams_match_prereg()
    test_hash_embedding_deterministic()
    test_action_one_hot()
    test_dormant_until_score_delta()
    test_picker_none_when_dormant()
    test_back_label_distances_clipped_at_horizon()
    test_path_clears_after_back_label()
    test_reset_wipes_state()
    test_predict_distance_returns_float()
    test_l2_find_all_frontiers()
    test_l2_select_action_with_layer3_picker()
    test_picker_after_activation_picks_lowest_predicted()
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
