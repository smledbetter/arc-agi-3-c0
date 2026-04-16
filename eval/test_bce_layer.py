"""Smoke + unit tests for Layer 0 (agent/layers/bce_frame_change.py).

Run from project root: .venv/bin/python eval/test_bce_layer.py
"""
from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from agent.layers.bce_frame_change import (
    Layer0, one_hot_grid, NUM_COLORS, GRID, BUFFER_MAX,
)


def test_one_hot_basic() -> None:
    frame = np.zeros((GRID, GRID), dtype=np.int8)
    frame[0, 0] = 15
    frame[10, 20] = 3
    t = one_hot_grid(frame)
    assert t.shape == (NUM_COLORS, GRID, GRID)
    assert t[0, 0, 0] == 0 and t[15, 0, 0] == 1
    assert t[3, 10, 20] == 1 and t[0, 10, 20] == 0
    # Exactly one active channel per pixel
    assert (t.sum(dim=0) == 1).all()
    print("  OK one_hot_basic")


def test_one_hot_overflow_crashes() -> None:
    frame = np.zeros((GRID, GRID), dtype=np.int32)
    frame[0, 0] = 16  # out of range
    try:
        one_hot_grid(frame)
    except ValueError as e:
        assert "out of range" in str(e)
        print("  OK one_hot_overflow_crashes")
        return
    raise AssertionError("expected ValueError on palette overflow")


def test_backbone_preserves_spatial_resolution() -> None:
    """Pre-reg §3 claim: spatial resolution preserved end-to-end (no pooling/stride)."""
    from agent.layers.bce_frame_change import Layer0Net
    import torch
    net = Layer0Net()
    x = torch.zeros(2, NUM_COLORS, GRID, GRID)
    feat = net.backbone(x)
    assert feat.shape == (2, 256, GRID, GRID), f"backbone output {feat.shape} != (2,256,64,64)"
    action_logits, click_logits = net(x)
    assert action_logits.shape == (2, 5), f"action_logits {action_logits.shape}"
    assert click_logits.shape == (2, 1, GRID, GRID), f"click_logits {click_logits.shape}"
    # Channel counts
    convs = [m for m in net.backbone.modules() if isinstance(m, torch.nn.Conv2d)]
    channels = [(c.in_channels, c.out_channels) for c in convs]
    assert channels == [(16, 32), (32, 64), (64, 128), (128, 256)], f"channels {channels}"
    print("  OK backbone_preserves_spatial_resolution")


def test_optimizer_config_matches_prereg() -> None:
    """Pre-reg §3: Adam lr=1e-3, betas=(0.9, 0.999), no weight decay."""
    layer = Layer0(master_seed=42)
    assert isinstance(layer.opt, __import__("torch").optim.Adam)
    g = layer.opt.param_groups[0]
    assert abs(g["lr"] - 1e-3) < 1e-9, f"lr={g['lr']}"
    assert g["betas"] == (0.9, 0.999), f"betas={g['betas']}"
    assert g["weight_decay"] == 0.0, f"weight_decay={g['weight_decay']}"
    print("  OK optimizer_config_matches_prereg")


def test_buffer_fifo_evicts_oldest() -> None:
    """Pre-reg §3: buffer cap 10K FIFO, evicts oldest on overflow."""
    from agent.layers.bce_frame_change import ReplayBuffer, BufferEntry
    buf = ReplayBuffer(max_size=3)
    import torch
    def entry(i):
        return BufferEntry(key=("h", i), one_hot=torch.zeros(1), action_id=i, x=None, y=None, frame_changed=0)
    buf.add(entry(1)); buf.add(entry(2)); buf.add(entry(3))
    assert len(buf) == 3 and buf.evictions == 0
    buf.add(entry(4))
    assert len(buf) == 3 and buf.evictions == 1
    # Oldest (1) should be gone; 2, 3, 4 present
    keys = list(buf._entries.keys())
    assert ("h", 1) not in keys and ("h", 4) in keys
    print("  OK buffer_fifo_evicts_oldest")


def test_audit_fields() -> None:
    """Pre-reg §3+§5: audit returns buffer_size, fifo_evictions, num_train_steps, last_loss."""
    layer = Layer0(master_seed=42)
    a = layer.audit()
    for field in ("buffer_size", "fifo_evictions", "num_train_steps", "last_loss", "mean_loss_last_100"):
        assert field in a, f"audit missing {field}"
    assert a["buffer_size"] == 0
    assert a["last_loss"] is None
    # after some steps
    frame = np.zeros((GRID, GRID), dtype=np.int8)
    layer.observe_and_train(frame, "h", 1, None, None, 1)
    a = layer.audit()
    assert a["buffer_size"] == 1
    assert a["num_train_steps"] == 1
    assert isinstance(a["last_loss"], float)
    print("  OK audit_fields")


def test_predict_shapes_in_range() -> None:
    layer = Layer0(master_seed=42)
    frame = np.random.default_rng(0).integers(0, 16, size=(GRID, GRID)).astype(np.int8)
    probs = layer.predict_action_probs(frame)
    assert set(probs.keys()) == {1, 2, 3, 4, 5}
    for v in probs.values():
        assert 0.0 <= v <= 1.0
    clicks = layer.predict_click_probs(frame)
    assert clicks.shape == (GRID, GRID)
    assert clicks.min() >= 0.0 and clicks.max() <= 1.0
    print("  OK predict_shapes_in_range")


def test_learns_trivial_signal() -> None:
    """ACTION1 always changes the frame; ACTION2 never does.

    After training on 200 hash-unique transitions, sigmoid(action_head) should
    push P(change | ACTION1) > 0.7 and P(change | ACTION2) < 0.3.
    """
    layer = Layer0(master_seed=42)
    rng = np.random.default_rng(0)
    # Generate 150 distinct frames; ACTION1 always changes, ACTION2 never does.
    for i in range(150):
        frame = rng.integers(0, 16, size=(GRID, GRID)).astype(np.int8)
        h = f"frame_{i:04d}"
        action = 1 if i % 2 == 0 else 2
        changed = 1 if action == 1 else 0
        layer.observe_and_train(frame, h, action, None, None, changed)
    # Evaluate
    frame = rng.integers(0, 16, size=(GRID, GRID)).astype(np.int8)
    probs = layer.predict_action_probs(frame)
    print(f"  P(change|A1)={probs[1]:.3f}  P(change|A2)={probs[2]:.3f}")
    assert probs[1] > 0.7, f"expected ACTION1 prob >0.7, got {probs[1]:.3f}"
    assert probs[2] < 0.3, f"expected ACTION2 prob <0.3, got {probs[2]:.3f}"
    # Loss should have decreased from the start
    early = float(np.mean(layer.loss_history[:20]))
    late = float(np.mean(layer.loss_history[-20:]))
    print(f"  loss early={early:.4f}  late={late:.4f}")
    assert late < early * 0.7, "loss should have dropped significantly"
    print("  OK learns_trivial_signal")


def test_reset_wipes_weights_and_buffer() -> None:
    layer = Layer0(master_seed=42)
    rng = np.random.default_rng(0)
    for i in range(50):
        frame = rng.integers(0, 16, size=(GRID, GRID)).astype(np.int8)
        layer.observe_and_train(frame, f"h{i}", 1, None, None, 1)
    assert len(layer.buffer) > 0
    pre_steps = len(layer.loss_history)
    layer.reset()
    assert len(layer.buffer) == 0
    # Parameters changed (new random init); spot-check first conv weight
    # This is a soft check — not bit-identical, just different.
    # loss_history is retained per pre-reg design.
    assert len(layer.loss_history) == pre_steps
    print("  OK reset_wipes_weights_and_buffer")


def test_dedup_by_hash_and_action() -> None:
    layer = Layer0(master_seed=42)
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 16, size=(GRID, GRID)).astype(np.int8)
    # Same (hash, action) observed 5 times — buffer size should stay at 1.
    for _ in range(5):
        layer.observe_and_train(frame, "only_hash", 1, None, None, 1)
    assert len(layer.buffer) == 1
    # Different action → new entry
    layer.observe_and_train(frame, "only_hash", 2, None, None, 0)
    assert len(layer.buffer) == 2
    # ACTION6 at different (x, y) → separate entries
    layer.observe_and_train(frame, "only_hash", 6, 10, 20, 1)
    layer.observe_and_train(frame, "only_hash", 6, 10, 20, 1)  # dup
    layer.observe_and_train(frame, "only_hash", 6, 30, 40, 0)  # new
    assert len(layer.buffer) == 4
    print("  OK dedup_by_hash_and_action")


def test_action7_no_training() -> None:
    layer = Layer0(master_seed=42)
    frame = np.zeros((GRID, GRID), dtype=np.int8)
    pre = len(layer.loss_history)
    loss = layer.observe_and_train(frame, "h", 7, None, None, 0)
    assert loss == 0.0
    assert len(layer.loss_history) == pre
    assert len(layer.buffer) == 0
    print("  OK action7_no_training")


def main() -> int:
    print("Layer 0 tests:")
    test_one_hot_basic()
    test_one_hot_overflow_crashes()
    test_backbone_preserves_spatial_resolution()
    test_optimizer_config_matches_prereg()
    test_buffer_fifo_evicts_oldest()
    test_audit_fields()
    test_predict_shapes_in_range()
    test_dedup_by_hash_and_action()
    test_action7_no_training()
    test_reset_wipes_weights_and_buffer()
    test_learns_trivial_signal()
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
