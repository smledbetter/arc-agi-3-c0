"""Per-step timing for Layer 0 BCE train_step. Determines Stage 1 feasibility on CPU."""
from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np

from agent.layers.bce_frame_change import Layer0, GRID

layer = Layer0(master_seed=0, num_threads=2)
rng = np.random.default_rng(0)

# Warm up — first call has graph construction + BN init + etc.
print("warming up (5 steps)...")
for i in range(5):
    frame = rng.integers(0, 16, size=(GRID, GRID)).astype(np.int8)
    layer.observe_and_train(frame, f"warm_{i}", 1, None, None, 1)

# Timed section
print("timing 50 steps...")
t0 = time.perf_counter()
for i in range(50):
    frame = rng.integers(0, 16, size=(GRID, GRID)).astype(np.int8)
    layer.observe_and_train(frame, f"bench_{i}", 1, None, None, 1)
dt = time.perf_counter() - t0

per_step_s = dt / 50
print(f"per-step: {per_step_s*1000:.1f} ms  ({dt:.1f}s / 50 steps)")
stage1_steps = 3 * 5 * 2000  # games × seeds × max_steps
projected = stage1_steps * per_step_s
print(f"Stage 1 projected wall: {projected/60:.1f} min  ({stage1_steps} steps)")
print(f"pre-reg §5 budget: 30 min → {'PASS' if projected < 1800 else 'FAIL'}")
