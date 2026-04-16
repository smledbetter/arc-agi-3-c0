# Stage 1 Pre-Registration — Amendment 2

**Committed:** 2026-04-16
**Amends:** `paper/stage1-preregistration.md` (commit `923ce4b`), specifically §4 Layer 3 `frame_hash_embedding`.
**Reason:** Internal contradiction in original §4 wording — discovered during Layer 3 implementation. No experiments have been run.

## What the original said

> `frame_hash_embedding`: 64-dim vector derived deterministically from the SHA256 hash bytes — reshape first 512 bits to 64 × int8 then cast to float and scale to ±1. No learned projection. Reproducible across seeds.

## The problem

SHA256 produces 256 bits (32 bytes), not 512 bits. "First 512 bits of SHA256" is impossible.

## Resolution (this amendment)

- Compute `expanded = SHA512(frame_hash.encode())` — yields 512 bits (64 bytes).
- For each of the 64 bytes, map `byte > 127 → +1.0`, else `-1.0`.
- Result: 64-d ±1 float vector. Deterministic, reproducible, no learned projection.

The original `frame_hash` produced by `agent/seed_replay.py:frame_hash()` is unchanged — still SHA256(state + level + grid bytes). Only the `hash_to_embedding` function used inside Layer 3's MLP input changes.

Implementation reference: `agent/layers/score_follow.py:hash_to_embedding()`.

## What is unchanged

All other §4 Layer 3 hyperparameters: MLP architecture (3 × 128 ReLU), Adam lr=1e-3, Huber loss δ=1.0, batch=32, back-label horizon=50, activation gate, rerank-not-override combination with L2.

## Metadata

- Author: Steven Ledbetter (Claude Code session, 2026-04-16).
- Triggered by: `test_hash_embedding_deterministic` failed assertion `e1.shape == (64,)` — got 32, because SHA256 hex has 64 chars = 32 bytes, not 64 bytes.
- No Stage 1 trajectories have been generated. Pre-Reg integrity preserved.
