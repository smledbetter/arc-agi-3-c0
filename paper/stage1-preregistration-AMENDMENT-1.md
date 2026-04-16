# Stage 1 Pre-Registration — Amendment 1

**Committed:** 2026-04-16
**Amends:** `paper/stage1-preregistration.md` (commit `923ce4b`)
**Reason:** Original §5 stated "Estimated wall time < 30 min total on CPU." Empirical measurement on the dev VPS (2 vCPU droplet) showed Layer 0 train_step at **4.6 s/step**, which projects to **38 hours** for the 30,000-step Stage 1 — 77× over budget. The wall-time estimate was wrong by two orders of magnitude.

**No experiments were run before this discovery.** The CPU bench (`tools/bench_layer0.py`) was the only execution; it measured per-step latency with no game environment, no scoring, no Stage 1 trajectories. No Stage 1 results have been observed.

The substrate switch from CPU to GPU is therefore a legitimate spec revision, not peek-and-tweak. Per the original pre-reg's §6 spirit ("Either the PASS numerical bar is met or the FAIL rule applies … no retroactive Stage 1 hyperparameter adjustments"), the rule is that we must not adjust the spec **after observing trial outcomes** — and we have not.

## Amendment

### A1.1. Substrate (extends §2 of the original)

- **Stage 1 substrate:** single CUDA GPU, RTX A4000-class or better (16+ GB VRAM, Ampere or newer), CUDA ≥ 12.4, PyTorch ≥ 2.5.
- **Reference box:** vast.ai instance ID `35094112` — RTX A4000, 12 vCPU, 32 GB RAM, image `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`, $0.086/hr.
- **Empirical bench (Layer 0 only, 50 steps, batch 32, 64×64 one-hot input):**
  - 2-vCPU droplet CPU: 4634 ms/step → 38 h projected
  - RTX A4000 GPU: 38 ms/step → 19 min projected
  - **Speedup: ~120×.**
- **Updated wall-time budget for §5:** Stage 1 ≤ 30 min on the reference box. If a future Stage 1 re-run uses different hardware, re-bench before running and re-confirm the 30-min budget.

### A1.2. Determinism on GPU (refines §2)

The original §2 said: *"Stage 3's LLM substrate is nondeterministic on GPU unless `torch.use_deterministic_algorithms(True)` + CUDA determinism env vars are set — Stage 3's pre-registration will specify the enforcement stack separately."*

The same caveat now applies to Stage 1 since Stage 1 also uses GPU.

- **Bitwise determinism is NOT enforced for Stage 1 on GPU.** `torch.use_deterministic_algorithms(True)` would force deterministic conv backward at substantial perf cost (verified empirically: ~3× slowdown for 4-conv nets on Ampere). The 30-min budget assumes non-deterministic algorithms.
- **What IS deterministic on GPU:** master seed → `torch.manual_seed(seed)` → reproducible weight init + CPU-side RNG (numpy.random.default_rng) → deterministic action choices.
- **What is NOT bitwise deterministic on GPU:** convolution backward (uses cuBLAS/cuDNN reduction order which can vary), tensor → CPU `.item()` conversion ordering. Two runs of the same trajectory will produce **identical action sequences and identical frame hashes** (because actions are chosen from CPU-side RNG before the next env.step), but **floating-point values in the BCE loss curve and per-action probabilities may differ at ~10⁻⁵ scale**.
- **Validator still in scope:** `tools/verify_seed_replay.py` validates trajectory-data identity (frame hashes + actions) modulo `wall_ms`, which is unaffected by GPU FP nondeterminism (its policy is uniform-random, no torch involvement).
- **Stage 2 / Stage 3 carry-forward:** since Layer 0 weights at end of trajectory are the seed-driven Adam state, and Adam SGD on GPU is non-bitwise-deterministic, the "PRNG-seed-identical across stages" claim from §2 means *the same seeds will be passed to all stages*; it does NOT mean Layer 0 weights at end of training will be bitwise-identical between Stage 1 and a re-run. This is acceptable — paired comparisons compare distributions over seeds, not single-trajectory floating-point identity.

### A1.3. Cost added to budget tracker (adds to original §5)

- Stage 1 GPU cost (vast.ai, RTX A4000 @ $0.086/hr × ≤1 h with margin) ≤ **$0.10**.
- Cumulative Stage 1 cost projection: ≤ **$0.20** including instance idle time during development of Layers 1–3.
- Within the original §3.1 ($50) and ≤ Stage 0 reserve ($10).

### A1.4. What is unchanged

All of §1, §3, §4, §6, §7 of the original pre-registration are unchanged. Hyperparameters (channel counts, lr, batch, buffer cap, filter threshold, reset protocol, decision rules) are still frozen.

## Amendment metadata

- Author: Steven Ledbetter (Claude Code session, 2026-04-16).
- Triggered by: Layer 0 CPU bench showing 4.6 s/step on dev VPS.
- No Stage 1 trajectories have been generated. No Stage 1 results have been observed. The amendment is purely about substrate and the wall-time budget.
- Reviewer: none (self-amended); the original skeptical-review pass from the agent was applied to v1.0 and the architecture remains as reviewed.
