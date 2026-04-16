# Stage 1 Pre-Registration

**Committed:** 2026-04-16
**Covers:** C0 Layers 0–3 on games A=sb26, B=r11l, C=su15.
**Purpose:** Lock in Stage 1 design choices BEFORE writing layer code so accidental peek-and-tweak is structurally impossible. All numeric hyperparameters in §2–6 are immutable once this file is committed; changes require a new pre-registration with a fresh set of seeds.

**Scope note:** Stage 1 is a **runability gate**, not a competitiveness gate. The PASS bar in §6 ("≥1 level on ≥2/3 games") is deliberately weak relative to published preview baselines (StochasticGoose 12.58% RHAE, dolphin-in-a-coma median 17/25). Its job is to confirm that Layers 0–3 are not broken before we spend effort on Layer 4 (Stage 2) and the C1/C1' ablation (Stage 3). Competitive evaluation happens at Stage 5 with the full public set and bootstrap CI. Future reviewers should not interpret a Stage 1 PASS as a positive claim about C0's competitiveness.

## 1. Games (committed in `tools/game_inventory.md`)

- **A = sb26-7fbdac44** (convention-aligned; 8 levels; baseline_sum 213; tag `keyboard_click`)
- **B = r11l-495a7899** (convention-shifted; 6 levels; baseline_sum 233; tag `click`)
- **C = su15-1944f8ab** (mixed; 9 levels; baseline_sum 361; tag `click`)

Engineering blinding enforced by `eval/held_out_wrapper.py` until Stage 4.

## 2. Seeds

Master PRNG seeds: **`[42, 7, 99, 1, 123]`**. Each seed drives (a) `arc.make(game_id, seed=...)` and (b) `numpy.random.default_rng(seed)` for every stochastic choice in the policy.

**Determinism claim (honest scope):** PRNG-seed-identical across stages — the same integer seeds will be fed into both master RNGs for Stage 2 (Layer-4 ablation) and Stage 3 (C1/C1' ablations), enabling paired-difference comparison. Bitwise-identical trajectories are only guaranteed when the policy is identical to Stage 1's (verified by `tools/verify_seed_replay.py`). Stage 2's Layer 4 introduces new deterministic state that will diverge the trajectory; Stage 3's LLM substrate is nondeterministic on GPU unless `torch.use_deterministic_algorithms(True)` + CUDA determinism env vars are set — Stage 3's pre-registration will specify the enforcement stack separately.

## 3. Layer 0 — Frame-change BCE classifier

**Spec (pinned from `architecture-candidates.md` §4.1):**

- Input tensor: `(16, 64, 64)` one-hot encoding of the grid. 16 channels = ARC palette. If a game emits values > 15, crash loudly — do not silently saturate.
- Backbone: 4 conv layers `{32, 64, 128, 256}` channels, kernel 3×3, stride 1, pad 1, batchnorm + ReLU between. Spatial resolution preserved end-to-end.
- Head A (ACTION1–5): global-avg-pool on the final feature map → `Linear(256, 5)` → sigmoid. Per-action frame-change probability.
- Head B (ACTION6 click): `Conv2d(256, 1, kernel=1)` → sigmoid → 64×64 probability map. Each position is the predicted frame-change probability for a click at that coordinate.

**Training:**

- Per-step online SGD. After each environment step, one gradient update.
- Optimizer: Adam, `lr=1e-3`, `betas=(0.9, 0.999)`, no weight decay.
- Batch size: 32, sampled uniformly from the replay buffer (if buffer has < 32 unique entries, train on what's there).
- Loss: per-transition BCE against the observed bit `frame_change ∈ {0, 1}`. For ACTION1–5, loss is BCE on the scalar head output for that action. For ACTION6(x, y), loss is BCE at the single (x, y) position of the output map; other positions are masked out of the loss.
- No gradient clipping; no learning-rate schedule.

**Hyperparameter provenance and status:**

- Channel counts `{32, 64, 128, 256}`, two-head split, and BCE-on-frame-change-bit come from `architecture-candidates.md` §4.1, which cites StochasticGoose (arXiv 2512.24156 writeup + github.com/DriesSmit/ARC3-solution).
- Adam `lr=1e-3`, batch=32, buffer=10K, no-clipping, no-schedule are untuned first-pass defaults chosen without pilot experiments. **If Stage 1 fails, these are the first diagnostics to revisit** — before declaring C0 non-functional, verify that Layer 0 is actually learning by inspecting per-step BCE loss curves and the rate of filtered-out actions.

**Replay buffer (hash-deduplicated):**

- Key for ACTION1–5: `(frame_hash, action_id)` — one entry per unique (state, action) pair.
- Key for ACTION6: `(frame_hash, 6, x, y)`.
- Value: `(one_hot_tensor, action_id, x_or_none, y_or_none, frame_changed_bit)`.
- Max size: 10,000 entries. FIFO eviction.
- **Audit requirement:** per trajectory, log final buffer size and whether FIFO eviction was ever triggered. If >95% of trajectories never hit 10K, the cap is cosmetic and should be reduced in the Stage 2 pre-registration (not silently).

**Reset protocol (with acknowledged risk):**

- Model weights re-initialized and replay buffer cleared **whenever `levels_completed` increments** (i.e., a level-up is detected). This matches `architecture-candidates.md` §4.1 spec ("Model and buffer reset between levels") and avoids cross-level behavior carry-over.
- Across seeds, a fresh model is created at trajectory start. Across games, fresh everything.
- **Known risk:** baseline_sums imply ~30–60 actions per level, so Layer 0 gets only ~40 gradient updates before a reset wipes it. Levels 2+ start with a random-init classifier and a near-empty buffer; Layers 1–3 carry most of the signal until Layer 0 warms back up. If Stage 1 FAILs on games A or C specifically, the reset protocol is the first diagnostic to revisit — test a variant that resets only the replay buffer (not the weights) and document as a supplemental finding, not a retroactive Stage 1 change.

## 4. Layers 1–3 specs (pinned)

- **Layer 1 (wander):** uniform sample from `obs.available_actions` filtered by Layer 0's per-action probability threshold (keep action `a` if `P(frame_change | a) ≥ 0.05`). If all actions are filtered out, fall back to uniform over unfiltered `available_actions`.
  - **Filter-health audit:** log per-step whether the filter rejected ≥1 action and whether the fallback (all-filtered → uniform) fired. **Pre-committed calibration rule:** if >50% of steps across the 15 Stage 1 trajectories trigger the fallback, the 0.05 threshold is miscalibrated. Document as a diagnostic in Stage 1 results; Layer 0's learning curve is probably the root cause. Do NOT retroactively adjust the threshold in Stage 1 — calibrate for Stage 2 via a fresh pre-registration.

- **Layer 2 (state graph):** hashed-state directed graph with edges keyed `(frame_hash_src, action, frame_hash_dst)`. When the graph has reachable states with untested legal actions, BFS to the nearest such state and take the untested action. Suppresses Layer 1 when it fires.

- **Layer 3 (score-delta back-label):** on positive `score_delta`, back-label every state on the traversed path with `distance_to_milestone = steps_since_hash`.
  - Regressor: MLP, 3 hidden layers × 128 units, ReLU. Input: `(frame_hash_embedding, action_id_one_hot_7)`. Output: scalar value predicting distance-to-milestone.
  - `frame_hash_embedding`: 64-dim vector derived deterministically from the SHA256 hash bytes — reshape first 512 bits to 64 × int8 then cast to float and scale to ±1. No learned projection. Reproducible across seeds.
  - Back-label horizon: clip `distance_to_milestone` at 50. States >50 steps before the milestone are not labeled.
  - Optimizer: Adam `lr=1e-3`, Huber loss (delta=1.0), batch=32, no schedule. Same first-pass-untuned caveat as Layer 0.
  - Retrain cadence: one gradient step after each new back-label is added to the training set.
  - **Combination with L2:** Layer 3 does NOT override Layer 2's BFS target. When Layer 2 has a reachable untested-action frontier, L3 reranks the frontier candidates by predicted value (ascending distance-to-milestone) and picks the lowest-distance one. If Layer 3 has never trained (no positive deltas yet), L2 uses its default BFS nearest-first order.
  - **Activation gate:** Layer 3 is dormant until the first positive `score_delta` in the current level. Before then, it emits no preference and L2's default order applies.

Subsumption order (highest priority wins): **L3-rerank-of-L2 > L2 > L1 > L0-filter**.

## 5. Stage 1 evaluation protocol and frozen analyses

- Per `(game, seed)`: one trajectory, `max_steps = 2000`, `arm = "c0_layers03"`.
- 3 games × 5 seeds = 15 trajectories. Estimated wall time < 30 min total on CPU.
- Artifacts: JSONL traces under `traces/stage1/<short>/c0_layers03/seed{N}.jsonl` per `agent/trace_logger.py` schema.

**Frozen aggregation (moved up from §7 per review):**

- **Primary metric:** `actions_to_first_level_up` per `(game, seed)`, treated as right-censored at `max_steps=2000`. If no level-up occurred in a trajectory, the value is recorded as `None` in the JSONL summary and treated as right-censored in aggregation (NOT imputed, NOT dropped).
- **Per-game aggregation:** Kaplan-Meier median over the 5 seeds. Ties at 2000 resolve as censored. Primary reported statistic: per-game median actions-to-first-level-up + 90% bootstrap CI.
- **Bootstrap method:** percentile bootstrap, B=10,000 resamples at the (seed)-level (nonparametric, stratified by game). Not BCa, not basic — percentile, frozen.
- **Secondary metrics:** total `levels_cleared` per `(game, seed)`; wall seconds per trajectory; filter-fallback rate (§4); buffer-eviction audit (§3).
- **Plot set (frozen before running):**
  1. Per-game strip plot of 5 `actions_to_first_level_up` values (censored as upward arrows at 2000).
  2. Per-game bar of `levels_cleared` mean across seeds.
  3. Layer 0 BCE loss trace (per-step, per-game median ± IQR across seeds).
  4. Filter-fallback rate per-game.
- **Analysis script:** `tools/analyze_stage1.py`. Signature: `aggregate(traces_dir: Path, bootstrap_B=10000) -> Stage1Results`. May be iterated during debugging for non-numerical issues (logging, plot cosmetics); numerical outputs must match the frozen spec above.

## 6. Pre-committed decision rules (mirror testing-plan.md §5, with tightening flagged)

- **Stage 1 PASS:** ≥1 level cleared on ≥2 of 3 games, with ≥3 of 5 seeds clearing ≥1 level on each passing game. Proceed to Stage 2 (Layer 4).
  - **Tightening vs testing-plan.md §4:** the plan states only "≥1 level on ≥2 of 3 games" without a seed sub-threshold. This pre-reg adds "≥3 of 5 seeds per passing game" to reduce the chance that a single lucky seed on a single game produces a PASS. Tightening acknowledged — reduces conjunction feasibility below testing-plan §7's 70% per-gate estimate. Consider this the honest bar.

- **Stage 1 FAIL (0 levels cleared on ≥2 of 3 games, or the per-game seed threshold not met):** expand to all 25 public games as diagnostic (budget permitting, ≤$200). If still <1 level cleared on ≥10 games, C0 fundamentally doesn't work → document and decide (pivot to C1-primary vs abandon). No retroactive Stage 1 hyperparameter adjustments.

- **No AMBIGUOUS clause.** Either the PASS numerical bar is met or the FAIL rule applies. Previous drafts included a clause allowing two additional seeds on marginal outcomes; that clause is dropped because adding seeds after inspecting 15 trajectories is equivalent to optional stopping / "two more subjects" p-hacking.

## 7. What is NOT pre-committed (may be iterated during Stage 1 debugging)

- Exact bytes of `frame_hash` (currently SHA256 of grid bytes + state + level). Collision rate is load-bearing only at very long trajectories; debug if collisions observed.
- Logging verbosity / INFO-line suppression.
- Cosmetic fields in `tools/analyze_stage1.py` plots (titles, colors, axis labels). Numerical outputs are frozen in §5.
- Stage 1 narrative writeup in `paper/stage1_results.md` (this is a reporting artifact, not a decision input).

Anything in §1–6 is frozen until Stage 1 completes.

## 8. Pre-registration metadata

- Git commit containing this file: (committed alongside this file).
- Claude Code session: 2026-04-16.
- Author: Steven Ledbetter <smledbetter@gmail.com>.
