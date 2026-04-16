# Stage 1 Results

**Date:** 2026-04-16
**Pre-registration:** [`stage1-preregistration.md`](./stage1-preregistration.md), [`AMENDMENT-1`](./stage1-preregistration-AMENDMENT-1.md), [`AMENDMENT-2`](./stage1-preregistration-AMENDMENT-2.md)
**Code commit:** repository state at this commit; analyzer `tools/analyze_stage1.py`.
**Substrate:** vast.ai instance `35094112` (RTX A4000, 12 vCPU, 32 GB RAM), per Amendment 1.

## TL;DR

Stage 1 **FAILed the pre-registered PASS bar.** 1 of 3 games met the per-game threshold (≥3 of 5 seeds clearing ≥1 level); pre-reg requires ≥2 of 3. The pre-registered hypothesis (sb26 = convention-aligned = easiest; r11l = convention-shifted = hardest) was **inverted** — only r11l cleared, and it did so via random clicks rather than via Layers 0–3 doing useful work. Trace analysis identifies a specific failure mechanism: **Layer 0's frame-change BCE classifier carries almost no information when most actions change the frame**, which is the case on all three Stage 1 games. Per pre-reg §6, this triggers the FAIL diagnostic — expand to all 25 public games (1 seed each).

## 1. Setup recap

- **Games:** A=`sb26-7fbdac44`, B=`r11l-495a7899`, C=`su15-1944f8ab` (justifications in `tools/game_inventory.md`).
- **Seeds:** `[42, 7, 99, 1, 123]`.
- **Architecture:** C0 Layers 0–3 per pre-reg §3–4. Frame-change BCE (4-conv {32,64,128,256}, two heads), uniform-filtered wander, hash-state graph with BFS-frontier, score-delta back-label MLP with rerank-of-frontier picker.
- **Reset protocol:** model + L2 graph + L3 wiped on each `levels_completed` increment; preserved across env GAME_OVER auto-resets within a trajectory.
- **Termination:** `max_steps = 2000` per trajectory. GAME_OVER → `env.reset()` and continue (counts toward max_steps; no agent-state reset).
- **Total compute:** 15 trajectories on RTX A4000, wall ≈ 22 minutes, cost ≈ $0.04. Within Amendment 1's $0.20 budget.

## 2. Frozen aggregation per pre-reg §5

| Game | n_seeds | events | censored | KM median (actions to first level-up) | 90% bootstrap CI | levels_cleared (mean ± individual) | wall (s, mean) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **sb26** | 5 | 0 | 5 | inf (no event observed) | (inf, inf) | 0.00 (0,0,0,0,0) | 112.5 |
| **r11l** | 5 | 5 | 0 | **371** | (199, 1174) | 1.00 (1,1,1,1,1) | 84.7 |
| **su15** | 5 | 0 | 5 | inf (no event observed) | (inf, inf) | 0.00 (0,0,0,0,0) | 62.5 |

Method: percentile bootstrap, B=10,000 resamples at the seed level (per pre-reg §5; `tools/analyze_stage1.py` is the canonical implementation). Right-censoring at 2000 steps treated via Kaplan-Meier; bootstrap medians of all-censored resamples encoded as +inf.

### Action source mix (steps summed across 5 seeds per game)

| Game | available actions | L2-untested | L2-path | L2-bfs | L1-wander |
|---|---|---:|---:|---:|---:|
| sb26 | {5, 6, 7} | 2,212 | 5,435 | 564 | 1,788 |
| r11l | {6} | 2,535 | 6,788 | 631 | 46 |
| su15 | {6, 7} | 5,499 | 2,963 | 1,292 | 246 |

L2 (state-graph) dominates source attribution on all games — Layer 1 (wander) fires only when L2 has no untested action at the current state and no reachable visited frontier. r11l's L1 share is strikingly low (0.5%), reflecting that r11l keeps producing new states from each click, so L2-path replay drives nearly every step.

## 3. Plots

(see `paper/stage1_figs/`)

- `levels_cleared_per_game.png` — bar of mean levels cleared with seed-level scatter. Shows r11l = 1.0 ± 0, sb26 = su15 = 0.
- `action_source_mix.png` — stacked bar of L0/L1/L2/L3 contribution to step decisions per game.

Two plots from pre-reg §5 are deferred: per-step Layer 0 BCE loss curve and filter-fallback rate. Both require schema extension (`StepRecord` lacks `bce_loss` and `wander_audit` fields). Will be added in Stage 2's pre-registration so the trace already has them when Stage 2 runs.

## 4. Mechanism analysis (the actual finding)

The 0/15 → 5/15 → 0/15 result across A → B → C looks like a hypothesis inversion. The trace data shows it isn't, exactly — the pre-reg hypothesis was about a mechanism (visual-salience prior collapse) that turned out not to be load-bearing for our agent.

### 4.1 Layer 0's BCE filter is uninformative on these games

The 0.05 frame-change-probability threshold is meant to filter out actions that won't change the frame. But on all three Stage 1 games:

- **sb26:** 1608–1732 of 2000 steps changed the frame (80–87%). ACTION5 (the dominant keyboard action) changes the frame on essentially every step.
- **r11l:** 2000/2000 = 100% frame change rate.
- **su15:** 1740–1854 of 2000 (87–93%).

When most actions change the frame, Layer 0 has nothing to *learn* and nothing to *filter*. The 0.05 threshold either passes everything (no filter effect) or filters everything if the head's prior collapses to ~0 in the early random-init period (then triggers the all-filtered → uniform fallback). Either way, Layer 1's wander is effectively uniform-random over `available_actions`.

### 4.2 The click map gives no spatial signal without scoring

Layer 0's ACTION6 click head is `Conv2d(256, 1, 1×1) → sigmoid → 64×64`. The intent is a learned "click here = likely to change frame" map. But:

- The head only learns from observed (state, action_id, x, y, frame_changed) tuples.
- Most clicks change the frame (above), so the head trains toward a near-uniform "yes" everywhere.
- Without a scoring signal, the head has no way to differentiate "useful click here" from "useless click here."

Concretely: on r11l where every step is ACTION6, the level-up clicks across 5 seeds were `(56, 24)`, `(49, 29)`, `(26, 32)`, `(23, 10)`, `(14, 13)` — scattered across the grid with no obvious cluster. Random sampling from a near-uniform click map eventually hits the right pixel; ~370 clicks on average for r11l's level 1.

### 4.3 Layer 3 (score-follow) is dormant 2/3 of the time

L3 only trains after a positive `score_delta` and only modifies the frontier-rerank picker once activated. On sb26 and su15 (zero score deltas across all 5 seeds), L3 never trains and contributes zero policy bias. On r11l, L3 activates once per trajectory (after the level-1 clear), then immediately gets reset by the level-up reset protocol. Net effect: L3's MLP is essentially never used to direct exploration in Stage 1.

This is consistent with the pre-reg's "activation gate" choice (§4) but makes the gate an own-goal in sparse-reward regimes — the layer designed to *escape* sparse-reward exploration only fires *after* the agent has already escaped it.

### 4.4 What's actually driving Stage 1 outcomes

Stage 1's results sort by **click-affordance density**, not by visual convention:

- r11l: every click anywhere in the play area produces a frame change; one of those clicks (per level) produces a score event. Density of "scoring clicks" / "all clicks" is high enough that random sampling finds one.
- sb26: scoring requires both a specific action sequence (ACTION5 navigation) and specific click targets. Density is low; random exploration doesn't accumulate enough information to find them.
- su15: scoring requires specific click targets (no keyboard sequence). Click density of "scoring clicks" / "all clicks" is too low for random sampling within 2000 steps.

This is a refinement of the pre-reg hypothesis, not a refutation per se — visual conventions are correlated with click-affordance density (button UIs need *specific* clicks; gameplay-sprite UIs are more "click anywhere"), but our agent doesn't perceive the conventions; it only experiences the affordance density.

## 5. Pre-reg hypothesis vs observed

| | Pre-reg prediction | Observed | Implied mechanism |
|---|---|---|---|
| sb26 (A) | Layers 0–3 work; clears multiple levels | 0/5 cleared any level | Sequence-dependence + low click affordance density defeats random+graph |
| r11l (B) | Layers 0–3 struggle; needs Layer 4 | 5/5 cleared exactly 1 level | High click affordance + simple scoring rule allows random hit |
| su15 (C) | Partial: Layer 4 modest lift | 0/5 cleared any level | Click-only with low affordance density = same failure as sb26 |

The pre-reg's "visual-salience prior collapse" hypothesis (§4.6) predicts the wrong sort of variation. A revised, narrower hypothesis: **C0 (Layers 0–3 only) requires high click-affordance density to score; will clear games where ~any click in some region progresses, will fail on games requiring specific click targets.** The 22-game diagnostic (§7) tests this.

## 6. PASS/FAIL verdict (per pre-reg §6)

- PASS criterion: ≥1 level cleared on ≥2 of 3 games, with ≥3 of 5 seeds per passing game.
- Observed: 1 of 3 games (r11l) meets the per-game threshold (5/5 ≥ 3/5). 0 of 3 in the other direction.
- **Verdict: FAIL.**

Per pre-reg §6: "Stage 1 FAIL: expand to all 25 public games as diagnostic (budget permitting, ≤$200). If still <1 level cleared on ≥10 games, C0 fundamentally doesn't work → document and decide (pivot to C1-primary vs abandon). No retroactive Stage 1 hyperparameter adjustments."

No retroactive hyperparameter adjustments have been made. All Stage 1 numbers above are the frozen-spec output.

## 7. Diagnostic plan

**Run all 25 public games × 1 seed (seed=42, already validated for determinism), `max_steps=2000`, same architecture.**

- 22 new trajectories (3 already done at this seed).
- Wall ≈ 35 min on the same vast.ai instance.
- Cost ≈ $0.05.
- Decision rule (pre-registered here, before running): if ≥10 of 25 games clear ≥1 level, the click-affordance-density hypothesis (§4.4) is supported and Stage 2 (Layer 4) is the right next move. If <10 clear, C0 is fundamentally insufficient at this scale; switch to evaluating C1-primary or pivot.
- Held-out wrapper widened to allow all 25 game IDs for this diagnostic only; restored to A/B/C after.

## 8. Limitations

- n=5 seeds × 3 games is small. The 22-game diagnostic addresses the games dimension; n_seeds stays at 1 for the diagnostic to keep cost down. A FAIL on diagnostic with 1 seed could be an unlucky seed.
- "r11l PASS via random click" means the PASS isn't a positive signal about C0's exploration logic; it's a positive signal about r11l being click-affordance-dense. Don't read the 1/3 PASS as a partial endorsement.
- Bitwise GPU determinism is not enforced (Amendment 1 §A1.2). Trajectory data (frame hashes, action sequences) is reproducible from master seed; per-step Layer 0 loss values may differ at ~10⁻⁵ scale across re-runs. This affects nothing in the analysis above.
- The frame-change-as-target choice for Layer 0's BCE was inherited from StochasticGoose (`architecture-candidates.md` §4.1). Our finding (it's uninformative when most actions change the frame) suggests either (a) StochasticGoose's preview-game environments had different action-frame-change distributions, or (b) StochasticGoose's downstream layers compensated for the weak signal. Either is testable but out of Stage 1's scope.

## 9. Honest framing for engagement

Per testing-plan.md §11, public engagement was scoped to "after Stage 1 pass." Stage 1 did not pass. Engagement deferred at minimum until the 22-game diagnostic completes. If the diagnostic also FAILs (<10 of 25 games cleared), the natural next public artifact is a short writeup of the failure mechanism (§4) rather than a result claim — frame as "what we learned about why frame-change BCE doesn't carry exploration agents through sparse-reward games," which is a contribution even without a positive headline number.

---

**Status:** Stage 1 frozen. Diagnostic queued (§7). Stage 2 (Layer 4) is conditional on diagnostic outcome; if C0 fundamentally doesn't work, Layer 4 question reframes from "lift" to "rescue."
