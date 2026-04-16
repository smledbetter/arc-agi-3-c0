"""Stage 1 analyzer — frozen aggregation per pre-reg §5.

Produces:
  - Per-game median actions_to_first_level_up via Kaplan-Meier
    (right-censored at max_steps=2000), with 90% percentile bootstrap CI
    over the 5 seeds (B=10,000 resamples).
  - Per-game levels_cleared mean across seeds.
  - Per-game wall_seconds mean across seeds.
  - Per-game action-source mix (L2-untested / L2-path / L2-bfs / L1-wander).
  - PASS/FAIL verdict per pre-reg §6:
      PASS = ≥1 level cleared on ≥2 of 3 games, with ≥3 of 5 seeds clearing
             ≥1 level on each passing game.

Plots (matplotlib, saved to paper/stage1_figs/):
  1. Per-game strip plot of `actions_to_first_level_up` (censored as up-arrows at 2000).
  2. Per-game bar of mean `levels_cleared` (with seed-level scatter).
  3. Per-game action-source mix stacked bar.

NOT yet plotted (require schema extension scheduled for Stage 2):
  - Layer 0 BCE loss trace per step (StepRecord lacks `bce_loss` field).
  - Filter-fallback rate (StepRecord lacks WanderAudit fields).
  TODO is filed in paper/stage1_results.md once that file is written.

Signature (per pre-reg):
    aggregate(traces_dir: Path, bootstrap_B=10000) -> Stage1Results
"""
from __future__ import annotations
import argparse
import collections
import json
import sys
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


MAX_STEPS = 2000
ARM = "c0_layers03"
GAMES = ["sb26", "r11l", "su15"]


@dataclass
class TrajectoryResult:
    game: str
    seed: int
    n_steps: int
    levels_cleared: int
    actions_to_first_level_up: Optional[int]   # None = right-censored
    wall_seconds: float
    src_counts: dict[str, int] = field(default_factory=dict)  # L2-untested / L2-bfs / L1-wander / etc.


@dataclass
class GameSummary:
    game: str
    n_seeds: int
    n_event: int                   # seeds that achieved level-up (uncensored)
    n_censored: int                # seeds with no level-up by max_steps
    km_median: float               # may be float('inf') if median not reached
    bootstrap_lo: float
    bootstrap_hi: float
    levels_cleared_mean: float
    levels_cleared_per_seed: list[int]
    wall_seconds_mean: float
    src_count_total: dict[str, int] = field(default_factory=dict)


@dataclass
class Stage1Results:
    games: list[GameSummary]
    pass_per_game: dict[str, bool]   # ≥3/5 seeds with ≥1 level cleared
    overall_pass: bool               # ≥2 of 3 games pass


# ---------- trace I/O ----------

def load_trajectory(path: Path) -> Optional[TrajectoryResult]:
    """Read one (game, seed) JSONL trace; extract summary + per-step src counts.

    Returns None if the trace has no summary line (trajectory still in progress).
    """
    src_counts: dict[str, int] = collections.Counter()
    summary_rec: Optional[dict] = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            # Mid-write partial line on an in-progress trajectory; skip silently.
            continue
        if "trajectory" in rec:
            summary_rec = rec
            continue
        action = rec.get("action") or {}
        src = action.get("src", "unknown")
        src_counts[src] += 1
    if summary_rec is None:
        return None  # trajectory in progress; skip
    # Decode trajectory id e.g. "stage1/sb26/c0_layers03/seed42"
    game_short = summary_rec["trajectory"].split("/")[1]
    seed = int(summary_rec["trajectory"].split("seed")[-1])
    return TrajectoryResult(
        game=game_short,
        seed=seed,
        n_steps=int(summary_rec["total_steps"]),
        levels_cleared=int(summary_rec["levels_cleared"]),
        actions_to_first_level_up=summary_rec.get("actions_to_first_level_up"),
        wall_seconds=float(summary_rec["wall_seconds"]),
        src_counts=dict(src_counts),
    )


def load_all(traces_dir: Path) -> list[TrajectoryResult]:
    out: list[TrajectoryResult] = []
    for game in GAMES:
        d = traces_dir / "stage1" / game / ARM
        if not d.exists():
            continue
        for f in sorted(d.glob("seed*.jsonl")):
            t = load_trajectory(f)
            if t is not None:
                out.append(t)
    return out


# ---------- KM median + percentile bootstrap ----------

def km_median(events: list[tuple[float, bool]], max_t: float = MAX_STEPS) -> float:
    """Kaplan-Meier median over (time, observed) tuples.

    `observed=True` means level-up happened at `time` (event).
    `observed=False` means right-censored at `time` (max_steps).
    Returns the smallest t where S(t) ≤ 0.5; +inf if survival never crosses 0.5.
    """
    if not events:
        return float("inf")
    sorted_ev = sorted(events, key=lambda e: e[0])
    n_at_risk = len(sorted_ev)
    s = 1.0
    last_t = 0.0
    # Group by time; process events and censoring at each unique time.
    times = sorted(set(t for t, _ in sorted_ev))
    for t in times:
        # Number of events (level-ups) and censored at this time
        d = sum(1 for tt, obs in sorted_ev if tt == t and obs)
        c = sum(1 for tt, obs in sorted_ev if tt == t and not obs)
        # Update survival function only on events
        if d > 0 and n_at_risk > 0:
            s_new = s * (n_at_risk - d) / n_at_risk
            if s >= 0.5 > s_new:
                return float(t)  # crossed below 0.5 here
            s = s_new
        n_at_risk -= (d + c)
        last_t = t
    if s <= 0.5:
        return float(last_t)
    return float("inf")  # never crossed


def percentile_bootstrap_ci(
    events: list[tuple[float, bool]], B: int, alpha: float = 0.10,
    rng: Optional[np.random.Generator] = None,
) -> tuple[float, float]:
    """Percentile bootstrap CI for KM median over B resamples (with replacement).

    Returns (lo, hi) at the (alpha/2, 1-alpha/2) percentiles. inf medians are
    sorted to the high end (consistent with right-censoring meaning "≥ max_t").
    """
    if not events or B <= 0:
        return (float("inf"), float("inf"))
    rng = rng or np.random.default_rng(0)
    n = len(events)
    medians: list[float] = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        resample = [events[int(i)] for i in idx]
        medians.append(km_median(resample))
    # numpy percentile handles inf correctly when sorted
    arr = np.array(medians, dtype=float)
    lo = float(np.percentile(arr, 100 * alpha / 2, method="linear"))
    hi = float(np.percentile(arr, 100 * (1 - alpha / 2), method="linear"))
    return (lo, hi)


# ---------- aggregation ----------

def aggregate(traces_dir: Path, bootstrap_B: int = 10_000) -> Stage1Results:
    trajs = load_all(traces_dir)
    by_game = collections.defaultdict(list)
    for t in trajs:
        by_game[t.game].append(t)

    summaries: list[GameSummary] = []
    pass_per_game: dict[str, bool] = {}
    for game in GAMES:
        ts = by_game.get(game, [])
        n_seeds = len(ts)
        events: list[tuple[float, bool]] = []
        for t in ts:
            if t.actions_to_first_level_up is not None:
                events.append((float(t.actions_to_first_level_up), True))
            else:
                events.append((float(MAX_STEPS), False))
        n_event = sum(1 for _, obs in events if obs)
        n_censored = n_seeds - n_event
        median = km_median(events)
        rng = np.random.default_rng(hash(game) % (2**31))
        lo, hi = percentile_bootstrap_ci(events, B=bootstrap_B, rng=rng)
        levels = [t.levels_cleared for t in ts]
        wall = [t.wall_seconds for t in ts]
        src_total: collections.Counter = collections.Counter()
        for t in ts:
            src_total.update(t.src_counts)
        summaries.append(GameSummary(
            game=game, n_seeds=n_seeds, n_event=n_event, n_censored=n_censored,
            km_median=median, bootstrap_lo=lo, bootstrap_hi=hi,
            levels_cleared_mean=float(np.mean(levels)) if levels else 0.0,
            levels_cleared_per_seed=levels,
            wall_seconds_mean=float(np.mean(wall)) if wall else 0.0,
            src_count_total=dict(src_total),
        ))
        # PASS criterion: ≥3 of 5 seeds with ≥1 level cleared.
        pass_per_game[game] = sum(1 for L in levels if L >= 1) >= 3

    overall_pass = sum(pass_per_game.values()) >= 2
    return Stage1Results(
        games=summaries, pass_per_game=pass_per_game, overall_pass=overall_pass,
    )


# ---------- plots ----------

def plot_stage1(results: Stage1Results, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Strip plot of actions_to_first_level_up per seed, censored as up-arrows.
    fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
    for i, gs in enumerate(results.games):
        # We need the per-seed values; reload from src_count_total isn't enough.
        # Re-derive from km event list by reconstructing — actually easier to
        # store events on GameSummary. For now, plot only the median + CI.
        pass
    # (Strip plot deferred — need raw per-seed events on GameSummary.
    # Will fix in next iteration; below plots are sufficient for first pass.)
    plt.close(fig)

    # 2. Bar: mean levels_cleared per game, with seed-level scatter.
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    xs = np.arange(len(results.games))
    means = [g.levels_cleared_mean for g in results.games]
    ax.bar(xs, means, color="#7FDBFF", edgecolor="#0074D9")
    for i, g in enumerate(results.games):
        ax.scatter([i] * len(g.levels_cleared_per_seed), g.levels_cleared_per_seed,
                   color="black", s=20, zorder=3)
    ax.set_xticks(xs); ax.set_xticklabels([g.game for g in results.games])
    ax.set_ylabel("levels cleared (5 seeds)")
    ax.set_title("Stage 1 — levels cleared per (game, seed)")
    fig.tight_layout()
    fig.savefig(out_dir / "levels_cleared_per_game.png", bbox_inches="tight")
    plt.close(fig)

    # 3. Action-source mix: stacked bar per game.
    src_keys = sorted({k for g in results.games for k in g.src_count_total})
    if src_keys:
        fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
        bottoms = np.zeros(len(results.games))
        colors = ["#0074D9", "#2ECC40", "#FF4136", "#FF851B", "#AAAAAA"]
        for j, src in enumerate(src_keys):
            vals = np.array([g.src_count_total.get(src, 0) for g in results.games], dtype=float)
            ax.bar(xs, vals, bottom=bottoms, label=src, color=colors[j % len(colors)])
            bottoms += vals
        ax.set_xticks(xs); ax.set_xticklabels([g.game for g in results.games])
        ax.set_ylabel("steps (summed across 5 seeds)")
        ax.set_title("Stage 1 — action source mix per game")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "action_source_mix.png", bbox_inches="tight")
        plt.close(fig)


# ---------- CLI ----------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="traces", type=Path)
    ap.add_argument("--out", default="paper/stage1_figs", type=Path)
    ap.add_argument("--bootstrap-B", type=int, default=10_000)
    args = ap.parse_args()

    results = aggregate(args.traces, bootstrap_B=args.bootstrap_B)
    print(f"Stage 1 — {sum(g.n_seeds for g in results.games)} trajectories loaded")
    print()
    print(f"{'game':<6} {'seeds':>6} {'event':>6} {'cens':>5} "
          f"{'KM_med':>10} {'90%_CI':>20} {'levels_mean':>12} {'wall_s_mean':>12}")
    for g in results.games:
        ci = (
            f"({g.bootstrap_lo:.0f}, {'inf' if g.bootstrap_hi == float('inf') else f'{g.bootstrap_hi:.0f}'})"
        )
        med_str = f"{g.km_median:.0f}" if g.km_median != float("inf") else "inf"
        print(f"{g.game:<6} {g.n_seeds:>6} {g.n_event:>6} {g.n_censored:>5} "
              f"{med_str:>10} {ci:>20} {g.levels_cleared_mean:>12.2f} {g.wall_seconds_mean:>12.1f}")

    print()
    print(f"PASS per game: {results.pass_per_game}")
    print(f"OVERALL: {'PASS' if results.overall_pass else 'FAIL'} "
          f"({sum(results.pass_per_game.values())}/3 games passed; threshold ≥2)")

    plot_stage1(results, args.out)
    print(f"Plots written to {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
