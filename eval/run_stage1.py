"""Stage 1 runner: 3 games × N seeds × max_steps trajectories of C0 (Layers 0-3).

Pinned spec: paper/stage1-preregistration.md.
  - games: A=sb26, B=r11l, C=su15
  - seeds: [42, 7, 99, 1, 123]
  - max_steps = 2000
  - arm = "c0_layers03"

Writes JSONL traces to traces/stage1/<short>/c0_layers03/seed{N}.jsonl per
agent/trace_logger.py schema.

Usage:
    python eval/run_stage1.py                    # all games, all seeds
    python eval/run_stage1.py --games sb26       # single game
    python eval/run_stage1.py --seeds 42 --max-steps 100   # smoke
"""
from __future__ import annotations
import argparse
import sys
import os
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from arc_agi import Arcade

from agent.c0_agent import C0Agent
from agent.seed_replay import frame_hash, _ACTION_BY_VALUE
from agent.instrumentation import StepTimer
from agent.trace_logger import TraceLogger, StepRecord, TrajectorySummary


GAME_IDS = {
    "sb26": "sb26-7fbdac44",
    "r11l": "r11l-495a7899",
    "su15": "su15-1944f8ab",
}
DEFAULT_SEEDS = [42, 7, 99, 1, 123]
ARM = "c0_layers03"
STAGE = "stage1"
MAX_STEPS = 2000


def run_one(
    arcade: Arcade,
    game_short: str,
    master_seed: int,
    max_steps: int,
    base_dir: str = "traces",
    quiet: bool = False,
) -> dict:
    """Run a single (game, seed) trajectory. Returns audit + summary dict."""
    game_id = GAME_IDS[game_short]
    agent = C0Agent(master_seed=master_seed)
    env = arcade.make(game_id, seed=master_seed)
    obs = env.reset()
    prev_levels = int(obs.levels_completed)
    max_levels_cleared = prev_levels
    n_resets = 0
    actions_to_first_level_up: int | None = None

    src_frame = np.asarray(obs.frame[0])
    src_hash = frame_hash(obs)
    # Initialize L2 with the start state
    agent.layer2.observe_state(src_hash, list(obs.available_actions))

    t_start = time.perf_counter()

    with TraceLogger(STAGE, game_short, ARM, master_seed, base_dir=base_dir) as log:
        n_steps = 0
        for step in range(max_steps):
            n_steps = step + 1
            decision = agent.select_action(src_frame, src_hash, list(obs.available_actions))

            with StepTimer() as timer:
                obs = env.step(_ACTION_BY_VALUE[decision.action_id], data=decision.data)

            dst_frame = np.asarray(obs.frame[0])
            dst_hash = frame_hash(obs)
            score_delta = int(obs.levels_completed - prev_levels)
            frame_changed = dst_hash != src_hash

            agent.observe_transition(
                src_frame=src_frame,
                src_hash=src_hash,
                action_id=decision.action_id,
                x=decision.data["x"] if decision.data else None,
                y=decision.data["y"] if decision.data else None,
                dst_hash=dst_hash,
                dst_available_actions=list(obs.available_actions),
                frame_changed=frame_changed,
                score_delta=score_delta,
            )

            if score_delta > 0 and actions_to_first_level_up is None:
                actions_to_first_level_up = step + 1
            if score_delta > 0:
                # Reset model state for the new level per pre-reg §3.
                agent.reset_for_new_level()
            max_levels_cleared = max(max_levels_cleared, int(obs.levels_completed))

            log.log_step(StepRecord(
                step=step,
                frame_hash=dst_hash,
                action={"id": decision.action_id, "data": decision.data, "src": decision.source},
                reward=float(score_delta),
                score_delta=score_delta,
                frame_change=frame_changed,
                wall_ms=timer.wall_ms,
            ))

            prev_levels = int(obs.levels_completed)
            src_frame = dst_frame
            src_hash = dst_hash

            state_name = getattr(obs.state, "name", str(obs.state))
            if state_name == "WIN":
                break
            if state_name == "GAME_OVER":
                # Brooks-style: deaths are part of exploration. env.reset() and continue
                # within the same trajectory (resets count toward max_steps). Do NOT reset
                # agent state — accumulate exploration knowledge across deaths.
                n_resets += 1
                obs = env.reset()
                src_frame = np.asarray(obs.frame[0])
                src_hash = frame_hash(obs)
                agent.layer2.observe_state(src_hash, list(obs.available_actions))
                prev_levels = int(obs.levels_completed)  # restart level counter

        wall_seconds = time.perf_counter() - t_start

        log.log_summary(TrajectorySummary(
            trajectory=f"{STAGE}/{game_short}/{ARM}/seed{master_seed}",
            levels_cleared=max_levels_cleared,
            total_steps=n_steps,
            total_score=max_levels_cleared,
            actions_to_first_level_up=actions_to_first_level_up,
            wall_seconds=wall_seconds,
            cost_usd=0.0,
        ))

    audit = agent.audit()
    if not quiet:
        print(f"  {game_short} seed={master_seed:3d}: "
              f"steps={n_steps:4d} levels={max_levels_cleared} "
              f"first_lvl_up={actions_to_first_level_up} "
              f"resets={n_resets} "
              f"wall={wall_seconds:.1f}s "
              f"L2 states={audit['layer2']['n_states_visited']} "
              f"L3 active={audit['layer3']['activated']}")
    return {
        "game": game_short,
        "seed": master_seed,
        "n_steps": n_steps,
        "levels_cleared": max_levels_cleared,
        "actions_to_first_level_up": actions_to_first_level_up,
        "n_resets": n_resets,
        "wall_seconds": wall_seconds,
        "audit": audit,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="sb26,r11l,su15", help="comma-sep short game ids")
    ap.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)),
                    help="comma-sep ints")
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--base-dir", default="traces")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    games = [g.strip() for g in args.games.split(",") if g.strip()]
    seeds = [int(s) for s in args.seeds.split(",")]
    for g in games:
        if g not in GAME_IDS:
            raise ValueError(f"unknown game short {g!r}; allowed: {list(GAME_IDS)}")

    # Suppress chatty arc-agi INFO logs unless verbose
    logging.getLogger("arc_agi").setLevel(logging.WARNING)

    arcade = Arcade()
    print(f"Stage 1: games={games} seeds={seeds} max_steps={args.max_steps}")
    t0 = time.perf_counter()
    summaries = []
    for game in games:
        for seed in seeds:
            summaries.append(run_one(
                arcade, game, seed, args.max_steps,
                base_dir=args.base_dir, quiet=args.quiet,
            ))
    total_wall = time.perf_counter() - t0
    print(f"Done. Total wall: {total_wall:.1f}s ({total_wall/60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
