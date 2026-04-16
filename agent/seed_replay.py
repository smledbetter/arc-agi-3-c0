"""Deterministic trajectory runner. Load-bearing for all paired comparisons.

Contract (execution-plan.md §3.1):
  - A single master PRNG seed drives the full exploration trajectory.
  - No out-of-band randomness (environment, library internals).
  - Running the same seed twice with identical no-op policy produces
    bitwise-identical action sequences and frame hashes.

The master seed is passed both to `arc.make(seed=...)` (SDK-level
determinism) and to a `numpy.random.default_rng(...)` instance that the
policy uses for every stochastic choice. Nothing else may read randomness.
"""
from __future__ import annotations
import hashlib
from dataclasses import dataclass
from typing import Callable, Any

import numpy as np
from arc_agi import Arcade, LocalEnvironmentWrapper
from arcengine import GameAction, FrameDataRaw

_ACTION_BY_VALUE = {a.value: a for a in GameAction}

from agent.instrumentation import StepTimer, CostTracker
from agent.trace_logger import TraceLogger, StepRecord, TrajectorySummary


# A policy returns (GameAction, data_dict_for_ACTION6_or_None)
Action = tuple[GameAction, dict[str, Any] | None]
Policy = Callable[[FrameDataRaw, np.random.Generator], Action]


def uniform_random_policy(obs: FrameDataRaw, rng: np.random.Generator) -> Action:
    """No-op policy for validator: uniform-random over available_actions.

    For ACTION6 (click), draws (x, y) uniformly in [0, 63].
    """
    avail = obs.available_actions  # list[int]
    action_id = int(rng.choice(avail))
    action = _ACTION_BY_VALUE[action_id]
    if action == GameAction.ACTION6:
        x = int(rng.integers(0, 64))
        y = int(rng.integers(0, 64))
        return action, {"x": x, "y": y}
    return action, None


def frame_hash(obs: FrameDataRaw) -> str:
    """Stable hash of the observation's grid stack. Used for graph keys + determinism checks."""
    h = hashlib.sha256()
    h.update(str(obs.state).encode())
    h.update(int(obs.levels_completed).to_bytes(4, "big", signed=False))
    for layer in obs.frame:
        arr = np.ascontiguousarray(np.asarray(layer, dtype=np.int8))
        h.update(arr.tobytes())
        h.update(b"|")
    return h.hexdigest()


@dataclass
class RunResult:
    trajectory_id: str
    total_steps: int
    levels_cleared: int
    actions_to_first_level_up: int | None
    wall_seconds: float
    final_frame_hash: str


def run_trajectory(
    arcade: Arcade,
    game_id: str,
    master_seed: int,
    max_steps: int,
    policy: Policy,
    stage: str,
    arm: str,
    base_dir: str = "traces",
) -> RunResult:
    """Run a single deterministic trajectory, logging each step.

    Returns a RunResult for verification. JSONL trace written to
    traces/<stage>/<game_id_short>/<arm>/seed<seed>.jsonl.
    """
    short_game = game_id.split("-", 1)[0]  # e.g. "ls20-9607627b" -> "ls20"
    rng = np.random.default_rng(master_seed)
    env: LocalEnvironmentWrapper = arcade.make(game_id, seed=master_seed)
    cost = CostTracker()

    import time as _time
    t_start = _time.perf_counter()

    prev_levels = 0
    actions_to_first_level_up: int | None = None

    with TraceLogger(stage, short_game, arm, master_seed, base_dir=base_dir) as log:
        obs = env.reset()
        prev_hash = frame_hash(obs)

        for step in range(max_steps):
            with StepTimer() as timer:
                action, data = policy(obs, rng)
                obs = env.step(action, data=data)

            new_hash = frame_hash(obs)
            score_delta = int(obs.levels_completed - prev_levels)
            if score_delta > 0 and actions_to_first_level_up is None:
                actions_to_first_level_up = step + 1
            prev_levels = int(obs.levels_completed)

            log.log_step(StepRecord(
                step=step,
                frame_hash=new_hash,
                action={"id": int(action.value), "data": data},
                reward=float(score_delta),  # sparse reward = level-up
                score_delta=score_delta,
                frame_change=(new_hash != prev_hash),
                wall_ms=timer.wall_ms,
            ))
            prev_hash = new_hash

            state_name = getattr(obs.state, "name", str(obs.state))
            if state_name in ("WIN", "GAME_OVER"):
                break

        wall_seconds = _time.perf_counter() - t_start
        total_steps = step + 1

        trajectory_id = f"{stage}/{short_game}/{arm}/seed{master_seed}"
        log.log_summary(TrajectorySummary(
            trajectory=trajectory_id,
            levels_cleared=prev_levels,
            total_steps=total_steps,
            total_score=prev_levels,
            actions_to_first_level_up=actions_to_first_level_up,
            wall_seconds=wall_seconds,
            cost_usd=cost.usd_total,
        ))

    return RunResult(
        trajectory_id=trajectory_id,
        total_steps=total_steps,
        levels_cleared=prev_levels,
        actions_to_first_level_up=actions_to_first_level_up,
        wall_seconds=wall_seconds,
        final_frame_hash=prev_hash,
    )
