"""JSONL trace logger. Schema pinned in execution-plan.md §3.2.

Per-step line:
  {"step": int, "frame_hash": str, "action": {...}, "reward": float,
   "score_delta": int, "frame_change": bool, "wall_ms": float}

Per-trajectory summary (final line, distinguished by "trajectory" key):
  {"trajectory": "...", "levels_cleared": int, "total_steps": int,
   "total_score": int, "actions_to_first_level_up": int|None,
   "wall_seconds": float, "cost_usd": float}
"""
from __future__ import annotations
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass
class StepRecord:
    step: int
    frame_hash: str
    action: dict[str, Any]
    reward: float
    score_delta: int
    frame_change: bool
    wall_ms: float


@dataclass
class TrajectorySummary:
    trajectory: str
    levels_cleared: int
    total_steps: int
    total_score: int
    actions_to_first_level_up: int | None
    wall_seconds: float
    cost_usd: float


class TraceLogger:
    """Writes JSONL trace for one trajectory.

    Usage:
        with TraceLogger(stage="stage1", game="ls20", arm="c0", seed=42) as log:
            log.log_step(StepRecord(...))
            ...
            log.log_summary(TrajectorySummary(...))
    """

    def __init__(
        self,
        stage: str,
        game: str,
        arm: str,
        seed: int,
        base_dir: str | os.PathLike = "traces",
    ) -> None:
        self.stage = stage
        self.game = game
        self.arm = arm
        self.seed = seed
        self.trajectory_id = f"{stage}/{game}/{arm}/seed{seed}"
        self.path = Path(base_dir) / stage / game / arm / f"seed{seed}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = None

    def __enter__(self) -> "TraceLogger":
        self._fh = open(self.path, "w", encoding="utf-8")
        return self

    def __exit__(self, *exc) -> None:
        if self._fh is not None:
            self._fh.close()

    def log_step(self, rec: StepRecord) -> None:
        assert self._fh is not None
        self._fh.write(json.dumps(asdict(rec), sort_keys=True) + "\n")

    def log_summary(self, summary: TrajectorySummary) -> None:
        assert self._fh is not None
        self._fh.write(json.dumps(asdict(summary), sort_keys=True) + "\n")
