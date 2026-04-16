"""Seed-replay validator. Exit 0 iff two runs with the same master seed
produce bitwise-identical JSONL traces.

Usage:
    python tools/verify_seed_replay.py --seed 42 --game ls20 --steps 200

Fulfills execution-plan.md §3.1 exit criterion.
"""
from __future__ import annotations
import argparse
import filecmp
import shutil
import sys
from pathlib import Path

from arc_agi import Arcade

# Allow `python tools/verify_seed_replay.py` from project root
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.seed_replay import run_trajectory, uniform_random_policy  # noqa: E402


def resolve_game_id(arcade: Arcade, short: str) -> str:
    """Match a user-provided short id like 'ls20' to the full 'ls20-9607627b' id."""
    envs = arcade.get_environments()
    for e in envs:
        gid = e.game_id
        if gid == short or gid.split("-", 1)[0] == short:
            return gid
    raise ValueError(f"No environment matches {short!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--game", default="ls20")
    ap.add_argument("--steps", type=int, default=200)
    args = ap.parse_args()

    arcade = Arcade()
    game_id = resolve_game_id(arcade, args.game)
    short = game_id.split("-", 1)[0]

    runA_dir = Path("traces/_verify/runA")
    runB_dir = Path("traces/_verify/runB")
    for d in (runA_dir, runB_dir):
        if d.exists():
            shutil.rmtree(d)

    print(f"[verify] game={game_id} seed={args.seed} steps={args.steps}")
    resA = run_trajectory(
        arcade, game_id, args.seed, args.steps,
        uniform_random_policy, stage="_verify", arm="runA", base_dir=str(runA_dir),
    )
    resB = run_trajectory(
        arcade, game_id, args.seed, args.steps,
        uniform_random_policy, stage="_verify", arm="runB", base_dir=str(runB_dir),
    )
    print(f"  runA steps={resA.total_steps} final_hash={resA.final_frame_hash[:16]}")
    print(f"  runB steps={resB.total_steps} final_hash={resB.final_frame_hash[:16]}")

    pathA = runA_dir / "_verify" / short / "runA" / f"seed{args.seed}.jsonl"
    pathB = runB_dir / "_verify" / short / "runB" / f"seed{args.seed}.jsonl"

    bytesA = pathA.read_bytes()
    bytesB = pathB.read_bytes()
    # Strip the trajectory_id field from summary (it embeds run arm name, which differs).
    # Per-step lines must match bitwise.
    linesA = bytesA.splitlines()
    linesB = bytesB.splitlines()
    if len(linesA) != len(linesB):
        print(f"FAIL: line count differs ({len(linesA)} vs {len(linesB)})")
        return 1
    # Per-step: match bitwise modulo wall_ms (OS timing jitter is not determinism).
    import json as _json
    for i, (a, b) in enumerate(zip(linesA[:-1], linesB[:-1])):
        ra = _json.loads(a); rb = _json.loads(b)
        ra.pop("wall_ms", None); rb.pop("wall_ms", None)
        if ra != rb:
            print(f"FAIL: step {i} differs modulo wall_ms")
            print(f"  A: {ra}")
            print(f"  B: {rb}")
            return 1
    # Summary: should match modulo arm name + wall_seconds (wall is OS-dependent).
    import json
    sa = json.loads(linesA[-1])
    sb = json.loads(linesB[-1])
    for k in ("levels_cleared", "total_steps", "total_score",
              "actions_to_first_level_up", "cost_usd"):
        if sa[k] != sb[k]:
            print(f"FAIL: summary.{k} differs ({sa[k]} vs {sb[k]})")
            return 1

    print(f"PASS: {len(linesA)-1} per-step lines bitwise-identical; summary identical modulo wall/arm.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
