"""Minimal JSONL trace reader. Splits per-step records from the trailing summary."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterator


def iter_steps(path: str | Path) -> Iterator[dict]:
    """Yield per-step records (lines that do NOT have a "trajectory" key)."""
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if "trajectory" not in rec:
            yield rec


def read_summary(path: str | Path) -> dict:
    """Return the trajectory summary line (last line with a "trajectory" key)."""
    for line in reversed(Path(path).read_text(encoding="utf-8").splitlines()):
        rec = json.loads(line)
        if "trajectory" in rec:
            return rec
    raise ValueError(f"No summary line in {path}")


if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    n = 0
    for _ in iter_steps(path):
        n += 1
    print(f"{path}: {n} steps, summary={read_summary(path)}")
