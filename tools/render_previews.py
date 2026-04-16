"""Render the first observed frame of each of 25 public games to games/previews/<short>.png.

One-shot helper for Stage-1 game selection (execution-plan.md section 3.4).
Uses a reasonable 16-color palette; exact hex is not load-bearing for selection.
"""
from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from arc_agi import Arcade


PALETTE = [
    "#000000", "#0074D9", "#FF4136", "#2ECC40",
    "#FFDC00", "#AAAAAA", "#F012BE", "#FF851B",
    "#7FDBFF", "#870C25", "#AA0000", "#55AA00",
    "#0000AA", "#AAAA00", "#00AAAA", "#FFFFFF",
]
CMAP = ListedColormap(PALETTE)


def main() -> int:
    arc = Arcade()
    envs = arc.get_environments()
    out_dir = Path("games/previews")
    out_dir.mkdir(parents=True, exist_ok=True)
    for e in envs:
        short = e.game_id.split("-", 1)[0]
        env = arc.make(e.game_id, seed=0)
        obs = env.reset()
        grid = np.asarray(obs.frame[0], dtype=np.int8)
        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
        ax.imshow(grid, cmap=CMAP, vmin=0, vmax=15)
        title = f"{e.title}  ({e.game_id})\ntags={e.tags} baseline_actions={e.baseline_actions}"
        ax.set_title(title, fontsize=8)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"{short}.png", bbox_inches="tight")
        plt.close(fig)
        print(f"  rendered {short}  tags={e.tags}")
    print(f"done, wrote {len(envs)} previews to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
