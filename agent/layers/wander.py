"""Layer 1 — wander policy.

Pinned spec: paper/stage1-preregistration.md §4.

  uniform sample from obs.available_actions filtered by Layer 0's per-action
  probability threshold (keep action a if P(frame_change | a) >= 0.05). If all
  actions are filtered out, fall back to uniform over unfiltered available_actions.

ACTION6 click position is sampled uniformly from grid cells whose click-head
probability >= 0.05; if no cell clears the threshold, uniform over the full
64×64 grid (this matches the keyboard-action fallback rule applied per-cell).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np

from agent.layers.bce_frame_change import Layer0, ACTION6, NUM_KEY_ACTIONS, GRID

FILTER_THRESHOLD = 0.05  # pinned in pre-reg §4


@dataclass
class WanderAudit:
    rejected_any: bool      # filter rejected at least one available action
    fallback_fired: bool    # all available actions filtered out → uniform fallback
    n_available: int
    n_kept: int


def wander(
    layer0: Layer0,
    frame: np.ndarray,
    available_actions: list[int],
    rng: np.random.Generator,
    threshold: float = FILTER_THRESHOLD,
) -> tuple[int, Optional[dict[str, int]], WanderAudit]:
    """Pick an action via Layer 0-filtered uniform sampling.

    Returns: (action_id, data_dict_or_None_for_ACTION6, audit).
    """
    action_probs = layer0.predict_action_probs(frame)  # {1..5: float}
    click_map = None  # lazy — only compute if ACTION6 is in available_actions

    # Score each available action with its Layer 0 keep-probability.
    keep: dict[int, bool] = {}
    for aid in available_actions:
        if 1 <= aid <= NUM_KEY_ACTIONS:
            keep[aid] = action_probs[aid] >= threshold
        elif aid == ACTION6:
            if click_map is None:
                click_map = layer0.predict_click_probs(frame)
            keep[aid] = float(click_map.max()) >= threshold
        else:
            # ACTION7 (undo) and any future actions: no Layer 0 signal → always keep
            keep[aid] = True

    kept = [aid for aid, k in keep.items() if k]
    rejected_any = len(kept) < len(available_actions)
    fallback_fired = len(kept) == 0
    if fallback_fired:
        kept = list(available_actions)

    audit = WanderAudit(
        rejected_any=rejected_any,
        fallback_fired=fallback_fired,
        n_available=len(available_actions),
        n_kept=len(kept),
    )

    chosen = int(rng.choice(kept))
    if chosen == ACTION6:
        if click_map is None:
            click_map = layer0.predict_click_probs(frame)
        x, y = _sample_click(click_map, rng, threshold)
        return chosen, {"x": int(x), "y": int(y)}, audit
    return chosen, None, audit


def _sample_click(
    click_map: np.ndarray,
    rng: np.random.Generator,
    threshold: float,
) -> tuple[int, int]:
    """Uniform sample (x, y) over cells with click_map >= threshold.

    Per-cell fallback: if no cell clears the threshold, uniform over the full grid.
    """
    mask = click_map >= threshold
    if not mask.any():
        x = int(rng.integers(0, GRID))
        y = int(rng.integers(0, GRID))
        return x, y
    ys, xs = np.where(mask)
    idx = int(rng.integers(0, len(xs)))
    return int(xs[idx]), int(ys[idx])
