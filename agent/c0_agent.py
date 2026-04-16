"""C0 agent — composes Layers 0-3 via subsumption.

Pinned spec: paper/stage1-preregistration.md §4.

Subsumption priority (highest first):
  L3-rerank-of-L2 > L2 > L1 > L0-filter

Mechanism:
  - On each step:
      1. L2.select_action(current_hash, available, click_sampler, picker=L3.make_picker())
         If L3 is dormant (no positive score_delta yet this level), picker is None
         and L2 falls back to nearest-BFS. Once L3 activates, picker reranks.
      2. If L2 returns None (graph exhausted from current state) → L1 wander.
      3. L0's filter is implicit inside L1's wander.
  - After env.step:
      - L0 trains on (frame, action, frame_changed)
      - L2 records edge + dst's available_actions
      - L3 records step on its path
      - On positive score_delta: L3 back-labels + trains
      - On levels_completed increment: reset L0, L2, L3 for the new level
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np

from agent.layers.bce_frame_change import Layer0, ACTION6
from agent.layers.state_graph import StateGraph
from agent.layers.score_follow import Layer3
from agent.layers.wander import wander, _sample_click, FILTER_THRESHOLD


@dataclass
class StepDecision:
    action_id: int
    data: Optional[dict[str, int]]
    source: str  # "L2-untested" | "L2-path" | "L2-bfs" | "L1-wander"
    wander_audit: Optional[object] = None  # WanderAudit if L1 fired


class C0Agent:
    """C0 = Layers 0-3 composed via subsumption."""

    def __init__(self, master_seed: int, device: str = "auto") -> None:
        self.master_seed = master_seed
        self.rng = np.random.default_rng(master_seed + 1)  # offset from layer rngs
        self.layer0 = Layer0(master_seed, device=device)
        self.layer2 = StateGraph(rng=np.random.default_rng(master_seed + 2))
        self.layer3 = Layer3(master_seed, device=device)

    def reset_for_new_level(self) -> None:
        """Wipe model state between levels per pre-reg §3 reset protocol.

        Re-seeds L2's RNG so post-reset behavior is reproducible from master_seed.
        """
        self.layer0.reset()
        self.layer2 = StateGraph(rng=np.random.default_rng(self.master_seed + 2))
        self.layer3.reset()

    def select_action(
        self,
        frame: np.ndarray,
        frame_hash: str,
        available_actions: list[int],
    ) -> StepDecision:
        """Pick the next action using L2 (with L3 picker if active), falling back to L1."""
        click_sampler = self._make_click_sampler(frame)
        picker = self.layer3.make_frontier_picker()

        # Track whether L2 source was untested-here, path-replay, or fresh BFS.
        # We expose this via a quick check before/after the call.
        pre_path_len = len(self.layer2._path)
        pre_bfs = self.layer2._bfs_targets

        result = self.layer2.select_action(
            frame_hash, available_actions, click_sampler, frontier_picker=picker,
        )

        if result is not None:
            action_id, data = result
            if self.layer2._bfs_targets > pre_bfs:
                source = "L2-bfs"
            elif pre_path_len > 0:
                source = "L2-path"
            else:
                source = "L2-untested"
            return StepDecision(action_id=action_id, data=data, source=source)

        # L2 missed → L1 wander
        action_id, data, audit = wander(
            self.layer0, frame, available_actions, self.rng,
        )
        return StepDecision(action_id=action_id, data=data, source="L1-wander", wander_audit=audit)

    def observe_transition(
        self,
        src_frame: np.ndarray,
        src_hash: str,
        action_id: int,
        x: Optional[int],
        y: Optional[int],
        dst_hash: str,
        dst_available_actions: list[int],
        frame_changed: bool,
        score_delta: int,
    ) -> None:
        """Update all layers with the observed transition.

        Caller is responsible for triggering reset_for_new_level() on level-up.
        """
        # L0 trains on the (src_frame, action, frame_changed) signal.
        self.layer0.observe_and_train(src_frame, src_hash, action_id, x, y, frame_changed)
        # L2 records the edge and the destination state's legal actions.
        self.layer2.observe_transition(src_hash, action_id, x, y, dst_hash)
        self.layer2.observe_state(dst_hash, dst_available_actions)
        # L3 records the path step (for back-labeling on future positive deltas).
        self.layer3.record_step(src_hash, action_id)
        # L3 back-labels + trains on a positive score delta.
        if score_delta > 0:
            self.layer3.on_score_delta(score_delta)

    def _make_click_sampler(self, frame: np.ndarray):
        """Return a callable that samples (x, y) for ACTION6 via Layer 0's click map."""
        def sampler() -> tuple[int, int]:
            cm = self.layer0.predict_click_probs(frame)
            return _sample_click(cm, self.rng, FILTER_THRESHOLD)
        return sampler

    def audit(self) -> dict:
        from dataclasses import asdict, is_dataclass
        l2 = self.layer2.audit()
        return {
            "layer0": self.layer0.audit(),
            "layer2": asdict(l2) if is_dataclass(l2) else l2,
            "layer3": self.layer3.audit(),
        }
