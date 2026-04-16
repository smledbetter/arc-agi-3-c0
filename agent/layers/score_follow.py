"""Layer 3 — score-delta back-label + frontier rerank.

Pinned spec: paper/stage1-preregistration.md §4.

  Layer 3 (score-delta back-label): on positive score_delta, back-label every
  state on the traversed path with distance_to_milestone = steps_since_hash.
  - Regressor: MLP, 3 hidden layers × 128 units, ReLU.
  - Input: (frame_hash_embedding 64-d, action_id_one_hot_7).
  - frame_hash_embedding: deterministic from SHA256 first 512 bits → 64×int8 → ±1 float.
  - Back-label horizon: clip distance_to_milestone at 50.
  - Adam lr=1e-3, Huber loss delta=1.0, batch=32.
  - Retrain cadence: one gradient step after each new back-label is added.
  - Combination with L2: rerank L2's frontier candidates by predicted distance
    (ascending). DOES NOT override BFS — just picks among frontiers.
  - Activation gate: dormant until first positive score_delta in current level.
"""
from __future__ import annotations
import collections
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


HASH_EMBED_DIM = 64
NUM_ACTIONS = 7
INPUT_DIM = HASH_EMBED_DIM + NUM_ACTIONS
HIDDEN = 128
N_HIDDEN_LAYERS = 3
LR = 1e-3
HUBER_DELTA = 1.0
BATCH_SIZE = 32
HORIZON = 50  # back-label distance cap


def hash_to_embedding(frame_hash: str) -> torch.Tensor:
    """frame_hash → 64-dim ±1 float vector. Deterministic, reproducible.

    Pre-reg §4 said "first 512 bits of SHA256 → 64 int8 → ±1," but SHA256 only
    produces 256 bits. Amendment 2 (paper/stage1-preregistration-AMENDMENT-2.md)
    clarifies: expand the hash via SHA512(frame_hash.hex) to get 512 bits, then
    map each byte (>127 → +1, ≤127 → -1) to a 64-d ±1 vector.
    """
    import hashlib
    expanded = hashlib.sha512(frame_hash.encode()).digest()  # 64 bytes = 512 bits
    arr = np.frombuffer(expanded, dtype=np.uint8)
    return torch.tensor(np.where(arr > 127, 1.0, -1.0), dtype=torch.float32)


def action_to_one_hot(action_id: int) -> torch.Tensor:
    """Action id ∈ {1..7} → 7-dim one-hot."""
    v = torch.zeros(NUM_ACTIONS, dtype=torch.float32)
    if 1 <= action_id <= NUM_ACTIONS:
        v[action_id - 1] = 1.0
    return v


class Layer3Net(nn.Module):
    """Distance-to-milestone regressor."""

    def __init__(self) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d_in = INPUT_DIM
        for _ in range(N_HIDDEN_LAYERS):
            layers.append(nn.Linear(d_in, HIDDEN))
            layers.append(nn.ReLU(inplace=True))
            d_in = HIDDEN
        layers.append(nn.Linear(d_in, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # (B,)


@dataclass
class TrainExample:
    embedding: torch.Tensor   # 64-d
    action_one_hot: torch.Tensor  # 7-d
    distance: float


class Layer3:
    """Back-label trainer + rerank picker."""

    def __init__(self, master_seed: int, device: str = "auto") -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        torch.manual_seed(master_seed + 7919)  # offset to avoid Layer 0 collision
        self.net = Layer3Net().to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=LR, betas=(0.9, 0.999))
        self.training_set: list[TrainExample] = []
        self.rng = np.random.default_rng(master_seed + 7919)
        self.activated: bool = False  # gates rerank picker
        # Path memory for back-labeling: deque of (frame_hash, action_id) for the
        # last `HORIZON+1` steps in this level.
        self.path: collections.deque[tuple[str, int]] = collections.deque(maxlen=HORIZON + 1)
        self.loss_history: list[float] = []

    def reset(self) -> None:
        """Wipe weights + back-label set + path. Called on level-up per pre-reg §3."""
        torch.manual_seed(int(self.rng.integers(0, 2**31)))
        self.net = Layer3Net().to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=LR, betas=(0.9, 0.999))
        self.training_set = []
        self.path.clear()
        self.activated = False

    def record_step(self, frame_hash: str, action_id: int) -> None:
        """Record a step on the current path (for future back-labeling)."""
        self.path.append((frame_hash, int(action_id)))

    def on_score_delta(self, score_delta: int) -> None:
        """If score_delta > 0, back-label the recent path and train one step."""
        if score_delta <= 0:
            return
        # Back-label every (state, action) in self.path with distance from end of path.
        new_examples: list[TrainExample] = []
        path_list = list(self.path)
        n = len(path_list)
        for i, (h, aid) in enumerate(path_list):
            distance = float(min(n - 1 - i, HORIZON))
            new_examples.append(TrainExample(
                embedding=hash_to_embedding(h),
                action_one_hot=action_to_one_hot(aid),
                distance=distance,
            ))
        self.training_set.extend(new_examples)
        # One gradient step per new back-labeled example, per pre-reg.
        for _ in range(len(new_examples)):
            self._train_step()
        self.activated = True
        # Clear path so subsequent steps start a new horizon.
        self.path.clear()

    def _train_step(self) -> float:
        n = len(self.training_set)
        if n == 0:
            return 0.0
        bs = min(BATCH_SIZE, n)
        idxs = self.rng.choice(n, size=bs, replace=False)
        x = torch.stack([
            torch.cat([self.training_set[i].embedding, self.training_set[i].action_one_hot])
            for i in idxs
        ]).to(self.device)
        targets = torch.tensor(
            [self.training_set[i].distance for i in idxs],
            dtype=torch.float32, device=self.device,
        )
        self.net.train()
        pred = self.net(x)
        loss = F.huber_loss(pred, targets, delta=HUBER_DELTA)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        v = float(loss.item())
        self.loss_history.append(v)
        return v

    @torch.no_grad()
    def predict_distance(self, frame_hash: str, action_id: int) -> float:
        """Predicted distance-to-milestone for (state, action)."""
        self.net.eval()
        emb = hash_to_embedding(frame_hash).to(self.device)
        oh = action_to_one_hot(action_id).to(self.device)
        x = torch.cat([emb, oh]).unsqueeze(0)
        return float(self.net(x).cpu().item())

    def make_frontier_picker(self):
        """Return a frontier_picker callable for StateGraph.select_action.

        If not yet activated (no positive score_delta seen in this level),
        the picker is a no-op that returns the first (shallowest) candidate
        — matching L2's default behavior.

        When activated, reranks candidates by predicted distance from the
        target_hash + first action of the path (ascending), tie-break by depth.
        """
        if not self.activated:
            return None  # signal "use default" to StateGraph

        def picker(candidates):
            best = None
            best_score = None
            for cand in candidates:
                target_hash, depth, path = cand
                if not path:
                    continue
                first_action = path[0][0]
                pred = self.predict_distance(target_hash, first_action)
                # Score: predicted distance, tie-break by BFS depth (prefer shallower).
                score = (pred, depth)
                if best is None or score < best_score:
                    best = cand
                    best_score = score
            return best if best is not None else candidates[0]

        return picker

    def audit(self) -> dict:
        return {
            "activated": self.activated,
            "training_set_size": len(self.training_set),
            "num_train_steps": len(self.loss_history),
            "last_loss": self.loss_history[-1] if self.loss_history else None,
            "path_len": len(self.path),
        }
