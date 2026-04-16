"""Layer 0 — frame-change BCE classifier.

Pinned spec: paper/stage1-preregistration.md §3. Reset contract: on level-up,
caller invokes Layer0.reset() which wipes weights and buffer. Across seeds
and games, construct a fresh Layer0.

Caller contract: pass the 2D grid (the single layer from obs.frame[0]), a
stable frame hash string, the action id (1-7), and click (x, y) or (None, None).
If a game emits multi-layer frames, extract the relevant layer upstream — this
module only knows about 2D (H, W) int grids.
"""
from __future__ import annotations
import collections
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_COLORS = 16
GRID = 64
NUM_KEY_ACTIONS = 5        # ACTION1..ACTION5 get the pooled scalar head
ACTION6 = 6                # ACTION6 gets the per-pixel click map
BUFFER_MAX = 10_000
BATCH_SIZE = 32
LR = 1e-3


def one_hot_grid(frame: np.ndarray) -> torch.Tensor:
    """(H, W) int grid → (16, H, W) float32 one-hot.

    Crashes loudly on palette overflow (values >15 or <0) per pre-reg §3.
    """
    arr = np.asarray(frame)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D grid, got shape {arr.shape}")
    lo, hi = int(arr.min()), int(arr.max())
    if lo < 0 or hi >= NUM_COLORS:
        raise ValueError(
            f"frame color out of range [0,{NUM_COLORS - 1}]: min={lo} max={hi}"
        )
    h, w = arr.shape
    idx = torch.from_numpy(arr.astype(np.int64))  # (H, W)
    t = torch.zeros(NUM_COLORS, h, w, dtype=torch.float32)
    t.scatter_(0, idx.unsqueeze(0), 1.0)
    return t


class Layer0Net(nn.Module):
    """4-conv ConvNet with a pooled action head and a 1×1 click head."""

    def __init__(self) -> None:
        super().__init__()

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        self.backbone = nn.Sequential(
            block(NUM_COLORS, 32),
            block(32, 64),
            block(64, 128),
            block(128, 256),
        )
        self.action_head = nn.Linear(256, NUM_KEY_ACTIONS)
        self.click_head = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(x)                       # (B, 256, H, W)
        pooled = feat.mean(dim=(2, 3))                # (B, 256)
        action_logits = self.action_head(pooled)      # (B, 5)
        click_logits = self.click_head(feat)          # (B, 1, H, W)
        return action_logits, click_logits


@dataclass
class BufferEntry:
    key: tuple                  # dedup key: (hash, action_id[, x, y])
    one_hot: torch.Tensor       # (16, 64, 64) float32
    action_id: int              # 1-5 or 6
    x: Optional[int]            # click x for ACTION6, else None
    y: Optional[int]
    frame_changed: int          # 0 or 1


class ReplayBuffer:
    """Hash-deduplicated FIFO buffer.

    Re-observing an existing (state, action) key overwrites the entry
    (refreshes frame_changed bit) and moves it to most-recent position.
    """

    def __init__(self, max_size: int = BUFFER_MAX) -> None:
        self.max = max_size
        self._entries: collections.OrderedDict[tuple, BufferEntry] = collections.OrderedDict()
        self.evictions = 0

    def __len__(self) -> int:
        return len(self._entries)

    def add(self, entry: BufferEntry) -> None:
        if entry.key in self._entries:
            self._entries[entry.key] = entry
            self._entries.move_to_end(entry.key)
            return
        self._entries[entry.key] = entry
        if len(self._entries) > self.max:
            self._entries.popitem(last=False)
            self.evictions += 1

    def sample(self, rng: np.random.Generator, batch: int) -> list[BufferEntry]:
        n = min(batch, len(self._entries))
        if n == 0:
            return []
        keys = list(self._entries.keys())
        chosen = rng.choice(len(keys), size=n, replace=False)
        return [self._entries[keys[int(i)]] for i in chosen]


class Layer0:
    """Online trainer + predictor. One instance per (game, seed, level)."""

    def __init__(self, master_seed: int, device: str = "auto", num_threads: int | None = 2) -> None:
        self.master_seed = master_seed
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if num_threads is not None and self.device.type == "cpu":
            torch.set_num_threads(num_threads)
        torch.manual_seed(master_seed)
        self.net = Layer0Net().to(self.device)
        self.opt = torch.optim.Adam(
            self.net.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=0.0
        )
        self.buffer = ReplayBuffer(BUFFER_MAX)
        self.rng = np.random.default_rng(master_seed)
        self.loss_history: list[float] = []

    def reset(self) -> None:
        """Wipe weights and buffer. Called on level-up per pre-reg §3."""
        torch.manual_seed(self.master_seed + len(self.loss_history))  # new but reproducible
        self.net = Layer0Net().to(self.device)
        self.opt = torch.optim.Adam(
            self.net.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=0.0
        )
        self.buffer = ReplayBuffer(BUFFER_MAX)
        # loss_history carried across — useful for per-trajectory plots

    def observe_and_train(
        self,
        frame: np.ndarray,
        frame_hash: str,
        action_id: int,
        x: Optional[int],
        y: Optional[int],
        frame_changed: bool,
    ) -> float:
        """Record transition, take one SGD step.

        Returns training loss, or 0.0 if the buffer is empty or action is
        unsupported (ACTION7/RESET — no training signal).
        """
        if action_id in range(1, NUM_KEY_ACTIONS + 1):
            key: tuple = (frame_hash, action_id)
        elif action_id == ACTION6:
            if x is None or y is None:
                raise ValueError("ACTION6 requires (x, y)")
            key = (frame_hash, ACTION6, int(x), int(y))
        else:
            return 0.0  # ACTION7 / RESET don't train the frame-change head

        one_hot = one_hot_grid(frame)
        self.buffer.add(BufferEntry(
            key=key, one_hot=one_hot, action_id=action_id,
            x=x, y=y, frame_changed=int(frame_changed),
        ))
        return self._train_step()

    def _train_step(self) -> float:
        samples = self.buffer.sample(self.rng, BATCH_SIZE)
        if not samples:
            return 0.0
        B = len(samples)
        self.net.train()
        x_batch = torch.stack([s.one_hot for s in samples]).to(self.device)  # (B,16,64,64)
        action_logits, click_logits = self.net(x_batch)                      # (B,5), (B,1,H,W)

        # Vectorized per-sample logit gather. One backward graph for the batch.
        action_ids = np.array([s.action_id for s in samples], dtype=np.int64)
        xs = np.array([s.x if s.x is not None else 0 for s in samples], dtype=np.int64)
        ys = np.array([s.y if s.y is not None else 0 for s in samples], dtype=np.int64)
        changed = np.array([s.frame_changed for s in samples], dtype=np.float32)

        is_keyboard = torch.from_numpy(action_ids <= NUM_KEY_ACTIONS).to(self.device)
        aid_index = torch.from_numpy(
            np.where(action_ids <= NUM_KEY_ACTIONS, action_ids - 1, 0)
        ).to(self.device)
        xs_t = torch.from_numpy(xs).to(self.device)
        ys_t = torch.from_numpy(ys).to(self.device)
        target = torch.from_numpy(changed).to(self.device)

        batch_idx = torch.arange(B, device=self.device)
        action_selected = action_logits.gather(1, aid_index.unsqueeze(1)).squeeze(1)  # (B,)
        click_selected = click_logits[batch_idx, 0, ys_t, xs_t]                       # (B,)
        logit = torch.where(is_keyboard, action_selected, click_selected)             # (B,)

        loss = F.binary_cross_entropy_with_logits(logit, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        val = float(loss.item())
        self.loss_history.append(val)
        return val

    @torch.no_grad()
    def predict_action_probs(self, frame: np.ndarray) -> dict[int, float]:
        """Return {1: p1, 2: p2, ..., 5: p5} — P(frame_change | ACTIONi)."""
        self.net.eval()
        x = one_hot_grid(frame).unsqueeze(0).to(self.device)
        action_logits, _ = self.net(x)
        probs = torch.sigmoid(action_logits[0]).cpu().numpy()
        return {i + 1: float(probs[i]) for i in range(NUM_KEY_ACTIONS)}

    @torch.no_grad()
    def predict_click_probs(self, frame: np.ndarray) -> np.ndarray:
        """Return (H, W) array of P(frame_change | ACTION6 at (x, y))."""
        self.net.eval()
        x = one_hot_grid(frame).unsqueeze(0).to(self.device)
        _, click_logits = self.net(x)
        return torch.sigmoid(click_logits[0, 0]).cpu().numpy()

    def audit(self) -> dict:
        """Health snapshot for Stage 1 audit tables (pre-reg §3 §5)."""
        return {
            "buffer_size": len(self.buffer),
            "fifo_evictions": self.buffer.evictions,
            "num_train_steps": len(self.loss_history),
            "last_loss": self.loss_history[-1] if self.loss_history else None,
            "mean_loss_last_100": (
                float(np.mean(self.loss_history[-100:]))
                if self.loss_history else None
            ),
        }
