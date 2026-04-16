"""Wall-clock and cost tracking per step and per trajectory."""
from __future__ import annotations
import time
from dataclasses import dataclass, field


class StepTimer:
    """Context manager: `with StepTimer() as t: ...; t.wall_ms` gives milliseconds."""
    def __enter__(self) -> "StepTimer":
        self._t0 = time.perf_counter_ns()
        return self

    def __exit__(self, *exc) -> None:
        self.wall_ms = (time.perf_counter_ns() - self._t0) / 1_000_000.0


@dataclass
class CostTracker:
    """Accumulator for API cost (USD). Always 0 for C0; used by C1/C1' ablations."""
    usd_total: float = 0.0
    usd_by_step: list[float] = field(default_factory=list)

    def add(self, usd: float) -> None:
        self.usd_total += usd
        self.usd_by_step.append(usd)

    def per_step(self, step: int) -> float:
        return self.usd_by_step[step] if step < len(self.usd_by_step) else 0.0
