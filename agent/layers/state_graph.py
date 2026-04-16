"""Layer 2 — hashed-state graph with BFS to untested-action frontier.

Pinned spec: paper/stage1-preregistration.md §4.

  Layer 2 (state graph): hashed-state directed graph with edges keyed
  (frame_hash_src, action, frame_hash_dst). When the graph has reachable
  states with untested legal actions, BFS to the nearest such state and
  take the untested action. Suppresses Layer 1 when it fires.

Design decisions made here, NOT pinned in pre-reg (defensible defaults):

1. ACTION6 lumping. The pre-reg edge key is (src_hash, action, dst_hash) with
   action = single int. For ACTION6, that means we treat "any click at state h"
   as one action. Once we click ANYWHERE at h, ACTION6 is marked tested at h.
   Click position diversity is delegated to Layer 1's `_sample_click`. Without
   this lumping the untested set per state would be 4096 entries (one per cell)
   and the graph would not converge in 2000 steps.

2. BFS targets visited states only. A state is "visited" iff we've observed
   its `available_actions`. Edges that point to unvisited states are followed
   blindly during path execution but not chosen as BFS targets — we only know
   the untested set for visited states.

3. Path replay uses stored (action_id, x, y) tuples. If the env is deterministic
   on (state, action, click), replay is exact. ACTION6 click coords are stored
   on the edge.

4. If BFS finds no frontier, `select_action` returns None and Layer 1 takes over.
"""
from __future__ import annotations
import collections
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class Edge:
    action_id: int
    x: Optional[int]      # ACTION6 click x; None for keyboard actions
    y: Optional[int]
    dst_hash: str


# Click sampler: callable that returns (x, y) for an ACTION6 click at the
# current state. Provided by the caller (typically Layer 0 + Layer 1's logic).
ClickSampler = Callable[[], tuple[int, int]]


@dataclass
class GraphAudit:
    n_states_visited: int
    n_edges: int
    n_states_with_untested: int
    fired_count: int           # how many select_action() calls returned a non-None action
    bfs_target_count: int      # how many calls used a BFS target (vs taking immediate untested)


class StateGraph:
    """Directed multigraph + untested-action bookkeeping + BFS frontier search."""

    def __init__(self) -> None:
        self.edges: dict[str, list[Edge]] = collections.defaultdict(list)
        self.untested: dict[str, set[int]] = {}    # legal actions not yet tried at h
        self.visited: set[str] = set()              # states whose `available_actions` we know
        self._path: list[tuple[int, Optional[int], Optional[int]]] = []
        self._fired = 0
        self._bfs_targets = 0

    def reset(self) -> None:
        """Wipe state. Called on level-up per pre-reg §3 reset protocol."""
        self.edges.clear()
        self.untested.clear()
        self.visited.clear()
        self._path.clear()
        self._fired = 0
        self._bfs_targets = 0

    def observe_state(self, current_hash: str, available_actions: list[int]) -> None:
        """Register that we are now at `current_hash` with given legal actions."""
        if current_hash not in self.visited:
            self.untested[current_hash] = set(int(a) for a in available_actions)
            self.visited.add(current_hash)

    def observe_transition(
        self,
        src_hash: str,
        action_id: int,
        x: Optional[int],
        y: Optional[int],
        dst_hash: str,
    ) -> None:
        """Record edge src --action--> dst and mark action_id as tested at src."""
        self.edges[src_hash].append(Edge(action_id=int(action_id), x=x, y=y, dst_hash=dst_hash))
        if src_hash in self.untested:
            self.untested[src_hash].discard(int(action_id))

    def select_action(
        self,
        current_hash: str,
        available_actions: list[int],
        click_sampler: ClickSampler,
    ) -> Optional[tuple[int, Optional[dict[str, int]]]]:
        """Pick an action via Layer 2 logic, or return None to defer to Layer 1.

        Returns (action_id, data_dict_or_None_for_ACTION6) on hit, None on miss.
        """
        self.observe_state(current_hash, available_actions)

        # 1. If current state has untested legal actions, pick one (priority over path).
        legal_set = set(int(a) for a in available_actions)
        untested_here = self.untested.get(current_hash, set()) & legal_set
        if untested_here:
            action_id = min(untested_here)  # deterministic pick: smallest id first
            self._fired += 1
            self._path.clear()  # local untested takes priority over a stale path
            return self._with_data(action_id, click_sampler)

        # 2. If we have a path queued (from a previous BFS), pop the next step.
        while self._path:
            action_id, x, y = self._path.pop(0)
            if action_id not in legal_set:
                # Path stale: env's available_actions changed. Re-plan.
                self._path.clear()
                break
            self._fired += 1
            if action_id == 6:
                return action_id, {"x": int(x), "y": int(y)} if x is not None else None
            return action_id, None

        # 3. BFS to nearest visited state with non-empty untested.
        path = self._bfs_to_frontier(current_hash)
        if path is None:
            return None  # graph exhausted from here → Layer 1 takes over
        self._path = path
        action_id, x, y = self._path.pop(0)
        if action_id not in legal_set:
            # First step of fresh BFS path is illegal at current state — graph is stale.
            self._path.clear()
            return None
        self._fired += 1
        self._bfs_targets += 1
        if action_id == 6:
            return action_id, {"x": int(x), "y": int(y)} if x is not None else None
        return action_id, None

    def _with_data(
        self, action_id: int, click_sampler: ClickSampler,
    ) -> tuple[int, Optional[dict[str, int]]]:
        if action_id == 6:
            x, y = click_sampler()
            return action_id, {"x": int(x), "y": int(y)}
        return action_id, None

    def _bfs_to_frontier(self, start: str) -> Optional[list[tuple[int, Optional[int], Optional[int]]]]:
        """Return a list of (action_id, x, y) steps from start to the nearest
        visited state with non-empty untested. None if no such state reachable."""
        if start not in self.edges and start not in self.visited:
            return None
        # BFS over edges. Track parent pointers to reconstruct path.
        parents: dict[str, tuple[str, Edge]] = {}
        seen = {start}
        queue = collections.deque([start])
        target: Optional[str] = None
        while queue:
            h = queue.popleft()
            # Is this a frontier state? (visited and has untested actions)
            if h != start and h in self.visited and self.untested.get(h):
                target = h
                break
            for edge in self.edges.get(h, []):
                if edge.dst_hash in seen:
                    continue
                seen.add(edge.dst_hash)
                parents[edge.dst_hash] = (h, edge)
                queue.append(edge.dst_hash)
        if target is None:
            return None
        # Reconstruct path: walk parents from target back to start.
        steps: list[tuple[int, Optional[int], Optional[int]]] = []
        cur = target
        while cur != start:
            parent_h, edge = parents[cur]
            steps.append((edge.action_id, edge.x, edge.y))
            cur = parent_h
        steps.reverse()
        return steps

    def audit(self) -> GraphAudit:
        return GraphAudit(
            n_states_visited=len(self.visited),
            n_edges=sum(len(es) for es in self.edges.values()),
            n_states_with_untested=sum(1 for s in self.untested.values() if s),
            fired_count=self._fired,
            bfs_target_count=self._bfs_targets,
        )
