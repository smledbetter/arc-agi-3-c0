"""Tests for Layer 2 (StateGraph)."""
from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.layers.state_graph import StateGraph


def click_sampler_const() -> tuple[int, int]:
    return (10, 20)


def test_empty_graph_returns_none() -> None:
    g = StateGraph()
    res = g.select_action("h0", [], click_sampler_const)
    # Empty available_actions → untested is empty → BFS finds nothing → None
    assert res is None
    print("  OK empty_graph_returns_none")


def test_picks_untested_at_current_state() -> None:
    g = StateGraph()
    res = g.select_action("h0", [1, 2, 3], click_sampler_const)
    assert res is not None
    aid, data = res
    assert aid in (1, 2, 3)
    assert data is None
    a = g.audit()
    assert a.n_states_visited == 1
    assert a.n_states_with_untested == 1  # 2 actions still untested at h0
    print("  OK picks_untested_at_current_state")


def test_marks_action_tested_after_observation() -> None:
    g = StateGraph()
    g.select_action("h0", [1, 2], click_sampler_const)
    g.observe_transition("h0", 1, None, None, "h1")
    assert 1 not in g.untested["h0"]
    assert 2 in g.untested["h0"]
    print("  OK marks_action_tested_after_observation")


def test_action6_returns_xy_via_sampler() -> None:
    g = StateGraph()
    res = g.select_action("h0", [6], lambda: (5, 7))
    assert res is not None
    aid, data = res
    assert aid == 6
    assert data == {"x": 5, "y": 7}
    print("  OK action6_returns_xy_via_sampler")


def test_returns_none_when_all_tested_no_neighbors() -> None:
    """Single state, all actions exhausted, no edges → None (Layer 1 takes over)."""
    g = StateGraph()
    g.select_action("h0", [1, 2], click_sampler_const)
    g.observe_transition("h0", 1, None, None, "h0")  # self-loop
    g.select_action("h0", [1, 2], click_sampler_const)
    g.observe_transition("h0", 2, None, None, "h0")
    res = g.select_action("h0", [1, 2], click_sampler_const)
    assert res is None
    print("  OK returns_none_when_all_tested_no_neighbors")


def test_bfs_finds_frontier_at_distance_2() -> None:
    """Build h0 → h1 → h2 where h2 has untested actions; BFS from h0 finds it."""
    g = StateGraph()
    # Visit h0, take action 1 → h1
    g.select_action("h0", [1, 2], click_sampler_const)  # picks 1
    g.observe_transition("h0", 1, None, None, "h1")
    # Visit h1, take action 1 → h2
    g.select_action("h1", [1], click_sampler_const)  # only action 1 available
    g.observe_transition("h1", 1, None, None, "h2")
    # Visit h2 (no actions taken yet so untested is full)
    g.observe_state("h2", [3, 4])  # 3, 4 untested at h2
    # Now back at h0 with only action 2 left untested locally.
    # Actually action 2 is still untested at h0, so it picks that first.
    res = g.select_action("h0", [1, 2], click_sampler_const)
    assert res == (2, None), f"expected to take untested action 2 at h0, got {res}"
    g.observe_transition("h0", 2, None, None, "h2")
    # h0 now has nothing untested. BFS should find h2 (which has 3, 4 untested).
    res = g.select_action("h0", [1, 2], click_sampler_const)
    assert res is not None
    aid, _ = res
    # The path is h0 --1--> h1 --1--> h2 OR h0 --2--> h2 (1 hop).
    # BFS picks shortest, so the 1-hop path: take action 2 at h0, then on
    # arrival at h2 we'd pick 3 or 4. But action 2 is no longer untested at h0
    # (we took it). Wait — observing the transition marked it tested. So h0's
    # untested = {} now. BFS expands from h0 via outgoing edges. Edge to h2
    # via action 2 puts h2 at depth 1. h2 has untested {3, 4} → target.
    # Path = [(2, None, None)]. We pop and return action 2.
    assert aid == 2, f"expected BFS path-step action 2, got {aid}"
    print("  OK bfs_finds_frontier_at_distance_2")


def test_reset_wipes_everything() -> None:
    g = StateGraph()
    g.select_action("h0", [1, 2], click_sampler_const)
    g.observe_transition("h0", 1, None, None, "h1")
    a = g.audit()
    assert a.n_states_visited > 0
    g.reset()
    a = g.audit()
    assert a.n_states_visited == 0
    assert a.n_edges == 0
    assert a.fired_count == 0
    print("  OK reset_wipes_everything")


def test_audit_fields() -> None:
    g = StateGraph()
    a = g.audit()
    for f in ("n_states_visited", "n_edges", "n_states_with_untested",
              "fired_count", "bfs_target_count"):
        assert hasattr(a, f)
    print("  OK audit_fields")


def main() -> int:
    print("Layer 2 (state graph) tests:")
    test_empty_graph_returns_none()
    test_picks_untested_at_current_state()
    test_marks_action_tested_after_observation()
    test_action6_returns_xy_via_sampler()
    test_returns_none_when_all_tested_no_neighbors()
    test_bfs_finds_frontier_at_distance_2()
    test_reset_wipes_everything()
    test_audit_fields()
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
