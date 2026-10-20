"""
AdaptIQ — test_planner.py
=========================
Tests for the planning layer (Phase 3). Dependency-free:

    python test_planner.py

Checks that plans are prerequisite-respecting, that the required set is exactly
goals + transitive prerequisites, that the partial order keeps independent
topics unordered, that next_step always proposes an eligible topic, and that the
interleaved simulation reaches its goal (even when a topic fails and forces a
replan).
"""

from __future__ import annotations

import sys

from knowledge_base import KnowledgeBase
from bkt import MasteryTracker
from planner import (required_topics, astar_study_order, partial_order_levels,
                     ordering_constraints, next_step, mastered_ids, simulate,
                     MASTERY_THRESHOLD)

KB = KnowledgeBase()
_failures: list = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f"  <-- {detail}" if (detail and not cond) else ""))
    if not cond:
        _failures.append(name)


def mastery_with(mastered) -> dict:
    S = set(mastered)
    return {t.id: (1.0 if t.id in S else 0.0) for t in KB.all_topics()}


def is_valid_order(order, mastered) -> bool:
    """Every topic appears only after all its prerequisites."""
    seen = set(mastered)
    for t in order:
        if not all(p in seen for p in KB.prerequisites(t)):
            return False
        seen.add(t)
    return True


# --------------------------------------------------------------------------- #
def test_required_set() -> None:
    print("test_required_set")
    req = required_topics(KB, ["dynamic_programming"], set())
    check("DP from scratch requires {arrays, recursion, dynamic_programming}",
          req == {"arrays", "recursion", "dynamic_programming"},
          f"got {sorted(req)}")
    req2 = required_topics(KB, ["dynamic_programming"], {"arrays", "recursion"})
    check("already-mastered prereqs are excluded",
          req2 == {"dynamic_programming"}, f"got {sorted(req2)}")
    check("fully-mastered goal needs nothing",
          required_topics(KB, ["arrays"], {"arrays"}) == set())


def test_astar_order_valid() -> None:
    print("test_astar_order_valid")
    for goals in (["dynamic_programming"], ["graphs"], ["bst", "heaps"],
                  ["sorting", "hashing"]):
        order = astar_study_order(KB, set(), goals)
        req = required_topics(KB, goals, set())
        check(f"{goals}: order respects prerequisites",
              is_valid_order(order, set()))
        check(f"{goals}: order covers exactly the required set",
              set(order) == req, f"order={order} req={sorted(req)}")
        check(f"{goals}: plan length == |required| (optimal)",
              len(order) == len(req))
    check("already-mastered goal yields empty plan",
          astar_study_order(KB, {"arrays"}, ["arrays"]) == [])


def test_partial_order() -> None:
    print("test_partial_order")
    # Sorting and Hashing: Recursion and Hashing are independent (both need only
    # already-required Arrays), so they must share a level.
    levels = partial_order_levels(KB, set(), ["sorting", "hashing"])
    flat = [t for lvl in levels for t in lvl]
    check("level 0 is Arrays alone", levels[0] == ["arrays"], f"got {levels}")
    same_level = any(("recursion" in lvl and "hashing" in lvl) for lvl in levels)
    check("Recursion and Hashing are in the same (independent) level", same_level,
          f"levels={levels}")
    # A flattened level order is itself a valid study order.
    check("flattened levels respect prerequisites", is_valid_order(flat, set()))
    # Ordering constraints must include the real prerequisite edges.
    edges = set(ordering_constraints(KB, set(), ["sorting"]))
    check("constraint arrays->recursion present", ("arrays", "recursion") in edges)
    check("constraint recursion->sorting present", ("recursion", "sorting") in edges)


def test_next_step() -> None:
    print("test_next_step")
    # From scratch toward DP, the only eligible required topic is Arrays.
    check("first step toward DP is Arrays",
          next_step(KB, mastery_with(set()), ["dynamic_programming"]) == "arrays")
    # With Arrays mastered, Recursion becomes the next step.
    check("after Arrays, next step is Recursion",
          next_step(KB, mastery_with({"arrays"}), ["dynamic_programming"]) == "recursion")
    # Goal already met -> None.
    check("next_step is None when goal mastered",
          next_step(KB, mastery_with({"arrays"}), ["arrays"]) is None)
    # next_step must always be an eligible, required topic.
    m = mastery_with({"arrays", "linked_lists"})
    ns = next_step(KB, m, ["graphs"])
    check("proposed step is eligible", KB.is_eligible(ns, mastered_ids(m)))
    check("proposed step is required",
          ns in required_topics(KB, ["graphs"], mastered_ids(m)))


def test_simulate_reaches_goal() -> None:
    print("test_simulate_reaches_goal")
    # Diligent learner (defaults to correct answers) reaches the goal.
    tr = MasteryTracker.from_knowledge_base(KB)
    trace, reached = simulate(KB, tr, ["dynamic_programming"], learner={})
    check("diligent learner reaches DP", reached)
    check("DP actually mastered at the end", tr.is_mastered("dynamic_programming"))

    # Learner who fails Recursion first, then recovers, still reaches the goal;
    # and must not master DP before Recursion (prerequisite enforced by replanning).
    tr2 = MasteryTracker.from_knowledge_base(KB)
    learner = {"recursion": [False, False, True, False] + [True] * 20}
    trace2, reached2 = simulate(KB, tr2, ["dynamic_programming"], learner)
    check("struggling-then-recovering learner still reaches DP", reached2)
    joined = " | ".join(trace2)
    rec_done = next((i for i, l in enumerate(trace2)
                     if "Recursion" in l and "mastered; advancing" in l), None)
    dp_done = next((i for i, l in enumerate(trace2)
                    if "Dynamic Programming" in l and "mastered; advancing" in l), None)
    check("Recursion is mastered before Dynamic Programming",
          rec_done is not None and (dp_done is None or rec_done < dp_done),
          f"rec_done={rec_done} dp_done={dp_done}")


def main() -> int:
    tests = [
        test_required_set,
        test_astar_order_valid,
        test_partial_order,
        test_next_step,
        test_simulate_reaches_goal,
    ]
    print("=" * 74)
    print("AdaptIQ planner — test suite")
    print("=" * 74)
    for t in tests:
        t()
    print("=" * 74)
    if _failures:
        print(f"RESULT: {len(_failures)} check(s) FAILED: {', '.join(_failures)}")
        return 1
    print("RESULT: all checks PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
