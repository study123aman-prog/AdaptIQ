"""
AdaptIQ — planner.py
====================
The planning layer (Phase 3, CO5). It decides *what the learner should study
next, and in what order*, to reach a set of goal topics — respecting the
prerequisite semantic net and the current BKT mastery estimates.

It provides the four planning ideas the syllabus asks for:

  * **State-space search** — states are sets of mastered topics; actions are
    "study(T)" (applicable only when T's prerequisites are mastered); the goal is
    "all target topics mastered". `astar_study_order` runs A* with an admissible
    heuristic (number of required topics still unmastered) to return an optimal,
    prerequisite-respecting study order.
  * **Partial-order planning** — `partial_order_levels` groups the required
    topics into levels; topics in the same level are mutually independent
    (e.g. Sorting vs. Hashing) and may be studied in any order. Any linearisation
    consistent with the levels is a valid plan (least commitment).
  * **Planning & acting** — `simulate` interleaves teaching, quizzing (via the
    BKT tracker) and belief updates instead of planning once and walking away.
  * **Conditional planning / replanning** — the plan is recomputed from the
    updated mastery after every study session, so when an answer diverges from
    the plan's assumption (a topic that was supposed to be learned isn't), the
    planner reacts: it keeps the learner on that topic and dependents stay
    blocked, rather than marching ahead.

Dependency-free, like the rest of the engine. Eligibility/prerequisite facts
come from the Phase-0 knowledge base; mastery comes from the Phase-1 BKT tracker.
"""

from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from knowledge_base import KnowledgeBase
from bkt import MasteryTracker

try:                                           # single source of truth for the bar
    from rules import MASTERY_THRESHOLD
except Exception:                              # pragma: no cover - standalone fallback
    MASTERY_THRESHOLD = 0.95


class PlannerError(ValueError):
    """Raised when a plan is requested that cannot be satisfied."""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def mastered_ids(mastery: Dict[str, float],
                 threshold: float = MASTERY_THRESHOLD) -> Set[str]:
    """Topics whose BKT estimate clears the mastery threshold."""
    return {t for t, p in mastery.items() if p >= threshold}


def required_topics(kb: KnowledgeBase, goals, mastered: Set[str]) -> Set[str]:
    """Everything that must still be studied: goals + all their transitive
    prerequisites, minus whatever is already mastered."""
    req: Set[str] = set()
    for g in goals:
        kb.get_topic(g)                        # validates the id
        req.add(g)
        req |= kb.ancestors(g)
    return {t for t in req if t not in mastered}


def _order_key(kb: KnowledgeBase, topic: str) -> Tuple[int, str]:
    """Deterministic, pedagogically sensible tie-break: easier topics first."""
    t = kb.get_topic(topic)
    return (t.difficulty, t.id)


# --------------------------------------------------------------------------- #
# State-space search (A*)
# --------------------------------------------------------------------------- #

def astar_study_order(kb: KnowledgeBase, mastered: Set[str], goals,
                      threshold: float = MASTERY_THRESHOLD) -> List[str]:
    """
    A* over mastery states. Returns a prerequisite-respecting study order that
    reaches the goal (every target topic mastered). Each action studies one
    topic; the heuristic h = |required - studied| is admissible and consistent,
    so the first plan A* completes is optimal in length (= |required|).
    """
    mastered = set(mastered)
    required = required_topics(kb, goals, mastered)
    if not required:
        return []

    def eligible(topic: str, studied: frozenset) -> bool:
        return all(p in mastered or p in studied
                   for p in kb.prerequisites(topic))

    start: frozenset = frozenset()
    goal: frozenset = frozenset(required)

    def h(studied: frozenset) -> int:
        return len(required - studied)

    counter = 0
    frontier: List[Tuple[int, int, int, frozenset, List[str]]] = [
        (h(start), 0, counter, start, [])
    ]
    best_g: Dict[frozenset, int] = {start: 0}

    while frontier:
        f, g, _, studied, plan = heapq.heappop(frontier)
        if studied == goal:
            return plan
        if g > best_g.get(studied, 1 << 30):
            continue
        for topic in sorted(required - set(studied), key=lambda t: _order_key(kb, t)):
            if not eligible(topic, studied):
                continue
            nxt = frozenset(studied | {topic})
            ng = g + 1
            if ng < best_g.get(nxt, 1 << 30):
                best_g[nxt] = ng
                counter += 1
                heapq.heappush(frontier, (ng + h(nxt), ng, counter, nxt, plan + [topic]))

    raise PlannerError("goal is unreachable (unexpected for an acyclic net)")


# --------------------------------------------------------------------------- #
# Partial-order planning
# --------------------------------------------------------------------------- #

def partial_order_levels(kb: KnowledgeBase, mastered: Set[str], goals,
                         threshold: float = MASTERY_THRESHOLD) -> List[List[str]]:
    """
    Group the required topics into dependency levels. A topic's level is the
    length of its longest prerequisite chain *within the required set*; topics in
    the same level are mutually independent and may be studied in any order.
    """
    mastered = set(mastered)
    required = required_topics(kb, goals, mastered)
    rank: Dict[str, int] = {}
    for t in kb.topological_order():           # prereqs are ranked before dependents
        if t not in required:
            continue
        preds = [p for p in kb.prerequisites(t) if p in required]
        rank[t] = 0 if not preds else 1 + max(rank[p] for p in preds)
    levels: Dict[int, List[str]] = defaultdict(list)
    for t, r in rank.items():
        levels[r].append(t)
    return [sorted(levels[r], key=lambda t: _order_key(kb, t)) for r in sorted(levels)]


def ordering_constraints(kb: KnowledgeBase, mastered: Set[str], goals,
                         threshold: float = MASTERY_THRESHOLD) -> List[Tuple[str, str]]:
    """The must-precede edges (p -> t) among required topics — the only ordering
    the plan actually commits to."""
    mastered = set(mastered)
    required = required_topics(kb, goals, mastered)
    edges = []
    for t in required:
        for p in kb.prerequisites(t):
            if p in required:
                edges.append((p, t))
    return sorted(edges)


# --------------------------------------------------------------------------- #
# Acting: the immediate next step (recomputed = replanning)
# --------------------------------------------------------------------------- #

def next_step(kb: KnowledgeBase, mastery: Dict[str, float], goals,
              threshold: float = MASTERY_THRESHOLD) -> Optional[str]:
    """
    The single best topic to study right now, given the *current* mastery.
    Returns None when all goals are mastered. Because it reads live mastery, it
    is itself the act of replanning: call it after every observation.
    """
    mastered = mastered_ids(mastery, threshold)
    required = required_topics(kb, goals, mastered)
    if not required:
        return None
    eligible = [t for t in required if kb.is_eligible(t, mastered)]
    if not eligible:                           # should not happen on a DAG
        raise PlannerError("no eligible topic though goals remain — net inconsistent")
    return sorted(eligible, key=lambda t: _order_key(kb, t))[0]


# --------------------------------------------------------------------------- #
# Explanation
# --------------------------------------------------------------------------- #

def explain_plan(kb: KnowledgeBase, mastered: Set[str], goals,
                 threshold: float = MASTERY_THRESHOLD) -> str:
    name = lambda t: kb.get_topic(t).name
    mastered = set(mastered)
    required = required_topics(kb, goals, mastered)
    goal_names = ", ".join(name(g) for g in goals)

    if not required:
        return f"Goal(s) [{goal_names}] already mastered — nothing to plan."

    order = astar_study_order(kb, mastered, goals, threshold)
    levels = partial_order_levels(kb, mastered, goals, threshold)

    lines = [f"Goal: master {goal_names}.",
             f"Topics to study ({len(required)}), prerequisite-respecting order (A*):"]
    studied: Set[str] = set(mastered)
    for i, t in enumerate(order, 1):
        preds = [p for p in kb.prerequisites(t)]
        if preds:
            reason = "after " + ", ".join(name(p) for p in preds)
        else:
            reason = "foundational — no prerequisites"
        lines.append(f"  {i}. {name(t)}  ({reason})")
        studied.add(t)

    lines.append("\nPartial-order view (topics on the same line are independent):")
    for r, level in enumerate(levels):
        lines.append(f"  level {r}: " + ", ".join(name(t) for t in level))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Planning + acting + replanning (interleaved)
# --------------------------------------------------------------------------- #

def simulate(kb: KnowledgeBase, tracker: MasteryTracker, goals,
             learner: Dict[str, List[bool]], session_size: int = 6,
             max_sessions: int = 30, threshold: float = MASTERY_THRESHOLD):
    """
    Interleave planning and acting. Each session: replan from current mastery,
    pick the next topic, quiz the learner (answers drawn from `learner[topic]`,
    defaulting to correct), update beliefs, and report. Returns (trace, reached).
    """
    name = lambda t: kb.get_topic(t).name
    idx: Dict[str, int] = defaultdict(int)
    trace: List[str] = []

    for session in range(1, max_sessions + 1):
        mastery = tracker.snapshot()
        mastered = mastered_ids(mastery, threshold)
        if not required_topics(kb, goals, mastered):
            trace.append(f"[session {session}] GOAL REACHED — "
                         f"{', '.join(name(g) for g in goals)} mastered.")
            return trace, True

        plan = astar_study_order(kb, mastered, goals, threshold)
        nxt = plan[0]
        trace.append(f"[session {session}] replan → " +
                     " → ".join(name(t) for t in plan))

        stream = learner.get(nxt, [])
        before = mastery[nxt]
        answers: List[bool] = []
        for _ in range(session_size):
            i = idx[nxt]
            a = stream[i] if i < len(stream) else True      # default to correct
            idx[nxt] += 1
            tracker.observe(nxt, a)
            answers.append(a)
            if tracker.is_mastered(nxt, threshold):
                break
        after = tracker.mastery_of(nxt)
        got, tot = sum(answers), len(answers)

        if tracker.is_mastered(nxt, threshold):
            trace.append(f"    study {name(nxt)}: {got}/{tot} correct → "
                         f"mastery {before:.2f}→{after:.2f}  ✓ mastered; advancing.")
        else:
            blocked = sorted({name(d) for d in required_topics(kb, goals, mastered)
                              if nxt in kb.ancestors(d)})
            tail = (f" dependents still blocked: {', '.join(blocked)}."
                    if blocked else "")
            trace.append(f"    study {name(nxt)}: {got}/{tot} correct → "
                         f"mastery {before:.2f}→{after:.2f}  ✗ not yet; "
                         f"REPLAN keeps {name(nxt)}.{tail}")

    return trace, False


# --------------------------------------------------------------------------- #
# Demo — run `python planner.py`
# --------------------------------------------------------------------------- #

def _demo() -> None:
    kb = KnowledgeBase()
    name = lambda t: kb.get_topic(t).name

    print("=" * 74)
    print("AdaptIQ — planner (Phase 3)")
    print("=" * 74)

    print("\nA) STATE-SPACE SEARCH + PARTIAL ORDER — goal: Dynamic Programming "
          "(from scratch)\n")
    print(explain_plan(kb, set(), ["dynamic_programming"]))

    print("\nB) PARTIAL ORDER with independent goals — Sorting and Hashing:\n")
    print(explain_plan(kb, set(), ["sorting", "hashing"]))

    print("\nC) PLANNING + ACTING + REPLANNING — goal: Dynamic Programming.")
    print("   Scripted learner: aces Arrays, but FAILS Recursion the first time,")
    print("   then recovers. Watch the plan hold on Recursion before advancing.\n")
    tracker = MasteryTracker.from_knowledge_base(kb)
    learner = {
        "arrays": [True] * 8,
        # recursion: a rough first session, then steady correct answers
        "recursion": [False, False, True, False] + [True] * 18,
        "dynamic_programming": [True] * 12,
    }
    trace, reached = simulate(kb, tracker, ["dynamic_programming"], learner)
    for line in trace:
        print("   " + line)
    print(f"\n   goal reached: {reached}")


if __name__ == "__main__":
    _demo()
