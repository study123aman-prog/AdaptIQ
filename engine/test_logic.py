"""
AdaptIQ — test_logic.py
=======================
Tests for the logic/rule engine (Phase 2). Dependency-free:

    python test_logic.py

The strongest tests cross-check the *logic* engine against the Phase-0 *graph*
methods in knowledge_base.py: two independent implementations (logical inference
vs. hand-written graph traversal) must agree on eligibility and on transitive
prerequisites. If they disagree, one of them is wrong.
"""

from __future__ import annotations

import sys

from knowledge_base import KnowledgeBase
from logic_engine import (Var, Literal, Rule, unify, substitute,
                          forward_chain, prove, PropLit, resolution_refutation)
from rules import (kb_facts, build_program, unlocked_topics, transitive_prereqs,
                   explain_locked, resolution_demo, MASTERY_THRESHOLD)

KB = KnowledgeBase()
_failures: list = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f"  <-- {detail}" if (detail and not cond) else ""))
    if not cond:
        _failures.append(name)


def mastery_with(mastered) -> dict:
    """A mastery vector that puts exactly `mastered` above threshold."""
    S = set(mastered)
    return {t.id: (1.0 if t.id in S else 0.0) for t in KB.all_topics()}


# --------------------------------------------------------------------------- #
def test_unification() -> None:
    print("test_unification")
    p = Var("p")
    s = unify(Literal("prereq", (p, "dynamic_programming")),
              Literal("prereq", ("recursion", "dynamic_programming")))
    check("binds ?p = recursion", s is not None and s.get(p) == "recursion")
    check("substitute applies the binding",
          substitute(Literal("prereq", (p, "dynamic_programming")), s or {})
          == Literal("prereq", ("recursion", "dynamic_programming")))
    check("clashing constants do not unify",
          unify(Literal("prereq", ("arrays", "trees")),
                Literal("prereq", ("arrays", "bst"))) is None)
    check("different predicates do not unify",
          unify(Literal("topic", ("arrays",)),
                Literal("prereq", ("arrays",))) is None)


def test_forward_chain_eligibility_matches_kb() -> None:
    """Forward-chained eligible(T) must match KnowledgeBase.is_eligible for the
    same mastered set, across several learner states."""
    print("test_forward_chain_eligibility_matches_kb")
    scenarios = [
        set(),
        {"arrays"},
        {"arrays", "linked_lists"},
        {"arrays", "linked_lists", "recursion"},
        {t.id for t in KB.all_topics()},        # everything mastered
    ]
    all_ok = True
    for S in scenarios:
        logic_elig = set(unlocked_topics(KB, mastery_with(S)))
        graph_elig = {t.id for t in KB.all_topics() if KB.is_eligible(t.id, S)}
        if logic_elig != graph_elig:
            all_ok = False
            check(f"eligibility matches for mastered={sorted(S)}", False,
                  f"logic={sorted(logic_elig)} graph={sorted(graph_elig)}")
    check("forward-chained eligibility == KB.is_eligible (all scenarios)", all_ok)


def test_transitive_prereqs_match_kb_ancestors() -> None:
    """Recursive forward chaining of needs(T,P) must equal KB.ancestors(T)."""
    print("test_transitive_prereqs_match_kb_ancestors")
    all_ok = True
    for t in KB.all_topics():
        logic_anc = set(transitive_prereqs(KB, t.id))
        graph_anc = set(KB.ancestors(t.id))
        if logic_anc != graph_anc:
            all_ok = False
            check(f"needs-closure matches ancestors for {t.id}", False,
                  f"logic={sorted(logic_anc)} graph={sorted(graph_anc)}")
    check("transitive 'needs' closure == KB.ancestors (every topic)", all_ok)


def test_backward_chaining() -> None:
    print("test_backward_chaining")
    mastery = mastery_with({"arrays", "linked_lists"})
    facts, rules = build_program(KB, mastery)

    # stacks_queues needs arrays + linked_lists -> both mastered -> provable
    p_ok = prove(Literal("eligible", ("stacks_queues",)), facts, rules)
    check("eligible(stacks_queues) is proved", p_ok.ok)

    # dynamic_programming needs recursion -> not mastered -> fails on mastered(recursion)
    p_bad = prove(Literal("eligible", ("dynamic_programming",)), facts, rules)
    check("eligible(dynamic_programming) fails", not p_bad.ok)
    check("failing leaf is mastered(recursion)",
          Literal("mastered", ("recursion",)) in p_bad.failing_leaves(),
          f"leaves={p_bad.failing_leaves()}")

    # explanation text should name recursion and say LOCKED
    text = explain_locked(KB, mastery, "dynamic_programming")
    check("explanation says LOCKED", "LOCKED" in text)
    check("explanation names Recursion", "Recursion" in text)


def test_resolution() -> None:
    print("test_resolution")
    entailed, steps = resolution_demo()
    check("eligible_dp is entailed (reaches empty clause)", entailed)
    check("resolution recorded at least one step", len(steps) >= 1)

    # A clearly non-entailed query: {a} does not entail b.
    a, b = PropLit("a"), PropLit("b")
    ent2, _ = resolution_refutation([frozenset({a})], b)
    check("{a} does NOT entail b", not ent2)

    # Modus ponens via resolution: {a}, {¬a ∨ b} entail b.
    ent3, _ = resolution_refutation(
        [frozenset({a}), frozenset({a.negate(), b})], b)
    check("{a, a→b} entails b", ent3)


def main() -> int:
    tests = [
        test_unification,
        test_forward_chain_eligibility_matches_kb,
        test_transitive_prereqs_match_kb_ancestors,
        test_backward_chaining,
        test_resolution,
    ]
    print("=" * 74)
    print("AdaptIQ logic/rule engine — test suite")
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
