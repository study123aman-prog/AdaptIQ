"""
AdaptIQ — rules.py
==================
The domain logic program (Phase 2). This is where the three knowledge sources
meet the inference engine:

  * the **knowledge base** (topics + the prerequisite semantic net)  — Phase 0
  * the **BKT mastery estimates** (which topics the learner has mastered) — Phase 1
  * the **logic engine** (unification, chaining, resolution)           — logic_engine.py

From these it builds a definite-clause program and exposes the pedagogy-level
questions AdaptIQ actually needs to answer:

  * `unlocked_topics(...)`     — forward chaining: what is the learner eligible for?
  * `explain_locked(...)`      — backward chaining: *why* is a topic still locked?
  * `transitive_prereqs(...)`  — recursive forward chaining: everything a topic needs
  * `resolution_demo()`        — a propositional resolution refutation set-piece

Semantics
---------
`mastered(T)`  : the learner's BKT estimate for T is at/above a threshold.
`prereq(P, T)` : P is a direct prerequisite of T   (edge P -> T in the net).
`eligible(T)`  : every direct prerequisite of T is mastered  (matches
                 KnowledgeBase.is_eligible — eligibility is about prerequisites,
                 independent of whether T itself is mastered yet).
`needs(T, P)`  : P is a prerequisite of T, directly or transitively.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from knowledge_base import KnowledgeBase
from bkt import MasteryTracker
from logic_engine import (
    Var, Literal, Rule, forward_chain, prove, ask,
    PropLit, resolution_refutation,
)

MASTERY_THRESHOLD = 0.95


# --------------------------------------------------------------------------- #
# Translating the knowledge base + mastery state into logical facts and rules
# --------------------------------------------------------------------------- #

def kb_facts(kb: KnowledgeBase) -> Set[Literal]:
    """topic(T) and prereq(P, T) facts straight from the semantic net."""
    facts: Set[Literal] = set()
    for t in kb.all_topics():
        facts.add(Literal("topic", (t.id,)))
        for pre in t.prerequisites:
            facts.add(Literal("prereq", (pre, t.id)))
    return facts


def mastered_facts(mastery: Dict[str, float],
                   threshold: float = MASTERY_THRESHOLD) -> Set[Literal]:
    """mastered(T) for every topic whose BKT estimate clears the threshold.

    This is the concrete bridge from the *probabilistic* student model to the
    *symbolic* rule engine: a fuzzy P(mastered) becomes a crisp logical fact.
    """
    return {Literal("mastered", (t,)) for t, p in mastery.items() if p >= threshold}


def eligibility_rules(kb: KnowledgeBase) -> List[Rule]:
    """One grounded rule per topic:  eligible(T) :- mastered(p1), ..., mastered(pn).

    A topic with no prerequisites becomes the fact  eligible(T).
    """
    rules: List[Rule] = []
    for t in kb.all_topics():
        body = tuple(Literal("mastered", (p,)) for p in t.prerequisites)
        rules.append(Rule(Literal("eligible", (t.id,)), body))
    return rules


def transitivity_rules() -> List[Rule]:
    """General (variable-carrying) recursive rules for transitive prerequisites:

        needs(T, P) :- prereq(P, T).
        needs(T, P) :- prereq(M, T), needs(M, P).
    """
    T, P, M = Var("T"), Var("P"), Var("M")
    return [
        Rule(Literal("needs", (T, P)), (Literal("prereq", (P, T)),)),
        Rule(Literal("needs", (T, P)), (Literal("prereq", (M, T)), Literal("needs", (M, P)))),
    ]


def build_program(kb: KnowledgeBase, mastery: Dict[str, float],
                  threshold: float = MASTERY_THRESHOLD):
    """The eligibility program: (facts, rules)."""
    facts = kb_facts(kb) | mastered_facts(mastery, threshold)
    rules = eligibility_rules(kb)
    return facts, rules


# --------------------------------------------------------------------------- #
# Pedagogy-level queries
# --------------------------------------------------------------------------- #

def unlocked_topics(kb: KnowledgeBase, mastery: Dict[str, float],
                    threshold: float = MASTERY_THRESHOLD) -> List[str]:
    """Forward chaining: every topic the learner is currently eligible for."""
    facts, rules = build_program(kb, mastery, threshold)
    closure, _ = forward_chain(facts, rules)
    return sorted(f.args[0] for f in closure if f.pred == "eligible")


def transitive_prereqs(kb: KnowledgeBase, topic: str) -> List[str]:
    """Recursive forward chaining: everything `topic` needs (direct + transitive)."""
    facts = kb_facts(kb)
    closure, _ = forward_chain(facts, transitivity_rules())
    return sorted(f.args[1] for f in closure
                  if f.pred == "needs" and f.args[0] == topic)


def explain_locked(kb: KnowledgeBase, mastery: Dict[str, float], topic: str,
                   threshold: float = MASTERY_THRESHOLD) -> str:
    """
    Backward chaining: produce the 'why is this (un)locked?' reasoning trace.

    Proves the goal eligible(topic); on success it lists the satisfied
    prerequisites, on failure it names the unmastered prerequisite(s).
    """
    kb.get_topic(topic)                                   # validates the id
    facts, rules = build_program(kb, mastery, threshold)
    goal = Literal("eligible", (topic,))
    proof = prove(goal, facts, rules)

    name = lambda tid: kb.get_topic(tid).name
    prereqs = kb.prerequisites(topic)
    header = f"Query: eligible({topic})?  —  backward chaining\n"

    if not prereqs:
        return (header +
                f"'{name(topic)}' is a foundational topic (no prerequisites) → "
                f"ALWAYS ELIGIBLE.\n\nProof tree:\n{proof.render()}")

    satisfied = [p for p in prereqs if Literal("mastered", (p,)) in facts]
    missing = [p for p in prereqs if Literal("mastered", (p,)) not in facts]

    lines = [header, f"Rule:  eligible({topic}) :- " +
             ", ".join(f"mastered({p})" for p in prereqs) + "."]
    if satisfied:
        lines.append("  satisfied: " + ", ".join(name(p) for p in satisfied))
    if missing:
        lines.append("  UNMET:     " + ", ".join(name(p) for p in missing))

    if proof.ok:
        lines.append(f"\n→ '{name(topic)}' is UNLOCKED: all prerequisites are mastered.")
    else:
        miss_names = ", ".join(name(p) for p in missing)
        lines.append(f"\n→ '{name(topic)}' is LOCKED. Master first: {miss_names}.")

    lines.append("\nProof tree:\n" + proof.render())
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Resolution set-piece (propositional)
# --------------------------------------------------------------------------- #

def resolution_demo() -> Tuple[bool, list]:
    """
    Prove, by resolution refutation, that the learner is eligible for Dynamic
    Programming given that recursion is mastered and the eligibility rule
    'mastered_recursion → eligible_dp'. Returns (entailed, steps).
    """
    mr = PropLit("mastered_recursion")
    edp = PropLit("eligible_dp")
    clauses = [
        frozenset({mr}),                       # fact:  mastered_recursion
        frozenset({mr.negate(), edp}),         # rule:  mastered_recursion → eligible_dp
    ]
    return resolution_refutation(clauses, edp)


# --------------------------------------------------------------------------- #
# Demo — run `python rules.py`
# --------------------------------------------------------------------------- #

def _demo() -> None:
    kb = KnowledgeBase()
    tracker = MasteryTracker.from_knowledge_base(kb)

    # A learner who has drilled Arrays and Linked Lists to mastery, but has NOT
    # yet mastered Recursion. (Mastery comes from the BKT engine, Phase 1.)
    tracker.observe_sequence("arrays", [True] * 8)
    tracker.observe_sequence("linked_lists", [True] * 10)
    mastery = tracker.snapshot()

    name = lambda tid: kb.get_topic(tid).name
    print("=" * 74)
    print("AdaptIQ — logic / rule engine (Phase 2)")
    print("=" * 74)
    mastered = sorted(t for t, p in mastery.items() if p >= MASTERY_THRESHOLD)
    print("Mastered (BKT ≥ %.2f): %s\n" % (
        MASTERY_THRESHOLD, ", ".join(name(t) for t in mastered)))

    print("1) FORWARD CHAINING — topics the learner is eligible for now:")
    for t in unlocked_topics(kb, mastery):
        print(f"     • {name(t)}")

    print("\n2) UNIFICATION — ask prereq(?p, dynamic_programming):")
    for s, fact in ask(Literal("prereq", (Var("p"), "dynamic_programming")),
                       kb_facts(kb), []):
        bound = s[Var("p")]
        print(f"     binds ?p = {bound}  ({name(bound)})   [{fact}]")

    print("\n3) BACKWARD CHAINING — why is Dynamic Programming locked?\n")
    print(explain_locked(kb, mastery, "dynamic_programming"))

    print("\n   (contrast) Stacks & Queues:\n")
    print(explain_locked(kb, mastery, "stacks_queues"))

    print("\n4) TRANSITIVE PREREQUISITES of Dynamic Programming (recursive FC):")
    print("     " + ", ".join(name(p) for p in transitive_prereqs(kb, "dynamic_programming")))

    print("\n5) RESOLUTION — refutation proof that eligible_dp follows from")
    print("   { mastered_recursion ,  mastered_recursion → eligible_dp }:")
    entailed, steps = resolution_demo()
    for c1, c2, r in steps:
        shown = "∅ (empty clause)" if len(r) == 0 else "{" + ", ".join(map(repr, r)) + "}"
        print(f"     resolve {{{', '.join(map(repr, c1))}}} , "
              f"{{{', '.join(map(repr, c2))}}}  ⟹  {shown}")
    print(f"     entailed: {entailed}")


if __name__ == "__main__":
    _demo()
