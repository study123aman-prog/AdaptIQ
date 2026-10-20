"""
AdaptIQ — knowledge_base.py
===========================

The knowledge layer of the AdaptIQ adaptive DSA tutor (Phase 0).

This module loads the DSA *topic frames* and *question frames* from JSON and
exposes them through a small, well-defined query API. It also builds two
knowledge structures required by the AI syllabus:

  * a **prerequisite semantic net** — a directed acyclic graph (DAG) where an
    edge  prereq -> topic  means "prereq must be mastered before topic".
  * a small **ontology** — the is-a / used-in relations between concepts.

Design note
-----------
The graph algorithms (transitive prerequisites, topological ordering, cycle
detection) are implemented *from scratch* in plain Python rather than pulling in
a third-party graph library. For a data-structures tutor this is deliberate:
the traversals are gradeable, fully transparent, and have zero dependencies, so
the engine runs anywhere. `draw_graph.py` is the only place a plotting library
is used, and even that degrades gracefully.

Everything here is pure knowledge representation and semantic-net *reasoning*
(eligibility, unlocking, "why is this locked?"). The probabilistic mastery
tracking (BKT) lands in Phase 1; the full logic/rule engine in Phase 2.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

# --------------------------------------------------------------------------- #
# Frame definitions (a "frame" = an object with named slots)
# --------------------------------------------------------------------------- #

_BKT_KEYS = ("p_init", "p_learn", "p_slip", "p_guess")


@dataclass
class Misconception:
    """A named misconception attached to a topic; questions reference these."""
    id: str
    text: str


@dataclass
class Topic:
    """A DSA topic represented as a frame with named slots."""
    id: str
    name: str
    category: str                       # DataStructure | Algorithm | Technique
    prerequisites: List[str]            # ids of direct prerequisites (semantic-net edges)
    difficulty: int                     # 1 (foundational) .. 5 (advanced)
    concepts: List[str] = field(default_factory=list)
    misconceptions: List[Misconception] = field(default_factory=list)
    is_a: Optional[str] = None          # ontology: this topic is-a <class>
    used_in: List[str] = field(default_factory=list)  # ontology: used-in <topics>
    bkt_params: Dict[str, float] = field(default_factory=dict)

    def misconception_ids(self) -> Set[str]:
        return {m.id for m in self.misconceptions}


@dataclass
class Question:
    """A single practice question, also a frame."""
    id: str
    topic: str
    difficulty: str                     # easy | medium | hard
    stem: str
    options: List[str]
    answer_index: int
    explanation: str
    misconception: Optional[str] = None  # id of a Misconception on the parent topic

    def is_correct(self, choice_index: int) -> bool:
        return choice_index == self.answer_index


# --------------------------------------------------------------------------- #
# Custom error type
# --------------------------------------------------------------------------- #

class KnowledgeBaseError(ValueError):
    """Raised when the knowledge base fails a consistency/validation check."""


# --------------------------------------------------------------------------- #
# The knowledge base
# --------------------------------------------------------------------------- #

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TOPICS_PATH = os.path.join(_HERE, "topics.json")
DEFAULT_QUESTIONS_PATH = os.path.join(_HERE, "questions.json")

DIFFICULTY_ORDER = {"easy": 0, "medium": 1, "hard": 2}


class KnowledgeBase:
    """
    Loads topic + question frames and exposes the prerequisite semantic net,
    the ontology, and a query/explainability API.
    """

    def __init__(self, topics_path: str = DEFAULT_TOPICS_PATH,
                 questions_path: str = DEFAULT_QUESTIONS_PATH,
                 validate: bool = True):
        self.topics: Dict[str, Topic] = {}
        self.questions: Dict[str, Question] = {}
        self._questions_by_topic: Dict[str, List[Question]] = {}

        self._load_topics(topics_path)
        self._load_questions(questions_path)
        self._build_reverse_edges()

        if validate:
            self.validate()

    # ---------------------------------------------------------------- loading

    def _load_topics(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for raw in data["topics"]:
            miscon = [Misconception(**m) for m in raw.get("common_misconceptions", [])]
            topic = Topic(
                id=raw["id"],
                name=raw["name"],
                category=raw["category"],
                prerequisites=list(raw.get("prerequisites", [])),
                difficulty=int(raw["difficulty"]),
                concepts=list(raw.get("concepts", [])),
                misconceptions=miscon,
                is_a=raw.get("is_a"),
                used_in=list(raw.get("used_in", [])),
                bkt_params=dict(raw.get("bkt_params", {})),
            )
            if topic.id in self.topics:
                raise KnowledgeBaseError(f"Duplicate topic id: {topic.id}")
            self.topics[topic.id] = topic

    def _load_questions(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for raw in data["questions"]:
            q = Question(
                id=raw["id"],
                topic=raw["topic"],
                difficulty=raw["difficulty"],
                stem=raw["stem"],
                options=list(raw["options"]),
                answer_index=int(raw["answer_index"]),
                explanation=raw["explanation"],
                misconception=raw.get("misconception"),
            )
            if q.id in self.questions:
                raise KnowledgeBaseError(f"Duplicate question id: {q.id}")
            self.questions[q.id] = q
            self._questions_by_topic.setdefault(q.topic, []).append(q)

    def _build_reverse_edges(self) -> None:
        """Precompute, for each topic, the set of topics that directly require it."""
        self._dependents: Dict[str, List[str]] = {tid: [] for tid in self.topics}
        for tid, topic in self.topics.items():
            for pre in topic.prerequisites:
                if pre in self._dependents:
                    self._dependents[pre].append(tid)

    # ------------------------------------------------------------- validation

    def validate(self) -> None:
        """Run every consistency check; raise KnowledgeBaseError on the first failure."""
        # 1. Prerequisites must reference real topics.
        for topic in self.topics.values():
            for pre in topic.prerequisites:
                if pre not in self.topics:
                    raise KnowledgeBaseError(
                        f"Topic '{topic.id}' lists unknown prerequisite '{pre}'.")

        # 2. BKT parameters must be present and valid probabilities.
        for topic in self.topics.values():
            for key in _BKT_KEYS:
                if key not in topic.bkt_params:
                    raise KnowledgeBaseError(
                        f"Topic '{topic.id}' is missing BKT parameter '{key}'.")
                val = topic.bkt_params[key]
                if not (0.0 <= val <= 1.0):
                    raise KnowledgeBaseError(
                        f"Topic '{topic.id}' BKT parameter '{key}'={val} is outside [0,1].")
            # Degenerate-model guard used in the BKT literature: slip + guess must be < 1
            # (otherwise a wrong answer would raise mastery — the model becomes nonsensical).
            if topic.bkt_params["p_slip"] + topic.bkt_params["p_guess"] >= 1.0:
                raise KnowledgeBaseError(
                    f"Topic '{topic.id}' has p_slip + p_guess >= 1 (degenerate BKT model).")

        # 3. The prerequisite graph must be acyclic.
        cycle = self._find_cycle()
        if cycle:
            raise KnowledgeBaseError(
                "Prerequisite graph contains a cycle: " + " -> ".join(cycle))

        # 4. Every question must reference a real topic, a valid answer index,
        #    and (if tagged) a misconception that exists on its topic.
        for q in self.questions.values():
            if q.topic not in self.topics:
                raise KnowledgeBaseError(
                    f"Question '{q.id}' references unknown topic '{q.topic}'.")
            if not (0 <= q.answer_index < len(q.options)):
                raise KnowledgeBaseError(
                    f"Question '{q.id}' answer_index {q.answer_index} out of range.")
            if len(q.options) < 2:
                raise KnowledgeBaseError(f"Question '{q.id}' needs at least 2 options.")
            if q.difficulty not in DIFFICULTY_ORDER:
                raise KnowledgeBaseError(
                    f"Question '{q.id}' has invalid difficulty '{q.difficulty}'.")
            if q.misconception is not None:
                if q.misconception not in self.topics[q.topic].misconception_ids():
                    raise KnowledgeBaseError(
                        f"Question '{q.id}' tags misconception '{q.misconception}' "
                        f"not defined on topic '{q.topic}'.")

        # 5. used_in ontology links should reference real topics (warn-level -> strict here).
        for topic in self.topics.values():
            for used in topic.used_in:
                if used not in self.topics:
                    raise KnowledgeBaseError(
                        f"Topic '{topic.id}' used_in references unknown topic '{used}'.")

    # --------------------------------------------------- graph algorithms (scratch)

    def _find_cycle(self) -> Optional[List[str]]:
        """DFS-based cycle detection over the prerequisite graph. Returns a cycle path or None."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {tid: WHITE for tid in self.topics}
        stack: List[str] = []

        def dfs(node: str) -> Optional[List[str]]:
            color[node] = GRAY
            stack.append(node)
            for nxt in self.topics[node].prerequisites:
                if color[nxt] == GRAY:                 # back-edge -> cycle
                    idx = stack.index(nxt)
                    return stack[idx:] + [nxt]
                if color[nxt] == WHITE:
                    found = dfs(nxt)
                    if found:
                        return found
            color[node] = BLACK
            stack.pop()
            return None

        for tid in self.topics:
            if color[tid] == WHITE:
                found = dfs(tid)
                if found:
                    return found
        return None

    def ancestors(self, topic_id: str) -> Set[str]:
        """All *transitive* prerequisites of a topic (everything that must come before it)."""
        self._require(topic_id)
        seen: Set[str] = set()

        def dfs(node: str) -> None:
            for pre in self.topics[node].prerequisites:
                if pre not in seen:
                    seen.add(pre)
                    dfs(pre)

        dfs(topic_id)
        return seen

    def descendants(self, topic_id: str) -> Set[str]:
        """All topics that transitively depend on this one (what it eventually unlocks)."""
        self._require(topic_id)
        seen: Set[str] = set()

        def dfs(node: str) -> None:
            for dep in self._dependents[node]:
                if dep not in seen:
                    seen.add(dep)
                    dfs(dep)

        dfs(topic_id)
        return seen

    def topological_order(self) -> List[str]:
        """
        A valid learning sequence via Kahn's algorithm: no topic appears before
        any of its prerequisites. Ties are broken by (difficulty, id) for a
        stable, sensible order.
        """
        indeg = {tid: len(t.prerequisites) for tid, t in self.topics.items()}
        ready = sorted([tid for tid, d in indeg.items() if d == 0],
                       key=self._order_key)
        order: List[str] = []
        while ready:
            node = ready.pop(0)
            order.append(node)
            newly_ready = []
            for dep in self._dependents[node]:
                indeg[dep] -= 1
                if indeg[dep] == 0:
                    newly_ready.append(dep)
            # keep 'ready' sorted so the sequence is deterministic
            ready = sorted(ready + newly_ready, key=self._order_key)
        if len(order) != len(self.topics):
            raise KnowledgeBaseError("Topological sort failed — graph is cyclic.")
        return order

    def _order_key(self, tid: str):
        t = self.topics[tid]
        return (t.difficulty, t.id)

    # ------------------------------------------------- semantic-net reasoning

    def is_eligible(self, topic_id: str, mastered: Set[str]) -> bool:
        """A topic is eligible iff all of its direct prerequisites are mastered."""
        self._require(topic_id)
        return all(pre in mastered for pre in self.topics[topic_id].prerequisites)

    def eligible_topics(self, mastered: Set[str]) -> List[str]:
        """Every not-yet-mastered topic whose prerequisites are all satisfied."""
        return [tid for tid in self.topological_order()
                if tid not in mastered and self.is_eligible(tid, mastered)]

    def missing_prerequisites(self, topic_id: str, mastered: Set[str]) -> List[str]:
        """Direct prerequisites still not mastered — the reason a topic is locked."""
        self._require(topic_id)
        return [pre for pre in self.topics[topic_id].prerequisites if pre not in mastered]

    # ------------------------------------------------------------- basic queries

    def get_topic(self, topic_id: str) -> Topic:
        self._require(topic_id)
        return self.topics[topic_id]

    def all_topics(self) -> List[Topic]:
        return list(self.topics.values())

    def prerequisites(self, topic_id: str) -> List[str]:
        return list(self.get_topic(topic_id).prerequisites)

    def dependents(self, topic_id: str) -> List[str]:
        self._require(topic_id)
        return list(self._dependents[topic_id])

    def questions_for(self, topic_id: str,
                      difficulty: Optional[str] = None) -> List[Question]:
        self._require(topic_id)
        qs = list(self._questions_by_topic.get(topic_id, []))
        if difficulty is not None:
            qs = [q for q in qs if q.difficulty == difficulty]
        return sorted(qs, key=lambda q: (DIFFICULTY_ORDER[q.difficulty], q.id))

    def _require(self, topic_id: str) -> None:
        if topic_id not in self.topics:
            raise KnowledgeBaseError(f"Unknown topic '{topic_id}'.")

    # ------------------------------------------------- explainability helpers
    # These return human-readable strings — the seeds of the reasoning trace
    # that later phases (planner, tutor loop) will surface to the learner.

    def describe_topic(self, topic_id: str) -> str:
        t = self.get_topic(topic_id)
        pres = ", ".join(self.topics[p].name for p in t.prerequisites) or "none"
        unlocks = ", ".join(self.topics[d].name for d in self._dependents[t.id]) or "none"
        lines = [
            f"{t.name}  [{t.category}, difficulty {t.difficulty}/5]",
            f"  is-a: {t.is_a or 'n/a'}",
            f"  prerequisites: {pres}",
            f"  unlocks (direct): {unlocks}",
            f"  key concepts: {', '.join(t.concepts)}",
        ]
        return "\n".join(lines)

    def explain_eligibility(self, topic_id: str, mastered: Set[str]) -> str:
        """
        A backward-looking explanation of whether a topic is available yet — the
        Phase-0 seed of the 'Why is this locked?' feature (full backward chaining
        arrives in Phase 2).
        """
        t = self.get_topic(topic_id)
        if not t.prerequisites:
            return f"'{t.name}' is a foundational topic with no prerequisites — always available."
        missing = self.missing_prerequisites(topic_id, mastered)
        if not missing:
            got = ", ".join(self.topics[p].name for p in t.prerequisites)
            return (f"'{t.name}' is UNLOCKED: all prerequisites are mastered ({got}).")
        missing_names = ", ".join(self.topics[m].name for m in missing)
        return (f"'{t.name}' is LOCKED. Missing prerequisite(s): {missing_names}. "
                f"Master those first.")

    def explain_learning_order(self) -> str:
        order = self.topological_order()
        pretty = " -> ".join(self.topics[t].name for t in order)
        return "A valid prerequisite-respecting learning path:\n  " + pretty


# --------------------------------------------------------------------------- #
# CLI / self-check — run `python knowledge_base.py` to validate + preview.
# --------------------------------------------------------------------------- #

def _demo() -> None:
    kb = KnowledgeBase()  # validates on load
    n_topics = len(kb.topics)
    n_qs = len(kb.questions)
    print("=" * 70)
    print(f"AdaptIQ knowledge base loaded and VALIDATED.")
    print(f"  topics    : {n_topics}")
    print(f"  questions : {n_qs}  (avg {n_qs / n_topics:.1f} per topic)")
    print("=" * 70)

    print("\n" + kb.explain_learning_order() + "\n")

    # Semantic-net reasoning demo: transitive prerequisites of Dynamic Programming.
    dp_ancestors = kb.ancestors("dynamic_programming")
    print("Everything you must master before Dynamic Programming:")
    print("  " + ", ".join(sorted(kb.topics[a].name for a in dp_ancestors)))

    # Eligibility / 'why locked' demo.
    print("\nEligibility demo (a learner who has mastered only Arrays):")
    mastered = {"arrays"}
    print("  eligible now: " +
          ", ".join(kb.topics[t].name for t in kb.eligible_topics(mastered)))
    print("  " + kb.explain_eligibility("dynamic_programming", mastered))
    print("  " + kb.explain_eligibility("recursion", mastered))

    # Question-bank spot check.
    print("\nSample question (recursion, hard):")
    q = kb.questions_for("recursion", difficulty="hard")[0]
    print(f"  Q: {q.stem}")
    for i, opt in enumerate(q.options):
        mark = "*" if i == q.answer_index else " "
        print(f"    {mark} {chr(65 + i)}. {opt}")
    print(f"  Explanation: {q.explanation}")


if __name__ == "__main__":
    _demo()
