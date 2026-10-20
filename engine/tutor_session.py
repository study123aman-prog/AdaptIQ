"""
AdaptIQ — tutor_session.py
==========================
The headless session controller (Phase 4). It ties the whole engine together
into a single object the UI can drive:

    knowledge base  (topics, questions, prerequisites)   — Phase 0
    BKT tracker     (per-topic P(mastered))              — Phase 1
    planner         (what to study next, replanning)     — Phase 3
    explanations    (why this question / mastery trace)  — throughout

Deliberately UI-free (no Streamlit import) so the adaptive logic can be unit
tested in isolation; app.py is a thin presentation layer over this class.

Adaptive behaviour
-------------------
* Which TOPIC to serve next comes from the planner (next eligible topic on the
  shortest prerequisite-respecting path to the goal). Recomputed every turn, so
  it naturally replans as mastery changes.
* Which QUESTION within that topic is chosen by current mastery: a weak learner
  gets easy questions, a strong one gets hard questions — so difficulty tracks
  the BKT estimate. Unseen questions are preferred.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from knowledge_base import KnowledgeBase, Question, DIFFICULTY_ORDER
from bkt import MasteryTracker
from planner import (next_step, required_topics, astar_study_order,
                     mastered_ids, explain_plan, MASTERY_THRESHOLD)

# Mastery bands that decide how hard a question to serve within a topic.
_DIFF_BY_NAME = {v: k for k, v in DIFFICULTY_ORDER.items()}   # 0->easy,1->medium,2->hard


def target_difficulty(p_mastery: float) -> str:
    """Pick a question difficulty appropriate to the current mastery estimate."""
    if p_mastery < 0.40:
        return "easy"
    if p_mastery < 0.75:
        return "medium"
    return "hard"


@dataclass
class Item:
    """The question currently being served."""
    topic: str
    topic_name: str
    question: Question
    difficulty: str
    target: str


@dataclass
class TutorSession:
    kb: KnowledgeBase
    tracker: MasteryTracker
    goals: List[str]
    threshold: float = MASTERY_THRESHOLD
    _seen: Dict[str, set] = field(default_factory=dict)
    _current: Optional[Item] = None
    history: List[dict] = field(default_factory=list)

    # ------------------------------------------------------------ construction
    @classmethod
    def start(cls, goals, threshold: float = MASTERY_THRESHOLD,
              kb: Optional[KnowledgeBase] = None) -> "TutorSession":
        kb = kb or KnowledgeBase()
        for g in goals:
            kb.get_topic(g)                        # validate goal ids
        tracker = MasteryTracker.from_knowledge_base(kb)
        return cls(kb=kb, tracker=tracker, goals=list(goals), threshold=threshold,
                   _seen={t.id: set() for t in kb.all_topics()})

    # ----------------------------------------------------------------- status
    def name(self, tid: str) -> str:
        return self.kb.get_topic(tid).name

    def mastered(self) -> set:
        return mastered_ids(self.tracker.snapshot(), self.threshold)

    def is_complete(self) -> bool:
        return not required_topics(self.kb, self.goals, self.mastered())

    def progress(self) -> dict:
        """How many of the required topics are done, for a progress bar."""
        mastered = self.mastered()
        remaining = required_topics(self.kb, self.goals, mastered)
        needed = required_topics(self.kb, self.goals, set())   # from scratch
        done = len(needed) - len(remaining)
        return {"done": done, "total": len(needed),
                "fraction": (done / len(needed)) if needed else 1.0}

    # -------------------------------------------------------- question serving
    def next_item(self) -> Optional[Item]:
        """Choose the next (topic, question). Returns None when goals are met."""
        topic = next_step(self.kb, self.tracker.snapshot(), self.goals, self.threshold)
        if topic is None:
            self._current = None
            return None
        question = self._pick_question(topic)
        self._current = Item(topic=topic, topic_name=self.name(topic),
                             question=question, difficulty=question.difficulty,
                             target=target_difficulty(self.tracker.mastery_of(topic)))
        return self._current

    @property
    def current(self) -> Optional[Item]:
        return self._current

    def _pick_question(self, topic: str) -> Question:
        seen = self._seen.setdefault(topic, set())
        all_qs = self.kb.questions_for(topic)
        pool = [q for q in all_qs if q.id not in seen] or all_qs   # repeat if exhausted
        target = target_difficulty(self.tracker.mastery_of(topic))
        tdist = lambda q: abs(DIFFICULTY_ORDER[q.difficulty] - DIFFICULTY_ORDER[target])
        return sorted(pool, key=lambda q: (tdist(q), DIFFICULTY_ORDER[q.difficulty], q.id))[0]

    # ----------------------------------------------------------------- answering
    def submit(self, choice_index: int) -> dict:
        """Grade the current question, update beliefs, return a feedback bundle."""
        if self._current is None:
            raise RuntimeError("No current question — call next_item() first.")
        item = self._current
        q = item.question
        correct = q.is_correct(choice_index)
        before = self.tracker.mastery_of(item.topic)
        step = self.tracker.observe(item.topic, correct)
        self._seen.setdefault(item.topic, set()).add(q.id)

        misconception_text = None
        if (not correct) and q.misconception:
            for m in self.kb.get_topic(item.topic).misconceptions:
                if m.id == q.misconception:
                    misconception_text = m.text
                    break

        feedback = {
            "topic": item.topic,
            "topic_name": item.topic_name,
            "question_id": q.id,
            "chosen_index": choice_index,
            "correct_index": q.answer_index,
            "correct": correct,
            "explanation": q.explanation,
            "misconception": misconception_text,
            "narration": step.explain(item.topic_name),
            "mastery_before": before,
            "mastery_after": step.mastery,
            "predicted_correct": step.predicted_correct,
            "mastered_now": self.tracker.is_mastered(item.topic, self.threshold),
        }
        self.history.append(feedback)
        self._current = None
        return feedback

    # ------------------------------------------------------- explanations / views
    def why_this(self) -> str:
        """Glass-box: explain why the current topic/question was chosen."""
        if self._current is None:
            return "No question is being served right now."
        item = self._current
        mastered = self.mastered()
        plan = astar_study_order(self.kb, mastered, self.goals, self.threshold)
        plan_names = " → ".join(self.name(t) for t in plan) if plan else "—"
        prereqs = self.kb.prerequisites(item.topic)
        unlocks = self.kb.dependents(item.topic)
        p = self.tracker.mastery_of(item.topic)
        pc = self.tracker.probability_correct(item.topic)

        lines = [
            f"Goal: master {', '.join(self.name(g) for g in self.goals)}.",
            f"Chosen topic: {item.topic_name}.",
            ("  • eligible now: " +
             ("prerequisites mastered (" + ", ".join(self.name(x) for x in prereqs) + ")"
              if prereqs else "foundational, no prerequisites")),
            f"  • it is the next step on the shortest path: {plan_names}",
            f"  • current P(mastered) = {p:.2f}; predicted P(correct) = {pc:.2f}, "
            f"so a {item.target} question was selected",
        ]
        if unlocks:
            lines.append("  • mastering it moves you toward: " +
                         ", ".join(self.name(x) for x in unlocks))
        return "\n".join(lines)

    def mastery_rows(self) -> List[dict]:
        """Per-topic mastery, in a sensible (prerequisite-first) order, for the dashboard."""
        snap = self.tracker.snapshot()
        rows = []
        for tid in self.kb.topological_order():
            t = self.kb.get_topic(tid)
            rows.append({
                "id": tid,
                "topic": t.name,
                "category": t.category,
                "difficulty": t.difficulty,
                "mastery": round(snap[tid], 3),
                "mastered": snap[tid] >= self.threshold,
            })
        return rows

    def plan_overview(self) -> str:
        return explain_plan(self.kb, self.mastered(), self.goals, self.threshold)


if __name__ == "__main__":
    # Quick headless playthrough: always answer correctly, print the trace.
    s = TutorSession.start(["dynamic_programming"])
    print("Plan:\n" + s.plan_overview() + "\n")
    turns = 0
    while not s.is_complete() and turns < 300:
        item = s.next_item()
        if item is None:
            break
        fb = s.submit(item.question.answer_index)      # answer correctly
        turns += 1
        if fb["mastered_now"]:
            print(f"  mastered {fb['topic_name']} after reaching "
                  f"{fb['mastery_after']:.2f}")
    print(f"\nComplete: {s.is_complete()} in {turns} questions.")
