"""
AdaptIQ — test_session.py
=========================
Tests for the headless tutor session controller (Phase 4). Dependency-free:

    python test_session.py

Streamlit cannot be installed in the build sandbox, so all adaptive behaviour
lives in TutorSession (which has no UI dependency) and is verified here; app.py
is only a thin presentation layer over this.
"""

from __future__ import annotations

import sys

from knowledge_base import KnowledgeBase
from tutor_session import TutorSession, target_difficulty

KB = KnowledgeBase()
_failures: list = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f"  <-- {detail}" if (detail and not cond) else ""))
    if not cond:
        _failures.append(name)


# --------------------------------------------------------------------------- #
def test_target_difficulty() -> None:
    print("test_target_difficulty")
    check("low mastery -> easy", target_difficulty(0.20) == "easy")
    check("0.40 boundary -> medium", target_difficulty(0.40) == "medium")
    check("mid mastery -> medium", target_difficulty(0.60) == "medium")
    check("high mastery -> hard", target_difficulty(0.90) == "hard")


def test_first_topic_and_adaptivity() -> None:
    print("test_first_topic_and_adaptivity")
    s = TutorSession.start(["dynamic_programming"], kb=KB)
    item = s.next_item()
    check("first topic toward DP is Arrays", item.topic == "arrays",
          f"got {item.topic}")
    # Arrays prior is 0.40 -> target medium; Recursion prior 0.15 -> easy.
    check("served difficulty matches the mastery band when available",
          item.difficulty == item.target or
          item.target not in {q.difficulty for q in KB.questions_for("arrays")},
          f"served={item.difficulty} target={item.target}")


def test_unseen_rotation() -> None:
    print("test_unseen_rotation")
    s = TutorSession.start(["dynamic_programming"], kb=KB)
    first = s.next_item()
    s.submit(first.question.answer_index)        # correct; stays on Arrays a while
    second = s.next_item()
    check("still on Arrays (not yet mastered)", second.topic == "arrays")
    check("a different (unseen) question is served",
          second.question.id != first.question.id,
          f"{first.question.id} vs {second.question.id}")


def test_wrong_answer_feedback() -> None:
    print("test_wrong_answer_feedback")
    s = TutorSession.start(["dynamic_programming"], kb=KB)
    item = s.next_item()
    wrong = (item.question.answer_index + 1) % len(item.question.options)
    fb = s.submit(wrong)
    check("wrong answer is graded wrong", fb["correct"] is False)
    check("wrong answer lowers mastery (Arrays net-fall)",
          fb["mastery_after"] < fb["mastery_before"],
          f"{fb['mastery_before']:.3f} -> {fb['mastery_after']:.3f}")
    check("feedback carries the explanation", bool(fb["explanation"]))


def test_full_playthrough_completes() -> None:
    print("test_full_playthrough_completes")
    s = TutorSession.start(["dynamic_programming"], kb=KB)
    turns = 0
    while not s.is_complete() and turns < 400:
        item = s.next_item()
        if item is None:
            break
        # always eligible + required
        if not (KB.is_eligible(item.topic, s.mastered())
                and item.topic in {"arrays", "recursion", "dynamic_programming"}):
            check("served topic is eligible+on-path", False,
                  f"topic={item.topic} mastered={sorted(s.mastered())}")
            break
        s.submit(item.question.answer_index)     # answer correctly
        turns += 1
    check("diligent learner completes the goal", s.is_complete())
    check("Dynamic Programming is mastered", s.tracker.is_mastered("dynamic_programming"))
    check("completed in a sane number of questions", 0 < turns < 400,
          f"turns={turns}")


def test_views() -> None:
    print("test_views")
    s = TutorSession.start(["dynamic_programming"], kb=KB)
    s.next_item()
    rows = s.mastery_rows()
    check("mastery_rows has one row per topic", len(rows) == len(KB.all_topics()))
    check("rows are in prerequisite-first order (Arrays first)",
          rows[0]["id"] == "arrays")
    why = s.why_this()
    check("why_this names the goal", "Dynamic Programming" in why)
    check("why_this explains the path", "shortest path" in why)
    prog = s.progress()
    check("initial progress is 0 of 3", prog["done"] == 0 and prog["total"] == 3,
          f"{prog}")


def main() -> int:
    tests = [
        test_target_difficulty,
        test_first_topic_and_adaptivity,
        test_unseen_rotation,
        test_wrong_answer_feedback,
        test_full_playthrough_completes,
        test_views,
    ]
    print("=" * 74)
    print("AdaptIQ tutor session — test suite")
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
