"""
AdaptIQ — app.py  (Streamlit UI, Phase 4 fast-path)
===================================================
A thin presentation layer over tutor_session.TutorSession. All the adaptive
logic lives in the engine; this file only renders it and collects clicks.

Run it locally (Streamlit is not installable in the build sandbox):

    cd engine
    pip install -r requirements.txt
    streamlit run app.py

What you get:
  * pick a goal topic and a mastery threshold;
  * answer adaptively-chosen questions;
  * see the correct/incorrect feedback, the explanation, any misconception, and
    the BKT reasoning trace;
  * a "Why this question?" panel (the glass-box feature);
  * a live mastery dashboard and the study plan.
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from knowledge_base import KnowledgeBase
from tutor_session import TutorSession

_HERE = os.path.dirname(os.path.abspath(__file__))
_GRAPH_PNG = os.path.join(_HERE, "prerequisite_graph.png")

st.set_page_config(page_title="AdaptIQ — Explainable DSA Tutor",
                   page_icon="🧭", layout="wide")


@st.cache_resource
def get_kb() -> KnowledgeBase:
    """Load + validate the knowledge base once per server process."""
    return KnowledgeBase()


KB = get_kb()
NAME_TO_ID = {t.name: t.id for t in KB.all_topics()}
ID_TO_NAME = {t.id: t.name for t in KB.all_topics()}


# --------------------------------------------------------------------------- #
# Sidebar — session setup
# --------------------------------------------------------------------------- #
def sidebar() -> None:
    st.sidebar.title("🧭 AdaptIQ")
    st.sidebar.caption("An explainable adaptive DSA tutor — Bayesian Knowledge "
                       "Tracing + a prerequisite semantic net.")

    default_goal = ["Dynamic Programming"] if "Dynamic Programming" in NAME_TO_ID else []
    goal_names = st.sidebar.multiselect(
        "Goal topic(s) — what do you want to master?",
        options=list(NAME_TO_ID.keys()), default=default_goal)
    threshold = st.sidebar.slider("Mastery threshold", 0.80, 0.99, 0.95, 0.01)

    if st.sidebar.button("Start / Restart session", type="primary",
                         use_container_width=True):
        if not goal_names:
            st.sidebar.warning("Pick at least one goal topic.")
        else:
            goals = [NAME_TO_ID[n] for n in goal_names]
            s = TutorSession.start(goals, threshold=threshold, kb=KB)
            st.session_state.session = s
            _serve_next()
            st.rerun()

    st.sidebar.divider()
    st.sidebar.caption("Tip: start with Dynamic Programming to watch the plan "
                       "walk Arrays → Recursion → DP.")


# --------------------------------------------------------------------------- #
# State helpers
# --------------------------------------------------------------------------- #
def _serve_next() -> None:
    """Advance to the next question (or the done state) and stash the trace."""
    s: TutorSession = st.session_state.session
    item = s.next_item()
    st.session_state.item = item
    st.session_state.feedback = None
    st.session_state.why = s.why_this() if item else ""
    st.session_state.phase = "quiz" if item else "done"


# --------------------------------------------------------------------------- #
# Main panels
# --------------------------------------------------------------------------- #
def quiz_panel() -> None:
    s: TutorSession = st.session_state.session
    item = st.session_state.item
    phase = st.session_state.phase

    prog = s.progress()
    st.progress(prog["fraction"],
                text=f"Progress: {prog['done']} / {prog['total']} required topics mastered")

    if phase == "done" or s.is_complete():
        st.success("🎉 Goal reached — all required topics are mastered!")
        st.balloons()
        return

    q = item.question
    st.subheader(f"{item.topic_name}  ·  {item.difficulty} question")
    st.markdown(f"**{q.stem}**")

    opts = q.options
    disabled = (phase == "feedback")
    choice = st.radio(
        "Your answer", options=list(range(len(opts))),
        format_func=lambda i: f"{chr(65 + i)}.  {opts[i]}",
        index=(st.session_state.feedback["chosen_index"] if disabled else None),
        key=f"ans_{q.id}", disabled=disabled)

    if phase == "quiz":
        if st.button("Submit answer", type="primary"):
            if choice is None:
                st.warning("Select an option first.")
            else:
                st.session_state.feedback = s.submit(choice)
                st.session_state.phase = "feedback"
                st.rerun()

    if phase == "feedback":
        fb = st.session_state.feedback
        if fb["correct"]:
            st.success(f"✅ Correct.")
        else:
            st.error(f"❌ Not quite. Correct answer: "
                     f"{chr(65 + fb['correct_index'])}. {opts[fb['correct_index']]}")
        st.info(fb["explanation"])
        if fb["misconception"]:
            st.warning(f"Common misconception: {fb['misconception']}")

        c1, c2 = st.columns(2)
        c1.metric("P(mastered) for this topic",
                  f"{fb['mastery_after']:.2f}",
                  delta=f"{fb['mastery_after'] - fb['mastery_before']:+.2f}")
        c2.metric("Predicted P(correct) before you answered",
                  f"{fb['predicted_correct']:.2f}")
        st.caption("🧠 " + fb["narration"])
        if fb["mastered_now"]:
            st.success(f"You've now mastered {fb['topic_name']}!")

        if st.button("Next question ▶", type="primary"):
            _serve_next()
            st.rerun()


def why_panel() -> None:
    with st.expander("❓ Why this question?  (the glass-box reasoning trace)",
                     expanded=True):
        st.text(st.session_state.get("why", "") or
                "Start a session to see the tutor's reasoning.")


def dashboard_panel() -> None:
    s: TutorSession = st.session_state.session
    st.subheader("📊 Mastery dashboard")
    df = pd.DataFrame(s.mastery_rows())
    chart_df = df.set_index("topic")[["mastery"]]
    st.bar_chart(chart_df, height=280)
    st.caption(f"Threshold = {s.threshold:.2f}. A topic counts as mastered once "
               "its bar crosses the threshold.")
    st.dataframe(
        df[["topic", "category", "difficulty", "mastery", "mastered"]],
        hide_index=True, use_container_width=True)

    with st.expander("🗺️ Study plan (A* + partial order)"):
        st.text(s.plan_overview())
    if os.path.exists(_GRAPH_PNG):
        with st.expander("🔗 Prerequisite semantic net"):
            st.image(_GRAPH_PNG, use_container_width=True)


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
def main() -> None:
    sidebar()
    st.title("AdaptIQ — Explainable Adaptive DSA Tutor")

    if "session" not in st.session_state:
        st.info("👈 Pick a goal topic in the sidebar and press "
                "**Start / Restart session** to begin.")
        with st.expander("What makes this different from a quiz app?", expanded=True):
            st.markdown(
                "- It keeps a **probabilistic estimate of your mastery** of each "
                "topic (Bayesian Knowledge Tracing) and updates it after every answer.\n"
                "- It **respects prerequisites**: you won't see Dynamic Programming "
                "until you've shown you understand Recursion.\n"
                "- It can **explain every decision** — why this question, why a topic "
                "is still locked, and what your mastery is.")
        return

    left, right = st.columns([3, 2], gap="large")
    with left:
        quiz_panel()
        why_panel()
    with right:
        dashboard_panel()


if __name__ == "__main__":
    main()
