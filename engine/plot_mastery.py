"""
AdaptIQ — plot_mastery.py
=========================
Visualise how the BKT engine's mastery estimate evolves as a learner answers
questions. Produces `mastery_over_time.png`: one trajectory per topic, with
green markers for correct answers and red crosses for incorrect ones, a dotted
line at each topic's prior (p_init), and a dashed line at the mastery threshold.

The two scripted scenarios are chosen to contrast learning rates: Arrays
(high p_learn) is picked up quickly despite a stumble, while Dynamic Programming
(low p_learn, higher slip) climbs slowly even once answers start landing — the
curves make the role of each parameter visible.

matplotlib is the only external dependency (same policy as draw_graph.py); the
engine itself (bkt.py) stays dependency-free.

Usage:
    python plot_mastery.py [output.png]
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless: render straight to a file
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from knowledge_base import KnowledgeBase
from bkt import MasteryTracker

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(_HERE, "mastery_over_time.png")

THRESHOLD = 0.95

# Deterministic answer scripts (1 = correct, 0 = wrong) that tell a small story.
SCENARIOS = {
    "arrays":              [1, 1, 0, 1, 1, 1, 1, 1],
    "dynamic_programming": [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1],
}
PALETTE = {"arrays": "#1a73e8", "dynamic_programming": "#8e24aa"}


def _trajectory(tracker: MasteryTracker, topic: str, answers):
    """Return (xs, ys, corrects): mastery after each answer, starting at the prior."""
    xs = [0]
    ys = [tracker.mastery_of(topic)]   # x=0 is the prior, before any answer
    corrects = []
    for i, a in enumerate(answers, start=1):
        res = tracker.observe(topic, bool(a))
        xs.append(i)
        ys.append(res.mastery)
        corrects.append(bool(a))
    return xs, ys, corrects


def draw(output_path: str = DEFAULT_OUT) -> str:
    kb = KnowledgeBase()
    tracker = MasteryTracker.from_knowledge_base(kb)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    for topic, answers in SCENARIOS.items():
        tracker.reset(topic)
        xs, ys, corrects = _trajectory(tracker, topic, answers)
        name = kb.get_topic(topic).name
        color = PALETTE.get(topic, "#444444")

        ax.plot(xs, ys, "-", color=color, linewidth=2.2, label=name, zorder=2)
        # hollow marker for the prior
        ax.scatter([xs[0]], [ys[0]], marker="o", s=60, facecolor="white",
                   edgecolor=color, linewidth=1.8, zorder=3)
        # filled markers for each answered attempt
        for x, y, c in zip(xs[1:], ys[1:], corrects):
            ax.scatter([x], [y], marker=("o" if c else "X"), s=70,
                       color=("#2e7d32" if c else "#c62828"),
                       edgecolor="white", linewidth=0.8, zorder=4)
        # dotted baseline at this topic's prior
        ax.axhline(ys[0], color=color, linestyle=":", linewidth=1, alpha=0.45, zorder=1)

    ax.axhline(THRESHOLD, color="#555555", linestyle="--", linewidth=1.2, zorder=1)
    ax.text(0.0, THRESHOLD + 0.012, f"mastery threshold = {THRESHOLD:.2f}",
            fontsize=9, color="#555555")

    ax.set_xlabel("attempt number  (0 = prior, before any answer)", fontsize=11)
    ax.set_ylabel("P(mastered)", fontsize=11)
    ax.set_ylim(0, 1.03)
    ax.set_xlim(-0.3, max(len(a) for a in SCENARIOS.values()) + 0.3)
    ax.set_title("AdaptIQ — Bayesian Knowledge Tracing: mastery over time",
                 fontsize=14, fontweight="bold", pad=12)
    ax.grid(True, alpha=0.25)

    # Legend: topic lines plus correct/incorrect marker keys.
    handles, _ = ax.get_legend_handles_labels()
    handles += [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2e7d32",
               markeredgecolor="white", markersize=9, label="correct answer"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor="#c62828",
               markeredgecolor="white", markersize=9, label="incorrect answer"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor="#888888", markersize=9, label="prior (p_init)"),
    ]
    ax.legend(handles=handles, loc="center right", framealpha=0.95, fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT
    print("Saved mastery plot ->", draw(out))
