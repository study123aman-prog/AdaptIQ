"""
AdaptIQ — plot_param_recovery.py
================================
Visualise the Phase-6 parameter learning: for a few topics, simulate attempt
logs from the authored ("true") BKT parameters, re-estimate the parameters by
maximum likelihood, and show true vs. fitted side by side. Close bars mean the
learner successfully recovered the generating parameters from data alone.

matplotlib only (same policy as the other plot scripts). Takes ~20 s because it
runs the optimiser for each topic.

Usage:
    python plot_param_recovery.py [output.png]
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from knowledge_base import KnowledgeBase
from bkt import BKTParams
from learn_params import simulate_logs, fit_bkt

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(_HERE, "param_recovery.png")

TOPICS = ["arrays", "recursion", "dynamic_programming"]
PARAMS = ["p_init", "p_learn", "p_slip", "p_guess"]


def draw(output_path: str = DEFAULT_OUT) -> str:
    kb = KnowledgeBase()
    fig, axes = plt.subplots(1, len(TOPICS), figsize=(4.6 * len(TOPICS), 4.6),
                             dpi=150, sharey=True)
    if len(TOPICS) == 1:
        axes = [axes]

    x = range(len(PARAMS))
    width = 0.38
    for ax, tid in zip(axes, TOPICS):
        true = BKTParams.from_dict(kb.get_topic(tid).bkt_params)
        logs = simulate_logs(true, n_learners=500, n_opportunities=16, seed=7)
        fitted, _ = fit_bkt(logs, restarts=4, seed=1)

        tvals = [getattr(true, p) for p in PARAMS]
        fvals = [getattr(fitted, p) for p in PARAMS]
        ax.bar([i - width / 2 for i in x], tvals, width, label="true",
               color="#1a73e8", edgecolor="#0b3d91")
        ax.bar([i + width / 2 for i in x], fvals, width, label="fitted (MLE)",
               color="#f9ab00", edgecolor="#9a6700")
        for i, (tv, fv) in enumerate(zip(tvals, fvals)):
            ax.text(i - width / 2, tv + 0.01, f"{tv:.2f}", ha="center",
                    va="bottom", fontsize=7.5)
            ax.text(i + width / 2, fv + 0.01, f"{fv:.2f}", ha="center",
                    va="bottom", fontsize=7.5)
        ax.set_title(kb.get_topic(tid).name, fontsize=11, fontweight="bold")
        ax.set_xticks(list(x))
        ax.set_xticklabels(PARAMS, rotation=20)
        ax.set_ylim(0, 0.75)
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("probability")
    axes[0].legend(loc="upper right", fontsize=9)
    fig.suptitle("AdaptIQ — BKT parameter recovery by maximum likelihood\n"
                 "(true parameters vs. parameters re-estimated from simulated logs)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT
    print("Saved parameter-recovery plot ->", draw(out))
