"""
AdaptIQ — draw_graph.py
=======================

Render the prerequisite semantic net as a PNG so the knowledge structure is
visible at a glance (foundations on the left, advanced topics on the right).

Layout is computed from scratch: each topic's column = its "rank" (the longest
prerequisite chain leading to it), so an edge always points left-to-right.
Nodes are coloured by intrinsic difficulty (1..5). Matplotlib is the only
external dependency; the layout and all graph traversal are hand-written, so
no graph library is used.

Usage:
    python draw_graph.py [output.png]
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")  # headless: render to file, no display needed
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from knowledge_base import KnowledgeBase

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(_HERE, "prerequisite_graph.png")


def _wrap_label(name: str, max_line: int = 12) -> str:
    """Balance a topic name onto (at most) two lines for tidy box labels."""
    if len(name) <= max_line:
        return name
    words = name.split()
    if len(words) == 1:
        return name
    # Choose the split point that most evenly balances the two lines.
    best_split, best_diff = 1, 10 ** 9
    for i in range(1, len(words)):
        left = " ".join(words[:i])
        right = " ".join(words[i:])
        diff = abs(len(left) - len(right))
        if diff < best_diff:
            best_diff, best_split = diff, i
    return " ".join(words[:best_split]) + "\n" + " ".join(words[best_split:])


def _compute_ranks(kb: KnowledgeBase) -> Dict[str, int]:
    """rank = length of the longest prerequisite chain ending at a topic."""
    rank: Dict[str, int] = {}
    for tid in kb.topological_order():   # guarantees prereqs are ranked first
        pres = kb.prerequisites(tid)
        rank[tid] = 0 if not pres else 1 + max(rank[p] for p in pres)
    return rank


def draw(output_path: str = DEFAULT_OUT) -> str:
    kb = KnowledgeBase()
    rank = _compute_ranks(kb)

    # Group topics into columns by rank; order within a column by difficulty.
    columns: Dict[int, List[str]] = defaultdict(list)
    for tid, r in rank.items():
        columns[r].append(tid)
    for r in columns:
        columns[r].sort(key=lambda t: (kb.get_topic(t).difficulty, kb.get_topic(t).name))

    max_rows = max(len(c) for c in columns.values())
    n_cols = max(columns) + 1

    # Assign (x, y) positions. Columns spread on x; nodes centred on y per column.
    x_gap, y_gap = 3.6, 2.1
    pos: Dict[str, tuple] = {}
    for r, tids in columns.items():
        n = len(tids)
        offset = (max_rows - n) / 2.0
        for i, tid in enumerate(tids):
            pos[tid] = (r * x_gap, (offset + i) * y_gap)

    fig_w = max(9, n_cols * 2.7)
    fig_h = max(6, max_rows * 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)

    norm = Normalize(vmin=1, vmax=5)
    cmap = plt.get_cmap("YlOrRd")

    box_w, box_h = 2.3, 1.02
    edge_off = box_w / 2 + 0.05

    # Edges first (so nodes draw on top). prereq -> topic, left to right.
    for tid in kb.topics:
        x2, y2 = pos[tid]
        for pre in kb.prerequisites(tid):
            x1, y1 = pos[pre]
            arrow = FancyArrowPatch(
                (x1 + edge_off, y1), (x2 - edge_off, y2),
                arrowstyle="-|>", mutation_scale=14,
                connectionstyle="arc3,rad=0.06",
                color="#9aa0a6", linewidth=1.3, zorder=1,
            )
            ax.add_patch(arrow)

    # Nodes.
    for tid, (x, y) in pos.items():
        t = kb.get_topic(tid)
        color = cmap(norm(t.difficulty))
        box = FancyBboxPatch(
            (x - box_w / 2, y - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.02,rounding_size=0.18",
            linewidth=1.4, edgecolor="#3c4043", facecolor=color, zorder=2,
        )
        ax.add_patch(box)
        # Dark text on light fills, light text on dark (high-difficulty) fills.
        txt_color = "#202124" if t.difficulty <= 3 else "white"
        ax.text(x, y, _wrap_label(t.name), ha="center", va="center", fontsize=8.5,
                fontweight="bold", color=txt_color, zorder=3)

    # Cosmetics.
    ax.set_xlim(-1.6, (n_cols - 1) * x_gap + 1.6)
    ax.set_ylim(-1.4, max_rows * y_gap)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("AdaptIQ — DSA Prerequisite Semantic Net\n"
                 "(arrows point from prerequisite to dependent topic)",
                 fontsize=13, fontweight="bold", pad=16)

    # Difficulty colour legend.
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, ticks=[1, 2, 3, 4, 5])
    cbar.set_label("intrinsic difficulty", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT
    saved = draw(out)
    print(f"Saved prerequisite graph -> {saved}")
