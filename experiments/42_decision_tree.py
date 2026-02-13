#!/usr/bin/env python3
"""
Experiment 42: Decision Tree for Decay Method Selection

Generates a publication-quality decision tree / flowchart summarizing
the practical guidance from 41 VDD experiments. Helps practitioners
choose the right decay strategy based on their knowledge-base
characteristics and drift patterns.

No external dependencies beyond matplotlib.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path
import shutil

RESULTS_DIR = Path(__file__).parent.parent / "results"
ARXIV_DIR = Path(__file__).parent.parent / "arxiv_submission" / "figures"

C = {
    "start": "#CFD8DC",
    "q": "#BBDEFB",
    "rec": "#C8E6C9",
    "rec_bold": "#81C784",
    "caution": "#FFF9C4",
    "note": "#E1BEE7",
    "ev": "#ECEFF1",
    "edge": "#37474F",
    "txt": "#212121",
}


def build():
    fig, ax = plt.subplots(figsize=(14, 9.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9.5)
    ax.axis("off")
    ax.set_aspect("equal")

    def rbox(cx, cy, w, h, text, fc, fs=8.5, bold=False,
             ec="#546E7A", lw=1.1):
        p = FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.08",
            facecolor=fc, edgecolor=ec, linewidth=lw,
            alpha=0.95, zorder=2, transform=ax.transData)
        ax.add_patch(p)
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal",
                color=C["txt"], linespacing=1.2, zorder=3)
        return (cx, cy, w, h)

    def arrow(x0, y0, x1, y1, color=None, lw=1.2, rad=0):
        if color is None:
            color = C["edge"]
        cs = f"arc3,rad={rad}"
        a = FancyArrowPatch(
            (x0, y0), (x1, y1),
            arrowstyle="->,head_width=0.15,head_length=0.12",
            color=color, linewidth=lw, zorder=1,
            connectionstyle=cs, mutation_scale=12)
        ax.add_patch(a)

    def label(x, y, text, fs=7, color=None, ha="center", va="center"):
        if color is None:
            color = C["edge"]
        ax.text(x, y, text, ha=ha, va=va, fontsize=fs, fontstyle="italic",
                color=color, zorder=4,
                bbox=dict(boxstyle="round,pad=0.06", fc="white",
                          ec="none", alpha=0.9))

    # ================================================================
    # Layout in data coordinates (14 x 9.5)
    # ================================================================

    # Y rows (top to bottom)
    Y0 = 8.7     # start
    Y1 = 7.5     # Q1
    Y1L = 6.5    # accumulation leaf
    Y2 = 5.5     # Q2
    Y3 = 4.0     # Q3
    Y4 = 2.2     # leaves
    Yev = 0.65   # evidence

    # X positions
    XM = 5.5     # main column
    XL = 1.5     # accumulation
    XR = 10.8    # VDD default

    # Leaf X
    L1 = 1.4
    L2 = 4.5
    L3 = 7.6
    L4 = 10.7

    # Side panel
    XS = 12.5
    YS1 = 8.5
    YS2 = 7.3

    # ================================================================
    # BOXES
    # ================================================================

    # Start
    rbox(XM, Y0, 2.6, 0.65,
         "Which Decay Method\nShould I Use?",
         C["start"], fs=11, bold=True, ec="#78909C", lw=1.8)

    # Q1
    rbox(XM, Y1, 3.4, 0.7,
         "Is your knowledge base\naccumulation or replacement?",
         C["q"], fs=9.5, bold=True)

    # Accumulation
    rbox(XL, Y1L, 2.2, 1.1,
         "No Decay\n(or minimal)\n\nOld facts stay valid.",
         C["rec"], fs=8.5)

    # Q2
    rbox(XM, Y2, 2.8, 0.7,
         "Do you know your\ndrift pattern?",
         C["q"], fs=9.5, bold=True)

    # VDD default
    rbox(XR, Y2, 2.4, 1.3,
         "VDD\n(safest default)\n\nAdaptive, never worst.\nNo manual tuning.",
         C["rec_bold"], fs=9, bold=True, ec="#2E7D32", lw=2.0)

    # Q3
    rbox(XM, Y3, 3.2, 0.6,
         "What is the drift pattern?",
         C["q"], fs=9.5, bold=True)

    # Leaf 1: Recency
    rbox(L1, Y4, 2.2, 1.3,
         "Recency\n(\u03bb \u2265 0.5)\n\nConstant high drift.\nNews, social media.",
         C["rec"], fs=8.5)

    # Leaf 2: Time-weighted
    rbox(L2, Y4, 2.2, 1.3,
         "Time-Weighted\n(\u03b1 = 0.01)\n\nGradual, predictable\nevolution.",
         C["rec"], fs=8.5)

    # Leaf 3: Static
    rbox(L3, Y4, 2.2, 1.3,
         "Static Decay\n(\u03bb \u2248 0.1)\n\nMostly stable,\nrare changes.",
         C["caution"], fs=8.5)

    # Leaf 4: VDD reversions
    rbox(L4, Y4, 2.2, 1.3,
         "VDD\n\nReversions possible.\nNon-monotonic drift.",
         C["rec"], fs=8.5)

    # Side: labeled data
    rbox(XS, YS1, 1.8, 0.55,
         "Have labeled\ntemporal data?",
         C["note"], fs=8, bold=True, ec="#7B1FA2", lw=1.3)

    # Side: online lambda
    rbox(XS, YS2, 1.8, 0.9,
         "Online Lambda\n\nMatches VDD\n(d = \u22120.056)\nbut needs labels.",
         C["caution"], fs=7.5, ec="#F57F17", lw=1.1)

    # ================================================================
    # ARROWS
    # ================================================================
    g = 0.08

    # Start -> Q1
    arrow(XM, Y0 - 0.65 / 2 - g, XM, Y1 + 0.7 / 2 + g)

    # Q1 -> Accumulation (down-left)
    arrow(XM - 3.4 / 2 - g, Y1 - 0.15, XL + 2.2 / 2 + g, Y1L + 0.35,
          color="#2E7D32", rad=0.25)
    label(2.8, Y1 - 0.05, "Accumulation", fs=8, color="#2E7D32")

    # Q1 -> Q2 (down)
    arrow(XM, Y1 - 0.7 / 2 - g, XM, Y2 + 0.7 / 2 + g)
    label(XM + 0.5, (Y1 - 0.35 + Y2 + 0.35) / 2, "Replacement",
          fs=8, ha="left")

    # Q2 -> VDD default (right)
    arrow(XM + 2.8 / 2 + g, Y2, XR - 2.4 / 2 - g, Y2,
          color="#1565C0")
    label((XM + 1.4 + XR - 1.2) / 2, Y2 + 0.38,
          "No / Mixed / Unpredictable", fs=7.5, color="#1565C0")

    # Q2 -> Q3 (down)
    arrow(XM, Y2 - 0.7 / 2 - g, XM, Y3 + 0.6 / 2 + g)
    label(XM - 0.45, (Y2 - 0.35 + Y3 + 0.3) / 2, "Yes",
          fs=8, ha="right")

    # Q3 -> 4 leaves via bus bar
    q3b = Y3 - 0.3 - g
    lt = Y4 + 1.3 / 2 + g
    jy = (q3b + lt) / 2

    # Vertical stub from Q3 down
    ax.plot([XM, XM], [q3b, jy], color=C["edge"], lw=1.2,
            zorder=1, solid_capstyle="round")
    # Horizontal bar
    ax.plot([L1, L4], [jy, jy], color=C["edge"], lw=1.2,
            zorder=1, solid_capstyle="round")

    leaf_labels = [
        (L1, "Constant\nhigh drift"),
        (L2, "Gradual"),
        (L3, "Mostly\nstable"),
        (L4, "Reversions"),
    ]
    for lx, pl in leaf_labels:
        arrow(lx, jy - 0.03, lx, lt)
        ax.plot([lx, lx], [jy - 0.06, jy + 0.06],
                color=C["edge"], lw=1.6, zorder=1)
        label(lx, jy + 0.30, pl, fs=7)

    # Side: start -> side question
    arrow(XM + 2.6 / 2 + g, Y0 - 0.05,
          XS - 1.8 / 2 - g, YS1 + 0.1,
          color="#7B1FA2", lw=0.9, rad=-0.15)
    label((XM + 1.3 + XS - 0.9) / 2, Y0 + 0.22,
          "Side path", fs=6.5, color="#7B1FA2")

    # Side question -> online lambda
    arrow(XS, YS1 - 0.55 / 2 - g, XS, YS2 + 0.9 / 2 + g,
          color="#7B1FA2", lw=0.9)
    label(XS + 0.4, (YS1 - 0.275 + YS2 + 0.45) / 2,
          "Yes", fs=7, color="#7B1FA2", ha="left")

    # ================================================================
    # Evidence bar
    # ================================================================
    rbox(6.0, Yev, 12.0, 0.7,
         ("Key evidence:   VDD \"never worst\" across 42 experiments   |   "
          "Time-weighted best for gradual (0.87\u20130.89 acc)   |   "
          "Recency best for constant drift\n"
          "Online \u03bb matches VDD but requires supervision   |   "
          "All sigmoid activations converge at k \u2265 5   |   "
          "13.5% precision, reframed honestly"),
         C["ev"], fs=7.5, ec="#90A4AE", lw=0.7)

    # ================================================================
    # Legend
    # ================================================================
    handles = [
        mpatches.Patch(fc=C["start"], ec="#78909C", label="Start"),
        mpatches.Patch(fc=C["q"], ec="#546E7A", label="Decision question"),
        mpatches.Patch(fc=C["rec"], ec="#546E7A", label="Recommended method"),
        mpatches.Patch(fc=C["caution"], ec="#546E7A",
                       label="Conditional / caution"),
        mpatches.Patch(fc=C["note"], ec="#7B1FA2",
                       label="Optional (needs labels)"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=7,
              framealpha=0.90, edgecolor="#B0BEC5",
              title="Legend", title_fontsize=7.5,
              bbox_to_anchor=(0.0, 0.0),
              bbox_transform=ax.transAxes)

    # ================================================================
    # Title & footer
    # ================================================================
    fig.suptitle(
        "Practitioner Decision Tree: Choosing a Decay Strategy",
        fontsize=14, fontweight="bold", y=0.98, color=C["txt"])

    ax.text(7.0, 0.15,
            "Based on 42 experiments across 3 domains (React, Python, "
            "Node.js) with 120 real-world facts.  "
            "See VDD paper for full analysis.",
            ha="center", va="center", fontsize=6.5, color="#757575",
            fontstyle="italic")

    fig.tight_layout(rect=[0, 0.0, 1, 0.96])
    return fig


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ARXIV_DIR.mkdir(parents=True, exist_ok=True)

    print("Experiment 42: Decision Tree for Decay Method Selection")
    print("=" * 60)

    fig = build()

    out = RESULTS_DIR / "42_decision_tree.png"
    fig.savefig(out, dpi=350, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"  Saved: {out}")

    out2 = ARXIV_DIR / "42_decision_tree.png"
    shutil.copy2(out, out2)
    print(f"  Copied: {out2}")

    plt.close(fig)
    print("\nDone.")


if __name__ == "__main__":
    main()
