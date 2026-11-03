"""Kinematic bowler over training: accuracy rises, tolerance to release timing does not.

Reads results/kinematic/equifinality_training/<N>M/<run>_summary.json (scripts/kin_equifinality.py
on each 5M checkpoint, no noise, 21 starts) and plots per run: good % without noise, good %
under 12 ms release-timing jitter (exact expectation from the landing curve), and the landing
slope at release.

    python scripts/plot_kin_training.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results" / "kinematic" / "equifinality_training"
RUNS = ["a250_T3a_0", "a250_T3a_1", "a270_T3a_0", "a270_T3a_1", "a290_T3a_1"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # categorical slots 1-5 (validated)
MARKERS = ["o", "s", "^", "D", "v"]                                  # second channel: 3 hues are < 3:1
INK, INK2, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e4e3de", "#fcfcfb"
STEPS = [5, 10, 15, 20, 25, 30]


def main():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), facecolor=SURFACE)
    panels = [("good deliveries, no noise (%)", lambda n: 100 * n["good"] / n["deliveries"], (0, 105)),
              ("good with 12 ms timing jitter (%)", lambda n: n["good_with_timing_jitter_pct"], (0, 105)),
              ("landing moves, m per 10 ms of timing", lambda n: n["median_abs_slope_m_per_10ms"], (0, 15))]
    for ax, (title, f, ylim) in zip(axes, panels):
        for run, c, mk in zip(RUNS, COLORS, MARKERS):
            y = [f(json.load(open(SRC / f"{s}M" / f"kin_bowl_torque_{run}_summary.json"))["none"]) for s in STEPS]
            ang, _, seed = run.split("_")
            ax.plot(STEPS, y, color=c, lw=2, marker=mk, ms=8, mec=SURFACE, mew=1.5, label=f"{ang} seed {seed}")
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(INK2)
        ax.tick_params(colors=INK2, labelsize=10)
        ax.set_ylim(*ylim)
        ax.set_xticks(STEPS)
        ax.set_xlabel("training steps (millions)", color=INK, fontsize=11)
        ax.set_title(title, color=INK, fontsize=12, loc="left")
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower right")
    fig.tight_layout()
    out = ROOT / "results" / "kinematic" / "equifinality_training.png"
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
