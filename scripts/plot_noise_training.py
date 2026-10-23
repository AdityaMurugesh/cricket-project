"""Does training under motor noise make a policy robust to it? (run15)

Matched pairs: the same release angle, actuator budget and seed, trained without
noise and trained with it, both scored under that noise (scripts/noise_robustness.py,
200 episodes). One line per pair. Only pairs where both final policies release.

    python scripts/plot_noise_training.py results/run15/noise_robustness
"""
import csv
import statistics as st
import sys
from pathlib import Path

import matplotlib.pyplot as plt

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BUDGET = {"s100": ("#2a78d6", "1.0x actuators"), "s150": ("#eb6834", "1.5x actuators")}
PANELS = [("timing", "release-timing jitter, SD 12 ms"), ("torque", "torque noise, SD 15% of command"),
          ("both", "both noises")]


def main():
    d = Path(sys.argv[1])
    R = {}
    for f in sorted(d.glob("*.csv")):
        for r in csv.DictReader(open(f)):
            p = r["run"].split("_")   # throw_ppo_run15_<train>_a<ang>_s<scale>_e0003_<seed>
            R[(p[3], p[4], p[5], p[-1], r["condition"])] = float(r["good_pct"])
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2), sharey=True, facecolor=SURFACE)
    for ax, (cond, title) in zip(axes, PANELS):
        diffs = []
        for (tc, a, s, sd, test), g in sorted(R.items()):
            if tc != cond or test != cond or ("none", a, s, sd, cond) not in R:
                continue
            g0 = R[("none", a, s, sd, cond)]
            diffs.append(g - g0)
            ax.plot([0, 1], [g0, g], color=BUDGET[s][0], lw=2, alpha=0.75, zorder=2)
            ax.scatter([0, 1], [g0, g], s=56, color=BUDGET[s][0], edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.set_title(title, loc="left", color=INK, fontsize=10.5, pad=34)
        ax.text(0, 1.02, f"{len(diffs)} pairs: mean {st.mean(diffs):+.1f} points, median {st.median(diffs):+.1f}\n"
                         f"{sum(x > 0 for x in diffs)} better, {sum(x < 0 for x in diffs)} worse",
                transform=ax.transAxes, ha="left", va="bottom", color=INK_2, fontsize=8.5)
        ax.set_xticks([0, 1], ["trained\nwithout noise", "trained\nwith it"])
        ax.set_xlim(-0.3, 1.3)
        ax.set_ylim(-3, 105)
        ax.set_facecolor(SURFACE)
        ax.grid(True, axis="y", color=GRID, lw=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(AXIS)
    axes[0].set_ylabel("legal + good length under that noise (%)", color=INK_2)
    handles = [plt.Line2D([], [], marker="o", color=c, lw=2, markersize=7, label=l) for c, l in BUDGET.values()]
    axes[2].legend(handles=handles, frameon=False, loc="upper right", labelcolor=INK_2, fontsize=9)
    fig.tight_layout()
    out = d.parent / "noise_training.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
