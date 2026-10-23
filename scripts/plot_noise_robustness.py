"""Noise-free policies scored under motor noise: good % against release speed.

Reads scripts/noise_robustness.py output. Left: release-timing jitter; right:
signal-dependent torque noise. Colour = actuator budget.

    python scripts/plot_noise_robustness.py results/run13b/noise_robustness
"""
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BUDGET = {"100": ("#2a78d6", "1.0x actuators"), "150": ("#eb6834", "1.5x actuators")}


def main():
    d = Path(sys.argv[1])
    rows = [r for f in sorted(d.glob("*.csv")) for r in csv.DictReader(open(f))]
    speed = {r["run"]: float(r["mean_speed_kmh"]) for r in rows if r["condition"] == "none"}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True, facecolor=SURFACE)
    for ax, (cond, title) in zip(axes, [("timing", "release-timing jitter, SD 12 ms"),
                                        ("torque", "torque noise, SD 15% of command")]):
        for r in rows:
            if r["condition"] != cond:
                continue
            scale = r["run"].split("_s")[-1][:3]
            ax.scatter(speed[r["run"]], float(r["good_pct"]), s=64, color=BUDGET[scale][0],
                       edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.set_title(title, loc="left", color=INK, fontsize=10.5)
        ax.set_xlabel("release speed of the noise-free delivery (km/h)", color=INK_2)
        ax.set_ylim(-3, 105)
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, lw=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(AXIS)
    axes[0].set_ylabel("legal + good length under noise (%)\n(100% without noise)", color=INK_2)
    handles = [plt.Line2D([], [], marker="o", ls="", markersize=8, color=c, label=l) for c, l in BUDGET.values()]
    axes[1].legend(handles=handles, frameon=False, loc="center right", labelcolor=INK_2, fontsize=9)
    fig.tight_layout()
    out = d.parent / "noise_robustness.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
