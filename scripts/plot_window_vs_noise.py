"""The equifinal window predicts robustness to release-timing jitter (run15).

x: good % under 12 ms jitter predicted from the noise-free delivery alone --
scripts/equifinality.py's p_jitter_12ms, the N(0, 12 ms)-weighted share of release
instants within +/-40 ms whose landing is in the zone. y: good % measured over 200
jittered episodes (scripts/noise_robustness.py). Every final policy that releases.

    python scripts/plot_window_vs_noise.py results/run15
"""
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
TRAIN = {"none": ("#2a78d6", "trained without noise"), "timing": ("#eb6834", "trained with timing jitter"),
         "torque": ("#1f9e89", "trained with torque noise"), "both": ("#8a5cd0", "trained with both")}


def key(run):
    p = run.split("_")   # throw_ppo_run15_<train>_a<ang>_s<scale>_e0003_<seed>
    return p[3], p[4], p[5], p[-1]


def main():
    d = Path(sys.argv[1])
    win = {}
    for f in sorted((d / "equifinality").glob("*.csv")):
        r = list(csv.DictReader(open(f)))[-1]
        if r["land_x"]:
            win[key(r["run"])] = 100 * float(r["p_jitter_12ms"])
    meas = {}
    for f in sorted((d / "noise_robustness").glob("*.csv")):
        for r in csv.DictReader(open(f)):
            if r["condition"] == "timing":
                meas[key(r["run"])] = float(r["good_pct"])
    fig, ax = plt.subplots(figsize=(5.6, 5.0), facecolor=SURFACE)
    xs, ys = [], []
    for k, x in sorted(win.items()):
        if k not in meas:
            continue
        xs.append(x)
        ys.append(meas[k])
        ax.scatter(x, meas[k], s=56, color=TRAIN[k[0]][0], edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.plot([0, 100], [0, 100], color=AXIS, lw=1, ls="--", zorder=1)
    r = np.corrcoef(xs, ys)[0, 1]
    mae = np.mean(np.abs(np.array(xs) - np.array(ys)))
    ax.set_title("the equifinal window predicts robustness to jitter", loc="left", color=INK, fontsize=10.5, pad=22)
    ax.text(0, 1.02, f"{len(xs)} final policies: r = {r:.2f}, mean error {mae:.1f} points",
            transform=ax.transAxes, color=INK_2, fontsize=8.5)
    ax.set_xlabel("predicted from the noise-free landing curve (%)", color=INK_2)
    ax.set_ylabel("measured, 200 episodes with 12 ms jitter (%)", color=INK_2)
    ax.set_xlim(-3, 103)
    ax.set_ylim(-3, 103)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)
    handles = [plt.Line2D([], [], marker="o", ls="", markersize=7, color=c, label=l) for c, l in TRAIN.values()]
    ax.legend(handles=handles, frameon=False, loc="upper left", labelcolor=INK_2, fontsize=8.5)
    fig.tight_layout()
    out = d / "window_vs_noise.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
