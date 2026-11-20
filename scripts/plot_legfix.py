"""Before / after the leg clean-up (scripts/fix_legs.py): each foot's height and sliding speed around release.

    python scripts/plot_legfix.py --delivery T3a   -> results/kinematic/legfix/legfix_T3a.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COL = {"r": "#2a78d6", "l": "#eb6834"}
NAME = {"r": "back (right) foot", "l": "front (left) foot"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delivery", default="T3a")
    a = ap.parse_args()
    z = np.load(ROOT / "data/mocap" / f"legfix_{a.delivery}_feet.npz")
    rep = json.load(open(ROOT / "results/kinematic/legfix" / f"legfix_{a.delivery}.json"))
    t = z["t"]
    w = (t >= -0.62) & (t <= 0.40)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 5.6), sharex=True)
    fig.patch.set_facecolor(BG)
    for c, s in enumerate("rl"):
        for r, what in enumerate(("height", "speed")):
            ax = axes[r, c]
            ax.set_facecolor(BG)
            for pl in rep["plants"][s]:
                t0, t1 = pl["used_stance_s"]
                ax.axvspan(t0, t1, color=COL[s], alpha=0.10, lw=0)
            for tag, ls, lw, col, lab in (("before", (0, (4, 3)), 1.4, INK2, "as fitted"),
                                          ("after", "-", 2.0, COL[s], "cleaned")):
                X = z[f"{tag}_{s}_heel"]
                Y = z[f"{tag}_{s}_ball"]
                if what == "height":
                    v = np.minimum(X[:, 2], Y[:, 2]) * 100
                else:                   # the lower sole point's speed: the one on the floor (a pivot is not a slide)
                    vh, vb = (np.hypot(*np.gradient(P[:, :2], t, axis=0).T) for P in (X, Y))
                    v = np.where(X[:, 2] <= Y[:, 2], vh, vb)
                ax.plot(t[w], v[w], ls=ls, lw=lw, color=col, label=lab)
            ax.axvline(0, color=INK2, lw=0.8)
            ax.axvline(rep["data_end_s"], color=INK2, lw=0.8, ls=":")
            ax.grid(color=GRID, lw=0.8)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.tick_params(colors=INK2, labelsize=8.5)
            if what == "height":
                ax.set_title(NAME[s], color=INK, fontsize=10.5, loc="left")
                ax.set_ylim(-3, 52)
                if c == 0:
                    ax.set_ylabel("lowest sole point (cm)", color=INK, fontsize=9)
                    ax.text(rep["data_end_s"] + 0.01, 47, "last leg\nmarkers", fontsize=7.5, color=INK2, va="top")
                    ax.text(0.01, 47, "release", fontsize=7.5, color=INK2, va="top")
            else:
                ax.set_ylim(0, 8)
                ax.set_xlabel("time from release (s)", color=INK, fontsize=9)
                if c == 0:
                    ax.set_ylabel("lowest point's speed (m/s)", color=INK, fontsize=9)
            if r == 0 and c == 1:
                ax.legend(fontsize=8.5, frameon=False, loc="upper right")
    b, af = rep["before"]["all_feet"], rep["after"]["all_feet"]
    fig.suptitle(f"Legs after the jump, {a.delivery}: shaded = foot planted (video contacts, trimmed to leg reach).  "
                 f"Footskate {b['skate_m']:.1f} m -> {af['skate_m']:.1f} m", x=0.01, ha="left", color=INK, fontsize=10.5)
    fig.tight_layout()
    out = ROOT / "results/kinematic/legfix" / f"legfix_{a.delivery}.png"
    fig.savefig(out, dpi=150, facecolor=BG)
    print("wrote", out)


if __name__ == "__main__":
    main()
