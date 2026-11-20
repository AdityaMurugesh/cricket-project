"""CMA-ES alone vs CMA-ES followed by RL, under the same noise (results/kinematic/cmaes_vs_rl/<label>.json from
scripts/compare_cmaes_rl.py). One panel per comparison: the good-ball rate at that panel's timing skill against the
motor noise (0, 5%, 15% per command), for each candidate, labelled with its speed (km/h, no motor noise). The last
cell is the key.

    python scripts/plot_cmaes_vs_rl.py --labels prelim_a270_fast prelim_a290_j2 ... --sd 2 2 4 8 12 \
        --titles ... --out results/kinematic/cmaes_vs_rl/cmaes_vs_rl_prelim.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "results/kinematic/cmaes_vs_rl"
INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, GREEN, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
# kind -> colour, marker, short name; a second candidate of the same kind in a panel is drawn dashed
STYLE = {"cmaes": (BLUE, "o", "CMA-ES"), "cmaes_tn": (GREEN, "D", "CMA-ES, motor-noise opt."),
         "bc": (INK2, "s", "clone"), "rl": (ORANGE, "o", "RL"), "rl_tn": (YELLOW, "D", "RL, motor-noise trained"),
         "rl_scratch": (ORANGE, "^", "RL from scratch")}
KEY = [("cmaes", "CMA-ES plan: open loop, the same torques whatever happens"),
       ("rl", "RL policy: closed loop, sees the arm every 20 ms"),
       ("cmaes_tn", "CMA-ES plan optimised under motor noise"),
       ("bc", "the plan cloned into the network, before RL"),
       ("rl_tn", "RL trained with motor noise")]


def kind(name):
    for k in ("cmaes_tn", "cmaes", "bc", "rl_tn", "rl_scratch", "rl"):
        if name.startswith(k):
            return k
    return "rl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--sd", nargs="+", type=float, required=True, help="timing skill per label (ms)")
    ap.add_argument("--titles", nargs="+", default=None)
    ap.add_argument("--ncols", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    n = len(a.labels)
    nrow = int(np.ceil((n + 1) / a.ncols))
    fig, axes = plt.subplots(nrow, a.ncols, figsize=(4.0 * a.ncols, 3.5 * nrow), squeeze=False)
    fig.patch.set_facecolor(BG)
    axes = axes.ravel()
    kinds_seen = set()
    for p, lab in enumerate(a.labels):
        ax = axes[p]
        d = json.load(open(DIR / f"{lab}.json"))
        sd = a.sd[p]
        first = next(k for k in d if not k.startswith("_"))
        conds = ["none"] + sorted([c for c in d[first] if c.startswith("t0")], key=lambda c: float(c[1:]))
        xs = np.arange(len(conds))
        ax.set_facecolor(BG)
        count = {}
        for name, item in d.items():
            if name.startswith("_"):
                continue
            k = kind(name)
            kinds_seen.add(k)
            col, mk, short = STYLE[k]
            count[k] = count.get(k, 0) + 1
            ls = "-" if count[k] == 1 and k != "rl_scratch" else "--"
            ys = [100 * item[c]["p_cont"][f"{sd:g}"] for c in conds]
            kmh = item["none"]["speed_kmh"] or 0
            ax.plot(xs, ys, ls=ls, marker=mk, color=col, lw=2, ms=6.5, mec=BG, mew=1.2,
                    label=f"{short} {kmh:.0f} km/h")
        ax.set_xticks(xs)
        ax.set_xticklabels(["none" if c == "none" else f"{float(c[1:]) * 100:g}%" for c in conds])
        ax.set_title(a.titles[p] if a.titles else lab, color=INK, fontsize=10.5, loc="left")
        ax.grid(color=GRID, lw=0.8)
        ax.set_ylim(-3, 103)
        ax.set_xlim(-0.25, len(conds) - 0.75)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=8.5)
        ax.legend(fontsize=8, frameon=True, facecolor=BG, edgecolor=GRID, loc="best", handlelength=3.2, numpoints=1)
        if p % a.ncols == 0:
            ax.set_ylabel("good balls (%)", color=INK, fontsize=9.5)
        if p >= n - a.ncols:
            ax.set_xlabel("motor noise per 20 ms command", color=INK, fontsize=9)
    key = axes[n]
    key.axis("off")
    y = 0.92
    key.text(0.0, y, "How to read it", fontsize=10.5, color=INK, fontweight="bold", transform=key.transAxes)
    for k, text in KEY:
        if k not in kinds_seen:
            continue
        y -= 0.13
        col, mk, _ = STYLE[k]
        key.plot([0.02, 0.12], [y + 0.02, y + 0.02], color=col, lw=2, marker=mk, ms=6, transform=key.transAxes)
        key.text(0.16, y, text, fontsize=8.8, color=INK2, transform=key.transAxes)
    y -= 0.17
    key.text(0.0, y, "Each point: 63 deliveries (21 starts x 3 noise draws),\nthe same draws for every bowler. "
             "Good-ball rate at\nthe panel's release-timing error.", fontsize=8.5, color=INK2,
             transform=key.transAxes, va="top")
    for ax in axes[n + 1:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(a.out, dpi=150, facecolor=BG)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
