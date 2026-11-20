"""CMA-ES alone vs CMA-ES followed by RL, under the same noise (results/kinematic/cmaes_vs_rl/<label>.json from
scripts/compare_cmaes_rl.py). One panel per comparison: the good-ball rate at that panel's timing skill against the
motor noise (0, 5%, 15% per command), for each candidate; speeds (km/h, no motor noise) in the legend.

    python scripts/plot_cmaes_vs_rl.py --labels prelim_a270_j4 prelim_a270_j12 ... --out <png>
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
# fixed colour per kind of candidate (validated categorical slots 1, 2, 4 + a neutral)
STYLE = {"cmaes": ("#2a78d6", "-", "o"), "cmaes_tn": ("#1baf7a", "-", "D"), "bc": (INK2, ":", "s"),
         "rl": ("#eb6834", "-", "o"), "rl_tn": ("#eda100", "-", "D"), "rl_scratch": ("#eb6834", "--", "^")}
NAMES = {"cmaes": "CMA-ES plan (open loop)", "cmaes_tn": "CMA-ES plan optimised under motor noise",
         "bc": "clone of the plan (before RL)", "rl": "RL policy (closed loop)",
         "rl_tn": "RL trained with motor noise", "rl_scratch": "RL from scratch"}


def kind(name):
    for k in ("cmaes_tn", "cmaes", "bc", "rl_tn", "rl_scratch", "rl"):
        if name.startswith(k):
            return k
    return "rl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--sd", nargs="+", type=float, default=None, help="timing skill per label (ms); default from label")
    ap.add_argument("--titles", nargs="+", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    n = len(a.labels)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n + 0.6, 4.3), sharey=True)
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor(BG)
    for p, (lab, ax) in enumerate(zip(a.labels, axes)):
        d = json.load(open(DIR / f"{lab}.json"))
        sd = a.sd[p] if a.sd else (2.0 if "fast" in lab else float(lab.split("_j")[-1].split("_")[0]))
        conds = ["none"] + sorted([c for c in d[next(k for k in d if not k.startswith("_"))] if c.startswith("t0")],
                                  key=lambda c: float(c[1:]))
        xs = [0.0] + [float(c[1:]) * 100 for c in conds[1:]]
        ax.set_facecolor(BG)
        for name, item in d.items():
            if name.startswith("_"):
                continue
            k = kind(name)
            col, ls, mk = STYLE[k]
            ys = [100 * item[c]["p_cont"][f"{sd:g}"] for c in conds]
            kmh = item["none"]["speed_kmh"] or 0
            extra = name[len(k):].strip("_")
            ax.plot(xs, ys, ls=ls, marker=mk, color=col, lw=2, ms=7, mec=BG, mew=1.5,
                    label=f"{NAMES[k]}{' ' + extra if extra else ''}: {kmh:.0f} km/h")
        ax.set_xticks(xs)
        ax.set_xticklabels(["none" if x == 0 else f"{x:g}%" for x in xs])
        ax.set_xlabel("motor noise (per 20 ms command)", color=INK, fontsize=9)
        ax.set_title(a.titles[p] if a.titles else f"{lab}: timing error {sd:g} ms", color=INK, fontsize=10, loc="left")
        ax.grid(color=GRID, lw=0.8)
        ax.set_ylim(-3, 103)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=8.5)
        ax.legend(fontsize=7.2, frameon=False, loc="upper right")
    axes[0].set_ylabel("good balls (%)", color=INK)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150, facecolor=BG)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
