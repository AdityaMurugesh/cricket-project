"""How RL moves away from the CMA-ES start: the good-ball rate of the PPO checkpoints of the cvr runs (hpc/packs/cvr_2.txt,
scored by compare_cmaes_rl.py as results/kinematic/cmaes_vs_rl/cvr_a270_<skill>_ck.json), against training steps, with
no motor noise (solid) and with 15% motor noise (dashed), for PPO trained noise-free (orange) and with motor noise
(yellow). Step 0 is the behaviour clone (from cvr_a270_<skill>.json's bc item), and the grey line is the CMA-ES plan
itself under 15% motor noise. One panel per timing skill, at that skill's release error.

    python scripts/plot_cvr_learning.py --skills j2 j4 j8 j12 fast --sd 2 4 8 12 2
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "results/kinematic/cmaes_vs_rl"
INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, YELLOW = "#2a78d6", "#eb6834", "#eda100"
NAME = re.compile(r"^(rl|rltn)_(\d+)(?:_steps)?$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skills", nargs="+", default=["j2", "j4", "j8", "j12", "fast"])
    ap.add_argument("--sd", nargs="+", type=float, default=[2, 4, 8, 12, 2])
    ap.add_argument("--noisy", default="t0.15", help="the motor-noise condition drawn dashed")
    ap.add_argument("--out", default=str(DIR / "cvr_learning.png"))
    a = ap.parse_args()
    n = len(a.skills)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.7), squeeze=False, sharey=True)
    fig.patch.set_facecolor(BG)
    first = None
    for ax, skill, sd in zip(axes[0], a.skills, a.sd):
        ax.set_facecolor(BG)
        key = f"{sd:g}"
        ck_f, main_f = DIR / f"cvr_a270_{skill}_ck.json", DIR / f"cvr_a270_{skill}.json"
        if not ck_f.exists():
            ax.axis("off")
            ax.text(0.5, 0.5, f"{skill}: not scored yet", color=INK2, fontsize=10, ha="center", transform=ax.transAxes)
            continue
        first = first or ax
        ck = json.load(open(ck_f))
        main = json.load(open(main_f)) if main_f.exists() else {}
        bc = main.get("bc")
        for who, col in (("rl", ORANGE), ("rltn", YELLOW)):
            pts = sorted((int(m.group(2)), v) for k, v in ck.items() if (m := NAME.match(k)) and m.group(1) == who)
            if not pts:
                continue
            for cond, ls in (("none", "-"), (a.noisy, "--")):
                xs = [0.0] if bc else []
                ys = [100 * bc[cond]["p_cont"][key]] if bc else []
                xs += [s / 1e6 for s, _ in pts]
                ys += [100 * v[cond]["p_cont"][key] for _, v in pts]
                ax.plot(xs, ys, color=col, ls=ls, lw=2, marker="o", ms=4, mec=BG, mew=0.8)
            print(f"{skill} {who}: final ({pts[-1][0] / 1e6:.1f}M) none {100 * pts[-1][1]['none']['p_cont'][key]:.0f}% "
                  f"{a.noisy} {100 * pts[-1][1][a.noisy]['p_cont'][key]:.0f}%  "
                  f"{pts[-1][1]['none']['speed_kmh'] or 0:.0f} km/h")
        if "cmaes" in main:
            ax.axhline(100 * main["cmaes"][a.noisy]["p_cont"][key], color=BLUE, lw=1.4, ls="--")
            ax.axhline(100 * main["cmaes"]["none"]["p_cont"][key], color=BLUE, lw=1.4)
        ax.set_title(f"{'fastest ball' if skill == 'fast' else skill[1:] + ' ms'} (good rate at {sd:g} ms)",
                     color=INK, fontsize=10, loc="left")
        ax.set_xlabel("PPO training steps (millions)", color=INK, fontsize=9)
        ax.set_ylim(-3, 103)
        ax.grid(color=GRID, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=8.5)
    if first is not None:
        first.set_ylabel("good balls (%)", color=INK, fontsize=9.5)
        first.yaxis.set_tick_params(labelleft=True)
    handles = [plt.Line2D([], [], color=ORANGE, lw=2, label="RL, trained noise-free"),
               plt.Line2D([], [], color=YELLOW, lw=2, label="RL, trained with motor noise"),
               plt.Line2D([], [], color=BLUE, lw=1.4, label="the CMA-ES plan"),
               plt.Line2D([], [], color=INK2, lw=2, ls="-", label="scored with no motor noise"),
               plt.Line2D([], [], color=INK2, lw=2, ls="--", label=f"scored with {float(a.noisy[1:]) * 100:g}% motor noise")]
    ncol = 5 if n >= 4 else 3
    fig.legend(handles=handles, loc="lower center", ncol=ncol, fontsize=8.5, frameon=False)
    fig.tight_layout(rect=(0, 0.08 if ncol == 5 else 0.15, 1, 1))
    fig.savefig(a.out, dpi=150, facecolor=BG)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
