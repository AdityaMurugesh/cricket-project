"""Figure for the whole-body tracking runs on the dynamic G1 (scripts/train_g1_track.py).

(a) how far into the delivery the controller gets when it starts at the clip's first frame, over training;
(b) the share of training episodes (random start points) that reach the clip's end;
(c) the start-point test (scripts/eval_g1_track_starts.py): from each start, where it falls (or that it finishes).
The delivery's phases (from the reference's foot heights) are shaded.

Inputs (copied from the HPC): results/kinematic/g1/track_<run>_eval.csv, results/kinematic/g1/track_starts_<run>.json
    python scripts/plot_g1_track.py  -> results/kinematic/g1/track_summary.png
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
G1 = ROOT / "results" / "kinematic" / "g1"
RUNS = [("g1track_a025_s0", "corrections ±0.25 rad, lr 3e-4"),
        ("g1track_a025_lr1e4_s1", "±0.25 rad, lr 1e-4"),
        ("g1track_a050_lr1e4_s2", "±0.5 rad, lr 1e-4")]
COLORS = ["#257de0", "#f47230", "#00b894"]          # categorical slots 1-3, fixed order
PHASES = [(-1.0, -0.55, "run-up\n(3-3.7 m/s)"), (-0.55, -0.24, "bound"), (-0.24, 0.0, "delivery\nstride"),
          (0.0, 0.34, "follow-\nthrough")]
SHADE = ["#f4f4f4", "#e6e6e6", "#f4f4f4", "#e6e6e6"]
INK, INK2 = "#191919", "#666666"


def phases(ax, axis="y"):
    for (a, b, name), c in zip(PHASES, SHADE):
        (ax.axhspan if axis == "y" else ax.axvspan)(a, b, color=c, zorder=0, lw=0)
    if axis == "y":
        for a, b, name in PHASES:
            ax.text(1.01, (a + b) / 2, name, transform=ax.get_yaxis_transform(), va="center", fontsize=7.5,
                    color=INK2)
    else:
        for a, b, name in PHASES:
            ax.text((a + b) / 2, 0.02, name.split("(")[0].strip(), transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=7.5, color=INK2)
    (ax.axhline if axis == "y" else ax.axvline)(0.0, color=INK2, lw=0.8, ls="--", zorder=1)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


def main():
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw=dict(wspace=0.42))
    for (run, label), col in zip(RUNS, COLORS):
        f = G1 / f"track_{run}_eval.csv"
        if not f.exists():
            continue
        rows = list(csv.DictReader(open(f)))
        x = [int(r["timesteps"]) / 1e6 for r in rows]
        y = [0.34 if r["completed"] == "True" else float(r["reached_t_rel"]) for r in rows]
        axs[0].plot(x, y, "-o", color=col, lw=2, ms=5, label=label, zorder=3)
        axs[1].plot(x, [float(r["rsi_completed_pct"]) for r in rows], "-o", color=col, lw=2, ms=5, label=label,
                    zorder=3)
        s = G1 / f"track_starts_{run}.json"
        if s.exists():
            st = json.loads(s.read_text())["starts"]
            t0 = [r["start_t_rel"] for r in st]
            t1 = [0.34 if r["completed"] else r["reached_t_rel"] for r in st]
            axs[2].plot(t0, t1, "-", color=col, lw=2, zorder=3, label=label)
            done = [(a, b) for a, b, r in zip(t0, t1, st) if r["completed"]]
            fell = [(a, b) for a, b, r in zip(t0, t1, st) if not r["completed"]]
            if done:
                axs[2].plot(*zip(*done), "o", color=col, ms=7, mec="white", mew=1.5, zorder=4)
            if fell:
                axs[2].plot(*zip(*fell), "o", color="white", ms=6, mec=col, mew=1.8, zorder=4)
    phases(axs[0])
    axs[0].set_ylim(-1.0, 0.42)
    axs[0].set_xlabel("training steps (millions)", color=INK)
    axs[0].set_ylabel("furthest point reached (s from release)", color=INK)
    axs[0].set_title("(a) Start at the clip's first frame (1.0 s before release):\nhow far it gets before falling",
                     fontsize=9.5, color=INK, loc="left")
    axs[1].set_ylim(0, 100)
    axs[1].set_xlabel("training steps (millions)", color=INK)
    axs[1].set_ylabel("training episodes reaching the end (%)", color=INK)
    axs[1].set_title("(b) Practice runs (random start points)\nthat get to the end of the clip",
                     fontsize=9.5, color=INK, loc="left")
    axs[1].legend(fontsize=8, frameon=False, loc="lower right")
    phases(axs[2], axis="x")
    axs[2].axhline(0.0, color=INK2, lw=0.8, ls="--", zorder=1)
    axs[2].plot([-1, 0.34], [-1, 0.34], color=INK2, lw=0.8, zorder=1)
    axs[2].set_xlim(-1.0, 0.36)
    axs[2].set_ylim(-1.0, 0.42)
    axs[2].set_xlabel("start point (s from release)", color=INK)
    axs[2].set_ylabel("furthest point reached (s from release)", color=INK)
    axs[2].set_title("(c) The 25M-step policies started at different points:\nfilled = gets to the end (+0.34 s), open = falls",
                     fontsize=9.5, color=INK, loc="left")
    for ax in axs:
        style(ax)
    fig.suptitle("Dynamic Unitree G1: learned whole-body tracking controller (PPO, DeepMimic-style)",
                 fontsize=11, color=INK, x=0.06, y=1.04, ha="left")
    out = G1 / "track_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(out.relative_to(ROOT))


if __name__ == "__main__":
    main()
