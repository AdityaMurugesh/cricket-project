"""Before/after figure for the find-then-lose collapse: run11 vs run13b (resolve at release).

Good % (legal and in the 10.9-12.9 m zone) over training, one line per run,
x = deliveries bowled so far (run13b episodes are much shorter, so timesteps
would not compare like with like). Reads reward_terms.py summaries.

    python scripts/plot_collapse_fix.py
"""
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BUDGET = {"100": ("#2a78d6", "1.0x actuators"), "150": ("#eb6834", "1.5x actuators")}

PANELS = [
    ("run11", "throw_ppo_run11_", "run11: PPO as before"),
    ("run13b", "throw_ppo_run13b_noshape_", "run13b: resolve at release, no shaping"),
    ("run13b", "throw_ppo_run13b_toterm_", "run13b: resolve at release, timeout = failure"),
]


def final_good(results_dir, prefix):
    """How many final policies' deterministic deliveries are good, from the eval.json copies."""
    n = good = 0
    for f in sorted((results_dir / "eval").glob(f"{prefix}*.json")):
        n += 1
        good += bool(json.load(open(f))["deterministic"]["good"])
    return good, n


def main():
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True, facecolor=SURFACE)
    for ax, (sweep, prefix, title) in zip(axes, PANELS):
        d = ROOT / "results" / sweep
        for f in sorted((d / "reward_terms").glob(f"{prefix}*.csv")):
            rows = list(csv.DictReader(open(f)))
            scale = f.stem.split("_s")[-1][:3]
            x, total = [], 0
            for r in rows:
                total += int(r["episodes"])
                x.append(total / 1000)
            ax.plot(x, [float(r["good_pct"]) for r in rows], color=BUDGET[scale][0], lw=1.5,
                    alpha=0.85, drawstyle="steps-mid", solid_capstyle="round")
        good, n = final_good(d, prefix)
        ax.set_title(f"{title}\nfinal policy bowls a good length: {good} / {n} runs",
                     loc="left", color=INK, fontsize=10.5, linespacing=1.5)
        ax.set_facecolor(SURFACE)
        ax.set_ylim(0, 105)
        ax.set_xlabel("deliveries bowled in training (thousands)", color=INK_2)
        ax.grid(True, color=GRID, lw=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(AXIS)
    axes[0].set_ylabel("legal + good length (% of deliveries)", color=INK_2)
    handles = [plt.Line2D([], [], color=c, lw=2, label=l) for c, l in BUDGET.values()]
    axes[-1].legend(handles=handles, frameon=False, loc="center right", labelcolor=INK_2, fontsize=9)
    fig.tight_layout()
    out = ROOT / "results" / "run13b" / "collapse_fix.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
