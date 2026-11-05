"""Full-body bowler trained with 12 ms timing noise: PPO batch 64 against batch 4096, same seeds.

Reads <run>_summary.json from results/kinematic/noise_training/<N>M (5M checkpoints, no noise in the eval,
21 starts) and results/kinematic/equifinality (the final model). Prints the pairs and the stability counts.

    python scripts/plot_batch_noise.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results" / "kinematic"
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e4e3de", "#fcfcfb"
MILLIONS = [5, 10, 15, 20, 25, 30]
POLICIES = [(a, s) for a in (250, 270, 290) for s in (0, 1)]
GROUPS = [("batch 64", "kin_bowl_torque_t12", BLUE), ("batch 4096", "kin_bowl_torque_b4096_t12", ORANGE)]


def series(prefix, a, s):
    name = f"{prefix}_a{a}_T3a_{s}_summary.json"
    files = [R / "noise_training" / f"{m}M" / name for m in MILLIONS] + [R / "equifinality" / name]
    return [json.load(open(f))["none"] for f in files]


def main():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), facecolor=SURFACE, sharex=True)
    x = list(range(len(MILLIONS) + 1))
    late_ok = {}
    for label, prefix, c in GROUPS:
        late_ok[label] = 0
        for a, s in POLICIES:
            rows = series(prefix, a, s)
            good = [100 * r["good"] / r["deliveries"] for r in rows]
            jit = [r["good_with_timing_jitter_pct"] for r in rows]
            axes[0].plot(x, good, color=c, lw=2, marker="o", ms=5, mec=SURFACE, mew=1, alpha=0.85)
            axes[1].plot(x, jit, color=c, lw=2, marker="o", ms=5, mec=SURFACE, mew=1, alpha=0.85)
            late_ok[label] += sum(r["good"] >= 19 for r in rows[3:])     # 20M, 25M, 30M, final
    for ax, ylabel, title in ((axes[0], "good-length deliveries, no noise in the eval (%)", "Does it stay accurate?"),
                              (axes[1], "good under 12 ms release-timing jitter (%)", "Does it tolerate timing error?")):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(INK2)
        ax.tick_params(colors=INK2, labelsize=10)
        ax.set_xticks(x, [f"{m}M" for m in MILLIONS] + ["final"])
        ax.set_ylim(0, 104)
        ax.set_xlabel("training steps", color=INK, fontsize=11)
        ax.set_ylabel(ylabel, color=INK, fontsize=11)
        ax.set_title(title, color=INK, fontsize=13, loc="left")
    handles = [plt.Line2D([], [], color=c, lw=2, marker="o", ms=5, label=f"PPO {label} (6 runs: 3 release angles x 2 seeds)")
               for label, _, c in GROUPS]
    fig.legend(handles=handles, frameon=False, fontsize=10, labelcolor=INK, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out = R / "batch_noise.png"
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print("wrote", out)
    print("checkpoints at 20M, 25M, 30M, final with at least 19/21 good, of 24:", late_ok)
    print("policy    batch 64: good, jitter, km/h, |slope|    batch 4096: good, jitter, km/h, |slope|")
    for a, s in POLICIES:
        cells = []
        for _, prefix, _ in GROUPS:
            r = series(prefix, a, s)[-1]
            cells.append(f"{r['good']:2d}/21 {r['good_with_timing_jitter_pct']:4.0f}% {r['mean_speed_kmh']:3.0f} {r['median_abs_slope_m_per_10ms']:.2f}")
        print(f"a{a} s{s}   " + "    ".join(cells))


if __name__ == "__main__":
    main()
