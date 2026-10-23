"""Plot run10's best trained delivery per (budget, release angle) against the scripted ceiling.

Lines: scripted constant-torque ceiling from frontier_scripted_fine.csv.
Dots: fastest legal, in-zone deterministic delivery any run10 policy reached in
that cell -- best over seeds, entropy settings, final policy and checkpoints.
Reads results/run10/*.csv only, so it runs locally without torch.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "run10"

# sequential red ramp, light -> dark with budget (adjacent pairs pass the CVD check)
BUDGET_COLORS = {0.75: "#d9777c", 1.0: "#a8464b", 1.5: "#6e1a1e"}
SURFACE, INK, INK_2, GRID = "#ffffff", "#0b0b0b", "#52514e", "#e4e3df"


def load_ceiling(path):
    out = defaultdict(dict)
    for r in csv.DictReader(open(path)):
        if r["best_speed_kmh"]:
            out[float(r["actuator_scale"])][float(r["release_angle_deg"])] = float(r["best_speed_kmh"])
    return out


def load_best(path):
    out = defaultdict(dict)
    for r in csv.DictReader(open(path)):
        cands = []
        if r["det_good"] == "True":
            cands.append(float(r["det_speed_kmh"]))
        if r["ckpt_best_speed_kmh"]:
            cands.append(float(r["ckpt_best_speed_kmh"]))
        if cands:
            scale, ang = float(r["actuator_scale"]), float(r["release_angle_deg"])
            out[scale][ang] = max(out[scale].get(ang, 0.0), *cands)
    return out


def label(scale):
    return f"{scale:g}×"


def plot(ceiling, best, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.4))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=10, length=0)

    for scale, color in BUDGET_COLORS.items():
        xs = sorted(ceiling[scale])
        ys = [ceiling[scale][x] for x in xs]
        ax.plot(xs, ys, color=color, linewidth=2, solid_capstyle="round",
                label=f"{label(scale)} budget", zorder=2)
        ax.annotate(label(scale), (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=11, fontweight="bold", color=INK)
        pts = sorted(best[scale].items())
        ax.scatter([a for a, _ in pts], [s for _, s in pts], s=64, color=color,
                   edgecolor=SURFACE, linewidth=2, zorder=3)

    ax.set_xlim(220, 290)
    ax.set_ylim(20, 60)
    ax.set_yticks(range(20, 61, 10))
    ax.set_xlabel("Release angle (°, 270 = arm vertical)", color=INK_2, fontsize=11)
    ax.set_ylabel("Release speed (km/h)", color=INK_2, fontsize=11)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.07), ncol=3, frameon=False,
              fontsize=11, labelcolor=INK_2, handlelength=1.6)
    ax.text(0.5, 1.03, "lines: scripted constant-torque ceiling    dots: best legal in-zone "
            "PPO delivery (any seed, final or checkpoint)",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=9, color=INK_2)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, facecolor=SURFACE)
    print(f"wrote {out_png}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier", default=str(RESULTS / "frontier_scripted_fine.csv"))
    parser.add_argument("--results", default=str(RESULTS / "run10_results.csv"))
    parser.add_argument("--out", default=str(RESULTS / "run10_best_vs_ceiling.png"))
    args = parser.parse_args()
    plot(load_ceiling(args.frontier), load_best(args.results), args.out)


if __name__ == "__main__":
    main()
