"""Pace vs the bowler's timing skill: the fastest good delivery at each required good rate, for each
release-timing SD, from every frontier optimum re-scored at every SD (scripts/frontier_cross_sigma.py,
interpolated good rate). Writes results/kinematic/frontier/skill_frontier.png and skill_frontier.json.

  python scripts/plot_skill_frontier.py
"""
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "results" / "kinematic" / "frontier"
TAUS = [0.25, 0.5, 0.6, 0.75, 0.85, 0.9, 0.95]
SDS = ["1", "2", "4", "8", "12", "16"]


def envelope(rows, angle, sd):
    """For each tau, the fastest delivery whose interpolated good rate at this SD is at least tau."""
    out = []
    for t in TAUS:
        ok = [r for r in rows if r["angle"] == angle and r["p_cont"].get(sd, 0.0) >= t]
        best = max(ok, key=lambda r: r["speed_kmh"], default=None)
        out.append(None if best is None else round(best["speed_kmh"], 1))
    return out


def main():
    rows = []
    for f in sorted(glob.glob(str(D / "cross_sigma_c" / "*.json"))):
        rows += [r for r in json.load(open(f)) if r.get("good")]
    table = {a: {sd: envelope(rows, a, sd) for sd in SDS} for a in (270.0, 290.0)}
    json.dump({"taus": TAUS, "by_angle": {str(int(a)): v for a, v in table.items()}, "n_deliveries": len(rows)},
              open(D / "skill_frontier.json", "w"), indent=1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    cmap = plt.get_cmap("viridis")
    for ax, a in zip(axes, (270.0, 290.0)):
        for k, sd in enumerate(SDS):
            ys = table[a][sd]
            pts = [(t, y) for t, y in zip(TAUS, ys) if y is not None]
            if pts:
                ax.plot(*zip(*pts), "o-", color=cmap(k / (len(SDS) - 1)), label=f"{sd} ms", lw=2, ms=4)
        ax.set_title(f"release {int(a)}°")
        ax.set_xlabel("required good rate under timing error")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("fastest good delivery (km/h)")
    axes[1].legend(title="timing SD", frameon=False)
    fig.suptitle("Pace vs timing skill: the arm's best delivery for a bowler with a given release-timing SD")
    fig.tight_layout()
    fig.savefig(D / "skill_frontier.png", dpi=130)
    for a in table:
        print(f"release {int(a)}:  tau " + "  ".join(f"{t:>5}" for t in TAUS))
        for sd in SDS:
            print(f"   SD {sd:>2} ms      " + "  ".join(f"{y if y is not None else '-':>5}" for y in table[a][sd]))


if __name__ == "__main__":
    main()
