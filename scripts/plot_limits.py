"""The pace/tolerance frontier under each arm-limit profile (envs/arm_limits.py; supervisor, 20 Nov: "a human-like and a
real-robot set of actuator and joint limits"). One panel per release angle: the fastest legal, good-length delivery
CMA-ES finds (km/h) against the good rate it must keep under the timing skill's release error, for the skeleton's own
motors (250 N m, every result before 21 Nov) and the human, Unitree G1 and Unitree H1 limits. A hollow marker is the
best delivery found that does NOT meet that requirement (labelled with what it reached).

    python scripts/plot_limits.py            (reads results/kinematic/frontier/limits/*_lim<profile>.json)
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FR = ROOT / "results/kinematic/frontier"
INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
PROFILES = [("model", "skeleton model (250 N m)", INK2, "o"), ("human", "human-like", "#2a78d6", "o"),
            ("g1", "Unitree G1", "#eb6834", "s"), ("h1", "Unitree H1", "#1baf7a", "D")]


def records(angle, profile, sd):
    base = f"frontier_a{angle}_h500_k20_j{sd}_pc_drag_se"
    f = FR / f"{base}.json" if profile == "model" else FR / "limits" / f"{base}_lim{profile}.json"
    if not f.exists():
        return None
    return sorted(json.load(open(f)), key=lambda r: r["tau"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--angles", nargs="+", type=int, default=[270, 290])
    ap.add_argument("--sd", type=int, default=4, help="timing skill (ms) the frontiers were run at")
    ap.add_argument("--taus", nargs="+", type=float, default=[0.0, 0.5, 0.75])
    ap.add_argument("--out", default=str(ROOT / "results/kinematic/frontier/limits/limits_frontier.png"))
    a = ap.parse_args()
    fig, axes = plt.subplots(1, len(a.angles), figsize=(5.6 * len(a.angles), 4.3), squeeze=False, sharey=True)
    fig.patch.set_facecolor(BG)
    rows = []
    for ax, angle in zip(axes[0], a.angles):
        ax.set_facecolor(BG)
        for prof, label, col, mk in PROFILES:
            recs = records(angle, prof, a.sd)
            if recs is None:
                continue
            recs = [r for r in recs if any(abs(r["tau"] - t) < 1e-9 for t in a.taus)]
            xs = [100 * r["tau"] for r in recs]
            ys = [r["speed_kmh"] if r.get("released") else float("nan") for r in recs]
            ax.plot(xs, ys, color=col, lw=2, zorder=2)
            for r, x, y in zip(recs, xs, ys):
                ok = bool(r.get("feasible"))
                ax.plot([x], [y], marker=mk, ms=8, color=col, mfc=col if ok else BG, mec=col, mew=1.8, zorder=3)
                if not ok and r.get("released"):
                    p = 100 * (r.get("p_jitter_12ms") or 0.0)
                    why = "no good ball" if not r.get("good") else f"reaches {p:.0f}%"
                    ax.annotate(why, (x, y), textcoords="offset points", xytext=(7, -3), fontsize=7.5, color=INK2)
                rows.append((angle, prof, r["tau"], r.get("speed_kmh"), ok, r.get("p_jitter_12ms"), r.get("good")))
            ax.plot([], [], color=col, marker=mk, lw=2, ms=7, label=label)
        ax.set_title(f"release angle {angle}°", color=INK, fontsize=11, loc="left")
        ax.set_xlabel(f"required good balls under {a.sd} ms release error (%)", color=INK, fontsize=9.5)
        ax.set_xticks([100 * t for t in a.taus])
        ax.set_xticklabels([f"{100 * t:g}" + ("
(fastest)" if t == 0 else "") for t in a.taus])
        ax.grid(color=GRID, lw=0.8)
        ax.set_ylim(bottom=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=8.5)
    axes[0][0].set_ylabel("fastest delivery CMA-ES finds (km/h)", color=INK, fontsize=9.5)
    axes[0][0].legend(fontsize=8.5, frameon=True, facecolor=BG, edgecolor=GRID, loc="lower left")
    fig.text(0.01, 0.01, "Filled: meets the requirement (legal, good length and line, good often enough). "
             "Hollow: the best found that does not.", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150, facecolor=BG)
    print("wrote", a.out)
    for angle, prof, tau, kmh, ok, p, good in rows:
        print(f"a{angle} {prof:6s} tau {tau:.2f}: {kmh or 0:6.1f} km/h  feasible {ok!s:5s}  good {good!s:5s}  "
              f"p {100 * (p or 0):5.1f}%")


if __name__ == "__main__":
    main()
