"""The trajectory-optimised pace/tolerance frontier against every trained PPO bowler.

x: good rate under 12 ms release-timing jitter (p_jitter_12ms, exact from the noise-free landing curve).
y: release speed, km/h.
Lines: scripts/kin_frontier.py, the fastest legal good-length delivery the arm can make at each
required tolerance (one start, 1.0 s before the captured release; open loop; per release angle).
Points: PPO finals of the torque-controlled kinematic bowler that bowl at least 19/21 good
(results/kinematic/equifinality/*_summary.json, condition "none"; their tolerance and speed are means over
21 starts). Hollow = trained without timing noise.

    cricket/Scripts/python.exe scripts/plot_frontier.py
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
COLORS = {250: "#2a78d6", 270: "#eb6834", 290: "#1baf7a"}
INK, INK2, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e4e3de", "#fcfcfb"


FR = ROOT / "results" / "kinematic" / "frontier"


def envelope(recs):
    """Feasible points not beaten on both speed and tolerance by another (reruns pooled)."""
    recs = [r for r in recs if r["feasible"]]
    return sorted([r for r in recs if not any(q["speed_kmh"] > r["speed_kmh"] and q["p_jitter_12ms"] > r["p_jitter_12ms"]
                                              for q in recs)], key=lambda r: r["p_jitter_12ms"])


def frontier(suffix="_k20"):
    """Per angle: frontier_a<ang><suffix>.json pooled with any rerun of the same tag in a subfolder."""
    out = {}
    for ang in (250, 270, 290):
        recs = [r for f in FR.glob(f"**/frontier_a{ang}{suffix}.json") for r in json.load(open(f))]
        if envelope(recs):
            out[ang] = envelope(recs)
    return out


def rl_points():
    pts = []
    eq = ROOT / "results" / "kinematic" / "equifinality"
    for f in list(eq.glob("kin_bowl_torque_*_summary.json")) + list(eq.glob("kin2_*_summary.json")):
        s = json.load(open(f))
        n = s["none"]
        m = re.search(r"_a(\d+)_T3a", s["run"])
        if not m or n["good"] < 19:
            continue
        pts.append({"run": s["run"], "angle": int(m[1]), "noise": "_t12" in s["run"], "kin2": s["run"].startswith("kin2"),
                    "p": n["good_with_timing_jitter_pct"] / 100.0, "kmh": n["mean_speed_kmh"]})
    return pts


def main():
    F, P = frontier(), rl_points()
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2})
    fig, ax = plt.subplots(figsize=(7.5, 5.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.8)
    for ang, recs in sorted(F.items()):
        recs = sorted(recs, key=lambda r: r["p_jitter_12ms"])
        ax.plot([r["p_jitter_12ms"] * 100 for r in recs], [r["speed_kmh"] for r in recs], "-o", color=COLORS[ang],
                lw=2.2, ms=5, label=f"optimised arm, release {ang}°")
    for p in P:
        c = COLORS.get(p["angle"], INK2)
        if p["kin2"]:
            ax.scatter(p["p"] * 100, p["kmh"], s=90, marker="*", zorder=4, facecolor=c, edgecolor=INK, lw=0.7)
            continue
        ax.scatter(p["p"] * 100, p["kmh"], s=34, marker="s" if p["noise"] else "o", zorder=3,
                   facecolor=c if p["noise"] else "none", edgecolor=c, lw=1.3)
    ax.scatter([], [], marker="s", facecolor=INK2, edgecolor=INK2, label="PPO, trained with timing noise")
    ax.scatter([], [], marker="o", facecolor="none", edgecolor=INK2, label="PPO, trained without")
    ax.scatter([], [], marker="*", s=90, facecolor=INK2, edgecolor=INK, lw=0.7, label="PPO, exact tolerance reward (new)")
    ax.set_xlabel("good length under 12 ms release-timing jitter (%)")
    ax.set_ylabel("release speed (km/h)")
    ax.set_xlim(0, 100)
    ax.set_title("What the arm can do vs what PPO found (all legal, 10.9-12.9 m)", color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    out = FR / "frontier.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print("wrote", out)
    for ang, recs in sorted(F.items()):
        for r in recs:
            print(f"a{ang} tau {r['tau']:.2f}: {r['speed_kmh']:5.1f} km/h  p_jit {r['p_jitter_12ms']:.2f}  "
                  f"elev {r.get('elevation_deg') or 0:5.1f}  slope {r.get('slope_m_per_10ms') or 0:5.2f} m/10ms")


def budgets():
    """Release 270 deg: the frontier at 0.5x, 0.75x, 1x and 1.5x the arm's torque, and with the 0.5 s handover."""
    curves = {"0.5x torque": "_k20_q0p5", "0.75x torque": "_k20_q0p75", "1x torque": "_k20", "1.5x torque": "_k20_q1p5",
              "1x, arm handover 0.5 s": "_h500_k20"}
    cols = ["#9ec5f0", "#2a78d6", "#1f1f1e", "#eb6834", "#1baf7a"]
    fig, ax = plt.subplots(figsize=(7.5, 5.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.8)
    for (lab, suf), c in zip(curves.items(), cols):
        recs = frontier(suf).get(270)
        if recs:
            ax.plot([r["p_jitter_12ms"] * 100 for r in recs], [r["speed_kmh"] for r in recs], "-o", color=c, lw=2,
                    ms=4.5, label=lab, ls="--" if "handover" in lab else "-")
            for r in recs:
                print(f"{lab:24s} {r['speed_kmh']:5.1f} km/h  p_jit {r['p_jitter_12ms']:.2f}")
    ax.set_xlabel("good length under 12 ms release-timing jitter (%)")
    ax.set_ylabel("release speed (km/h)")
    ax.set_xlim(0, 100)
    ax.set_title("Actuator budget: release 270°, optimised arm", color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FR / "frontier_budget.png", dpi=160)
    print("wrote", FR / "frontier_budget.png")


if __name__ == "__main__":
    main()
    budgets()
