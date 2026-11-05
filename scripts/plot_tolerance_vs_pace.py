"""Timing tolerance against pace, for the two-joint arm (run15) and the full-body kinematic bowler.

y: how far the landing moves per 10 ms of release timing, at the actual release (|slope|).
x: release speed (km/h), noise-free.
Two-joint arm: final policy of every run15 run that bowls legally
(results/run15/equifinality, results/run15/noise_robustness, condition "none").
Kinematic bowler: the torque finals, median over 21 starts (results/kinematic/equifinality);
torque a290 s0 is left out (it releases when the window opens, not on the arm).
Real bowler: raw hand/wrist markers, where the landing a ball would have crosses the middle of the
good-length zone (results/kinematic/equifinal_mechanism.json; there is no ball in the capture).

    python scripts/plot_tolerance_vs_pace.py
"""
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]   # categorical slots 1-6 (markers differ too)
INK, INK2, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e4e3de", "#fcfcfb"


def two_joint():
    d = ROOT / "results" / "run15"
    speed = {}
    for f in (d / "noise_robustness").glob("*.csv"):
        for r in csv.DictReader(open(f)):
            if r["condition"] == "none" and r["mean_speed_kmh"]:
                speed[r["run"]] = float(r["mean_speed_kmh"])
    pts = {"free": [], "noise": []}
    for f in (d / "equifinality").glob("*.csv"):
        rows = list(csv.DictReader(open(f)))
        last = rows[-1]                                  # the final model is the last row
        if last["legal"] != "True" or not last["slope_m_per_10ms"] or last["run"] not in speed:
            continue
        cond = re.search(r"run15_(none|timing|torque|both)_", last["run"]).group(1)
        pts["free" if cond == "none" else "noise"].append((speed[last["run"]], abs(float(last["slope_m_per_10ms"]))))
    return pts


def kinematic(pattern="kin_bowl_torque_a*_summary.json"):
    """Final models that release; noise-free a290 s0 is left out (window-edge releases)."""
    out = []
    for f in sorted((ROOT / "results" / "kinematic" / "equifinality").glob(pattern)):
        s = json.load(open(f))
        n = s["none"]
        if s["run"] == "kin_bowl_torque_a290_T3a_0" or not n["released"]:
            continue
        out.append((n["mean_speed_kmh"], n["median_abs_slope_m_per_10ms"], s["run"][16:].replace("_T3a_", " s")))
    return out


def main():
    tj, kin, kin_noise = two_joint(), kinematic(), kinematic("kin_bowl_torque_t12*_summary.json")
    kin += kinematic("kin_bowl_torque_b4096_a*_summary.json")
    kin_b4096 = kinematic("kin_bowl_torque_b4096_t12_a*_summary.json")
    human = [r for r in json.load(open(ROOT / "results" / "kinematic" / "equifinal_mechanism.json"))
             if r["source"] == "measured"]
    fig, ax = plt.subplots(figsize=(9, 6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    series = [(tj["free"], "two-joint arm, trained without noise", "o"),
              (tj["noise"], "two-joint arm, trained with noise", "s"),
              ([(x, y) for x, y, _ in kin], "full-body bowler, trained without noise", "D"),
              ([(x, y) for x, y, _ in kin_noise], "full-body bowler, trained with noise", "^"),
              ([(x, y) for x, y, _ in kin_b4096], "full-body bowler, trained with noise, PPO batch 4096", "v"),
              ([(r["speed_kmh"], abs(r["slope_m_per_10ms"])) for r in human],
               "real hand/wrist markers (hypothetical landing, no ball)", "*")]
    for (pts, label, mk), c in zip(series, COLORS):
        ax.plot([p[0] for p in pts], [p[1] for p in pts], mk, ms=17 if mk == "*" else 9, color=c, mec=SURFACE, mew=1,
                ls="none", label=f"{label} (n = {len(pts)})")
    for x, y, lab in kin:
        ax.annotate(lab, (x, y), xytext=(8, 6), textcoords="offset points", fontsize=10, color=INK)
    for r in human:
        ax.annotate(f"real {r['run']}", (r["speed_kmh"], abs(r["slope_m_per_10ms"])), xytext=(9, 3),
                    textcoords="offset points", fontsize=10, color=INK)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=10)
    ax.set_xlabel("release speed (km/h)", color=INK, fontsize=11)
    ax.set_ylabel("landing moves, m per 10 ms of release timing", color=INK, fontsize=11)
    ax.set_title("Faster deliveries tolerate less timing error", color=INK, fontsize=13, loc="left")
    ax.legend(frameon=False, fontsize=10, labelcolor=INK, loc="upper left")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    out = ROOT / "results" / "kinematic" / "tolerance_vs_pace.png"
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print("wrote", out, {k: len(v) for k, v in tj.items()}, len(kin))


if __name__ == "__main__":
    main()
