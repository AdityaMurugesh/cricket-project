"""Speed vs accuracy, CMA-ES against RL, in the v3 model (air drag + hard elbow stop, arm handover 0.5 s).

Lines: the CMA-ES envelope for each timing skill (release-timing SD): for every accuracy level, the fastest
good optimum (270 or 290 deg) whose good rate at that SD is at least that level (every optimum re-scored at
every SD: results/kinematic/frontier/cross_sigma_v3). Filled markers: v3 RL bowlers trained for that SD,
scored at it (results/kinematic/equifinality/curves_at_sd.json). Hollow markers: the 110 km/h RL fast
bowler (trained with perfect timing) scored at each SD.

  python scripts/plot_speed_accuracy.py      -> results/kinematic/frontier/speed_vs_accuracy.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
XS = ROOT / "results/kinematic/frontier/cross_sigma_v3"
CAS = ROOT / "results/kinematic/equifinality/curves_at_sd.json"
OUT = ROOT / "results/kinematic/frontier/speed_vs_accuracy.png"

# timing skill -> (colour: validated categorical slots 1-4, label, RL bowler trained for it)
SKILLS = [(2, "#2a78d6", "2 ms (elite)", "kin2_v3_tolr2_tt0p75_bcclone_h500_a290_T3a_0"),
          (4, "#eb6834", "4 ms (very good)", "kin2_v3_tolr4_tt0p5_bcfast_h500_a270_T3a_0"),
          (8, "#1baf7a", "8 ms (good amateur)", "kin2_v3_tolr8_tt0p5_bcfast_h500_a270_T3a_0"),
          (12, "#eda100", "12 ms (ordinary)", "kin2_v3_tolr12_resfast_a290_T3a_1")]
FAST = "kin2_v3_nf_ls32_bcfast_h500_a270_T3a_0"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def envelope(sd, levels):
    rows = [r for f in ("v3_a270.json", "v3_a290.json") for r in json.load(open(XS / f)) if r.get("good")]
    return np.array([max([r["speed_kmh"] for r in rows if r["p_cont"].get(str(sd), 0) >= a - 1e-9] or [np.nan])
                     for a in levels])


def main():
    cas = json.load(open(CAS))
    levels = np.linspace(0.02, 1.0, 197)
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for sd, col, lab, rl in SKILLS:
        env = envelope(sd, levels)
        ax.step(levels * 100, env, where="post", color=col, lw=2, label=f"CMA-ES limit, {lab}")
        r = cas[rl]
        ax.plot(100 * r["p_cont"][f"{sd:g}"], r["speed_kmh"], "o", ms=10, color=col, mec="#fcfcfb", mew=2,
                zorder=5)
        f = cas[FAST]
        ax.plot(100 * f["p_cont"][f"{sd:g}"], f["speed_kmh"], "o", ms=9, mfc="#fcfcfb", mec=col, mew=2, zorder=5)
    ax.annotate("RL fast bowler (110 km/h, trained with perfect timing):\nthe same ball, scored at 2 / 4 / 8 / 12 ms",
                xy=(100 * cas[FAST]["p_cont"]["4"], cas[FAST]["speed_kmh"]), xytext=(30, 128),
                fontsize=8.5, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.annotate("filled dots: RL bowlers trained" + chr(10) + "for that timing skill", xy=(100 * cas[SKILLS[3][3]]["p_cont"]["12"], 56.3),
                xytext=(62, 38), fontsize=8.5, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.plot([], [], "o", ms=9, color=INK2, label="RL bowler trained for that timing")
    ax.plot([], [], "o", ms=9, mfc="#fcfcfb", mec=INK2, mew=2, label="RL 110 km/h bowler at that timing")
    ax.set_xlim(0, 104)
    ax.set_ylim(30, 142)
    ax.set_xlabel("accuracy: % of balls on a good length, with the bowler's timing error", color=INK)
    ax.set_ylabel("ball speed (km/h)", color=INK)
    ax.set_title("Speed vs accuracy: the CMA-ES limit and the RL bowlers (new model, v3)", color=INK,
                 fontsize=11.5, loc="left")
    ax.grid(color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2)
    ax.legend(loc="lower left", fontsize=8.5, frameon=True, facecolor="#fcfcfb", edgecolor=GRID)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, facecolor=fig.get_facecolor())
    print("wrote", OUT)
    for sd, _, _, rl in SKILLS:
        e = envelope(sd, np.array([0.5, 0.75, 0.9]))
        print(f"{sd:2d} ms  CMA-ES @50/75/90%: {e}   RL {cas[rl]['speed_kmh']:.1f} km/h @ {cas[rl]['p_cont'][f'{sd:g}']:.0%}"
              f"   fast RL @ {cas[FAST]['p_cont'][f'{sd:g}']:.0%}")


if __name__ == "__main__":
    main()
