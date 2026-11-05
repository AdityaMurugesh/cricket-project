"""Figure from scripts/equifinal_mechanism.py: why some deliveries tolerate timing error, and the real hand.

Left: where the ball would land against release timing. Right: the two terms that move the landing per
10 ms of later release (speed gained, elevation lost); the dot is their sum with the small height term.

    python scripts/plot_equifinal_mechanism.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
INK, INK2, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e4e3de", "#fcfcfb"
BLUE, ORANGE, GREEN, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
ZONE = (10.9, 12.9)


def short(r):
    if r["source"] == "measured":
        return f"real bowler {r['run']} ({r['marker']})"
    n = r["run"][16:].replace("_T3a_", " s")
    return ("tolerant " if "t12" in n else "brittle ") + n.replace("b4096_t12_", "").replace("b4096_", "b4096 ")


def main():
    res = json.load(open(ROOT / "results" / "kinematic" / "equifinal_mechanism.json"))
    by = {r["run"]: r for r in res}
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.5, 7), facecolor=SURFACE, gridspec_kw=dict(width_ratios=[1.05, 1]))

    # left: landing curves, a few representatives
    a.axhspan(*ZONE, color=GREEN, alpha=0.16, lw=0)
    a.text(-39, 11.9, "good length", va="center", fontsize=10, color=INK2)
    curves = [("kin_bowl_torque_b4096_t12_a250_T3a_0", GREEN, "-", "simulated, trained with timing noise (a250 s0): 45 km/h"),
              ("kin_bowl_torque_b4096_t12_a270_T3a_1", YELLOW, "-", "simulated, trained with timing noise (a270 s1): 54 km/h"),
              ("kin_bowl_torque_a270_T3a_0", BLUE, "-", "simulated, trained without noise (a270 s0): 67 km/h"),
              ("T3a", ORANGE, "--", "real hand marker, T3a: 67 km/h (hypothetical landing, no ball)"),
              ("T2b", ORANGE, ":", "real hand marker, T2b: 72 km/h (hypothetical landing, no ball)")]
    for key, c, ls, label in curves:
        r = by[key]
        marker = "o" if r["source"] == "measured" else None
        a.plot(r["curve"]["t_ms"], r["curve"]["length_m"], ls=ls, color=c, lw=2, marker=marker, ms=5, label=label,
               mec=SURFACE, mew=1)
    a.set_xlim(-40, 40)
    a.set_ylim(0, 28)
    a.set_xlabel("release timing (ms); 0 = actual release (real hand: where its landing crosses mid-zone)",
                 color=INK, fontsize=10)
    a.set_ylabel("where the ball would land (m beyond release)", color=INK, fontsize=11)
    a.set_title("Landing against release timing", color=INK, fontsize=13, loc="left")
    a.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1)

    # right: the two terms, every run computed; most tolerant at the top
    rows = sorted(res, key=lambda r: abs(r["slope_m_per_10ms"]), reverse=True)
    for i, r in enumerate(rows):
        b.barh(i, r["term_speed"], color=GREEN, height=0.62, edgecolor=SURFACE, lw=1,
               label="speed gained" if i == 0 else None)
        b.barh(i, r["term_elevation"], color=BLUE, height=0.62, edgecolor=SURFACE, lw=1,
               label="elevation lost" if i == 0 else None)
        b.plot(r["sum_terms"], i, "o", color=INK, ms=8, mec=SURFACE, mew=1.5, ls="none", zorder=3,
               label="net effect on landing" if i == 0 else None)
    b.set_yticks(range(len(rows)))
    b.set_yticklabels([f"{short(r)}, {r['speed_kmh']:.0f} km/h" for r in rows], fontsize=9, color=INK)
    b.axvline(0, color=INK2, lw=1)
    b.set_xlabel("landing moves, m per 10 ms later release", color=INK, fontsize=11)
    b.set_title("Why: speed gain against elevation loss", color=INK, fontsize=13, loc="left")
    b.legend(frameon=False, fontsize=10, labelcolor=INK, loc="upper center", bbox_to_anchor=(0.3, -0.13), ncol=3)

    for ax in (a, b):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(INK2)
        ax.tick_params(colors=INK2, labelsize=10)
    b.tick_params(axis="y", colors=INK)
    fig.tight_layout()
    out = ROOT / "results" / "kinematic" / "equifinal_mechanism.png"
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
