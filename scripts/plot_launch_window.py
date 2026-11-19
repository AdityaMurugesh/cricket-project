"""Teaching figure: the launch window. Where the ball would pitch if it left the hand a few ms early or late,
for three bowlers (noise-free, start 1.0 s before the captured release), against the good-length band, with the
timing-error distributions of an elite (2 ms SD) and an ordinary (12 ms SD) bowler underneath.

  python scripts/plot_launch_window.py      -> results/kinematic/launch_window.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "results" / "kinematic" / "equifinality"
H = 0.002
BOWLERS = [
    ("kin_bowl_torque_b4096_h500_bctau0_a270_T3a_0", "fast, trained without timing noise", "#c0392b"),
    ("kin2_tolr2_tt0p75_bcclone_h500_a290_T3a_0", "trained for 2 ms timing (elite)", "#8e44ad"),
    ("kin2_tolr12_resfast_a290_T3a_1", "trained for 12 ms timing (ordinary)", "#2471a3"),
]


def curve(run):
    d = json.load(open(E / f"{run}_curves.json"))
    key = min((k for k in d if k.startswith("none_")), key=lambda k: abs(float(k.split("_")[1]) - 1.0))
    c = np.array(d[key], dtype=float)
    return c[:, 0] * H * 1000, c[:, 1], c[np.argmin(np.abs(c[:, 0])), 4] * 3.6


def main():
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 6.2), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax.axhspan(10.9, 12.9, color="#27ae60", alpha=0.18)
    ax.text(39, 12.8, "good length\n10.9-12.9 m", color="#1e8449", fontsize=9, va="top", ha="right")
    for run, label, col in BOWLERS:
        t, L, kmh = curve(run)
        ins = (L >= 10.9) & (L <= 12.9)
        ax.plot(t, L, color=col, lw=2.2, label=f"{kmh:.0f} km/h, {label}")
        if ins.any():
            ax.plot(t[ins], L[ins], color=col, lw=6, alpha=0.35, solid_capstyle="butt")
    ax.axvline(0, color="k", lw=0.8, ls=":")
    ax.set_ylim(4, 20)
    ax.set_ylabel("where the ball pitches (m)")
    ax.set_title("The launch window: release a few ms early or late, and the length moves", fontsize=11)
    ax.legend(frameon=True, framealpha=0.9, fontsize=8.5, loc="lower left")
    ax.grid(alpha=0.25)
    t = np.linspace(-40, 40, 400)
    for sd, col in ((2, "#8e44ad"), (12, "#2471a3")):
        ax2.fill_between(t, np.exp(-0.5 * (t / sd) ** 2) / sd, color=col, alpha=0.35, label=f"timing error, SD {sd} ms")
    ax2.set_yticks([])
    ax2.set_xlabel("release time relative to the bowler's intended release (ms)")
    ax2.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    out = ROOT / "results" / "kinematic" / "launch_window.png"
    fig.savefig(out, dpi=140)
    print("wrote", out)


if __name__ == "__main__":
    main()
