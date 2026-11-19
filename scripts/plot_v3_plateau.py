"""Teaching figure: an equifinal plateau found by the v3 optimiser. Where the ball pitches if it leaves the hand a few ms
early or late, for three v3 optima at 270 deg (arm handover 0.5 s): the 130 km/h fastest delivery (a ~2 ms window), and
two tolerant ones whose landing point stays in the good-length band over 25-30 ms of release timing because the
speed and angle errors cancel.

  python scripts/plot_v3_plateau.py      -> results/kinematic/frontier/v3_plateau.png (+ prints each window)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.kin_equifinality as ke  # noqa: E402
import scripts.kin_frontier as kf  # noqa: E402

D = ROOT / "results" / "kinematic" / "frontier"
H = 0.002
SHOW = [("frontier_a270_h500_k20_j2_pc_drag_se.json", 0.0, "#c0392b"),
        ("frontier_a270_h500_k20_j8_pc_drag_se.json", 0.5, "#8e44ad"),
        ("frontier_a270_h500_k20_j8_pc_drag_se.json", 0.9, "#2471a3")]


def curve(rec):
    kf._init(rec["angle"], (10.9, 12.9), rec.get("handover_s"), None, rec.get("env2") or {}, rec["jitter_sd_ms"],
             rec.get("max_extension_deg", 15.0), True, rec.get("delivery", "T3a"))
    kf.KNOTS = np.array(rec["knot_times_s"])
    kf._M = kf.interp_matrix(kf.decision_times(kf._env))
    table = np.clip(kf._M @ np.array(rec["knots"]).reshape(len(kf.KNOTS), -1), -1.0, 1.0)
    r, c = ke.delivery(kf._env, kf.OpenLoop(kf._env, table), kf.BEFORE_S, 0)
    return r, np.array(c, dtype=float)


def main():
    fig, ax = plt.subplots(figsize=(8, 4.3))
    ax.axhspan(10.9, 12.9, color="#27ae60", alpha=0.18)
    ax.text(38, 12.8, "good length\n10.9-12.9 m", color="#1e8449", fontsize=9, va="top", ha="right")
    for f, tau, col in SHOW:
        rec = next(r for r in json.load(open(D / f)) if r.get("feasible") and abs(r["tau"] - tau) < 1e-9)
        r, c = curve(rec)
        t, L = c[:, 0] * H * 1000, c[:, 1]
        ins = (L >= 10.9) & (L <= 12.9)
        kmh = r["speed_m_s"] * 3.6
        lab = f"{kmh:.0f} km/h: good {100 * r['p_jitter_cont']:.0f}% of the time at {rec['jitter_sd_ms']:g} ms timing SD"
        ax.plot(t, L, color=col, lw=2.2, label=lab)
        ax.plot(np.where(ins, t, np.nan), np.where(ins, L, np.nan), color=col, lw=6, alpha=0.35, solid_capstyle="butt")
        sp = c[ins, 4] * 3.6 if ins.any() else []
        print(f"{kmh:6.1f} km/h  window {ins.sum() * 2} ms of samples in band, "
              f"from {t[ins].min() if ins.any() else 0:+.0f} to {t[ins].max() if ins.any() else 0:+.0f} ms; "
              f"speed across it {min(sp):.0f}-{max(sp):.0f} km/h" if len(sp) else "")
    ax.axvline(0, color="k", lw=0.8, ls=":")
    ax.set_xlim(-40, 40)
    ax.set_ylim(4, 20)
    ax.set_xlabel("release time relative to the planned release (ms)")
    ax.set_ylabel("where the ball pitches (m)")
    ax.set_title("v3 optimiser, 270°: the fast ball has a ~2 ms window; the tolerant ones hold the length for 25-30 ms",
                 fontsize=10)
    ax.legend(frameon=True, framealpha=0.9, fontsize=8.5, loc="lower left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(D / "v3_plateau.png", dpi=140)
    print("wrote", D / "v3_plateau.png")


if __name__ == "__main__":
    main()
