"""Constrained RL (PPO-Lagrangian) against the reward-penalty baseline it was paired with, during training. One column
per pair (--pair <lagrangian run>:<baseline run>:<title>), three rows, binned by training steps:
  illegal balls (% of released, the nominal release), with the Lagrangian run's target
  speed of the good balls (legal, good length and line; km/h)
  lambda, the price the Lagrangian run's law currently has (logs/<run>/lagrange.csv)
Reads logs/<run>/episodes.csv (train_runup's EpisodeLogger) and logs/<run>/lagrange.csv (LagrangeCallback).

    python scripts/plot_lagrange.py --pair kin2_v3_lagr0p01_tolr12_h500_a270_T3a_0:kin2_v3_tolr12_h500_a270_T3a_0:"12 ms, target 1%" ...
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def binned(run, bin_steps):
    """Per bin of training steps: centre (M steps), illegal % of released, mean good-ball speed (km/h)."""
    f = ROOT / "logs" / run / "episodes.csv"
    if not f.exists():
        return None
    ts, rel, legal, good, kmh = [], [], [], [], []
    with open(f, newline="") as fh:
        for row in csv.DictReader(fh):
            ts.append(int(row["timesteps"]))
            rel.append(row["released"] == "1")
            legal.append(row["legal"] == "1")
            good.append(row["good"] == "1")
            kmh.append(float(row["speed_kmh"] or 0.0))
    ts, rel, legal, good, kmh = map(np.asarray, (ts, rel, legal, good, kmh))
    b = ts // bin_steps
    xs, ill, spd = [], [], []
    for k in np.unique(b):
        m = b == k
        r = m & rel
        xs.append((k + 0.5) * bin_steps / 1e6)
        ill.append(100.0 * np.mean(~legal[r]) if r.any() else np.nan)
        g = m & good
        spd.append(np.mean(kmh[g]) if g.sum() >= 20 else np.nan)
    return np.array(xs), np.array(ill), np.array(spd)


def lagrange(run):
    f = ROOT / "logs" / run / "lagrange.csv"
    if not f.exists():
        return None
    rows = list(csv.DictReader(open(f, newline="")))
    return (np.array([int(r["timesteps"]) for r in rows]) / 1e6, np.array([float(r["mean_cost"]) for r in rows]),
            np.array([float(r["lambda"]) for r in rows]))


def target_of(run):
    """The --lagrangian target from the run name (lagr0p01 -> 0.01)."""
    for part in run.split("_"):
        if part.startswith("lagr"):
            return float(part[4:].replace("p", "."))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs="+", required=True, help="<lagrangian run>:<baseline run>:<title>")
    ap.add_argument("--bin-steps", type=int, default=500_000)
    ap.add_argument("--out", default=str(ROOT / "results/kinematic/constrained/constrained_rl.png"))
    a = ap.parse_args()
    pairs = [p.split(":", 2) for p in a.pair]
    fig, axes = plt.subplots(3, len(pairs), figsize=(4.3 * len(pairs), 8.2), squeeze=False, sharex="col")
    fig.patch.set_facecolor(BG)
    for c, (run, base, title) in enumerate(pairs):
        got = {}
        for name, col, lab in ((base, BLUE, "reward penalty (as before)"), (run, ORANGE, "Lagrangian (law as a cost)")):
            d = binned(name, a.bin_steps)
            if d is None:
                print("no episodes.csv for", name)
                continue
            got[name] = d
            xs, ill, spd = d
            axes[0][c].plot(xs, ill, color=col, lw=2, label=lab)
            axes[1][c].plot(xs, spd, color=col, lw=2, label=lab)
        if len(got) == 2:      # both at the same training step: the shorter run's last bin
            x_end = min(got[base][0][-1], got[run][0][-1])
            for name, who in ((base, "baseline  "), (run, "Lagrangian")):
                xs, ill, spd = got[name]
                i = int(np.argmin(np.abs(xs - x_end)))
                print(f"{title} | {who} at {xs[i]:.2f}M steps: illegal {ill[i]:.1f}% of released, "
                      f"good-ball speed {spd[i]:.1f} km/h")
        tgt = target_of(run)
        if tgt is not None:
            axes[0][c].axhline(100 * tgt, color=INK2, lw=1.2, ls="--", label=f"target {100 * tgt:g}%")
        lg = lagrange(run)
        if lg is not None:
            axes[2][c].plot(lg[0], lg[2], color=ORANGE, lw=2)
            print(f"{title}: lambda at the end {lg[2][-1]:.3f}, max {lg[2].max():.3f}")
        axes[0][c].set_title(title, color=INK, fontsize=10.5, loc="left")
        axes[2][c].set_xlabel("training steps (millions)", color=INK, fontsize=9)
    for r, lab in enumerate(("illegal balls (% of released)", "good-ball speed (km/h)", "λ (price of an illegal ball)")):
        axes[r][0].set_ylabel(lab, color=INK, fontsize=9.5)
    for ax in axes.ravel():
        ax.set_facecolor(BG)
        ax.grid(color=GRID, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=8.5)
    for ax in axes[0]:
        ax.set_ylim(bottom=0)
    axes[0][0].legend(fontsize=8, frameon=True, facecolor=BG, edgecolor=GRID, loc="upper right")
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150, facecolor=BG)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
