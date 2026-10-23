"""Split each training episode's task reward into its terms and track them over training.

ThrowEnv._reward pays once per episode:
    legal and in zone:  hit (+1) + pace (speed_weight * release speed, m/s)
    legal, out of zone: miss (-distance to the zone)
    illegal / no ball:  miss (-distance) + illegal (-illegal_penalty)
The potential-based shaping is not in episodes.csv's reward and sums to ~-Phi(s0)
per episode (identical starts), so it cannot rank deliveries; it is left out.

Each episode is decomposed from its logged reward, so the terms add up to it
exactly, and pace is cross-checked against speed_weight * release_speed_m_s.

    # on the HPC (no matplotlib needed): one small CSV per run
    python scripts/reward_terms.py summarize logs/throw_ppo_run10_* --out results/run10/reward_terms
    # locally: plot
    python scripts/reward_terms.py plot results/run10/reward_terms
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEED_WEIGHT, ILLEGAL_PENALTY, MAX_EXT = 0.5, 2.0, 15.0
TERMS = ["hit", "pace", "miss", "illegal"]


def fnum(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else x


def decompose(r, zone):
    """Terms of one episode, plus its outcome class."""
    reward, ext, land = fnum(r["reward"]), fnum(r["elbow_extension_deg"]), fnum(r["landing_x"])
    legal = ext is not None and ext <= MAX_EXT
    in_zone = land is not None and zone[0] <= land <= zone[1]
    t = dict.fromkeys(TERMS, 0.0)
    if legal and in_zone:
        t["hit"], t["pace"], cls = 1.0, reward - 1.0, "good"
    elif legal:
        t["miss"], cls = reward, ("short" if land is not None and land < zone[0] else "long")
    else:
        t["illegal"], t["miss"] = -ILLEGAL_PENALTY, reward + ILLEGAL_PENALTY
        cls = "no_release" if land is None else "illegal_cls"
    return t, cls, reward


def summarize(run_dirs, out_dir, buckets, zone):
    out_dir.mkdir(parents=True, exist_ok=True)
    for d in run_dirs:
        d = Path(d)
        if not (d / "episodes.csv").exists():
            print("skip (no episodes.csv):", d)
            continue
        rows = list(csv.DictReader(open(d / "episodes.csv")))
        if not rows:
            continue
        steps_end = int(rows[-1]["timesteps"])
        acc = [dict(n=0, reward=0.0, speed=0.0, n_speed=0, land=0.0, n_land=0, ext=0.0, n_ext=0,
                    pace_err=0.0, **dict.fromkeys(TERMS, 0.0),
                    **dict.fromkeys(["good", "short", "long", "illegal_cls", "no_release"], 0))
               for _ in range(buckets)]
        for r in rows:
            b = acc[min(buckets - 1, int(int(r["timesteps"]) * buckets / (steps_end + 1)))]
            t, cls, reward = decompose(r, zone)
            b["n"] += 1
            b["reward"] += reward
            b[cls] += 1
            for k in TERMS:
                b[k] += t[k]
            sp, land, ext = fnum(r["release_speed_m_s"]), fnum(r["landing_x"]), fnum(r["elbow_extension_deg"])
            if sp is not None:
                b["speed"] += sp * 3.6
                b["n_speed"] += 1
                if cls == "good":
                    b["pace_err"] = max(b["pace_err"], abs(t["pace"] - SPEED_WEIGHT * sp))
            if land is not None:
                b["land"] += land
                b["n_land"] += 1
            if ext is not None:
                b["ext"] += ext
                b["n_ext"] += 1
        out = out_dir / f"{d.name}.csv"
        with open(out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["bucket", "timesteps_end", "episodes", "reward", *TERMS,
                        "good_pct", "short_pct", "long_pct", "illegal_pct", "no_release_pct",
                        "speed_kmh", "land_x", "extension_deg", "max_pace_check_err"])
            for i, b in enumerate(acc):
                n = max(b["n"], 1)
                w.writerow([i, int((i + 1) * (steps_end + 1) / buckets), b["n"],
                            *(round(b[k] / n, 4) for k in ["reward", *TERMS]),
                            *(round(100 * b[k] / n, 2) for k in ["good", "short", "long", "illegal_cls", "no_release"]),
                            round(b["speed"] / b["n_speed"], 2) if b["n_speed"] else "",
                            round(b["land"] / b["n_land"], 3) if b["n_land"] else "",
                            round(b["ext"] / b["n_ext"], 2) if b["n_ext"] else "",
                            round(b["pace_err"], 6)])
        print("wrote", out, f"({len(rows)} episodes)")


def plot(in_dir, out_png, runs):
    import matplotlib.pyplot as plt

    INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
    TERM_COLORS = {"hit": "#7aa96b", "pace": "#2f6db3", "miss": "#d9a03f", "illegal": "#a8464b"}
    CLS_COLORS = {"good_pct": "#2f6db3", "short_pct": "#d9a03f", "long_pct": "#8a6fb3",
                  "illegal_pct": "#a8464b", "no_release_pct": "#b9b8b3"}
    CLS_LABELS = {"good_pct": "legal, in zone", "short_pct": "legal, short", "long_pct": "legal, long",
                  "illegal_pct": "illegal", "no_release_pct": "no release"}

    def load(name):
        rows = list(csv.DictReader(open(in_dir / f"throw_ppo_run10_{name}.csv")))
        return {k: [fnum(r[k]) for r in rows] for k in rows[0]}

    fig, axes = plt.subplots(3, len(runs), figsize=(4.3 * len(runs), 9.5), facecolor="white",
                             sharex="col", gridspec_kw={"height_ratios": [1.3, 1, 1]})
    for j, (name, title) in enumerate(runs):
        s = load(name)
        x = [v / 1e6 for v in s["timesteps_end"]]
        ax_t, ax_c, ax_k = axes[0][j], axes[1][j], axes[2][j]
        # mean reward per episode, split into terms (positive terms stack up, negative down)
        pos, neg = [0.0] * len(x), [0.0] * len(x)
        for k in TERMS:
            v = s[k]
            base = pos if k in ("hit", "pace") else neg
            ax_t.fill_between(x, base, [b + vi for b, vi in zip(base, v)], color=TERM_COLORS[k],
                              alpha=0.85, lw=0, label=k, step="mid")
            for i in range(len(x)):
                base[i] += v[i]
        ax_t.plot(x, s["reward"], color=INK, lw=1.6, label="total", drawstyle="steps-mid")
        ax_t.axhline(0, color=INK_2, lw=0.8)
        ax_t.set_title(title, color=INK, fontsize=10.5, loc="left")
        # what the deliveries were
        bottom = [0.0] * len(x)
        for k, c in CLS_COLORS.items():
            ax_c.fill_between(x, bottom, [b + v for b, v in zip(bottom, s[k])], color=c, lw=0,
                              label=CLS_LABELS[k], step="mid")
            bottom = [b + v for b, v in zip(bottom, s[k])]
        ax_c.set_ylim(0, 100)
        # pace vs accuracy: release speed and where the ball lands
        ax_k.plot(x, s["speed_kmh"], color=TERM_COLORS["pace"], lw=1.6, drawstyle="steps-mid")
        ax_k.tick_params(axis="y", colors=TERM_COLORS["pace"])
        ax_l = ax_k.twinx()
        ax_l.axhspan(6.0, 8.0, color="#a8464b", alpha=0.12, lw=0)
        ax_l.plot(x, s["land_x"], color=INK, lw=1.4, drawstyle="steps-mid")
        ax_l.set_ylim(0, 12)
        ax_k.set_xlabel("training timesteps (M)", color=INK)
        if j == 0:
            ax_t.set_ylabel("mean task reward per episode", color=INK)
            ax_c.set_ylabel("deliveries (%)", color=INK)
            ax_k.set_ylabel("release speed, km/h", color=TERM_COLORS["pace"])
        if j == len(runs) - 1:
            ax_l.set_ylabel("mean landing (m), zone shaded", color=INK)
        for a in (ax_t, ax_c, ax_k):
            a.grid(True, color=GRID, lw=0.7)
            a.set_axisbelow(True)
            for sp in ("top",):
                a.spines[sp].set_visible(False)
    axes[0][0].legend(frameon=False, fontsize=8.5, loc="lower left", ncol=2)
    axes[1][0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    print("wrote", out_png)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summarize")
    s.add_argument("runs", nargs="+")
    s.add_argument("--out", default=str(ROOT / "results" / "run10" / "reward_terms"))
    s.add_argument("--buckets", type=int, default=40)
    s.add_argument("--target-range", type=float, nargs=2, default=[6.0, 8.0])
    q = sub.add_parser("plot")
    q.add_argument("in_dir")
    q.add_argument("--out", default=str(ROOT / "results" / "run10" / "run10_reward_terms.png"))
    a = p.parse_args()
    if a.cmd == "summarize":
        summarize(a.runs, Path(a.out), a.buckets, tuple(a.target_range))
    else:
        runs = [  # one per failure mode, then a run that kept the zone, for contrast
            ("a250_s075_e0003_1", "0.75x, 250 deg: lost, now lands short"),
            ("a240_s150_e0003_0", "1.5x, 240 deg: lost, now illegal"),
            ("a270_s150_e0003_0", "1.5x, 270 deg: lost late, now illegal"),
            ("a265_s100_e0003_1", "1.0x, 265 deg: kept the zone"),
        ]
        plot(Path(a.in_dir), Path(a.out), runs)


if __name__ == "__main__":
    main()
