"""Why do run10 policies find the legal zone and then lose it?

Left: training-time good% (legal and in zone, stochastic episodes) over the
run, for every run that reached >=30 % and ended below 10 % ("lost"), coloured
by how its final policy fails, against the runs that kept it (grey).
Right: lost / kept / never-found per release angle.
Prints the supporting numbers: reward falls with good% in every lost run (so
this is the optimiser leaving a good policy, not the reward preferring a bad
one), and every lost run that saved checkpoints has a good one.

Reads logs/throw_ppo_run10_*/curve.json + ckpt_eval.json and
results/run10/run10_results.csv. Writes results/run10/run10_collapse.png.
"""
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
RESULTS = ROOT / "results" / "run10"

SURFACE, INK, INK_2, GRID = "#ffffff", "#0b0b0b", "#52514e", "#e4e3df"
SHORT, ILLEGAL, KEPT, LOST = "#2f6db3", "#a8464b", "#b9b8b3", "#3d3c39"


def fnum(v):
    return float(v) if v not in ("", None) else None


def load():
    runs = []
    for r in csv.DictReader(open(RESULTS / "run10_results.csv")):
        d = LOGS / f"throw_ppo_run10_{r['tag']}"
        curve = json.loads((d / "curve.json").read_text()) if (d / "curve.json").exists() else []
        if not curve:
            continue
        ck = d / "ckpt_eval.json"
        ck = json.loads(ck.read_text())["checkpoints"] if ck.exists() else None
        good = np.array([b["good_pct"] for b in curve])
        status = "never"
        if good.max() >= 30:
            status = "lost" if good[-1] < 10 else "kept" if good[-1] >= 30 else "partial"
        # how the final (stochastic) policy fails: mostly illegal, or legal but off the zone
        mode = "illegal" if (fnum(r["stoch_legal_pct"]) or 0) < 50 else "short"
        runs.append({
            "tag": r["tag"], "angle": float(r["release_angle_deg"]),
            "scale": float(r["actuator_scale"]), "status": status, "mode": mode,
            "steps": np.array([b["timesteps_end"] for b in curve]) / 1e6,
            "good": good, "reward": np.array([b["mean_reward"] for b in curve]),
            "ckpt_good": None if ck is None else [c["deterministic"]["good"] for c in ck],
        })
    return runs


def report(runs):
    lost = [r for r in runs if r["status"] == "lost"]
    kept = [r for r in runs if r["status"] == "kept"]
    fell = sum(r["reward"][r["good"].argmax()] > r["reward"][-1] for r in lost)
    corr = np.median([np.corrcoef(r["good"], r["reward"])[0, 1] for r in runs if r["good"].std() > 0])
    with_ck = [r for r in lost if r["ckpt_good"] is not None]
    rescued = sum(any(r["ckpt_good"]) for r in with_ck)
    modes = Counter((r["mode"], r["scale"]) for r in lost)
    print(f"{len(runs)} runs: {len(lost)} lost, {len(kept)} kept")
    print(f"mean reward fell from peak to final in {fell}/{len(lost)} lost runs; "
          f"median within-run corr(good%, reward) = {corr:.2f}")
    print(f"lost runs with checkpoints: {len(with_ck)}, of which {rescued} have a legal in-zone checkpoint")
    print("failure mode of lost runs (mode, budget):", dict(modes))


def plot(runs, out_png):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURFACE,
                                 gridspec_kw={"width_ratios": [1.3, 1]})
    for a in (ax, bx):
        a.set_facecolor(SURFACE)
        a.grid(True, color=GRID, lw=0.8)
        a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(INK_2)
        a.tick_params(colors=INK_2)

    for r in runs:
        if r["status"] == "kept":
            ax.plot(r["steps"], r["good"], color=KEPT, lw=1, alpha=0.7, zorder=1)
    for r in runs:
        if r["status"] == "lost":
            ax.plot(r["steps"], r["good"], color=ILLEGAL if r["mode"] == "illegal" else SHORT,
                    lw=1.6, alpha=0.9, zorder=2)
    ax.plot([], [], color=KEPT, lw=2, label=f"kept the zone ({sum(r['status'] == 'kept' for r in runs)} runs)")
    ax.plot([], [], color=SHORT, lw=2, label="lost: still legal, lands short")
    ax.plot([], [], color=ILLEGAL, lw=2, label="lost: now illegal (2: no release)")
    ax.set_xlabel("training timesteps (M)", color=INK)
    ax.set_ylabel("legal and in zone (% of training episodes)", color=INK)
    ax.set_ylim(-3, 103)
    ax.set_title("Runs that found the zone, then lost it", color=INK, loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, labelcolor=INK)

    angles = sorted({r["angle"] for r in runs})
    x = np.arange(len(angles))
    count = {s: [sum(r["angle"] == a and r["status"] == s for r in runs) for a in angles]
             for s in ("lost", "kept", "never", "partial")}
    bottom = np.zeros(len(angles))
    for s, colour, label in (("kept", KEPT, "kept"), ("lost", LOST, "found, then lost"),
                             ("partial", "#e0d6c4", "ended 10-30 %"), ("never", "#efeeea", "never found")):
        bx.bar(x, count[s], bottom=bottom, color=colour, label=label, width=0.75, edgecolor=SURFACE)
        bottom += count[s]
    bx.set_xticks(x, [f"{a:g}" for a in angles], fontsize=8)
    bx.set_xlabel("release angle (deg; 270 = vertical)", color=INK)
    bx.set_ylabel("runs", color=INK)
    bx.set_title("Losing it happens at the low angles", color=INK, loc="left", fontsize=12)
    bx.legend(frameon=False, fontsize=9, labelcolor=INK)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150, facecolor=SURFACE)
    print("wrote", out_png)


def main():
    runs = load()
    report(runs)
    plot(runs, RESULTS / "run10_collapse.png")


if __name__ == "__main__":
    main()
