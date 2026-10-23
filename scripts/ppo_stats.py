"""Read PPO's own training stats out of a run's job.log and line them up with what it bowled.

SB3 prints one table per rollout (action_std/*, train/approx_kl, ...). This
parses those tables and joins them with the run's reward-term buckets
(scripts/reward_terms.py summarize), to ask whether the find-then-lose
collapse tracks the exploration noise or the size of the policy updates.

    python scripts/ppo_stats.py logs/throw_ppo_run11_* --terms results/run11/reward_terms \
        --out results/run11/ppo_stats.png
"""
import argparse
import csv
import re
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROW_RE = re.compile(r"^\|\s+(\S+)\s+\|\s+(\S*)\s+\|$")
KEYS = ["total_timesteps", "elbow", "shoulder", "approx_kl", "clip_fraction", "entropy_loss",
        "explained_variance", "value_loss"]


def parse_log(path):
    """One dict per SB3 table: {total_timesteps, elbow, shoulder, approx_kl, ...}."""
    tables, cur = [], {}
    for line in open(path, errors="replace"):
        line = line.rstrip()
        if line.startswith("---"):
            if "total_timesteps" in cur:
                tables.append(cur)
            cur = {}
            continue
        m = ROW_RE.match(line)
        if m and m.group(1) in KEYS and m.group(2):
            try:
                cur[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
    return tables


def load_terms(path):
    rows = list(csv.DictReader(open(path)))
    return [{k: (float(v) if v not in ("", None) else None) for k, v in r.items()} for r in rows]


def at(tables, steps, key):
    """Mean of `key` over the tables that fall in the bucket ending at `steps`."""
    vals = [t[key] for t in tables if key in t and steps[0] < t["total_timesteps"] <= steps[1]]
    return st.mean(vals) if vals else None


def summarize(runs, terms_dir):
    """Per run: noise and update size at the good% peak vs the end of training."""
    out = []
    for d in runs:
        d = Path(d)
        log, terms = d / "job.log", terms_dir / f"{d.name}.csv"
        if not (log.exists() and terms.exists()):
            continue
        tables, b = parse_log(log), load_terms(terms)
        edges = [0] + [r["timesteps_end"] for r in b]
        peak = max(range(len(b)), key=lambda i: b[i]["good_pct"])
        rec = {"run": d.name, "peak_good_pct": b[peak]["good_pct"], "end_good_pct": b[-1]["good_pct"],
               "peak_M": b[peak]["timesteps_end"] / 1e6}
        for name, i in (("peak", peak), ("end", len(b) - 1)):
            for k in ("shoulder", "elbow", "approx_kl", "clip_fraction"):
                rec[f"{name}_{k}"] = at(tables, (edges[i], edges[i + 1]), k)
            rec[f"{name}_speed_kmh"] = b[i]["speed_kmh"]
        out.append(rec)
    return out


def plot(runs, terms_dir, out_png, picks):
    import matplotlib.pyplot as plt

    INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
    BLUE, ORANGE, RED, GREEN = "#2f6db3", "#d9a03f", "#a8464b", "#7aa96b"
    runs = [Path(r) for r in runs if Path(r).name in picks]
    fig, axes = plt.subplots(3, len(runs), figsize=(4.3 * len(runs), 8.5), facecolor="white",
                             sharex="col", squeeze=False)
    for j, d in enumerate(runs):
        tables, b = parse_log(d / "job.log"), load_terms(terms_dir / f"{d.name}.csv")
        x = [r["timesteps_end"] / 1e6 for r in b]
        tx = [t["total_timesteps"] / 1e6 for t in tables]
        a0, a1, a2 = axes[0][j], axes[1][j], axes[2][j]
        a0.plot(x, [r["good_pct"] for r in b], color=BLUE, lw=1.8, drawstyle="steps-mid")
        a0.set_ylim(0, 100)
        a0.set_title(picks[d.name], color=INK, fontsize=10.5, loc="left")
        a0s = a0.twinx()
        a0s.plot(x, [r["speed_kmh"] for r in b], color=INK_2, lw=1.2, ls="--", drawstyle="steps-mid")
        a1.plot(tx, [t.get("shoulder") for t in tables], color=ORANGE, lw=1.4, label="shoulder")
        a1.plot(tx, [t.get("elbow") for t in tables], color=RED, lw=1.4, label="elbow")
        a1.set_yscale("log")
        a2.plot(tx, [t.get("approx_kl") for t in tables], color=GREEN, lw=0.9, label="approx KL")
        a2.set_yscale("log")
        a2.set_xlabel("training timesteps (M)", color=INK)
        if j == 0:
            a0.set_ylabel("legal + in zone (%)", color=BLUE)
            a1.set_ylabel("action noise std (log)", color=INK)
            a2.set_ylabel("approx KL per update (log)", color=INK)
            a1.legend(frameon=False, fontsize=8.5)
        if j == len(runs) - 1:
            a0s.set_ylabel("release speed, km/h (dashed)", color=INK_2)
        for a in (a0, a1, a2):
            a.grid(True, color=GRID, lw=0.7)
            a.set_axisbelow(True)
            a.spines["top"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    print("wrote", out_png)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="run log dirs holding job.log")
    p.add_argument("--terms", required=True, help="reward_terms.py summarize output dir")
    p.add_argument("--out", default=None, help="png; omit to only print the table")
    p.add_argument("--csv", default=None, help="also write the per-run summary here")
    p.add_argument("--pick", nargs="*", default=[], metavar="RUN=TITLE",
                   help="runs to plot, as dir_name=title")
    a = p.parse_args()
    rows = summarize(a.runs, Path(a.terms))
    fmt = lambda v: "   -  " if v is None else f"{v:6.3f}"
    print(f"{'run':38s} good% peak->end  std shoulder peak->end  std elbow peak->end  KL peak->end")
    for r in rows:
        print(f"{r['run']:38s} {r['peak_good_pct']:5.1f} -> {r['end_good_pct']:5.1f}   "
              f"{fmt(r['peak_shoulder'])} -> {fmt(r['end_shoulder'])}   "
              f"{fmt(r['peak_elbow'])} -> {fmt(r['end_elbow'])}   "
              f"{fmt(r['peak_approx_kl'])} -> {fmt(r['end_approx_kl'])}")
    if a.csv and rows:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    if a.out:
        picks = dict(s.split("=", 1) for s in a.pick)
        plot(a.runs, Path(a.terms), a.out, picks)


if __name__ == "__main__":
    main()
