"""Aggregate run10's per-checkpoint eval.json files into the speed/accuracy/legality curve.

Reads logs/throw_ppo_run10_*/eval.json (copied back from the HPC) and the
scripted frontier CSV, writes results/run10/run10_results.csv, a markdown
table, and a two-panel figure. No torch needed, so it runs locally.
"""
import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TAG_RE = re.compile(r"a(?P<ang>[\d.]+)_s(?P<s100>\d+)_e(?P<ent>\d+)_(?P<seed>\d+)$")

# reference palette, categorical slots 1-3 (validated all-pairs, light surface)
BUDGET_COLORS = {0.75: "#2a78d6", 1.0: "#eb6834", 1.5: "#1baf7a"}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def ent_value(code):
    # "0003" -> 0.003, "001" -> 0.01
    return float("0." + code[1:]) if code.startswith("0") else float(code)


def load_evals(log_root):
    rows = []
    for path in sorted(Path(log_root).glob("throw_ppo_run10_*/eval.json")):
        tag = path.parent.name.replace("throw_ppo_run10_", "")
        m = TAG_RE.match(tag)
        if not m:
            continue
        e = json.loads(path.read_text())
        d, s = e["deterministic"], e["stochastic"]
        rows.append({
            "tag": tag,
            "release_angle_deg": float(m["ang"]),
            "actuator_scale": int(m["s100"]) / 100.0,
            "ent_coef": ent_value(m["ent"]),
            "seed": int(m["seed"]),
            "det_released": d["released"],
            "det_speed_kmh": d["speed_kmh"],
            "det_land_x": d["land_x"],
            "det_extension_deg": d["extension_deg"],
            "det_legal": d["legal"],
            "det_good": d["good"],
            "det_release_shoulder_deg": d["release_shoulder_deg"],
            "stoch_good_pct": s["good_pct"],
            "stoch_legal_pct": s["legal_pct"],
            "stoch_zone_pct": s["in_zone_pct"],
            "stoch_released_pct": s["released_pct"],
            "stoch_mean_speed_good_kmh": s["mean_speed_good_kmh"],
            "stoch_mean_land_x": s["mean_land_x"],
            "action_std": " ".join(f"{v:.2f}" for v in e["action_std"]),
        })
    return rows


def load_frontier(path):
    out = defaultdict(dict)
    if not Path(path).exists():
        return out
    for r in csv.DictReader(open(path)):
        v = r["best_speed_kmh"]
        out[float(r["actuator_scale"])][float(r["release_angle_deg"])] = float(v) if v else None
    return out


def fmt(v, spec="{:.1f}", none="--"):
    return none if v is None else spec.format(v)


def markdown(rows, frontier, ent):
    lines = [f"Main grid (ent_coef={ent}). Deterministic policy per seed: speed if the delivery "
             "was legal AND in the 6-8 m zone, else why not. good% = stochastic, 200 episodes.",
             "",
             "| budget | angle | scripted ceiling | seed 0 (det) | seed 1 (det) | good% s0 / s1 |",
             "|---|---|---|---|---|---|"]

    def cell(r):
        if r is None:
            return "not run"
        if not r["det_released"]:
            return "no release"
        if r["det_good"]:
            return f"**{r['det_speed_kmh']:.1f}**"
        why = "illegal" if not r["det_legal"] else f"lands {fmt(r['det_land_x'], '{:.1f}')} m"
        return f"{fmt(r['det_speed_kmh'])} ({why})"

    by = {(r["actuator_scale"], r["release_angle_deg"], r["seed"]): r
          for r in rows if abs(r["ent_coef"] - ent) < 1e-9}
    for scale in sorted({k[0] for k in by}):
        for ang in sorted({k[1] for k in by if k[0] == scale}):
            r0, r1 = by.get((scale, ang, 0)), by.get((scale, ang, 1))
            ceil = frontier.get(scale, {}).get(ang)
            g = " / ".join(fmt(r["stoch_good_pct"], "{:.0f}") if r else "--" for r in (r0, r1))
            lines.append(f"| {scale:.2f}x | {ang:.0f} | {fmt(ceil, '{:.1f}', 'unreachable')} | "
                         f"{cell(r0)} | {cell(r1)} | {g} |")
    return "\n".join(lines)


def headline(rows, frontier, ent):
    lines = ["| budget | best RL legal in-zone (det) | at angle | scripted ceiling | at angle |",
             "|---|---|---|---|---|"]
    for scale in sorted({r["actuator_scale"] for r in rows}):
        good = [r for r in rows if r["actuator_scale"] == scale and r["det_good"]
                and abs(r["ent_coef"] - ent) < 1e-9]
        best = max(good, key=lambda r: r["det_speed_kmh"], default=None)
        f = {a: v for a, v in frontier.get(scale, {}).items() if v is not None}
        fa = max(f, key=f.get) if f else None
        lines.append(f"| {scale:.2f}x | {fmt(best['det_speed_kmh'] if best else None)} km/h | "
                     f"{fmt(best['release_angle_deg'] if best else None, '{:.0f}')} | "
                     f"{fmt(f.get(fa) if fa else None)} km/h | {fmt(fa, '{:.1f}')} |")
    return "\n".join(lines)


def plot(rows, frontier, ent, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7.5), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2]})
    fig.patch.set_facecolor(SURFACE)
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(INK_2)
        ax.tick_params(colors=INK_2, labelsize=9)

    for scale, color in BUDGET_COLORS.items():
        f = frontier.get(scale, {})
        xs = sorted(f)
        # break the line where the zone is unreachable
        seg_x, seg_y = [], []
        for x in xs + [None]:
            y = f.get(x) if x is not None else None
            if y is None:
                if seg_x:
                    ax1.plot(seg_x, seg_y, color=color, linewidth=2, solid_capstyle="round")
                seg_x, seg_y = [], []
            else:
                seg_x.append(x)
                seg_y.append(y)
        reach = [(x, f[x]) for x in xs if f.get(x) is not None]
        if reach:
            lx, ly = reach[-1]
            ax1.annotate(f"{scale:g}x budget", (lx, ly), xytext=(6, 0), textcoords="offset points",
                         va="center", fontsize=9, color=INK)

        mine = [r for r in rows if r["actuator_scale"] == scale and abs(r["ent_coef"] - ent) < 1e-9]
        good = [r for r in mine if r["det_good"]]
        ax1.scatter([r["release_angle_deg"] for r in good], [r["det_speed_kmh"] for r in good],
                    s=64, color=color, edgecolor=SURFACE, linewidth=2, zorder=3,
                    label=f"{scale:g}x budget")
        by_ang = defaultdict(list)
        for r in mine:
            by_ang[r["release_angle_deg"]].append(r["stoch_good_pct"])
        angs = sorted(by_ang)
        ax2.plot(angs, [sum(by_ang[a]) / len(by_ang[a]) for a in angs], color=color,
                 linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE,
                 markeredgewidth=2, label=f"{scale:g}x budget")

    ax1.annotate("ceilings coincide here: the landing\nconstraint binds, not the torque",
                 (240, 30.1), xytext=(236, 36), fontsize=8.5, color=INK_2,
                 arrowprops={"arrowstyle": "-", "color": INK_2, "linewidth": 0.8})
    ax1.set_ylabel("Release speed, km/h", color=INK, fontsize=10)
    ax1.set_title("Fastest legal delivery landing in the 6-8 m zone, by release angle\n"
                  "lines: scripted constant-torque ceiling   dots: trained policy (deterministic, per seed)",
                  loc="left", fontsize=10.5, color=INK)
    ax1.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    ax2.set_ylabel("Legal & in zone, % of\nstochastic deliveries", color=INK, fontsize=10)
    ax2.set_xlabel("Release angle (shoulder, deg; 270 = arm vertical)", color=INK, fontsize=10)
    ax2.set_ylim(-3, 103)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, facecolor=SURFACE)
    print(f"wrote {out_png}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", default=str(ROOT / "logs"))
    parser.add_argument("--frontier", default=str(ROOT / "logs" / "frontier_scripted_fine.csv"))
    parser.add_argument("--out-dir", default=str(ROOT / "results" / "run10"))
    parser.add_argument("--ent", type=float, default=0.003, help="main-grid ent_coef")
    args = parser.parse_args()

    rows = load_evals(args.log_root)
    if not rows:
        sys.exit(f"no run10 eval.json files under {args.log_root}")
    frontier = load_frontier(args.frontier)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "run10_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["ent_coef"], r["actuator_scale"],
                                                r["release_angle_deg"], r["seed"])))
    md = ["# run10 results", "", f"{len(rows)} evaluated checkpoints.", "",
          "## Headline per budget", "", headline(rows, frontier, args.ent), "",
          "## Full grid", "", markdown(rows, frontier, args.ent), ""]
    ents = sorted({r["ent_coef"] for r in rows} - {args.ent})
    for e in ents:
        md += [f"## Control column, ent_coef={e} (seed 0 only)", "", markdown(rows, frontier, e), ""]
    (out / "run10_results.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    plot(rows, frontier, args.ent, out / "run10_frontier.png")


if __name__ == "__main__":
    main()
