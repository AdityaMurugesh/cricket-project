"""v3 model (air drag + hard elbow stop) against the old model, from the frontier runs of 20 Nov.

Each frontier record is optimised for its own timing SD and scored at it (interpolated good rate). For a required good
rate tau the "best" is the fastest feasible record in that file with p >= tau (records found for a higher tau count too).
  v3       results/kinematic/frontier/frontier_a<A>_h500_k20[_j<SD>]_pc_drag_se.json
  control  results/kinematic/frontier/v3_control/frontier_a<A>_h500_k20[_j<SD>]_pc.json  (old model, same starts)
  ablation drag only (_drag), elbow only (_se), at 270 deg, SD 2 and 12
  bodies   _drag_se_<T1a|T2b|T4b> at 270 deg, SD 2 and 12 (T3a = the v3 file)
The headline comparison uses envelopes: every optimum of a model re-scored at every SD (frontier_cross_sigma.py ->
cross_sigma_v3/v3_a<A>.json and control_a<A>.json), so an optimiser that got lucky at one SD does not distort it; the
old (18-19 Nov) handover envelope comes from skill_frontier.json.
Writes results/kinematic/frontier/v3_compare.png and v3_compare.json and prints the tables.

  python scripts/compare_v3.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "results" / "kinematic" / "frontier"
TAUS = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95]
SDS = [1, 2, 4, 8, 12]


def fname(angle, sd, suffix, sub=""):
    j = "" if sd == 12 else f"_j{sd}"
    return D / sub / f"frontier_a{angle}_h500_k20{j}_pc{suffix}.json"


def best(path, tau):
    if not path.exists():
        return None
    ok = [r for r in json.load(open(path))
          if r.get("feasible") and r.get("good") and (r.get("p_jitter_12ms") or 0.0) >= tau - 1e-9]
    if not ok:
        return None
    r = max(ok, key=lambda r: r["speed_kmh"])
    return round(r["speed_kmh"], 1)


def row(path):
    return [best(path, t) for t in TAUS]


def fmt(xs):
    return "  ".join(f"{x if x is not None else '-':>6}" for x in xs)


ETAUS = [0.5, 0.6, 0.75, 0.85, 0.9, 0.95]


def envelope(xs_file, sd, found_at=None):
    """For each tau in ETAUS, the fastest good delivery whose interpolated good rate at this SD is >= tau.
    found_at: only optima searched at these SDs (the control was searched at 1, 2 and 12 ms only)."""
    if not xs_file.exists():
        return [None] * len(ETAUS)
    rows = [r for r in json.load(open(xs_file)) if r.get("good")
            and (found_at is None or r.get("found_at_sd_ms", 12.0) in found_at)]
    out = []
    for t in ETAUS:
        ok = [r["speed_kmh"] for r in rows if r["p_cont"].get(str(sd), 0.0) >= t - 1e-9]
        out.append(round(max(ok), 1) if ok else None)
    return out


def main():
    X = D / "cross_sigma_v3"
    old = json.load(open(D / "skill_frontier.json"))
    env = {"taus": ETAUS}
    for a in (270, 290):
        for sd in SDS:
            env[f"{a}_{sd}"] = {"v3": envelope(X / f"v3_a{a}.json", sd),
                                "v3_same_searches": envelope(X / f"v3_a{a}.json", sd, (1.0, 2.0, 12.0)),
                                "control": envelope(X / f"control_a{a}.json", sd),
                                "old_18nov": [old["by_angle_handover"][str(a)][str(sd)][old["taus"].index(t)]
                                              if t in old["taus"] else None for t in ETAUS]}
    print("ENVELOPES (every optimum re-scored at every SD), tau " + "  ".join(f"{t:>5}" for t in ETAUS))
    for a in (270, 290):
        for sd in SDS:
            for k in ("v3", "v3_same_searches", "control", "old_18nov"):
                print(f"  {a} {sd:>2} ms {k:17s}" + fmt(env[f"{a}_{sd}"][k]))
    out = {"envelopes": env, "taus": TAUS, "v3": {}, "control": {}, "ablation": {}, "bodies": {}}
    for a in (270, 290):
        for sd in SDS:
            out["v3"][f"{a}_{sd}"] = row(fname(a, sd, "_drag_se"))
            out["control"][f"{a}_{sd}"] = row(fname(a, sd, "", "v3_control"))
    for sd in (2, 12):
        for k, suf in (("drag only", "_drag"), ("hard elbow only", "_se"), ("both (v3)", "_drag_se")):
            out["ablation"][f"{k}, {sd} ms"] = row(fname(270, sd, suf))
        out["ablation"][f"old model (control), {sd} ms"] = row(fname(270, sd, "", "v3_control"))
        for body in ("T3a", "T1a", "T2b", "T4b"):
            out["bodies"][f"{body}, {sd} ms"] = row(fname(270, sd, "_drag_se" + ("" if body == "T3a" else f"_{body}")))
    nh = D / "frontier_a270_k20_pc_drag_se.json"
    out["no_handover_270_12ms"] = row(nh)
    json.dump(out, open(D / "v3_compare.json", "w"), indent=1)

    print("tau:" + " " * 30 + fmt(TAUS))
    for a in (270, 290):
        for sd in SDS:
            print(f"{a} deg {sd:>2} ms  v3       " + " " * 12 + fmt(out["v3"][f"{a}_{sd}"]))
            print(f"{a} deg {sd:>2} ms  control  " + " " * 12 + fmt(out["control"][f"{a}_{sd}"]))
    print("ablation (270 deg, handover):")
    for k, v in out["ablation"].items():
        print(f"  {k:34s}" + fmt(v))
    print("bodies (270 deg, handover, v3):")
    for k, v in out["bodies"].items():
        print(f"  {k:34s}" + fmt(v))
    print("no handover, 270, 12 ms, v3:      " + fmt(out["no_handover_270_12ms"]))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    cmap = plt.get_cmap("viridis")
    for ax, a in zip(axes, (270, 290)):
        for k, sd in enumerate(SDS):
            col = cmap(k / (len(SDS) - 1))
            for key, ls, lab in (("v3_same_searches", "-", f"{sd} ms"), ("control", "--", None)):
                pts = [(t, y) for t, y in zip(ETAUS, env[f"{a}_{sd}"][key]) if y is not None]
                if pts:
                    ax.plot(*zip(*pts), ls, marker="o" if ls == "-" else None, ms=4, lw=2 if ls == "-" else 1.2,
                            color=col, label=lab)
        ax.set_title(f"release {a}°, arm handover 0.5 s", fontsize=10)
        ax.set_xlabel("required good rate at the bowler's timing SD")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("fastest good delivery (km/h)")
    axes[1].legend(title="timing SD", frameon=False, ncol=2, fontsize=8, loc="upper right")
    axes[0].text(0.02, 0.03, "solid: v3 model (air drag + hard elbow stop)\ndashed: old model; same starts, settings and searches",
                 transform=axes[0].transAxes, fontsize=8, va="bottom")
    fig.suptitle("Does the more realistic model change the pace vs timing-skill result?")
    fig.tight_layout()
    fig.savefig(D / "v3_compare.png", dpi=130)
    print("wrote", D / "v3_compare.png")


if __name__ == "__main__":
    main()
