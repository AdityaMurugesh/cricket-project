"""CMA-ES alone against CMA-ES followed by RL, under the same noise (supervisor's question, 20 Nov meeting).

Every candidate bowls in the same simulator (the v3 model: air drag, hard elbow stop, arm handover 0.5 s) and
is scored the same way:
  plan  a CMA-ES optimum (scripts/kin_frontier.py record, <json>:<tau>): open loop, the same torques whatever
        happens
  ppo   a PPO network (checkpoints/*.zip): the behaviour clone of a plan before any RL (*_init.zip), or after
        PPO fine-tuning. It sees the arm's state every 20 ms, so it can react
Noise conditions (the same draws for every candidate: common random numbers, kin_equifinality's seeds):
  none    no motor noise: 21 starts, all on the 20 ms decision grid, so with the handover they are one delivery
  t<sd>   signal-dependent motor noise (each decision's command u -> u (1 + sd N(0, 1)), Harris & Wolpert),
          21 starts x --noise-seeds draws (t0.15 is kin_equifinality's "torque" condition)
On top of either, release-timing error N(0, SD): every delivery's landing curve gives its exact good rate at
each SD (p_good_continuous, as curves_at_sd.py); a ball that is not released counts 0.

    python scripts/compare_cmaes_rl.py --label a270_j4_t0.75 \
        --plan cmaes=results/kinematic/frontier/frontier_a270_h500_k20_j4_pc_drag_se.json:0.75 \
        --ppo rl=checkpoints/kin2_v3_cvr_j4_h500_a270_T3a_0.zip
  -> results/kinematic/cmaes_vs_rl/<label>.json (items merged into an existing file)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.kin_equifinality import delivery, p_good_continuous  # noqa: E402

OUT = ROOT / "results/kinematic/cmaes_vs_rl"
SDS_MS = (1, 2, 4, 8, 12)


class TimePlan:
    """A frontier record as a 'model': the planned torques at the decision's time relative to the captured
    release (piecewise linear between knots, flat outside them, as kin_frontier.interp_matrix)."""

    def __init__(self, rec, env):
        self.t = np.array(rec["knot_times_s"])
        self.k = np.array(rec["knots"])
        self.env = env

    def predict(self, obs, deterministic=True):
        tr = self.env.t - self.env.t_ref_release
        a = np.array([np.interp(tr, self.t, self.k[:, j]) for j in range(self.k.shape[1])])
        return np.clip(a, -1.0, 1.0), None


def load_record(spec):
    path, tau = spec.rsplit(":", 1)
    return next(r for r in json.load(open(ROOT / path)) if abs(r["tau"] - float(tau)) < 1e-9)


def env_kw_from_record(rec):
    kw = {"delivery": rec.get("delivery", "T3a"), "release_angle_deg": rec["angle"],
          "arm_handover_s": rec.get("handover_s"), "target_range": [10.9, 12.9]}
    for k, v in (rec.get("env2") or {}).items():
        kw[k] = v
    return kw


def make_env(kw, torque_sd):
    from envs.kinematic_bowl_env2 import KinematicBowlEnv2
    kw = dict(kw, rsi=False, tolerance_reward_sd_ms=0.0, release_timing_sd_ms=0.0, torque_noise_sd=torque_sd)
    for k in ("target_range", "target_sample"):
        if kw.get(k) is not None:
            kw[k] = tuple(kw[k])
    if kw.get("elbow_solref") is not None:
        kw["elbow_solref"] = list(kw["elbow_solref"])
    kw.pop("tolerance_target", None)
    return KinematicBowlEnv2(**kw)


def score(env, model, starts, seeds):
    H = env.m.opt.timestep
    ps = {sd: [] for sd in SDS_MS}
    speeds, nominal_good, released = [], 0, 0
    for before in starts:
        for s in seeds:
            r, c = delivery(env, model, float(before), 1000 * s + int(round(before * 1000)))
            if not r["released"] or c is None:
                for sd in SDS_MS:
                    ps[sd].append(0.0)
                continue
            released += 1
            speeds.append(r["speed_m_s"] * 3.6)
            nominal_good += bool(r["good"])
            c = np.array(c, dtype=float)
            off, x, y, good = c[:, 0] * H, c[:, 1], c[:, 2], c[:, 3] > 0.5
            inband = (env.target_min <= x) & (x <= env.target_max) & (np.abs(y) <= env.line_tol_m)
            ok = good | ~inband        # as curves_at_sd.py: a sample fails legality/window/plane only if in band
            for sd in SDS_MS:
                ps[sd].append(p_good_continuous(off, x, y, ok, env.target_min, env.target_max, env.line_tol_m,
                                                sd / 1000.0))
    n = len(starts) * len(seeds)
    return {"n": n, "released": released, "good_exact_release": nominal_good,
            "speed_kmh": float(np.mean(speeds)) if speeds else None,
            "speed_sd_kmh": float(np.std(speeds)) if speeds else None,
            "p_cont": {f"{sd:g}": float(np.mean(ps[sd])) for sd in SDS_MS}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--plan", nargs="*", default=[], help="name=<frontier json>:<tau>")
    p.add_argument("--ppo", nargs="*", default=[], help="name=<checkpoint zip>")
    p.add_argument("--env-from", default=None,
                   help="frontier record <json>:<tau> whose model to bowl in (default: the first --plan)")
    p.add_argument("--torque-sd", type=float, nargs="+", default=[0.05, 0.15],
                   help="motor-noise SDs to score under (condition t<sd>), besides none")
    p.add_argument("--noise-seeds", type=int, default=10)
    p.add_argument("--starts", type=int, default=21)
    a = p.parse_args()
    from stable_baselines3 import PPO
    items = {}
    for spec in a.plan:
        name, rs = spec.split("=", 1)
        items[name] = ("plan", rs)
    for spec in a.ppo:
        name, path = spec.split("=", 1)
        items[name] = ("ppo", path)
    ref = load_record(a.env_from or a.plan[0].split("=", 1)[1])
    kw = env_kw_from_record(ref)
    starts = np.linspace(0.8, 1.2, a.starts)
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"{a.label}.json"
    res = json.load(open(f)) if f.exists() else {}
    res["_env"] = {"kw": kw, "torque_sd": a.torque_sd, "noise_seeds": a.noise_seeds, "starts": a.starts}
    for name, (kind, spec) in items.items():
        out = {"kind": kind, "source": spec}
        conds = [("none", 0.0, [0])] + [(f"t{sd:g}", sd, range(1, a.noise_seeds + 1)) for sd in a.torque_sd]
        for cond, sd, seeds in conds:
            env = make_env(kw, sd)
            if kind == "plan":
                rec = load_record(spec)
                assert rec["angle"] == ref["angle"] and rec.get("handover_s") == ref.get("handover_s"), spec
                model = TimePlan(rec, env)
                out.update(plan_kmh=rec["speed_kmh"], plan_tau=rec["tau"], plan_sd_ms=rec.get("jitter_sd_ms"),
                           plan_torque_noise_sd=rec.get("torque_noise_sd", 0.0))
            else:
                model = PPO.load(ROOT / spec, device="cpu")
            out[cond] = score(env, model, starts if sd > 0 else starts[:1], list(seeds))
            s = out[cond]
            print(f"{a.label:22s} {name:10s} {cond:6s} {s['speed_kmh'] or 0:6.1f} km/h  released {s['released']}/{s['n']}"
                  f"  exact-release good {s['good_exact_release']}  " +
                  "  ".join(f"p{k}={v:.2f}" for k, v in s["p_cont"].items()), flush=True)
        res[name] = out
        json.dump(res, open(f, "w"), indent=1)
    print("wrote", f)


if __name__ == "__main__":
    main()
