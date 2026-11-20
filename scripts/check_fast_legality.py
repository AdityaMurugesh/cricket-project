"""Re-check fast deliveries' legality with the ICC-style 'max' measure (most-flexed point after
arm-horizontal -> release), and report hyperextension and joint speeds."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import scripts.kin_equifinality as ke
import scripts.kin_frontier as kf
from envs.legality import elbow_extension_deg
D = ROOT / "results/kinematic/frontier"

def from_frontier(f, tau=0.0):
    rec = next(r for r in json.load(open(D / f)) if r.get("feasible") and abs(r["tau"] - tau) < 1e-9)
    kf._init(rec["angle"], (10.9, 12.9), rec.get("handover_s"), None, rec.get("env2") or {}, 12.0,
             rec.get("max_extension_deg", 15.0), False, rec.get("delivery", "T3a"))
    kf.KNOTS = np.array(rec["knot_times_s"]); kf._M = kf.interp_matrix(kf.decision_times(kf._env))
    tab = np.clip(kf._M @ np.array(rec["knots"]).reshape(len(kf.KNOTS), -1), -1, 1)
    return kf._env, kf.OpenLoop(kf._env, tab), kf.BEFORE_S

def from_rl(run):
    from stable_baselines3 import PPO
    kw = json.load(open(ROOT / "checkpoints" / f"{run}.env2.json"))
    env = ke.make_env2(kw, 0.0, None)
    return env, PPO.load(ROOT / "checkpoints" / f"{run}.zip", device="cpu"), kf.BEFORE_S

cases = [("optimiser, old model, 270 (the 113 km/h)", lambda: from_frontier("frontier_a270_k20.json")),
         ("optimiser, v3, handover (the 130 km/h)", lambda: from_frontier("frontier_a270_h500_k20_j2_pc_drag_se.json")),
         ("optimiser, old model, law test (129.6)", lambda: from_frontier("throwtest/frontier_a270_k20.json")),
         ("RL fast bowler, v3 (110 km/h)", lambda: from_rl("kin2_v3_nf_ls32_bcfast_h500_a270_T3a_0"))]
for name, mk in cases:
    env, model, before = mk()
    out, rows, arm, elbow, t0 = ke.run_episode(env, model, before, 0)
    rel = len(rows) - 1
    arm, elbow = np.asarray(arm), np.asarray(elbow)
    ep = elbow_extension_deg(arm, elbow, rel + 1, mode="endpoint")
    mx = elbow_extension_deg(arm, elbow, rel + 1, mode="max")
    h = env.m.opt.timestep
    s = np.sin(np.radians(arm[: rel + 2])); hz = [i for i in range(len(s) - 1) if s[i] * s[i + 1] <= 0][-1]
    seg = elbow[hz: rel + 2]
    w_arm = np.max(np.abs(np.diff(arm[hz - 50: rel + 2]))) / h
    w_elb = np.max(np.abs(np.diff(elbow[hz - 50: rel + 2]))) / h
    print(f"{name}: {out['speed_m_s']*3.6:.1f} km/h good={out['good']} | ext endpoint {ep:+.1f}  ICC-style(max) {mx:+.1f} deg"
          f" | elbow over the phase {seg.min():+.1f}..{seg.max():+.1f} deg (flexion +) | min elbow overall {elbow[:rel+2].min():+.1f}"
          f" | peak arm-circle speed {w_arm:.0f} deg/s, peak elbow speed {w_elb:.0f} deg/s, horizontal->release {(rel+1-hz)*h*1000:.0f} ms")
