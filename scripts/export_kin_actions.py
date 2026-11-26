"""Record a kinematic2 / G1 PPO policy's deterministic delivery as plain action lists (.json), on the HPC.

Torch can't load on the Windows laptop, but these envs are deterministic without noise, so the policy's delivery
is fully described by its actions: one list with the release on (the delivery that is scored) and one with the
release switched off, run on into the follow-through (what render_kinematic.py shows after the ball has gone).
Replay anywhere with  python scripts/render_kinematic.py --actions results/kinematic/actions/<run>.json

    python scripts/export_kin_actions.py --run kin2_g1_fast_h500_a270_T3aL_0 [--model-path ...] [--before 1.0]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.kin_equifinality import make_env2, run_episode  # noqa: E402


class Recording:
    def __init__(self, model):
        self.model, self.acts = model, []

    def predict(self, obs, deterministic=True):
        a, s = self.model.predict(obs, deterministic=True)
        self.acts.append(np.asarray(a, float).tolist())
        return a, s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", nargs="+", required=True)
    p.add_argument("--model-path", default=None, help="one run only: instead of checkpoints/<run>.zip")
    p.add_argument("--before", type=float, default=1.0)
    p.add_argument("--flight-s", type=float, default=0.8)
    p.add_argument("--out-dir", default=str(ROOT / "results/kinematic/actions"))
    a = p.parse_args()
    from stable_baselines3 import PPO
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for run in a.run:
        kw2 = json.load(open(ROOT / "checkpoints" / f"{run}.env2.json"))
        env = make_env2(dict(kw2), 0.0)
        model = PPO.load(a.model_path or ROOT / "checkpoints" / f"{run}.zip", device="cpu")
        on = Recording(model)
        out, rows, _, _, t0 = run_episode(env, on, a.before, 0)
        rec = {"run": run, "env2": kw2, "before_s": a.before, "outcome": {k: out.get(k) for k in
               ("released", "why", "speed_m_s", "land_x", "lateral_m", "extension_deg", "legal", "good")},
               "on": on.acts, "off": []}
        if out["released"]:
            t_rel = t0 + len(rows) * env.m.opt.timestep
            off = Recording(model)
            run_episode(env, off, a.before, 0, release=False, t_stop=t_rel + a.flight_s + env.dt)
            rec["off"] = off.acts
        f = out_dir / f"{run}.json"
        json.dump(rec, open(f, "w"))
        o = rec["outcome"]
        print(run, "released" if o["released"] else o["why"],
              f"{(o['speed_m_s'] or 0) * 3.6:.1f} km/h land {o['land_x']} good {o['good']}", "->", f, flush=True)


if __name__ == "__main__":
    main()
