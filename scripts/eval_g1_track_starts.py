"""How late in the delivery must the dynamic G1 start for its tracking controller to get through to the end?

A trained controller (scripts/train_g1_track.py) is run deterministically from many points of the clip (the robot
placed on the reference there, as in training's random starts) to the clip's end. The answer separates the run-up
(fast running, then the bound) from the delivery stride, release and follow-through.

    python scripts/eval_g1_track_starts.py --run g1track_a050_lr1e4_s2 --action-scale 0.5 [--ckpt <zip>]
  -> results/kinematic/g1/track_starts_<run>.json (+ logs/<run>/rollout_k<k>.npz for render_g1_track.py --rollout)
Runs on the HPC (needs torch). A few seconds per start.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--ckpt", default=None, help="policy zip (default checkpoints/<run>.zip); its vecnorm is "
                                                "<zip stem>_vecnorm.pkl")
    p.add_argument("--delivery", default="T3aLf")
    p.add_argument("--action-scale", type=float, default=0.25)
    p.add_argument("--from", dest="t_from", type=float, default=-1.0)
    p.add_argument("--to", dest="t_to", type=float, default=0.35)
    p.add_argument("--every", type=int, default=3, help="start every this many 20 ms steps")
    a = p.parse_args()

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from envs.g1_track_env import G1TrackEnv
    from train_g1_track import evaluate
    torch.set_num_threads(1)

    zip_ = Path(a.ckpt).resolve() if a.ckpt else ROOT / "checkpoints" / f"{a.run}.zip"
    vn_path = zip_.with_name(zip_.stem + "_vecnorm.pkl")
    mk = lambda: G1TrackEnv(delivery=a.delivery, action_scale=a.action_scale, rsi=False, t_from=a.t_from,
                            t_to=a.t_to)
    vn = VecNormalize.load(str(vn_path), DummyVecEnv([mk]))
    vn.training, vn.norm_reward = False, False
    model = PPO.load(str(zip_), device="cpu")
    env = mk()
    rows, first_done = [], None
    for k0 in range(0, env.n_grid - 5, a.every):
        e = evaluate(model, vn, a.delivery, a.action_scale, a.t_from, a.t_to, record=True, k0=k0, env=env)
        q = e.pop("qpos"); dt = e.pop("dt"); t_start = e.pop("t_start_rel"); e.pop("k0")
        e["k0"] = k0
        rows.append(e)
        print(f"start k={k0:2d} ({e['start_t_rel']:+.2f} s): {'COMPLETED' if e['completed'] else 'fell'} at "
              f"{e['reached_t_rel']:+.2f} s ({e['steps']}/{e['of_steps']}), joint err {e['joint_err_deg']:.1f} deg, "
              f"pelvis err {e['root_err_m']:.3f} m", flush=True)
        if k0 == 0 or (e["completed"] and first_done is None):
            np.savez(ROOT / "logs" / a.run / f"rollout_k{k0}.npz", qpos=q, dt=dt, t_start_rel=t_start, k0=k0)
        if e["completed"] and first_done is None:
            first_done = k0
    # the latest start that does NOT complete: from any later start the controller always finishes
    fails = [r["k0"] for r in rows if not r["completed"]]
    always_from = None
    later = [r for r in rows if not fails or r["k0"] > max(fails)]
    if later:
        always_from = later[0]["start_t_rel"]
    out = {"run": a.run, "ckpt": str(zip_.relative_to(ROOT)), "action_scale": a.action_scale,
           "earliest_completed_start_t_rel": next((r["start_t_rel"] for r in rows if r["completed"]), None),
           "always_completes_from_t_rel": always_from, "starts": rows}
    dst = ROOT / "results" / "kinematic" / "g1" / f"track_starts_{a.run}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1))
    print("earliest completed start:", out["earliest_completed_start_t_rel"], "| always completes from:",
          always_from, "->", dst.relative_to(ROOT), flush=True)


if __name__ == "__main__":
    main()
