"""Deterministic replays of a full-bowler (stage 3) policy from several start times.

Per start: the delivery (or why there was none), release height, how far the
bowling arm turned (unwrapped upper-arm angle, ThrowEnv convention), how well
the body tracked the mocap, and where the pelvis ended up vs the reference.
A training score can be exploration noise; this is what the policy does.

    python scripts/eval_runup.py --model-path checkpoints/runup_bowl_a250_T3a_0_ckpts/step_25000000_steps.zip \
        --release-angle 250 [--out results/stage3/a250_25M.json]
"""
import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import PPO

from envs.humanoid_bowler import ref_at
from envs.runup_bowl_env import RunupBowlEnv


def replay(env, model, i0):
    env.reset(seed=0)
    env.t = env._start_t = i0 / env.ref["rate"]
    env.d.qpos[:] = env.ref["qpos"][i0]
    env.d.qvel[:] = env.ref["qvel"][i0]
    mujoco.mj_forward(env.m, env.d)
    env.start_arm_tracking()
    obs = env._obs(env.t)
    pose = []
    while True:
        a, _ = model.predict(obs, deterministic=True)
        obs, r, te, tr, info = env.step(a)
        pose.append(info["terms"]["pose"])
        if te or tr:
            break
    o = dict(info["outcome"])
    q_ref = ref_at(env.ref, min(env.t, env.t_ref_release), env.m)[0]
    arm = np.array(env._arm_hist)
    return {"start_t": round(i0 / env.ref["rate"], 2), "end_t": round(env.t, 2),
            "released": o["released"], "why": o.get("why", ""),
            "speed_kmh": round(o["speed_m_s"] * 3.6, 1), "land_x": o["land_x"] and round(o["land_x"], 2),
            "legal": o["legal"], "good": o["good"],
            "extension_deg": o["extension_deg"] and round(o["extension_deg"], 1),
            "release_height_m": o.get("release_height_m") and round(o["release_height_m"], 2),
            "arm_turned_deg": round(float(arm[-1] - arm.min()), 0),
            "mean_pose": round(float(np.mean(pose)), 3),
            "pelvis_err_m": round(float(np.linalg.norm(env.d.qpos[:3] - q_ref[:3])), 2)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--delivery", default="T3a")
    p.add_argument("--release-angle", type=float, default=250.0)
    p.add_argument("--max-pelvis-drift", type=float, default=None, help="as in training (m)")
    p.add_argument("--starts", type=int, default=8, help="start times spread over the allowed range")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    env = RunupBowlEnv(delivery=a.delivery, rsi=False, release_angle_deg=a.release_angle,
                       max_pelvis_drift_m=a.max_pelvis_drift)
    model = PPO.load(a.model_path, device="cpu")
    rows = [replay(env, model, int(i)) for i in np.linspace(0, env.rsi_last_index, a.starts)]
    for r in rows:
        print(r)
    good = sum(r["good"] for r in rows)
    print(f"good {good}/{len(rows)}  released {sum(r['released'] for r in rows)}/{len(rows)}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"model_path": a.model_path, "release_angle": a.release_angle, "rows": rows},
                  open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
