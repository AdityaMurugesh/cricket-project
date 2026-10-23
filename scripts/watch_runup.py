"""Watch a run-up imitation policy in the MuJoCo viewer, with the mocap reference as a ghost.

The simulated bowler is drawn solid; the captured reference pose at the same
instant is drawn translucent on top of it, so tracking (and falling) is
visible at a glance. Starts from the first frame of the run-up, or from
--start-s into it (on the reference pose and velocity there, as in training).

    python scripts/watch_runup.py --model-path checkpoints/runup_ppo_T3a_0_ckpts/step_5000000_steps.zip
    python scripts/watch_runup.py --model-path ... --check     # no window: print how far it gets
"""
import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import PPO

from envs.humanoid_bowler import ref_at
from envs.runup_imitation_env import RunupImitationEnv

GHOST_RGBA = (0.35, 0.65, 1.0, 0.30)


def run(env, model, viewer=None, slow=2.0, start_s=0.0):
    obs, _ = env.reset(seed=0)
    if start_s > 0:
        i0 = int(start_s * env.ref["rate"])
        env.t = env._start_t = i0 / env.ref["rate"]
        env.d.qpos[:] = env.ref["qpos"][i0]
        env.d.qvel[:] = env.ref["qvel"][i0]
        mujoco.mj_forward(env.m, env.d)
        obs = env._obs(env.t)
    ghost = mujoco.MjData(env.m)
    opt, pert = mujoco.MjvOption(), mujoco.MjvPerturb()
    R, n, info = 0.0, 0, {}
    while True:
        a, _ = model.predict(obs, deterministic=True)
        obs, r, te, tr, info = env.step(a)
        R += r
        n += 1
        if viewer is not None:
            if not viewer.is_running():
                return None
            ghost.qpos[:] = ref_at(env.ref, env.t, env.m)[0]
            mujoco.mj_kinematics(env.m, ghost)
            with viewer.lock():
                viewer.user_scn.ngeom = 0
                mujoco.mjv_addGeoms(env.m, ghost, opt, pert, mujoco.mjtCatBit.mjCAT_DYNAMIC, viewer.user_scn)
                for i in range(viewer.user_scn.ngeom):
                    viewer.user_scn.geoms[i].rgba[:] = GHOST_RGBA
                viewer.cam.lookat[:] = [env.d.qpos[0], env.d.qpos[1], 0.9]
                viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                   f"t = {env.t:.2f} s of {env.t_end:.2f} (release)",
                                   f"tracking reward {r:.2f}  |  solid: simulated, blue ghost: mocap reference")])
            viewer.sync()
            time.sleep(env.dt * slow)
        if te or tr:
            break
    return {"steps": n, "t": env.t, "t_end": env.t_end, "return": R, "fell": info["fell"],
            "reached_release": info["reached_release"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--delivery", default="T3a")
    p.add_argument("--slow", type=float, default=2.0, help="slow-motion factor")
    p.add_argument("--check", action="store_true", help="no viewer: print the outcome")
    p.add_argument("--start-s", type=float, default=0.0, help="start this far into the run-up")
    a = p.parse_args()
    env = RunupImitationEnv(delivery=a.delivery, rsi=False)
    model = PPO.load(a.model_path)
    if a.check:
        print(run(env, model, start_s=a.start_s))
        return
    with mujoco.viewer.launch_passive(env.m, env.d) as viewer:
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 4.5, 100, -12
        while viewer.is_running():
            out = run(env, model, viewer, a.slow, a.start_s)
            if out is None:
                break
            print(out, flush=True)
            t_hold = time.time() + 1.5
            while time.time() < t_hold and viewer.is_running():
                viewer.sync()
                time.sleep(0.02)


if __name__ == "__main__":
    main()
