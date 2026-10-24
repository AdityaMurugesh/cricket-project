"""Watch a run-up imitation policy in the MuJoCo viewer, with the mocap reference as a ghost.

The simulated bowler is drawn solid; the captured reference pose at the same
instant is drawn translucent on top of it, so tracking (and falling) is
visible at a glance. Starts from the first frame of the run-up, or from
--start-s into it (on the reference pose and velocity there, as in training).

    python scripts/watch_runup.py --model-path checkpoints/runup_ppo_T3a_0_ckpts/step_5000000_steps.zip
    python scripts/watch_runup.py --model-path ... --check     # no window: print how far it gets
    python scripts/watch_runup.py --task bowl --model-path checkpoints/runup_bowl_a250_T3a_0.zip
    python scripts/watch_runup.py --task kinematic --model-path checkpoints/kin_bowl_a250_T3a_0.zip
      (the captured body, only the bowling arm is the policy's; --start-s defaults to 1 s before release)
      (stage 3: the ball is drawn in the hand, then its flight; the caption gives the delivery)
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
from envs.kinematic_bowl_env import KinematicBowlEnv
from envs.runup_bowl_env import BALL_R, G, RunupBowlEnv
from envs.runup_imitation_env import FIRST_START_INDEX, RunupImitationEnv

GHOST_RGBA = (0.35, 0.65, 1.0, 0.30)
BALL_RGBA = (0.8, 0.1, 0.1, 1.0)


def add_sphere(scn, pos, rgba, size=BALL_R):
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([size, 0, 0]),
                        np.asarray(pos, float), np.eye(3).ravel(), np.asarray(rgba, np.float32))
    scn.ngeom += 1


def flight_arc(p, v, n=40):
    """Drag-free path of the ball from release to the ground (as RunupBowlEnv scores it)."""
    t_fl = (v[2] + np.sqrt(max(v[2] ** 2 + 2 * G * (p[2] - BALL_R), 0.0))) / G
    ts = np.linspace(0, t_fl, n)
    return [p + v * t - np.array([0, 0, 0.5 * G * t * t]) for t in ts]


def caption(env, r):
    o = getattr(env, "_outcome", None)
    if o is None:
        return f"tracking reward {r:.2f}  |  solid: simulated, blue ghost: mocap reference"
    if not o["released"]:
        return f"no ball: {o['why']}"
    zone = f"{env.target_min:g}-{env.target_max:g} m"
    return (f"released {o['speed_m_s'] * 3.6:.1f} km/h, lands {o['land_x']:.2f} m (zone {zone}), "
            f"elbow extension {o['extension_deg']:+.1f} deg -> {'LEGAL' if o['legal'] else 'ILLEGAL'}"
            f"{'  GOOD' if o['good'] else ''}")


def run(env, model, viewer=None, slow=2.0, start_s=0.0):
    obs, _ = env.reset(seed=0)
    if isinstance(env, KinematicBowlEnv):
        if start_s > 0:
            obs, _ = env.reset(seed=0, options={"before_s": env.t_ref_release - start_s})
    elif start_s > 0:
        i0 = max(int(start_s * env.ref["rate"]), FIRST_START_INDEX)
        env.t = env._start_t = i0 / env.ref["rate"]
        env.d.qpos[:] = env.ref["qpos"][i0]
        env.d.qvel[:] = env.ref["qvel"][i0]
        mujoco.mj_forward(env.m, env.d)
        obs = env._obs(env.t)
    ghost = mujoco.MjData(env.m)
    opt, pert = mujoco.MjvOption(), mujoco.MjvPerturb()
    R, n, info = 0.0, 0, {}
    bowl = isinstance(env, RunupBowlEnv)
    arc = None
    if bowl:
        env.start_arm_tracking()   # the state may have been moved to start_s above
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
                if bowl:
                    o = env._outcome
                    if o is None:
                        add_sphere(viewer.user_scn, env._grip(), BALL_RGBA)
                    elif o["released"]:
                        arc = arc or flight_arc(env._grip(), env._grip_velocity())
                        for q in arc:
                            add_sphere(viewer.user_scn, q, BALL_RGBA, 0.6 * BALL_R)
                viewer.cam.lookat[:] = [env.d.qpos[0], env.d.qpos[1], 0.9]
                viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                   f"t = {env.t:.2f} s of {env.t_end:.2f} (release)",
                                   caption(env, r))])
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
    p.add_argument("--task", default="imitate", choices=["imitate", "bowl", "kinematic"])
    p.add_argument("--release-angle", type=float, default=250.0)
    p.add_argument("--slow", type=float, default=2.0, help="slow-motion factor")
    p.add_argument("--check", action="store_true", help="no viewer: print the outcome")
    p.add_argument("--start-s", type=float, default=0.0, help="start this far into the run-up")
    a = p.parse_args()
    if a.task == "kinematic":
        env = KinematicBowlEnv(delivery=a.delivery, rsi=False, release_angle_deg=a.release_angle)
    elif a.task == "bowl":
        env = RunupBowlEnv(delivery=a.delivery, rsi=False, release_angle_deg=a.release_angle)
    else:
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
