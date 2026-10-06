"""Play a captured delivery on the humanoid; the ball is physically simulated.

The body is puppeteered kinematically from data/mocap/reference_motion_<D>.npz
(bowling-arm angles and forward travel measured; pelvis height and legs held
at BowlerEnv's delivery-stride pose -- see mocap/retarget.py). The ball rides
the hand on BowlerEnv's weld and is let go at the MEASURED release instant
with the hand site's finite-difference velocity, exactly as demo_bowler.py
does for the scripted swing.

    python scripts/demo_mocap.py --delivery T3a            # headless, prints the throw
    python scripts/demo_mocap.py --delivery T3a --frames   # also saves rendered frames
    python scripts/demo_mocap.py --delivery T3a --view     # MuJoCo viewer
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402

from envs.bowler_env import BowlerEnv  # noqa: E402
from envs.runup import LEG_JOINT_NAMES  # noqa: E402

DATA = ROOT / "data" / "mocap"


class MocapPlayback:
    """Smooth (cubic-spline) access to a 100 Hz reference motion at sim rate."""

    def __init__(self, path, model):
        z = np.load(path)
        self.info = json.loads(str(z["info"]))
        self.dt_ref = float(z["timestep"])
        self.k_rel = int(z["release_index"])
        q = z["qpos"]
        root = model.joint("root_free").qposadr[0]
        sh = model.joint("shoulder").qposadr[0]
        el = model.joint("elbow").qposadr[0]
        cols = {"x": q[:, root], "shoulder": np.degrees(q[:, sh]), "elbow": np.degrees(q[:, el])}
        ok = ~np.isnan(cols["x"]) & ~np.isnan(cols["shoulder"]) & ~np.isnan(cols["elbow"])
        # longest contiguous measured stretch that contains the release
        idx = np.where(ok)[0]
        s = e = self.k_rel
        if not ok[self.k_rel]:
            raise ValueError("reference motion has no arm data at the release frame")
        while s - 1 >= 0 and ok[s - 1]:
            s -= 1
        while e + 1 < len(ok) and ok[e + 1]:
            e += 1
        self.k0, self.k1 = s, e
        t = (np.arange(s, e + 1) - self.k_rel) * self.dt_ref
        self.t0, self.t1 = t[0], t[-1]            # seconds relative to release
        self.splines = {k: CubicSpline(t, v[s:e + 1]) for k, v in cols.items()}
        self.legs = {n: float(np.degrees(q[0, model.joint(n).qposadr[0]])) for n in LEG_JOINT_NAMES}
        self.root_z = float(q[0, root + 2])

    def pose(self, t):
        t = float(np.clip(t, self.t0, self.t1))
        return (float(self.splines["x"](t)), 0.0, self.root_z), self.legs, \
            float(self.splines["shoulder"](t)), float(np.clip(self.splines["elbow"](t), 0, 160))


def play(env, mp, viewer=None, frame_times=(), renderer=None, cam=None, real_time=True):
    env.reset()
    dt = env.dt
    frames = {}
    want = sorted(frame_times)

    def grab(t):
        while want and t >= want[0] - 1e-9:
            if renderer is not None:
                renderer.update_scene(env.data, camera=cam)
                frames[want[0]] = renderer.render().copy()
            want.pop(0)

    def sync():
        if viewer is not None:
            viewer.sync()
            if real_time:
                time.sleep(dt)

    t = mp.t0
    n_pre = int(round(-mp.t0 / dt))
    for i in range(n_pre + 1):
        t = mp.t0 + i * dt
        env._prev_hand_pos = env.data.site_xpos[env.hand_site_id].copy()
        env._write_pose(*mp.pose(t))
        grab(t)
        sync()
        if viewer is not None and not viewer.is_running():
            return None, frames
    env.release_ball()
    done, info, steps = False, {}, 0
    while not done:
        t += dt
        if t <= mp.t1 and env._frozen_body_qpos is not None:
            root_xyz, legs, sh, el = mp.pose(t)
            fq = env._frozen_body_qpos
            fq[env.root_qpos_adr] = root_xyz[0]
            fq[env.shoulder_qpos_adr] = np.radians(sh)
            fq[env.elbow_qpos_adr] = np.radians(el)
        obs, reward, done, info = env.flight_step()
        grab(t)
        sync()
        steps += 1
        if viewer is not None and not viewer.is_running():
            return None, frames
    info["reward"] = reward
    return info, frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delivery", default="T3a")
    ap.add_argument("--view", action="store_true", help="open the MuJoCo viewer (loops on Enter)")
    ap.add_argument("--frames", action="store_true", help="save rendered frames to data/mocap/plots/")
    args = ap.parse_args()

    env = BowlerEnv(max_steps=1500)
    mp = MocapPlayback(DATA / f"reference_motion_{args.delivery}.npz", env.model)
    print(f"[{args.delivery}] playing {mp.t0 * 1000:.0f} ms to release, measured arm data to "
          f"{mp.t1 * 1000:+.0f} ms; {mp.info.get('legs')}")

    renderer = cam = None
    frame_times = ()
    if args.frames:
        renderer = mujoco.Renderer(env.model, 360, 480)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [0.0, 0.0, 1.0]
        cam.distance, cam.azimuth, cam.elevation = 4.0, 90, -8
        frame_times = (-0.20, -0.10, -0.05, 0.0, 0.05, 0.30)

    viewer = None
    if args.view:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(env.model, env.data)
        viewer.cam.lookat[:] = [2.0, 0, 0.9]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 10, 90, -12

    while True:
        info, frames = play(env, mp, viewer, frame_times, renderer, cam, real_time=viewer is not None)
        if info is None:
            break
        sp = info["release_speed"]
        print(f"simulated ball release speed: {sp * 3.6:.1f} km/h")
        print(f"landing (x, y): {info['landing_pos']}  "
              f"(target zone x in [{env.target_min}, {env.target_max}])  reward {info['reward']:.2f}")
        if frames:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, len(frames), figsize=(3 * len(frames), 2.6))
            for ax, (tt, img) in zip(np.atleast_1d(axes), sorted(frames.items())):
                ax.imshow(img); ax.set_axis_off(); ax.set_title(f"{tt * 1000:+.0f} ms", fontsize=8)
            fig.suptitle(f"{args.delivery} played on assets/bowler.xml (release at 0 ms)", fontsize=9)
            fig.tight_layout()
            out = DATA / "plots" / f"demo_{args.delivery}.png"
            fig.savefig(out, dpi=90); plt.close(fig)
            print(f"frames -> {out}")
        if viewer is None or not viewer.is_running():
            break
        try:
            if input("Enter to replay, q to quit > ").strip().lower() == "q":
                break
        except EOFError:
            break
    if viewer is not None:
        viewer.close()


if __name__ == "__main__":
    main()
