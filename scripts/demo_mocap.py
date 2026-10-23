"""Play a captured delivery on the humanoid, run-up included; the ball is
physically simulated.

The body is puppeteered kinematically from data/mocap/reference_motion_<D>.npz,
from standing still at the top of the run-up to the follow-through. Measured:
forward travel, the bob of each running step and the jump, and the bowling
arm over the delivery. Not measured: the legs (a running cycle locked to the
measured steps) and the arm before the delivery (carry pose) -- see
mocap/retarget.py. The ball rides the hand on BowlerEnv's weld and is let go
at the MEASURED release instant with the hand site's finite-difference
velocity, exactly as demo_bowler.py does for the scripted swing.

    python scripts/demo_mocap.py --delivery T3a            # headless, prints the throw
    python scripts/demo_mocap.py --delivery T3a --frames   # also saves rendered frames
    python scripts/demo_mocap.py --delivery T3a --view     # MuJoCo viewer

The captured markers are drawn as orange dots next to the humanoid (from
data/mocap/markers_<D>.npz, raw and unlabelled), so what is measured and
what is animated can be told apart by eye; --no-markers hides them.
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
MARKER_RGBA = np.array([1.0, 0.55, 0.0, 1.0], np.float32)
MARKER_RADIUS = 0.018


class MarkerPlayback:
    """The raw exported markers (data/mocap/markers_<D>.npz), on the same
    clock and in the same world frame as the reference motion. Nearest
    100 Hz frame; NaN rows are markers not seen in that frame."""

    def __init__(self, path):
        z = np.load(path)
        self.pos = z["pos"]
        self.dt_ref = float(z["timestep"])
        self.k_rel = int(z["release_index"])

    def at(self, t):
        k = int(round(t / self.dt_ref)) + self.k_rel
        return self.pos[k] if 0 <= k < len(self.pos) else None


def draw_markers(scene, pts):
    """Append one sphere per visible marker to an mjvScene."""
    if pts is None:
        return
    for p in pts:
        if np.isnan(p[0]) or scene.ngeom >= scene.maxgeom:
            continue
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([MARKER_RADIUS, 0, 0]), p.astype(np.float64),
                            np.eye(3).ravel(), MARKER_RGBA)
        scene.ngeom += 1


class MocapPlayback:
    """Smooth (cubic-spline) access to a 100 Hz reference motion at sim rate.

    t0 is the first frame (standing at the top of the run-up), t1 the end of
    the measured arm data after release; both in seconds relative to release.
    Nothing is played after t1, so every channel is splined up to t1 only:
    the end condition at t1 is then the same for all of them, and the same as
    it was before the run-up was added (it matters when t1 is a frame or two
    after release, as in T4b)."""

    def __init__(self, path, model):
        z = np.load(path)
        self.info = json.loads(str(z["info"]))
        self.events = json.loads(str(z["events"]))
        self.dt_ref = float(z["timestep"])
        self.k_rel = int(z["release_index"])
        q = z["qpos"]
        root = model.joint("root_free").qposadr[0]
        cols = {"x": q[:, root], "z": q[:, root + 2],
                "shoulder": np.degrees(q[:, model.joint("shoulder").qposadr[0]]),
                "elbow": np.degrees(q[:, model.joint("elbow").qposadr[0]])}
        for n in LEG_JOINT_NAMES:
            cols[n] = np.degrees(q[:, model.joint(n).qposadr[0]])
        k1 = self.events["arm_measured_to"]
        if any(np.isnan(v[:k1 + 1]).any() for v in cols.values()):
            raise ValueError(f"{path.name}: qpos has gaps before the end of the arm data; "
                             "re-run scripts/mocap_process.py")
        t = (np.arange(k1 + 1) - self.k_rel) * self.dt_ref
        self.t0, self.t1 = t[0], t[-1]
        self.splines = {k: CubicSpline(t, v[:k1 + 1]) for k, v in cols.items()}

    def pose(self, t):
        t = float(np.clip(t, self.t0, self.t1))
        s = self.splines
        legs = {n: float(s[n](t)) for n in LEG_JOINT_NAMES}
        return (float(s["x"](t)), 0.0, float(s["z"](t))), legs, \
            float(s["shoulder"](t)), float(np.clip(s["elbow"](t), 0, 160))


def play(env, mp, viewer=None, frame_times=(), renderer=None, cam=None, real_time=True,
         markers=None):
    env.reset()
    dt = env.dt
    frames = {}
    want = sorted(frame_times)

    def follow(c):
        # track the bowler along the run-up; hold the view once the ball is away
        if c is not None and not env.released:
            c.lookat[0] = env.data.qpos[env.root_qpos_adr]

    def grab(t):
        while want and t >= want[0] - 1e-9:
            if renderer is not None:
                follow(cam)
                renderer.update_scene(env.data, camera=cam)
                if markers is not None:
                    draw_markers(renderer.scene, markers.at(want[0]))
                frames[want[0]] = renderer.render().copy()
            want.pop(0)

    def sync(t):
        if viewer is not None:
            follow(viewer.cam)
            if markers is not None:
                with viewer.lock():
                    viewer.user_scn.ngeom = 0
                    draw_markers(viewer.user_scn, markers.at(t))
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
        sync(t)
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
        sync(t)
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
    ap.add_argument("--no-markers", action="store_true",
                    help="hide the captured marker dots drawn next to the humanoid")
    args = ap.parse_args()

    env = BowlerEnv(max_steps=1500)
    mp = MocapPlayback(DATA / f"reference_motion_{args.delivery}.npz", env.model)
    markers = None
    if not args.no_markers:
        markers = MarkerPlayback(DATA / f"markers_{args.delivery}.npz")
        print("  orange dots: the captured markers as exported, unlabelled and unfiltered")
    ev, ms = mp.events, lambda k: (k - mp.k_rel) * mp.dt_ref * 1000
    print(f"[{args.delivery}] run-up from {mp.t0 * 1000:.0f} ms (standing), take-off "
          f"{ms(ev['takeoff']):.0f} ms, top of jump {ms(ev['jump_peak']):.0f} ms, release 0 ms; "
          f"measured arm from {ms(ev['arm_measured_from']):.0f} to {mp.t1 * 1000:+.0f} ms")
    print(f"  legs: {mp.info.get('legs')}")

    renderer = cam = None
    frame_times = ()
    if args.frames:
        renderer = mujoco.Renderer(env.model, 360, 480)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [0.0, 0.0, 1.0]
        cam.distance, cam.azimuth, cam.elevation = 3.2, 90, -5
        jump = [ms(ev[k]) / 1000 for k in ("takeoff", "jump_peak") if ev[k] >= 0]
        frame_times = tuple(sorted({-2.0, -1.5, -1.0, *jump, -0.20, -0.10, 0.0, 0.30}))

    viewer = None
    if args.view:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(env.model, env.data)
        viewer.cam.lookat[:] = [0.0, 0, 0.9]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 6, 90, -12

    while True:
        info, frames = play(env, mp, viewer, frame_times, renderer, cam,
                            real_time=viewer is not None, markers=markers)
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
            cols = int(np.ceil(len(frames) / 2))
            fig, axes = plt.subplots(2, cols, figsize=(4 * cols, 6.4))
            for ax in axes.ravel():
                ax.set_axis_off()
            for ax, (tt, img) in zip(axes.ravel(), sorted(frames.items())):
                ax.imshow(img); ax.set_title(f"{tt * 1000:+.0f} ms", fontsize=9)
            fig.suptitle(f"{args.delivery} played on assets/bowler.xml (release at 0 ms)"
                         + ("" if markers is None else "; orange dots = captured markers"), fontsize=10)
            fig.tight_layout(h_pad=2.0)
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
