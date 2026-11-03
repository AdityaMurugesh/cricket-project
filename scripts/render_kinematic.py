"""Video of one kinematic-bowler delivery: side view following the bowler | view down the pitch.

What is what, and the video says so on every frame:
  body   the motion capture (T3a skeleton fit), replayed exactly
  arm    the learned policy (deterministic), simulated
  ball   simulated: in the hand until the env releases it, then the same drag-free
         flight the env scores (RunupBowlEnv._release); the length x line zone is drawn
After release the arm carries on under the policy (the same replay with release off,
as in scripts/kin_equifinality.py); the body holds its last captured frame once the
capture ends (0.4 s after release).

Frames are rendered here (MuJoCo); the mp4 is written with OpenCV from loco-venv.

    python scripts/render_kinematic.py --run kin_bowl_torque_a270_T3a_0
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import PPO

from envs.kinematic_bowl_env import KinematicBowlEnv
from envs.runup_bowl_env import BALL_R, G
from scripts.kin_equifinality import RUN_RE, run_episode

ROOT = Path(__file__).resolve().parent.parent
LOCO_PY = ROOT / "loco-venv" / "Scripts" / "python.exe"
W, H = 640, 540
WRITE = r"""
import sys, glob, cv2
tmp, out, fps = sys.argv[1], sys.argv[2], float(sys.argv[3])
lines = sys.argv[4].split("|")
files = sorted(glob.glob(f"{tmp}/f_*.png"))
first = cv2.imread(files[0])
w = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (first.shape[1], first.shape[0]))
for i, f in enumerate(files):
    img = cv2.imread(f)
    for k, s in enumerate(lines):
        for col, th in (((0, 0, 0), 4), ((255, 255, 255), 1)):    # white on a black outline
            cv2.putText(img, s, (14, 28 + 26 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, th, cv2.LINE_AA)
    w.write(img)
w.release()
print("wrote", out, len(files), "frames")
"""


def add_geom(scn, kind, size, pos, rgba, mat=None):
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], kind, np.asarray(size, float), np.asarray(pos, float),
                        np.eye(3).flatten() if mat is None else mat, np.asarray(rgba, np.float32))
    scn.ngeom += 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="e.g. kin_bowl_torque_a270_T3a_0")
    p.add_argument("--before", type=float, default=1.0, help="start this long before the captured release (s)")
    p.add_argument("--slow", type=int, default=4, help="slow-motion factor (25 fps video)")
    p.add_argument("--flight-s", type=float, default=0.8, help="how long to show after release")
    p.add_argument("--out", default=None, help="default results/kinematic/videos/<run>.mp4")
    a = p.parse_args()
    m = RUN_RE.match(a.run)
    env = KinematicBowlEnv(delivery=m["delivery"], rsi=False, release_angle_deg=float(m["ang"]),
                           target_range=(10.9, 12.9), arm_control=m["ctl"])
    model = PPO.load(ROOT / "checkpoints" / f"{a.run}.zip", device="cpu")

    out1, rows, _, _, t0 = run_episode(env, model, a.before, 0)
    if not out1["released"]:
        sys.exit(f"no delivery from this start: {out1['why']}")
    h = env.m.opt.timestep
    rel = len(rows) - 1
    t_rel = t0 + (rel + 1) * h
    p_rel, v_rel, _ = rows[rel]
    # the same delivery, release off, to keep the arm moving after the ball has gone. Record qpos per step
    qpos = []
    track = env._track_arm

    def keep():
        track()
        qpos.append(env.d.qpos.copy())
    env._track_arm = keep
    try:
        run_episode(env, model, a.before, 0, release=False, t_stop=t_rel + a.flight_s + env.dt)
    finally:
        env._track_arm = track
    n = min(len(qpos), rel + 1 + int(a.flight_s / h))

    # ball: at the grip until release, then ballistic until it lands
    d = mujoco.MjData(env.m)
    stride = int(round(1.0 / (25 * a.slow) / h))           # physics steps per video frame
    x0, zone = p_rel[0], (10.9, 12.9)
    rend = mujoco.Renderer(env.m, H, W)
    tmp = Path(tempfile.mkdtemp(prefix="kinvid_"))
    land_t = (v_rel[2] + np.sqrt(v_rel[2] ** 2 + 2 * G * (p_rel[2] - BALL_R))) / G
    for f, i in enumerate(range(0, n, stride)):
        d.qpos[:] = qpos[i]
        mujoco.mj_kinematics(env.m, d)
        if i <= rel:
            R = d.xmat[env._hand].reshape(3, 3)
            ball = d.xpos[env._hand] + R @ np.array([0.035, -0.09, 0.0])
        else:
            tf = min((i - rel) * h, land_t)
            ball = p_rel + v_rel * tf + np.array([0, 0, -0.5 * G * tf ** 2])
        imgs = []
        for view in ("side", "pitch"):
            cam = mujoco.MjvCamera()
            if view == "side":
                cam.lookat[:] = [d.qpos[0], d.qpos[1], 1.0]
                cam.distance, cam.azimuth, cam.elevation = 4.0, 90, -8
            else:
                cam.lookat[:] = [x0 + 6.0, 0.0, 0.6]
                # from behind the bowler, a little to the off side, down the pitch (+x)
                cam.distance, cam.azimuth, cam.elevation = 11.0, 15, -14
            rend.update_scene(d, cam)
            scn = rend.scene
            # the good-length zone on the ground (length from the release point x line), and the ball
            add_geom(scn, mujoco.mjtGeom.mjGEOM_BOX, [(zone[1] - zone[0]) / 2, env.line_tol_m, 0.002],
                     [x0 + sum(zone) / 2, 0.0, 0.003], [0.11, 0.69, 0.48, 0.45])
            add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [BALL_R * (1.0 if view == "side" else 2.5)] * 3, ball,
                     [0.85, 0.1, 0.1, 1.0])
            imgs.append(rend.render().copy())
        plt.imsave(tmp / f"f_{f:05d}.png", np.concatenate(imgs, axis=1))
    o = out1
    lines = [f"Body: motion capture replay ({m['delivery']}).  Bowling arm: learned policy ({a.run[9:]}).  "
             "Ball: simulated.",
             f"{o['speed_m_s'] * 3.6:.0f} km/h, lands {o['land_x']:.1f} m, {abs(o['lateral_m']):.2f} m off line, "
             f"elbow extension {o['extension_deg']:.1f} deg ({'good' if o['good'] else 'not good'}).  "
             f"{a.slow}x slow motion.",
             "Green = good-length zone (10.9-12.9 m, +/-0.5 m line). Right view: ball drawn 2.5x size."]
    out = Path(a.out) if a.out else ROOT / "results" / "kinematic" / "videos" / f"{a.run}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(LOCO_PY), "-c", WRITE, str(tmp), str(out), "25", "|".join(lines)], check=True)


if __name__ == "__main__":
    main()
