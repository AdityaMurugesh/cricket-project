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
from envs.ballistics import BALL_R, position
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
    p.add_argument("--run", default=None, help="e.g. kin_bowl_torque_a270_T3a_0")
    p.add_argument("--frontier", default=None, help="instead of a policy: a kin_frontier.py json; the arm plays that "
                                                     "record's optimised open-loop torques")
    p.add_argument("--tau", type=float, default=0.0, help="with --frontier: which record (its required good rate)")
    p.add_argument("--before", type=float, default=1.0, help="start this long before the captured release (s)")
    p.add_argument("--slow", type=int, default=4, help="slow-motion factor (25 fps video)")
    p.add_argument("--flight-s", type=float, default=0.8, help="how long to show after release")
    p.add_argument("--out", default=None, help="default results/kinematic/videos/<run>.mp4")
    p.add_argument("--target-m", type=float, default=None, help="any-length (target_sample) bowlers: band centre")
    p.add_argument("--model-path", default=None, help="instead of checkpoints/<run>.zip")
    p.add_argument("--label", default=None, help="what the arm is, for the caption (default: the run name)")
    p.add_argument("--delivery", default=None,
                   help="replay another captured body (e.g. T3aL, the legs cleaned by scripts/fix_legs.py)")
    p.add_argument("--robot", default=None, choices=["g1"], help="the Unitree G1 bowler (envs/g1_bowl_env.py)")
    p.add_argument("--captured-arm", action="store_true",
                   help="with --robot: the arm too replays the (retargeted) capture, to the end: no policy")
    p.add_argument("--angle", type=float, default=270.0, help="with --captured-arm: release angle")
    p.add_argument("--any", action="store_true", help="with --frontier: the record even if infeasible")
    a = p.parse_args()
    if a.captured_arm:
        from envs.g1_bowl_env import G1BowlEnv
        from envs.ballistics import CRICKET_DRAG_K
        env = G1BowlEnv(delivery=a.delivery or "T3aL", rsi=False, release_angle_deg=a.angle, arm_handover_s=-1.0,
                        drag_k=CRICKET_DRAG_K)
        model = type("Zero", (), {"predict": lambda self, o, deterministic=True: (np.zeros(7), None)})()
        m = {"delivery": a.delivery or "T3aL"}
        a.run = a.run or f"g1_captured_arm_a{int(a.angle)}"
        a.label = a.label or "the captured arm, retargeted (no policy)"
    elif a.frontier:                  # a trajectory-optimisation record: replay its knots open loop
        import json
        import scripts.kin_frontier as kf
        recs = [r for r in json.load(open(a.frontier)) if (r.get("feasible") or a.any) and abs(r["tau"] - a.tau) < 1e-6]
        if not recs:
            sys.exit(f"no feasible record with tau {a.tau} in {a.frontier}")
        rec = recs[0]
        kf._init(rec["angle"], (10.9, 12.9), rec.get("handover_s"), None, rec.get("env2") or {}, 12.0,
                 rec.get("max_extension_deg", 15.0), False, a.delivery or rec.get("delivery", "T3a"))
        kf.KNOTS = np.array(rec["knot_times_s"])
        kf._M = kf.interp_matrix(kf.decision_times(kf._env))
        env = kf._env
        model = kf.OpenLoop(env, np.clip(kf._M @ np.array(rec["knots"]).reshape(len(kf.KNOTS), -1), -1.0, 1.0))
        m = {"delivery": a.delivery or rec.get("delivery", "T3a")}
        a.run = a.run or f"{Path(a.frontier).stem}_tau{a.tau:g}"
    env2 = ROOT / "checkpoints" / f"{a.run}.env2.json"
    if a.frontier or a.captured_arm:
        pass
    elif env2.exists():               # kinematic2 runs: rebuild the env as trained, without training noise
        import json
        from scripts.kin_equifinality import make_env2
        kw2 = json.load(open(env2))
        if a.delivery:
            kw2["delivery"] = a.delivery
        env = make_env2(kw2, 0.0, a.target_m)
        m = {"delivery": kw2.get("delivery", "T3a")}
    else:
        m = RUN_RE.match(a.run)
        env = KinematicBowlEnv(delivery=m["delivery"], rsi=False, release_angle_deg=float(m["ang"]),
                               target_range=(10.9, 12.9), arm_control=m["ctl"],
                               arm_handover_s=int(m["ho"]) / 1000 if m["ho"] else None)
    if not (a.frontier or a.captured_arm):
        model = PPO.load(a.model_path or ROOT / "checkpoints" / f"{a.run}.zip", device="cpu")

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
    x0, zone = p_rel[0], (env.target_min, env.target_max)
    dk = getattr(env, "drag_k", 0.0)
    rend = mujoco.Renderer(env.m, H, W)
    tmp = Path(tempfile.mkdtemp(prefix="kinvid_"))
    land_x = out1["land_x"] + x0
    g1v = hasattr(env, "_grip_site")
    for f, i in enumerate(range(0, n, stride)):
        d.qpos[:] = qpos[i]
        mujoco.mj_kinematics(env.m, d)
        if i <= rel and hasattr(env, "_grip_site"):        # the G1's grip point on its hand
            ball = d.site_xpos[env._grip_site].copy()
        elif i <= rel:
            R = d.xmat[env._hand].reshape(3, 3)
            ball = d.xpos[env._hand] + R @ np.array([0.035, -0.09, 0.0])
        else:
            ball = np.array(position(p_rel, v_rel, dk, (i - rel) * h))
            if ball[0] >= land_x or ball[2] <= BALL_R:      # pitched: hold it where it landed
                ball = np.array([land_x, out1["lateral_m"], BALL_R])
        imgs = []
        for view in ("side", "pitch"):
            cam = mujoco.MjvCamera()
            if view == "side":
                cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.8 if g1v else 1.0]
                cam.distance, cam.azimuth, cam.elevation = (3.0 if g1v else 4.0), 90, -8
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
    g1 = hasattr(env, "_grip_site")
    body = ("Unitree G1 humanoid (its own body, masses, joint ranges); body motion = the capture retargeted onto it"
            if g1 else None) or (f"motion capture replay ({m['delivery'][:-1]}), legs cleaned (feet planted at the video's contacts)"
            if m["delivery"].endswith("L") else f"motion capture replay ({m['delivery']})")
    lines = [f"Body: {body}.  Bowling arm: {a.label or 'learned policy (' + a.run + ')'}.  "
             f"Ball: simulated{' with air drag' if dk > 0 else ''}.",
             f"{o['speed_m_s'] * 3.6:.0f} km/h, lands {o['land_x']:.1f} m, {abs(o['lateral_m']):.2f} m off line, "
             f"elbow extension {o['extension_deg']:.1f} deg ({'good' if o['good'] else 'not good'}).  "
             f"{a.slow}x slow motion.",
             f"Green = target zone ({zone[0]:.1f}-{zone[1]:.1f} m, +/-0.5 m line). Right view: ball drawn 2.5x size."]
    out = Path(a.out) if a.out else ROOT / "results" / "kinematic" / "videos" / f"{a.run}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(LOCO_PY), "-c", WRITE, str(tmp), str(out), "25", "|".join(lines)], check=True)


if __name__ == "__main__":
    main()
