"""Video of a tracking-controller rollout on the dynamic G1 (logs/<run>/final_rollout.npz from train_g1_track.py),
side by side with the reference motion it tracks (a translucent copy). Runs on the laptop (no torch: the rollout is
the recorded qpos). Side view | view from behind; 2x slow motion.

    python scripts/render_g1_track.py --run g1track_a025_s0 [--delivery T3aLf]
  -> results/kinematic/videos/<run>.mp4
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from envs.g1_track_env import G1TrackEnv  # noqa: E402
from scripts.render_kinematic import LOCO_PY, WRITE  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--rollout", default=None, help="default logs/<run>/final_rollout.npz")
    p.add_argument("--delivery", default="T3aLf")
    p.add_argument("--label", default=None)
    p.add_argument("--suffix", default="", help="appended to the video name")
    a = p.parse_args()
    z = np.load(a.rollout or ROOT / "logs" / a.run / "final_rollout.npz")
    Q, dt, t_start = z["qpos"], float(z["dt"]), float(z["t_start_rel"])
    k0 = int(z["k0"]) if "k0" in z.files else 0      # the reference grid step the rollout started on
    env = G1TrackEnv(delivery=a.delivery, rsi=False)
    m, d = env.m, mujoco.MjData(env.m)
    dref = mujoco.MjData(m)
    rend = mujoco.Renderer(m, 540, 640)
    tmp = Path(tempfile.mkdtemp(prefix="g1track_"))
    n_ref = env.n_grid
    for f, q in enumerate(Q):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        dref.qpos[:] = env.gq[min(k0 + f, n_ref - 1)]
        dref.qpos[1] += 1.0                       # the reference drawn 1 m to the left, for comparison
        mujoco.mj_kinematics(m, dref)
        imgs = []
        for az in (90, 180):
            cam = mujoco.MjvCamera()
            cam.lookat[:] = [q[0], q[1] + 0.5, 0.7]
            cam.distance, cam.azimuth, cam.elevation = 3.6, az, -8
            rend.update_scene(d, cam)
            scn = rend.scene
            # add the reference body's geoms, translucent
            n0 = scn.ngeom
            mujoco.mjv_addGeoms(m, dref, mujoco.MjvOption(), mujoco.MjvPerturb(), mujoco.mjtCatBit.mjCAT_DYNAMIC, scn)
            for g in range(n0, scn.ngeom):
                scn.geoms[g].rgba[:] = [0.35, 0.55, 0.95, 0.35]
            imgs.append(rend.render().copy())
        plt.imsave(tmp / f"f_{f:05d}.png", np.concatenate(imgs, axis=1))
    done = len(Q) - 1
    lines = [f"Unitree G1, WHOLE BODY PHYSICALLY SIMULATED, learned tracking controller ({a.label or a.run})",
             f"Solid: the controller. Blue, 1 m aside: the reference. Starts {abs(t_start):.2f} s "
             f"before release; {done} of {n_ref - 1 - k0} steps ({done * dt:.2f} s). 2x slow motion."]
    out = ROOT / "results" / "kinematic" / "videos" / f"{a.run}{a.suffix}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(LOCO_PY), "-c", WRITE, str(tmp), str(out), str(round(0.5 / dt, 3)), "|".join(lines)], check=True)


if __name__ == "__main__":
    main()
