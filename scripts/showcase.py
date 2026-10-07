"""Replay recorded run10 deliveries in the MuJoCo viewer, looping until the window is closed.

Pure MuJoCo (no torch), so it runs on the laptop: the action sequences were
recorded on the HPC by scripts/export_actions.py. ThrowEnv has no reset
noise, so replaying the actions reproduces the policy's delivery.
  python scripts/showcase.py            # watch (slow motion, loops)
  python scripts/showcase.py --check    # headless: confirm replays match the HPC
"""
import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from envs.throw_env import ThrowEnv

ROOT = Path(__file__).resolve().parent.parent
ORDER = [  # tells the story: one per budget, then a chuck, then a run that drifted
    "a265_s075_e0003_0_final",
    "a270_s100_e001_0_step_4000000_steps",
    "a275_s150_e0003_2_step_3000000_steps",
    "a240_s150_e0003_0_final",
    "a270_s150_e0003_0_step_2000000_steps",
    "a270_s150_e0003_0_final",
]


def load(name, replay_dir):
    z = np.load(Path(replay_dir) / f"{name}.npz")
    return {
        "name": name,
        "actions": z["actions"],
        "angle": float(z["release_angle_deg"]),
        "scale": float(z["actuator_scale"]),
        "frame_skip": int(z["frame_skip"]),
        "label": str(z["label"]),
        "recorded": json.loads(str(z["outcome"])),
    }


def replay(rec, viewer=None, display=None, slow=1.0, hold_s=1.5):
    """Simulate in this delivery's own env (its actuator budget); mirror the state
    onto the viewer's display model, which shares the geometry."""
    env = ThrowEnv(release_mode="fixed_angle", release_angle_deg=rec["angle"],
                   actuator_scale=rec["scale"], speed_weight=0.5)
    env.reset()
    dt = env.model.opt.timestep
    status = "swinging"
    info = {}
    done = False
    for i in range(10_000):
        action = rec["actions"][i // rec["frame_skip"]] if i // rec["frame_skip"] < len(rec["actions"]) \
            else np.zeros(2)
        _, _, done, info = env.step(action)
        if info["released"] and status == "swinging":
            ext = info["elbow_extension_deg"]
            status = (f"released {info['release_speed'] * 3.6:.1f} km/h at shoulder "
                      f"{info['shoulder_at_release_deg']:.0f} deg | elbow extension "
                      f"{ext:+.1f} deg -> {'LEGAL' if info['legal'] else 'ILLEGAL (>15)'}")
        if viewer is not None:
            if not viewer.is_running():
                return None
            mirror(env, display)
            overlay(viewer, rec, status, info)
            viewer.sync()
            time.sleep(dt * slow)
        if done:
            break
    if viewer is not None:
        land = info.get("landing_pos")
        if land:
            zone = env.target_min <= land[0] <= env.target_max
            status += f" | lands {land[0]:.2f} m {'IN ZONE' if zone else 'MISS'}"
        t_end = time.time() + hold_s
        while time.time() < t_end and viewer.is_running():
            overlay(viewer, rec, status, info)
            viewer.sync()
            time.sleep(0.02)
    return info


def mirror(env, display):
    display.data.qpos[:] = env.data.qpos
    display.data.qvel[:] = env.data.qvel
    mujoco.mj_forward(display.model, display.data)


def overlay(viewer, rec, status, info):
    viewer.set_texts([
        (mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
         rec["label"], f"actuator budget {rec['scale']:g}x | release fixed at {rec['angle']:.0f} deg"),
        (mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
         status, "target zone: red box, 6-8 m | close the window to stop"),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-dir", default=str(ROOT / "results" / "replays"))
    parser.add_argument("--only", nargs="*", default=None, help="names (without .npz) to play")
    parser.add_argument("--slow", type=float, default=4.0, help="slow-motion factor (1 = real time)")
    parser.add_argument("--check", action="store_true", help="headless: compare with the HPC outcome")
    args = parser.parse_args()

    names = args.only or [n for n in ORDER if (Path(args.replay_dir) / f"{n}.npz").exists()]
    recs = [load(n, args.replay_dir) for n in names]

    if args.check:
        for r in recs:
            info = replay(r)
            land = info.get("landing_pos")
            got = (info["release_speed"] * 3.6 if info.get("release_speed") else None,
                   land[0] if land else None, info.get("legal"))
            want = r["recorded"]
            print(f"{r['name']:40s} laptop {got[0]:.2f} km/h land {got[1]:.3f} legal={got[2]} | "
                  f"HPC {want['speed_kmh']:.2f} km/h land {want['land_x']:.3f} legal={want['legal']}")
        return

    import mujoco.viewer
    display = ThrowEnv()
    viewer = mujoco.viewer.launch_passive(display.model, display.data)
    viewer.cam.lookat[:] = [3.5, 0, 0.8]
    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 12, 90, -12
    while viewer.is_running():
        for r in recs:
            print(f"> {r['label']}", flush=True)
            if replay(r, viewer, display, slow=args.slow) is None:
                break
    viewer.close()


if __name__ == "__main__":
    main()
