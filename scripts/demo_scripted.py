"""Hand-scripted swing and release on ThrowEnv, no learned policy. --view to watch."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from envs.throw_env import ThrowEnv


def play_episode(env, viewer, shoulder, elbow, release_action):
    env.reset()
    done = False
    t = 0
    info = {}
    reward = 0.0
    while not done:
        release = release_action
        obs, reward, done, info = env.step([shoulder, elbow, release])
        t += 1
        if viewer is not None:
            if not viewer.is_running():
                return None
            viewer.sync()
            time.sleep(env.model.opt.timestep)
    info["steps"] = t
    info["reward"] = reward
    return info


def print_result(info, env):
    speed = info["release_speed"]
    speed_str = f"{speed * 3.6:.1f} km/h" if speed is not None else "n/a"
    print(f"steps: {info['steps']}")
    print(f"released: {info['released']}")
    print(f"release_speed: {speed_str}")
    print(f"release geometry (shoulder, elbow, height): "
          f"{info['shoulder_at_release_deg']}, {info['elbow_at_release_deg']}, "
          f"{info['release_height_m']}")
    print(f"landing_pos (x, y): {info['landing_pos']}")
    print(f"timeout (never landed): {info['timeout'] and not env.landed}")
    print(f"final reward: {info['reward']}")
    print(f"target zone: [{env.target_min}, {env.target_max}]")


def run(view, shoulder, elbow, release_angle, max_steps):
    env = ThrowEnv(max_steps=max_steps)
    release_action = env.release_action_for_angle(release_angle)

    viewer = None
    if view:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(env.model, env.data)
        # pull the camera back so the whole flight is visible
        viewer.cam.lookat[:] = [3.5, 0, 0.8]
        viewer.cam.distance = 12
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -15

    while True:
        info = play_episode(env, viewer, shoulder, elbow, release_action)
        if info is None:
            break  # viewer window was closed mid-throw
        print(f"\n[shoulder={shoulder} elbow={elbow} release_angle={release_angle}]")
        print_result(info, env)

        if viewer is None:
            break  # headless: one run and out

        if not viewer.is_running():
            break

        print("\nEnter to replay, 'shoulder elbow release_angle' to change "
              "(e.g. '1.0 0.5 120'), or 'q' to quit:")
        try:
            line = input("> ").strip()
        except EOFError:
            break

        if line.lower() == "q":
            break
        if line:
            parts = line.split()
            if len(parts) == 3:
                try:
                    shoulder, elbow, release_angle = float(parts[0]), float(parts[1]), float(parts[2])
                    release_action = env.release_action_for_angle(release_angle)
                except ValueError:
                    print("Couldn't parse that -- keeping previous parameters.")
            else:
                print("Expected 3 values: shoulder elbow release_angle -- keeping previous parameters.")

    if viewer is not None:
        viewer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", action="store_true", help="open the MuJoCo viewer and allow replay")
    parser.add_argument("--shoulder", type=float, default=1.0, help="shoulder torque in [-1, 1]")
    parser.add_argument("--elbow", type=float, default=0.0, help="elbow torque in [-1, 1]")
    parser.add_argument("--release-angle", type=float, default=265.0,
                         help="shoulder angle (deg) at which to let go; 270 is straight up")
    parser.add_argument("--max-steps", type=int, default=900)
    args = parser.parse_args()
    run(view=args.view, shoulder=args.shoulder, elbow=args.elbow,
        release_angle=args.release_angle, max_steps=args.max_steps)
