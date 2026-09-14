"""Full bowler demo: scripted run-up into a scripted bowling swing, with
the ball going through real physics from the moment of release.

Run-up and swing are both kinematic (see envs/bowler_env.py for why) --
this is a visual demo, not the RL-trainable environment. Run with --view
to watch it; the same replay-and-edit loop as demo_scripted.py lets you
retune parameters between throws without restarting.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from envs.bowler_env import BowlerEnv


def play_episode(env, viewer, release_frac, real_time=True):
    env.reset()
    dt = env.dt

    n_run = round(env.run_duration / dt)
    for i in range(n_run + 1):
        env.run_up_step(i * dt)
        if viewer is not None:
            if not viewer.is_running():
                return None
            viewer.sync()
            if real_time:
                time.sleep(dt)

    release_t = env.delivery_duration * release_frac
    n_rel = round(release_t / dt)
    t2 = 0.0
    for i in range(n_rel + 1):
        env.delivery_step(t2)
        t2 += dt
        if viewer is not None:
            if not viewer.is_running():
                return None
            viewer.sync()
            if real_time:
                time.sleep(dt)
    env.release_ball()

    done = False
    info = {}
    reward = 0.0
    steps = 0
    while not done:
        obs, reward, done, info = env.flight_step()
        steps += 1
        if viewer is not None:
            if not viewer.is_running():
                return None
            viewer.sync()
            if real_time:
                time.sleep(dt)

    info["steps"] = n_run + n_rel + 2 + steps
    info["reward"] = reward
    return info


def print_result(info, env):
    speed = info["release_speed"]
    speed_str = f"{speed * 3.6:.1f} km/h" if speed is not None else "n/a"
    print(f"steps: {info['steps']}")
    print(f"release_speed: {speed_str}")
    print(f"landing_pos (x, y): {info['landing_pos']}")
    print(f"timeout (never landed): {info['timeout'] and not env.landed}")
    print(f"final reward: {info['reward']}")
    print(f"target zone: [{env.target_min}, {env.target_max}]")


def run(view, release_frac, max_steps):
    env = BowlerEnv(max_steps=max_steps)

    viewer = None
    if view:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(env.model, env.data)
        viewer.cam.lookat[:] = [0.0, 0, 0.9]
        viewer.cam.distance = 16
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -12

    while True:
        info = play_episode(env, viewer, release_frac)
        if info is None:
            break  # viewer window was closed mid-throw
        print(f"\n[release_frac={release_frac}]")
        print_result(info, env)

        if viewer is None:
            break
        if not viewer.is_running():
            break

        print("\nEnter to replay, a number (0-1) to change the release "
              "fraction of the swing, or 'q' to quit:")
        try:
            line = input("> ").strip()
        except EOFError:
            break

        if line.lower() == "q":
            break
        if line:
            try:
                release_frac = float(line)
            except ValueError:
                print("Couldn't parse that -- keeping previous release fraction.")

    if viewer is not None:
        viewer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", action="store_true", help="open the MuJoCo viewer and allow replay")
    parser.add_argument("--release-frac", type=float, default=0.56,
                         help="fraction of the delivery swing at which the ball is released")
    parser.add_argument("--max-steps", type=int, default=900)
    args = parser.parse_args()
    run(view=args.view, release_frac=args.release_frac, max_steps=args.max_steps)
