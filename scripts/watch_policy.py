"""Watch a trained PPO policy throw, live, in the MuJoCo viewer.

Same viewer setup as demo_scripted.py, but driven by a loaded model's
actions instead of hand-picked torques/release-step. In --view mode this
loops interactively (replay / next / quit) instead of running a fixed
number of episodes and closing -- same interaction pattern as
demo_scripted.py's replay prompt.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import PPO

from envs.throw_env_gym import ThrowEnvGym


def play_episode(env, model, viewer, deterministic, seed):
    obs, _ = env.reset(seed=seed)
    done = False
    info = {}
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if viewer is not None:
            if not viewer.is_running():
                return None
            viewer.sync()
            time.sleep(env._env.model.opt.timestep)
    return info


def print_result(info, env, episode_num, seed):
    speed = info.get("release_speed")
    speed_str = f"{speed * 3.6:.1f} km/h" if speed else "n/a"
    landing = info.get("landing_pos")
    print(f"\n[episode {episode_num}, seed={seed}]")
    if not info.get("released"):
        # a spent delivery: the arm swung up through the release window and
        # out the far side still holding the ball. Worth calling out here --
        # on screen it just looks like a swing with no throw.
        print(f"released: False  (delivery spent={info.get('spent')}, "
              f"arm reached {info.get('max_shoulder_deg', float('nan')):.0f} deg)")
    else:
        print("released: True")
        # the geometry that separates a bowling action from a sling. 270 deg is
        # straight up, the real overarm release point, and a bowler's elbow is
        # near 0 there. On screen a sling and a bowl can look similar until you
        # know where the ball actually left the hand.
        print(f"release geometry: shoulder {info['shoulder_at_release_deg']:.1f} deg, "
              f"elbow {info['elbow_at_release_deg']:.1f} deg, "
              f"height {info['release_height_m']:.2f} m")
    print(f"release_speed: {speed_str}")
    print(f"landing_pos (x, y): {landing}")
    print(f"elbow_extension_deg: {info.get('elbow_extension_deg')}  "
          f"(legal={info.get('legal')}, limit {env._env.max_legal_extension_deg} deg)")
    print(f"episode reward: {info.get('episode_reward')}")
    print(f"target zone: [{env._env.target_min}, {env._env.target_max}]")


def setup_camera(viewer, lookat, distance, azimuth, elevation):
    viewer.cam.lookat[:] = lookat
    viewer.cam.distance = distance
    viewer.cam.azimuth = azimuth
    viewer.cam.elevation = elevation


def run(model_path, view, n_episodes, deterministic, seed,
        cam_lookat, cam_distance, cam_azimuth, cam_elevation, speed_weight):
    # speed_weight must match what the checkpoint was TRAINED with, or the
    # rewards printed here are computed under a different objective than the
    # one the policy optimised. It does not change what you see -- the policy's
    # actions depend only on the observation -- only whether the numbers
    # alongside it mean anything.
    env = ThrowEnvGym(speed_weight=speed_weight)
    model = PPO.load(model_path)

    viewer = None
    if view:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(env._env.model, env._env.data)
        setup_camera(viewer, cam_lookat, cam_distance, cam_azimuth, cam_elevation)

    ep = 0
    cur_seed = seed
    while True:
        info = play_episode(env, model, viewer, deterministic, cur_seed)
        if info is None:
            break  # viewer closed mid-throw
        ep += 1
        print_result(info, env, ep, cur_seed)

        if viewer is None:
            # headless: run exactly n_episodes back-to-back, no prompt
            cur_seed += 1
            if ep >= n_episodes:
                break
            continue

        if not viewer.is_running():
            break

        print("\nEnter to replay same throw, 'n' for a new seed, "
              "'c azimuth elevation distance' to move the camera "
              "(e.g. 'c 45 -20 10'), or 'q' to quit:")
        try:
            line = input("> ").strip()
        except EOFError:
            break

        if line.lower() == "q":
            break
        if line.lower() == "n":
            cur_seed += 1
        elif line.startswith("c "):
            parts = line.split()[1:]
            if len(parts) == 3:
                try:
                    az, el, dist = (float(p) for p in parts)
                    setup_camera(viewer, cam_lookat, dist, az, el)
                except ValueError:
                    print("Couldn't parse that -- keeping previous camera.")
            else:
                print("Expected 3 values: azimuth elevation distance -- keeping previous camera.")
        # bare Enter: replay the same seed again

    if viewer is not None:
        viewer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--view", action="store_true", help="open the MuJoCo viewer")
    parser.add_argument("--episodes", type=int, default=5,
                         help="headless mode only -- ignored in --view mode, which loops until you quit")
    parser.add_argument("--stochastic", action="store_true",
                         help="sample actions instead of using the policy's deterministic mean")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cam-lookat", type=float, nargs=3, default=[3.5, 0, 0.8],
                         help="camera look-at point, e.g. --cam-lookat 3.5 0 0.8")
    parser.add_argument("--cam-distance", type=float, default=12)
    parser.add_argument("--cam-azimuth", type=float, default=90)
    parser.add_argument("--cam-elevation", type=float, default=-15)
    parser.add_argument("--speed-weight", type=float, default=0.1,
                         help="must match what the checkpoint was trained with, or the "
                              "printed rewards use a different objective than the policy "
                              "optimised. checkpoints/_local_sw03 was trained at 0.3.")
    args = parser.parse_args()
    run(args.model_path, args.view, args.episodes, not args.stochastic, args.seed,
        args.cam_lookat, args.cam_distance, args.cam_azimuth, args.cam_elevation,
        args.speed_weight)
