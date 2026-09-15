"""Watch a trained PPO policy throw, live, in the MuJoCo viewer.

Same viewer setup as demo_scripted.py, but driven by a loaded model's
actions instead of hand-picked torques/release-step.
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


def print_result(info, env, episode_num):
    speed = info.get("release_speed")
    speed_str = f"{speed * 3.6:.1f} km/h" if speed else "n/a"
    landing = info.get("landing_pos")
    print(f"\n[episode {episode_num}]")
    print(f"released: {info.get('released')}")
    print(f"release_speed: {speed_str}")
    print(f"landing_pos (x, y): {landing}")
    print(f"elbow_extension_deg: {info.get('elbow_extension_deg')}")
    print(f"episode reward: {info.get('episode_reward')}")
    print(f"target zone: [{env._env.target_min}, {env._env.target_max}]")


def run(model_path, view, n_episodes, deterministic, seed):
    env = ThrowEnvGym()
    model = PPO.load(model_path)

    viewer = None
    if view:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(env._env.model, env._env.data)
        viewer.cam.lookat[:] = [3.5, 0, 0.8]
        viewer.cam.distance = 12
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -15

    ep = 0
    while ep < n_episodes:
        info = play_episode(env, model, viewer, deterministic, seed + ep)
        if info is None:
            break  # viewer closed mid-throw
        ep += 1
        print_result(info, env, ep)
        if viewer is not None and not viewer.is_running():
            break

    if viewer is not None:
        viewer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--view", action="store_true", help="open the MuJoCo viewer")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--stochastic", action="store_true",
                         help="sample actions instead of using the policy's deterministic mean")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(args.model_path, args.view, args.episodes, not args.stochastic, args.seed)
