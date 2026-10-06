"""Compare a checkpoint's deterministic vs stochastic behaviour and its release profile."""
import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import PPO

from envs.throw_env_gym import ThrowEnvGym


def rollout(env, model, deterministic, seed):
    obs, _ = env.reset(seed=seed)
    done = False
    info = {}
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return info


def describe(info):
    if not info.get("released"):
        return (f"NO RELEASE (delivery spent={info.get('spent')}, "
                f"arm reached {info.get('max_shoulder_deg', float('nan')):.0f} deg), "
                f"reward {info.get('episode_reward', float('nan')):+.2f}")
    landing = info.get("landing_pos")
    speed = info.get("release_speed") or 0.0
    ext = info.get("elbow_extension_deg")
    target = info.get("release_target_deg")
    asked = f" (asked for {target:.1f})" if target is not None else ""
    where = f"landed {landing[0]:.2f} m" if landing else "never landed"
    return (f"released at shoulder {info['shoulder_at_release_deg']:.1f} deg{asked}, "
            f"elbow {info['elbow_at_release_deg']:.1f} deg, "
            f"height {info['release_height_m']:.2f} m | {speed * 3.6:.1f} km/h | "
            f"{where} | extension {ext:+.2f} deg, legal={info.get('legal')} "
            f"| reward {info.get('episode_reward', float('nan')):+.2f}")


def release_profile(env, model):
    """Release action vs shoulder angle along a deterministic rollout."""
    std = float(np.exp(model.policy.log_std.detach().cpu().numpy()[2]))
    mode = env._env.release_mode
    obs, _ = env.reset(seed=0)
    rows = []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        shoulder = float(np.degrees(env._env.data.qpos[0]))
        if mode == "target_angle":
            # the release angle the policy is asking for
            third = env._env.release_target_deg(action[2])
        else:
            third = 100.0 * 0.5 * (1.0 - math.erf((0.0 - float(action[2])) / (std * math.sqrt(2.0))))
        rows.append((shoulder, float(action[2]), third))
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    return std, mode, rows


def run(model_path, episodes, seed, speed_weight):
    # speed_weight has to match training or the rewards printed are meaningless
    env = ThrowEnvGym(speed_weight=speed_weight)
    model = PPO.load(model_path)
    lo, hi = env._env.release_window_min, env._env.release_window_max

    print("=== deterministic (the policy you would report) ===")
    det = [rollout(env, model, True, seed + i) for i in range(3)]
    for i, info in enumerate(det):
        print(f"  seed {seed + i}: {describe(info)}")
    det_released = sum(1 for i in det if i.get("released"))

    print(f"\n=== stochastic, {episodes} episodes (what the training log sees) ===")
    stoch = [rollout(env, model, False, seed + 1000 + i) for i in range(episodes)]
    rel = [i for i in stoch if i.get("released")]
    legal = [i for i in rel if i.get("legal")]
    zone = [i for i in rel if i.get("landing_pos")
            and env._env.target_min <= i["landing_pos"][0] <= env._env.target_max]
    speeds = [i["release_speed"] * 3.6 for i in rel]
    angles = [i["shoulder_at_release_deg"] for i in rel]
    print(f"  released {len(rel)}/{episodes}  legal {len(legal)}/{max(1, len(rel))}  "
          f"in zone {len(zone)}/{max(1, len(rel))}")
    if speeds:
        print(f"  speed  mean {np.mean(speeds):.1f} km/h, best {max(speeds):.1f} km/h")
        print(f"  release angle  mean {np.mean(angles):.1f} deg, "
              f"sd {np.std(angles):.1f} deg  (window is {lo:.0f}-{hi:.0f})")

    if det_released == 0 and rel:
        print("\n  !! The deterministic policy never releases but the stochastic one")
        print("     almost always does. The release is being fired by exploration")
        print("     noise, not chosen.")

    std, mode, rows = release_profile(env, model)
    print(f"\n=== release channel profile (mode={mode}, action std = {std:.2f}) ===")
    # a big std here means release timing is basically random
    if std > 4.0:
        print(f"  !! std {std:.1f} here is far above the torque dimensions --")
        print("     sampled release is close to random.")
    third_label = "target deg" if mode == "target_angle" else "P(fires) %"
    print(f"  {'shoulder':>9}  {'mean action':>12}  {third_label:>11}")
    in_window = [r for r in rows if lo - 30 <= r[0] <= hi + 10]
    for shoulder, mean_a, third in in_window[:: max(1, len(in_window) // 12)]:
        mark = "  <- in window" if lo <= shoulder <= hi else ""
        print(f"  {shoulder:9.1f}  {mean_a:+12.3f}  {third:11.1f}{mark}")
    windowed = [r for r in rows if lo <= r[0] <= hi]
    if windowed:
        spread = max(r[1] for r in windowed) - min(r[1] for r in windowed)
        print(f"\n  mean action varies by {spread:.3f} across the release window.")
        if mode == "threshold" and spread < 0.2:
            print("  That is flat -- the policy is not timing the release at all.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--episodes", type=int, default=100,
                        help="stochastic episodes to sample for the statistics")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--speed-weight", type=float, default=0.1,
                        help="must match the value the checkpoint was trained with")
    args = parser.parse_args()
    run(args.model_path, args.episodes, args.seed, args.speed_weight)
