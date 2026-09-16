"""Inspect a trained checkpoint's actual decision-making, not just its scores.

A training log can show a 100% release rate, 100% ICC legality and a
healthy reward while the policy has learned nothing at all about WHEN to
let go of the ball. That happened on 2026-09-16: the release channel's
mean sat flat at about -0.45 across the whole release window with a
standard deviation of 1.73, so roughly 39% of samples fired regardless of
arm angle. Over the ~12 decisions the arm spends inside the window that
is a ~99.8% chance of releasing somewhere, which is where the 100%
release rate came from -- but the release instant was the first coin flip
to come up heads, not a decision. The same policy evaluated
deterministically never released at all.

So this reports three things the episode CSV cannot:

  1. the deterministic rollout -- the policy you would actually report,
     which is the distribution's mean action rather than a sample
  2. stochastic rollout statistics, which is what the training log saw
  3. the release-channel profile: mean action and firing probability as a
     function of shoulder angle

If (1) and (2) disagree, the behaviour is coming from exploration noise.
If the profile in (3) is flat across the window, release timing is not
being controlled at all, whatever the success rate says.

Speeds in km/h throughout -- see the speed-units memory.
"""
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
    return (f"released at shoulder {info['shoulder_at_release_deg']:.1f} deg, "
            f"elbow {info['elbow_at_release_deg']:.1f} deg, "
            f"height {info['release_height_m']:.2f} m | {speed * 3.6:.1f} km/h | "
            f"landed {landing[0]:.2f} m" if landing else "no landing") + (
            f" | extension {ext:+.2f} deg, legal={info.get('legal')} "
            f"| reward {info.get('episode_reward', float('nan')):+.2f}")


def release_profile(env, model):
    """Mean release action and its firing probability vs shoulder angle,
    walked along a deterministic rollout. A flat column means the policy
    is not timing the release at all."""
    std = float(np.exp(model.policy.log_std.detach().cpu().numpy()[2]))
    obs, _ = env.reset(seed=0)
    rows = []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        shoulder = float(np.degrees(env._env.data.qpos[0]))
        p_fire = 0.5 * (1.0 - math.erf((0.0 - float(action[2])) / (std * math.sqrt(2.0))))
        rows.append((shoulder, float(action[2]), p_fire))
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    return std, rows


def run(model_path, episodes, seed, speed_weight):
    # must match what the checkpoint was TRAINED with, or the rewards here
    # are computed under a different objective than the policy optimised.
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
        print("     noise, not chosen -- see this file's docstring.")

    std, rows = release_profile(env, model)
    print(f"\n=== release channel profile (action std = {std:.2f}) ===")
    print(f"  {'shoulder':>9}  {'mean action':>12}  {'P(fires)':>9}")
    in_window = [r for r in rows if lo - 30 <= r[0] <= hi + 10]
    for shoulder, mean_a, p in in_window[:: max(1, len(in_window) // 12)]:
        mark = "  <- in window" if lo <= shoulder <= hi else ""
        print(f"  {shoulder:9.1f}  {mean_a:+12.3f}  {p:8.1%}{mark}")
    windowed = [r for r in rows if lo <= r[0] <= hi]
    if windowed:
        spread = max(r[1] for r in windowed) - min(r[1] for r in windowed)
        print(f"\n  mean action varies by only {spread:.3f} across the release window.")
        if spread < 0.2:
            print("  That is flat -- the policy is not timing the release at all.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--episodes", type=int, default=100,
                        help="stochastic episodes to sample for the statistics")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--speed-weight", type=float, default=0.1,
                        help="must match what the checkpoint was trained with, or the "
                             "printed rewards use a different objective than the policy "
                             "optimised.")
    args = parser.parse_args()
    run(args.model_path, args.episodes, args.seed, args.speed_weight)
