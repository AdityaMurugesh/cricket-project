"""Evaluate one checkpoint and write a JSON row: deterministic delivery + stochastic stats.

Machine-readable counterpart to inspect_policy.py, so a sweep's checkpoints can
be aggregated without scraping text. Speeds in km/h.
"""
import argparse
import json
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


def row(info, env):
    landing = info.get("landing_pos")
    land_x = float(landing[0]) if landing else None
    speed = info.get("release_speed")
    in_zone = land_x is not None and env._env.target_min <= land_x <= env._env.target_max
    return {
        "released": bool(info.get("released")),
        "spent": bool(info.get("spent")),
        "speed_kmh": speed * 3.6 if speed else None,
        "land_x": land_x,
        "extension_deg": info.get("elbow_extension_deg"),
        "legal": bool(info.get("legal")),
        "in_zone": bool(in_zone),
        "good": bool(info.get("legal")) and bool(in_zone),
        "release_shoulder_deg": info.get("shoulder_at_release_deg"),
        "release_elbow_deg": info.get("elbow_at_release_deg"),
        "release_height_m": info.get("release_height_m"),
        "episode_reward": info.get("episode_reward"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--speed-weight", type=float, default=0.5)
    parser.add_argument("--release-mode", default="fixed_angle")
    parser.add_argument("--release-angle", type=float, default=None)
    parser.add_argument("--actuator-scale", type=float, default=1.0)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    env = ThrowEnvGym(speed_weight=args.speed_weight, release_mode=args.release_mode,
                      release_angle_deg=args.release_angle, actuator_scale=args.actuator_scale)
    model = PPO.load(args.model_path)

    det = row(rollout(env, model, True, args.seed), env)
    stoch = [row(rollout(env, model, False, args.seed + 1000 + i), env)
             for i in range(args.episodes)]
    n = len(stoch)
    good = [r for r in stoch if r["good"]]
    rel = [r for r in stoch if r["released"]]

    def pct(items):
        return 100.0 * len(items) / n if n else 0.0

    out = {
        "tag": args.tag,
        "model_path": args.model_path,
        "release_mode": args.release_mode,
        "release_angle_deg": args.release_angle,
        "actuator_scale": args.actuator_scale,
        "speed_weight": args.speed_weight,
        "deterministic": det,
        "stochastic": {
            "episodes": n,
            "released_pct": pct(rel),
            "legal_pct": pct([r for r in stoch if r["legal"]]),
            "in_zone_pct": pct([r for r in stoch if r["in_zone"]]),
            "good_pct": pct(good),
            "mean_speed_kmh": float(np.mean([r["speed_kmh"] for r in rel])) if rel else None,
            "mean_speed_good_kmh": float(np.mean([r["speed_kmh"] for r in good])) if good else None,
            "best_speed_good_kmh": float(max(r["speed_kmh"] for r in good)) if good else None,
            "mean_land_x": float(np.mean([r["land_x"] for r in rel if r["land_x"] is not None]))
                           if any(r["land_x"] is not None for r in rel) else None,
            "sd_land_x": float(np.std([r["land_x"] for r in rel if r["land_x"] is not None]))
                         if any(r["land_x"] is not None for r in rel) else None,
            "mean_extension_deg": float(np.mean([r["extension_deg"] for r in rel
                                                 if r["extension_deg"] is not None]))
                                  if any(r["extension_deg"] is not None for r in rel) else None,
        },
        "action_std": [float(s) for s in np.exp(model.policy.log_std.detach().cpu().numpy())],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    d = det
    print(f"deterministic: released={d['released']} "
          f"{(d['speed_kmh'] or 0):.1f} km/h land {d['land_x']} legal={d['legal']} good={d['good']}")
    s = out["stochastic"]
    print(f"stochastic x{n}: good {s['good_pct']:.0f}%  legal {s['legal_pct']:.0f}%  "
          f"zone {s['in_zone_pct']:.0f}%  mean good speed {s['mean_speed_good_kmh']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
