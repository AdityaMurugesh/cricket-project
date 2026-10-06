"""Evaluate every periodic checkpoint of one run10 run; write ckpt_eval.json in its log dir.

The deterministic delivery decides "best": legal and in zone first, then
release speed. Stochastic stats are recorded alongside so a lucky
deterministic point can be told apart from a policy that has the zone.
  python scripts/eval_ckpts.py --run throw_ppo_run10_a260_s075_e0003_2 --episodes 50
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import PPO

from envs.throw_env_gym import ThrowEnvGym
from scripts.eval_checkpoint import rollout, row

ROOT = Path(__file__).resolve().parent.parent
TAG_RE = re.compile(r"throw_ppo_run10_a(?P<ang>[\d.]+)_s(?P<s100>\d+)_e\d+_\d+$")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="run name, e.g. throw_ppo_run10_a260_s075_e0003_2")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--speed-weight", type=float, default=0.5)
    args = parser.parse_args()

    m = TAG_RE.match(args.run)
    if not m:
        sys.exit(f"not a run10 run name: {args.run}")
    ang, scale = float(m["ang"]), int(m["s100"]) / 100.0
    ckpt_dir = ROOT / "checkpoints" / f"{args.run}_ckpts"
    paths = sorted(ckpt_dir.glob("step_*_steps.zip"),
                   key=lambda p: int(p.stem.split("_")[1]))
    final = ROOT / "checkpoints" / f"{args.run}.zip"
    if final.exists():
        paths.append(final)
    if not paths:
        sys.exit(f"no checkpoints for {args.run}")

    env = ThrowEnvGym(speed_weight=args.speed_weight, release_mode="fixed_angle",
                      release_angle_deg=ang, actuator_scale=scale)
    results = []
    for p in paths:
        model = PPO.load(p)
        det = row(rollout(env, model, True, 0), env)
        stoch = [row(rollout(env, model, False, 1000 + i), env) for i in range(args.episodes)]
        good = [r for r in stoch if r["good"]]
        steps = int(p.stem.split("_")[1]) if p.parent == ckpt_dir else model.num_timesteps
        results.append({
            "checkpoint": p.name, "timesteps": steps, "deterministic": det,
            "stoch_good_pct": 100.0 * len(good) / len(stoch),
            "stoch_mean_speed_good_kmh": float(np.mean([r["speed_kmh"] for r in good])) if good else None,
        })
        d = det
        print(f"{p.name:28s} det {'GOOD' if d['good'] else 'miss'} {(d['speed_kmh'] or 0):5.1f} km/h "
              f"land {d['land_x']}  stoch good {results[-1]['stoch_good_pct']:.0f}%", flush=True)

    good_det = [r for r in results if r["deterministic"]["good"]]
    best = max(good_det, key=lambda r: r["deterministic"]["speed_kmh"]) if good_det else None
    out = {"run": args.run, "release_angle_deg": ang, "actuator_scale": scale,
           "checkpoints": results, "best": best}
    (ROOT / "logs" / args.run / "ckpt_eval.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
