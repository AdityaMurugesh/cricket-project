"""Record a checkpoint's deterministic delivery as an action sequence (.npz).

Torch can't load on the Windows laptop, but ThrowEnv has no reset noise, so a
deterministic policy's delivery is fully described by its action sequence:
record it here (HPC), replay it anywhere with scripts/showcase.py (pure MuJoCo).
  python scripts/export_actions.py --run throw_ppo_run10_a275_s150_e0003_2 --label "1.5x budget, 275 deg"
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

ROOT = Path(__file__).resolve().parent.parent
TAG_RE = re.compile(r"throw_ppo_run10_a(?P<ang>[\d.]+)_s(?P<s100>\d+)_e\d+_\d+$")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--checkpoint", default=None,
                        help="a file in checkpoints/<run>_ckpts/ (default: the final policy)")
    parser.add_argument("--label", default="")
    parser.add_argument("--out-dir", default=str(ROOT / "results" / "replays"))
    args = parser.parse_args()

    m = TAG_RE.match(args.run)
    ang, scale = float(m["ang"]), int(m["s100"]) / 100.0
    path = (ROOT / "checkpoints" / f"{args.run}_ckpts" / args.checkpoint if args.checkpoint
            else ROOT / "checkpoints" / f"{args.run}.zip")
    env = ThrowEnvGym(speed_weight=0.5, release_mode="fixed_angle",
                      release_angle_deg=ang, actuator_scale=scale)
    model = PPO.load(path)

    obs, _ = env.reset(seed=0)
    actions, done, info = [], False, {}
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        actions.append(np.asarray(action, dtype=np.float64))
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    land = info.get("landing_pos")
    outcome = {
        "released": bool(info.get("released")),
        "speed_kmh": info["release_speed"] * 3.6 if info.get("release_speed") else None,
        "land_x": float(land[0]) if land else None,
        "extension_deg": info.get("elbow_extension_deg"),
        "legal": bool(info.get("legal")),
    }
    name = args.run.replace("throw_ppo_run10_", "") + (
        "_" + Path(args.checkpoint).stem if args.checkpoint else "_final")
    out = Path(args.out_dir) / f"{name}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, actions=np.array(actions), release_angle_deg=ang, actuator_scale=scale,
             frame_skip=env.frame_skip, label=args.label, outcome=json.dumps(outcome))
    print(f"{out.name}: {len(actions)} steps, {outcome}")


if __name__ == "__main__":
    main()
