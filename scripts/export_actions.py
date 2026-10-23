"""Record a checkpoint's deterministic delivery as an action sequence (.npz).

Torch can't load on the Windows laptop, but ThrowEnv has no reset noise, so a
deterministic policy's delivery is fully described by its action sequence:
record it here (HPC), replay it anywhere with scripts/showcase.py (pure MuJoCo).
  python scripts/export_actions.py --run throw_ppo_run10_a275_s150_e0003_2 --label "1.5x budget, 275 deg"
  python scripts/export_actions.py --run throw_ppo_run14_noshape_a260_s150_e0003_3 --target-range 10.9 12.9 \n      --out-dir results/replays_good_length --label "1.5x, 260 deg"
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
from scripts.eval_ckpts import TAG_RE

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--checkpoint", default=None,
                        help="a file in checkpoints/<run>_ckpts/ (default: the final policy)")
    parser.add_argument("--label", default="")
    parser.add_argument("--out-dir", default=str(ROOT / "results" / "replays"))
    parser.add_argument("--target-range", type=float, nargs=2, default=[6.0, 8.0])
    args = parser.parse_args()

    m = TAG_RE.match(args.run)
    ang, scale = float(m["ang"]), int(m["s100"]) / 100.0
    path = (ROOT / "checkpoints" / f"{args.run}_ckpts" / args.checkpoint if args.checkpoint
            else ROOT / "checkpoints" / f"{args.run}.zip")
    env = ThrowEnvGym(speed_weight=0.5, release_mode="fixed_angle",
                      release_angle_deg=ang, actuator_scale=scale,
                      target_range=tuple(args.target_range))
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
        "good": bool(bool(info.get("legal")) and bool(land) and
                     args.target_range[0] <= land[0] <= args.target_range[1]),
    }
    name = re.sub(r"^throw_ppo_run\d+\w*?_(?=a\d)", "", args.run) + (
        "_" + Path(args.checkpoint).stem if args.checkpoint else "_final")
    out = Path(args.out_dir) / f"{name}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, actions=np.array(actions), release_angle_deg=ang, actuator_scale=scale,
             frame_skip=env.frame_skip, label=args.label, outcome=json.dumps(outcome),
             target_range=np.array(args.target_range))
    print(f"{out.name}: {len(actions)} steps, {outcome}")


if __name__ == "__main__":
    main()
