"""Score trained policies with motor noise switched on (evaluation only, no training).

The policy acts deterministically; only the environment is noisy:
  timing  release-timing jitter, SD --timing-ms (default 12 ms, Muller & Loosch)
  torque  signal-dependent torque noise, SD --torque-sd x command (default 0.15)
  both    both at once
A policy trained without noise that keeps the zone under timing jitter is
equifinal in Muller & Loosch's sense; one that does not was never asked to be.

    python scripts/noise_robustness.py --run throw_ppo_run13b_noshape_a240_s150_e0003_0 \
        --target-range 10.9 12.9 --out results/run13b/noise_robustness.csv
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import PPO, SAC

from envs.throw_env_gym import ThrowEnvGym
from scripts.eval_checkpoint import rollout, row
from scripts.eval_ckpts import TAG_RE

ROOT = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", nargs="+", required=True)
    p.add_argument("--checkpoint", default=None, help="file in checkpoints/ to use instead of the final model")
    p.add_argument("--target-range", type=float, nargs=2, default=[6.0, 8.0])
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--timing-ms", type=float, default=12.0)
    p.add_argument("--torque-sd", type=float, default=0.15)
    p.add_argument("--root", default=str(ROOT))
    p.add_argument("--out", required=True)
    a = p.parse_args()
    root, zone = Path(a.root), tuple(a.target_range)
    conditions = {"none": (0.0, 0.0), "timing": (a.timing_ms, 0.0),
                  "torque": (0.0, a.torque_sd), "both": (a.timing_ms, a.torque_sd)}
    out_rows = []
    for run in a.run:
        m = TAG_RE.match(run)
        if not m:
            sys.exit(f"not a sweep run name: {run}")
        Algo = SAC if m["algo"] == "sac" else PPO
        path = root / "checkpoints" / (a.checkpoint or f"{run}.zip")
        model = Algo.load(path)
        for cond, (t_ms, q_sd) in conditions.items():
            env = ThrowEnvGym(speed_weight=0.5, release_mode="fixed_angle",
                              release_angle_deg=float(m["ang"]), actuator_scale=int(m["s100"]) / 100,
                              target_range=zone, release_timing_sd_ms=t_ms, torque_noise_sd=q_sd)
            eps = [row(rollout(env, model, True, 5000 + i), env) for i in range(a.episodes)]
            lands = [r["land_x"] for r in eps if r["land_x"] is not None]
            good = [r for r in eps if r["good"]]
            rec = {"run": run, "condition": cond, "timing_sd_ms": t_ms, "torque_sd": q_sd,
                   "episodes": len(eps),
                   "good_pct": 100.0 * len(good) / len(eps),
                   "legal_pct": 100.0 * sum(r["legal"] for r in eps) / len(eps),
                   "mean_land_x": float(np.mean(lands)) if lands else None,
                   "sd_land_x": float(np.std(lands)) if lands else None,
                   "mean_speed_kmh": float(np.mean([r["speed_kmh"] for r in eps if r["speed_kmh"]]))
                                     if any(r["speed_kmh"] for r in eps) else None}
            out_rows.append(rec)
            print(f"{run} {cond:6s} good {rec['good_pct']:5.1f}%  legal {rec['legal_pct']:5.1f}%  "
                  f"land {rec['mean_land_x'] or 0:5.2f} +/- {rec['sd_land_x'] or 0:4.2f} m  "
                  f"{rec['mean_speed_kmh'] or 0:4.1f} km/h", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)


if __name__ == "__main__":
    main()
