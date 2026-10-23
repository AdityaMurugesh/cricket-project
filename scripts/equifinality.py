"""Equifinality of a policy's delivery (Muller & Loosch 1999; Reiser et al. 2011), for every checkpoint.

If the ball had been let go at another instant near the actual release, where
would it land? A trajectory is equifinal when many release instants still hit.
Per checkpoint, from its deterministic delivery:
  window_ms  span of release instants around the actual one that land in the zone
  sd_40ms    SD of hypothetical landings for release +/-40 ms (Reiser's measure)
Hypothetical landings are drag-free ballistic from the hand's position and
velocity, shifted so the actual release lands where the simulator says it did
(the curve alone came out 0.2-0.6 m short on the run10 replays; the shift
removes that offset).

    python scripts/equifinality.py --run throw_ppo_run11_a230_s100_e0003_2 --target-range 10.9 12.9
"""
import argparse
import csv
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import PPO, SAC

from envs.throw_env import ThrowEnv
from envs.throw_env_gym import ThrowEnvGym
from scripts.eval_ckpts import TAG_RE

ROOT = Path(__file__).resolve().parent.parent
HALF_WINDOW_S = 0.040


def ballistic_x(p, v, g=9.81, r=0.036):
    z0 = p[2] - r
    return p[0] + v[0] * (v[2] + np.sqrt(max(v[2] ** 2 + 2 * g * z0, 0.0))) / g


def policy_actions(model, env):
    obs, _ = env.reset(seed=0)
    acts, done, info = [], False, {}
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        acts.append(np.asarray(a, dtype=float))
        obs, _, term, trunc, info = env.step(a)
        done = term or trunc
    return acts, info


def measure(acts, frame_skip, ang, scale, zone):
    """Replay the actions physics step by physics step, tracking the 'release now' landing."""
    env = ThrowEnv(release_mode="fixed_angle", release_angle_deg=ang, actuator_scale=scale,
                   speed_weight=0.5, target_range=zone)
    env.reset()
    m, d, sid, dt = env.model, env.data, env.hand_site_id, env.model.opt.timestep
    k = int(round(HALF_WINDOW_S / dt))
    land, rel, v6, info = [], None, np.zeros(6), {}
    for i in range(len(acts) * frame_skip):
        _, _, done, info = env.step(acts[i // frame_skip])
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, sid, v6, 0)
        land.append(ballistic_x(d.site_xpos[sid].copy(), v6[3:].copy()))
        if rel is None and info.get("released"):
            rel = i
        if rel is not None and i - rel >= k:
            break
    # window_early_ms / window_late_ms: how far before / after the actual release the
    # in-zone window reaches (late side capped at 40 ms: the replay stops there)
    # p_jitter_12ms: chance of landing in the zone with release jitter N(0, 12 ms), from the
    # whole +/-40 ms landing curve (the window above is only the unbroken stretch);
    # slope_m_per_10ms: how fast the landing moves with release time, at the release
    out = {"legal": bool(info.get("legal")), "land_x": None, "window_ms": 0.0, "window_early_ms": 0.0,
           "window_late_ms": 0.0, "p_jitter_12ms": 0.0, "slope_m_per_10ms": None,
           "sd_40ms": None, "early_40ms": None, "late_40ms": None}
    while not (env.landed or env.spent or info.get("timeout")):
        _, _, done, info = env.step(acts[-1])
    if rel is None or info.get("landing_pos") is None:
        return out
    actual = float(info["landing_pos"][0])
    L = np.array(land) + (actual - land[rel])
    w = L[max(rel - k, 0): rel + k + 1]
    inz = (L >= zone[0]) & (L <= zone[1])
    a = b = rel
    if inz[rel]:
        while a > 0 and inz[a - 1]:
            a -= 1
        while b < len(L) - 1 and inz[b + 1]:
            b += 1
        out["window_ms"] = (b - a + 1) * dt * 1000
        out["window_early_ms"] = (rel - a) * dt * 1000
        out["window_late_ms"] = (b - rel) * dt * 1000
    lo = max(rel - k, 0)
    t = (np.arange(lo, lo + len(w)) - rel) * dt
    g = np.exp(-0.5 * (t / 0.012) ** 2)
    out["p_jitter_12ms"] = float((g * inz[lo: lo + len(w)]).sum() / g.sum()) if out["legal"] else 0.0
    if 0 < rel < len(L) - 1:
        out["slope_m_per_10ms"] = float((L[rel + 1] - L[rel - 1]) / (2 * dt) * 0.010)
    out.update(land_x=actual, sd_40ms=float(w.std()), early_40ms=float(w[0]), late_40ms=float(w[-1]))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, nargs="+")
    p.add_argument("--target-range", type=float, nargs=2, default=[6.0, 8.0])
    p.add_argument("--root", default=str(ROOT))
    p.add_argument("--out", default=None, help="csv; default logs/<run>/equifinality.csv")
    a = p.parse_args()
    zone, root = tuple(a.target_range), Path(a.root)
    for run in a.run:
        mt = TAG_RE.match(run)
        if not mt:
            sys.exit(f"not a sweep run name: {run}")
        Algo = SAC if mt["algo"] == "sac" else PPO
        ang, scale = float(mt["ang"]), int(mt["s100"]) / 100.0
        ckpt_dir = root / "checkpoints" / f"{run}_ckpts"
        paths = sorted(ckpt_dir.glob("step_*_steps.zip"), key=lambda q: int(q.stem.split("_")[1]))
        final = root / "checkpoints" / f"{run}.zip"
        paths += [final] if final.exists() else []
        genv = ThrowEnvGym(speed_weight=0.5, release_mode="fixed_angle", release_angle_deg=ang,
                           actuator_scale=scale, target_range=zone)
        rows = []
        for q in paths:
            model = Algo.load(q)
            acts, _ = policy_actions(model, genv)
            r = measure(acts, genv.frame_skip, ang, scale, zone)
            steps = int(q.stem.split("_")[1]) if q.parent == ckpt_dir else model.num_timesteps
            rows.append({"run": run, "checkpoint": q.name, "timesteps": steps, **r})
            print(f"{run} {q.name:26s} legal={r['legal']!s:5s} land {r['land_x'] or 0:6.2f} "
                  f"window {r['window_ms']:5.1f} ms  sd40 {r['sd_40ms'] or 0:.2f} m", flush=True)
        out = Path(a.out) if a.out else root / "logs" / run / "equifinality.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        if rows:
            with open(out, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)


if __name__ == "__main__":
    main()
