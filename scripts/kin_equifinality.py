"""Equifinality, covariation and variability of the kinematic bowler's deliveries.

The same questions scripts/equifinality.py and noise_robustness.py asked of
ThrowEnv's two-joint arm, now for KinematicBowlEnv (captured body, learned arm).

Per delivery (one start time, optionally with torque noise):
  1. replay it as trained: the actual release and outcome
  2. replay it again with release switched off, so the arm carries on as the
     policy drives it. Nothing the ball does feeds back into the arm, so up to
     the release the two replays are identical, and past it the second one is
     the arm the policy would have swung had the ball not gone yet
  3. for every physics step within +/-40 ms of the actual release: where the ball
     would land (length and line, the env's own ballistic), and whether that
     release would be good (legal elbow at that instant, arm in the bowling
     plane, inside the release window, in the length x line box)
From the curve (Muller & Loosch 1999; Reiser et al. 2011):
  window_ms        unbroken stretch of good release instants around the actual one
  p_jitter_12ms    good rate under release-timing jitter N(0, 12 ms). The jitter only
                   moves the release along this same arm path, so this is the exact
                   expected rate (up to |jitter| > 40 ms, p < 0.001), not a prediction
  slope_m_per_10ms how far the length moves per 10 ms of release timing
  sd_40ms          SD of hypothetical lengths over +/-40 ms (Reiser's measure)

Across a run's deliveries (21 starts, and 21 starts x --noise-seeds with torque noise):
  covariation (Muller & Loosch's surrogate test): release speed, elevation and height
  are shuffled independently across deliveries, which keeps each one's spread but
  breaks how they go together; covar_ratio = SD(length, shuffled) / SD(length, actual).
  Above 1: the parameters co-vary so that the length stays put.
  variability (Andrews et al. 2024's comparison): SD of release time (ms) and CV of
  release speed and of the arm-circle duration (arm hanging -> release).

    python scripts/kin_equifinality.py --run kin_bowl_torque_a270_T3a_0 --out results/kinematic/equifinality
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import PPO

from envs.kinematic_bowl_env import KinematicBowlEnv
from envs.legality import elbow_extension_deg
from envs.runup_bowl_env import BALL_R, G

ROOT = Path(__file__).resolve().parent.parent
HALF_WINDOW_S = 0.040
JITTER_SD_S = 0.012
RUN_RE = re.compile(r"kin_bowl_(?P<ctl>torque|pd)(?:_t[\d.]+)?(?:_q[\d.]+)?_a(?P<ang>\d+)_(?P<delivery>\w+?)_(?P<seed>\d+)$")


class Recorder:
    """Wraps env._track_arm (called after every physics step) to log the ball at the grip."""

    def __init__(self, env):
        self.env, self.rows = env, []
        self._track = env._track_arm
        env._track_arm = self

    def __call__(self):
        self._track()
        e = self.env
        self.rows.append((e._grip().copy(), e._grip_velocity().copy(), e._arm_out_of_plane_deg()))


def land(p, v):
    """RunupBowlEnv._release's ballistic: length beyond the release point, and line."""
    z0 = p[2] - BALL_R
    t_fl = (v[2] + np.sqrt(max(v[2] ** 2 + 2 * G * z0, 0.0))) / G
    return float(v[0] * t_fl), float(p[1] + v[1] * t_fl)


def run_episode(env, model, before_s, seed, release=True, t_stop=None):
    """One delivery from before_s ahead of the captured release; returns the env's outcome and recording."""
    angle = env.release_angle_deg
    env.release_angle_deg = angle if release else np.inf
    t_end = env.t_end
    if t_stop is not None:
        env.t_end = t_stop
    rec = Recorder(env)
    try:
        obs, _ = env.reset(seed=seed, options={"before_s": before_s})
        rec.rows.clear()
        while True:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, te, tr, info = env.step(a)
            if te or tr:
                break
    finally:
        env._track_arm = rec._track
        env.release_angle_deg, env.t_end = angle, t_end
    return dict(info["outcome"]), rec.rows, list(env._arm_hist), list(env._elbow_hist), env._start_t


def delivery(env, model, before_s, seed):
    out, rows, _, _, t0 = run_episode(env, model, before_s, seed)
    h = env.m.opt.timestep
    r = {"before_s": round(before_s, 3), "seed": seed, **{k: out.get(k) for k in
         ("released", "why", "speed_m_s", "land_x", "lateral_m", "extension_deg", "legal", "good",
          "release_t", "release_height_m", "arm_out_of_plane_deg")}}
    if not out["released"]:
        return r, None
    rel = len(rows) - 1                           # physics step of the release
    # out["release_t"] is the start of the releasing decision (20 ms steps); this is the instant
    r["release_instant_s"] = t0 + (rel + 1) * h
    k = int(round(HALF_WINDOW_S / h))
    # the same delivery with release switched off, run 40 ms (+1 decision) past the actual release
    _, rows2, arm, elbow, _ = run_episode(env, model, before_s, seed, release=False,
                                          t_stop=out["release_t"] + HALF_WINDOW_S + env.dt)
    p, v, _ = rows2[rel]
    assert np.allclose(land(p, v), (out["land_x"], out["lateral_m"]), atol=1e-6), "replays diverged"
    lo, hi = max(rel - k, 0), min(rel + k, len(rows2) - 1)
    curve = []
    for i in range(lo, hi + 1):
        p, v, plane = rows2[i]
        x, y = land(p, v)
        t = t0 + (i + 1) * h                      # rows[i] is after physics step i + 1
        ext = elbow_extension_deg(arm, elbow, i + 1)   # arm_hist[0] is the start state
        legal = ext is not None and ext <= 15.0
        ok = (legal and plane <= env.max_arm_out_of_plane_deg and env.t_window <= t <= env.t_end
              and abs(elbow[i + 1]) <= env.max_release_elbow_deg)
        good = ok and env.target_min <= x <= env.target_max and abs(y) <= env.line_tol_m
        curve.append((i - rel, x, y, good, float(np.linalg.norm(v))))
    off = np.array([c[0] for c in curve]) * h
    L = np.array([c[1] for c in curve])
    good = np.array([c[3] for c in curve])
    j = int(np.where(off == 0)[0][0])
    a = b = j
    if good[j]:
        while a > 0 and good[a - 1]:
            a -= 1
        while b < len(good) - 1 and good[b + 1]:
            b += 1
    w = np.exp(-0.5 * (off / JITTER_SD_S) ** 2)
    r.update(window_ms=(b - a + 1) * h * 1000 if good[j] else 0.0,
             window_early_ms=(j - a) * h * 1000 if good[j] else 0.0,
             window_late_ms=(b - j) * h * 1000 if good[j] else 0.0,
             p_jitter_12ms=float((w * good).sum() / w.sum()),
             slope_m_per_10ms=float((L[j + 1] - L[j - 1]) / (2 * h) * 0.010) if 0 < j < len(L) - 1 else None,
             sd_40ms=float(L.std()), early_40ms=float(L[0]), late_40ms=float(L[-1]))
    # release parameters for the covariation test, and the arm-circle duration
    assert bool(good[j]) == bool(out["good"]), "curve at the release disagrees with the env"
    p, v, _ = rows2[rel]
    # in the x-z plane, so that length = f(speed, elevation, height) exactly
    r["inplane_speed_m_s"] = float(np.hypot(v[0], v[2]))
    r["elevation_deg"] = float(np.degrees(np.arctan2(v[2], v[0])))
    hang = [i for i in range(1, rel + 1) if arm[i - 1] < 90.0 <= arm[i]]
    r["circle_s"] = (rel + 1 - hang[-1]) * h if hang else None
    return r, curve


def covariation(rows, n_perm=500, seed=0):
    """Muller & Loosch's surrogate test on the released deliveries' length."""
    rel = [r for r in rows if r["released"]]
    if len(rel) < 5:
        return {}
    s = np.array([r["inplane_speed_m_s"] for r in rel])
    e = np.radians([r["elevation_deg"] for r in rel])
    z = np.array([r["release_height_m"] for r in rel])

    def length(s, e, z):
        vx, vz = s * np.cos(e), s * np.sin(e)       # in-plane: line is a separate question
        return vx * (vz + np.sqrt(np.maximum(vz ** 2 + 2 * G * (z - BALL_R), 0.0))) / G

    actual = length(s, e, z)
    rng = np.random.default_rng(seed)
    sh = [length(rng.permutation(s), rng.permutation(e), rng.permutation(z)).std() for _ in range(n_perm)]
    t = np.array([r["release_instant_s"] for r in rel])
    speed = np.array([r["speed_m_s"] for r in rel])
    circ = np.array([r["circle_s"] for r in rel if r["circle_s"]])
    return {"n": len(rel), "sd_length_m": float(actual.std()), "sd_length_shuffled_m": float(np.mean(sh)),
            "covar_ratio": float(np.mean(sh) / actual.std()) if actual.std() > 0 else None,
            "sd_release_t_ms": float(t.std() * 1000), "cv_speed": float(speed.std() / speed.mean()),
            "cv_circle": float(circ.std() / circ.mean()) if len(circ) > 1 else None,
            "mean_speed_kmh": float(speed.mean() * 3.6), "corr_speed_elevation": float(np.corrcoef(s, e)[0, 1]),
            "corr_speed_height": float(np.corrcoef(s, z)[0, 1])}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", nargs="+", required=True, help="e.g. kin_bowl_torque_a270_T3a_0")
    p.add_argument("--model-path", default=None, help="instead of checkpoints/<run>.zip (one run only)")
    p.add_argument("--starts", type=int, default=21)
    p.add_argument("--noise-seeds", type=int, default=10, help="torque-noise deliveries per start (torque runs)")
    p.add_argument("--torque-sd", type=float, default=0.15)
    p.add_argument("--target-range", type=float, nargs=2, default=[10.9, 12.9])
    p.add_argument("--root", default=str(ROOT))
    p.add_argument("--out", required=True, help="directory: <run>_deliveries.csv, <run>_curves.json, <run>_summary.json")
    a = p.parse_args()
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for run in a.run:
        m = RUN_RE.match(run)
        if not m:
            sys.exit(f"not a kinematic run name: {run}")
        model = PPO.load(a.model_path or root / "checkpoints" / f"{run}.zip", device="cpu")
        conds = {"none": 0.0}
        if m["ctl"] == "torque" and a.noise_seeds > 0:
            conds["torque"] = a.torque_sd
        rows, curves, summary = [], {}, {"run": run, "model": str(a.model_path or f"checkpoints/{run}.zip")}
        for cond, sd in conds.items():
            env = KinematicBowlEnv(delivery=m["delivery"], rsi=False, release_angle_deg=float(m["ang"]),
                                   target_range=tuple(a.target_range), arm_control=m["ctl"], torque_noise_sd=sd)
            seeds = [0] if sd == 0 else range(1, a.noise_seeds + 1)
            cond_rows = []
            for before in np.linspace(*env.start_range, a.starts):
                for s in seeds:
                    r, c = delivery(env, model, float(before), 1000 * s + int(round(before * 1000)))
                    r = {"run": run, "condition": cond, **r}
                    cond_rows.append(r)
                    if c is not None:
                        curves[f"{cond}_{r['before_s']}_{s}"] = c
            rel = [r for r in cond_rows if r["released"]]
            summary[cond] = {
                "deliveries": len(cond_rows), "released": len(rel), "good": sum(r["good"] for r in cond_rows),
                "legal": sum(r["legal"] for r in cond_rows),
                # good rate with release-timing jitter on top (exact expectation, see the docstring)
                "good_with_timing_jitter_pct": 100.0 * sum(r.get("p_jitter_12ms", 0.0) for r in cond_rows)
                / len(cond_rows),
                "median_window_ms": float(np.median([r["window_ms"] for r in rel])) if rel else None,
                "median_abs_slope_m_per_10ms": float(np.median([abs(r["slope_m_per_10ms"]) for r in rel
                                                                if r["slope_m_per_10ms"] is not None]))
                if rel else None,
                **covariation(cond_rows)}
            rows += cond_rows
            s_ = summary[cond]
            print(f"{run} {cond:6s} good {s_['good']}/{s_['deliveries']}  +jitter {s_['good_with_timing_jitter_pct']:5.1f}%"
                  f"  window {s_['median_window_ms']} ms  covar {s_.get('covar_ratio')}", flush=True)
        with open(out / f"{run}_deliveries.csv", "w", newline="") as fh:
            keys = list(dict.fromkeys(k for r in rows for k in r))
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        # per delivery: (physics steps from the actual release, length, line, good, speed) per instant
        json.dump(curves, open(out / f"{run}_curves.json", "w"),
                  default=lambda x: bool(x) if isinstance(x, np.bool_) else float(x))
        json.dump(summary, open(out / f"{run}_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
