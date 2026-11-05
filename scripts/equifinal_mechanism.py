"""What makes a delivery tolerant to release timing, and where the real bowler sits.

Moving the release later changes the ball's speed, elevation and height, and each of those moves the
landing. To first order:
    dL/dt = dL/ds * ds/dt  +  dL/de * de/dt  +  dL/dz * dz/dt
Here ds/dt, de/dt and dz/dt are measured along the arm's path over +/-10 ms and the partials come from
the env's ballistic, all per 10 ms of release timing. A flat landing curve needs small terms or terms
that cancel: that is Muller & Loosch's equifinal path, taken one delivery at a time.

Three sources, never mixed up:
  simulated  learned arm on the replayed capture, deterministic: release at t = 0 is the env's own
  bowled     the same policy bowling with the env's release-timing jitter switched ON, to check
             p_jitter_12ms (the expected rate from the noise-free curve) against real deliveries
  measured   raw hand/wrist markers (data/mocap/series_*.npz, 100 Hz, central differences). There is NO
             ball in the capture, so the landing is hypothetical: where a ball would land if it left with
             the marker's velocity. A marker on the back of the hand isn't the ball (it misses the fingers
             and the wrist's snap). The hand-speed peak lands far too short (4-6 m), so t = 0 for the humans
             is where the hypothetical landing crosses the middle of the good-length zone.

    python scripts/equifinal_mechanism.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import PPO

from envs.kinematic_bowl_env import KinematicBowlEnv
from envs.runup_bowl_env import BALL_R, G
from scripts.kin_equifinality import RUN_RE, run_episode

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "kinematic" / "equifinal_mechanism.json"
ZONE = (10.9, 12.9)
RUNS = ["kin_bowl_torque_a250_T3a_0", "kin_bowl_torque_a270_T3a_0", "kin_bowl_torque_b4096_a250_T3a_1"] + [
    f"kin_bowl_torque_b4096_t12_a{a}_T3a_{s}" for a in (250, 270, 290) for s in (0, 1)]
HUMANS = [("T3a", "hand"), ("T2b", "hand"), ("T1a", "wrist"), ("T4b", "wrist")]
STARTS = np.linspace(0.8, 1.2, 5)
JITTER_DELIVERIES = 200


def length(s, e, z):
    vx, vz = s * np.cos(e), s * np.sin(e)
    return vx * (vz + np.sqrt(vz ** 2 + 2 * G * (z - BALL_R))) / G


def terms(s, e, z, ds, de, dz):
    """Per unit of whatever ds, de, dz are per: the three first-order terms of dL/dt."""
    d = 1e-5
    return ((length(s + d, e, z) - length(s - d, e, z)) / (2 * d) * ds,
            (length(s, e + d, z) - length(s, e - d, z)) / (2 * d) * de,
            (length(s, e, z + d) - length(s, e, z - d)) / (2 * d) * dz)


def simulated(run):
    m = RUN_RE.match(run)
    env = KinematicBowlEnv(delivery=m["delivery"], rsi=False, release_angle_deg=float(m["ang"]),
                           arm_control=m["ctl"])
    model = PPO.load(ROOT / "checkpoints" / f"{run}.zip", device="cpu")
    h = env.m.opt.timestep
    k = int(round(0.040 / h))
    k10 = int(round(0.010 / h))
    rows, curve = [], None
    for before in STARTS:
        out, rec, _, _, _ = run_episode(env, model, float(before), 1)
        rel = len(rec) - 1
        _, rec2, arm, _, _ = run_episode(env, model, float(before), 1, release=False,
                                         t_stop=out["release_t"] + 0.040 + env.dt)

        def state(i):
            p, v, _ = rec2[i]
            return np.hypot(v[0], v[2]), np.arctan2(v[2], v[0]), p[2]

        lo, hi = max(rel - k, 0), min(rel + k, len(rec2) - 1)
        c = [state(i) for i in range(lo, hi + 1)]
        s, e, z = state(rel)
        a, b = state(rel - k10), state(rel + k10)
        ds, de, dz = (b[0] - a[0]) / 2, (b[1] - a[1]) / 2, (b[2] - a[2]) / 2       # per 10 ms
        t_s, t_e, t_z = terms(s, e, z, ds, de, dz)
        rows.append(dict(good=bool(out["good"]), speed_kmh=s * 3.6, elevation_deg=np.degrees(e), height_m=z,
                         length_m=out["land_x"], kmh_per_10ms=ds * 3.6, deg_per_10ms=np.degrees(de),
                         arm_deg_s=(arm[rel + 1 + k10] - arm[rel + 1 - k10]) / (2 * k10 * h),
                         term_speed=t_s, term_elevation=t_e, term_height=t_z, sum_terms=t_s + t_e + t_z,
                         slope_m_per_10ms=(length(*b) - length(*a)) / 2))
        if abs(before - 1.0) < 1e-9:
            curve = dict(t_ms=[(i - rel) * h * 1000 for i in range(lo, hi + 1)],
                         length_m=[float(length(*x)) for x in c])
    # bowl for real with the env's jitter on: the deterministic policy, the same starts spread out
    envj = KinematicBowlEnv(delivery=m["delivery"], rsi=False, release_angle_deg=float(m["ang"]),
                            arm_control=m["ctl"], release_timing_sd_ms=12.0)
    good = 0
    befores = np.linspace(0.8, 1.2, JITTER_DELIVERIES)
    for i in range(JITTER_DELIVERIES):
        obs, _ = envj.reset(seed=50_000 + i, options={"before_s": float(befores[i])})
        while True:
            obs, _, te, tr, info = envj.step(model.predict(obs, deterministic=True)[0])
            if te or tr:
                break
        good += bool(info["outcome"]["good"])
    keys = rows[0].keys() - {"good"}
    return dict(run=run, source="simulated", starts=len(rows), good=sum(r["good"] for r in rows),
                **{k_: float(np.mean([r[k_] for r in rows])) for k_ in keys},
                jitter_good_pct_bowled=100 * good / JITTER_DELIVERIES, curve=curve)


def measured(name, key):
    d = np.load(ROOT / "data" / "mocap" / f"series_{name}.npz")
    P, rate = d[key], float(d["rate"])
    V = np.gradient(P, 1 / rate, axis=0)
    fwd, up = V[:, 0] * float(d["direction"]), V[:, 1]          # markers: x along the pitch, y up
    s, e, z = np.hypot(fwd, up), np.arctan2(up, fwd), P[:, 1]
    with np.errstate(invalid="ignore"):
        L = length(s, e, z)
    rel = int(np.argmin(np.abs(d["t"] - float(d["release_frame"]) / rate)))
    # the crossing of the zone's middle, coming down the curve before the hand-speed peak
    mid = sum(ZONE) / 2
    idx = [i for i in range(rel - 8, rel) if L[i] >= mid > L[i + 1]]
    if not idx:
        raise ValueError(f"{name}: no crossing of {mid} m before the hand-speed peak")
    i = idx[-1]
    frac = (L[i] - mid) / (L[i] - L[i + 1])
    t_cross = (i + frac - rel) * 10.0                            # ms relative to the peak (negative: before)
    sc = s[i] + frac * (s[i + 1] - s[i])
    j = i if frac < 0.5 else i + 1                               # nearest frame, for the three terms
    ds, de, dz = (s[j + 1] - s[j - 1]) / 2, (e[j + 1] - e[j - 1]) / 2, (z[j + 1] - z[j - 1]) / 2   # per 10 ms
    t_s, t_e, t_z = terms(s[j], e[j], z[j], ds, de, dz)
    # good rate under N(0, 12 ms) along this curve, length only: there is no ball to give the line
    frames = np.arange(max(i - 4, 0), min(i + 6, len(L)))
    ok = ~np.isnan(L[frames])
    tt = (frames[ok] - i - frac) * 10.0
    fine = np.arange(max(tt.min(), -40), min(tt.max(), 40) + 0.01, 0.5)
    Lf = np.interp(fine, tt, L[frames[ok]])
    w = np.exp(-0.5 * (fine / 12.0) ** 2)
    in_zone = (Lf >= ZONE[0]) & (Lf <= ZONE[1])
    return dict(run=name, source="measured", marker=key, speed_kmh=float(sc) * 3.6,
                elevation_deg=float(np.degrees(e[i] + frac * (e[i + 1] - e[i]))),
                height_m=float(z[i] + frac * (z[i + 1] - z[i])), ms_before_speed_peak=float(-t_cross),
                length_at_peak_m=float(L[rel]), slope_m_per_10ms=float(L[i + 1] - L[i]),
                kmh_per_10ms=float(ds * 3.6), deg_per_10ms=float(np.degrees(de)), term_speed=float(t_s),
                term_elevation=float(t_e), term_height=float(t_z), sum_terms=float(t_s + t_e + t_z),
                window_ms=float(in_zone.sum() * 0.5),
                length_good_pct_12ms=float(100 * (w * in_zone).sum() / w.sum()),
                curve=dict(t_ms=[float(x) for x in tt], length_m=[float(x) for x in L[frames[ok]]]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="*", help="just these simulated runs, printed and not saved")
    a = p.parse_args()
    res = [simulated(r) for r in a.runs] if a.runs else [simulated(r) for r in RUNS] + [measured(*h) for h in HUMANS]
    if not a.runs:
        OUT.write_text(json.dumps(res, indent=1))
    for r in res:
        name = r["run"][16:] if r["source"] == "simulated" else f"{r['run']} {r['marker']}"
        line = (f"{name:24s} {r['speed_kmh']:5.1f} km/h elev {r['elevation_deg']:6.1f} h {r['height_m']:.2f} "
                f"speed {r['term_speed']:+5.2f} elev {r['term_elevation']:+6.2f} height {r['term_height']:+5.2f} "
                f"sum {r['sum_terms']:+6.2f} slope {r['slope_m_per_10ms']:+6.2f} "
                f"{r['kmh_per_10ms']:+5.1f} km/h, {r['deg_per_10ms']:+6.1f} deg per 10 ms ")
        if r["source"] == "simulated":
            line += f"arm {r['arm_deg_s']:.0f} deg/s; bowled under jitter {r['jitter_good_pct_bowled']:.1f}%, good {r['good']}/{r['starts']}"
        else:
            line += (f"window {r['window_ms']:.1f} ms, length-good under jitter {r['length_good_pct_12ms']:.1f}%, "
                     f"{r['ms_before_speed_peak']:.0f} ms before peak ({r['length_at_peak_m']:.1f} m there)")
        print(line)


if __name__ == "__main__":
    main()
