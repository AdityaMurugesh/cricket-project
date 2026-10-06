"""Scripted speed/accuracy frontier: no learning, pure MuJoCo (runs locally, no torch).

For each actuator budget and fixed release angle, search constant-torque swings
(shoulder torque u_s on a grid, refined by bisection; a few elbow torques) and
report the fastest delivery that is both ICC-legal and lands in the target zone.
This is the reference ceiling the run10 RL sweep is compared against: a learned
policy can use time-varying torque, but at a fixed release angle with a near-
straight arm the landing point is set almost entirely by release speed, so the
in-zone speed band is a property of the physics rather than of the controller.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from envs.throw_env import ThrowEnv

ROOT = Path(__file__).resolve().parent.parent


def deliver(env, u_shoulder, u_elbow):
    env.reset()
    done = False
    info = {}
    while not done:
        _, _, done, info = env.step([u_shoulder, u_elbow])
    land = info["landing_pos"][0] if info.get("landing_pos") else None
    return {
        "u_shoulder": u_shoulder, "u_elbow": u_elbow,
        "released": info["released"],
        "speed_kmh": info["release_speed"] * 3.6 if info["release_speed"] else None,
        "land_x": land,
        "extension_deg": info["elbow_extension_deg"],
        "legal": info["legal"],
        "release_shoulder_deg": info["shoulder_at_release_deg"],
        "release_height_m": info["release_height_m"],
    }


def good(r, env):
    return (r["released"] and r["legal"] and r["land_x"] is not None
            and env.target_min <= r["land_x"] <= env.target_max)


def frontier_point(scale, angle, elbow_ctrls, n_grid, n_bisect):
    env = ThrowEnv(release_mode="fixed_angle", release_angle_deg=angle, actuator_scale=scale)
    best = None
    tried = 0
    for ue in elbow_ctrls:
        grid = [deliver(env, float(u), ue) for u in np.linspace(0.0, 1.0, n_grid)]
        tried += len(grid)
        cands = [r for r in grid if good(r, env)]
        # refine the far zone edge: the fastest in-zone throw sits where landing crosses target_max
        for a, b in zip(grid, grid[1:]):
            if a["land_x"] is None or b["land_x"] is None:
                continue
            lo, hi = a, b
            if not ((lo["land_x"] - env.target_max) * (hi["land_x"] - env.target_max) < 0):
                continue
            for _ in range(n_bisect):
                mid = deliver(env, 0.5 * (lo["u_shoulder"] + hi["u_shoulder"]), ue)
                tried += 1
                if mid["land_x"] is None:
                    break
                if good(mid, env):
                    cands.append(mid)
                if (mid["land_x"] - env.target_max) * (lo["land_x"] - env.target_max) < 0:
                    hi = mid
                else:
                    lo = mid
        for r in cands:
            if best is None or r["speed_kmh"] > best["speed_kmh"]:
                best = r
    full = deliver(env, 1.0, elbow_ctrls[0])
    return best, full, tried


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scales", type=float, nargs="+", default=[0.75, 1.0, 1.5])
    parser.add_argument("--angles", type=float, nargs="+",
                        default=[230, 240, 250, 260, 270, 280, 290, 300])
    parser.add_argument("--elbow-ctrls", type=float, nargs="+", default=[-0.2, -1.0, 0.0])
    parser.add_argument("--grid", type=int, default=41)
    parser.add_argument("--bisect", type=int, default=12)
    parser.add_argument("--out", default=str(ROOT / "logs" / "frontier_scripted.csv"))
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"{'scale':>5} {'angle':>5} | {'best legal in-zone':>32} | {'full-torque swing':>26}")
    for scale in args.scales:
        for angle in args.angles:
            best, full, tried = frontier_point(scale, angle, args.elbow_ctrls, args.grid, args.bisect)
            if best:
                b = (f"{best['speed_kmh']:5.1f} km/h land {best['land_x']:5.2f} m "
                     f"u_s={best['u_shoulder']:.3f}")
            else:
                b = "none reachable"
            f = (f"{full['speed_kmh']:5.1f} km/h land {full['land_x']:5.2f} m"
                 if full["speed_kmh"] and full["land_x"] is not None else "no release")
            print(f"{scale:5.2f} {angle:5.0f} | {b:>32} | {f:>26}   ({tried} sims)", flush=True)
            rows.append({
                "actuator_scale": scale, "release_angle_deg": angle,
                "best_speed_kmh": best["speed_kmh"] if best else None,
                "best_land_x": best["land_x"] if best else None,
                "best_u_shoulder": best["u_shoulder"] if best else None,
                "best_u_elbow": best["u_elbow"] if best else None,
                "best_extension_deg": best["extension_deg"] if best else None,
                "best_release_shoulder_deg": best["release_shoulder_deg"] if best else None,
                "best_release_height_m": best["release_height_m"] if best else None,
                "full_torque_speed_kmh": full["speed_kmh"],
                "full_torque_land_x": full["land_x"],
            })
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
