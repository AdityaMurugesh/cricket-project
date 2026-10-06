"""Bucket a run's episodes.csv over training and write curve.json next to it.

Shows whether a run found the zone and kept it, or found it and drifted
out (seen in run10). No torch: plain csv, so it runs anywhere.
  python scripts/curve_summary.py logs/throw_ppo_run10_*/episodes.csv --buckets 10
"""
import argparse
import csv
import json
import statistics as stats
from pathlib import Path


def _f(v):
    return float(v) if v not in (None, "") else None


def curve(csv_path, buckets, target=(6.0, 8.0), max_ext=15.0):
    rows = list(csv.DictReader(open(csv_path)))
    n = len(rows)
    out = []
    for b in range(buckets):
        chunk = rows[b * n // buckets:(b + 1) * n // buckets]
        if not chunk:
            continue
        good, speeds_good, lands, rewards = 0, [], [], []
        for r in chunk:
            x, ext, sp = _f(r["landing_x"]), _f(r["elbow_extension_deg"]), _f(r["release_speed_m_s"])
            rewards.append(_f(r["reward"]) or 0.0)
            if x is not None:
                lands.append(x)
            if (x is not None and target[0] <= x <= target[1]
                    and ext is not None and ext <= max_ext and sp is not None):
                good += 1
                speeds_good.append(sp * 3.6)
        out.append({
            "bucket": b,
            "timesteps_end": int(float(chunk[-1]["timesteps"])),
            "episodes": len(chunk),
            "good_pct": 100.0 * good / len(chunk),
            "mean_speed_good_kmh": stats.mean(speeds_good) if speeds_good else None,
            "mean_land_x": stats.mean(lands) if lands else None,
            "mean_reward": stats.mean(rewards),
        })
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csvs", nargs="+")
    parser.add_argument("--buckets", type=int, default=10)
    args = parser.parse_args()
    for c in args.csvs:
        c = Path(c)
        if not c.exists() or c.stat().st_size == 0:
            continue
        cur = curve(c, args.buckets)
        (c.parent / "curve.json").write_text(json.dumps(cur, indent=1))
        peak = max(cur, key=lambda r: r["good_pct"]) if cur else None
        if peak:
            print(f"{c.parent.name}: peak good {peak['good_pct']:.0f}% at bucket {peak['bucket']}, "
                  f"final {cur[-1]['good_pct']:.0f}%")


if __name__ == "__main__":
    main()
