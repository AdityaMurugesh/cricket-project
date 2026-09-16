"""Summarise a training run's episodes.csv into the numbers that matter.

The training log's own reward/loss curves say whether PPO is optimising;
they say nothing about whether the thing it found is a bowling action.
That takes release geometry, which is why throw_env.py logs
elbow_at_release_deg and release_height_m. run6 looked healthy in the SB3
console output while releasing the ball from behind its back at chest
height.

Speeds are reported in km/h throughout -- see the speed-units memory.
"""
import argparse
import csv
import statistics as stats
from pathlib import Path


def _f(row, key):
    v = row.get(key)
    if v is None or v == "":
        return None
    return float(v)


def summarise(rows, max_legal_extension_deg, target_range):
    n = len(rows)
    released = [r for r in rows if _f(r, "release_speed_m_s") is not None]
    landed = [r for r in released if _f(r, "landing_x") is not None]
    legal = [r for r in released
             if _f(r, "elbow_extension_deg") is not None
             and _f(r, "elbow_extension_deg") <= max_legal_extension_deg]
    lo, hi = target_range
    on_target = [r for r in landed if lo <= _f(r, "landing_x") <= hi]
    # the actual research metric: speed among throws that were BOTH legal
    # and accurate, since that is the "subject to" in the research question
    good = [r for r in on_target
            if _f(r, "elbow_extension_deg") is not None
            and _f(r, "elbow_extension_deg") <= max_legal_extension_deg]

    def mean(items, key, scale=1.0):
        vals = [_f(r, key) for r in items]
        vals = [v * scale for v in vals if v is not None]
        return stats.mean(vals) if vals else float("nan")

    def sd(items, key):
        vals = [_f(r, key) for r in items]
        vals = [v for v in vals if v is not None]
        return stats.pstdev(vals) if len(vals) > 1 else float("nan")

    return {
        "episodes": n,
        "released_pct": 100.0 * len(released) / n if n else 0.0,
        "legal_pct": 100.0 * len(legal) / len(released) if released else 0.0,
        "on_target_pct": 100.0 * len(on_target) / len(released) if released else 0.0,
        "legal_and_on_target_pct": 100.0 * len(good) / len(released) if released else 0.0,
        "mean_speed_kmh": mean(released, "release_speed_m_s", 3.6),
        "best_speed_kmh": max([_f(r, "release_speed_m_s") * 3.6 for r in good], default=float("nan")),
        "mean_speed_good_kmh": mean(good, "release_speed_m_s", 3.6),
        "mean_reward": mean(rows, "reward"),
        # the two release-geometry numbers that separate a bowl from a sling
        "mean_release_shoulder_deg": mean(released, "shoulder_at_release_deg"),
        "mean_release_elbow_deg": mean(released, "elbow_at_release_deg"),
        "mean_release_height_m": mean(released, "release_height_m"),
        # what the policy asked for, averaged, plus how tightly it holds to
        # it. A large sd here means release timing is still being decided by
        # exploration noise rather than by the policy.
        "mean_release_target_deg": mean(released, "release_target_deg"),
        "sd_release_target_deg": sd(released, "release_target_deg"),
    }


# (key, column header, format). Headers are abbreviated so a run fits one
# terminal line: rel% = released, legal% is of released throws, zone% is
# landed in the target zone, good% is legal AND in the zone -- the actual
# success rate. kmh/good and kmh/best are over good throws only, since
# speed on an illegal or wayward delivery is not a result.
FIELDS = [
    ("episodes", "eps", "{:.0f}"),
    ("released_pct", "rel%", "{:.0f}"),
    ("legal_pct", "legal%", "{:.0f}"),
    ("on_target_pct", "zone%", "{:.0f}"),
    ("legal_and_on_target_pct", "good%", "{:.0f}"),
    ("mean_speed_kmh", "kmh/all", "{:.1f}"),
    ("mean_speed_good_kmh", "kmh/good", "{:.1f}"),
    ("best_speed_kmh", "kmh/best", "{:.1f}"),
    ("mean_reward", "reward", "{:.2f}"),
    ("mean_release_shoulder_deg", "rel@sh", "{:.0f}"),
    ("mean_release_elbow_deg", "rel@el", "{:.0f}"),
    ("mean_release_height_m", "rel@z", "{:.2f}"),
    ("mean_release_target_deg", "want@sh", "{:.0f}"),
    ("sd_release_target_deg", "want:sd", "{:.1f}"),
]


def run(csv_path, buckets, max_legal_extension_deg, target_range):
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        print("no episodes logged")
        return
    size = max(1, len(rows) // buckets)
    chunks = [("all", rows)]
    chunks += [(f"ep {i}-{min(i + size, len(rows))}", rows[i:i + size])
               for i in range(0, len(rows), size)]

    header = f"{'segment':>14} " + " ".join(f"{h:>8}" for _, h, _ in FIELDS)
    print(header)
    print("-" * len(header))
    for label, chunk in chunks:
        s = summarise(chunk, max_legal_extension_deg, target_range)
        cells = " ".join(f"{fmt.format(s[k]):>8}" for k, _, fmt in FIELDS)
        print(f"{label:>14} {cells}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="path to a run's episodes.csv")
    parser.add_argument("--buckets", type=int, default=8,
                        help="split the run into this many equal segments to show progress")
    parser.add_argument("--max-legal-extension", type=float, default=15.0)
    parser.add_argument("--target-range", type=float, nargs=2, default=[6.0, 8.0])
    args = parser.parse_args()
    run(Path(args.csv_path), args.buckets, args.max_legal_extension, args.target_range)
