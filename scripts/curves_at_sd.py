"""Mean good rate of a scored bowler under release-timing SDs other than the scorer's 12 ms, from its
saved landing curves (kin_equifinality *_curves.json, noise-free starts), with the curve interpolated between
2 ms samples (kin_equifinality.p_good_continuous).

  python scripts/curves_at_sd.py "kin2_tolr*" --sd-ms 1 2 4 8 12
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.kin_equifinality import p_good_continuous  # noqa: E402

H = 0.002          # physics step (assets/bowler.xml)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pattern")
    p.add_argument("--sd-ms", type=float, nargs="+", default=[1, 2, 4, 8, 12, 16])
    p.add_argument("--lo", type=float, default=10.9)
    p.add_argument("--hi", type=float, default=12.9)
    p.add_argument("--line", type=float, default=0.5)
    a = p.parse_args()
    out = {}
    for f in sorted(glob.glob(str(ROOT / "results/kinematic/equifinality" / f"{a.pattern}_curves.json"))):
        run = Path(f).name[: -len("_curves.json")]
        d = json.load(open(f))
        curves = [np.array(v, dtype=float) for k, v in d.items() if k.startswith("none_") and v]
        if not curves:
            continue
        res = {}
        for sd in a.sd_ms:
            ps = []
            for c in curves:
                off, x, y, good = c[:, 0] * H, c[:, 1], c[:, 2], c[:, 3] > 0.5
                inband = (a.lo <= x) & (x <= a.hi) & (np.abs(y) <= a.line)
                ok = good | ~inband          # a sample fails on legality/window/plane only if in band yet not good
                ps.append(p_good_continuous(off, x, y, ok, a.lo, a.hi, a.line, sd / 1000.0))
            res[f"{sd:g}"] = float(np.mean(ps))
        speed = float(np.mean([c[np.argmin(np.abs(c[:, 0])), 4] for c in curves]) * 3.6)
        out[run] = {"speed_kmh": speed, "n": len(curves), "p_cont": res}
        print(f"{run:46s} {speed:6.1f} km/h  " + "  ".join(f"p{k}={v:.2f}" for k, v in res.items()))
    if out:
        json.dump(out, open(ROOT / "results/kinematic/equifinality" / "curves_at_sd.json", "w"), indent=1)


if __name__ == "__main__":
    main()
