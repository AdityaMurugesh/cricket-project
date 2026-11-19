"""Re-score frontier optima under several release-timing SDs (pace vs the bowler's timing skill).

Each optimum in the given frontier JSONs is replayed once per SD in --sd-ms; "p" is the
2 ms-grid good rate, "p_cont" the interpolated one (kin_equifinality.p_good_continuous). That gives the whole skill curve for each delivery, so one can check that a frontier found for one SD
is not beaten, at that SD, by an optimum found for another.

  python scripts/frontier_cross_sigma.py results/kinematic/frontier/frontier_a290_k20*.json --out results/kinematic/frontier/cross_sigma_a290.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.kin_equifinality as ke  # noqa: E402
import scripts.kin_frontier as kf  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("jsons", nargs="+")
    p.add_argument("--sd-ms", type=float, nargs="+", default=[1, 2, 4, 8, 12, 16])
    p.add_argument("--out", required=True)
    a = p.parse_args()
    rows = []
    for f in a.jsons:
        for rec in json.load(open(f)):
            if rec.get("handover_s") is not None or rec.get("env2") or not rec.get("released", True):
                continue
            kf.KNOTS = np.array(rec["knot_times_s"])
            kf._init(rec["angle"], (10.9, 12.9), None, None, None, 12.0)
            x = np.array(rec["knots"]).ravel()
            ps, pc = {}, {}
            for sd in a.sd_ms:
                ke.JITTER_SD_S = sd / 1000.0
                r = kf.evaluate(x)
                ps[f"{sd:g}"] = r.get("p_jitter_grid", r.get("p_jitter_12ms")) or 0.0
                pc[f"{sd:g}"] = r.get("p_jitter_cont") or 0.0
            row = {"file": Path(f).name, "angle": rec["angle"], "tau": rec["tau"],
                   "found_at_sd_ms": rec.get("jitter_sd_ms", 12.0), "speed_kmh": (r.get("speed_m_s") or 0) * 3.6,
                   "good": r.get("good"), "p": ps, "p_cont": pc}
            rows.append(row)
            print(f"{row['file']:34s} tau {row['tau']:.2f} {row['speed_kmh']:6.1f} km/h good {row['good']} "
                  + " ".join(f"p{k}={v:.2f}/{pc[k]:.2f}" for k, v in ps.items()), flush=True)
    json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
