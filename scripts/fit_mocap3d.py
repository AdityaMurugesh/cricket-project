"""Fit the full-body 3D skeleton to every captured delivery (mocap/fit3d.py).

Slow (a few minutes per delivery), so it is separate from mocap_process.py,
which only summarises its output in REPORT.md. The participant's skeleton
scale is found once, on the calibration deliveries (the two with the most
complete bowling arm), and then held fixed for every delivery.

    python scripts/fit_mocap3d.py                     # all deliveries
    python scripts/fit_mocap3d.py --deliveries T3a    # some
    python scripts/fit_mocap3d.py --scale 1.02        # skip the scale search

Writes data/mocap/fit3d_<D>.npz; replay with scripts/demo_mocap3d.py.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mocap import c3dio, delivery as dl, fit3d, runup, tracks as tk  # noqa: E402

OUT = ROOT / "data" / "mocap"
LABELS = OUT / "labels.json"     # {"T3a": {"18": "left elbow", ...}}: identities given by hand
# body-part words -> bones they may sit on ("left"/"right" picks the side)
PART_BONES = {"head": ("head",), "forehead": ("head",), "temple": ("head",),
              "neck": ("torso",), "chest": ("torso",), "sternum": ("torso",), "back": ("torso",),
              "spine": ("torso",), "shoulder": ("torso",),
              "pelvis": ("pelvis",), "hip": ("pelvis",), "waist": ("pelvis",), "belt": ("pelvis",),
              "upper arm": ("humerus",), "elbow": ("humerus",), "inner elbow": ("humerus",),
              "forearm": ("ulna", "radius"),
              "wrist": ("radius", "ulna"), "hand": ("hand",),
              "thigh": ("femur",), "knee": ("femur", "tibia"), "shin": ("tibia",), "ankle": ("tibia",),
              "foot": ("calcn", "toes"), "heel": ("calcn",), "toe": ("calcn", "toes")}


def label_bones(text):
    """'left elbow' -> ('humerus_l',); 'head' -> ('head',); None if not understood."""
    t = text.lower().strip()
    side = "l" if t.startswith("left") else "r" if t.startswith("right") else None
    part = t.replace("left", "").replace("right", "").strip()
    bones = PART_BONES.get(part)
    if bones is None:
        return None
    return tuple(b if b in ("head", "torso", "pelvis") else f"{b}_{side or 'r'}" for b in bones)         if side or all(b in ("head", "torso", "pelvis") for b in bones) else None


# the IOR landmark a labelled part's marker is on: its attachment point is kept
# near that one (fit3d.SITE_LABEL_M), which also fixes the upper arm's roll.
# Wrist: each forearm bone has one landmark (styloid) anyway. "inner elbow" is
# the medial epicondyle: not in the AIM model, but camera 3's static-trial frame
# shows a marker on the inside of each elbow. Label a track so only if it sits
# level with the lateral elbow marker (on the hinge axis), ~11 cm across.
PART_SITE = {"shoulder": "SAE", "upper arm": "HUM", "elbow": "HLE", "inner elbow": "HME", "hand": "HM2"}


def label_site(text):
    """'left upper arm' -> 'L_HUM'; '' where the part has no single landmark."""
    t = text.lower().strip()
    side = "L" if t.startswith("left") else "R" if t.startswith("right") else None
    part = t.replace("left", "").replace("right", "").strip()
    return f"{side}_{PART_SITE[part]}" if side and part in PART_SITE else ""


def manual_pins(name):
    """{tid: {"bones": (...), "site": landmark}} from LABELS."""
    if not LABELS.exists():
        return {}
    import json
    out = {}
    for tid, text in json.loads(LABELS.read_text()).get(name, {}).items():
        bones = label_bones(text)
        if bones:
            out[int(tid)] = {"bones": bones, "site": label_site(text)}
    return out
CALIBRATION = ("T3a", "T4b")
POST_S = 0.4     # fitted after release
RETRY_FRACTION = 0.6   # a fit using fewer of the visible markers than this is retried (best_fit)
STAND_S = 0.3    # fitted before the run starts


def prepare(name, trials):
    """Everything fit3d.fit needs for one delivery."""
    _, n, r = next(d for d in dl.DELIVERIES if d[0] == name)
    T, trs = trials[n]
    res = dl.analyse(name, T, trs, r)
    usable = res.metrics.get("release_quality") in ("good", "lower bound (lost after peak)")
    rel, src = (res.release, "hand-speed peak") if usable else (r, "video")
    run = runup.analyse(name, T, trs, rel, res.direction, src)
    first = max(run.frame0, run.start - int(STAND_S * T.rate))
    last = min(T.n - 3, rel + int(POST_S * T.rate))
    # identities established by physics around release (mocap/identify.py),
    # only where the release is captured: elsewhere the bowling hand is lost
    # before release and the fastest point identify.py finds can be the front
    # (left) arm pulling down -- in T5a its "bowling wrist" is on the left side
    pins = {}
    if usable:
        for kind in ("hand", "wrist", "elbow", "proximal"):
            for tid in getattr(res.arm, kind):
                pins[tid] = kind
        if res.spine and res.spine.ids:
            for tid in res.spine.ids:
                pins[tid] = "spine"
    pins.update(manual_pins(name))   # labels given by hand win
    return dict(name=name, trial=T, tracks=trs, first=first, last=last, release=rel,
                direction=res.direction, x_offset=float(run.x_lab[run.i(rel)]),
                z_centre=float(np.nanmedian(T.pos[rel, :, 2])), run=run, pins=pins,
                release_source=src, hand_tracked=bool(usable))


def measure_arm_scale(trials, scale):
    """The participant's arms relative to the skeleton's: forearm length (elbow
    marker on the lateral epicondyle to the midpoint of the wrist styloid
    markers) in the deliveries whose arm was identified and checked on video,
    over the same distance on the skeleton. A model arm too long (0.274 vs
    ~0.25 m) can only put its hand on the hand marker by bending the elbow."""
    import mujoco
    lengths = []
    for name, n, r in dl.DELIVERIES:
        T, trs = trials[n]
        res = dl.analyse(name, T, trs, r)
        if not res.release_usable or not res.arm.elbow or len(res.arm.wrist) < 2:
            continue
        by = {t.tid: t for t in trs}
        for f in range(res.frame0, res.frame0 + 101):
            pts = [by[i].at(f) for i in res.arm.elbow[:1] + res.arm.wrist[:2]]
            if all(q is not None for q in pts):
                lengths.append(np.linalg.norm(pts[0] - (pts[1] + pts[2]) / 2))
    m = mujoco.MjModel.from_xml_string(fit3d.skeleton3d.build_xml(scale))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    P = lambda s: d.site_xpos[m.site(s).id]
    model = np.linalg.norm(P("m_R_HLE") - (P("m_R_RSP") + P("m_R_USP")) / 2)
    return float(np.median(lengths) / model), float(np.median(lengths))


def limb_templates(arm_scale, log=print):
    """Per limb, its mean fitted pose at each instant from release (its
    fit3d.TEMPLATE_S window) over the fits on disk where it was tracked then:
    an arm with a marker on its forearm or hand (and, for the front arm, on the
    upper arm); a leg with markers on two of thigh, shin and foot, one below
    the knee. The bowling arm comes from the deliveries with a tracked release
    only. {"arm_r": (rows from release, qpos values of fit3d.LIMB_JOINTS), ...}.
    The delivery is the same movement every time, so where a limb's markers are
    lost this is a far better guess than a hanging arm or a stiff leg -- still a
    guess, drawn translucent like every bone without a marker."""
    sk = fit3d.Skeleton(1.0, arm_scale)
    out = {}
    for limb in ("arm", "leg"):
        lo, hi = (int(round(x * 100)) for x in fit3d.TEMPLATE_S[limb])
        for side in ("r", "l"):
            qidx = fit3d.limb_dofs(sk.m, limb, side)[0]
            quat = slice(3, 7) if limb == "arm" else slice(0, 4)   # the ball joint's quaternion
            samples, used = {}, []
            for name, *_ in dl.DELIVERIES:
                path = OUT / f"fit3d_{name}.npz"
                if not path.exists():
                    continue
                z = np.load(path)
                if limb == "arm" and side == "r" and not bool(z["hand_tracked"]):
                    continue
                bodies = list(z["body_names"])
                sup = z["support"]
                col = lambda b: sup[:, bodies.index(f"{b}_{side}")]
                if limb == "arm":
                    ok = (col("ulna") | col("radius") | col("hand")) & (col("humerus") | (side == "r"))
                else:
                    low = col("tibia") | col("calcn") | col("toes")
                    ok = low & ((col("tibia").astype(int) + col("calcn") + col("toes") + col("femur")) >= 2)
                k, Q = int(z["release_index"]), z["qpos"]
                if len(Q[0]) != sk.m.nq:
                    continue
                rows = [rel for rel in range(lo, hi + 1) if 0 <= k + rel < len(Q) and ok[k + rel]]
                if rows:
                    used.append(name)
                for rel in rows:
                    samples.setdefault(rel, []).append(Q[k + rel, qidx])
            rels, vals = sorted(samples), []
            for rel in rels:
                V = np.array(samples[rel])
                qq = V[:, quat] * np.sign(V[:, quat] @ V[0, quat])[:, None]
                v = V.mean(0)
                v[quat] = qq.mean(0) / np.linalg.norm(qq.mean(0))
                vals.append(v)
            out[f"{limb}_{side}"] = (np.array(rels, int), np.array(vals))
            nn = [len(samples[r]) for r in rels]
            log(f"{limb} template {side}: {len(rels)} rows from {used}, "
                f"{min(nn) if nn else 0}-{max(nn) if nn else 0} deliveries per row")
    np.savez_compressed(OUT / "limb_template.npz",
                        **{f"{key}_{k}": v for key, (r, qv) in out.items() for k, v in (("rows", r), ("qpos", qv))})
    return {k: v for k, v in out.items() if len(v[0])}


def run_fit(p, scale, log, judge_runup=True, arm_scale=1.0, template=None):
    t0 = time.time()
    fr, wtracks = fit3d.fit(p["name"], p["trial"], p["tracks"], p["first"], p["last"], p["release"],
                            p["direction"], p["x_offset"], p["z_centre"], p["run"], p["pins"],
                            scale=scale, judge_runup=judge_runup, arm_scale=arm_scale,
                            template=template, log=log)
    log(f"  [{p['name']}] {time.time() - t0:.0f} s")
    return fr, wtracks


def best_fit(p, scale, log, arm_scale=1.0, template=None):
    """The first match is judged on the run-up frames only (the upright guess is
    good there and wrong in the delivery). That rescues some deliveries and costs
    others, so where the fit uses under RETRY_FRACTION of the visible markers it
    is re-run judging whole tracks, and the fit that drives the skeleton with
    more markers is kept (fewer markers at an equal residual explain less)."""
    fr, wtracks = run_fit(p, scale, log, arm_scale=arm_scale, template=template)
    T = p["trial"]
    visible = np.median(np.isfinite(T.pos[p["first"]:p["last"] + 1, :, 0]).sum(1))
    if np.median(fr.n_obs) < RETRY_FRACTION * visible:
        log(f"  [{p['name']}] {np.median(fr.n_obs):.0f} of {visible:.0f} visible markers used: "
            "retrying with the first match judged on whole tracks")
        fr2, wt2 = run_fit(p, scale, log, judge_runup=False, arm_scale=arm_scale, template=template)
        if np.median(fr2.n_obs) > np.median(fr.n_obs):
            log(f"  [{p['name']}] kept the retry ({np.median(fr2.n_obs):.0f} markers per frame)")
            return fr2, wt2
        log(f"  [{p['name']}] kept the first fit")
    return fr, wtracks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deliveries", default=",".join(d[0] for d in dl.DELIVERIES))
    ap.add_argument("--scale", type=float, default=None)
    ap.add_argument("--arm-scale", type=float, default=None)
    ap.add_argument("--no-template", action="store_true",
                    help="do not hold untracked limbs to the mean of the tracked deliveries")
    args = ap.parse_args()
    names = args.deliveries.split(",")
    logf = open(OUT / "fit3d.log", "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    trials = {}
    for p in c3dio.dynamic_trials():
        T = c3dio.load(p)
        trials[int(p.stem.split()[-1])] = (T, tk.build_tracks(T))
    scale = args.scale
    if scale is None:
        found = []
        for name in CALIBRATION:
            fr, _ = run_fit(prepare(name, trials), None, log)
            found.append(fr.scale)
        scale = float(np.median(found))
        log(f"participant scale {scale:.3f} (from {dict(zip(CALIBRATION, found))})")
    arm_scale = args.arm_scale
    if arm_scale is None:
        arm_scale, forearm = measure_arm_scale(trials, scale)
        log(f"arm scale {arm_scale:.3f} (forearm {forearm:.3f} m from the markers)")
    # from the fits on disk before this run replaces them (run twice from scratch)
    template = None if args.no_template else limb_templates(arm_scale, log)
    for name in names:
        p = prepare(name, trials)
        fr, wtracks = best_fit(p, scale, log, arm_scale, template)
        fit3d.save(OUT / f"fit3d_{name}.npz", fr, wtracks, p["trial"], p["direction"], p["x_offset"],
                   p["z_centre"], release_source=p["release_source"], hand_tracked=p["hand_tracked"],
                   pinned=np.array(sorted(p["pins"]), int), arm_scale=arm_scale)
    log("done")


if __name__ == "__main__":
    main()
