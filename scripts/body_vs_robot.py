"""Could a humanoid robot move its legs and trunk like the captured bowler? A kinematic check of the reference body
motion (data/mocap/ref_<delivery>.npz, the body every kinematic result rides on) against the joint ranges and joint
speed limits of the Unitree G1 (29-DoF) and H1, from their URDFs (unitree_ros: g1_29dof.urdf, h1.urdf <limit>).
The arm has its own limit profiles (envs/arm_limits.py); this is the rest of the body. Torques are NOT checked
(that needs the ground forces; and the robots are smaller and lighter than the bowler).

How the angles are compared. Every body frame of the skeleton has OpenSim axes in the default pose (x forward, y up,
z right; mocap/skeleton3d.py); the robots' are x forward, y left, z up. Each hip and lumbar ball joint's rotation is
expressed in robot axes and split in the robot's own joint order:
  G1 hip    pitch (y), roll (x), yaw (z)        H1 hip    yaw, roll, pitch
  G1 waist  yaw, roll, pitch                    H1 torso  yaw only (no trunk roll or pitch joint)
Knee: robot flexion = -knee_angle (the skeleton's knee is negative in flexion). Ankle pitch: robot = -ankle_angle
(positive = toes down on both robots). G1 ankle roll ~ the subtalar angle (its axis is oblique: approximate; the
range is symmetric, so only its size matters). The robots' signs: hip pitch negative = flexion; hip roll is
adduction-positive on the right, abduction-positive on the left (the URDF ranges are mirrored to match).
Speeds are central differences at the capture rate (100 Hz) of each robot joint coordinate.

Windows (s from release): the run-up's end and the bound, -1.0 to -0.6 (leg markers measured), and the delivery
stride, -0.6 to +0.3 (legs reconstructed after landing). A coordinate beyond the normal human range too (HUMAN_ROM)
is flagged: that is the fit, not the robot (e.g. the ~80 deg hip rotation in the bound, where the pelvis markers are
poor).

Caveats: the legs after -0.60 s are reconstructed (no leg markers: scripts/fix_legs.py) and the pelvis after -0.21 s
is the fit's; the capture rate limits the peak speeds it can show.

    python scripts/body_vs_robot.py [--delivery T3aL]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# URDF <limit lower upper (deg) velocity (rad/s) effort (N m)>, unitree_ros master, fetched 21 Nov 2026
LIMITS = {
    "G1": {"hip_pitch": (-145.0, 165.0, 32, 88), "hip_roll_r": (-170.0, 30.0, 32, 88),
           "hip_roll_l": (-30.0, 170.0, 32, 88), "hip_yaw": (-158.0, 158.0, 32, 88),
           "knee": (-5.0, 165.0, 20, 139), "ankle_pitch": (-50.0, 30.0, 30, 35), "ankle_roll": (-15.0, 15.0, 30, 35),
           "waist_yaw": (-150.0, 150.0, 32, 88), "waist_roll": (-29.8, 29.8, 30, 35),
           "waist_pitch": (-29.8, 29.8, 30, 35)},
    "H1": {"hip_pitch": (-179.9, 145.0, 23, 200), "hip_roll_r": (-24.6, 24.6, 23, 200),
           "hip_roll_l": (-24.6, 24.6, 23, 200), "hip_yaw": (-24.6, 24.6, 23, 200),
           "knee": (-14.9, 117.5, 14, 300), "ankle_pitch": (-49.8, 29.8, 9, 40), "ankle_roll": (0.0, 0.0, 0, 0),
           "waist_yaw": (-134.6, 134.6, 23, 200), "waist_roll": (0.0, 0.0, 0, 0), "waist_pitch": (0.0, 0.0, 0, 0)},
}
# normal adult range of motion (deg, approximate textbook values), to flag what is beyond a human too
HUMAN_ROM = {"hip_yaw": 50.0, "knee": (-10.0, 150.0), "ankle_pitch": (-40.0, 55.0)}
# OpenSim axes -> robot axes: x_r = x, y_r = -z, z_r = y (rows: robot axes in OpenSim coordinates)
T = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
ORDER = {("G1", "hip"): ("YXZ", ("pitch", "roll", "yaw")), ("H1", "hip"): ("ZXY", ("yaw", "roll", "pitch")),
         ("G1", "waist"): ("ZXY", ("yaw", "roll", "pitch")), ("H1", "waist"): ("ZXY", ("yaw", "roll", "pitch"))}


def ball_angles(q_wxyz, robot, part):
    """(n, 4) MuJoCo quaternions (child in parent, OpenSim axes) -> {coordinate: degrees (n,)} in the robot's chain."""
    R = Rotation.from_quat(q_wxyz[:, [1, 2, 3, 0]]).as_matrix()
    Rr = np.einsum("ij,njk,lk->nil", T, R, T)
    seq, names = ORDER[(robot, part)]
    e = np.unwrap(Rotation.from_matrix(Rr).as_euler(seq), axis=0)
    return {n: np.degrees(e[:, i]) for i, n in enumerate(names)}


def coords(qpos, m, robot):
    adr = lambda name: m.jnt_qposadr[m.joint(name).id]
    out = {}
    for s in ("r", "l"):
        h = ball_angles(qpos[:, adr(f"hip_{s}"):adr(f"hip_{s}") + 4], robot, "hip")
        out[f"hip_pitch_{s}"], out[f"hip_roll_{s}"], out[f"hip_yaw_{s}"] = h["pitch"], h["roll"], h["yaw"]
        out[f"knee_{s}"] = -np.degrees(qpos[:, adr(f"knee_angle_{s}")])
        out[f"ankle_pitch_{s}"] = -np.degrees(qpos[:, adr(f"ankle_angle_{s}")])
        out[f"ankle_roll_{s}"] = np.degrees(qpos[:, adr(f"subtalar_angle_{s}")])
    w = ball_angles(qpos[:, adr("lumbar"):adr("lumbar") + 4], robot, "waist")
    out["waist_yaw"], out["waist_roll"], out["waist_pitch"] = w["yaw"], w["roll"], w["pitch"]
    return out


def limit_key(name):
    if name.startswith("hip_roll"):
        return name
    return name[:-2] if name[-2:] in ("_r", "_l") else name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delivery", default="T3aL")
    ap.add_argument("--windows", nargs="+", default=["-1.0:-0.6", "-0.6:0.3"],
                    help="from:to, s from release (default: the bound with measured legs; the delivery stride)")
    ap.add_argument("--out", default=str(ROOT / "results/kinematic/body_vs_robot.json"))
    a = ap.parse_args()
    from envs.kinematic_bowl_env import KinematicBowlEnv
    env = KinematicBowlEnv(delivery=a.delivery, arm_control="torque")
    ref = np.load(ROOT / "data/mocap" / f"ref_{a.delivery}.npz", allow_pickle=True)
    qpos, rate, rel = ref["qpos"], float(ref["rate"]), int(ref["release_index"])
    t = (np.arange(len(qpos)) - rel) / rate
    res = {"delivery": a.delivery, "rate_hz": rate, "windows": {}}
    for spec in a.windows:
        f0, f1 = map(float, spec.split(":"))
        win = (t >= f0 - 1e-9) & (t <= f1 + 1e-9)
        res["windows"][spec] = report(qpos, env.m, rate, win, f"{a.delivery}, {f0:+.2f} to {f1:+.2f} s from release")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


def beyond_human(name, lo, hi):
    k = limit_key(name)
    if k == "hip_yaw":
        return max(abs(lo), abs(hi)) > HUMAN_ROM[k]
    if k in HUMAN_ROM:
        return lo < HUMAN_ROM[k][0] or hi > HUMAN_ROM[k][1]
    return False


def report(qpos, m, rate, win, title):
    out = {}
    for robot in ("G1", "H1"):
        L = LIMITS[robot]
        rows = {}
        for name, ang in coords(qpos, m, robot).items():
            lo, hi, vmax, eff = L[limit_key(name)]
            vel = np.gradient(np.radians(ang), 1.0 / rate)
            aw, vw = ang[win], np.abs(vel[win])
            out_rng = (aw < lo - 1e-6) | (aw > hi + 1e-6)
            rows[name] = {"min_deg": float(aw.min()), "max_deg": float(aw.max()), "range_deg": [lo, hi],
                          "peak_speed_rad_s": float(vw.max()), "speed_limit_rad_s": vmax,
                          "frac_out_of_range": float(out_rng.mean()), "frac_over_speed": float((vw > vmax).mean()),
                          "no_joint": vmax == 0, "beyond_human": bool(beyond_human(name, aw.min(), aw.max()))}
        out[robot] = rows
        print(f"== {robot}  ({title})")
        print(f"   {'joint':14s} {'human min..max (deg)':>22s} {'robot range':>16s} {'out':>5s}   "
              f"{'peak speed':>10s} {'limit':>6s} {'over':>5s}")
        for name, r in rows.items():
            if r["no_joint"]:
                print(f"   {name:14s} {r['min_deg']:9.1f} .. {r['max_deg']:6.1f}     (no such joint: must stay 0)")
                continue
            print(f"   {name:14s} {r['min_deg']:9.1f} .. {r['max_deg']:6.1f}   {r['range_deg'][0]:6.1f}..{r['range_deg'][1]:6.1f}"
                  f" {100 * r['frac_out_of_range']:4.0f}%   {r['peak_speed_rad_s']:7.1f}    {r['speed_limit_rad_s']:4.0f}"
                  f" {100 * r['frac_over_speed']:4.0f}%" + ("   beyond a human too: the fit" if r["beyond_human"] else ""))
    return out


if __name__ == "__main__":
    main()
