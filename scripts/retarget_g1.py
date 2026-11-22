"""Retarget the captured delivery onto the Unitree G1 humanoid (assets/unitree_g1, MuJoCo Menagerie g1.xml, 29 DoF).

Supervisor, 20 Nov: "try an actual humanoid robot model". Until now the robot limits were only on the arm; the body
was the captured skeleton. This puts the whole delivery on the G1's own body: its link lengths, masses and joint
ranges.

Method: per-frame inverse kinematics (scipy least_squares, the G1's joint ranges as bounds, warm-started from the
previous frame). The robot is 1.32 m tall and the bowler about 1.8 m, so positions cannot be matched; segment
DIRECTIONS can. Residuals (all unit vectors in world axes):
  thighs (hip -> knee), shanks (knee -> ankle), feet (heel -> toe), upper arms (shoulder -> elbow), forearms
  (elbow -> wrist), hands (wrist -> grip point), and the torso's frame (its three axes)
plus a small pull towards the previous frame (continuity) and towards the wrist joints' zero.
The pelvis is not solved for: its orientation is the capture's (OpenSim body axes x forward, y up, z right ->
the G1's x forward, y left, z up) and its position the capture's scaled by the leg-length ratio (G1 / bowler, hip to
ankle). Timing is unchanged, so the robot's speeds are the bowler's times that ratio.

What the G1 cannot do shows up as residuals: e.g. its waist rolls and pitches only +/-30 deg (the bowler side-bends
~50 deg at release, scripts/body_vs_robot.py), so the torso direction error there is real, and the arms then point
the captured way from a more upright trunk.

Writes data/mocap/ref_G1_<delivery>.npz (qpos (n, 36), qvel (n, 35) forward differences, rate, release_index,
scale; participant data: gitignored) and results/kinematic/g1/retarget_<delivery>.json (per-segment angle errors).

    python scripts/retarget_g1.py --delivery T3aL
"""
import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import butter, filtfilt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from envs.humanoid_bowler import build_model, MOCAP_DIR          # noqa: E402
from envs.runup_bowl_env import GRIP                               # noqa: E402

G1_XML = ROOT / "assets" / "unitree_g1" / "g1.xml"
G1_GRIP = np.array([0.085, 0.0, 0.0])      # ball centre in right_wrist_yaw_link (the rubber hand, envs/g1_bowl_env)
# skeleton body axes (OpenSim: x fwd, y up, z right) -> G1 body axes (x fwd, y left, z up): R_g1 = R_skel @ M
M = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])

# (name, skeleton (from, to), G1 (from, to), weight); a point is a body name, or "grip"
SEGMENTS = []
for s, S in (("r", "right"), ("l", "left")):
    SEGMENTS += [
        (f"thigh_{s}", (f"femur_{s}", f"tibia_{s}"), (f"{S}_hip_roll_link", f"{S}_knee_link"), 1.0),
        (f"shank_{s}", (f"tibia_{s}", f"talus_{s}"), (f"{S}_knee_link", f"{S}_ankle_roll_link"), 1.0),
        (f"foot_{s}", (f"calcn_{s}", f"toes_{s}"), (f"{S}_ankle_roll_link", f"{S}_toe"), 0.5),
        (f"upper_arm_{s}", (f"humerus_{s}", f"ulna_{s}"), (f"{S}_shoulder_roll_link", f"{S}_elbow_link"), 1.0),
        (f"forearm_{s}", (f"ulna_{s}", f"hand_{s}"), (f"{S}_elbow_link", f"{S}_wrist_roll_link"), 1.0),
        (f"hand_{s}", (f"hand_{s}", f"grip_{s}"), (f"{S}_wrist_yaw_link", f"{S}_grip"), 0.3),
    ]
TORSO_W = 0.7
CONT_W = 0.15
WRIST_W = 0.02


def g1_model():
    """The G1 with its position actuators removed (kinematics only here) and toe/grip sites added."""
    spec = mujoco.MjSpec.from_file(str(G1_XML))
    for a in list(spec.actuators):
        spec.delete(a)
    for k in list(spec.keys):
        spec.delete(k)
    for S in ("right", "left"):
        spec.body(f"{S}_ankle_roll_link").add_site(name=f"{S}_toe", pos=[0.12, 0.0, -0.03])
        g = G1_GRIP if S == "right" else G1_GRIP * [1, -1, 1]
        spec.body(f"{S}_wrist_yaw_link").add_site(name=f"{S}_grip", pos=g)
    return spec.compile()


class Points:
    """World position of a body origin or a site, by name."""

    def __init__(self, m, d, extra):
        self.m, self.d, self.extra = m, d, extra

    def __call__(self, name):
        if name in self.extra:
            return self.extra[name](self.d)
        sid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid >= 0:
            return self.d.site_xpos[sid]
        return self.d.xpos[self.m.body(name).id]


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def leg_length(m, d, hip, ankle):
    mujoco.mj_kinematics(m, d)
    return float(np.linalg.norm(d.xpos[m.body(hip).id] - d.xpos[m.body(ankle).id]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delivery", default="T3aL")
    ap.add_argument("--cutoff-hz", type=float, default=10.0, help="low-pass of the joint angles (0 = off)")
    args = ap.parse_args()

    ref = np.load(MOCAP_DIR / f"ref_{args.delivery}.npz")
    sk = build_model(arm_scale=float(ref["arm_scale"]))
    assert sk.nq == ref["qpos"].shape[1], (sk.nq, ref["qpos"].shape)
    sd = mujoco.MjData(sk)

    def skel_grip(side):
        hid = sk.body(f"hand_{side}").id
        g = GRIP if side == "r" else GRIP * [1, 1, -1]
        return lambda d: d.xpos[hid] + d.xmat[hid].reshape(3, 3) @ g
    sp = Points(sk, sd, {"grip_r": skel_grip("r"), "grip_l": skel_grip("l")})

    g1 = g1_model()
    gd = mujoco.MjData(g1)
    gp = Points(g1, gd, {})
    jids = [j for j in range(g1.njnt) if g1.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
    jnames = [g1.joint(j).name for j in jids]
    qadr = np.array([g1.jnt_qposadr[j] for j in jids])
    lo, hi = g1.jnt_range[jids, 0].copy(), g1.jnt_range[jids, 1].copy()
    wrist = np.array(["wrist" in n for n in jnames])
    torso = g1.body("torso_link").id

    # leg-length ratio, both in their default (standing) poses
    mujoco.mj_resetData(sk, sd)
    sk_leg = leg_length(sk, sd, "femur_r", "talus_r")
    mujoco.mj_resetData(g1, gd)
    g1_leg = leg_length(g1, gd, "right_hip_roll_link", "right_ankle_roll_link")
    scale = g1_leg / sk_leg
    print(f"leg (hip-ankle): bowler {sk_leg:.3f} m, G1 {g1_leg:.3f} m, scale {scale:.3f}")

    n = len(ref["qpos"])
    out_q = np.zeros((n, g1.nq))
    err = {name: np.zeros(n) for name, *_ in SEGMENTS}
    err["torso"] = np.zeros(n)
    x_prev = np.clip(np.zeros(len(jids)), lo, hi)
    frames = []
    for i in range(n):
        sd.qpos[:] = ref["qpos"][i]
        mujoco.mj_kinematics(sk, sd)
        targets = [(name, unit(sp(b) - sp(a)), (ga, gb), w) for name, (a, b), (ga, gb), w in SEGMENTS]
        R_pel = sd.xmat[sk.body("pelvis").id].reshape(3, 3) @ M
        R_tor = sd.xmat[sk.body("torso").id].reshape(3, 3) @ M
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, R_pel.flatten())
        root = np.concatenate([sd.xpos[sk.body("pelvis").id] * scale, quat])

        def fk(x):
            gd.qpos[:7] = root
            gd.qpos[qadr] = x
            mujoco.mj_kinematics(g1, gd)

        def resid(x):
            fk(x)
            r = []
            for name, u, (ga, gb), w in targets:
                r.append(w * (unit(gp(gb) - gp(ga)) - u))
            r.append(TORSO_W * (gd.xmat[torso].reshape(3, 3) - R_tor).flatten())
            r.append(CONT_W * (x - x_prev))
            r.append(WRIST_W * x[wrist])
            return np.concatenate(r)

        sol = least_squares(resid, np.clip(x_prev, lo, hi), bounds=(lo, hi), method="trf", xtol=1e-8, ftol=1e-8,
                            max_nfev=400 if i else 3000)
        x_prev = sol.x
        out_q[i, :7], out_q[i, qadr] = root, sol.x
        frames.append((targets, R_tor))

    # zero-phase low-pass of the joint angles: near a straight elbow the shoulder's twist is barely determined and
    # the per-frame solution can flip branches (126 rad/s shoulder-yaw spikes unfiltered), which no motor follows
    rate = float(ref["rate"])
    if args.cutoff_hz > 0:
        b, a = butter(4, args.cutoff_hz / (rate / 2.0))
        out_q[:, qadr] = np.clip(filtfilt(b, a, out_q[:, qadr], axis=0), lo, hi)
    for i, (targets, R_tor) in enumerate(frames):
        gd.qpos[:] = out_q[i]
        mujoco.mj_kinematics(g1, gd)
        for name, u, (ga, gb), w in targets:
            err[name][i] = np.degrees(np.arccos(np.clip(unit(gp(gb) - gp(ga)) @ u, -1.0, 1.0)))
        Rg = gd.xmat[torso].reshape(3, 3)
        err["torso"][i] = np.degrees(np.arccos(np.clip((np.trace(Rg.T @ R_tor) - 1.0) / 2.0, -1.0, 1.0)))

    qvel = np.zeros((n, g1.nv))
    for i in range(n - 1):
        mujoco.mj_differentiatePos(g1, qvel[i], 1.0 / rate, out_q[i], out_q[i + 1])
    qvel[-1] = qvel[-2]
    ri = int(ref["release_index"])
    dst = MOCAP_DIR / f"ref_G1_{args.delivery}.npz"
    np.savez(dst, qpos=out_q, qvel=qvel, rate=rate, release_index=ri, nq=g1.nq, delivery=f"G1_{args.delivery}",
             scale=scale, joint_names=np.array(jnames), **{f"err_{k}": v for k, v in err.items()})
    print("wrote", dst)

    # report: per-segment direction error (deg), whole delivery and around release
    win = slice(max(ri - 30, 0), min(ri + 10, n))
    rep = {"delivery": args.delivery, "scale": scale, "frames": n, "release_index": ri,
           "bowler_leg_m": sk_leg, "g1_leg_m": g1_leg,
           "segment_error_deg": {k: {"median": float(np.median(v)), "p95": float(np.percentile(v, 95)),
                                     "release_window_max": float(v[win].max()), "at_release": float(v[ri])}
                                 for k, v in err.items()},
           "joints_at_limit_frac": {nm: float(np.mean((np.abs(out_q[:, a] - l) < 1e-3) | (np.abs(out_q[:, a] - h) < 1e-3)))
                                    for nm, a, l, h in zip(jnames, qadr, lo, hi)}}
    rep["joints_at_limit_frac"] = {k: v for k, v in rep["joints_at_limit_frac"].items() if v > 0}
    od = ROOT / "results" / "kinematic" / "g1"
    od.mkdir(parents=True, exist_ok=True)
    (od / f"retarget_{args.delivery}.json").write_text(json.dumps(rep, indent=1))
    for k, v in rep["segment_error_deg"].items():
        print(f"  {k:12s} median {v['median']:5.1f}  p95 {v['p95']:5.1f}  release-window max {v['release_window_max']:5.1f}"
              f"  at release {v['at_release']:5.1f}")
    print("  at a joint limit:", {k: round(v, 2) for k, v in rep["joints_at_limit_frac"].items()})


if __name__ == "__main__":
    main()
