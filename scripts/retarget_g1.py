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

--feet (27 Nov, for the dynamic whole-body G1, scripts/g1_dynamic.py): the plain retarget's feet sink up to 11 cm
into the floor and its planted foot slides at ~1.4 m/s, because the pelvis path is the bowler's scaled and never
looks at the ground. With --feet the IK also solves the pelvis position (pulled weakly to the scaled capture), keeps
every foot sphere above the floor, and pins a foot flat on the floor while the captured foot is planted (lower foot
point within 3 cm of the ground for >= 30 ms): the 4 foot spheres are held where they first touched down.
--froude: same lengths x scale but the clock x sqrt(scale) (dynamic similarity: equal Froude number). With the
human timing a 0.74x robot "falls" at 0.74 g through the delivery's 0.38 s bound (both feet off the ground), which
no physical robot can; at sqrt(scale) its flight is ballistic under real gravity. Written as ref_G1_<delivery><suffix>.

    python scripts/retarget_g1.py --delivery T3aL --feet --froude --suffix f
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
ROOT_W = 0.3            # --feet: pelvis position pulled to the scaled capture (per metre)
ROOTC_W = 3.0           # --feet: pelvis continuity (per metre from previous frame + the capture's step)
PIN_W = 50.0            # --feet: planted foot spheres held in place (per metre)
FLOOR_W = 50.0          # --feet: no foot sphere below the floor (per metre)
ROOT_CUTOFF_HZ = 4.0    # --feet: low-pass of the solved pelvis path
FOOT_R = 0.005          # the G1's foot collision spheres (g1.xml class "foot")
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
    ap.add_argument("--feet", action="store_true", help="solve the pelvis position, feet on (not in) the floor, "
                    "planted feet pinned")
    ap.add_argument("--froude", action="store_true", help="clock x sqrt(scale): dynamically similar timing")
    ap.add_argument("--suffix", default="", help="written as ref_G1_<delivery><suffix>.npz (e.g. f)")
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
    feet = {S: [i for i in range(g1.ngeom) if g1.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE
                and abs(g1.geom_size[i][0] - FOOT_R) < 1e-9 and g1.body(g1.geom_bodyid[i]).name == f"{S}_ankle_roll_link"]
            for S in ("right", "left")}
    planted = {S: np.zeros(n, bool) for S in feet}
    if args.feet:
        # when is each captured foot planted: its lower point (heel or toe) within 3 cm of the floor, >= 30 ms
        low = {S: np.zeros(n) for S in feet}
        for i in range(n):
            sd.qpos[:] = ref["qpos"][i]
            mujoco.mj_kinematics(sk, sd)
            for S, s in (("right", "r"), ("left", "l")):
                low[S][i] = min(sd.xpos[sk.body(f"calcn_{s}").id][2], sd.xpos[sk.body(f"toes_{s}").id][2])
        ri0, rate0 = int(ref["release_index"]), float(ref["rate"])
        for S in feet:
            on = low[S] < low[S].min() + 0.03
            e = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
            for a0, b0 in zip(e[::2], e[1::2]):
                if b0 - a0 >= 3:
                    planted[S][a0:b0] = True
            e = np.flatnonzero(np.diff(np.r_[0, planted[S].astype(int), 0]))
            print(f"  {S} foot planted (s from release, human clock):",
                  [(round((a0 - ri0) / rate0, 2), round((b0 - 1 - ri0) / rate0, 2)) for a0, b0 in zip(e[::2], e[1::2])])
    pins = {S: None for S in feet}
    root_prev = prev_cap = None
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
        nr = 3 if args.feet else 0
        for S in feet:                          # a foot that lifted: its next touch-down gets a new pin
            if not planted[S][i]:
                pins[S] = None

        def fk(xx):
            gd.qpos[:7] = root
            if nr:
                gd.qpos[:3] = xx[:3]
            gd.qpos[qadr] = xx[nr:]
            mujoco.mj_kinematics(g1, gd)

        def resid(xx):
            fk(xx)
            x = xx[nr:]
            r = []
            if nr:
                r.append(ROOT_W * (xx[:3] - root[:3]))
                if root_prev is not None:
                    r.append(ROOTC_W * (xx[:3] - (root_prev + root[:3] - prev_cap)))
                for S, g in feet.items():
                    z = gd.geom_xpos[g, 2] - FOOT_R
                    r.append(FLOOR_W * np.minimum(z, 0.0))                      # never below the floor
                    if planted[S][i]:
                        if pins[S] is None:
                            r.append(FLOOR_W * z)                                # touch down flat
                        else:
                            r.append(PIN_W * (gd.geom_xpos[g] - pins[S]).flatten())
            for name, u, (ga, gb), w in targets:
                r.append(w * (unit(gp(gb) - gp(ga)) - u))
            r.append(TORSO_W * (gd.xmat[torso].reshape(3, 3) - R_tor).flatten())
            r.append(CONT_W * (x - x_prev))
            r.append(WRIST_W * x[wrist])
            return np.concatenate(r)

        if nr:
            r0 = root[:3] if root_prev is None else root_prev + (root[:3] - prev_cap)
            x0 = np.concatenate([r0, np.clip(x_prev, lo, hi)])
            blo, bhi = np.r_[[-np.inf] * 3, lo], np.r_[[np.inf] * 3, hi]
        else:
            x0, blo, bhi = np.clip(x_prev, lo, hi), lo, hi
        sol = least_squares(resid, x0, bounds=(blo, bhi), method="trf", xtol=1e-8, ftol=1e-8,
                            max_nfev=400 if i else 3000)
        x_prev = sol.x[nr:]
        out_q[i, :7], out_q[i, qadr] = root, sol.x[nr:]
        if nr:
            out_q[i, :3] = root_prev = sol.x[:3]
            prev_cap = root[:3]
            fk(sol.x)
            for S, g in feet.items():
                if planted[S][i] and pins[S] is None:
                    pins[S] = gd.geom_xpos[g].copy()
                    pins[S][:, 2] = FOOT_R
        frames.append((targets, R_tor))

    # zero-phase low-pass of the joint angles: near a straight elbow the shoulder's twist is barely determined and
    # the per-frame solution can flip branches (126 rad/s shoulder-yaw spikes unfiltered), which no motor follows
    rate = float(ref["rate"])
    if args.cutoff_hz > 0:
        b, a = butter(4, args.cutoff_hz / (rate / 2.0))
        out_q[:, qadr] = np.clip(filtfilt(b, a, out_q[:, qadr], axis=0), lo, hi)
        if args.feet:
            b2, a2 = butter(2, ROOT_CUTOFF_HZ / (rate / 2.0))
            out_q[:, :3] = filtfilt(b2, a2, out_q[:, :3], axis=0)
    for i, (targets, R_tor) in enumerate(frames):
        gd.qpos[:] = out_q[i]
        mujoco.mj_kinematics(g1, gd)
        for name, u, (ga, gb), w in targets:
            err[name][i] = np.degrees(np.arccos(np.clip(unit(gp(gb) - gp(ga)) @ u, -1.0, 1.0)))
        Rg = gd.xmat[torso].reshape(3, 3)
        err["torso"][i] = np.degrees(np.arccos(np.clip((np.trace(Rg.T @ R_tor) - 1.0) / 2.0, -1.0, 1.0)))

    if args.froude:                 # the same frames on a slower clock: lengths x scale, times x sqrt(scale)
        rate = rate / np.sqrt(scale)
        print(f"  froude: clock x {np.sqrt(scale):.3f}, {rate:.1f} frames per second")
    qvel = np.zeros((n, g1.nv))
    for i in range(n - 1):
        mujoco.mj_differentiatePos(g1, qvel[i], 1.0 / rate, out_q[i], out_q[i + 1])
    qvel[-1] = qvel[-2]
    ri = int(ref["release_index"])
    dst = MOCAP_DIR / f"ref_G1_{args.delivery}{args.suffix}.npz"
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
    (od / f"retarget_{args.delivery}{args.suffix}.json").write_text(json.dumps(rep, indent=1))
    for k, v in rep["segment_error_deg"].items():
        print(f"  {k:12s} median {v['median']:5.1f}  p95 {v['p95']:5.1f}  release-window max {v['release_window_max']:5.1f}"
              f"  at release {v['at_release']:5.1f}")
    print("  at a joint limit:", {k: round(v, 2) for k, v in rep["joints_at_limit_frac"].items()})


if __name__ == "__main__":
    main()
