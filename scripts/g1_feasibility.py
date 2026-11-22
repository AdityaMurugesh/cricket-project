"""Could the Unitree G1 itself make the body motion it is given? Checks on the retargeted reference
(data/mocap/ref_G1_<D>.npz, scripts/retarget_g1.py) against the G1's own limits.

  * joint ranges: by construction inside them (the retarget is bounded); how often each joint sits AT a limit is
    in results/kinematic/g1/retarget_<D>.json, and what that costs shows as the segment direction errors there.
  * joint speeds: each joint's speed (forward differences at 100 Hz) against its motor's no-load speed (URDF
    <limit velocity>). Above it the motor cannot drive the joint at all.
  * joint torques, where they can be computed without the ground forces: inverse dynamics (mj_inverse) gives, at a
    joint, the torque that the motion of everything BEYOND it needs (a tree: the backward Newton-Euler pass). For
    the waist (torso, head, both arms beyond it) and the arm joints that is exact whatever the feet do. The leg
    joints carry the ground reaction, which a kinematic reference does not have: not computed here (that needs the
    physics run, the next step). Compared with the Menagerie model's forcerange (Unitree's peak torques).
  * the bowling arm is reported for the CAPTURED arm motion (what a human arm did); in the G1 results the arm is the
    optimiser's / policy's and its motors are limited in the simulation itself.

Timing is the human's (the retarget keeps it), so a G1 at 0.74 x the bowler's size is asked for the bowler's
angular speeds; dynamic similarity (Froude) would let a smaller body take sqrt(0.74) = 0.86 x the time, which
would make the speeds 1.16 x HIGHER still. So this is not a harsh reading.

    python scripts/g1_feasibility.py --delivery T3aL
"""
import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from envs.g1_bowl_env import G1_XML, MOCAP_DIR      # noqa: E402

# URDF <limit velocity> (rad/s), unitree_ros g1_29dof (scripts/body_vs_robot.py, envs/arm_limits.py)
VEL = {"hip_pitch": 32, "hip_roll": 32, "hip_yaw": 32, "knee": 20, "ankle_pitch": 30, "ankle_roll": 30,
       "waist_yaw": 32, "waist_roll": 30, "waist_pitch": 30, "shoulder_pitch": 37, "shoulder_roll": 37,
       "shoulder_yaw": 37, "elbow": 37, "wrist_roll": 37, "wrist_pitch": 22, "wrist_yaw": 22}
WINDOWS = {"bound": (-1.0, -0.6), "delivery_stride": (-0.6, 0.3)}
NO_GRF = ("waist", "shoulder", "elbow", "wrist")     # joints whose torque does not depend on the ground forces


def kind(name):
    n = name.replace("_joint", "")
    for s in ("left_", "right_"):
        n = n.replace(s, "")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delivery", default="T3aL")
    a = ap.parse_args()
    a.from_s, a.to_s = -1.0, 0.3
    z = np.load(MOCAP_DIR / f"ref_G1_{a.delivery}.npz")
    q, v, rate, ri = z["qpos"], z["qvel"], float(z["rate"]), int(z["release_index"])
    spec = mujoco.MjSpec.from_file(str(G1_XML))
    for k in list(spec.keys):
        spec.delete(k)
    peak = {act.target: float(act.forcerange[1]) for act in spec.actuators}
    for act in list(spec.actuators):
        spec.delete(act)
    m = spec.compile()
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT | mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
    m.dof_frictionloss[:] = 0.0
    d = mujoco.MjData(m)
    i0, i1 = max(int(ri + a.from_s * rate), 0), min(int(ri + a.to_s * rate), len(q) - 2)
    acc = np.diff(v, axis=0) * rate
    tau = []
    for i in range(i0, i1 + 1):
        d.qpos[:], d.qvel[:], d.qacc[:] = q[i], v[i], acc[i]
        mujoco.mj_inverse(m, d)
        tau.append(d.qfrc_inverse.copy())
    tau = np.array(tau)
    rows = {}
    for j in range(m.njnt):
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name, da = m.joint(j).name, m.jnt_dofadr[j]
        r = {"speed_limit": VEL[kind(name)]}
        if kind(name).startswith(NO_GRF):
            r["torque_limit"] = peak[name]
        for wname, (s0, s1) in WINDOWS.items():
            k0, k1 = int(round((s0 - a.from_s) * rate)), int(round((s1 - a.from_s) * rate))
            w = np.abs(v[i0 + k0:i0 + k1 + 1, da])
            r[wname] = {"speed_p50": float(np.median(w)), "speed_p95": float(np.percentile(w, 95)),
                        "speed_max": float(w.max()), "frac_over_speed": float(np.mean(w > r["speed_limit"]))}
            if "torque_limit" in r:
                t = np.abs(tau[k0:k1 + 1, da])
                r[wname].update(torque_p50=float(np.median(t)), torque_p95=float(np.percentile(t, 95)),
                                torque_max=float(t.max()), frac_over_torque=float(np.mean(t > r["torque_limit"])))
        rows[name] = r
    series = {"t_s": ((np.arange(i0, i1 + 1) - ri) / rate).tolist(),
              **{m.joint(j).name: (tau[:, m.jnt_dofadr[j]] / peak[m.joint(j).name]).tolist()
                 for j in range(m.njnt) if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE
                 and kind(m.joint(j).name).startswith(NO_GRF)}}
    out = {"delivery": a.delivery, "windows_s": WINDOWS, "scale": float(z["scale"]), "joints": rows,
           "torque_over_limit_series": series}
    od = ROOT / "results" / "kinematic" / "g1"
    od.mkdir(parents=True, exist_ok=True)
    (od / f"feasibility_{a.delivery}.json").write_text(json.dumps(out, indent=1))
    for wname in WINDOWS:
        print(f"== {wname} {WINDOWS[wname]} s from release: speed p95 / limit (rad/s, % frames over); torque p50 p95 / "
              f"limit (N m, % over; waist and arms only)")
        for name, r in rows.items():
            w = r[wname]
            tq = (f"   torque {w['torque_p50']:5.1f} {w['torque_p95']:6.1f} / {r['torque_limit']:3.0f} "
                  f"({100 * w['frac_over_torque']:3.0f}%)") if "torque_limit" in r else ""
            print(f"  {name:28s} speed {w['speed_p95']:5.1f} / {r['speed_limit']:2d} ({100 * w['frac_over_speed']:3.0f}%){tq}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.2))
    tt = np.array(series["t_s"])
    for name, col in (("waist_yaw_joint", "#1f77b4"), ("waist_roll_joint", "#d62728"), ("waist_pitch_joint", "#ff7f0e"),
                      ("right_shoulder_pitch_joint", "#2ca02c"), ("right_shoulder_roll_joint", "#9467bd"),
                      ("left_shoulder_pitch_joint", "#8c564b")):
        ax.plot(tt, np.abs(series[name]), color=col, lw=1.3, label=name.replace("_joint", ""))
    ax.axhline(1.0, color="k", lw=1, ls="--")
    ax.axvspan(-1.0, -0.6, color="0.9", zorder=0)
    ax.text(-0.98, ax.get_ylim()[1] * 0.92, "run-up end + bound\n(pelvis poorly measured)", fontsize=8, va="top")
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("time from release (s)")
    ax.set_ylabel("torque needed / G1 peak torque")
    ax.set_title("Unitree G1 following the captured delivery: torque needed at the waist and shoulders (captured arm motion)",
                 fontsize=9)
    ax.legend(fontsize=7, ncol=2, loc="lower left")
    fig.tight_layout()
    fig.savefig(od / f"feasibility_{a.delivery}.png", dpi=150)
    print("wrote", od / f"feasibility_{a.delivery}.png")


if __name__ == "__main__":
    main()
