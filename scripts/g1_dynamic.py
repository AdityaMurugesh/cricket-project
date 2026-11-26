"""Can the real G1 do the whole delivery? A first DYNAMIC whole-body test (supervisor, 20 Nov: "main next step").

Until now the G1's body was prescribed (envs/g1_bowl_env.py: replayed exactly, no floor, no balance). Here the
whole robot is physically simulated: a free-floating base, floor contact through the G1's 8 foot spheres
(Menagerie g1.xml, friction 0.6), and all 29 joints on Unitree's own position servos (kp 500, critically damped,
torque clipped to the URDF effort: hips/knees 88-139 N m, ankles 50, waist 50-88, arms 25, wrists 5).
Each servo tracks the retargeted capture (data/mocap/ref_G1_<delivery>.npz, scripts/retarget_g1.py).

Two checks:
  1. ground   the reference's feet against the floor. The retarget scaled the pelvis path but never looked at
              the ground, so its feet sink below it. --ground-fix shifts the pelvis up/down (5 Hz filtered) so
              that the lower foot touches the floor in every frame: a robot can't put its feet under the floor
  2. balance  start in the reference state, track it with the servos only, and see whether the robot stays up:
              time to fall (pelvis below 0.45 m or trunk tilted > 60 deg), pelvis tracking error, and the share
              of frames each joint group is torque-saturated. Then the same with "help": a virtual spring-damper
              holding the pelvis on its reference pose (the "hand of god" of imitation-learning papers); the
              force and torque it needs say how far the motion is from what this robot can do on its own.
              The joint torques under help include the legs (the floor now carries the robot).

    python scripts/g1_dynamic.py [--delivery T3aL] [--before 1.0] [--video]
  -> results/kinematic/g1/dynamic_<delivery>.json, dynamic_<delivery>.png (+ videos/g1_dynamic_*.mp4)
"""
import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

G1_XML = ROOT / "assets" / "unitree_g1" / "g1.xml"
OUT = ROOT / "results" / "kinematic" / "g1"
GROUPS = {"legs": ("hip", "knee", "ankle"), "waist": ("waist",), "left arm": ("left_shoulder", "left_elbow", "left_wrist"),
          "right arm": ("right_shoulder", "right_elbow", "right_wrist")}


def build(timestep=0.002):
    spec = mujoco.MjSpec.from_file(str(G1_XML))
    for k in list(spec.keys):
        spec.delete(k)
    spec.option.timestep = timestep
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[60, 20, 0.1],
                            rgba=[0.45, 0.6, 0.4, 1])
    spec.worldbody.add_light(pos=[0, 0, 6], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 1080
    spec.visual.headlight.ambient, spec.visual.headlight.diffuse = [0.45] * 3, [0.6] * 3    # videos only
    return spec.compile()


def foot_geoms(m):
    return [i for i in range(m.ngeom) if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE
            and abs(m.geom_size[i][0] - 0.005) < 1e-9 and "ankle_roll" in m.body(m.geom_bodyid[i]).name]


def foot_heights(m, qpos):
    d, feet = mujoco.MjData(m), foot_geoms(m)
    out = []
    for q in qpos:
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        z = d.geom_xpos[feet][:, 2] - m.geom_size[feet][:, 0]
        x = d.geom_xpos[feet][:, :2]
        out.append((z[:4].min(), z[4:].min(), x[:4].mean(0), x[4:].mean(0)))
    zl = np.array([o[0] for o in out])
    zr = np.array([o[1] for o in out])
    return zl, zr, np.array([o[2] for o in out]), np.array([o[3] for o in out])


def ground_fix(m, ref, cutoff_hz=5.0):
    """Shift the pelvis vertically so the lower foot touches the floor in every frame (filtered)."""
    zl, zr, _, _ = foot_heights(m, ref["qpos"])
    off = -np.minimum(zl, zr)
    b, a = butter(2, cutoff_hz / (ref["rate"] / 2))
    off = filtfilt(b, a, off)
    q = ref["qpos"].copy()
    q[:, 2] += off
    qv = ref["qvel"].copy()
    qv[:, 2] += np.gradient(off) * ref["rate"]
    return dict(ref, qpos=q, qvel=qv), off


def interp_ref(ref, t):
    """Reference qpos/qvel at time t (s from the first frame): linear on joints, slerp on the base quaternion."""
    f = np.clip(t * ref["rate"], 0, len(ref["qpos"]) - 1.000001)
    i, w = int(f), f - int(f)
    q0, q1 = ref["qpos"][i], ref["qpos"][i + 1]
    q = (1 - w) * q0 + w * q1
    r = Rotation.from_quat([q0[[4, 5, 6, 3]], q1[[4, 5, 6, 3]]])
    from scipy.spatial.transform import Slerp
    qq = Slerp([0, 1], r)([w]).as_quat()[0]
    q[3:7] = qq[[3, 0, 1, 2]]
    v = (1 - w) * ref["qvel"][i] + w * ref["qvel"][i + 1]
    return q, v


def tilt_deg(quat_wxyz):
    R = Rotation.from_quat(quat_wxyz[[1, 2, 3, 0]]).as_matrix()
    return float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))


def simulate(m, ref, t0, t1, help_kp=0.0, record=False):
    """Track ref from t0 to t1 with the G1's servos; optional pelvis spring-damper 'help' (kp N/m, kp/10 N m/rad)."""
    d = mujoco.MjData(m)
    q, v = interp_ref(ref, t0)
    d.qpos[:], d.qvel[:] = q, v
    mujoco.mj_forward(m, d)
    pelvis = m.body("pelvis").id
    act_q = np.array([m.jnt_qposadr[m.actuator_trnid[a, 0]] for a in range(m.nu)])
    act_name = [m.actuator(a).name for a in range(m.nu)]
    lim = m.actuator_forcerange[:, 1]
    mass = m.body_subtreemass[0]
    kd = 2 * np.sqrt(help_kp * mass) if help_kp else 0.0
    n = int(round((t1 - t0) / m.opt.timestep))
    rows, frames, fell_t = [], [], None
    for k in range(n):
        t = t0 + k * m.opt.timestep
        q, v = interp_ref(ref, t)
        d.ctrl[:] = q[act_q]
        d.xfrc_applied[:] = 0
        if help_kp:
            e = q[:3] - d.qpos[:3]
            f = help_kp * e + kd * (v[:3] - d.qvel[:3])
            R_err = Rotation.from_quat(q[[4, 5, 6, 3]]) * Rotation.from_quat(d.qpos[[4, 5, 6, 3]]).inv()
            w_ref = Rotation.from_quat(q[[4, 5, 6, 3]]).apply(v[3:6])
            w = Rotation.from_quat(d.qpos[[4, 5, 6, 3]]).apply(d.qvel[3:6])
            tq = help_kp / 10 * R_err.as_rotvec() + kd / 10 * (w_ref - w)
            d.xfrc_applied[pelvis, :3], d.xfrc_applied[pelvis, 3:] = f, tq
        mujoco.mj_step(m, d)
        tf = d.actuator_force.copy()
        sat = np.abs(tf) >= lim - 1e-6
        tilt = tilt_deg(d.qpos[3:7])
        if fell_t is None and (d.qpos[2] < 0.45 or tilt > 60):
            fell_t = t
        rows.append({"t": t, "pelvis_err_m": float(np.linalg.norm(q[:3] - d.qpos[:3])), "tilt_deg": tilt,
                     "help_f": float(np.linalg.norm(d.xfrc_applied[pelvis, :3])),
                     "help_tq": float(np.linalg.norm(d.xfrc_applied[pelvis, 3:])),
                     "torque": tf, "sat": sat, "joint_err_deg": np.degrees(np.abs(q[act_q] - d.qpos[act_q]))})
        if record and k % 4 == 0:
            frames.append(d.qpos.copy())
        if fell_t is not None and not help_kp and t > fell_t + 0.3:
            break
    return rows, fell_t, act_name, lim, frames


def summarise(rows, fell_t, act_name, lim, t_rel, mass):
    T = np.array([r["torque"] for r in rows])
    S = np.array([r["sat"] for r in rows])
    E = np.array([r["joint_err_deg"] for r in rows])
    out = {"fell_s_rel_release": None if fell_t is None else round(fell_t - t_rel, 3),
           "pelvis_err_m_median": float(np.median([r["pelvis_err_m"] for r in rows])),
           "pelvis_err_m_max": float(np.max([r["pelvis_err_m"] for r in rows])),
           "help_force_N_p95": float(np.percentile([r["help_f"] for r in rows], 95)),
           "help_force_bw_p95": float(np.percentile([r["help_f"] for r in rows], 95) / (mass * 9.81)),
           "help_force_N_max": float(np.max([r["help_f"] for r in rows])),
           "help_torque_Nm_p95": float(np.percentile([r["help_tq"] for r in rows], 95)), "groups": {}, "joints": {}}
    for g, keys in GROUPS.items():
        idx = [i for i, nme in enumerate(act_name) if any(k in nme for k in keys)]
        out["groups"][g] = {"saturated_pct": float(100 * S[:, idx].any(1).mean()),
                            "joint_err_deg_median": float(np.median(E[:, idx]))}
    for i, nme in enumerate(act_name):
        out["joints"][nme] = {"limit_Nm": float(lim[i]), "p95_Nm": float(np.percentile(np.abs(T[:, i]), 95)),
                              "saturated_pct": float(100 * S[:, i].mean())}
    return out


def residual_wrench(m, ref, i0, i1):
    """Inverse dynamics of the reference (contacts off): the external force on the pelvis the motion demands, per
    frame (N). With a foot on the floor the floor can supply it (within friction / the foot's area); with both feet
    in the air nothing can, so any force there means the motion breaks physics (OpenSim calls it the 'residual')."""
    flags = m.opt.disableflags
    m.opt.disableflags = flags | mujoco.mjtDisableBit.mjDSBL_CONTACT
    d = mujoco.MjData(m)
    feet = foot_geoms(m)
    rate = ref["rate"]
    qacc = np.gradient(ref["qvel"], axis=0) * rate
    F, air = [], []
    try:
        for k in range(i0, i1):
            d.qpos[:], d.qvel[:], d.qacc[:] = ref["qpos"][k], ref["qvel"][k], qacc[k]
            mujoco.mj_inverse(m, d)
            F.append(np.linalg.norm(d.qfrc_inverse[:3]))
            z = d.geom_xpos[feet][:, 2] - m.geom_size[feet][:, 0]
            air.append(z.min() > 0.03)
    finally:
        m.opt.disableflags = flags
    return np.array(F), np.array(air)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--delivery", default="T3aL")
    p.add_argument("--before", type=float, default=1.0, help="start this long before the captured release (s)")
    p.add_argument("--after", type=float, default=0.35, help="stop this long after it")
    p.add_argument("--help-kp", type=float, nargs="+", default=[3000.0])
    p.add_argument("--video", action="store_true")
    a = p.parse_args()
    from envs.g1_bowl_env import load_g1_reference
    m = build()
    raw = load_g1_reference(a.delivery)
    mass = float(m.body_subtreemass[0])
    t_rel = raw["release_index"] / raw["rate"]
    t0, t1 = t_rel - a.before, min(t_rel + a.after, (len(raw["qpos"]) - 1) / raw["rate"])
    zl, zr, xl, xr = foot_heights(m, raw["qpos"])
    fixed, off = ground_fix(m, raw)
    zl2, zr2, _, _ = foot_heights(m, fixed["qpos"])
    i0, i1 = int(t0 * raw["rate"]), int(t1 * raw["rate"])
    res = {"delivery": a.delivery, "mass_kg": mass, "window_s_rel_release": [-a.before, round(t1 - t_rel, 3)],
           "ground": {"lower_foot_below_floor_cm_median": float(-100 * np.median(np.minimum(zl, zr)[i0:i1])),
                      "lower_foot_below_floor_cm_max": float(-100 * np.min(np.minimum(zl, zr)[i0:i1])),
                      "pelvis_shift_cm_median": float(100 * np.median(off[i0:i1])),
                      "after_fix_lower_foot_cm_abs_median": float(100 * np.median(np.abs(np.minimum(zl2, zr2)[i0:i1])))},
           "runs": {}}
    # foot sliding: horizontal speed of the lower (stance) foot while it is within 1 cm of the floor (after the fix)
    lower = np.where(zl2 < zr2, 0, 1)
    xs = np.where(lower[:, None] == 0, xl, xr)
    on = np.minimum(zl2, zr2) < 0.01
    sp = np.linalg.norm(np.gradient(xs, axis=0), axis=1) * raw["rate"]
    same = np.r_[True, lower[1:] == lower[:-1]]
    sl = sp[i0:i1][(on & same)[i0:i1]]
    res["ground"]["stance_foot_slide_m_s_median"] = float(np.median(sl)) if len(sl) else None
    res["ground"]["stance_foot_slide_m_s_p90"] = float(np.percentile(sl, 90)) if len(sl) else None
    F, air = residual_wrench(m, fixed, i0, i1)
    res["flight"] = {"frames_both_feet_off": int(air.sum()), "duration_s": float(air.sum() / raw["rate"]),
                     "force_needed_in_flight_bw_median": float(np.median(F[air]) / (mass * 9.81)) if air.any() else None,
                     "force_needed_on_ground_bw_median": float(np.median(F[~air]) / (mass * 9.81))}
    print("ground:", json.dumps(res["ground"]))
    print("flight:", json.dumps(res["flight"]))
    curves = {}
    for label, ref, hk in [("raw_nohelp", raw, 0.0), ("fixed_nohelp", fixed, 0.0)] + \
                          [(f"fixed_help{int(k)}", fixed, k) for k in a.help_kp]:
        rows, fell, names, lim, frames = simulate(m, ref, t0, t1, hk, record=a.video)
        s = summarise(rows, fell, names, lim, t_rel, mass)
        res["runs"][label] = s
        curves[label] = rows
        print(f"{label:16s} fell {s['fell_s_rel_release']}  pelvis err median {s['pelvis_err_m_median']:.3f} m  "
              f"help p95 {s['help_force_N_p95']:.0f} N ({s['help_force_bw_p95']:.2f} BW)  "
              + "  ".join(f"{g} sat {v['saturated_pct']:.0f}%" for g, v in s["groups"].items()), flush=True)
        if a.video:
            render(m, frames, label, a.delivery, t0 - t_rel)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / f"dynamic_{a.delivery}.json", "w"), indent=1)
    plot(res, curves, t_rel, zl, zr, zl2, zr2, raw["rate"], i0, i1, a.delivery)


def plot(res, curves, t_rel, zl, zr, zl2, zr2, rate, i0, i1, delivery):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(12, 7.5))
    t = np.arange(len(zl)) / rate - t_rel
    a = ax[0, 0]
    a.plot(t, 100 * np.minimum(zl, zr), color="#b5523b", label=f"reference {delivery} as given")
    a.plot(t, 100 * np.minimum(zl2, zr2), color="#2b7bb9", label="after the floor fix (pelvis shifted)")
    a.axhline(0, color="k", lw=0.8)
    a.axvspan(t[i0], t[i1], color="0.9", zorder=0)
    a.set(xlabel="time from release (s)", ylabel="lower foot height (cm)", title="1. Feet against the floor")
    a.legend(fontsize=8)
    a = ax[0, 1]
    for lab, rows in curves.items():
        tt = np.array([r["t"] for r in rows]) - t_rel
        a.plot(tt, [r["tilt_deg"] for r in rows], label=lab)
    a.axhline(60, color="k", ls=":", lw=0.8)
    a.set(xlabel="time from release (s)", ylabel="trunk tilt from vertical (deg)", title="2. Does it stay up?")
    a.legend(fontsize=8)
    a = ax[1, 0]
    for lab, rows in curves.items():
        if lab.startswith("fixed_help"):
            tt = np.array([r["t"] for r in rows]) - t_rel
            mass_w = res["mass_kg"] * 9.81
            a.plot(tt, np.array([r["help_f"] for r in rows]) / mass_w, label=lab)
    a.set(xlabel="time from release (s)", ylabel="help force on the pelvis (x body weight)",
          title="3. How much outside help it needs")
    a.legend(fontsize=8)
    a = ax[1, 1]
    lab = next((k for k in res["runs"] if k.startswith("fixed_help")), None)
    if lab:
        J = res["runs"][lab]["joints"]
        names = [n for n in J if not n.startswith("left_wrist") and not n.startswith("right_wrist")]
        r = [J[n]["p95_Nm"] / J[n]["limit_Nm"] for n in names]
        a.barh(range(len(names)), r, color=["#b5523b" if x >= 0.999 else "#2b7bb9" for x in r])
        a.axvline(1, color="k", lw=0.8)
        a.set_yticks(range(len(names)))
        a.set_yticklabels([n.replace("_joint", "") for n in names], fontsize=6)
        a.set(xlabel="p95 torque / Unitree limit", title=f"4. Joint torques with help ({lab})")
    fig.suptitle(f"Unitree G1, whole body simulated with floor contact: the retargeted delivery {delivery}")
    fig.tight_layout()
    fig.savefig(OUT / f"dynamic_{delivery}.png", dpi=130)
    print("wrote", OUT / f"dynamic_{delivery}.png")


def render(m, frames, label, delivery, t_start_rel):
    import subprocess
    import tempfile
    import matplotlib.pyplot as plt
    from scripts.render_kinematic import LOCO_PY, WRITE
    d = mujoco.MjData(m)
    rend = mujoco.Renderer(m, 540, 640)
    tmp = Path(tempfile.mkdtemp(prefix="g1dyn_"))
    for f, q in enumerate(frames[::2]):           # every 16 ms of simulation -> 25 fps = 2.5x slow motion
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [q[0], q[1], 0.7]
        cam.distance, cam.azimuth, cam.elevation = 3.2, 90, -8
        rend.update_scene(d, cam)
        img = rend.render().copy()
        cam.azimuth = 180
        rend.update_scene(d, cam)
        plt.imsave(tmp / f"f_{f:05d}.png", np.concatenate([img, rend.render().copy()], axis=1))
    out = ROOT / "results" / "kinematic" / "videos" / f"g1_dynamic_{label}.mp4"
    lines = [f"Unitree G1, WHOLE BODY PHYSICALLY SIMULATED (floor contact, its own servos + torque limits): {label}",
             "Servos track the retargeted capture" + (" (no outside help)" if "nohelp" in label else
                                                       " + a virtual spring holding the pelvis ('help')"),
             f"Starts {abs(t_start_rel):.1f} s before release. 2.5x slow motion. Left: side view, right: from behind."]
    subprocess.run([str(LOCO_PY), "-c", WRITE, str(tmp), str(out), "25", "|".join(lines)], check=True)


if __name__ == "__main__":
    main()
