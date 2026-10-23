"""Physics version of the mocap skeleton, for learning the run-up and trunk from the capture.

assets/skeleton3d/skeleton3d.xml is the kinematic skeleton the 11 deliveries were
fitted to (mocap/skeleton3d.py): LocoMuJoCo's Hamner-2010 body with ball-jointed
hips, lumbar, shoulders and neck, no actuators, no contacts. This module turns
it into something that can be simulated and controlled:

  * motors on every joint, with the strengths of LocoMuJoCo's skeleton_torque.xml
    (the RL-ready source model); three per ball joint, one per local axis
  * the feet (calcaneus and toes) collide with the floor; nothing else does
  * the shoulder-girdle slides are removed: they let the fit move each shoulder
    up to 6 cm against the ribcage, which a rigid physical torso cannot do
  * the free ball is removed (the delivery task comes later)
  * armature on every joint and implicit damping, so stiff PD control is stable
    at the 2 ms physics step

Control is PD to joint targets (DeepMimic-style): the policy, or a test, gives a
target pose; `pd_torques` turns it into motor commands. Ball joints use the
quaternion error, so there is no gimbal lock with the arm overhead.

The reference motion comes from data/mocap/fit3d_<D>.npz (participant data,
not in the repo): `load_reference` maps its qpos onto this model.
"""
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SKELETON_XML = ROOT / "assets" / "skeleton3d" / "skeleton3d.xml"
MOCAP_DIR = ROOT / "data" / "mocap"

# peak torque (N m) per joint, from skeleton_torque.xml's motor gears; ball joints
# take the source's three hinges' gears in its axis order (flexion, adduction/
# bending, rotation), applied about the local x, z, y axes below
HINGE_GEAR = {"knee_angle": 600, "ankle_angle": 500, "subtalar_angle": 50, "mtp_angle": 50,
              "elbow_flex": 250, "pro_sup": 250, "wrist_flex": 50, "wrist_dev": 50}
BALL_GEAR = {"hip": (530, 600, 275), "lumbar": (160, 100, 160), "shoulder": (250, 250, 250),
             "neck": (50, 50, 50)}
ARMATURE = 0.02          # kg m^2 per joint: keeps the light toes/wrists stable under PD
# 1x sagged 9 cm holding a standing pose; 2x holds it (open-loop PD still falls
# within ~1 s standing, ~0.3 s following the run-up: balance is the policy's job)
KP_PER_GEAR = 2.0        # PD stiffness, N m/rad per N m of peak torque
KD_PER_GEAR = 0.2        # damping (implicit, as joint damping), N m s/rad per N m

# the bowling arm (right) is never imitated: its legality must come from the
# simulation, not from a human reference (context.md, design decision)
BOWLING_ARM = ("shoulder_r", "elbow_flex_r", "pro_sup_r", "wrist_flex_r", "wrist_dev_r")
FEET = ("calcn_r", "toes_r", "calcn_l", "toes_l")


def _base(name):
    return name[:-2] if name.endswith(("_r", "_l")) else name


def build_model(timestep=0.002):
    spec = mujoco.MjSpec.from_file(str(SKELETON_XML))
    for j in [j for j in spec.joints if j.name.startswith("girdle_")]:
        spec.delete(j)
    spec.delete(spec.body("ball"))
    spec.option.timestep = timestep
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    for g in spec.geoms:
        g.contype, g.conaffinity = 0, 0
    floor = spec.geom("floor")
    floor.contype, floor.conaffinity = 1, 1
    for b in FEET:
        for g in spec.body(b).geoms:
            g.contype, g.conaffinity, g.condim = 1, 1, 3
            g.friction = [1.0, 0.005, 0.0001]

    for j in spec.joints:
        if j.type == mujoco.mjtJoint.mjJNT_FREE:
            continue
        j.armature = ARMATURE
        if j.type == mujoco.mjtJoint.mjJNT_HINGE:
            gear = HINGE_GEAR[_base(j.name)]
            j.damping = [KD_PER_GEAR * gear, 0.0, 0.0]
            a = spec.add_actuator(name=f"mot_{j.name}", target=j.name,
                                  trntype=mujoco.mjtTrn.mjTRN_JOINT)
            a.gear = [gear, 0, 0, 0, 0, 0]
            a.ctrllimited, a.ctrlrange = True, [-1.0, 1.0]
        elif j.type == mujoco.mjtJoint.mjJNT_BALL:
            gears = BALL_GEAR[_base(j.name)]
            # MuJoCo damps a ball joint isotropically (first entry): use the mean strength
            j.damping = [KD_PER_GEAR * float(np.mean(gears)), 0.0, 0.0]
            for axis, gear in zip("xzy", gears):
                a = spec.add_actuator(name=f"mot_{j.name}_{axis}", target=j.name,
                                      trntype=mujoco.mjtTrn.mjTRN_JOINT)
                g6 = [0.0] * 6
                g6["xyz".index(axis)] = gear
                a.gear = g6
                a.ctrllimited, a.ctrlrange = True, [-1.0, 1.0]
    return spec.compile()


class Joints:
    """Per-joint bookkeeping for PD control and tracking rewards."""

    def __init__(self, m):
        self.m = m
        self.items = []    # (name, type, qpos adr, dof adr, [actuator ids], [gears])
        for j in range(m.njnt):
            t = m.jnt_type[j]
            if t == mujoco.mjtJoint.mjJNT_FREE:
                continue
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            acts = [a for a in range(m.nu) if m.actuator_trnid[a, 0] == j]
            gears = [float(np.abs(m.actuator_gear[a, :3]).max()) for a in acts]
            self.items.append((name, t, m.jnt_qposadr[j], m.jnt_dofadr[j], acts, gears))
        self.names = [it[0] for it in self.items]
        # policy action size: one number per actuated DoF (3 per ball joint)
        self.n_act = m.nu
        hinges = [it for it in self.items if it[1] == mujoco.mjtJoint.mjJNT_HINGE]
        self._h_qadr = np.array([it[2] for it in hinges])
        self._h_act = np.array([it[4][0] for it in hinges])
        self._balls = [(it[2], it[4], [int(np.argmax(np.abs(m.actuator_gear[a_, :3]))) for a_ in it[4]])
                       for it in self.items if it[1] == mujoco.mjtJoint.mjJNT_BALL]

    def pd_torques(self, data, q_target):
        """Motor commands (ctrl in [-1, 1]) driving qpos toward q_target (same layout as qpos).
        Damping is in the model (implicit); this is the stiffness part only, kp*err/gear."""
        ctrl = np.zeros(self.m.nu)
        q = data.qpos
        ctrl[self._h_act] = KP_PER_GEAR * (q_target[self._h_qadr] - q[self._h_qadr])
        e, qi, qe = np.zeros(3), np.zeros(4), np.zeros(4)
        for qa, acts, axes in self._balls:
            # error rotation in the child frame: q^-1 * q_target, as a rotation vector
            mujoco.mju_negQuat(qi, q[qa:qa + 4])
            mujoco.mju_mulQuat(qe, qi, q_target[qa:qa + 4])
            mujoco.mju_quat2Vel(e, qe, 1.0)
            ctrl[acts] = KP_PER_GEAR * e[axes]
        return np.clip(ctrl, -1.0, 1.0)


def export_reference(model, delivery="T3a"):
    """Write data/mocap/ref_<D>.npz: joint angles only, on this model's layout.

    The fit file also holds the participant's raw marker positions; training (and
    the HPC) only needs the pose, so that is all this file carries."""
    ref = _from_fit(model, delivery)
    out = MOCAP_DIR / f"ref_{delivery}.npz"
    np.savez_compressed(out, qpos=ref["qpos"], rate=ref["rate"], release_index=ref["release_index"],
                        nq=model.nq, delivery=delivery)
    return out


def load_reference(model, delivery="T3a"):
    """The fitted delivery as (qpos, qvel) on this model's layout, at the fit's rate.

    Reads data/mocap/ref_<D>.npz (export_reference) if present, else the fit.
    Returns dict(qpos (n, nq), qvel (n, nv), rate, release_index, dt)."""
    f = MOCAP_DIR / f"ref_{delivery}.npz"
    if not f.exists():
        return _from_fit(model, delivery)
    z = np.load(f)
    assert int(z["nq"]) == model.nq, "reference was exported for a different model layout"
    return _with_velocity(model, z["qpos"], float(z["rate"]), int(z["release_index"]), delivery)


def _with_velocity(model, Q, rate, release_index, delivery):
    n = len(Q)
    V = np.zeros((n, model.nv))
    for i in range(n - 1):
        mujoco.mj_differentiatePos(model, V[i], 1.0 / rate, Q[i], Q[i + 1])
    V[-1] = V[-2]
    return {"qpos": Q, "qvel": V, "rate": rate, "release_index": release_index,
            "dt": 1.0 / rate, "delivery": delivery}


def _from_fit(model, delivery):
    z = np.load(MOCAP_DIR / f"fit3d_{delivery}.npz", allow_pickle=True)
    src = mujoco.MjModel.from_xml_string(str(z["xml"])) if "xml" in z.files else \
        mujoco.MjModel.from_xml_path(str(SKELETON_XML))
    Qs = z["qpos"]
    n = Qs.shape[0]
    Q = np.zeros((n, model.nq))
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        sj = mujoco.mj_name2id(src, mujoco.mjtObj.mjOBJ_JOINT, name)
        size = {0: 7, 1: 4, 2: 1, 3: 1}[int(model.jnt_type[j])]
        Q[:, model.jnt_qposadr[j]:model.jnt_qposadr[j] + size] = \
            Qs[:, src.jnt_qposadr[sj]:src.jnt_qposadr[sj] + size]
    return _with_velocity(model, Q, float(z["rate"]), int(z["release_index"]), delivery)


def ref_at(ref, t, model):
    """Reference pose and velocity at time t (s), interpolated between frames."""
    x = np.clip(t * ref["rate"], 0, len(ref["qpos"]) - 1.000001)
    i, f = int(x), x - int(x)
    q = np.empty(model.nq)
    v = (1 - f) * ref["qvel"][i] + f * ref["qvel"][i + 1]
    # integrate from frame i by the interpolated fraction of its velocity: exact on
    # quaternions, linear elsewhere
    q[:] = ref["qpos"][i]
    mujoco.mj_integratePos(model, q, ref["qvel"][i], f / ref["rate"])
    return q, v
