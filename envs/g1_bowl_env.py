"""The bowling task on the Unitree G1 humanoid: the robot's own body, its own arm, its own motors.

Supervisor, 20 Nov: "try an actual humanoid robot model, with two sets of actuator and joint limits". Until 21 Nov
only the arm's motors and ranges were the robot's (envs/arm_limits.py); the arm, its masses and the body it rode on
were the captured bowler's. Here everything is the G1 (MuJoCo Menagerie unitree_g1/g1.xml, 29 DoF, 1.32 m, 35 kg,
Unitree's masses and inertias):

  * the body (legs, pelvis, waist, left arm) replays the captured delivery RETARGETED onto the G1
    (scripts/retarget_g1.py: per-frame IK on segment directions inside the G1's joint ranges, pelvis path scaled by
    the leg-length ratio 0.74, same timing). It is prescribed exactly as in KinematicBowlEnv (huge armature plus
    the reference acceleration, reset every physics step; no contacts), so these results are, like all before,
    "what the arm can do from this body motion", not yet balance.
  * the bowling (right) arm is the G1's 7-joint arm: shoulder pitch, roll, yaw, elbow, wrist roll, pitch, yaw, with
    the G1's link lengths, masses and joint ranges, driven by 7 torque motors with the URDF's limits: 25 N m and
    37 rad/s (shoulder, elbow, wrist roll), 5 N m and 22 rad/s (wrist pitch, yaw), linear torque-speed (a DC motor,
    envs/arm_limits.py "g1"), the full torque when braking. Each joint keeps the G1's armature (0.01) and friction
    loss (0.3 N m); viscous damping is 5% of peak / no-load speed, for numerical stability only.
  * the ball is held at a point on the G1's rubber hand (G1_GRIP, 8.5 cm along the hand) and leaves at the release
    angle, exactly as with the human arm (the robot's hand has no fingers: holding it is assumed, as before).
  * legality: the G1's elbow joint is 0 with the forearm at right angles to the upper arm and +90 deg straight
    (range -60..+120 deg), so the elbow FLEXION fed to envs/legality is 90 - q (deg): straight = 0, the robot can
    hyperextend 30 deg. The ICC 15 deg rule is applied to it unchanged.

Everything else (release window, arm plane, line and length, drag, timing jitter, motor noise, the tolerance reward,
learned release, constrained RL) is KinematicBowlEnv2's, so kin_frontier, kin_equifinality, bc_from_frontier and
train_runup work on it unchanged (kin_frontier --robot g1).

Observation (20): phase (3), the arm's 7 joint angles and 7 velocities / 10, hand relative to the shoulder (3).
Action (7): the arm motors' commands in [-1, 1] (x peak torque).
"""
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from envs.kinematic_bowl_env import KinematicBowlEnv, BODY_ARMATURE
from envs.kinematic_bowl_env2 import KinematicBowlEnv2
from envs.runup_bowl_env import RunupBowlEnv

ROOT = Path(__file__).resolve().parent.parent
G1_XML = ROOT / "assets" / "unitree_g1" / "g1.xml"
MOCAP_DIR = ROOT / "data" / "mocap"
G1_GRIP = np.array([0.085, 0.0, 0.0])
ARM = ("right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
       "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint")
# URDF <limit effort velocity> (unitree_ros g1_29dof.urdf; the Menagerie forcerange agrees)
ARM_TORQUE = (25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0)
ARM_OMEGA = (37.0, 37.0, 37.0, 37.0, 37.0, 22.0, 22.0)


def build_g1(timestep=0.002, arm_torque=ARM_TORQUE):
    spec = mujoco.MjSpec.from_file(str(G1_XML))
    for a in list(spec.actuators):
        spec.delete(a)
    for k in list(spec.keys):
        spec.delete(k)
    spec.option.timestep = timestep
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    for name, tq in zip(ARM, arm_torque):
        a = spec.add_actuator()
        a.name, a.target = name.replace("_joint", "_motor"), name
        a.trntype = mujoco.mjtTrn.mjTRN_JOINT
        a.gear = [tq, 0, 0, 0, 0, 0]
        a.ctrllimited, a.ctrlrange = True, [-1.0, 1.0]
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 1080
    spec.visual.headlight.ambient, spec.visual.headlight.diffuse = [0.45] * 3, [0.6] * 3    # videos only
    # a floor and a light for the videos (contacts are off in the env: the body is prescribed)
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[60, 20, 0.1], rgba=[0.45, 0.6, 0.4, 1])
    spec.worldbody.add_light(pos=[0, 0, 6], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    spec.body("right_wrist_yaw_link").add_site(name="right_grip", pos=G1_GRIP, size=[0.036, 0, 0], rgba=[0.8, 0.1, 0.1, 1])
    return spec.compile()


def load_g1_reference(delivery):
    z = np.load(MOCAP_DIR / f"ref_G1_{delivery}.npz")
    return {"qpos": z["qpos"], "qvel": z["qvel"], "rate": float(z["rate"]), "release_index": int(z["release_index"]),
            "scale": float(z["scale"])}


class G1BowlEnv(KinematicBowlEnv2):
    """KinematicBowlEnv2's task and options on the G1 (its __init__ replaces the skeleton-building chain)."""

    def __init__(self, delivery="T3aL", start_before_release_s=(0.8, 1.2), torque_noise_sd=0.0,
                 release_timing_sd_ms=0.0, arm_handover_s=None, drag_k=0.0, elbow_solref=(0.004, 1.0),
                 tolerance_reward_sd_ms=0.0, learned_release=False, target_sample=None, tolerance_target=0.0,
                 lagrangian=False, arm_torque_scale=1.0, frame_skip=10, rsi=True, seed=None,
                 release_angle_deg=250.0, target_range=(10.9, 12.9), task_weight=10.0, speed_weight=0.5,
                 speed_power=0.0, speed_target_kmh=68.0, illegal_penalty=2.0, max_release_elbow_deg=40.0,
                 window_before_s=0.15, late_s=0.15, max_arm_out_of_plane_deg=50.0, line_tol_m=0.5,
                 limits="g1", **unused):
        gym.Env.__init__(self)
        if limits not in ("g1", None):
            raise ValueError("G1BowlEnv has the G1's own arm limits")
        unused.pop("elbow_solref", None)
        self.m = m = build_g1(arm_torque=tuple(t * arm_torque_scale for t in ARM_TORQUE))
        self.d = mujoco.MjData(m)
        self.ref = load_g1_reference(delivery)
        assert self.ref["qpos"].shape[1] == m.nq, "re-run scripts/retarget_g1.py"
        self.frame_skip, self.dt = frame_skip, m.opt.timestep * frame_skip
        self.rsi, self.rng = rsi, np.random.default_rng(seed)
        self.t = 0.0
        # -- RunupBowlEnv's delivery settings
        self.release_angle_deg = release_angle_deg
        self.target_min, self.target_max = target_range
        self.task_weight, self.speed_weight, self.speed_power = task_weight, speed_weight, speed_power
        self.speed_target_m_s = speed_target_kmh / 3.6
        self.illegal_penalty, self.max_release_elbow_deg = illegal_penalty, max_release_elbow_deg
        self.max_arm_out_of_plane_deg, self.line_tol_m = max_arm_out_of_plane_deg, line_tol_m
        self.follow_through_s = 0.0
        self.t_ref_last = (len(self.ref["qpos"]) - 1) / self.ref["rate"]
        self.t_ref_release = self.ref["release_index"] / self.ref["rate"]
        self.t_window = self.t_ref_release - window_before_s
        self.t_end = self.t_ref_release + late_s
        self._hand = m.body("right_wrist_yaw_link").id
        self._shoulder = self._shoulder_body = m.body("right_shoulder_roll_link").id
        self._elbow_body = m.body("right_elbow_link").id
        self._elbow_q = m.jnt_qposadr[m.joint("right_elbow_joint").id]
        self._grip_site = m.site("right_grip").id
        # -- KinematicBowlEnv's prescribed body
        self.drag_k = float(drag_k)
        m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        arm_j = [m.joint(n).id for n in ARM]
        self._arm_dofs = np.array([m.jnt_dofadr[j] for j in arm_j])
        self._arm_q = np.array([m.jnt_qposadr[j] for j in arm_j])
        self._body_dofs = np.setdiff1d(np.arange(m.nv), self._arm_dofs)
        self._body_q = np.setdiff1d(np.arange(m.nq), self._arm_q)
        m.dof_armature[self._body_dofs] = BODY_ARMATURE
        m.dof_damping[self._body_dofs] = 0.0
        m.dof_frictionloss[self._body_dofs] = 0.0
        if elbow_solref is not None:
            m.jnt_solref[m.joint("right_elbow_joint").id] = elbow_solref
        self._arm_acts = np.arange(m.nu)
        tq = np.asarray(ARM_TORQUE) * arm_torque_scale
        m.dof_damping[self._arm_dofs] = 0.05 * tq / np.asarray(ARM_OMEGA)
        self.arm_control = "torque"
        self.torque_noise_sd = torque_noise_sd
        self.release_timing_sd_ms = release_timing_sd_ms
        self._noise_rng = np.random.default_rng(seed)
        self.release_delay_s = 0.0
        V = self.ref["qvel"]
        self._ref_acc = np.vstack([np.diff(V, axis=0) * self.ref["rate"], np.zeros((1, m.nv))])
        self._arm_action = np.arange(len(ARM))
        self.start_range = start_before_release_s
        self.arm_handover_s = arm_handover_s
        self.t_handover = (self.t_ref_release - arm_handover_s) if arm_handover_s is not None else -np.inf
        self._ref_arm = RunupBowlEnv._captured_arm_angles(self)
        # -- KinematicBowlEnv2's options (limits: the G1's DC torque-speed curve, envs/arm_limits "g1")
        self.tolerance_reward_sd_s = tolerance_reward_sd_ms / 1000.0
        self.tolerance_target = tolerance_target
        self.learned_release = learned_release
        self.target_sample = target_sample
        self.actuator, self.omega_max, self.torque_scale = "damping", 40.0, 1.0
        self.limits = "g1"
        self.lagrangian, self.lam = lagrangian, 0.0
        self._gear_norm = m.actuator_gear[:, 0].copy()
        self._continues = self.tolerance_reward_sd_s > 0 or self.release_timing_sd_ms > 0
        self.action_space = spaces.Box(-1.0, 1.0, (len(ARM) + (1 if learned_release else 0),), np.float32)
        self._target_code = 0.0
        self.suppress_release = False
        self.fixed_target_m = None
        self._outcome = None
        n_obs = len(self._obs2())
        self.observation_space = spaces.Box(-np.inf, np.inf, (n_obs,), np.float32)

    # -- the G1's hand and elbow ------------------------------------------------
    def _grip(self):
        return self.d.site_xpos[self._grip_site].copy()

    def _grip_velocity(self):
        v6 = np.zeros(6)
        mujoco.mj_objectVelocity(self.m, self.d, mujoco.mjtObj.mjOBJ_SITE, self._grip_site, v6, 0)
        return v6[3:]

    def _elbow_flexion_deg(self):
        return 90.0 - float(np.degrees(self.d.qpos[self._elbow_q]))

    def start_arm_tracking(self):
        RunupBowlEnv.start_arm_tracking(self)
        self._elbow_hist = [self._elbow_flexion_deg()]

    def _track_arm(self):
        a = self._arm_angle_raw()
        a += 360.0 * np.round((self._arm_hist[-1] - a) / 360.0)
        self._arm_hist.append(a)
        self._elbow_hist.append(self._elbow_flexion_deg())

    def _drive_body(self, t):
        KinematicBowlEnv._drive_body(self, t)

    def captured_arm_torques(self, before_s=1.0):
        """Inverse dynamics: the arm commands (x peak, unclipped) that would make the G1 arm follow the RETARGETED
        captured arm, per 20 ms decision from before_s ahead of release, with the body as prescribed. A warm start
        for the optimiser ("bowl like the human, with the robot's motors"); > 1 means the robot is too weak."""
        m, d = self.m, mujoco.MjData(self.m)
        i0 = int(round(self.ref["release_index"] - before_s * self.ref["rate"]))
        n = int(np.ceil((self.t_end - i0 / self.ref["rate"]) / self.dt)) + 1
        out = []
        for k in range(n):
            t = i0 / self.ref["rate"] + k * self.dt
            j = int(np.clip(t * self.ref["rate"], 0, len(self._ref_acc) - 1))
            from envs.humanoid_bowler import ref_at
            q, v = ref_at(self.ref, min(t, self.t_ref_last), m)
            d.qpos[:], d.qvel[:] = q, v
            d.qacc[:] = self._ref_acc[j]
            mujoco.mj_inverse(m, d)
            out.append(d.qfrc_inverse[self._arm_dofs] / self._gear_norm)
        return np.array(out)
