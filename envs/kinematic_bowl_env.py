"""The captured bowler with a physics bowling arm: the run-up is the mocap, the arm is learned.

Stage 1 could not learn the run-up: the fitted reference's feet sink up to 5 cm
into the floor and slide at ~1 m/s while planted, so no physical body can follow
it (docs/two_week_plan.md, 2026-10-24). Here the whole body except the right arm
moves exactly as captured, and only the arm (shoulder, elbow, forearm, wrist) is
simulated and driven by the policy. Everything about the delivery is RunupBowlEnv's:
release angle, window, arm plane, line and length, legality from the SIMULATED
elbow (never from mocap).

How the body is prescribed: every non-arm DoF gets a huge armature A and the
generalised force A * qacc_ref. Its own equation of motion then gives
qacc = qacc_ref + O(1/A), so the arm's reaction cannot push the body, while the
arm still feels the body's acceleration through the coupled mass matrix (the
trunk's rotation whips the arm as it would on a rigid attachment). The body state
is also reset to the reference every physics step, so nothing drifts. A weld or
per-step teleport alone does neither: both distorted the arm in the v1 demo
(envs/bowler_env.py notes). There are no contacts: the body never touches the
floor in a way that matters, it just follows the capture.

Observation: phase, bowling-arm joint angles and velocities, hand relative to
the shoulder. Action: the arm's PD targets (RunupImitationEnv's absolute mapping),
or with arm_control="torque" its motor commands, as ThrowEnv's arm.
Reward: only the delivery (task_weight * RunupBowlEnv's task reward), at its end.
Episodes start 0.8-1.2 s before the captured release (fixed 1.0 s with rsi=False, or
reset(options={"before_s": x})),
with the arm in its captured pose there (an initial condition, nothing more).
"""
import mujoco
import numpy as np
from gymnasium import spaces

from envs.humanoid_bowler import BOWLING_ARM, ref_at
from envs.runup_bowl_env import RunupBowlEnv

BODY_ARMATURE = 1e6


class KinematicBowlEnv(RunupBowlEnv):
    def __init__(self, delivery="T3a", start_before_release_s=(0.8, 1.2), arm_control="pd", arm_top_speed=40.0, **kw):
        kw.setdefault("follow_through_s", 0.0)     # the body cannot fall
        super().__init__(delivery=delivery, **kw)
        m = self.m
        m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        arm_j = [m.joint(n).id for n in BOWLING_ARM]
        arm_dofs = np.concatenate([np.arange(m.jnt_dofadr[j], m.jnt_dofadr[j] + (3 if m.jnt_type[j] == 1 else 1))
                                   for j in arm_j])
        arm_q = np.concatenate([np.arange(m.jnt_qposadr[j], m.jnt_qposadr[j] + (4 if m.jnt_type[j] == 1 else 1))
                                for j in arm_j])
        self._arm_dofs, self._arm_q = arm_dofs, arm_q
        self._body_dofs = np.setdiff1d(np.arange(m.nv), arm_dofs)
        self._body_q = np.setdiff1d(np.arange(m.nq), arm_q)
        m.dof_armature[self._body_dofs] = BODY_ARMATURE
        m.dof_damping[self._body_dofs] = 0.0
        # the skeleton's PD damping (0.2 x peak torque: 50 N m s/rad at the shoulder)
        # caps the arm at ~5 rad/s, far short of a bowling arm's ~20: even PD-tracking
        # the captured arm path only reached 124 deg. Here each arm joint's damping is
        # its peak torque / arm_top_speed (rad/s): fast enough to bowl, but bounded
        # (ThrowEnv-like 0.3 let random actions spin the light arm to NaN)
        for j in arm_j:
            da = m.jnt_dofadr[j]
            n = 3 if m.jnt_type[j] == 1 else 1
            acts = [a for a in range(m.nu) if m.actuator_trnid[a, 0] == j]
            m.dof_damping[da:da + n] = np.abs(m.actuator_gear[acts, :3]).max() / arm_top_speed
        self.arm_control = arm_control
        self._arm_acts = np.array([a for a in range(m.nu) if m.actuator_trnid[a, 0] in arm_j])
        # reference acceleration of every DoF, per frame (velocity is a forward difference,
        # so this is constant between frames i and i + 1)
        V = self.ref["qvel"]
        self._ref_acc = np.vstack([np.diff(V, axis=0) * self.ref["rate"], np.zeros((1, m.nv))])
        self._full_action = np.zeros(self.n_act, dtype=np.float64)
        self._arm_action = np.concatenate([np.arange(sl.start, sl.stop) for name, t, qa, sl, bowling, rng
                                           in self._act_slices if bowling])
        self.start_range = start_before_release_s
        self.action_space = spaces.Box(-1.0, 1.0, (len(self._arm_action),), np.float32)
        self._shoulder_body = self.m.body("humerus_r").id
        n_obs = len(self._kin_obs())
        self.observation_space = spaces.Box(-np.inf, np.inf, (n_obs,), np.float32)

    # -- the prescribed body ------------------------------------------------
    def _drive_body(self, t):
        q, v = ref_at(self.ref, min(t, self.t_ref_last), self.m)
        self.d.qpos[self._body_q] = q[self._body_q]
        self.d.qvel[self._body_dofs] = v[self._body_dofs]
        i = int(np.clip(t * self.ref["rate"], 0, len(self._ref_acc) - 1))
        self.d.qfrc_applied[:] = 0.0
        self.d.qfrc_applied[self._body_dofs] = BODY_ARMATURE * self._ref_acc[i, self._body_dofs]

    def _kin_obs(self):
        phase = (self.t - self.t_ref_release) / 1.0
        hand = self.d.xpos[self._hand] - self.d.xpos[self._shoulder_body]
        return np.concatenate([[phase, np.sin(np.pi * phase), np.cos(np.pi * phase)],
                               self.d.qpos[self._arm_q], self.d.qvel[self._arm_dofs] / 10.0,
                               hand]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        lo, hi = self.start_range
        before = self.rng.uniform(lo, hi) if self.rsi else 1.0
        if options and "before_s" in options:       # eval / viewer: start this long before release
            before = float(options["before_s"])
        i0 = int(round(self.ref["release_index"] - before * self.ref["rate"]))
        self.t = self._start_t = i0 / self.ref["rate"]
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:] = self.ref["qpos"][i0]
        self.d.qvel[:] = self.ref["qvel"][i0]
        self._drive_body(self.t)
        mujoco.mj_forward(self.m, self.d)
        self.start_arm_tracking()
        return self._kin_obs(), {}

    def step(self, action):
        a = np.clip(action, -1.0, 1.0)
        if self.arm_control == "pd":
            self._full_action[:] = 0.0
            self._full_action[self._arm_action] = a
            target = self._targets(self._full_action)
        task = None
        h = self.m.opt.timestep
        for k in range(self.frame_skip):
            self._drive_body(self.t + k * h)
            self.d.ctrl[:] = 0.0
            if self.arm_control == "pd":
                self.d.ctrl[self._arm_acts] = self.J.pd_torques(self.d, target)[self._arm_acts]
            else:                       # ThrowEnv's: the action is the arm's torques
                self.d.ctrl[self._arm_acts] = a
            mujoco.mj_step(self.m, self.d)
            self._track_arm()
            if self.t >= self.t_window and self._arm_hist[-1] >= self.release_angle_deg:
                why = ("bent arm at the release angle" if abs(self._elbow_hist[-1]) > self.max_release_elbow_deg
                       else self._bowling_position())
                task = self._no_ball(why) if why else self._release()
                break
        self.t += self.dt
        if task is None and self.t >= self.t_end - 1e-9:
            task = self._no_ball("still holding the ball")
        done = task is not None
        r = self.task_weight * task if done else 0.0
        info = {"terms": dict.fromkeys(("pose", "vel", "ends", "root"), 0.0), "fell": False, "t": self.t,
                "start_t": self._start_t, "drift_m": 0.0,
                "reached_release": bool(self._outcome is not None and self._outcome["released"]),
                "task_reward": task, "outcome": self._outcome}
        return self._kin_obs(), float(r), bool(done), False, info
