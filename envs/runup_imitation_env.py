"""Learn the captured run-up and trunk motion on the physics bowler (DeepMimic-style).

The policy drives every joint of envs/humanoid_bowler.py through PD targets and
is rewarded for following the mocap reference (data/mocap/ref_<D>.npz) from the
start of the run-up to release. The root (pelvis) is NOT actuated: staying up,
running in and jumping into the delivery must come from the joints, so a
policy that tracks the reference has learned a physically valid run-up.

What is imitated, and what is not:
  * pelvis path, legs, trunk (lumbar), head and the front (left) arm: tracked
  * the bowling (right) arm: NOT tracked and NOT driven from the reference. Its
    PD targets are absolute (action mapped onto the joint range), so whatever it
    does is the policy's own; the bowling task (legality from the simulated
    joint angles, landing zone, pace) is added on top of this later. This is the
    project rule: a human reference may shape gross body style, never the elbow.

Reward per control step, as in DeepMimic (Peng et al. 2018), each term in (0, 1]:
    0.65 pose   exp(-2  * weighted sum of squared joint-angle errors)
    0.10 vel    exp(-0.1 * sum of squared joint-velocity errors)
    0.15 ends   exp(-40 * sum of squared errors of feet, front hand and head,
                 relative to the pelvis)
    0.10 root   exp(-10 * (pelvis position error^2 + 0.5 * orientation error^2))
The lumbar joint is weighted 2x ("trunk reference").

Episodes start at a random reference frame (reference state initialisation)
and end at release (terminal: there is nothing after it to bootstrap), or
early if the pelvis drops/rises 0.3 m away from the reference or tilts more
than 1 rad from it (a fall).
"""
import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from envs.humanoid_bowler import BOWLING_ARM, Joints, build_model, load_reference, ref_at

POSE_WEIGHT = {"lumbar": 2.0, "neck": 0.5, "hip": 1.0, "knee_angle": 1.0, "ankle_angle": 0.5,
               "subtalar_angle": 0.2, "mtp_angle": 0.2, "shoulder": 0.5, "elbow_flex": 0.3,
               "pro_sup": 0.1, "wrist_flex": 0.1, "wrist_dev": 0.1}
END_BODIES = ("calcn_r", "calcn_l", "hand_l", "head")
FALL_HEIGHT_M = 0.3
FALL_TILT_RAD = 1.0


def _base(name):
    return name[:-2] if name.endswith(("_r", "_l")) else name


def _quat_angle(q1, q2):
    """Rotation angle between two unit quaternions."""
    return 2.0 * np.arccos(min(1.0, abs(float(np.dot(q1, q2)))))


class RunupImitationEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, delivery="T3a", frame_skip=10, action_scale=0.5, rsi=True, seed=None,
                 arm_scale=1.0):
        super().__init__()
        self.m = build_model(arm_scale=arm_scale)
        self.d = mujoco.MjData(self.m)
        self.J = Joints(self.m)
        self.ref = load_reference(self.m, delivery)
        self.frame_skip = frame_skip
        self.dt = self.m.opt.timestep * frame_skip
        self.action_scale = action_scale
        self.rsi = rsi
        self.t_end = self.ref["release_index"] / self.ref["rate"]
        self.rng = np.random.default_rng(seed)

        # action layout follows the actuators: joint by joint, 3 per ball joint
        self._act_slices = []
        k = 0
        for name, t, qa, da, acts, gears in self.J.items:
            n = len(acts)
            self._act_slices.append((name, t, qa, slice(k, k + n), name in BOWLING_ARM,
                                     self.m.jnt_range[self.m.joint(name).id]))
            k += n
        self.n_act = k
        self.action_space = spaces.Box(-1.0, 1.0, (self.n_act,), np.float32)
        n_obs = len(self._obs(0.0))
        self.observation_space = spaces.Box(-np.inf, np.inf, (n_obs,), np.float32)

        self._body = {b: self.m.body(b).id for b in END_BODIES + ("pelvis",)}
        self._tracked = [(name, t, qa, da, POSE_WEIGHT[_base(name)])
                         for name, t, qa, da, acts, gears in self.J.items if name not in BOWLING_ARM]
        self._tracked_dofs = np.concatenate([
            np.arange(da, da + (3 if t == mujoco.mjtJoint.mjJNT_BALL else 1))
            for name, t, qa, da, w in self._tracked])
        self._ref_data = mujoco.MjData(self.m)
        self.t = 0.0

    # -- reference helpers -------------------------------------------------
    def _ref_ends(self, q_ref):
        self._ref_data.qpos[:] = q_ref
        mujoco.mj_kinematics(self.m, self._ref_data)
        pel = self._ref_data.xpos[self._body["pelvis"]]
        return np.array([self._ref_data.xpos[self._body[b]] - pel for b in END_BODIES])

    # -- gym API -----------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        last = getattr(self, "rsi_last_index", self.ref["release_index"] - 10)
        i0 = int(self.rng.integers(0, last)) if self.rsi else 0
        self.t = i0 / self.ref["rate"]
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:] = self.ref["qpos"][i0]
        self.d.qvel[:] = self.ref["qvel"][i0]
        mujoco.mj_forward(self.m, self.d)
        self._start_t = self.t
        return self._obs(self.t), {}

    def _targets(self, action):
        """PD targets: reference + offset for imitated joints, absolute for the bowling arm."""
        q_ref, _ = ref_at(self.ref, self.t + self.dt, self.m)
        target = q_ref.copy()
        a = np.clip(action, -1.0, 1.0)
        for name, t, qa, sl, bowling, rng in self._act_slices:
            if t == mujoco.mjtJoint.mjJNT_HINGE:
                if bowling:
                    lo, hi = rng
                    target[qa] = 0.5 * (lo + hi) + 0.5 * (hi - lo) * a[sl][0]
                else:
                    target[qa] = q_ref[qa] + self.action_scale * a[sl][0]
            else:
                base = np.array([1.0, 0, 0, 0]) if bowling else q_ref[qa:qa + 4]
                # bowling shoulder: up to a full turn, so the over-the-back swing to
                # release (~205 deg about local -z) is one continuous action
                scale = 2 * np.pi if bowling else self.action_scale
                # actuator order is local x, z, y; the rotation vector is (x, y, z)
                rv = np.zeros(3)
                for k, axis in enumerate("xzy"):
                    rv["xyz".index(axis)] = scale * a[sl][k]
                dq = np.zeros(4)
                mujoco.mju_axisAngle2Quat(dq, rv / (np.linalg.norm(rv) + 1e-12), np.linalg.norm(rv))
                out = np.zeros(4)
                mujoco.mju_mulQuat(out, base, dq)
                target[qa:qa + 4] = out
        return target

    def step(self, action):
        target = self._targets(action)
        for _ in range(self.frame_skip):
            self.d.ctrl[:] = self.J.pd_torques(self.d, target)
            mujoco.mj_step(self.m, self.d)
        self.t += self.dt
        q_ref, v_ref = ref_at(self.ref, self.t, self.m)
        r, terms = self._reward(q_ref, v_ref)
        fell = (abs(self.d.qpos[2] - q_ref[2]) > FALL_HEIGHT_M
                or _quat_angle(self.d.qpos[3:7], q_ref[3:7]) > FALL_TILT_RAD
                or not np.isfinite(self.d.qpos).all())
        done = self.t >= self.t_end - 1e-9
        info = {"terms": terms, "fell": bool(fell), "t": self.t, "start_t": self._start_t,
                "reached_release": bool(done and not fell)}
        if fell:
            r = 0.0
        return self._obs(self.t), float(r), bool(fell or done), False, info

    # -- reward and observation ---------------------------------------------
    def _reward(self, q_ref, v_ref):
        q, v = self.d.qpos, self.d.qvel
        pose_err = 0.0
        for name, t, qa, da, w in self._tracked:
            if t == mujoco.mjtJoint.mjJNT_HINGE:
                pose_err += w * (q[qa] - q_ref[qa]) ** 2
            else:
                pose_err += w * _quat_angle(q[qa:qa + 4], q_ref[qa:qa + 4]) ** 2
        vel_err = float(np.sum((v[self._tracked_dofs] - v_ref[self._tracked_dofs]) ** 2))
        pel = self.d.xpos[self._body["pelvis"]]
        ends = np.array([self.d.xpos[self._body[b]] - pel for b in END_BODIES])
        end_err = float(np.sum((ends - self._ref_ends(q_ref)) ** 2))
        root_err = float(np.sum((q[:3] - q_ref[:3]) ** 2)) + 0.5 * _quat_angle(q[3:7], q_ref[3:7]) ** 2
        terms = {"pose": np.exp(-2.0 * pose_err), "vel": np.exp(-0.1 * vel_err),
                 "ends": np.exp(-40.0 * end_err), "root": np.exp(-10.0 * root_err)}
        r = 0.65 * terms["pose"] + 0.10 * terms["vel"] + 0.15 * terms["ends"] + 0.10 * terms["root"]
        return r, terms

    def _obs(self, t):
        q_ref, _ = ref_at(self.ref, min(t, self.t_end), self.m)
        q, v = self.d.qpos, self.d.qvel
        phase = t / self.t_end
        return np.concatenate([
            [phase, np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)],
            [q[2]], q[3:7], q[7:],              # pelvis height and orientation, joint angles
            v[:6] / 10.0, v[6:] / 10.0,          # root and joint velocities (scaled)
            q[:3] - q_ref[:3],                   # where the pelvis is vs the reference
        ]).astype(np.float32)
