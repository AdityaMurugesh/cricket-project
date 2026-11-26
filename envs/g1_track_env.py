"""Whole-body motion tracking on the dynamic Unitree G1: can a learned controller keep the robot up through the delivery?

scripts/g1_dynamic.py (27 Nov) showed that the G1's own servos, simply told to follow the retargeted delivery, let it
fall within 0.15-0.45 s: joint servos know nothing about balance. Every legged robot that performs a captured motion
does it with a learned tracking controller (DeepMimic, Peng et al. 2018; the same recipe trains today's G1 motion
policies). This is that recipe, kept minimal:

  * physics: the whole G1 (Menagerie g1.xml) with floor contact (its 8 foot spheres), gravity, Unitree's position
    servos (kp 500, critically damped) clipped to the URDF torque limits. Nothing is prescribed.
  * action (29): a correction added to each servo's target, on top of the reference pose one step ahead:
    target = q_ref(t + dt) + action_scale * a, a in [-1, 1]. With a = 0 this is g1_dynamic's "servos only".
  * observation (~110): phase in the clip; pelvis height, its up-vector and velocities in its own frame; joint angles
    and velocities; and the reference one step ahead relative to the robot (joint-angle gap, pelvis height gap,
    pelvis position gap and heading in the pelvis frame).
  * reward (DeepMimic's terms, weights 0.5/0.1/0.2/0.2): joint angles, joint velocities, hand and foot positions,
    pelvis position and orientation, each exp(-k * error^2) so a perfect step pays 1.
  * episode: starts at a random time of the clip in the reference state (reference state initialisation), ends at
    the clip's end, or early if the robot falls (pelvis < 0.35 m, trunk tilted > 60 deg away from the reference's
    tilt, pelvis > 0.6 m off its path, or anything but a foot touching the floor).

The clip is the delivery stride: --from/--to seconds around the captured release (default -1.0 .. +0.35, G1 clock).
Reference: data/mocap/ref_G1_<delivery>.npz; T3aLf = feet pinned to the floor, Froude-scaled clock
(scripts/retarget_g1.py --feet --froude --suffix f).
"""
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parent.parent
G1_XML = ROOT / "assets" / "unitree_g1" / "g1.xml"
EE = ("left_wrist_yaw_link", "right_wrist_yaw_link", "left_ankle_roll_link", "right_ankle_roll_link")


def build_g1_dynamic(timestep=0.002):
    spec = mujoco.MjSpec.from_file(str(G1_XML))
    for k in list(spec.keys):
        spec.delete(k)
    spec.option.timestep = timestep
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[60, 20, 0.1],
                            rgba=[0.45, 0.6, 0.4, 1])
    spec.worldbody.add_light(pos=[0, 0, 6], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 1080
    spec.visual.headlight.ambient, spec.visual.headlight.diffuse = [0.45] * 3, [0.6] * 3
    return spec.compile()



class G1TrackEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, delivery="T3aLf", t_from=-1.0, t_to=0.35, frame_skip=10, action_scale=0.25, rsi=True,
                 seed=None, max_tilt_deg=60.0, max_root_err_m=0.6, min_height_m=0.35):
        super().__init__()
        self.m = m = build_g1_dynamic()
        self.d = mujoco.MjData(m)
        z = np.load(ROOT / "data" / "mocap" / f"ref_G1_{delivery}.npz")
        self.rq, self.rv, self.rate = z["qpos"].astype(float), z["qvel"].astype(float), float(z["rate"])
        self.t_rel = int(z["release_index"]) / self.rate
        self.t0 = max(self.t_rel + t_from, 0.0)
        self.t1 = min(self.t_rel + t_to, (len(self.rq) - 1) / self.rate)
        self.frame_skip, self.dt = frame_skip, m.opt.timestep * frame_skip
        self.action_scale, self.rsi = action_scale, rsi
        self.max_tilt, self.max_root_err, self.min_h = np.radians(max_tilt_deg), max_root_err_m, min_height_m
        self.rng = np.random.default_rng(seed)
        self.act_q = np.array([m.jnt_qposadr[m.actuator_trnid[a, 0]] for a in range(m.nu)])
        self.act_v = np.array([m.jnt_dofadr[m.actuator_trnid[a, 0]] for a in range(m.nu)])
        self.lo, self.hi = m.actuator_ctrlrange[:, 0].copy(), m.actuator_ctrlrange[:, 1].copy()
        self.ee = [m.body(n).id for n in EE]
        self.pelvis = m.body("pelvis").id
        self.floor = m.geom("floor").id
        self.foot_bodies = {m.body("left_ankle_roll_link").id, m.body("right_ankle_roll_link").id}
        # the reference on the control grid t0 + k dt (an episode always starts on it), with its pelvis rotation
        # matrix and hand/foot positions: computed once, so a step does no interpolation
        self.n_grid = int(np.floor((self.t1 - self.t0) / self.dt + 1e-9)) + 1
        dd = mujoco.MjData(m)
        self.gq, self.gv = np.zeros((self.n_grid + 1, m.nq)), np.zeros((self.n_grid + 1, m.nv))
        self.gR, self.gee = np.zeros((self.n_grid + 1, 3, 3)), np.zeros((self.n_grid + 1, len(self.ee), 3))
        for k in range(self.n_grid + 1):
            q, v = self._interp(self.t0 + k * self.dt)
            dd.qpos[:] = q
            mujoco.mj_kinematics(m, dd)
            self.gq[k], self.gv[k] = q, v
            self.gR[k] = dd.xmat[self.pelvis].reshape(3, 3)
            self.gee[k] = dd.xpos[self.ee]
        self.t1 = self.t0 + (self.n_grid - 1) * self.dt
        self.action_space = spaces.Box(-1.0, 1.0, (m.nu,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (self._obs_dim(),), np.float32)
        self.k = 0
        self.t = self.t0

    # -- the reference at any time t (s on the reference clock); used only to build the grid
    def _interp(self, t):
        f = np.clip(t * self.rate, 0, len(self.rq) - 1.000001)
        i, w = int(f), f - int(f)
        q = (1 - w) * self.rq[i] + w * self.rq[i + 1]
        r = Slerp([0, 1], Rotation.from_quat([self.rq[i, [4, 5, 6, 3]], self.rq[i + 1, [4, 5, 6, 3]]]))([w])
        q[3:7] = r.as_quat()[0][[3, 0, 1, 2]]
        v = (1 - w) * self.rv[i] + w * self.rv[i + 1]
        return q, v

    def _obs_dim(self):
        return 1 + 1 + 3 + 3 + 3 + 29 + 29 + 29 + 1 + 3 + 3

    def _obs(self):
        d, k = self.d, min(self.k + 1, self.n_grid)
        R = d.xmat[self.pelvis].reshape(3, 3)          # pelvis -> world
        qn, Rn = self.gq[k], self.gR[k]
        phase = self.k / (self.n_grid - 1)
        return np.concatenate([
            [phase], [d.qpos[2]], R[2], R.T @ d.qvel[:3], d.qvel[3:6],
            d.qpos[self.act_q], 0.1 * d.qvel[self.act_v],
            qn[self.act_q] - d.qpos[self.act_q], [qn[2] - d.qpos[2]],
            R.T @ (qn[:3] - d.qpos[:3]), rotvec(R.T @ Rn)]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = options or {}
        k = options.get("k")
        if k is None:
            k = int(self.rng.integers(0, self.n_grid - 5)) if self.rsi else 0
        self.k = int(k)
        self.t = self.t0 + self.k * self.dt
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:], self.d.qvel[:] = self.gq[self.k], self.gv[self.k]
        self.d.ctrl[:] = self.gq[self.k][self.act_q]
        mujoco.mj_forward(self.m, self.d)
        return self._obs(), {}

    def _bad_contact(self):
        d, m = self.d, self.m
        for i in range(d.ncon):
            c = d.contact[i]
            if c.geom1 == self.floor or c.geom2 == self.floor:
                other = c.geom2 if c.geom1 == self.floor else c.geom1
                if m.geom_bodyid[other] not in self.foot_bodies:
                    return True
        return False

    def step(self, action):
        a = np.clip(np.asarray(action, float), -1.0, 1.0)
        k = self.k + 1
        q, v, ee, Rref = self.gq[k], self.gv[k], self.gee[k], self.gR[k]
        self.d.ctrl[:] = np.clip(q[self.act_q] + self.action_scale * a, self.lo, self.hi)
        mujoco.mj_step(self.m, self.d, nstep=self.frame_skip)
        self.k, self.t = k, self.t0 + k * self.dt
        d = self.d
        # tracking errors against the reference at the new time
        ej = d.qpos[self.act_q] - q[self.act_q]
        ev = d.qvel[self.act_v] - v[self.act_v]
        eee = d.xpos[self.ee] - ee
        er = d.qpos[:3] - q[:3]
        R = d.xmat[self.pelvis].reshape(3, 3)
        eo = np.arccos(np.clip((np.trace(Rref.T @ R) - 1.0) / 2.0, -1.0, 1.0))
        r_p = np.exp(-5.0 * np.mean(ej ** 2))
        r_v = np.exp(-0.02 * np.mean(ev ** 2))
        r_e = np.exp(-20.0 * np.mean(np.sum(eee ** 2, 1)))
        r_r = np.exp(-10.0 * np.sum(er ** 2) - 2.0 * eo ** 2)
        reward = 0.5 * r_p + 0.1 * r_v + 0.2 * r_e + 0.2 * r_r
        # falls: the trunk's up-vector against the reference's, the pelvis height and path, a non-foot contact
        tilt_err = np.arccos(np.clip(R[:, 2] @ Rref[:, 2], -1.0, 1.0))
        fell = (d.qpos[2] < self.min_h or tilt_err > self.max_tilt or np.linalg.norm(er) > self.max_root_err
                or not np.isfinite(d.qpos).all() or self._bad_contact())
        done_clip = self.k >= self.n_grid - 1
        info = {"fell": bool(fell), "t_rel_release": self.t - self.t_rel,
                "joint_err_rad": float(np.sqrt(np.mean(ej ** 2))), "root_err_m": float(np.linalg.norm(er)),
                "ee_err_m": float(np.sqrt(np.mean(np.sum(eee ** 2, 1)))), "completed": bool(done_clip and not fell)}
        if fell:
            reward = 0.0
        return self._obs(), float(reward), bool(fell), bool(done_clip and not fell), info


def rotvec(M):
    """Rotation vector (axis x angle) of a rotation matrix."""
    ang = np.arccos(np.clip((np.trace(M) - 1.0) / 2.0, -1.0, 1.0))
    w = 0.5 * np.array([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]])
    s = np.sin(ang)
    return w * (ang / s) if s > 1e-6 else w
