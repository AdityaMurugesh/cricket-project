"""Humanoid bowler: fully scripted run-up + bowling swing, with the ball
going through real physics from the moment of release.

Both the run-up and the bowling arm swing are kinematic (joint angles set
directly from envs/runup.py's pure functions, no actuators, no forces) --
this is a visual demo of a run-up + bowling action, not the RL-trainable
environment. envs/throw_env.py (2-joint arm, actuator-driven, no run-up)
remains the environment used for actual policy training; see the
run-up-method scope decision.

An earlier version tried to make the arm actuator/torque-driven here too,
holding the root+legs in place with weld constraints. That didn't work:
even a stiff weld has finite compliance, and that was enough give under
the arm's reaction torque to badly distort the throw (release speed and
direction were both wrong). Since the run-up was already scripted, the
simplest fix was to script the whole swing the same way and only let the
ball be dynamically simulated, launched at the hand's velocity (measured
by finite difference) at the scripted release instant.
"""
from pathlib import Path

import mujoco
import numpy as np

from envs.runup import (runup_pose, delivery_arm_angles_deg, LEG_JOINT_NAMES,
                         DELIVERY_STRIDE_DEG, CARRY_ARM_DEG)

ASSET_PATH = Path(__file__).resolve().parent.parent / "assets" / "bowler.xml"


class BowlerEnv:
    def __init__(self, model_path=ASSET_PATH, target_range=(6.0, 8.0), max_steps=900,
                 run_duration=1.6, blend_duration=0.3, run_speed=4.0, start_x=-6.4,
                 pelvis_height=0.80, delivery_duration=0.34):
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep

        self.grip_eq_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "grip")
        self.ball_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self.ball_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
        self.ground_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        self.hand_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "hand")

        self.shoulder_qpos_adr = self.model.joint("shoulder").qposadr[0]
        self.elbow_qpos_adr = self.model.joint("elbow").qposadr[0]
        self.leg_qpos_adr = [self.model.joint(n).qposadr[0] for n in LEG_JOINT_NAMES]
        self.root_qpos_adr = self.model.joint("root_free").qposadr[0]
        self.ball_qpos_adr = self.model.joint("ball_free").qposadr[0]
        self.ball_dof_adr = self.model.joint("ball_free").dofadr[0]

        self.target_min, self.target_max = target_range
        self.max_steps = max_steps

        self.run_duration = run_duration
        self.blend_duration = blend_duration
        self.run_speed = run_speed
        self.start_x = start_x
        self.pelvis_height = pelvis_height
        self.delivery_duration = delivery_duration

        self.released = False
        self.landed = False
        self.step_count = 0
        self.release_speed = None
        self.landing_pos = None
        self._prev_hand_pos = None
        self._frozen_body_qpos = None  # body pose at release, held during flight

        self.reset()

    # ---- kinematic puppeteering (run-up and delivery swing) -----------

    def _write_pose(self, root_xyz, leg_deg, shoulder_deg, elbow_deg):
        x, y, z = root_xyz
        self.data.qpos[self.root_qpos_adr:self.root_qpos_adr + 3] = [x, y, z]
        self.data.qpos[self.root_qpos_adr + 3:self.root_qpos_adr + 7] = [1, 0, 0, 0]
        for adr, name in zip(self.leg_qpos_adr, LEG_JOINT_NAMES):
            self.data.qpos[adr] = np.radians(leg_deg[name])
        self.data.qpos[self.shoulder_qpos_adr] = np.radians(shoulder_deg)
        self.data.qpos[self.elbow_qpos_adr] = np.radians(elbow_deg)
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self._snap_ball_to_hand()

    def _snap_ball_to_hand(self):
        hand_pos = self.data.site_xpos[self.hand_site_id].copy()
        hand_quat = np.empty(4)
        mujoco.mju_mat2Quat(hand_quat, self.data.site_xmat[self.hand_site_id].copy())
        self.data.qpos[self.ball_qpos_adr:self.ball_qpos_adr + 3] = hand_pos
        self.data.qpos[self.ball_qpos_adr + 3:self.ball_qpos_adr + 7] = hand_quat
        mujoco.mj_forward(self.model, self.data)

    def run_up_step(self, t):
        """Advance the scripted run-up to time t seconds (0 to run_duration).
        Root translates, legs cycle, arm stays in the carry pose."""
        root_xyz, legs = runup_pose(t, self.run_duration, self.blend_duration,
                                     self.run_speed, self.start_x, self.pelvis_height)
        self._write_pose(root_xyz, legs, *CARRY_ARM_DEG)

    def delivery_step(self, t):
        """Advance the scripted bowling swing to time t seconds since the
        run-up ended (0 to delivery_duration). Root+legs stay frozen at
        DELIVERY_STRIDE_DEG; the arm eases from carry to release pose."""
        self._prev_hand_pos = self.data.site_xpos[self.hand_site_id].copy()
        root_xyz = (self.data.qpos[self.root_qpos_adr], 0.0, self.pelvis_height)
        shoulder_deg, elbow_deg = delivery_arm_angles_deg(t, self.delivery_duration)
        self._write_pose(root_xyz, DELIVERY_STRIDE_DEG, shoulder_deg, elbow_deg)

    def release_ball(self):
        """Release the ball with the hand's current velocity (finite
        difference against the previous delivery_step frame) and switch it
        to real dynamics. Call once, at the scripted release instant."""
        hand_pos = self.data.site_xpos[self.hand_site_id].copy()
        if self._prev_hand_pos is not None:
            velocity = (hand_pos - self._prev_hand_pos) / self.dt
        else:
            velocity = np.zeros(3)
        self.data.eq_active[self.grip_eq_id] = 0
        self.data.qvel[self.ball_dof_adr:self.ball_dof_adr + 3] = velocity
        self.release_speed = float(np.linalg.norm(velocity))
        self.released = True
        self._frozen_body_qpos = self.data.qpos.copy()
        mujoco.mj_forward(self.model, self.data)

    def follow_through_step(self, t):
        """Carry the scripted swing on past release, for t seconds since
        the delivery started, while the ball is already in free flight.

        The bowler used to stop dead the instant the ball left the hand,
        because flight_step() pins the whole body to its release-instant
        pose. A real delivery carries the arm on down and across the body,
        and that follow-through is most of what makes an action read as
        bowling rather than as a throw. This just keeps updating the pose
        flight_step() pins to, so it costs nothing extra.

        Purely cosmetic: by now the ball is a fully independent free body,
        so nothing here can affect its flight. Deliberately does NOT go
        through _write_pose(), which zeroes qvel and snaps the ball back
        into the hand -- both of which would destroy the throw.
        """
        if self._frozen_body_qpos is None:
            return
        shoulder_deg, elbow_deg = delivery_arm_angles_deg(t, self.delivery_duration)
        self._frozen_body_qpos[self.shoulder_qpos_adr] = np.radians(shoulder_deg)
        self._frozen_body_qpos[self.elbow_qpos_adr] = np.radians(elbow_deg)

    # ---- dynamic free-flight phase (mirrors envs/throw_env.py) --------

    def reset(self, seed=None):
        mujoco.mj_resetData(self.model, self.data)
        self.released = False
        self.landed = False
        self.step_count = 0
        self.release_speed = None
        self.landing_pos = None
        self._prev_hand_pos = None
        self.data.eq_active[self.grip_eq_id] = 1
        self.run_up_step(0.0)
        return self._obs()

    def flight_step(self):
        """Advance real dynamics by one step. Only meaningful after
        release_ball(). The puppeteered body (root/legs/arm) has no
        actuators, so it's teleport-corrected back to its release-instant
        pose every frame -- purely cosmetic, doesn't affect ball physics
        since the ball is a fully independent free body once released."""
        mujoco.mj_step(self.model, self.data)

        if self._frozen_body_qpos is not None:
            frozen = self._frozen_body_qpos
            self.data.qpos[self.root_qpos_adr:self.root_qpos_adr + 7] = frozen[self.root_qpos_adr:self.root_qpos_adr + 7]
            self.data.qpos[self.shoulder_qpos_adr] = frozen[self.shoulder_qpos_adr]
            self.data.qpos[self.elbow_qpos_adr] = frozen[self.elbow_qpos_adr]
            for adr in self.leg_qpos_adr:
                self.data.qpos[adr] = frozen[adr]
            mujoco.mj_forward(self.model, self.data)

        self.step_count += 1

        if self.released and not self.landed and self._ball_touched_ground():
            self.landed = True
            self.landing_pos = tuple(self._ball_pos()[:2])

        reward = self._reward()
        timeout = self.step_count >= self.max_steps
        done = self.landed or timeout
        info = {
            "released": self.released,
            "release_speed": self.release_speed,
            "landing_pos": self.landing_pos,
            "timeout": timeout,
        }
        return self._obs(), reward, done, info

    def _obs(self):
        return np.array([
            self.data.qpos[self.shoulder_qpos_adr],
            self.data.qpos[self.elbow_qpos_adr],
        ], dtype=np.float32)

    def _ball_pos(self):
        return self.data.xpos[self.ball_body_id].copy()

    def _ball_touched_ground(self):
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            geoms = {c.geom1, c.geom2}
            if self.ball_geom_id in geoms and self.ground_geom_id in geoms:
                return True
        return False

    def _reward(self):
        if not self.landed:
            return 0.0
        x = self.landing_pos[0]
        if self.target_min <= x <= self.target_max:
            return 1.0
        dist = min(abs(x - self.target_min), abs(x - self.target_max))
        return -dist
