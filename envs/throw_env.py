"""Minimal two-joint throwing environment.

Validates three things before any humanoid/legality work starts:
ball physics, the weld-release mechanism, and target-zone scoring.
No humanoid, no legality constraint, no run-up -- see context.md.
"""
from pathlib import Path

import mujoco
import numpy as np

ASSET_PATH = Path(__file__).resolve().parent.parent / "assets" / "throw_arm.xml"


class ThrowEnv:
    def __init__(self, model_path=ASSET_PATH, target_range=(6.0, 8.0), max_steps=900,
                 start_pose_deg=(100.0, 0.0), speed_weight=0.1):
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        self.grip_eq_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "grip")
        self.ball_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self.ball_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
        self.ground_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        self.hand_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "hand")

        self.target_min, self.target_max = target_range
        self.max_steps = max_steps
        self.speed_weight = speed_weight
        # cocked/loaded starting angle for the arm (shoulder, elbow), degrees.
        # 100 deg shoulder = arm hanging down and slightly behind the body,
        # like the bottom of a bowler's backswing -- not pointing at the
        # target, so the throw doesn't need an unnatural "wind away first"
        # phase before it can swing forward.
        self.start_pose = np.radians(start_pose_deg)

        self.released = False
        self.landed = False
        self.step_count = 0
        self.release_speed = None
        self.landing_pos = None

        self.reset()

    def reset(self, seed=None):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0], self.data.qpos[1] = self.start_pose
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

        # place the ball exactly at the hand for this starting pose, then
        # let the weld hold it there -- the weld's relative-pose target is
        # body-frame-local, so this works for any starting joint angles.
        hand_pos = self.data.site_xpos[self.hand_site_id].copy()
        hand_quat = np.empty(4)
        mujoco.mju_mat2Quat(hand_quat, self.data.site_xmat[self.hand_site_id].copy())
        self.data.qpos[2:5] = hand_pos
        self.data.qpos[5:9] = hand_quat

        self.data.eq_active[self.grip_eq_id] = 1
        self.released = False
        self.landed = False
        self.step_count = 0
        self.release_speed = None
        self.landing_pos = None
        mujoco.mj_forward(self.model, self.data)
        return self._obs()

    def step(self, action):
        """action: [shoulder_torque, elbow_torque, release] each in [-1, 1].
        release > 0 triggers a one-shot release of the weld."""
        action = np.asarray(action, dtype=np.float64)
        self.data.ctrl[:2] = np.clip(action[:2], -1.0, 1.0)

        if not self.released and action[2] > 0.0:
            self.release_speed = self._ball_linear_speed()
            self.data.eq_active[self.grip_eq_id] = 0
            self.released = True

        mujoco.mj_step(self.model, self.data)
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
        return np.concatenate([self.data.qpos[:2], self.data.qvel[:2]]).astype(np.float32)

    def _ball_pos(self):
        return self.data.xpos[self.ball_body_id].copy()

    def _ball_linear_speed(self):
        # free joint qvel layout: [0:3] linear, [3:6] angular, both world frame
        linvel = self.data.qvel[2:5]
        return float(np.linalg.norm(linvel))

    def _ball_touched_ground(self):
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            geoms = {c.geom1, c.geom2}
            if self.ball_geom_id in geoms and self.ground_geom_id in geoms:
                return True
        return False

    def _reward(self):
        # speed only counts once the accuracy constraint is met -- this is
        # the "subject to" in the research question (max speed subject to
        # landing accuracy), not a free-standing speed bonus, so a fast
        # miss can never outscore an accurate throw of any speed.
        if not self.landed:
            return 0.0
        x = self.landing_pos[0]
        if self.target_min <= x <= self.target_max:
            return 1.0 + self.speed_weight * self.release_speed
        dist = min(abs(x - self.target_min), abs(x - self.target_max))
        return -dist
