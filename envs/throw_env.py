"""Minimal two-joint throwing environment.

Validates three things before any humanoid/legality work starts:
ball physics, the weld-release mechanism, and target-zone scoring.
No humanoid, no run-up -- see context.md.

Elbow legality (ICC: extension <= 15 deg between arm-horizontal and
release) is now folded into the real reward, not just logged -- see
elbow-legality-design-decision and workflow-constraints memory. Once
run4's throws were fast and accurate, watching the trained policy live
showed why: the shoulder barely moved (100 -> 55 deg) and all the speed
came from an elbow flick, since nothing in the reward asked for
anything resembling a real bowling swing. Gating the accuracy/speed
bonus on legality (same "subject to" pattern speed already uses for
accuracy) forces a technique that actually swings the arm through
horizontal, since that's the only way to have a measurable, legal
extension at all.
"""
from pathlib import Path

import mujoco
import numpy as np

from envs.legality import elbow_extension_deg

ASSET_PATH = Path(__file__).resolve().parent.parent / "assets" / "throw_arm.xml"


class ThrowEnv:
    def __init__(self, model_path=ASSET_PATH, target_range=(6.0, 8.0), max_steps=900,
                 start_pose_deg=(100.0, 0.0), speed_weight=0.1, min_release_step=100,
                 max_legal_extension_deg=15.0, horizontal_selector="last_before_release",
                 extension_mode="endpoint"):
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
        # a random/untrained policy's release action is ~50% likely to fire
        # on almost any given step, so with no gate release happens at step
        # 0-1 nearly every episode -- before the arm has swung at all. That
        # collapses every episode into "drop the ball from the start pose",
        # which all score ~identically regardless of the policy's actions,
        # giving PPO no gradient to learn from (see workflow-constraints
        # memory: this is what happened in the first real training run).
        # Gating release below this step count forces a swing phase where
        # actions actually matter. 100 steps = 0.2s at this model's 0.002s
        # timestep; demo_scripted.py's hand-tuned manual release timing is
        # step 155, so 100 leaves room for the policy to find its own
        # timing without reproducing the degenerate instant-drop case.
        self.min_release_step = min_release_step
        # cocked/loaded starting angle for the arm (shoulder, elbow), degrees.
        # 100 deg shoulder = arm hanging down and slightly behind the body,
        # like the bottom of a bowler's backswing -- not pointing at the
        # target, so the throw doesn't need an unnatural "wind away first"
        # phase before it can swing forward.
        self.start_pose = np.radians(start_pose_deg)

        # ICC legality threshold + the two operational-definition choices
        # still pending Dr. Felton's reply (see elbow-legality-design-
        # decision memory) -- kept as config, not hardcoded, until resolved.
        self.max_legal_extension_deg = max_legal_extension_deg
        self.horizontal_selector = horizontal_selector
        self.extension_mode = extension_mode

        self.released = False
        self.landed = False
        self.step_count = 0
        self.release_speed = None
        self.landing_pos = None
        self.elbow_extension_deg = None
        self._shoulder_hist = []
        self._elbow_hist = []
        self._release_idx = None

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
        self.elbow_extension_deg = None
        self._shoulder_hist = [np.degrees(self.data.qpos[0])]
        self._elbow_hist = [np.degrees(self.data.qpos[1])]
        self._release_idx = None
        mujoco.mj_forward(self.model, self.data)
        return self._obs()

    def step(self, action):
        """action: [shoulder_torque, elbow_torque, release] each in [-1, 1].
        release > 0 triggers a one-shot release of the weld."""
        action = np.asarray(action, dtype=np.float64)
        self.data.ctrl[:2] = np.clip(action[:2], -1.0, 1.0)

        just_released = False
        if not self.released and self.step_count >= self.min_release_step and action[2] > 0.0:
            self.release_speed = self._ball_linear_speed()
            self.data.eq_active[self.grip_eq_id] = 0
            self.released = True
            just_released = True

        mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        self._shoulder_hist.append(np.degrees(self.data.qpos[0]))
        self._elbow_hist.append(np.degrees(self.data.qpos[1]))
        if just_released:
            self._release_idx = len(self._shoulder_hist) - 1
            self.elbow_extension_deg = elbow_extension_deg(
                self._shoulder_hist, self._elbow_hist, self._release_idx,
                horizontal_selector=self.horizontal_selector, mode=self.extension_mode)

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
            "elbow_extension_deg": self.elbow_extension_deg,
            "legal": (self.elbow_extension_deg is not None
                      and self.elbow_extension_deg <= self.max_legal_extension_deg),
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
        # speed only counts once accuracy AND legality are met -- this is
        # the "subject to" in the research question (max speed subject to
        # (1) legality, (2) landing accuracy), not a free-standing speed
        # bonus, so a fast/accurate-but-illegal throw scores the same as a
        # miss, never better. elbow_extension_deg is None if the arm never
        # swung through a horizontal reference before release (e.g. an
        # elbow-only flick, no real swing) -- treated as illegal too, since
        # that's not a bowling action the ICC rule could even evaluate.
        if not self.landed:
            if self.step_count < self.max_steps:
                return 0.0
            # timed out with the ball still airborne (e.g. released too
            # vertically to come down within max_steps) -- score it same as
            # a landed miss, using the ball's current position, rather than
            # returning a flat 0.0. A flat 0.0 scores better than almost any
            # real miss and would give PPO an incentive to loft the ball
            # into never landing at all instead of aiming for the zone (see
            # workflow-constraints memory, run3 local validation).
            x = self._ball_pos()[0]
        else:
            x = self.landing_pos[0]

        legal = (self.elbow_extension_deg is not None
                 and self.elbow_extension_deg <= self.max_legal_extension_deg)
        if self.target_min <= x <= self.target_max and legal:
            speed = self.release_speed if self.release_speed is not None else 0.0
            return 1.0 + self.speed_weight * speed
        dist = min(abs(x - self.target_min), abs(x - self.target_max))
        return -dist
