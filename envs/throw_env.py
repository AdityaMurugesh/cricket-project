"""Two-joint arm that throws a ball at a target zone on the ground."""
from pathlib import Path

import mujoco
import numpy as np

from envs.legality import elbow_extension_deg

ASSET_PATH = Path(__file__).resolve().parent.parent / "assets" / "throw_arm.xml"


class ThrowEnv:
    def __init__(self, model_path=ASSET_PATH, target_range=(6.0, 8.0), max_steps=1200,
                 start_pose_deg=(100.0, 0.0), speed_weight=0.1, min_release_step=0,
                 release_window_deg=(230.0, 310.0), max_release_elbow_deg=40.0,
                 release_mode="target_angle", release_latch=True,
                 max_legal_extension_deg=15.0, illegal_penalty=2.0,
                 horizontal_selector="last_before_release", extension_mode="endpoint"):
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        self.grip_eq_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "grip")
        self.ball_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self.ball_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
        self.ground_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        self.hand_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "hand")

        self.target_min, self.target_max = target_range
        # long enough for a lofted ball to come down
        self.max_steps = max_steps
        self.speed_weight = speed_weight
        # old step-count gate, off by default
        self.min_release_step = min_release_step

        # overarm only: shoulder angle at release, 270 = straight up
        self.release_window_min, self.release_window_max = release_window_deg

        # arm must be near straight to let go (not the ICC metric)
        self.max_release_elbow_deg = max_release_elbow_deg

        # "threshold": fire when action[2] > 0, "target_angle": fire at that angle
        self.release_mode = release_mode

        # pick the target angle once when entering the window
        self.release_latch = release_latch
        # bottom of the backswing, degrees
        self.start_pose = np.radians(start_pose_deg)

        # ICC rule, plus the two definition choices still open
        self.max_legal_extension_deg = max_legal_extension_deg
        self.illegal_penalty = illegal_penalty
        self.horizontal_selector = horizontal_selector
        self.extension_mode = extension_mode

        self.released = False
        self.landed = False
        self.spent = False
        self.step_count = 0
        self.release_speed = None
        self.release_height = None
        self.release_shoulder_deg = None
        self.release_elbow_deg = None
        self.release_target_commanded = None
        self.release_target_latched = None
        self.landing_pos = None
        self.elbow_extension_deg = None
        self._shoulder_hist = []
        self._elbow_hist = []
        self._release_idx = None
        self._max_shoulder_deg = None

        self.reset()

    def reset(self, seed=None):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0], self.data.qpos[1] = self.start_pose
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

        # put the ball in the hand, then the weld holds it
        hand_pos = self.data.site_xpos[self.hand_site_id].copy()
        hand_quat = np.empty(4)
        mujoco.mju_mat2Quat(hand_quat, self.data.site_xmat[self.hand_site_id].copy())
        self.data.qpos[2:5] = hand_pos
        self.data.qpos[5:9] = hand_quat

        self.data.eq_active[self.grip_eq_id] = 1
        self.released = False
        self.landed = False
        self.spent = False
        self.step_count = 0
        self.release_speed = None
        self.release_height = None
        self.release_shoulder_deg = None
        self.release_elbow_deg = None
        self.release_target_commanded = None
        self.release_target_latched = None
        self.landing_pos = None
        self.elbow_extension_deg = None
        self._shoulder_hist = [np.degrees(self.data.qpos[0])]
        self._elbow_hist = [np.degrees(self.data.qpos[1])]
        self._release_idx = None
        self._max_shoulder_deg = self._shoulder_hist[0]
        mujoco.mj_forward(self.model, self.data)
        return self._obs()

    def step(self, action):
        """action = [shoulder_torque, elbow_torque, release], each in [-1, 1]."""
        action = np.asarray(action, dtype=np.float64)
        self.data.ctrl[:2] = np.clip(action[:2], -1.0, 1.0)

        just_released = False
        # one-shot release: drop the weld and record the release state
        if not self.released and self._release_permitted() and self._release_commanded(action[2]):
            self.release_speed = self._ball_linear_speed()
            self.release_height = float(self._ball_pos()[2])
            self.release_shoulder_deg = float(np.degrees(self.data.qpos[0]))
            self.release_elbow_deg = float(np.degrees(self.data.qpos[1]))
            self.release_target_commanded = (
                self.release_target_latched if self.release_latch
                else self.release_target_deg(action[2])
            ) if self.release_mode == "target_angle" else None
            self.data.eq_active[self.grip_eq_id] = 0
            self.released = True
            just_released = True

        mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        # angle history feeds the elbow extension metric
        self._shoulder_hist.append(np.degrees(self.data.qpos[0]))
        self._elbow_hist.append(np.degrees(self.data.qpos[1]))
        self._max_shoulder_deg = max(self._max_shoulder_deg, self._shoulder_hist[-1])
        if just_released:
            self._release_idx = len(self._shoulder_hist) - 1
            self.elbow_extension_deg = elbow_extension_deg(
                self._shoulder_hist, self._elbow_hist, self._release_idx,
                horizontal_selector=self.horizontal_selector, mode=self.extension_mode)

        if self.released and not self.landed and self._ball_touched_ground():
            self.landed = True
            self.landing_pos = tuple(self._ball_pos()[:2])

        # swung past the window still holding the ball, delivery wasted
        if not self.released and self._shoulder_hist[-1] > self.release_window_max:
            self.spent = True

        reward = self._reward()
        timeout = self.step_count >= self.max_steps
        done = self.landed or timeout or self.spent

        info = {
            "released": self.released,
            "spent": self.spent,
            "release_speed": self.release_speed,
            "landing_pos": self.landing_pos,
            "timeout": timeout,
            "elbow_extension_deg": self.elbow_extension_deg,
            "legal": (self.elbow_extension_deg is not None
                      and self.elbow_extension_deg <= self.max_legal_extension_deg),
            "max_shoulder_deg": self._max_shoulder_deg,
            "shoulder_at_release_deg": self.release_shoulder_deg,
            "elbow_at_release_deg": self.release_elbow_deg,
            "release_height_m": self.release_height,
            "release_target_deg": self.release_target_commanded,
        }
        return self._obs(), reward, done, info

    # arm can step ~4 deg per tick, so the top of the window is unreachable
    RELEASE_TARGET_INSET_DEG = 10.0

    def release_target_deg(self, release_action):
        """Map action[2] in [-1, 1] onto the release window."""
        a = float(np.clip(release_action, -1.0, 1.0))
        top = max(self.release_window_min,
                  self.release_window_max - self.RELEASE_TARGET_INSET_DEG)
        return self.release_window_min + 0.5 * (a + 1.0) * (top - self.release_window_min)

    def release_action_for_angle(self, angle_deg):
        """Inverse of release_target_deg."""
        top = max(self.release_window_min,
                  self.release_window_max - self.RELEASE_TARGET_INSET_DEG)
        if top <= self.release_window_min:
            return -1.0
        frac = (float(angle_deg) - self.release_window_min) / (top - self.release_window_min)
        return float(np.clip(2.0 * frac - 1.0, -1.0, 1.0))

    def _release_commanded(self, release_action):
        """Is the policy asking to let go now?"""
        if self.release_mode == "threshold":
            return release_action > 0.0
        if self.release_mode != "target_angle":
            raise ValueError(f"unknown release_mode: {self.release_mode!r}")

        shoulder_deg = np.degrees(self.data.qpos[0])
        if self.release_latch:
            # latch on window entry
            if self.release_target_latched is None:
                if shoulder_deg < self.release_window_min:
                    return False
                self.release_target_latched = self.release_target_deg(release_action)
            target = self.release_target_latched
        else:
            target = self.release_target_deg(release_action)
        return shoulder_deg >= target

    def _release_permitted(self):
        """Is the arm in a position where letting go is allowed?"""
        if self.step_count < self.min_release_step:
            return False
        shoulder_deg = np.degrees(self.data.qpos[0])
        if not (self.release_window_min <= shoulder_deg <= self.release_window_max):
            return False
        return abs(np.degrees(self.data.qpos[1])) <= self.max_release_elbow_deg

    def _obs(self):
        return np.concatenate([self.data.qpos[:2], self.data.qvel[:2]]).astype(np.float32)

    def _ball_pos(self):
        return self.data.xpos[self.ball_body_id].copy()

    def _ball_linear_speed(self):
        # free joint qvel: [0:3] linear, [3:6] angular
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
        # still flying and not timed out: nothing to score yet
        if not self.landed:
            if self.step_count < self.max_steps and not self.spent:
                return 0.0
            # timed out or spent: score wherever the ball is now
            x = self._ball_pos()[0]
        else:
            x = self.landing_pos[0]

        legal = (self.elbow_extension_deg is not None
                 and self.elbow_extension_deg <= self.max_legal_extension_deg)
        in_zone = self.target_min <= x <= self.target_max
        dist = 0.0 if in_zone else min(abs(x - self.target_min), abs(x - self.target_max))

        # speed only counts if legal and in zone
        if legal and in_zone:
            speed = self.release_speed if self.release_speed is not None else 0.0
            return 1.0 + self.speed_weight * speed
        if legal:
            return -dist
        # illegal or never bowled: miss distance plus a flat penalty
        return -dist - self.illegal_penalty
