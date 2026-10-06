"""Gymnasium wrapper for ThrowEnv with frame skip and potential-based shaping."""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from envs.throw_env import ThrowEnv

_OBS_BOUND = np.array([4 * np.pi, 4 * np.pi, 50.0, 50.0], dtype=np.float32)


class ThrowEnvGym(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, frame_skip=5, use_shaping=True, shaping_gamma=0.9999,
                 swing_weight=30.0, straight_arm_weight=6.0, **throw_env_kwargs):
        super().__init__()
        self._env = ThrowEnv(**throw_env_kwargs)
        self.frame_skip = frame_skip
        self.use_shaping = use_shaping
        # must equal PPO's gamma or the shaping is no longer policy-invariant
        self.shaping_gamma = shaping_gamma
        self.swing_weight = swing_weight
        self.straight_arm_weight = straight_arm_weight
        self.observation_space = spaces.Box(low=-_OBS_BOUND, high=_OBS_BOUND, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self._episode_reward = 0.0
        self._episode_len = 0

    def _predicted_landing_x(self):
        """Ballistic landing x if the ball kept its current velocity."""
        env = self._env
        pos = env._ball_pos()
        vel = env.data.qvel[2:5]
        x0, z0 = float(pos[0]), float(pos[2])
        vx0, vz0 = float(vel[0]), float(vel[2])
        g = -env.model.opt.gravity[2]
        disc = vz0 * vz0 + 2 * g * z0
        if g <= 0 or disc < 0:
            return x0
        t = (vz0 + np.sqrt(disc)) / g
        return x0 + vx0 * t

    def _potential(self):
        """Shaping potential: predicted miss distance + swing progress + arm straightness."""
        x_pred = self._predicted_landing_x()
        env = self._env
        # how far the ball would land from the zone if released now
        if env.target_min <= x_pred <= env.target_max:
            dist_term = 0.0
        else:
            dist_term = -min(abs(x_pred - env.target_min), abs(x_pred - env.target_max))
        # progress round the swing arc towards the release point
        angle = env.data.qpos[0] * 180.0 / np.pi
        start_deg = float(np.degrees(env.start_pose[0]))
        release_deg = 0.5 * (env.release_window_min + env.release_window_max)
        progress = (angle - start_deg) / (release_deg - start_deg)
        swing_term = self.swing_weight * float(np.clip(progress, 0.0, 1.0))

        # straighten the arm over the last 60 deg of the climb
        elbow = abs(env.data.qpos[1] * 180.0 / np.pi)
        approach = float(np.clip((angle - (release_deg - 60.0)) / 60.0, 0.0, 1.0))
        straightness = float(np.clip(1.0 - elbow / 180.0, 0.0, 1.0))
        straight_term = self.straight_arm_weight * approach * straightness

        return dist_term + swing_term + straight_term

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs = self._env.reset(seed=seed)
        self._episode_reward = 0.0
        self._episode_len = 0
        return obs.astype(np.float32), {}

    def step(self, action):
        phi_before = self._potential() if self.use_shaping else 0.0

        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        for _ in range(self.frame_skip):
            obs, reward, done, info = self._env.step(action)
            total_reward += reward
            self._episode_len += 1

            # a spent delivery is a real outcome, not a cut-off
            terminated = bool(self._env.landed or self._env.spent)
            truncated = bool(info["timeout"] and not terminated)
            if terminated or truncated:
                break

        self._episode_reward += total_reward

        if self.use_shaping:
            # Phi must be 0 at terminal states
            phi_after = 0.0 if (terminated or truncated) else self._potential()
            shaped_reward = total_reward + self.shaping_gamma * phi_after - phi_before
        else:
            shaped_reward = total_reward

        if terminated or truncated:
            info = dict(info)
            info["episode_reward"] = self._episode_reward
            info["episode_len"] = self._episode_len

        return obs.astype(np.float32), shaped_reward, terminated, truncated, info
