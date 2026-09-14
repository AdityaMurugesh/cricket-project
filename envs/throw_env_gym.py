"""Gymnasium adapter for ThrowEnv, needed for stable-baselines3.

Kept separate from throw_env.py rather than changing ThrowEnv's own API,
since demo_scripted.py depends on ThrowEnv's current reset()/step() shape.

Also accumulates the episode's joint-angle history and attaches the
elbow-extension metric (via legality.py) to info on episode end, so
training-loop logging (release speed, landing position, elbow extension --
see workflow-constraints memory) is available from a single info dict.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from envs.throw_env import ThrowEnv
from envs.legality import elbow_extension_deg

_OBS_BOUND = np.array([4 * np.pi, 4 * np.pi, 50.0, 50.0], dtype=np.float32)


class ThrowEnvGym(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, **throw_env_kwargs):
        super().__init__()
        self._env = ThrowEnv(**throw_env_kwargs)
        self.observation_space = spaces.Box(low=-_OBS_BOUND, high=_OBS_BOUND, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self._shoulder_hist = []
        self._elbow_hist = []
        self._release_idx = None
        self._episode_reward = 0.0
        self._episode_len = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs = self._env.reset(seed=seed)
        self._shoulder_hist = [np.degrees(self._env.data.qpos[0])]
        self._elbow_hist = [np.degrees(self._env.data.qpos[1])]
        self._release_idx = None
        self._episode_reward = 0.0
        self._episode_len = 0
        return obs.astype(np.float32), {}

    def step(self, action):
        was_released = self._env.released
        obs, reward, done, info = self._env.step(action)
        self._episode_reward += reward
        self._episode_len += 1

        self._shoulder_hist.append(np.degrees(self._env.data.qpos[0]))
        self._elbow_hist.append(np.degrees(self._env.data.qpos[1]))
        if info["released"] and not was_released:
            self._release_idx = len(self._shoulder_hist) - 1

        terminated = bool(self._env.landed)
        truncated = bool(info["timeout"] and not self._env.landed)

        if terminated or truncated:
            info = dict(info)
            info["episode_reward"] = self._episode_reward
            info["episode_len"] = self._episode_len
            info["elbow_extension_deg"] = None
            if self._release_idx is not None:
                info["elbow_extension_deg"] = elbow_extension_deg(
                    self._shoulder_hist, self._elbow_hist, self._release_idx)

        return obs.astype(np.float32), reward, terminated, truncated, info
