"""Gymnasium adapter for ThrowEnv, needed for stable-baselines3.

Kept separate from throw_env.py rather than changing ThrowEnv's own API,
since demo_scripted.py depends on ThrowEnv's current reset()/step() shape.

Also accumulates the episode's joint-angle history and attaches the
elbow-extension metric (via legality.py) to info on episode end, so
training-loop logging (release speed, landing position, elbow extension --
see workflow-constraints memory) is available from a single info dict.

frame_skip repeats each policy action for that many underlying 0.002s
physics steps before the policy sees a new observation. Two training runs
(see workflow-constraints memory) showed PPO improving for the first ~40%
of a 2M-step run, then collapsing onto a near-zero-effort "release
immediately" policy for the rest -- consistent with the reward (a single
number at the end of up to 900 individual per-step decisions) being too
hard to credit-assign over that many decisions. frame_skip=5 cuts the
decision count roughly 5x (900 -> ~180) without changing episode duration
in physics time, which is the standard fix for this failure mode.
elbow/shoulder history is still recorded every physics substep (not every
decision), so the elbow-legality metric's precision doesn't depend on
frame_skip -- see elbow-legality-design-decision memory on why that must
stay exact.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from envs.throw_env import ThrowEnv
from envs.legality import elbow_extension_deg

_OBS_BOUND = np.array([4 * np.pi, 4 * np.pi, 50.0, 50.0], dtype=np.float32)


class ThrowEnvGym(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, frame_skip=5, **throw_env_kwargs):
        super().__init__()
        self._env = ThrowEnv(**throw_env_kwargs)
        self.frame_skip = frame_skip
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
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        for _ in range(self.frame_skip):
            was_released = self._env.released
            obs, reward, done, info = self._env.step(action)
            total_reward += reward
            self._episode_len += 1

            self._shoulder_hist.append(np.degrees(self._env.data.qpos[0]))
            self._elbow_hist.append(np.degrees(self._env.data.qpos[1]))
            if info["released"] and not was_released:
                self._release_idx = len(self._shoulder_hist) - 1

            terminated = bool(self._env.landed)
            truncated = bool(info["timeout"] and not self._env.landed)
            if terminated or truncated:
                break

        self._episode_reward += total_reward

        if terminated or truncated:
            info = dict(info)
            info["episode_reward"] = self._episode_reward
            info["episode_len"] = self._episode_len
            info["elbow_extension_deg"] = None
            if self._release_idx is not None:
                info["elbow_extension_deg"] = elbow_extension_deg(
                    self._shoulder_hist, self._elbow_hist, self._release_idx)

        return obs.astype(np.float32), total_reward, terminated, truncated, info
