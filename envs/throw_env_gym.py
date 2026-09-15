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

Reward shaping (added after frame_skip alone still wasn't enough -- run3
plateaued around reward -6, never reliably reaching the target zone; see
workflow-constraints memory): ThrowEnv's true reward is a single sparse
number handed out only at the very end of the episode (landing or
timeout), which is a hard credit-assignment problem for PPO across up to
~180 decisions. This wrapper adds a dense *potential-based* shaping term
on top, following Ng, Harada & Russell (1999): shaped_reward = raw_reward
+ gamma * Phi(s') - Phi(s). That specific form is provably policy-
invariant -- it cannot change what the optimal policy is, only how fast
PPO can find it, so it's safe to add without silently redefining the
research question's actual objective (max speed subject to legality +
landing accuracy).

Phi(s) here is "how close would the ball land (assuming free-fall
ballistic flight from its current position/velocity) to the target
zone" -- valid pre-release too, since the grip weld keeps the ball's
qvel equal to the hand's, so it's a live "if I released right now"
prediction that only sharpens as the swing builds real speed and
direction. This single distance-based signal naturally captures BOTH
aim and speed (a stationary ball's predicted landing is right at the
start pose, far from the zone; building speed toward the target moves
the predicted landing closer), so a separate speed term in Phi isn't
needed. The *actual* task reward used for episode_reward/CSV logging
(the real research metric) is never touched -- only the per-step signal
PPO trains on is shaped.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from envs.throw_env import ThrowEnv
from envs.legality import elbow_extension_deg

_OBS_BOUND = np.array([4 * np.pi, 4 * np.pi, 50.0, 50.0], dtype=np.float32)


class ThrowEnvGym(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, frame_skip=5, use_shaping=True, shaping_gamma=0.99, **throw_env_kwargs):
        super().__init__()
        self._env = ThrowEnv(**throw_env_kwargs)
        self.frame_skip = frame_skip
        self.use_shaping = use_shaping
        # must match the PPO model's own gamma for the shaping math to be
        # the correct potential-based form -- SB3 PPO defaults to 0.99 too.
        self.shaping_gamma = shaping_gamma
        self.observation_space = spaces.Box(low=-_OBS_BOUND, high=_OBS_BOUND, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self._shoulder_hist = []
        self._elbow_hist = []
        self._release_idx = None
        self._episode_reward = 0.0
        self._episode_len = 0

    def _predicted_landing_x(self):
        """Where the ball would land (flat ground, gravity-only free fall)
        if it kept its current position/velocity from here on. Valid
        pre-release (grip weld keeps ball qvel == hand velocity) and
        mid-flight (converges to the true landing_pos as it descends)."""
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
        x_pred = self._predicted_landing_x()
        env = self._env
        if env.target_min <= x_pred <= env.target_max:
            return 0.0
        return -min(abs(x_pred - env.target_min), abs(x_pred - env.target_max))

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
        phi_before = self._potential() if self.use_shaping else 0.0

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

        if self.use_shaping:
            phi_after = 0.0 if (terminated or truncated) else self._potential()
            shaped_reward = total_reward + self.shaping_gamma * phi_after - phi_before
        else:
            shaped_reward = total_reward

        if terminated or truncated:
            info = dict(info)
            info["episode_reward"] = self._episode_reward
            info["episode_len"] = self._episode_len
            info["elbow_extension_deg"] = None
            if self._release_idx is not None:
                info["elbow_extension_deg"] = elbow_extension_deg(
                    self._shoulder_hist, self._elbow_hist, self._release_idx)

        return obs.astype(np.float32), shaped_reward, terminated, truncated, info
