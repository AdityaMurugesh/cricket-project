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
ThrowEnv itself now tracks shoulder/elbow angle history and computes
elbow_extension_deg at every physics substep (not just every decision),
so the elbow-legality metric's precision doesn't depend on frame_skip --
see elbow-legality-design-decision memory on why that must stay exact.
This wrapper just forwards it through info rather than duplicating the
tracking.

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

Phi(s) has two terms. The first is "how close would the ball land
(assuming free-fall ballistic flight from its current position/
velocity) to the target zone" -- valid pre-release too, since the grip
weld keeps the ball's qvel equal to the hand's, so it's a live "if I
released right now" prediction that only sharpens as the swing builds
real speed and direction. That distance signal naturally captures BOTH
aim and speed (a stationary ball's predicted landing is right at the
start pose, far from the zone; building speed toward the target moves
the predicted landing closer), so a separate speed term isn't needed.

The second (swing_weight) is monotonic progress of the shoulder around
the swing arc toward the release orientation, which the distance term
alone does not ask for -- see _potential() for the geometry and for the
attractor bug that the first version of this term introduced. It spans
0 -> swing_weight across the full swing, so swing_weight is set to be
roughly commensurate with the distance term's metre-scale range.

The third (straight_arm_weight) rewards extending the elbow as the
release window approaches, so PPO can find ThrowEnv's hard
elbow-at-release gate instead of arriving at the top folded up and being
unable to let go.

A caveat that run6 paid for: because this whole block is potential-based,
NONE of these weights can change which policy is optimal -- that is the
entire point of the Ng/Harada/Russell form, and it is why tuning
swing_weight from 5 to 30 never stopped the policy slinging the ball from
behind its back. Shaping is a learning-speed knob only. Anything the task
must actually require belongs in ThrowEnv, as a reward term or a gate.

All three terms are functions of the current state only, and Phi is
forced to 0 at terminal states, so the Ng/Harada/Russell policy-invariance
guarantee still holds for the set. The *actual* task reward used for
episode_reward/CSV logging (the real research metric) is never touched
-- only the per-step signal PPO trains on is shaped.
"""
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
        # must match the PPO model's own gamma for the shaping math to be
        # the correct potential-based form. Note this is NOT SB3's default:
        # train_throw.py passes gamma=0.999 explicitly, because at 0.99 an
        # episode-terminal reward ~240 policy steps away is discounted to
        # 0.09 of its value, which made stalling to timeout beat throwing
        # and collapsed a validation run. See ThrowEnv._reward(). If you
        # change one of the two, change both. 0.999 was not enough either:
        # it put a GOOD throw above stalling but left the intermediate step
        # (swinging without managing to release) BELOW it, so the path from
        # "do nothing" to "bowl well" still ran downhill first and the
        # policy retreated. 0.9999 makes the ordering monotone.
        self.shaping_gamma = shaping_gamma
        self.swing_weight = swing_weight
        self.straight_arm_weight = straight_arm_weight
        self.observation_space = spaces.Box(low=-_OBS_BOUND, high=_OBS_BOUND, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
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
            dist_term = 0.0
        else:
            dist_term = -min(abs(x_pred - env.target_min), abs(x_pred - env.target_max))
        # The true reward also requires legality (see throw_env.py's
        # _reward()), but the distance term alone has no reason to prefer
        # a real swing over an elbow-only flick -- that mismatch is what
        # let run4 converge on a fast, accurate, but illegal technique.
        #
        # Swing geometry (verified against the compiled model): the
        # shoulder angle points straight DOWN at 90, horizontal BACKWARD
        # (away from the target) at 180, straight UP at 270, and
        # horizontal FORWARD at 360. Release velocity is tangential, so
        # the ball leaves perpendicular to the arm, not along it: at 270
        # (arm vertical) it leaves horizontally toward the target from
        # maximum height, which is the real overarm release point, while
        # at 360 the arm is sweeping DOWN through horizontal and drives
        # the ball into the ground a metre away. A scripted open-loop
        # swing confirms 270 is the optimum: 41.5 km/h, extension
        # -0.03 deg (legal), landing 7.82 m, reward +2.15.
        #
        # An earlier version of this term used
        # -swing_weight * min(|angle-180|, |angle-360|), i.e. "distance
        # to the nearest horizontal". That is wrong in a way that caused
        # the exact failure it was meant to fix: it peaks at 180 -- arm
        # pointing away from the target -- and bottoms out at 270, the
        # true release point, scoring it WORSE than never moving at all.
        # It pinned the arm backward and penalized the climb to release.
        #
        # Replaced with monotonic progress from the start pose to 270, so
        # every degree of the climb is rewarded, the maximum sits exactly
        # at the release orientation, and there is no interior attractor
        # to get stuck in. Flat past 270 so the follow-through is neither
        # required nor penalized.
        #
        # swing_weight has to be large because the distance term actively
        # fights the backswing: while the arm climbs from 100 to ~180 the
        # ball is being carried backward, so the "if I released right now"
        # prediction gets worse, bottoming out at -10.89 against -6.11 at
        # the start -- a -4.78 penalty for doing the one thing a bowling
        # action requires. Measured along the scripted optimal swing, the
        # weight needed just to break even at that worst point is ~16.6.
        # At the old 5.0 the climb was net -3.40 and Phi fell on 87 of 152
        # climb steps, so the policy correctly learned to stand still and
        # drop the ball; at 30.0 the residual dip is -0.13 over 17 steps,
        # which PPO crosses easily. Policy invariance holds at any weight
        # (this is still PBRS), so this is a gradient-shaping knob only --
        # it cannot bias the reported research metric.
        #
        # The climb now targets the centre of ThrowEnv's overarm release
        # window rather than a hardcoded 270, so the shaping and the hard
        # gate cannot drift apart if the window is reconfigured.
        angle = env.data.qpos[0] * 180.0 / np.pi
        start_deg = float(np.degrees(env.start_pose[0]))
        release_deg = 0.5 * (env.release_window_min + env.release_window_max)
        progress = (angle - start_deg) / (release_deg - start_deg)
        swing_term = self.swing_weight * float(np.clip(progress, 0.0, 1.0))

        # Third term: straighten the arm as the window approaches. Release
        # is hard-gated on elbow <= max_release_elbow_deg, so a policy that
        # arrives at the top still folded up simply cannot let go, and the
        # episode is wasted. Nothing else in Phi asks for extension -- the
        # distance term is if anything happy with a bent arm, since folding
        # up shortens the lever and calms the predicted landing point. This
        # just makes the gate findable; it cannot make an unstraight arm
        # legal or illegal, because it is still pure PBRS.
        #
        # Ramped in over the last 60 deg of the climb instead of applied
        # throughout: a straight arm early in the backswing is a longer
        # lever fighting gravity, and demanding it from the start would
        # penalise the very acceleration phase that produces the speed.
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

            # a spent delivery (arm swung past the release window still
            # holding the ball) is a real outcome with a real reward, not a
            # cut-off, so it counts as terminated rather than truncated --
            # otherwise SB3 would bootstrap a value estimate onto the end of
            # an episode that genuinely has no future.
            terminated = bool(self._env.landed or self._env.spent)
            truncated = bool(info["timeout"] and not terminated)
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

        return obs.astype(np.float32), shaped_reward, terminated, truncated, info
