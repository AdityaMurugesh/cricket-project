"""KinematicBowlEnv with the roadmap's Tier-1 options (docs/roadmap_2026-11-18.md, items 2-5).

Torque control only. Every option is off by default, and with all of them off this behaves exactly as
KinematicBowlEnv(arm_control="torque") (tests/check in scripts/check_env2.py). The old env is left
untouched so that every earlier result stays reproducible.

  tolerance_reward_sd_ms = s   (item 2, "equifinality reward") the delivery pays the EXACT expected task
      reward under release-timing jitter N(0, s), not one sampled draw. At the release trigger the arm is
      simulated on for 40 ms (the policy still acting, holding the ball), the release instants within
      +/-40 ms are each scored as a delivery (length, line, legality at that instant, window, arm plane,
      bent arm), and the reward is their N(0, s)-weighted mean. This is the expectation of what training
      with release_timing_sd_ms = s samples, and the same curve kin_equifinality scores (p_jitter_12ms).
  learned_release   (item 3) one extra action: the ball goes when it is > 0, instead of when the arm passes
      release_angle_deg. A command before the release window opens is ignored (otherwise an untrained
      policy only ever releases at once, and every such no ball pays the same); a bent arm is a no ball.
  release_timing_sd_ms (fixed angle or learned): the actual release is the trigger + N(0, sd), causally:
      a late draw simulates the arm on; an early one uses the grip state recorded that long ago (the
      ball's flight depends only on the grip's position and velocity at that instant).
  target_sample = (lo, hi)   (item 4) per episode the good-length band is [c - 1, c + 1] m with
      c ~ U(lo, hi), and (c - 11.9) / 5 is in the observation, so one policy bowls any length on demand.
  actuator = "hill", omega_max   (item 5) the motor's available torque falls with the joint's speed in
      the direction it pushes (Hill's force-velocity curve), (1 - w/w_max) / (1 + w/(k w_max)), k = 0.25,
      zero at w_max, and full when it brakes. The linear damping (peak torque / arm_top_speed) is then
      reduced to 5% of it, for numerical stability only.
  torque_scale   multiplies the bowling arm's motor gears: the actuator budget (item 1)
  limits = "human" | "g1" | "h1"   the arm's actuator and joint limits from envs/arm_limits.py (peak torque and
      torque-speed curve per motor, joint ranges, shoulder cone); the linear damping is then 5% of peak / no-load
      speed, for numerical stability only. None: the skeleton's own (250 N m, 40 rad/s damping), as before.
  tolerance_target = t   (with tolerance_reward_sd_ms) the good part of the expected reward is divided by
      max(p, t): above good rate t only speed pays, below it tolerance does, so the bowler settles at p = t at
      the most speed it can. Plain expected reward (t = 0) favours p ~ 0.85-0.95 on every skill frontier.

Observation: KinematicBowlEnv's (so its trained policies can be fine-tuned here), + [target code] with
target_sample.
"""
import mujoco
import numpy as np
from gymnasium import spaces

from envs.kinematic_bowl_env import KinematicBowlEnv
from envs import legality
from envs.legality import elbow_extension_deg
from envs.ballistics import land as ballistic_land

HALF_WINDOW_S = 0.040
HILL_K = 0.25


def land(p, v, k=0.0):
    """RunupBowlEnv._release's ballistic (envs/ballistics.py): length beyond the release point, and line."""
    return ballistic_land(p, v, k)


class KinematicBowlEnv2(KinematicBowlEnv):
    def __init__(self, tolerance_reward_sd_ms=0.0, learned_release=False, target_sample=None,
                 actuator="damping", omega_max=40.0, torque_scale=1.0, tolerance_target=0.0, limits=None, **kw):
        kw["arm_control"] = "torque"
        super().__init__(**kw)
        m = self.m
        self.tolerance_reward_sd_s = tolerance_reward_sd_ms / 1000.0
        self.tolerance_target = tolerance_target
        self.learned_release = learned_release
        self.target_sample = target_sample
        self.actuator = actuator
        self.omega_max = omega_max
        self.torque_scale = torque_scale
        if torque_scale != 1.0:
            m.actuator_gear[self._arm_acts, :3] *= torque_scale
            m.dof_damping[self._arm_dofs] *= torque_scale       # same top speed: damping = peak torque / top
        if actuator == "hill":
            m.dof_damping[self._arm_dofs] *= 0.05
        elif actuator != "damping":
            raise ValueError(actuator)
        self.limits = limits
        if limits is not None and limits != "model":
            from envs.arm_limits import PROFILES
            if actuator != "damping" or torque_scale != 1.0:
                raise ValueError("limits replaces actuator/torque_scale")
            P = PROFILES[limits]
            tq = np.asarray(P["torque"], float)
            g = m.actuator_gear[self._arm_acts, :3]
            m.actuator_gear[self._arm_acts, :3] = g / np.linalg.norm(g, axis=1, keepdims=True) * tq[:, None]
            om = np.asarray(P["omega_max"], float)
            for a_, t_, o_ in zip(self._arm_acts, tq, om):
                j = m.actuator_trnid[a_, 0]
                da = m.jnt_dofadr[j]
                nd = 3 if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_BALL else 1
                m.dof_damping[da:da + nd] = 0.05 * t_ / o_
            for name, (lo, hi) in P.get("range_deg", {}).items():
                j = m.joint(name).id
                m.jnt_range[j] = np.radians([lo, hi])
                m.jnt_limited[j] = 1
            if P.get("shoulder_cone_deg"):
                j = m.joint("shoulder_r").id
                m.jnt_range[j] = [0.0, np.radians(P["shoulder_cone_deg"])]
                m.jnt_limited[j] = 1
        self._gear_norm = np.linalg.norm(m.actuator_gear[self._arm_acts, :3], axis=1)
        self._continues = self.tolerance_reward_sd_s > 0 or self.release_timing_sd_ms > 0
        n_act = len(self._arm_action) + (1 if learned_release else 0)
        self.action_space = spaces.Box(-1.0, 1.0, (n_act,), np.float32)
        self._target_code = 0.0
        self.suppress_release = False        # kin_equifinality's "release switched off" replay
        self.fixed_target_m = None           # target_sample: score at this band centre instead of a draw
        n_obs = len(self._obs2())
        self.observation_space = spaces.Box(-np.inf, np.inf, (n_obs,), np.float32)

    # -- observation ---------------------------------------------------------
    def _obs2(self):
        extra = []                   # no post-release flag: the observation stays KinematicBowlEnv's
        if self.target_sample is not None:
            extra.append(self._target_code)
        return np.concatenate([self._kin_obs(), np.asarray(extra, np.float32)]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        _, info = super().reset(seed=seed, options=options)
        if self.target_sample is not None:
            if options and "target_m" in options:
                c = float(options["target_m"])
            elif self.fixed_target_m is not None:
                c = float(self.fixed_target_m)
            else:
                c = float(self.rng.uniform(*self.target_sample))
            self.target_min, self.target_max = c - 1.0, c + 1.0
            self._target_code = (c - 11.9) / 5.0
        self._hist = []              # per physics step: (t, grip p, grip v, arm plane deg)
        self._trigger_i = None       # physics step of the release command / fixed-angle trigger
        self._release_at = None      # physics step the ball actually goes (sampled jitter, late draw)
        return self._obs2(), info

    # -- actuation -----------------------------------------------------------
    def _torques(self, a):
        u = np.array(a[: len(self._arm_action)], dtype=np.float64)
        if self.limits is not None and self.limits != "model":
            from envs.arm_limits import available
            w = self.d.actuator_velocity[self._arm_acts] / self._gear_norm      # rad/s along each motor's axis
            u = u * available(self.limits, w, u)
        if self.actuator == "hill":
            w = self.d.actuator_velocity[self._arm_acts] / self._gear_norm      # rad/s along each motor's axis
            conc = np.sign(w) == np.sign(u)
            s = np.abs(w) / self.omega_max
            f = np.where(conc, np.clip(1.0 - s, 0.0, 1.0) / (1.0 + s / HILL_K), 1.0)
            u = u * f
        return u

    # -- scoring one release instant ------------------------------------------
    def _score_instant(self, i):
        """Task reward and outcome if the ball left right after physics step i (index into _hist)."""
        t, p, v, plane = self._hist[i]
        j = i + 1                                   # _arm_hist[0] is the episode's start state
        elbow = self._elbow_hist[j]
        if abs(elbow) > self.max_release_elbow_deg:
            return self._no_ball_at("bent arm at the release")
        if plane > self.max_arm_out_of_plane_deg:
            return self._no_ball_at("arm out of the bowling plane")
        if not (self.t_window - 1e-9 <= t <= self.t_end + 1e-9):
            return self._no_ball_at("outside the release window")
        x, y = land(p, v, self.drag_k)
        ext = elbow_extension_deg(self._arm_hist, self._elbow_hist, j)
        legal = ext is not None and ext <= legality.LEGAL_EXTENSION_DEG
        out = {"released": True, "speed_m_s": float(np.linalg.norm(v)), "land_x": x, "lateral_m": y,
               "extension_deg": ext, "legal": legal, "release_t": t, "release_height_m": float(p[2]),
               "arm_out_of_plane_deg": plane}
        out["good"] = bool(legal and self.target_min <= x <= self.target_max and abs(y) <= self.line_tol_m)
        self._outcome = out
        return self._task_reward(x, legal, y), out

    def _no_ball_at(self, why):
        r = self._no_ball(why)
        return r, self._outcome

    def _expected(self, c):
        """N(0, sd)-weighted task reward over the recorded instants within +/-40 ms of instant c."""
        h = self.m.opt.timestep
        k = int(round(HALF_WINDOW_S / h))
        lo, hi = max(c - k, 0), min(c + k, len(self._hist) - 1)
        offs = (np.arange(lo, hi + 1) - c) * h
        w = np.exp(-0.5 * (offs / self.tolerance_reward_sd_s) ** 2)
        w /= w.sum()
        R, P, G = 0.0, 0.0, 0.0
        for wi, i in zip(w, range(lo, hi + 1)):
            r, o = self._score_instant(i)
            R += wi * r
            P += wi * float(o["good"])
            G += wi * r * float(o["good"])
        if self.tolerance_target > 0 and P > 0:
            # target tolerance: the good part pays G / max(P, target). Above the target that is the good
            # deliveries' mean reward (speed only); below it, P / target of it (tolerance pays)
            R = R - G + G / max(P, self.tolerance_target)
        r0, o0 = self._score_instant(c)            # the nominal release, for logging
        o0 = dict(o0, p_jit=P, expected_reward=R)
        self._outcome = o0
        return R, o0

    # -- step ------------------------------------------------------------------
    def step(self, action):
        a = np.clip(action, -1.0, 1.0)
        if self.torque_noise_sd > 0:
            a = a * (1.0 + self.torque_noise_sd * self._noise_rng.standard_normal(a.shape))
        cmd = self.learned_release and float(np.clip(action, -1, 1)[-1]) > 0.0
        h = self.m.opt.timestep
        task = None
        for k in range(self.frame_skip):
            t_k = self.t + k * h
            self._drive_body(t_k)
            self.d.ctrl[:] = 0.0
            if t_k >= self.t_handover - 1e-9:
                self.d.ctrl[self._arm_acts] = self._torques(a)
            mujoco.mj_step(self.m, self.d)
            self._track_arm()
            t_after = t_k + h
            self._hist.append((t_after, self._grip().copy(), self._grip_velocity().copy(),
                               self._arm_out_of_plane_deg()))
            i = len(self._hist) - 1
            if self._trigger_i is None and not self.suppress_release:
                if self.learned_release:      # a command before the window opens is ignored (no early no-ball)
                    fire = cmd and t_after >= self.t_window - 1e-9
                else:
                    fire = self.t >= self.t_window and self._arm_hist[-1] >= self.release_angle_deg
                if fire:
                    self._trigger_i = i
                    if self.tolerance_reward_sd_s > 0:
                        self._end_i = i + int(round(HALF_WINDOW_S / h))
                    elif self.release_delay_s != 0.0:
                        self._release_at = i + int(round(self.release_delay_s / h))
                        if self._release_at <= i:           # an early draw: the state that long ago
                            task, _ = self._score_instant(max(self._release_at, 0))
                            break
                    else:
                        task, _ = self._score_instant(i)
                        break
            if self._trigger_i is not None and self.tolerance_reward_sd_s > 0 and i >= self._end_i:
                task, _ = self._expected(self._trigger_i)
                break
            if self._release_at is not None and i >= self._release_at:
                task, _ = self._score_instant(i)
                break
        self.t += self.dt
        if task is None and self.t >= self.t_end - 1e-9:
            if self._trigger_i is not None and self.tolerance_reward_sd_s > 0:
                task, _ = self._expected(self._trigger_i)     # ran out of episode: score what was recorded
            elif self._release_at is not None:
                task, _ = self._score_instant(len(self._hist) - 1)
            else:
                task = self._no_ball("still holding the ball")
        done = task is not None
        r = self.task_weight * task if done else 0.0
        info = {"terms": dict.fromkeys(("pose", "vel", "ends", "root"), 0.0), "fell": False, "t": self.t,
                "start_t": self._start_t, "drift_m": 0.0,
                "reached_release": bool(self._outcome is not None and self._outcome.get("released")),
                "task_reward": task, "outcome": self._outcome}
        return self._obs2(), float(r), bool(done), False, info
