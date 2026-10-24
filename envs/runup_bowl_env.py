"""The full bowler: run in like the capture, then bowl with the right arm (stage 3).

RunupImitationEnv's tracking reward keeps the run-up, trunk, legs, head and
front arm close to the mocap. On top of it, the right arm now has a job, the
same one ThrowEnv's two-joint arm learned:

  * the ball sits at the right hand's grip point (scripts/demo_mocap3d.GRIP)
  * it is released when the bowling arm passes --release-angle (fixed_angle, as
    in run10-run14), measured like ThrowEnv's shoulder angle: the upper arm in the
    x-z plane, 90 hanging down, 180 back, 270 straight up, 360 forward. Release
    only counts within window_before_s / late_s of the captured release; a bent
    arm (> 40 deg) at the release angle wastes the delivery, as in ThrowEnv
  * a release is only a delivery from a bowling position: the upper arm within
    max_arm_out_of_plane_deg of the x-z plane (the capture is at 41-44 deg as it
    passes 250-270, leaning away; a
    sideways sling's x-z projection can also cross the release angle) and the
    pelvis within max_release_tilt_deg / max_release_height_err_m of the capture
    (the policies released while toppling). Otherwise it is a no ball
  * the bowler must then stay up for follow_through_s (the capture has 0.4 s of
    follow-through after release): a fall there voids the delivery
  * legality: envs/legality.elbow_extension_deg on the SIMULATED shoulder angle and
    elbow flexion, every physics step, exactly as ThrowEnv (never from mocap)
  * the flight is resolved at release (drag-free ballistic, as MuJoCo's own ball
    with no air); the task reward is paid at the end of the follow-through
  * task reward = ThrowEnv's, scaled by task_weight, landing measured from the
    release point: legal + in zone 1 + 0.5 * speed (m/s); legal miss -distance;
    illegal -distance - 2. The zone is the length band AND within line_tol_m of
    the stump line (y = 0): the distance is to the nearest point of that box

No ball bowled -- a fall, drifting more than max_pelvis_drift_m from the captured
pelvis path (if set), a release from no bowling position, a fall in the
follow-through, or still holding it late_s after the captured release --
scores the held-ball value (ball at the hand, illegal), i.e. at least as badly as
any real delivery. Without that, falling just before a bad release would pay.

The bowling arm is still not imitated: its whole action is the policy's own.
"""
import mujoco
import numpy as np

from envs.humanoid_bowler import MOCAP_DIR
from envs.legality import elbow_extension_deg
from envs.runup_imitation_env import RunupImitationEnv

GRIP = np.array([0.035, -0.09, 0.0])   # ball centre in the right hand frame (demo_mocap3d.GRIP)
BALL_R = 0.036
G = 9.81


class RunupBowlEnv(RunupImitationEnv):
    def __init__(self, delivery="T3a", release_angle_deg=250.0, target_range=(10.9, 12.9),
                 task_weight=10.0, speed_weight=0.5, illegal_penalty=2.0, max_release_elbow_deg=40.0,
                 window_before_s=0.15, late_s=0.15, rsi_before_release_s=1.0, max_pelvis_drift_m=None,
                 max_arm_out_of_plane_deg=50.0, max_release_tilt_deg=20.0, max_release_height_err_m=0.15,
                 follow_through_s=0.3, line_tol_m=0.5, **kw):
        # the participant's arm length, as fitted (stored with the reference)
        arm = float(np.load(MOCAP_DIR / f"ref_{delivery}.npz")["arm_scale"])
        super().__init__(delivery=delivery, arm_scale=kw.pop("arm_scale", arm), **kw)
        # episodes may only start in the run-up, before the arm begins its circle:
        # a start mid-delivery would hand the bowling arm the human's swing
        self.rsi_last_index = int(self.ref["release_index"] - rsi_before_release_s * self.ref["rate"])
        self.release_angle_deg = release_angle_deg
        self.target_min, self.target_max = target_range
        self.task_weight = task_weight
        self.speed_weight = speed_weight
        self.illegal_penalty = illegal_penalty
        self.max_release_elbow_deg = max_release_elbow_deg
        # the fall test only looks at height and tilt, so a policy can stop running
        # in and bowl from a standstill metres behind the capture (the 30M-step
        # from-scratch runs did): with this set, drifting that far counts as a fall
        self.max_pelvis_drift_m = max_pelvis_drift_m
        self.max_arm_out_of_plane_deg = max_arm_out_of_plane_deg
        self.max_release_tilt_deg = max_release_tilt_deg
        self.max_release_height_err_m = max_release_height_err_m
        self.follow_through_s = follow_through_s
        self.line_tol_m = line_tol_m
        self.t_ref_last = (len(self.ref["qpos"]) - 1) / self.ref["rate"]
        self.t_ref_release = self.t_end
        self.t_window = self.t_end - window_before_s
        self.t_end = self.t_end + late_s          # episode may run a little past the captured release
        self._hand = self.m.body("hand_r").id
        self._shoulder = self.m.body("humerus_r").id
        self._elbow_body = self.m.body("ulna_r").id
        self._elbow_q = self.m.jnt_qposadr[self.m.joint("elbow_flex_r").id]

    # -- bowling-arm geometry ----------------------------------------------
    def _arm_angle_raw(self):
        """Upper-arm angle in the x-z plane, ThrowEnv convention, in [0, 360)."""
        v = self.d.xpos[self._elbow_body] - self.d.xpos[self._shoulder]
        return (90.0 + np.degrees(np.arctan2(-v[0], -v[2]))) % 360.0

    def _grip(self):
        R = self.d.xmat[self._hand].reshape(3, 3)
        return self.d.xpos[self._hand] + R @ GRIP

    def _grip_velocity(self):
        v6 = np.zeros(6)
        mujoco.mj_objectVelocity(self.m, self.d, mujoco.mjtObj.mjOBJ_BODY, self._hand, v6, 0)
        R = self.d.xmat[self._hand].reshape(3, 3)
        # point velocity = body linear velocity + omega x (R @ GRIP); both in world frame
        return v6[3:] + np.cross(v6[:3], R @ GRIP)

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.start_arm_tracking()
        return obs, info

    def start_arm_tracking(self):
        """Begin the shoulder/elbow history (for legality) at the current state."""
        a = self._arm_angle_raw()
        self._arm_hist = [a if a > 45.0 else a + 360.0]   # hanging (~90) is the start of the circle
        self._elbow_hist = [float(np.degrees(self.d.qpos[self._elbow_q]))]
        self._outcome = None
        self._pending = None      # task reward of a release, paid after the follow-through

    def _arm_out_of_plane_deg(self):
        v = self.d.xpos[self._elbow_body] - self.d.xpos[self._shoulder]
        return float(np.degrees(np.arcsin(min(1.0, abs(v[1]) / (np.linalg.norm(v) + 1e-12)))))

    def _bowling_position(self):
        """None if the body is in a position to bowl from, else why not."""
        q_ref, _ = ref_at_clamped(self, self.t)
        if (np.degrees(_tilt(self.d.qpos[3:7], q_ref[3:7])) > self.max_release_tilt_deg
                or abs(self.d.qpos[2] - q_ref[2]) > self.max_release_height_err_m):
            return "off balance at release"
        if self._arm_out_of_plane_deg() > self.max_arm_out_of_plane_deg:
            return "arm out of the bowling plane"
        return None

    def _track_arm(self):
        a = self._arm_angle_raw()
        prev = self._arm_hist[-1]
        # unwrap: take the branch nearest the previous sample
        a += 360.0 * np.round((prev - a) / 360.0)
        self._arm_hist.append(a)
        self._elbow_hist.append(float(np.degrees(self.d.qpos[self._elbow_q])))

    def _task_reward(self, land_x, legal, lateral=0.0):
        dy = max(abs(lateral) - self.line_tol_m, 0.0)
        in_zone = land_x is not None and self.target_min <= land_x <= self.target_max and dy == 0.0
        if land_x is None:
            dist = self.target_min
        else:
            dx = 0.0 if self.target_min <= land_x <= self.target_max else \
                min(abs(land_x - self.target_min), abs(land_x - self.target_max))
            dist = float(np.hypot(dx, dy))
            # capped at the held-ball distance: no release (fall, hold) is never
            # better than any release, even one thrown backwards
            dist = min(dist, self.target_min)
        if legal and in_zone:
            return 1.0 + self.speed_weight * self._outcome["speed_m_s"]
        if legal:
            return -dist
        return -dist - self.illegal_penalty

    def _release(self):
        p, v = self._grip(), self._grip_velocity()
        z0 = p[2] - BALL_R
        t_fl = (v[2] + np.sqrt(max(v[2] ** 2 + 2 * G * z0, 0.0))) / G
        land = float(v[0] * t_fl)                       # metres beyond the release point
        ext = elbow_extension_deg(self._arm_hist, self._elbow_hist, len(self._arm_hist) - 1)
        legal = ext is not None and ext <= 15.0
        lat = float(p[1] + v[1] * t_fl)
        self._outcome = {"released": True, "speed_m_s": float(np.linalg.norm(v)), "land_x": land,
                         "lateral_m": lat, "extension_deg": ext, "legal": legal, "release_t": self.t,
                         "release_height_m": float(p[2]), "arm_out_of_plane_deg": self._arm_out_of_plane_deg()}
        self._outcome["good"] = bool(legal and self.target_min <= land <= self.target_max
                                     and abs(lat) <= self.line_tol_m)
        return self._task_reward(land, legal, lat)

    def _no_ball(self, why):
        self._outcome = {"released": False, "why": why, "speed_m_s": 0.0, "land_x": None, "lateral_m": None,
                         "extension_deg": None, "legal": False, "good": False}
        return self._task_reward(None, False)

    def step(self, action):
        target = self._targets(action)
        task = None
        for _ in range(self.frame_skip):
            self.d.ctrl[:] = self.J.pd_torques(self.d, target)
            mujoco.mj_step(self.m, self.d)
            if self._pending is not None:
                continue                     # follow-through: the ball is gone
            self._track_arm()
            if self.t >= self.t_window and self._arm_hist[-1] >= self.release_angle_deg:
                why = ("bent arm at the release angle" if abs(self._elbow_hist[-1]) > self.max_release_elbow_deg
                       else self._bowling_position())
                if why:
                    task = self._no_ball(why)
                    break
                self._pending = self._release()
                self._t_release = self.t
                if self.follow_through_s <= 0:
                    task = self._pending
                    break
        self.t += self.dt
        q_ref, v_ref = ref_at_clamped(self, self.t)
        r, terms = self._reward(q_ref, v_ref)
        fell = (abs(self.d.qpos[2] - q_ref[2]) > 0.3
                or _tilt(self.d.qpos[3:7], q_ref[3:7]) > 1.0
                or not np.isfinite(self.d.qpos).all())
        drift = float(np.linalg.norm(self.d.qpos[:2] - q_ref[:2]))
        if task is None and self.max_pelvis_drift_m is not None and drift > self.max_pelvis_drift_m:
            fell = True
            task = self._no_ball("left the run-up")
        if task is None and fell:
            task = self._no_ball("fell in the follow-through" if self._pending is not None else "fell")
        if task is None and self._pending is not None and self.t >= self._t_release + self.follow_through_s - 1e-9:
            task = self._pending
        if task is None and self._pending is None and self.t >= self.t_end - 1e-9:
            task = self._no_ball("still holding the ball")
        done = task is not None
        if fell:
            r = 0.0
        if done:
            r += self.task_weight * task
        info = {"terms": terms, "fell": bool(fell), "t": self.t, "start_t": self._start_t, "drift_m": drift,
                "reached_release": bool(self._outcome is not None and self._outcome["released"]),
                "task_reward": task, "outcome": self._outcome}
        return self._obs(min(self.t, self.t_ref_last)), float(r), bool(done), False, info


def ref_at_clamped(env, t):
    """The capture runs 0.4 s past release, so the follow-through is tracked too."""
    from envs.humanoid_bowler import ref_at
    return ref_at(env.ref, min(t, env.t_ref_last), env.m)


def _tilt(q1, q2):
    return 2.0 * np.arccos(min(1.0, abs(float(np.dot(q1, q2)))))
