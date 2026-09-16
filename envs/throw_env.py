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

That was not enough, and run6 showed why. The trained policy releases at
shoulder 207 deg with the elbow bent 113 deg: the ball leaves from BEHIND
the body at 1.6 m, barely above the shoulder mount, slung forward and up
at 28 deg. That is a shot-put, not a bowling action. It passes the ICC
elbow check trivially, because a permanently flexed but rigid arm has
zero extension and is legal by construction (context.md says so
explicitly) -- so legality alone can never require a bowling action.

Two things follow. First, no amount of tuning swing_weight in
throw_env_gym.py could ever have fixed this: that term is potential-based
shaping, which is provably policy-INVARIANT (Ng, Harada & Russell 1999).
It changes how fast PPO finds the optimum, never which optimum it is. The
sling was genuinely optimal under the reward as written. Second, the fix
therefore belongs in the task definition, and context.md already names it
-- "Overarm, not underarm" is listed as a required hard episode filter
and had simply never been implemented.

So release is now gated on ARM GEOMETRY rather than on elapsed steps:
the ball can only leave the hand inside an overarm delivery window (arm
up near vertical, release_window_deg) and with a reasonably straight arm
(max_release_elbow_deg). A scripted swing through that window measures
41.5 km/h, landing 7.82 m, extension -0.02 deg -- strictly better than
run6's learned sling on speed AND accuracy, so this constrains the policy
to a region that is not merely legal but actually faster.

NOTE on scope: max_release_elbow_deg is an "is this a bowling action at
all" filter. It is NOT a redefinition of the ICC legality metric, which
remains exactly as settled -- extension (change) measured on the
simulated elbow between arm-horizontal and release, never absolute bend,
never from mocap. See the elbow-legality-design-decision memory. Both
gates are constructor parameters, not constants, so their effect on
achievable speed can be reported rather than hidden.
"""
from pathlib import Path

import mujoco
import numpy as np

from envs.legality import elbow_extension_deg

ASSET_PATH = Path(__file__).resolve().parent.parent / "assets" / "throw_arm.xml"


class ThrowEnv:
    def __init__(self, model_path=ASSET_PATH, target_range=(6.0, 8.0), max_steps=1200,
                 start_pose_deg=(100.0, 0.0), speed_weight=0.1, min_release_step=0,
                 release_window_deg=(230.0, 310.0), max_release_elbow_deg=40.0,
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
        # Raised from 900 (1.8 s) with the overarm release window. Releases
        # now happen higher and later in the arc, and one near the bottom of
        # the window is lofted enough to still be airborne at 1.8 s -- which
        # scored it off the ball's mid-flight position instead of its real
        # landing point. 1200 covers the whole window's flight times, so the
        # timeout branch of _reward() stops firing on genuine deliveries.
        self.max_steps = max_steps
        self.speed_weight = speed_weight
        # Superseded by release_window_deg below and now defaulted off. It
        # used to be the only release gate (=100 steps), added because an
        # untrained policy fires release on step 0-1 nearly every episode,
        # collapsing every episode into "drop the ball from the start pose"
        # with no gradient for PPO. The angular window subsumes that job --
        # the arm physically cannot reach 230 deg from the 100 deg start
        # pose without a real swing -- and unlike a step count it also pins
        # down WHERE in the swing the ball leaves, which is what run6's
        # sling exploited. Kept as a parameter so the old configuration is
        # still reproducible for the writeup.
        self.min_release_step = min_release_step

        # ICC Law 21.1, "overarm not underarm": the ball may only leave the
        # hand while the arm is up near vertical. Geometry (axis is +Y, arm
        # along +X at 0): the arm points DOWN at 90 deg, horizontal BACKWARD
        # at 180, straight UP at 270, horizontal FORWARD at 360. Release
        # velocity is tangential, so at 270 the ball leaves horizontally
        # toward the target from maximum height -- the real overarm release
        # point. The default window is 270 +/- 40.
        #
        # Because the shoulder joint's lower limit is 95 deg (throw_arm.xml)
        # and the start pose is 100, the ONLY way into this window is up and
        # over through 180 -- so entering it also guarantees the
        # arm-horizontal crossing that the ICC elbow metric is measured
        # from. The window is one-shot in practice: the joint's upper limit
        # is 380 deg, so an arm that sails past 310 without releasing cannot
        # come round again. That is the real timing problem a bowler has.
        self.release_window_min, self.release_window_max = release_window_deg

        # A bowler's arm is close to straight at release. run6 released with
        # the elbow bent 113 deg, which is a shot-put, and no ICC rule
        # forbids it: extension (change) was ~0 because the arm was bent
        # RIGIDLY, which context.md states is legal by construction. This is
        # therefore a separate "is this a bowling action at all" filter and
        # deliberately NOT part of the legality metric -- see the module
        # docstring and the elbow-legality-design-decision memory.
        self.max_release_elbow_deg = max_release_elbow_deg
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
        self.release_height = None
        self.release_shoulder_deg = None
        self.release_elbow_deg = None
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
        self.release_height = None
        self.release_shoulder_deg = None
        self.release_elbow_deg = None
        self.landing_pos = None
        self.elbow_extension_deg = None
        self._shoulder_hist = [np.degrees(self.data.qpos[0])]
        self._elbow_hist = [np.degrees(self.data.qpos[1])]
        self._release_idx = None
        self._max_shoulder_deg = self._shoulder_hist[0]
        mujoco.mj_forward(self.model, self.data)
        return self._obs()

    def step(self, action):
        """action: [shoulder_torque, elbow_torque, release] each in [-1, 1].
        release > 0 triggers a one-shot release of the weld."""
        action = np.asarray(action, dtype=np.float64)
        self.data.ctrl[:2] = np.clip(action[:2], -1.0, 1.0)

        just_released = False
        if not self.released and action[2] > 0.0 and self._release_permitted():
            self.release_speed = self._ball_linear_speed()
            self.release_height = float(self._ball_pos()[2])
            self.release_shoulder_deg = float(np.degrees(self.data.qpos[0]))
            self.release_elbow_deg = float(np.degrees(self.data.qpos[1]))
            self.data.eq_active[self.grip_eq_id] = 0
            self.released = True
            just_released = True

        mujoco.mj_step(self.model, self.data)
        self.step_count += 1

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
            # how far round the swing the arm actually got. elbow extension
            # only exists once a horizontal crossing does, so when it logs
            # blank there is no way to tell "never moved" from "swung but
            # stopped just short" -- which is the difference between a
            # policy that is failing and one that is nearly there.
            "max_shoulder_deg": self._max_shoulder_deg,
            "shoulder_at_release_deg": self.release_shoulder_deg,
            # the two numbers that separate a bowling action from run6's
            # sling, and neither is visible in speed/landing/extension:
            # where in the arc the ball left, and how straight the arm was.
            "elbow_at_release_deg": self.release_elbow_deg,
            "release_height_m": self.release_height,
        }
        return self._obs(), reward, done, info

    def _release_permitted(self):
        """Whether the ball is allowed to leave the hand right now: the arm
        must be inside the overarm delivery window and close to straight.
        A release action outside this is simply ignored (the grip holds),
        the same way min_release_step used to ignore early ones -- it is a
        constraint on the action space, not a reward penalty, so PPO cannot
        trade it away against speed."""
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
