"""Fit the full-body 3D skeleton (mocap/skeleton3d.py) to the raw markers.

The exported labels are unreliable (tracks.py), and the markers worn do not
fully match the AIM model either: a 3-marker group at knee height has
exactly a wrist-and-hand's spacing, and there is an extra elbow marker
(REPORT.md). So the fit does not assume WHICH IOR marker a track is. It
assumes only what is physically certain -- each marker sits at a fixed point
on one bone, a little outside it -- and decides, in alternation:

  1. pose: per frame, the joint angles that bring each track's attachment
     point (bone + fixed offset) onto the track -- damped least squares with
     MuJoCo Jacobians, Huber-robust, with a constant-velocity prior from the
     previous frames and a weak prior towards a neutral posture, so joints
     with no marker stay sensible;
  2. attachment: each track goes to the bone it stays most still relative to
     over its lifetime (RMS of its position in that bone's frame), its offset
     the mean of that position, kept within SHELL_M of the bone's surface and
     SITE_MAX_M of one of its anatomical landmarks (without the latter a marker
     slides along a long bone and the fit trades that for a contorted pose).
     Repeated with a tightening threshold.

Tracks left over are then tested one by one (grow): re-fit their stretch of
time with the track attached to each candidate bone, and keep the attachment
if the track fits and the other markers do not get worse. Each rule there
fixed a failure seen against the video: a bone with no marker on it or
further down its limb can bend to reach anything, so it only takes a track
already close; leg tracks stay on their side of the pelvis; a rigid partner's
bone is tried first; no upper-arm marker near the shoulder joint (that is the
acromion, which slides on a scapula the rigid model torso does not have);
among fits, the one that needs the least change of pose wins.

Initial guess: the skeleton standing at the marker cloud, facing the run-up,
legs in envs/runup.py's running cycle locked to the measured steps
(mocap/runup.py), blended into BowlerEnv's delivery stride after the top of
the jump (as mocap/retarget.py does) -- only so the foot and shin markers can
latch on; the final leg pose comes from the markers where there are any.
Three physical constraints hold throughout: feet stay above the floor; at
each measured step (mocap/runup.py) and from SETTLE_S before release a foot
is on it (alternating back from a left-foot take-off, front foot planted at
release); and the pelvis stays near level, so the trunk, not the pelvis,
does the leaning. Tracks already identified by physics (mocap/identify.py)
are pinned to their bone. The skeleton is scaled to the participant by the
lowest residual (scripts/fit_mocap3d.py fixes one scale for all deliveries).

Final pass: markers fade in and out over FADE frames; limb gaps of up to
INFILL_MAX frames are interpolated in joint space; Gauss-Seidel smoothing;
then an 8 Hz zero-lag low-pass on the joints, because QTM's re-identification
swaps the exported marker set wholesale at a few instants per trial.

The result is measured, not assumed: per frame, how many markers drive the
fit and how far they are from their attachment points; per bone, whether any
marker is on it. Bones with no marker are posed by the priors alone, and
scripts/demo_mocap3d.py draws them translucent.
"""
from dataclasses import dataclass, field

import mujoco
import numpy as np
from scipy.spatial import cKDTree

from envs.runup import DELIVERY_STRIDE_DEG, run_cycle_at_phase_deg
from mocap import retarget, skeleton3d
from mocap.signal import runs

SIGMA_M = 0.01            # marker noise scale for the pose solve
HUBER_M = 0.03            # residuals beyond this are down-weighted
ROUNDS_M = (0.10, 0.07, 0.05, 0.04, 0.035, 0.03)   # attachment RMS threshold per round
SHELL_M = (0.005, 0.06)   # marker centre from the bone surface: min, max
SITE_MAX_M = 0.08         # ... and at most this far from one of the bone's landmarks
SITE_LABEL_M = 0.04       # ... or from its own landmark, for a track labelled by hand
TILT_SD = np.sin(np.radians(25))   # pelvis kept near level (the trunk does the leaning)
SHIN_SD = np.sin(np.radians(35))   # a shin with no marker hangs near vertical
FLOOR_SD, CONTACT_SD = 0.005, 0.02  # feet above the floor; a stance foot on it
STANCE_HALF = 3           # frames either side of a measured step the stance foot is down
FADE = 6                  # frames over which a marker's weight ramps in and out
INFILL_MAX = 80           # frames: a limb gap up to this long is interpolated, not guessed
JOINT_CUTOFF_HZ = 12.0    # final zero-lag low-pass on the joints (8 Hz blunted the arm at release)
ARM_CUTOFF_HZ = 20.0      # ... and on the bowling arm, the fastest thing in the delivery
ELBOW_WINDOW_S = (-0.3, 0.05)   # the bowling elbow is straightest somewhere in here
UPPER_ARM_WEIGHT = 0.2          # upper-arm markers, which carry the elbow offset, in the last pass
ELBOW_HOLD_S = (-0.35, 0.15)    # bowling elbow held straight for display in here
ELBOW_BLEND_S = 0.1
MIN_TRACK_FRAMES = 15     # shorter tracks cannot show which bone they move with
# an elbow or knee within JAM_STOP_RAD of its straight stop for JAM_FRAMES+ frames, its
# markers beyond > JAM_RESID_M off (median), is on the wrong branch (unjam_limbs)
JAM_STOP_RAD, JAM_RESID_M, JAM_FRAMES = np.radians(3), 0.04, 5
# a limb without enough markers (arm: none on forearm or hand; leg: fewer than two,
# or none below the knee), inside its TEMPLATE_S window of release, is held to the
# template (the mean measured pose of that limb at that instant in the deliveries
# where it is tracked) this firmly; ramped in over TEMPLATE_RAMP rows
TEMPLATE_S = {"arm": (-0.9, 0.5), "leg": (-1.4, 0.4)}
TEMPLATE_SD = {"rot": np.radians(15), "slide": 0.01}
TEMPLATE_RAMP = 10
ARM_JOINTS = ("girdle_{s}_x", "girdle_{s}_y", "girdle_{s}_z", "shoulder_{s}", "elbow_flex_{s}",
              "pro_sup_{s}", "wrist_flex_{s}", "wrist_dev_{s}")
LEG_JOINTS = ("hip_{s}", "knee_angle_{s}", "ankle_angle_{s}", "subtalar_angle_{s}", "mtp_angle_{s}")
LIMB_JOINTS = {"arm": ARM_JOINTS, "leg": LEG_JOINTS}
# run-up trunk with fewer than 3 markers: kept near this forward lean (T2a's
# bare trunk at the start fell 50 deg sideways; with markers it leans 10-35 deg)
TRUNK_SD, TRUNK_LEAN_DEG = np.sin(np.radians(20)), 15
# (hinge, parent ball joint, bodies beyond it, index of its straight stop in the range,
#  restart bends deg, restart rolls of the parent bone about its long axis deg)
JAM_LIMBS = (("elbow_flex_{s}", "shoulder_{s}", ("ulna", "radius", "hand"), 0, (90,), (0, 90, 180, 270)),
             ("knee_angle_{s}", "hip_{s}", ("tibia", "talus", "calcn", "toes"), 1, (-45, -90), (0,)))
SCALES = (0.92, 0.96, 1.00, 1.04, 1.08)
GROW_MAX_M = 0.025        # a tested attachment is kept if the track fits within this (median)
GROW_COST_M = 0.004       # ... and the other markers' median residual worsens by less than this
MOVE_COST_M = 0.03        # score cost per radian of mean joint change needed to take a marker
SHOULDER_CLEAR_M = 0.08   # no upper-arm marker this close to the shoulder joint (that is the scapula)
RIGID_SD_M, RIGID_SWING_DEG, RIGID_FRAMES = 0.006, 30, 30   # same bone: spacing sd while turning
# a leg marker is on its own side of the pelvis (feet may land a little across)
SIDE_SLACK_M = {"femur": 0.02, "tibia": 0.02, "calcn": 0.06, "toes": 0.06}
UNSUPPORTED_MAX_M = 0.06  # a bone with no marker nearby only takes a track already this close
ALIGN_STEP_M = 0.1        # grid step of the initial-guess alignment (+/- 2 steps sideways and along)
RUNUP_JUDGE_S = 0.8       # the first match uses only frames this long before release (pre take-off)
# median marker height a foot or shin may take unsupported; no thigh band: the
# hands hang at thigh height, and in T3a a thigh band put the left wrist pair on
# the left femur, which then kicked out to reach it
LEG_BAND_M = {"calcn": (-0.05, 0.30), "toes": (-0.05, 0.20), "tibia": (0.03, 0.62)}
LEG_HEIGHT_MAX_M = {"femur": 0.75, "tibia": 0.62, "calcn": 0.30, "toes": 0.30}  # median, scale 1
# ... and a thigh marker is at the knee or above: lower ones are on the shin (T4b's
# whole right-shin cluster, 0.40-0.50 m, had gone on the thigh, freeing the knee)
LEG_HEIGHT_MIN_M = {"femur": 0.46}
DISTAL = {"femur_r": ("tibia_r",), "tibia_r": ("calcn_r",), "calcn_r": ("toes_r",),
          "femur_l": ("tibia_l",), "tibia_l": ("calcn_l",), "calcn_l": ("toes_l",),
          "humerus_r": ("ulna_r",), "ulna_r": ("radius_r",), "radius_r": ("hand_r",),
          "humerus_l": ("ulna_l",), "ulna_l": ("radius_l",), "radius_l": ("hand_l",),
          "pelvis": ("torso",), "torso": ("pelvis", "head")}
NEIGHBOURS = {"pelvis": ("torso", "femur_r", "femur_l"), "head": ("torso",),
              "torso": ("pelvis", "head", "humerus_r", "humerus_l"),
              "femur_r": ("pelvis", "tibia_r"), "tibia_r": ("femur_r", "calcn_r"),
              "calcn_r": ("tibia_r", "toes_r"), "toes_r": ("calcn_r",),
              "femur_l": ("pelvis", "tibia_l"), "tibia_l": ("femur_l", "calcn_l"),
              "calcn_l": ("tibia_l", "toes_l"), "toes_l": ("calcn_l",),
              "humerus_r": ("torso", "ulna_r"), "ulna_r": ("humerus_r", "radius_r"),
              "radius_r": ("ulna_r", "hand_r"), "hand_r": ("radius_r",),
              "humerus_l": ("torso", "ulna_l"), "ulna_l": ("humerus_l", "radius_l"),
              "radius_l": ("ulna_l", "hand_l"), "hand_l": ("radius_l",)}
BONES = ("pelvis", "torso", "head", "femur_r", "tibia_r", "calcn_r", "toes_r", "femur_l", "tibia_l",
         "calcn_l", "toes_l", "humerus_r", "ulna_r", "radius_r", "hand_r", "humerus_l",
         "ulna_l", "radius_l", "hand_l")
PRED_SD = {"root_pos": 0.03, "root_rot": np.radians(4), "joint": np.radians(12),
           "arm": np.radians(30), "girdle": 0.01}
POST_SD = {"joint": np.radians(60), "minor": np.radians(15), "girdle": 0.025,
           "neck": np.radians(30)}
MINOR = ("subtalar", "mtp", "wrist_dev", "pro_sup", "wrist_flex")
ARM = ("shoulder", "elbow", "pro_sup", "wrist")
PINS = {"hand": ("hand_r",), "wrist": ("radius_r", "ulna_r"), "elbow": ("humerus_r",),
        "proximal": ("humerus_r", "torso"), "spine": ("torso",)}


@dataclass
class WTrack:
    tid: int
    f0: int                 # first frame, index into the fit window
    P: np.ndarray           # (n, 3) world positions
    aim: str                # QTM's majority label, reported only
    pinned: tuple = ()      # allowed bones, if identified by physics or by hand
    site: str = ""          # its landmark (skeleton3d site, e.g. "L_HUM"), if labelled by hand

    @property
    def f1(self):
        return self.f0 + len(self.P) - 1


@dataclass
class Fit3D:
    name: str
    first: int              # absolute frame of row 0
    release: int            # row of the release
    rate: float
    scale: float = 1.0
    qpos: np.ndarray = None
    attach: dict = field(default_factory=dict)      # tid -> (bone, (3,) offset in its frame)
    rms: dict = field(default_factory=dict)         # tid -> RMS motion relative to its bone, m
    n_obs: np.ndarray = None
    resid_med: np.ndarray = None
    resid_p90: np.ndarray = None
    support: np.ndarray = None                      # (n, nbody) bool: a marker on the body
    history: list = field(default_factory=list)
    xml: str = ""
    elbow_offset_deg: float = float("nan")          # removed from the bowling elbow (straighten_elbow)


def to_world(xyz, direction, x_offset, z_centre):
    """Lab (x along the room, y up, z lateral) -> world (x forward, y left, z up),
    the frame of mocap.retarget.marker_cloud."""
    return np.stack([direction * xyz[..., 0] - x_offset,
                     -direction * (xyz[..., 2] - z_centre), xyz[..., 1]], -1)


def world_tracks(tracks, first, last, direction, x_offset, z_centre):
    out = []
    for t in tracks:
        s, e = max(t.start, first), min(t.end, last)
        if e - s + 1 < MIN_TRACK_FRAMES:
            continue
        P = to_world(t.xyz[s - t.start:e - t.start + 1], direction, x_offset, z_centre)
        out.append(WTrack(t.tid, s - first, P, t.aim_label))
    return out


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def _quat_axis(axis, ang):
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, np.asarray(axis, float), ang)
    return q


class Skeleton:
    def __init__(self, scale=1.0, arm_scale=1.0):
        self.scale, self.arm_scale = scale, arm_scale
        self.xml = skeleton3d.build_xml(scale, arm_scale=arm_scale)
        m = self.m = mujoco.MjModel.from_xml_string(self.xml)
        self.d = mujoco.MjData(m)
        self.bones = np.array([m.body(b).id for b in BONES])
        self.nq = m.joint("ball_free").qposadr[0]     # body coordinates come first
        self.nv = m.joint("ball_free").dofadr[0]
        self.q0 = m.qpos0.copy()
        self.q_up = self.q0[3:7].copy()               # pelvis upright, facing +x
        self.jacp = np.zeros((3, m.nv))
        self.jacr = np.zeros((3, m.nv))
        self.pelvis = m.body("pelvis").id
        self.stance = None    # per fit row: feet on the floor (see stance_schedule)
        self.W_post = None    # per fit row: posture-prior weights (template-held arms), or None
        self.tmpl_rows = {}   # "arm_r" etc. -> (n,) bool: rows where that limb has a template
        self.upright = None   # per fit row: run-up, where a sparsely marked trunk is held near upright
        self.torso = m.body("torso").id
        lean = np.radians(TRUNK_LEAN_DEG)
        self.lean_axis = np.array([np.sin(lean), 0.0, np.cos(lean)])
        self.soles = {s: [m.site(f"sole_{p}_{s}").id for p in ("heel", "ball")] for s in ("l", "r")}
        self.shins = {s: (m.body(f"tibia_{s}").id,
                          {m.body(f"{b}_{s}").id for b in ("tibia", "talus", "calcn", "toes")})
                      for s in ("l", "r")}
        self.surface = {}
        for b in self.bones:
            verts = []
            for g in range(m.ngeom):
                if m.geom_bodyid[g] == b and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                    verts.append(skeleton3d._body_vertices(m, m.body(b).name, m.geom(g).name))
            self.surface[b] = cKDTree(np.concatenate(verts))
        self.sites_on = {}
        for s in range(m.nsite):
            if m.site(s).name.startswith("m_"):
                self.sites_on.setdefault(int(m.site_bodyid[s]), []).append(m.site_pos[s].copy())
        self.sites_on = {b: np.array(v) for b, v in self.sites_on.items()}
        self.site_named = {m.site(s).name[2:]: (int(m.site_bodyid[s]), m.site_pos[s].copy())
                           for s in range(m.nsite) if m.site(s).name.startswith("m_")}
        pred, post = np.zeros(m.nv), np.zeros(m.nv)
        for j in range(m.njnt):
            name, a, n = m.joint(j).name, m.jnt_dofadr[j], {0: 6, 1: 3, 2: 1, 3: 1}[int(m.jnt_type[j])]
            if name == "ball_free":
                continue
            if name == "root":
                pred[a:a + 3] = PRED_SD["root_pos"] ** -2
                pred[a + 3:a + 6] = PRED_SD["root_rot"] ** -2
                continue
            if name.startswith("girdle"):   # a few cm, rarely; slow
                pred[a:a + n] = PRED_SD["girdle"] ** -2
                post[a:a + n] = POST_SD["girdle"] ** -2
                continue
            arm = any(name.startswith(k) for k in ARM)
            pred[a:a + n] = PRED_SD["arm" if arm else "joint"] ** -2
            post[a:a + n] = POST_SD["neck" if name == "neck" else
                                    "minor" if any(k in name for k in MINOR) else "joint"] ** -2
        self.w_pred, self.w_post = pred[:self.nv], post[:self.nv]

    def pose(self, pos, yaw=0.0, legs_deg=None):
        """Neutral pose at pos facing yaw; legs_deg: envs.runup leg angles."""
        q = self.q0.copy()
        q[0:3] = pos
        mujoco.mju_mulQuat(q[3:7], _quat_axis([0, 0, 1], yaw), self.q_up)
        if legs_deg:
            for s in ("l", "r"):
                side = "left" if s == "l" else "right"
                a = self.m.joint(f"hip_{s}").qposadr[0]
                q[a:a + 4] = _quat_axis([0, 0, 1], np.radians(legs_deg[f"{side}_hip"]))
                q[self.m.joint(f"knee_angle_{s}").qposadr[0]] = np.radians(legs_deg[f"{side}_knee"])
        return q

    def frames_of(self, Q):
        """World pose of every bone in every frame: (n, nb, 3), (n, nb, 3, 3)."""
        pos = np.empty((len(Q), len(self.bones), 3))
        rot = np.empty((len(Q), len(self.bones), 3, 3))
        for i, q in enumerate(Q):
            self.d.qpos[:] = q
            mujoco.mj_kinematics(self.m, self.d)
            pos[i] = self.d.xpos[self.bones]
            rot[i] = self.d.xmat[self.bones].reshape(-1, 3, 3)
        return pos, rot

    def _diff(self, qa, qb):
        out = np.zeros(self.m.nv)
        mujoco.mj_differentiatePos(self.m, out, 1.0, qa, qb)
        return out[:self.nv]

    def _clamp(self, q):
        m = self.m
        for j in range(m.njnt):
            if not m.jnt_limited[j]:
                continue
            a = m.jnt_qposadr[j]
            if m.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                q[a] = np.clip(q[a], *m.jnt_range[j])
            elif m.jnt_type[j] == mujoco.mjtJoint.mjJNT_BALL:
                w = np.clip(abs(q[a]), 0, 1)
                ang, lim = 2 * np.arccos(w), m.jnt_range[j][1]
                if ang > lim:
                    axis = q[a + 1:a + 4] / (np.linalg.norm(q[a + 1:a + 4]) + 1e-12) * np.sign(q[a] or 1)
                    q[a] = np.cos(lim / 2)
                    q[a + 1:a + 4] = axis * np.sin(lim / 2)
        return q

    def solve(self, q, obs, q_pred, q_post, stance=(), iters=15, w_post=None, upright=False):
        """One frame. obs: list of (body id, (3,) offset, (3,) target);
        q_pred: constant-velocity prediction; q_post: posture prior;
        stance: feet ("l"/"r") that are on the floor in this frame."""
        m, d, nv = self.m, self.d, self.nv
        mask = np.arange(nv) >= 6
        for _ in range(iters):
            d.qpos[:] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            wp = self.w_post if w_post is None else w_post
            H = np.diag(self.w_pred + wp + 1e-6)
            g = self.w_pred * self._diff(q_pred, q) + wp * self._diff(q_post, q) * mask
            if obs:
                J = np.empty((3 * len(obs), nv))
                r = np.empty(3 * len(obs))
                fade = np.empty(len(obs))
                for i, (b, o, target, fw) in enumerate(obs):
                    p = d.xpos[b] + d.xmat[b].reshape(3, 3) @ o
                    mujoco.mj_jac(m, d, self.jacp, None, p, b)
                    J[3 * i:3 * i + 3] = self.jacp[:, :nv]
                    r[3 * i:3 * i + 3] = p - target
                    fade[i] = fw
                dist = np.linalg.norm(r.reshape(-1, 3), axis=1)
                w = np.where(dist < HUBER_M, 1.0, HUBER_M / np.maximum(dist, 1e-9)) * fade
                w = np.repeat(w, 3) / SIGMA_M ** 2
                H += J.T @ (w[:, None] * J)
                g += J.T @ (w * r)
            # pelvis level: its long axis (body y) near world z
            mujoco.mj_jacBody(m, d, None, self.jacr, self.pelvis)
            up = d.xmat[self.pelvis].reshape(3, 3)[:, 1]
            Jup = (-_skew(up) @ self.jacr[:, :nv])[:2]
            H += Jup.T @ Jup / TILT_SD ** 2
            g += Jup.T @ up[:2] / TILT_SD ** 2
            # run-up trunk with too few markers to orient it: near a running lean
            if upright and sum(o[0] == self.torso and o[3] > 0.5 for o in obs) < 3:
                mujoco.mj_jacBody(m, d, None, self.jacr, self.torso)
                ax = d.xmat[self.torso].reshape(3, 3)[:, 1]
                Jax = -_skew(ax) @ self.jacr[:, :nv]
                H += Jax.T @ Jax / TRUNK_SD ** 2
                g += Jax.T @ (ax - self.lean_axis) / TRUNK_SD ** 2
            # a shin with at most one marker on it or below hangs (a driven-up knee
            # bends instead of kicking the lower leg out straight; one marker near
            # the knee places the knee, not the shin: T3b's back leg, marked only at
            # the top of the shin after release, kicked up behind)
            marked = [o[0] for o in obs if o[3] > 0.5]
            for side, (tib, below) in self.shins.items():
                if sum(b in below for b in marked) >= 2:
                    continue
                mujoco.mj_jacBody(m, d, None, self.jacr, tib)
                ax = d.xmat[tib].reshape(3, 3)[:, 1]
                Jax = (-_skew(ax) @ self.jacr[:, :nv])[:2]
                H += Jax.T @ Jax / SHIN_SD ** 2
                g += Jax.T @ ax[:2] / SHIN_SD ** 2
            # floor, and the stance foot on it
            for side, sids in self.soles.items():
                for sid in sids:
                    z = d.site_xpos[sid, 2]
                    if z < 0 or side in stance:
                        mujoco.mj_jacSite(m, d, self.jacp, None, sid)
                        Jz = self.jacp[2, :nv]
                        wz = (FLOOR_SD if z < 0 else CONTACT_SD) ** -2
                        H += wz * np.outer(Jz, Jz)
                        g += wz * Jz * z
            dv = -np.linalg.solve(H, g)
            full = np.zeros(m.nv)
            full[:nv] = dv
            mujoco.mj_integratePos(m, q, full, 1.0)
            self._clamp(q)
            if np.abs(dv).max() < 1e-4:
                break
        return q


def body_centre(wtracks, n):
    """Per-frame median of the visible tracks (world), gaps interpolated."""
    c = np.full((n, 3), np.nan)
    for f in range(n):
        pts = [tr.P[f - tr.f0] for tr in wtracks if tr.f0 <= f <= tr.f1]
        if len(pts) >= 3:
            c[f] = np.median(pts, axis=0)
    k = np.arange(n)
    ok = ~np.isnan(c[:, 0])
    for j in range(3):
        c[:, j] = np.interp(k, k[ok], c[ok, j])
    return c


def initial_poses(sk, centre, run, frames):
    """Standing at the marker cloud, facing +x, legs in the measured-step running
    cycle. frames: absolute frame numbers of the rows of centre."""
    phi = retarget.stride_phase(run, frames)
    amp = np.clip(run.v_fwd[frames - run.frame0] / retarget.FULL_STRIDE_MS, 0, 1)
    jp = run.jump_peak if run.jump_peak >= 0 else run.release - int(0.47 * run.rate)
    settle = run.release - int(retarget.SETTLE_S * run.rate)
    w = retarget._smoothstep((frames - jp) / max(settle - jp, 1))
    Q = []
    for i, c in enumerate(centre):
        cyc = run_cycle_at_phase_deg(phi[i], retarget.SWING_AMP_DEG * amp[i],
                                     retarget.KNEE_LIFT_DEG * amp[i])
        legs = {k: (1 - w[i]) * cyc[k] + w[i] * DELIVERY_STRIDE_DEG[k] for k in cyc}
        Q.append(sk.pose([c[0], c[1], 0.95 * sk.scale], 0.0, legs))
    return np.array(Q)


def stance_schedule(run, first, n):
    """Per row, the feet on the floor: the measured steps alternate back from a
    left-foot take-off (right-arm bowler); the front (left) foot is planted
    from SETTLE_S before release to the end."""
    out = [()] * n
    for j, f in enumerate(reversed(run.steps)):
        foot = "l" if j % 2 == 0 else "r"
        for g in range(f - STANCE_HALF, f + STANCE_HALF + 1):
            if 0 <= g - first < n:
                out[g - first] = (foot,)
    settle = run.release - int(retarget.SETTLE_S * run.rate)
    for g in range(settle, first + n):
        if 0 <= g - first < n:
            out[g - first] = ("l",)
    return out


def fit_poses(sk, wtracks, attach, n, centre, Q_prior, fade=False):
    """Sequential per-frame solve, predicted at (damped) constant velocity, with
    a weak posture prior Q_prior (the initial guess: a joint with no marker
    relaxes towards it instead of drifting)."""
    Q = Q_prior.copy()
    obs_of = _obs_lists(sk, wtracks, attach, n, fade)
    sk.W_post = template_weights(sk, obs_of)
    return _solve_range(sk, obs_of, centre, Q_prior, Q, 0, n - 1)


def _rowkw(sk, f):
    """Skeleton.solve's per-row arguments."""
    return {"w_post": None if sk.W_post is None else sk.W_post[f],
            "upright": bool(sk.upright is not None and sk.upright[f])}


def limb_dofs(m, limb, side):
    """qpos and dof indices of one limb's joints (LIMB_JOINTS), and which dofs slide."""
    qidx, vidx, slide = [], [], []
    for name in LIMB_JOINTS[limb]:
        j = m.joint(name.format(s=side)).id
        ball = m.jnt_type[j] == mujoco.mjtJoint.mjJNT_BALL
        qa, va = m.jnt_qposadr[j], m.jnt_dofadr[j]
        qidx += list(range(qa, qa + (4 if ball else 1)))
        vidx += list(range(va, va + (3 if ball else 1)))
        slide += [m.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE] * (3 if ball else 1)
    return np.array(qidx), np.array(vidx), np.array(slide)


def limb_seen(m, limb, side, obs):
    """Enough markers on this limb in one row's observations to pose it itself:
    arm, one on the forearm or hand; leg, two, at least one below the knee."""
    w = [o[0] for o in obs if o[3] > 0.5]
    if limb == "arm":
        fore = {m.body(f"{b}_{side}").id for b in ("ulna", "radius", "hand")}
        return any(b in fore for b in w)
    low = {m.body(f"{b}_{side}").id for b in ("tibia", "talus", "calcn", "toes")}
    return sum(b in low for b in w) >= 1 and sum(b in low or b == m.body(f"femur_{side}").id for b in w) >= 2


def apply_template(sk, Q0, template, release_row):
    """Set Q0's limb joints to the template (fit_mocap3d.limb_templates: "arm_r"
    etc. -> (rows from release, qpos values)) where it has one; note those rows."""
    sk.tmpl_rows = {}
    for key, (rel, vals) in (template or {}).items():
        limb, side = key.split("_")
        qidx = limb_dofs(sk.m, limb, side)[0]
        rows = release_row + np.asarray(rel)
        ok = (rows >= 0) & (rows < len(Q0))
        for r, v in zip(rows[ok], np.asarray(vals)[ok]):
            Q0[r, qidx] = v
        mask = np.zeros(len(Q0), bool)
        mask[rows[ok]] = True
        sk.tmpl_rows[key] = mask


def template_weights(sk, obs_of):
    """Per-row posture weights: a limb without enough markers (limb_seen) where it
    has a template is held to it (TEMPLATE_SD), ramped in over TEMPLATE_RAMP rows
    after it last had them, so the limb does not jump. None if no template."""
    if not sk.tmpl_rows:
        return None
    m, n = sk.m, len(obs_of)
    W = np.tile(sk.w_post, (n, 1))
    for key, mask in sk.tmpl_rows.items():
        limb, side = key.split("_")
        _, vidx, slide = limb_dofs(m, limb, side)
        boost = np.where(slide, TEMPLATE_SD["slide"] ** -2, TEMPLATE_SD["rot"] ** -2)
        age = TEMPLATE_RAMP
        for f in range(n):
            seen = limb_seen(m, limb, side, obs_of[f])
            age = 0 if seen else age + 1
            if mask[f] and not seen:
                u = min(1.0, age / TEMPLATE_RAMP)
                W[f, vidx] = np.maximum(W[f, vidx], u * boost)
    return W


def _obs_lists(sk, wtracks, attach, n, fade=False):
    """Per frame: (body id, offset, target, weight). With fade (final pass only;
    the attachment rounds use full weights), a track's weight ramps up over its
    first FADE frames and down over its last, so the skeleton does not snap
    when markers appear or vanish."""
    obs_of = [[] for _ in range(n)]
    for tr in wtracks:
        if tr.tid in attach:
            b, o = attach[tr.tid]
            bid = sk.m.body(b).id
            for f in range(tr.f0, tr.f1 + 1):
                fw = min(1.0, (f - tr.f0 + 1) / FADE, (tr.f1 - f + 1) / FADE) if fade else 1.0
                obs_of[f].append((bid, o, tr.P[f - tr.f0], fw))
    return obs_of


def smooth(sk, wtracks, attach, Q, Q_prior, passes=2):
    """Gauss-Seidel smoothing: each frame re-solved against the markers with
    its neighbours' midpoint as the prediction (penalises acceleration rather
    than velocity, and looks both ways in time)."""
    n = len(Q)
    obs_of = _obs_lists(sk, wtracks, attach, n, fade=True)
    sk.W_post = template_weights(sk, obs_of)
    for _ in range(passes):
        for f in range(1, n - 1):
            mid = Q[f - 1].copy()
            full = np.zeros(sk.m.nv)
            full[:sk.nv] = 0.5 * sk._diff(Q[f - 1], Q[f + 1])
            mujoco.mj_integratePos(sk.m, mid, full, 1.0)
            Q[f] = sk.solve(Q[f].copy(), obs_of[f], mid, Q_prior[f],
                            sk.stance[f] if sk.stance is not None else (), iters=8, **_rowkw(sk, f))
    return Q


def lowpass_qpos(m, Q, rate, cutoff=JOINT_CUTOFF_HZ):
    """Zero-lag Butterworth on every coordinate of the body's qpos (quaternions
    made sign-continuous first, renormalised after). The exported marker set
    switches wholesale at QTM's re-identification instants, so the pose is
    held by different markers either side of them; this spreads the step."""
    from scipy.signal import butter, filtfilt
    Q = Q.copy()
    for j in range(m.njnt):
        if m.joint(j).name == "ball_free":
            continue
        a = m.jnt_qposadr[j]
        t = int(m.jnt_type[j])
        quats = [a + 3] if t == 0 else [a] if t == 1 else []
        for qa in quats:
            for f in range(1, len(Q)):
                if Q[f, qa:qa + 4] @ Q[f - 1, qa:qa + 4] < 0:
                    Q[f, qa:qa + 4] *= -1
    nq = m.joint("ball_free").qposadr[0]
    pad = min(15, len(Q) - 1)
    raw = Q[:, :nq].copy()
    b, a_ = butter(2, cutoff / (rate / 2))
    Q[:, :nq] = filtfilt(b, a_, raw, axis=0, padtype="odd", padlen=pad)
    # the bowling arm is the fastest thing in the delivery: filter it less, or the
    # skeleton's hand falls centimetres behind its markers at release
    b, a_ = butter(2, ARM_CUTOFF_HZ / (rate / 2))
    arm = filtfilt(b, a_, raw, axis=0, padtype="odd", padlen=pad)
    for name in ("girdle_r_x", "girdle_r_y", "girdle_r_z", "shoulder_r", "elbow_flex_r", "pro_sup_r",
                 "wrist_flex_r", "wrist_dev_r"):
        j = m.joint(name).id
        a = m.jnt_qposadr[j]
        n = 4 if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_BALL else 1
        Q[:, a:a + n] = arm[:, a:a + n]
    for j in range(m.njnt):
        t, a = int(m.jnt_type[j]), m.jnt_qposadr[j]
        if m.joint(j).name == "ball_free":
            continue
        for qa in ([a + 3] if t == 0 else [a] if t == 1 else []):
            Q[:, qa:qa + 4] /= np.linalg.norm(Q[:, qa:qa + 4], axis=1, keepdims=True)
    return Q


def straighten_elbow(sk, Q, release_row, rate):
    """The bowling elbow for display: flexion relative to its straightest point
    in the delivery (ELBOW_WINDOW_S around release), held straight and steady
    through the delivery itself (ELBOW_HOLD_S, blended in and out over
    ELBOW_BLEND_S); returns it and the offset removed.

    With this marker set the absolute elbow angle carries a constant offset:
    the one upper-arm marker sits off the humeral axis, and the elbow marker on
    the epicondyle, so a straight arm reads ~46 deg bent in T3a (REPORT.md),
    while every camera view shows it straight. As in the 2D replay
    (mocap/retarget.py), the arm is taken to reach full extension during the
    delivery -- the usual straight-arm action, an assumption, not a measurement.
    Within the delivery the fitted elbow, even offset-corrected, swings 20-27 deg
    in the last 100 ms (T3a) -- fit error, not the arm: the markers' own change
    there is ~5 deg (REPORT.md). Shown frame by frame it would look like a
    throw, so the delivery is held straight instead, and the hand sits a few
    cm off its markers at release. This is for the picture only: the elbow
    from mocap is not a legality measurement (legality stays simulated)."""
    a = sk.m.joint("elbow_flex_r").qposadr[0]
    lo = max(0, release_row + int(ELBOW_WINDOW_S[0] * rate))
    hi = min(len(Q) - 1, release_row + int(ELBOW_WINDOW_S[1] * rate))
    off = float(Q[lo:hi + 1, a].min())
    el = np.clip(Q[:, a] - off, 0, None)
    t = (np.arange(len(Q)) - release_row) / rate
    h0, h1 = ELBOW_HOLD_S
    w = np.clip(np.minimum((t - (h0 - ELBOW_BLEND_S)) / ELBOW_BLEND_S,
                           ((h1 + ELBOW_BLEND_S) - t) / ELBOW_BLEND_S), 0, 1)
    return (1 - w) * el, off


def refine_arm(sk, wtracks, attach, Q, elbow=None):
    """Last pass: the bowling arm alone re-solved to its markers in every frame,
    the rest of the body kept as smoothed. The smoothing that removes QTM's
    marker-swap jumps from the trunk also made the arm lag its markers by up to
    16 cm just after release in T3a; re-solved, it is within 2-3 cm.
    elbow: per-frame elbow flexion to hold (straighten_elbow); the upper-arm
    markers, which carry the offset, are then down-weighted (UPPER_ARM_WEIGHT)
    and the shoulder and girdle take up the difference."""
    n = len(Q)
    obs_of = _obs_lists(sk, wtracks, attach, n, fade=True)
    arm_bodies = {sk.m.body(b).id for b in ("humerus_r", "ulna_r", "radius_r", "hand_r")}
    hum = sk.m.body("humerus_r").id
    if elbow is not None:
        obs_of = [[(b, o, t, w * (UPPER_ARM_WEIGHT if b == hum else 1.0)) for b, o, t, w in obs]
                  for obs in obs_of]
    ej = sk.m.joint("elbow_flex_r").id
    rng = sk.m.jnt_range[ej].copy()
    idx = []
    for name in ("girdle_r_x", "girdle_r_y", "girdle_r_z", "shoulder_r", "elbow_flex_r",
                 "pro_sup_r", "wrist_flex_r", "wrist_dev_r"):
        j = sk.m.joint(name).id
        a = sk.m.jnt_qposadr[j]
        idx += list(range(a, a + (4 if sk.m.jnt_type[j] == mujoco.mjtJoint.mjJNT_BALL else 1)))
    Q = Q.copy()
    ea = sk.m.jnt_qposadr[ej]
    try:
        for f in range(n):
            if not any(o[0] in arm_bodies for o in obs_of[f]):
                continue
            q0 = Q[f].copy()
            if elbow is not None:   # hold the elbow: clamp its range to the target
                sk.m.jnt_range[ej] = (elbow[f] - 1e-3, elbow[f] + 1e-3)
                q0[ea] = elbow[f]
            q = sk.solve(q0, obs_of[f], q0, q0,
                         sk.stance[f] if sk.stance is not None else (), iters=20, **_rowkw(sk, f))
            Q[f, idx] = q[idx]
    finally:
        sk.m.jnt_range[ej] = rng
    return Q


def _frame_cost(sk, q, obs):
    """Huber cost of one frame's markers (as in Skeleton.solve), and their median distance."""
    sk.d.qpos[:] = q
    mujoco.mj_kinematics(sk.m, sk.d)
    dist = np.array([np.linalg.norm(sk.d.xpos[b] + sk.d.xmat[b].reshape(3, 3) @ o - t) for b, o, t, _ in obs])
    if not len(dist):
        return 0.0, 0.0
    w = np.array([fw for *_, fw in obs])
    hub = np.where(dist < HUBER_M, 0.5 * dist ** 2, HUBER_M * (dist - 0.5 * HUBER_M))
    return float((w * hub).sum()), float(np.median(dist))


def unjam_limbs(sk, wtracks, attach, Q, centre, Q_prior, log=print, name=""):
    """A hinge at its straight stop while the markers beyond it are far off is a
    wrong solution branch, not a straight limb. Elbow: with two markers along
    the upper arm its roll is free; rolled the wrong way, bending moves the
    hand away from its markers, so the solver straightens the elbow as far as
    it goes instead (T4a's front arm, folded across the body in the run-up,
    came out hyperextended with its hand 28 cm off). Knee: a leg marked only on
    the foot can reach it by hip alone, and the knee sits at its stop while the
    foot trails its swinging markers (T3a's left foot, 39 cm). Each jammed
    stretch is restarted at its first frame from bent hinges (elbow also at four
    upper-arm rolls; a knee is not rolled, which could bend it backwards); the
    best start is carried through the stretch by the usual sequential solve,
    and kept only if the markers fit better over it."""
    n = len(Q)
    m = sk.m
    obs_of = _obs_lists(sk, wtracks, attach, n)
    stance = sk.stance if sk.stance is not None else [()] * n
    for hinge, ball, distal, straight, bends, rolls in JAM_LIMBS:
        for s in ("l", "r"):
            beyond = {m.body(f"{b}_{s}").id for b in distal}
            ej = m.joint(hinge.format(s=s)).id
            ea, stop = m.jnt_qposadr[ej], m.jnt_range[ej][straight]
            sa = m.joint(ball.format(s=s)).qposadr[0]
            jam = np.zeros(n, bool)
            for f in range(n):
                ob = [o for o in obs_of[f] if o[0] in beyond]
                if ob and abs(Q[f, ea] - stop) < JAM_STOP_RAD:
                    jam[f] = _frame_cost(sk, Q[f], ob)[1] > JAM_RESID_M
            for a, b in runs(jam):
                if b - a + 1 < JAM_FRAMES:
                    continue
                hi = min(n - 1, b + JAM_FRAMES)
                best_c, best_q = _frame_cost(sk, Q[a], obs_of[a])[0], None
                for roll in rolls:
                    for bend in bends:
                        q = Q[a].copy()
                        mujoco.mju_mulQuat(q[sa:sa + 4], Q[a, sa:sa + 4], _quat_axis([0, 1, 0], np.radians(roll)))
                        q[ea] = np.radians(bend)
                        # q as its own prediction: damping only, no pull back to the start
                        q = sk.solve(q, obs_of[a], q, Q_prior[a], stance[a], iters=40,
                                     **_rowkw(sk, a))
                        c = _frame_cost(sk, q, obs_of[a])[0]
                        if c < best_c:
                            best_c, best_q = c, q.copy()
                if best_q is None:
                    continue
                Qn = Q.copy()
                Qn[a] = best_q
                _solve_range(sk, obs_of, centre, Q_prior, Qn, a + 1, hi)
                old = sum(_frame_cost(sk, Q[f], obs_of[f])[0] for f in range(a, hi + 1))
                new = sum(_frame_cost(sk, Qn[f], obs_of[f])[0] for f in range(a, hi + 1))
                if new < old:
                    left = sum(abs(Qn[f, ea] - stop) < JAM_STOP_RAD for f in range(a, b + 1))
                    log(f"  [{name}] {hinge.format(s=s)} unjammed over {(b - a + 1) / 100:.2f} s "
                        f"(marker cost {new / old:.0%} of before; {left} frames still at the stop)")
                    Q = Qn
    return Q


def _slerp(a, b, u):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    dot = float(a @ b)
    if dot < 0:
        b, dot = -b, -dot
    if dot > 0.9995:
        q = a + u * (b - a)
        return q / np.linalg.norm(q)
    th = np.arccos(dot)
    return (np.sin((1 - u) * th) * a + np.sin(u * th) * b) / np.sin(th)


def infill(sk, Q, support, max_gap=INFILL_MAX):
    """Gap-fill in joint space: where no marker is on a joint's limb (the bone it
    moves or any bone beyond) for at most max_gap frames between two observed
    stretches, interpolate the joint between them (slerp for ball joints),
    instead of leaving it to the initial guess. Longer gaps keep the guess."""
    m = sk.m
    below = {b: {b} for b in range(m.nbody)}
    for b in range(1, m.nbody):
        p = m.body_parentid[b]
        while p > 0:
            below[p].add(b)
            p = m.body_parentid[p]
    Q = Q.copy()
    n = len(Q)
    for j in range(m.njnt):
        if m.joint(j).name in ("root", "ball_free"):
            continue
        seen = support[:, sorted(below[m.jnt_bodyid[j]])].any(axis=1)
        held = np.zeros(n, bool)   # rows where a template holds this limb joint
        for key, mask in sk.tmpl_rows.items():
            limb, side = key.split("_")
            if m.joint(j).name in {a.format(s=side) for a in LIMB_JOINTS[limb]}:
                held |= mask
        a = m.jnt_qposadr[j]
        ball = m.jnt_type[j] == mujoco.mjtJoint.mjJNT_BALL
        for s, e in runs(~seen):
            if s == 0 or e == n - 1 or e - s + 1 > max_gap or held[s:e + 1].any():
                continue
            qa, qb = Q[s - 1], Q[e + 1]
            for f in range(s, e + 1):
                u = (f - s + 1) / (e - s + 2)
                if ball:
                    Q[f, a:a + 4] = _slerp(qa[a:a + 4], qb[a:a + 4], u)
                else:
                    Q[f, a] = (1 - u) * qa[a] + u * qb[a]
    return Q


def _solve_range(sk, obs_of, centre, Q_prior, Q, lo, hi):
    """Re-solve rows lo..hi of Q in place, warm-started from the rows before."""
    q_prev = Q[lo - 1].copy() if lo >= 1 else Q_prior[0].copy()
    q_prev2 = Q[lo - 2].copy() if lo >= 2 else q_prev.copy()
    for f in range(lo, hi + 1):
        v = sk._diff(q_prev2, q_prev) * 0.8
        full = np.zeros(sk.m.nv)
        full[:sk.nv] = v
        q_pred = q_prev.copy()
        mujoco.mj_integratePos(sk.m, q_pred, full, 1.0)
        if f == 0:
            q_pred = Q_prior[0].copy()
        elif len(obs_of[f]) < 4:      # too few markers: follow the cloud, relax towards the guess
            q_pred[0:2] = q_prev[0:2] + (centre[f, :2] - centre[f - 1, :2])
        Q[f] = sk.solve(q_pred.copy(), obs_of[f], q_pred, Q_prior[f],
                        sk.stance[f] if sk.stance is not None else (), **_rowkw(sk, f))
        q_prev2, q_prev = q_prev, Q[f].copy()
    return Q


def _to_shell(sk, b, o, site=""):
    """Where a marker can sit on bone b: within SITE_MAX_M of one of the bone's
    anatomical landmarks (mocap/skeleton3d.py), and between SHELL_M[0] and
    SHELL_M[1] from the bone's surface. Without the first, a marker on a long
    bone can slide along it and the fit trades that for a contorted pose.
    site: the track's own landmark, if labelled by hand; it is then kept within
    SITE_LABEL_M of that one. Any humerus landmark would let a lateral upper-arm
    marker sit in front or medial, and the arm's roll then follows whichever
    side the marker was put on (T4a's front arm came out rolled half a turn)."""
    named = sk.site_named.get(site)
    S, rmax = (named[1][None], SITE_LABEL_M) if named and named[0] == b else (sk.sites_on.get(b), SITE_MAX_M)
    for _ in range(2):
        if S is not None:
            dist = np.linalg.norm(S - o, axis=1)
            i = int(dist.argmin())
            if dist[i] > rmax:
                o = S[i] + (o - S[i]) * rmax / dist[i]
        dist, i = sk.surface[b].query(o)
        v = sk.surface[b].data[i]
        if dist > SHELL_M[1]:
            o = v + (o - v) * SHELL_M[1] / dist
        elif dist < SHELL_M[0]:
            out = o - sk.surface[b].data.mean(0)
            o = v + out / (np.linalg.norm(out) + 1e-9) * SHELL_M[0]
    return o


def attach_tracks(sk, Q, wtracks, max_rms, upto=None):
    """Each track to the bone it moves with (lowest RMS in that bone's frame).
    upto: judge only frames before this row -- for the first match against the
    upright-running guess, which is good in the run-up and badly wrong in the
    delivery (a 3 s trunk track would otherwise look 20 cm off its bone)."""
    pos, rot = sk.frames_of(Q)
    attach, rms = {}, {}
    for tr in wtracks:
        f1 = tr.f1 if upto is None else min(tr.f1, upto - 1)
        if f1 - tr.f0 + 1 < MIN_TRACK_FRAMES:
            continue
        R, p = rot[tr.f0:f1 + 1], pos[tr.f0:f1 + 1]
        local = np.einsum("tbji,tbj->tbi", R, tr.P[:f1 - tr.f0 + 1, None, :] - p)      # (t, nb, 3)
        best = None
        for k, b in enumerate(sk.bones):
            name = BONES[k]
            if tr.pinned and name not in tr.pinned:
                continue
            if name.split("_")[0] in SIDE_SLACK_M and not _side_ok(sk, Q, tr, name):
                continue
            if not _plausible(sk, name, local[:, k].mean(0)):
                continue
            o = _to_shell(sk, b, local[:, k].mean(0), tr.site)
            e = float(np.sqrt(np.mean(np.sum((local[:, k] - o) ** 2, axis=1))))
            if best is None or e < best[0]:
                best = (e, name, o)
        if best and (best[0] < max_rms or tr.pinned):
            attach[tr.tid] = (best[1], best[2])
            rms[tr.tid] = best[0]
    return attach, rms


def _local(sk, Q, tr, k):
    """Track tr in bone k's frame, over its frames: (t, 3)."""
    pos, rot = sk.frames_of(Q[tr.f0:tr.f1 + 1])
    return np.einsum("tji,tj->ti", rot[:, k], tr.P - pos[:, k])


def revalidate(sk, Q, wtracks, attach, max_rms):
    """Re-estimate each attachment's offset on its own bone; drop it if the
    track no longer moves with that bone (pinned tracks are kept)."""
    out, rms = {}, {}
    for tr in wtracks:
        if tr.tid not in attach:
            continue
        b = attach[tr.tid][0]
        k = BONES.index(b)
        L = _local(sk, Q, tr, k)
        o = _to_shell(sk, sk.bones[k], L.mean(0), tr.site)
        e = float(np.sqrt(np.mean(np.sum((L - o) ** 2, axis=1))))
        if e < max_rms or tr.pinned:
            out[tr.tid], rms[tr.tid] = (b, o), e
    return out, rms


def rigid_partners(wtracks):
    """{tid: [tids it is rigid with]}: spacing sd < RIGID_SD_M while the pair turns > RIGID_SWING_DEG."""
    out = {}
    for i, a in enumerate(wtracks):
        for b in wtracks[i + 1:]:
            s, e = max(a.f0, b.f0), min(a.f1, b.f1)
            if e - s + 1 < RIGID_FRAMES:
                continue
            v = a.P[s - a.f0:e - a.f0 + 1] - b.P[s - b.f0:e - b.f0 + 1]
            dist = np.linalg.norm(v, axis=1)
            u = v / dist[:, None]
            mean = u.mean(0) / np.linalg.norm(u.mean(0))
            swing = np.degrees(np.arccos(np.clip(u @ mean, -1, 1))).max()
            if dist.std() < RIGID_SD_M and swing > RIGID_SWING_DEG:
                out.setdefault(a.tid, []).append(b.tid)
                out.setdefault(b.tid, []).append(a.tid)
    return out


def _window_resid(sk, Q, wtracks, attach, lo, hi, skip):
    pos, rot = sk.frames_of(Q[lo:hi + 1])
    d = []
    for tr in wtracks:
        if tr.tid == skip or tr.tid not in attach:
            continue
        s, e = max(tr.f0, lo), min(tr.f1, hi)
        if s > e:
            continue
        b, o = attach[tr.tid]
        k = BONES.index(b)
        pred = pos[s - lo:e - lo + 1, k] + rot[s - lo:e - lo + 1, k] @ o
        d.append(np.linalg.norm(pred - tr.P[s - tr.f0:e - tr.f0 + 1], axis=1))
    return float(np.median(np.concatenate(d))) if d else 0.0


def _side_ok(sk, Q, tr, bone):
    """A leg track must sit on its bone's side of the pelvis (SIDE_SLACK_M) and
    below the bone's height cap (LEG_HEIGHT_MAX_M: the hands hang at thigh
    height, and a thigh with no other marker bends to follow one)."""
    kind, _, side = bone.partition("_")
    if kind not in SIDE_SLACK_M:
        return True
    z = np.median(tr.P[:, 2])
    if z > LEG_HEIGHT_MAX_M[kind] * sk.scale or z < LEG_HEIGHT_MIN_M.get(kind, -1.0) * sk.scale:
        return False
    lat = []
    for i in range(0, len(tr.P), 5):
        sk.d.qpos[:] = Q[tr.f0 + i]
        mujoco.mj_kinematics(sk.m, sk.d)
        right = sk.d.xmat[sk.pelvis].reshape(3, 3)[:, 2]          # pelvis z = the right
        lat.append((tr.P[i] - sk.d.xpos[sk.pelvis]) @ right)
    lat = float(np.median(lat)) * (1 if side == "r" else -1)
    return lat > -SIDE_SLACK_M[kind]


def _plausible(sk, bone, local_mean):
    """Anatomy the shell and landmark rules do not catch. A marker near the
    shoulder joint is on the scapula (acromion), which slides over the rigid
    model torso and so fits no bone well; on the upper arm it would lift the
    arm to reach it, so it is not allowed there (IOR's upper-arm markers sit
    mid-arm and at the elbow)."""
    if bone.startswith("humerus"):
        return np.linalg.norm(local_mean) > SHOULDER_CLEAR_M * sk.scale
    return True


def _leg_band_ok(bone, z):
    """A leg bone with no marker yet may still be tried for a track at the right
    height (median, world z) for it: a guessed leg can be 40 cm off a foot marker,
    which then never latches on. Height is what keeps a guessed foot from
    grabbing a back marker (the failure the support rule was made for)."""
    band = LEG_BAND_M.get(bone.split("_")[0])
    return band is not None and band[0] <= z <= band[1]


def _bone_support(wtracks, attach, n):
    """{bone: (n,) bool} -- an attached track on the bone in that frame."""
    sup = {b: np.zeros(n, bool) for b in BONES}
    for tr in wtracks:
        if tr.tid in attach:
            sup[attach[tr.tid][0]][tr.f0:tr.f1 + 1] = True
    return sup


def grow(sk, Q, wtracks, attach, centre, Q_prior, log, name):
    """Test attachments for unattached tracks by re-fitting their stretch of time
    with them attached; keep one if it fits and does not degrade the others.

    A bone with no marker on it or its neighbours is posed by the priors alone
    and can bend to reach any track, so a good fit there is no evidence: such a
    bone is only tried if the track already fits it within UNSUPPORTED_MAX_M.
    Candidates are tried in tiers, and the first tier with a fit wins: the bone
    of a rigid partner; then its neighbours and bones with marker support; then
    the rest. Otherwise a loosely held limb next door wins on fit alone."""
    n = len(Q)
    partners = rigid_partners(wtracks)
    added = 0
    for tr in sorted((t for t in wtracks if t.tid not in attach), key=lambda t: -len(t.P)):
        pos, rot = sk.frames_of(Q[tr.f0:tr.f1 + 1])
        local = np.einsum("tbji,tbj->tbi", rot, tr.P[:, None, :] - pos)
        e = [np.sqrt(np.mean(np.sum((local[:, k] - local[:, k].mean(0)) ** 2, axis=1)))
             for k in range(len(sk.bones))]
        sup = _bone_support(wtracks, attach, n)
        mates = {attach[p][0] for p in partners.get(tr.tid, []) if p in attach}
        own_sup, near_sup = set(), set()
        z = float(np.median(tr.P[:, 2]))
        by_height = [BONES.index(b) for b in BONES if _leg_band_ok(b, z)]
        for k in dict.fromkeys(list(np.argsort(e)[:4]) + by_height):
            b = BONES[k]
            on = sup[b][tr.f0:tr.f1 + 1].mean() >= 0.5
            # only the bone itself or one further out holds a limb bone's orientation
            out = [sup[nb] for nb in DISTAL.get(b, ())] or [np.zeros(n, bool)]
            near = np.any(out, axis=0)[tr.f0:tr.f1 + 1].mean() >= 0.5
            if on:
                own_sup.add(b)
            elif near or e[k] < UNSUPPORTED_MAX_M or _leg_band_ok(b, z):
                near_sup.add(b)
        tiers = [mates, {nb for b in mates for nb in NEIGHBOURS[b]} | own_sup, near_sup]
        tiers = [{b for b in t if _side_ok(sk, Q, tr, b)} for t in tiers]
        if tr.pinned:
            tiers = [t & set(tr.pinned) for t in tiers]
        lo, hi = max(0, tr.f0 - 3), min(n - 1, tr.f1 + 3)
        before = _window_resid(sk, Q, wtracks, attach, lo, hi, tr.tid)
        best = None
        tried = set()
        for tier in tiers:
            if best:
                break
            for b in tier - tried:
                tried.add(b)
                best = _try_bone(sk, Q, wtracks, attach, centre, Q_prior, tr, b, local, lo, hi,
                                 before, best)
        if best:
            attach[tr.tid] = (best[1], best[2])
            Q = best[3]
            added += 1
            log(f"  [{name}] + T{tr.tid} ({len(tr.P)} frames) on {best[1]}: fits within "
                f"{best[4] * 100:.1f} cm, others {best[5] * 100:+.1f} cm")
    return Q, attach, added


def _try_bone(sk, Q, wtracks, attach, centre, Q_prior, tr, b, local, lo, hi, before, best):
    """Re-fit [lo, hi] with tr attached to bone b; return the better of this and best."""
    n = len(Q)
    k = BONES.index(b)
    raw = local[:, k].mean(0)
    o0 = _to_shell(sk, sk.bones[k], raw, tr.site)
    if not _plausible(sk, b, raw):
        return best
    att = dict(attach)
    att[tr.tid] = (b, o0)
    Qh = Q.copy()
    for _ in range(2):
        _solve_range(sk, _obs_lists(sk, wtracks, att, n), centre, Q_prior, Qh, lo, hi)
        att[tr.tid] = (b, _to_shell(sk, sk.bones[k], _local(sk, Qh, tr, k).mean(0), tr.site))
    L = _local(sk, Qh, tr, k)
    own = float(np.median(np.linalg.norm(L - att[tr.tid][1], axis=1)))
    worse = _window_resid(sk, Qh, wtracks, att, lo, hi, tr.tid) - before
    # how much the body had to move to take the marker: prefer the explanation
    # that changes the pose least (a shoulder-height marker on the torso, not
    # on an upper arm lifted 60 deg to reach it)
    moved = float(np.mean([np.linalg.norm(sk._diff(Q[f], Qh[f])[6:]) for f in range(lo, hi + 1)]))
    if own < GROW_MAX_M and worse < GROW_COST_M:
        score = own + 2 * max(worse, 0) + MOVE_COST_M * moved
        if best is None or score < best[0]:
            best = (score, b, att[tr.tid][1], Qh, own, worse)
    return best


def residuals(sk, Q, wtracks, attach):
    n = len(Q)
    pos, rot = sk.frames_of(Q)
    per = [[] for _ in range(n)]
    support = np.zeros((n, sk.m.nbody), bool)
    for tr in wtracks:
        if tr.tid not in attach:
            continue
        b, o = attach[tr.tid]
        k = BONES.index(b)
        pred = pos[tr.f0:tr.f1 + 1, k] + rot[tr.f0:tr.f1 + 1, k] @ o
        for i, dd in enumerate(np.linalg.norm(pred - tr.P, axis=1)):
            per[tr.f0 + i].append(dd)
        support[tr.f0:tr.f1 + 1, sk.m.body(b).id] = True
    n_obs = np.array([len(p) for p in per])
    med = np.array([np.median(p) if p else np.nan for p in per])
    p90 = np.array([np.percentile(p, 90) if p else np.nan for p in per])
    return n_obs, med, p90, support


def _rounds(sk, wtracks, n, centre, Q_prior, attach, thresholds, name, log, res=None):
    for r, thr in enumerate(thresholds):
        Q = fit_poses(sk, wtracks, attach, n, centre, Q_prior)
        new, rms = attach_tracks(sk, Q, wtracks, thr)
        changed = sum(attach.get(t, (None,))[0] != new.get(t, (None,))[0] for t in set(attach) | set(new))
        attach = new
        n_obs, med, *_ = residuals(sk, Q, wtracks, attach)
        if res is not None:
            res.history.append((thr, len(attach), changed, float(np.nanmedian(med))))
        log(f"  [{name}] threshold {thr * 100:.1f} cm: {len(attach)} tracks attached, {changed} "
            f"changed bone, median residual {np.nanmedian(med) * 100:.1f} cm")
    return Q, attach, rms


def fit(name, trial, tracks, first, last, release, direction, x_offset, z_centre, run, pins=None,
        scale=None, judge_runup=True, arm_scale=1.0, template=None, log=print):
    """Fit one window [first, last] (absolute frames). pins: {tid: key of PINS}.
    scale: the participant's skeleton scale; None searches SCALES for it.
    arm_scale: the arms' length relative to that (fit_mocap3d.measure_arm_scale)."""
    n = last - first + 1
    wtracks = world_tracks(tracks, first, last, direction, x_offset, z_centre)
    for tr in wtracks:
        if pins and tr.tid in pins:
            v = pins[tr.tid]
            if isinstance(v, dict):   # labelled by hand: {"bones": (...), "site": "L_HUM"}
                tr.pinned, tr.site = tuple(v["bones"]), v.get("site", "")
            else:
                tr.pinned = PINS[v] if isinstance(v, str) else tuple(v)
    centre = body_centre(wtracks, n)
    res = Fit3D(name, first, release - first, trial.rate)

    frames = first + np.arange(n)
    stance = stance_schedule(run, first, n)
    sk = Skeleton(scale or 1.0, arm_scale)
    sk.stance = stance
    Q0 = initial_poses(sk, centre, run, frames)
    runup_end = res.release - int(RUNUP_JUDGE_S * trial.rate) if judge_runup else None
    shift = _align(sk, Q0, wtracks, runup_end)
    Q0[:, 0:3] += shift
    apply_template(sk, Q0, template, res.release)
    sk.upright = np.arange(n) < res.release - int(RUNUP_JUDGE_S * trial.rate)
    centre = centre + shift
    log(f"  [{name}] initial guess shifted by {np.round(shift * 100).astype(int)} cm to fit the markers")
    attach, rms = attach_tracks(sk, Q0, wtracks, ROUNDS_M[0] * 1.5, upto=runup_end)
    Q, attach, rms = _rounds(sk, wtracks, n, centre, Q0, attach, ROUNDS_M, name, log, res)
    Q, attach, rms = _grow_rounds(sk, Q, wtracks, attach, centre, Q0, name, log)

    if scale is None:
        sk, Q, Q0, attach, rms = _scale_search(sk, wtracks, attach, centre, run, frames, stance,
                                               name, log)
    Q = fit_poses(sk, wtracks, attach, n, centre, Q0, fade=True)
    Q = unjam_limbs(sk, wtracks, attach, Q, centre, Q0, log, name)
    support = residuals(sk, Q, wtracks, attach)[3]
    Q_fill = infill(sk, Q, support)
    Q = smooth(sk, wtracks, attach, Q_fill, Q_fill)
    Q = lowpass_qpos(sk.m, Q, trial.rate)
    # a pinned bowling forearm/hand marker seen at release (manual labels also pin the front arm)
    arm_pinned = any(tr.pinned and tr.tid in attach and attach[tr.tid][0] in ("ulna_r", "radius_r", "hand_r")
                     and tr.f0 <= res.release <= tr.f1 for tr in wtracks)
    if arm_pinned:   # bowling arm tracked through release: show the elbow straightened
        elbow, res.elbow_offset_deg = straighten_elbow(sk, Q, res.release, trial.rate)
        res.elbow_offset_deg = float(np.degrees(res.elbow_offset_deg))
        log(f"  [{name}] bowling elbow: {res.elbow_offset_deg:.0f} deg marker offset removed")
        Q = refine_arm(sk, wtracks, attach, Q, elbow)
    else:
        Q = refine_arm(sk, wtracks, attach, Q)
    res.scale, res.qpos, res.attach, res.rms, res.xml = sk.scale, Q, attach, rms, sk.xml
    res.n_obs, res.resid_med, res.resid_p90, res.support = residuals(sk, Q, wtracks, attach)
    log(f"  [{name}] final: scale {sk.scale:.2f}, {len(attach)} of {len(wtracks)} tracks attached, "
        f"median residual {np.nanmedian(res.resid_med) * 100:.1f} cm, "
        f"{np.median(res.n_obs):.0f} markers per frame")
    return res, wtracks


def _align(sk, Q0, wtracks, upto, step=ALIGN_STEP_M):
    """Shift of the whole initial guess that lets the most tracks attach. The
    marker cloud's median, where the guess starts, is off the body whenever the
    visible markers are lopsided (several on the bowling arm, few on the trunk),
    and the attachment rounds cannot recover from a start that misses the trunk."""
    best = None
    for dx in (-2, -1, 0, 1, 2):
        for dy in (-2, -1, 0, 1, 2):
            for dz in (-1, 0, 1):
                shift = np.array([dx, dy, dz]) * step
                Q = Q0.copy()
                Q[:, 0:3] += shift
                attach, rms = attach_tracks(sk, Q, wtracks, ROUNDS_M[0] * 1.5, upto=upto)
                score = (len(attach), -sum(rms.values()), -np.linalg.norm(shift))
                if best is None or score > best[0]:
                    best = (score, shift)
    return best[1]


def _scale_search(sk, wtracks, attach, centre, run, frames, stance, name, log):
    """Uniform skeleton scale with the lowest median residual (every 2nd frame,
    attachments fixed), then re-attach at that scale."""
    n = len(frames)
    sub = np.arange(0, n, 2)
    wsub = [t for t in (_sub(tr, sub) for tr in wtracks) if len(t.P)]
    best = None
    for s in SCALES:
        sks = Skeleton(s, sk.arm_scale)
        sks.stance = [stance[i] for i in sub]
        att = {t: (b, o * s / sk.scale) for t, (b, o) in attach.items()}
        Qs = fit_poses(sks, wsub, att, len(sub), centre[sub], initial_poses(sks, centre[sub], run, frames[sub]))
        _, med, *_ = residuals(sks, Qs, wsub, att)
        score = float(np.nanmedian(med))
        log(f"  [{name}] scale {s:.2f}: median residual {score * 100:.2f} cm")
        if best is None or score < best[0]:
            best = (score, s)
    new = Skeleton(best[1], sk.arm_scale)
    new.stance = stance
    attach = {t: (b, o * best[1] / sk.scale) for t, (b, o) in attach.items()}
    Q0 = initial_poses(new, centre, run, frames)
    Q = fit_poses(new, wtracks, attach, n, centre, Q0)
    attach, rms = revalidate(new, Q, wtracks, attach, ROUNDS_M[-1])
    Q, attach, rms = _grow_rounds(new, Q, wtracks, attach, centre, Q0, name, log)
    return new, Q, Q0, attach, rms


def save(path, res, wtracks, trial, direction, x_offset, z_centre, **extra):
    """Everything the replay and the report need, without the source c3d or
    LocoMuJoCo: poses, the fitted model, the window's raw markers in the same
    world frame (NaN = not seen) flagged by whether the fit used them, and the
    quality numbers."""
    n = len(res.qpos)
    frames = res.first + np.arange(n)
    raw = to_world(trial.pos[frames], direction, x_offset, z_centre)
    used = np.zeros(raw.shape[:2], bool)
    for tr in wtracks:
        if tr.tid not in res.attach:
            continue
        for i, f in enumerate(range(tr.f0, tr.f1 + 1)):
            dd = np.linalg.norm(raw[f] - tr.P[i], axis=1)
            if np.isfinite(dd).any():
                j = int(np.nanargmin(dd))
                used[f, j] = dd[j] < 1e-6
    tids = sorted(res.attach)
    m = mujoco.MjModel.from_xml_string(res.xml)
    np.savez_compressed(
        path, qpos=res.qpos, first_frame=res.first, release_index=res.release, rate=res.rate,
        scale=res.scale, xml=res.xml, markers=raw.astype(np.float32), marker_used=used,
        n_obs=res.n_obs, resid_med=res.resid_med, resid_p90=res.resid_p90, support=res.support,
        body_names=np.array([m.body(b).name for b in range(m.nbody)]),
        attach_tid=np.array(tids, int), attach_bone=np.array([res.attach[t][0] for t in tids]),
        attach_offset=np.array([res.attach[t][1] for t in tids]).reshape(-1, 3),
        attach_rms=np.array([res.rms.get(t, np.nan) for t in tids]), n_tracks=len(wtracks),
        elbow_offset_deg=res.elbow_offset_deg,
        direction=direction, x_offset=x_offset, z_centre=z_centre, **extra)


def _grow_rounds(sk, Q, wtracks, attach, centre, Q_prior, name, log, max_rounds=3):
    rms = {}
    for _ in range(max_rounds):
        Q, attach, added = grow(sk, Q, wtracks, attach, centre, Q_prior, log, name)
        Q = fit_poses(sk, wtracks, attach, len(Q), centre, Q_prior)
        attach, rms = revalidate(sk, Q, wtracks, attach, ROUNDS_M[-1])
        _, med, *_ = residuals(sk, Q, wtracks, attach)
        log(f"  [{name}] grow: +{added}, {len(attach)} attached after re-check, median residual "
            f"{np.nanmedian(med) * 100:.1f} cm")
        if not added:
            break
    return Q, attach, rms


def _sub(tr, sub):
    """A track restricted to the frames in `sub` (sorted indices), re-indexed."""
    keep = (sub >= tr.f0) & (sub <= tr.f1)
    if not keep.any():
        return WTrack(tr.tid, 0, np.zeros((0, 3)), tr.aim, tr.pinned, tr.site)
    idx = sub[keep]
    return WTrack(tr.tid, int(np.argmax(keep)), tr.P[idx - tr.f0], tr.aim, tr.pinned, tr.site)
