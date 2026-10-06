"""Map a captured delivery onto the planar bowling arm of assets/bowler.xml.

The humanoid's bowling arm is two hinges about the same lateral axis
(shoulder, elbow), on a torso that BowlerEnv keeps upright, so the
retargeting problem reduces to two angles per frame in the delivery
(sagittal) plane, in the model's own convention:

  shoulder theta: direction of the arm, atan2(-up, forward); 90 = hanging
      down, 180 = horizontal behind, 270 = vertical, 360 = forward.
  elbow: the forearm's further rotation in the same sense, 0 = straight.

Two choices are forced by the markers available, and both are recorded in
the output rather than hidden:

  * theta comes from the WHOLE-arm direction (upper-arm marker -> wrist
    centre), not the upper arm alone. The only upper-arm marker found sits
    ~18 cm from the elbow marker, slightly off the humeral axis, so the
    short upper-arm vector carries a large constant angular offset; the
    43 cm baseline to the wrist is much less sensitive to it, and equals
    the upper-arm direction whenever the elbow is straight.
  * the model elbow is the measured flexion minus its minimum over the
    delivery (the same marker offset makes the absolute flexion read
    ~40 deg too high -- see REPORT.md). This assumes the arm reaches
    full extension at some point in the delivery, which is the usual
    straight-arm action but is an assumption, not a measurement.

Root position: horizontal travel of the trunk (spine column if identified,
otherwise the median of every visible non-arm marker), scaled by nothing
-- the model and the bowler cover the same ground. Root height and the
legs are NOT measured (no pelvis or leg markers could be identified around
these deliveries) and are held at BowlerEnv's delivery-stride pose; the
output flags this.
"""
import numpy as np

from envs.runup import DELIVERY_STRIDE_DEG, LEG_JOINT_NAMES

MODEL_SHOULDER_RANGE = (60.0, 450.0)
MODEL_ELBOW_RANGE = (0.0, 160.0)


def _unwrap_deg(x):
    out = np.full_like(x, np.nan)
    ok = ~np.isnan(x)
    if ok.any():
        out[ok] = np.degrees(np.unwrap(np.radians(x[ok])))
    return out


def planar_arm(res):
    """(shoulder_deg, elbow_deg) series in the model convention, over the
    delivery window, plus a dict describing how they were made."""
    d = res.direction
    prox, elbow, wrist = res.series["proximal"], res.series["elbow"], res.series["wrist"]
    if np.all(np.isnan(wrist[:, 0])):
        wrist = res.series["hand"]
    arm = wrist - prox
    theta = np.degrees(np.arctan2(-arm[:, 1], d * arm[:, 0]))
    theta = _unwrap_deg(np.mod(theta, 360))
    # put the release value in [180, 450) so the series lives in the joint's range
    k = res.release - res.frame0
    if not np.isnan(theta[k]):
        shift = 360 * np.floor((theta[k] - 180) / 360)
        theta = theta - shift
    flex = res.series["elbow_flexion"]
    lo, hi = max(k - 30, 0), min(k + 15, len(flex))
    offset = float(np.nanmin(flex[lo:hi]))
    model_elbow = np.clip(flex - offset, *MODEL_ELBOW_RANGE)
    info = {"theta_source": "upper-arm marker -> wrist centre",
            "elbow_offset_removed_deg": offset,
            "theta_at_release_deg": float(theta[k]),
            "elbow_at_release_deg": float(model_elbow[k])}
    return theta, model_elbow, info


def root_track(res, tracks):
    """Forward position (delivery frame X, metres) of the trunk over the window."""
    n = len(res.series["elbow"])
    if res.spine and res.spine.ids:
        top, bot = res.series["spine_top"], res.series["spine_bottom"]
        c = (top + bot) / 2
        x = res.direction * c[:, 0]
        src = "spine column midpoint"
    else:
        arm_ids = set(res.arm.all_ids())
        x = np.full(n, np.nan)
        for i in range(n):
            f = res.frame0 + i
            vals = [t.at(f)[0] for t in tracks if t.tid not in arm_ids and t.start <= f <= t.end]
            if len(vals) >= 4:
                x[i] = res.direction * np.median(vals)
        src = "median of visible non-arm markers"
    return x, src


def reference_motion(res, tracks, model):
    """Per-frame qpos for assets/bowler.xml over the delivery window.

    Frames where the arm angles are missing are linearly interpolated if
    the gap is <= 5 frames, else left NaN (mask provided)."""
    theta, elbow, info = planar_arm(res)
    x, src = root_track(res, tracks)
    n = len(theta)
    t = np.arange(n)
    measured = ~np.isnan(theta) & ~np.isnan(elbow)
    for arr in (theta, elbow, x):
        ok = ~np.isnan(arr)
        if ok.sum() >= 2:
            gaps = np.interp(t, t[ok], arr[ok])
            # only bridge short gaps
            from mocap.signal import runs
            for s, e in runs(~ok):
                if s > 0 and e < n - 1 and e - s + 1 <= 5:
                    arr[s:e + 1] = gaps[s:e + 1]
    # root x relative to its value at release, placed so that release
    # happens where BowlerEnv's scripted delivery releases (x = 0)
    k = res.release - res.frame0
    x = x - x[k]
    qpos = np.full((n, model.nq), np.nan)
    root = model.joint("root_free").qposadr[0]
    qpos[:, root:root + 3] = np.stack([x, np.zeros(n), np.full(n, 0.80)], 1)
    qpos[:, root + 3:root + 7] = [1, 0, 0, 0]
    for name in LEG_JOINT_NAMES:
        qpos[:, model.joint(name).qposadr[0]] = np.radians(DELIVERY_STRIDE_DEG[name])
    qpos[:, model.joint("shoulder").qposadr[0]] = np.radians(theta)
    qpos[:, model.joint("elbow").qposadr[0]] = np.radians(elbow)
    ball = model.joint("ball_free").qposadr[0]
    qpos[:, ball:ball + 7] = np.nan   # the ball is simulated, not prescribed
    info.update(root_source=src, root_height="held at 0.80 m (not measured)",
                legs="held at BowlerEnv DELIVERY_STRIDE_DEG (not measured)")
    return qpos, measured, info
