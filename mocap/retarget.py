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

The reference motion covers the whole run-up, from standing still to
release, plus the follow-through. What is measured and what is not:

  root forward travel   MEASURED. Around release: the trunk (spine column if
                        identified, else the median of the visible non-arm
                        markers), as before. Earlier in the run-up: the
                        label-free whole-body position of mocap/runup.py,
                        blended in over SPLICE_S. Not scaled -- the model
                        and the bowler cover the same ground, and the
                        markers drawn by scripts/demo_mocap.py
                        (marker_cloud) sit in the same frame.
  root height           MEASURED shape, model-anchored level. The whole-body
                        bob of each running step and the jump (mocap/runup.py)
                        ride on the model's straight-leg stance height, so the
                        model's foot is on the ground at each measured
                        mid-stance. Settles to BowlerEnv's delivery-stride
                        height SETTLE_S before release, where the measured
                        signal stops being pelvis-like (trunk flexion).
  legs                  NOT MEASURED (no foot or leg marker can be identified
                        through the run-up). envs/runup.py's running cycle,
                        phase-locked to the measured step times, its
                        amplitude scaled with measured speed, then blended
                        into BowlerEnv's delivery-stride pose between the top
                        of the jump and SETTLE_S before release. Which foot
                        is down at each step is not measured either: steps
                        alternate backwards from a left-foot take-off, as for
                        a right-arm bowler (the model's bowling arm is right).
  bowling arm           MEASURED over the delivery (see above). Before the
                        measured stretch the arm is held in BowlerEnv's carry
                        pose and blended into the first measured frame over
                        ARM_BLEND_S.
"""
import numpy as np

from envs.runup import CARRY_ARM_DEG, DELIVERY_STRIDE_DEG, LEG_JOINT_NAMES, run_cycle_at_phase_deg
from mocap.signal import runs

MODEL_SHOULDER_RANGE = (60.0, 450.0)
MODEL_ELBOW_RANGE = (0.0, 160.0)
Z_STANCE = 0.84         # model hip-to-sole height with a straight, vertical leg
Z_DELIVERY = 0.80       # BowlerEnv's pelvis height in the delivery stride
FULL_STRIDE_MS = 3.0    # forward speed (m/s) at which the running cycle reaches full amplitude
SWING_AMP_DEG, KNEE_LIFT_DEG = 28.0, 35.0   # envs/runup.py's full-speed cycle
ARM_BLEND_S = 0.3
SETTLE_S = 0.1          # ~front-foot contact: delivery-stride pose and height reached by then
STAND_S = 0.3           # standing still shown before the run starts
SPLICE_S = 0.3          # run-up path blends into the trunk track over this, before the delivery window


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


def _smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


def _bridge_short_gaps(arr, max_gap=5):
    n = len(arr)
    t = np.arange(n)
    ok = ~np.isnan(arr)
    if ok.sum() >= 2:
        lin = np.interp(t, t[ok], arr[ok])
        for s, e in runs(~ok):
            if s > 0 and e < n - 1 and e - s + 1 <= max_gap:
                arr[s:e + 1] = lin[s:e + 1]
    return arr


def stride_phase(run, frames):
    """Running-cycle phase (rad) per frame: right foot at mid-stance at 0 mod
    2*pi, left at pi (envs.runup.run_cycle_at_phase_deg). Locked to the
    measured steps, the last of which (take-off) is the left foot. Before the
    first detected step the phase advances with distance travelled at the
    first detected step length, so the legs slow down with the body; after
    take-off it keeps the last step's rate."""
    x = run.x[frames - run.frame0]
    s = np.array(run.steps)
    if len(s) == 0:
        # no steps found: 1 m steps, left foot down 0.73 s before release (the
        # median take-off time where it was found)
        x0 = run.x[run.i(run.release - int(0.73 * run.rate))]
        return np.pi + np.pi * (x - x0) / 1.0
    k = len(s)
    phi_s = np.pi * (1 - (k - 1 - np.arange(k)))     # ..., -pi, 0, pi (take-off)
    phi = np.interp(frames, s, phi_s)
    xs = run.x[s - run.frame0]
    step_len = xs[1] - xs[0] if k >= 2 else 1.0
    step_t = s[-1] - s[-2] if k >= 2 else 0.3 * run.rate
    before, after = frames < s[0], frames > s[-1]
    phi[before] = phi_s[0] - np.pi * (xs[0] - x[before]) / step_len
    phi[after] = phi_s[-1] + np.pi * (frames[after] - s[-1]) / step_t
    return phi


def root_height_running(run, frames):
    """Model root height from the measured whole-body bob: the stance height at
    each measured mid-stance, plus the measured rise above the line joining
    them. Standing (before the first step): stance height. After take-off:
    the measured rise above the take-off point -- the jump."""
    h = run.h[frames - run.frame0]
    s = np.array(run.steps)
    if len(s) == 0:
        return np.full(len(frames), Z_STANCE)
    env = np.interp(frames, s, run.h[s - run.frame0])
    env[frames < s[0]] = h[frames < s[0]]
    return Z_STANCE + h - env


def marker_cloud(trial, direction, frames, x_offset, z_centre):
    """Every exported marker over `frames`, in the reference motion's world
    frame: x forward with the trunk at x = 0 at release (x_offset is the
    trunk's direction-signed lab x then, events["x_offset_m"]), y = the
    model's left, z up. For display: the raw label columns, NaN where a
    marker is missing, labels unused -- exactly what the export holds."""
    P = trial.pos[frames]            # lab: x along the room, y up, z lateral
    out = np.empty_like(P)
    out[..., 0] = direction * P[..., 0] - x_offset
    out[..., 1] = -direction * (P[..., 2] - z_centre)   # bowler's right is the model's -y
    out[..., 2] = P[..., 1]
    return out


def reference_motion(res, run, tracks, model):
    """Per-frame qpos for assets/bowler.xml from STAND_S before the run starts
    until the end of the delivery window (see the module docstring for what
    is measured). Returns (qpos, measured_mask, events, info): measured_mask
    marks frames whose bowling-arm angles are measured; events holds frame
    indices into qpos."""
    theta, elbow, info = planar_arm(res)
    x_trunk, src = root_track(res, tracks)
    for arr in (theta, elbow, x_trunk):
        _bridge_short_gaps(arr)
    k_rel_d = res.release - res.frame0

    first = max(run.frame0, run.start - int(STAND_S * run.rate))
    last = min(run.frame0 + len(run.x) - 1, res.frame0 + len(theta) - 1)
    frames = np.arange(first, last + 1)
    n = len(frames)
    k = res.release - first
    d_idx = frames - res.frame0                      # index into the delivery window
    in_d = (d_idx >= 0) & (d_idx < len(theta))

    # root x: the trunk track around release (unchanged, so the release is),
    # the whole body's position through the run-up, in the same frame as the
    # drawn markers (marker_cloud). The offset between the two is
    # interpolated inside the delivery window and fades out over SPLICE_S
    # outside it, so there is no jump and no constant shift of the run-up.
    x_off = x_trunk[k_rel_d]
    x_run = run.x_lab[frames - run.frame0] - x_off
    x_d = np.full(n, np.nan)
    x_d[in_d] = x_trunk[d_idx[in_d]] - x_off
    ok = np.where(~np.isnan(x_d))[0]
    idx = np.arange(n)
    off = np.interp(idx, ok, (x_d - x_run)[ok])
    gap = np.maximum(ok[0] - idx, idx - ok[-1]).clip(0)
    x = x_run + off * (1 - _smoothstep(gap / (SPLICE_S * run.rate)))

    # settle into the delivery stride between the top of the jump and SETTLE_S before release
    jp = run.jump_peak if run.jump_peak >= 0 else res.release - int(0.47 * run.rate)
    settle = res.release - int(SETTLE_S * run.rate)
    w = _smoothstep((frames - jp) / max(settle - jp, 1))
    z = (1 - w) * root_height_running(run, frames) + w * Z_DELIVERY

    amp = np.clip(run.v_fwd[frames - run.frame0] / FULL_STRIDE_MS, 0, 1)
    phi = stride_phase(run, frames)
    legs = {name: np.empty(n) for name in LEG_JOINT_NAMES}
    for i in range(n):
        cyc = run_cycle_at_phase_deg(phi[i], SWING_AMP_DEG * amp[i], KNEE_LIFT_DEG * amp[i])
        for name in LEG_JOINT_NAMES:
            legs[name][i] = (1 - w[i]) * cyc[name] + w[i] * DELIVERY_STRIDE_DEG[name]

    # arm: measured over the contiguous stretch containing release, carry pose
    # blended into it before, NaN after (nothing is known there)
    meas = ~np.isnan(theta) & ~np.isnan(elbow)
    if not meas[k_rel_d]:
        raise ValueError(f"{res.name}: no arm data at the release frame")
    ka = kb = k_rel_d
    while ka - 1 >= 0 and meas[ka - 1]:
        ka -= 1
    while kb + 1 < len(meas) and meas[kb + 1]:
        kb += 1
    a_abs, b_abs = res.frame0 + ka, res.frame0 + kb
    sh, el = np.full(n, np.nan), np.full(n, np.nan)
    for i, f in enumerate(frames):
        if f < a_abs:
            u = _smoothstep(1 - (a_abs - f) / (ARM_BLEND_S * run.rate))
            sh[i] = CARRY_ARM_DEG[0] + u * (theta[ka] - CARRY_ARM_DEG[0])
            el[i] = CARRY_ARM_DEG[1] + u * (elbow[ka] - CARRY_ARM_DEG[1])
        elif f <= b_abs:
            sh[i], el[i] = theta[f - res.frame0], elbow[f - res.frame0]
    measured = (frames >= a_abs) & (frames <= b_abs)

    qpos = np.full((n, model.nq), np.nan)
    root = model.joint("root_free").qposadr[0]
    qpos[:, root:root + 3] = np.stack([x, np.zeros(n), z], 1)
    qpos[:, root + 3:root + 7] = [1, 0, 0, 0]
    for name in LEG_JOINT_NAMES:
        qpos[:, model.joint(name).qposadr[0]] = np.radians(legs[name])
    qpos[:, model.joint("shoulder").qposadr[0]] = np.radians(sh)
    qpos[:, model.joint("elbow").qposadr[0]] = np.radians(el)
    ball = model.joint("ball_free").qposadr[0]
    qpos[:, ball:ball + 7] = np.nan   # the ball is simulated, not prescribed

    events = {"first_frame": first, "x_offset_m": float(x_trunk[k_rel_d]),
              "release": k, "run_start": run.start - first, "settle": settle - first,
              "arm_measured_from": a_abs - first, "arm_measured_to": b_abs - first,
              "steps": [int(f - first) for f in run.steps],
              "takeoff": run.takeoff - first if run.takeoff >= 0 else -1,
              "jump_peak": run.jump_peak - first if run.jump_peak >= 0 else -1}
    info.update(
        root_x=f"measured: {src} around release, label-free whole-body travel (mocap/runup.py) "
               "earlier in the run-up",
        root_height=f"measured whole-body bob and jump on the model's {Z_STANCE} m stance height; "
                    f"settles to {Z_DELIVERY} m (delivery stride) {SETTLE_S * 1000:.0f} ms before "
                    "release",
        legs="NOT measured: envs/runup.py running cycle phase-locked to the measured steps "
             "(left-foot take-off assumed), amplitude scaled with measured speed, blended into "
             "BowlerEnv DELIVERY_STRIDE_DEG from the top of the jump",
        arm_before_measured=f"carry pose {CARRY_ARM_DEG} blended into the first measured frame "
                            f"({run.ms(a_abs):.0f} ms) over {ARM_BLEND_S * 1000:.0f} ms - not "
                            "measured",
        run_start_ms=run.ms(run.start), steps_detected=len(run.steps))
    return qpos, measured, events, info
