"""Per-delivery analysis: events, bowling-arm kinematics, label-free metrics.

Lab frame of these files: x along the lab's long axis (the run-up runs
along +x or -x), y vertical (up), z lateral. Each delivery is re-expressed
in a delivery frame: X = horizontal direction of travel, Y = up, Z = X x Y
(the bowler's right when facing down the pitch).

Every metric is reported only when the markers it needs were identified
with evidence (see identify.py); otherwise it is NaN and the reason is
recorded. Nothing is filled in from assumptions.
"""
from dataclasses import dataclass, field

import numpy as np

from envs.legality import elbow_extension_deg
from mocap import identify as idf
from mocap import signal

# (trial number, approximate release frame, end of the room bowled towards)
# found from the hand-speed peak where the hand was tracked, and from the
# synchronised 25 fps Miqus video otherwise. Every trial except 1 holds two
# deliveries: one towards each end of the room.
DELIVERIES = [
    ("T1a", 1, 492), ("T2a", 2, 450), ("T2b", 2, 981), ("T3a", 3, 418),
    ("T3b", 3, 1040), ("T4a", 4, 405), ("T4b", 4, 1073), ("T5a", 5, 850),
    ("T5b", 5, 1496), ("T6a", 6, 467), ("T6b", 6, 1065),
]

PRE, POST = 60, 40          # analysis window around release, frames
MAX_GAP_ARM = 5             # frames; arm moves too fast to bridge longer gaps
KMH = 3.6
# In the four deliveries where the hand is tracked through release it peaks
# at 1.78-1.95 m (arm near vertical). A "peak" below 1.7 m means the marker
# was lost on the way up, so it is not a release.
RELEASE_MIN_HEIGHT_M = 1.7
# Timing windows used to tell front- from back-foot contact when only one
# plant is tracked: FFC precedes release by ~60-150 ms in bowling studies,
# BFC by ~200-400 ms.
FFC_WINDOW_MS = (-160, 20)
BFC_WINDOW_MS = (-450, -150)


@dataclass
class DeliveryResult:
    name: str
    trial: int
    rate: float
    frame0: int                                   # first frame of the window
    direction: int = 0                            # +1 bowling towards +x, -1 towards -x
    release: int = -1                             # release frame (absolute), -1 if unknown
    release_ok: bool = False
    arm: idf.ArmChain = None
    series: dict = field(default_factory=dict)    # name -> (n,3) or (n,) arrays over the window
    metrics: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    spine: object = None
    release_usable: bool = False

    @property
    def t(self):
        n = len(next(iter(self.series.values()))) if self.series else 0
        return (np.arange(n) + self.frame0) / self.rate


def _series(tracks, ids, a, b):
    """(b-a+1, 3) position of a landmark: one track id, or the mean of a pair
    (only where both members are visible)."""
    by = {t.tid: t for t in tracks}
    n = b - a + 1
    if not ids:
        return np.full((n, 3), np.nan)
    stack = []
    for i in ids:
        x = np.full((n, 3), np.nan)
        t = by[i]
        s, e = max(a, t.start), min(b, t.end)
        if s <= e:
            x[s - a:e - a + 1] = t.xyz[s - t.start:e - t.start + 1]
        stack.append(x)
    return np.mean(stack, axis=0)   # NaN wherever any member is missing


def _clean(x, rate):
    x, _ = signal.fill_gaps(x, MAX_GAP_ARM)
    return signal.lowpass(x, rate)


def _direction(tracks, rate, a, b):
    """Sign of the median x-velocity of everything visible in [a, b]."""
    vx = []
    for t in tracks:
        s, e = max(a, t.start), min(b, t.end)
        if e - s < 4:
            continue
        x = t.xyz[s - t.start:e - t.start + 1, 0]
        vx.append((x[-1] - x[0]) / ((e - s) / rate))
    return (1 if np.median(vx) > 0 else -1) if vx else 0


def body_speed(tracks, rate, a, b, direction):
    """Label-free whole-body forward speed over [a, b], m/s: the median over
    every visible track of its mean forward velocity. Markers on swinging
    limbs scatter either side; the median follows the trunk."""
    v = []
    for t in tracks:
        s, e = max(a, t.start), min(b, t.end)
        if e - s < 8:
            continue
        x = t.xyz[s - t.start:e - t.start + 1, 0]
        v.append(direction * (x[-1] - x[0]) / ((e - s) / rate))
    return (float(np.median(v)), len(v)) if v else (float("nan"), 0)


def foot_contacts(tracks, rate, a, b, direction):
    """Stance phases of low markers: (onset frame, x position, track id).

    A stance is >= 3 frames with the marker below 0.15 m and slower than
    0.8 m/s. Onsets of different markers within 4 frames and 0.3 m of each
    other are the same foot plant and are merged (earliest onset kept)."""
    plants = []
    for t in tracks:
        if t.n < 3:
            continue
        sp = idf.track_speed(t, rate)
        low = (t.xyz[:, 1] < 0.15) & (sp < 0.8)
        for s, e in signal.runs(low):
            if e - s + 1 < 3:
                continue
            f = t.start + s
            if a <= f <= b:
                plants.append((f, float(t.xyz[s, 0]), t.tid, float(t.xyz[s, 2])))
    plants.sort()
    merged = []
    for p in plants:
        if merged and p[0] - merged[-1][0] <= 4 and abs(p[1] - merged[-1][1]) < 0.3:
            continue
        merged.append(p)
    return merged


def analyse(name, trial, tracks, r_guess):
    rate = trial.rate
    a, b = max(0, r_guess - PRE), min(trial.n - 1, r_guess + POST)
    res = DeliveryResult(name, int(name[1]), rate, a)
    res.direction = _direction(tracks, rate, r_guess - 50, r_guess - 20)

    arm = idf.identify_arm(tracks, rate, r_guess)
    res.arm = arm
    hand = _clean(_series(tracks, arm.hand[:1], a, b), rate)
    wrist = _clean(_series(tracks, arm.wrist[:2], a, b), rate)
    if len(arm.wrist) == 2 and np.isnan(wrist[:, 0]).mean() > 0.5:
        # the pair is rarely co-visible: fall back to the better-covered marker
        singles = [_clean(_series(tracks, [i], a, b), rate) for i in arm.wrist]
        wrist = min(singles, key=lambda x: np.isnan(x[:, 0]).sum())
        res.notes.append("wrist centre unavailable; single wrist marker used")
    elbow = _clean(_series(tracks, arm.elbow[:2], a, b), rate)
    if len(arm.elbow) == 2 and np.isnan(elbow[:, 0]).mean() > 0.5:
        singles = [_clean(_series(tracks, [i], a, b), rate) for i in arm.elbow]
        elbow = min(singles, key=lambda x: np.isnan(x[:, 0]).sum())
        res.notes.append("elbow centre unavailable; single elbow marker used")
    prox = _clean(_series(tracks, arm.proximal[:1], a, b), rate)
    res.series.update(hand=hand, wrist=wrist, elbow=elbow, proximal=prox)

    # distal point for speed/release: hand if tracked, else wrist
    distal, distal_name = (hand, "hand") if not np.all(np.isnan(hand[:, 0])) else (wrist, "wrist")
    v = signal.velocity(distal, rate)
    spd = np.linalg.norm(v, axis=1)
    res.series["distal_speed"] = spd
    res.metrics["distal_marker"] = distal_name if not np.all(np.isnan(distal[:, 0])) else "none"

    # release proxy: peak distal speed while the point is above 1.4 m
    lo, hi = r_guess - 40 - a, r_guess + 20 - a
    m = np.zeros(len(spd), bool)
    m[max(lo, 0):hi + 1] = True
    m &= ~np.isnan(spd) & (distal[:, 1] > 1.4)
    if m.any():
        k = int(np.nanargmax(np.where(m, spd, -1)))
        res.release = a + k
        # trustworthy only if the marker is tracked through the peak
        tracked = bool(k >= 5 and k + 3 < len(spd) and not np.isnan(spd[k - 5:k + 4]).any())
        high = bool(distal[k, 1] >= RELEASE_MIN_HEIGHT_M)
        res.release_ok = tracked and high
        res.metrics["release_quality"] = ("good" if res.release_ok else
                                          "lower bound (lost after peak)" if high else
                                          "not captured")
        res.metrics["peak_distal_speed_kmh"] = float(spd[k] * KMH)
        res.metrics["release_height_m"] = float(distal[k, 1])
        raw = signal.velocity(_series(tracks, (arm.hand[:1] if distal_name == "hand" else arm.wrist[:2]), a, b), rate)
        rs = np.linalg.norm(raw, axis=1)
        win_rs = rs[max(k - 2, 0):k + 3]
        res.metrics["peak_distal_speed_raw_kmh"] = (float(np.nanmax(win_rs) * KMH)
                                                    if not np.all(np.isnan(win_rs)) else float("nan"))
        if not high:
            res.notes.append(f"fastest tracked point peaks at {distal[k, 1]:.2f} m, below the "
                             f"{RELEASE_MIN_HEIGHT_M} m an overarm release reaches: the hand was "
                             "not tracked through release, so no release time or speed")
        elif not tracked:
            res.notes.append("distal marker lost within 30 ms of its speed peak; release time "
                             "and speed are lower bounds")
    else:
        res.notes.append("no tracked hand/wrist point above 1.4 m near the expected release")

    # bowling-arm angles in the delivery frame
    usable = res.metrics.get("release_quality") in ("good", "lower bound (lost after peak)")
    res.release_usable = usable
    chain_ok = arm.ok_for_elbow_angle()
    if chain_ok and usable:
        u = elbow - prox
        f_ = wrist - elbow if not np.all(np.isnan(wrist[:, 0])) else hand - elbow
        cosang = np.sum(u * f_, 1) / np.linalg.norm(u, axis=1) / np.linalg.norm(f_, axis=1)
        flex = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
        fwd = res.direction * u[:, 0]
        theta = np.degrees(np.arctan2(-u[:, 1], fwd))          # model convention
        theta = np.where(np.isnan(theta), np.nan, np.mod(theta, 360))
        res.series.update(elbow_flexion=flex, shoulder_theta=theta,
                          upper_arm_elev=np.degrees(np.arcsin(np.clip(u[:, 1] / np.linalg.norm(u, axis=1), -1, 1))))
        _arm_metrics(res, flex, theta, a)
        if not res.release_ok:
            res.notes.append("elbow values use a release frame that is a lower bound (marker lost "
                             "just after its speed peak)")
    elif chain_ok:
        res.notes.append("arm chain complete but release not captured; no elbow-extension value")
    else:
        res.notes.append("bowling-arm chain incomplete (needs wrist/hand + elbow + upper-arm marker): "
                         f"hand={arm.hand} wrist={arm.wrist} elbow={arm.elbow} proximal={arm.proximal}")

    # trunk: column of back markers
    spine = idf.identify_spine(tracks, rate, res.release if res.release >= 0 else r_guess,
                               exclude=set(arm.all_ids()))
    res.spine = spine
    if spine.ids:
        top = _clean(_series(tracks, spine.ids[:1], a, b), rate)
        bot = _clean(_series(tracks, spine.ids[-1:], a, b), rate)
        ax = top - bot
        X = np.array([res.direction, 0, 0]); Y = np.array([0, 1, 0]); Z = np.cross(X, Y)
        lat = np.degrees(np.arctan2(-(ax @ Z), ax @ Y))   # + = towards the non-bowling (left) side
        fwd = np.degrees(np.arctan2(ax @ X, ax @ Y))      # + = leaning forward
        res.series.update(trunk_lateral=lat, trunk_forward=fwd, spine_top=top, spine_bottom=bot)
        if usable and not np.isnan(lat[res.release - a]):
            res.metrics["trunk_lateral_flexion_at_release_deg"] = float(lat[res.release - a])
            res.metrics["trunk_forward_flexion_at_release_deg"] = float(fwd[res.release - a])
    else:
        res.notes.append("trunk angle not measurable: " + "; ".join(spine.evidence))

    # label-free whole-body measures
    vb, nb = body_speed(tracks, rate, r_guess - 50, r_guess - 25, res.direction)
    res.metrics["approach_speed_kmh"] = vb * KMH
    res.metrics["approach_speed_ntracks"] = nb
    plants = foot_contacts(tracks, rate, r_guess - 70, r_guess + 30, res.direction)
    res.metrics["foot_plants"] = [(f, round(x, 3), tid) for f, x, tid, z in plants]
    _contact_metrics(res, plants)
    return res


def _arm_metrics(res, flex, theta, a):
    k = res.release - a
    seg = theta[:k + 1]
    ok = ~np.isnan(seg)
    if not ok.any():
        res.notes.append("upper-arm angle missing before release")
        return
    # unwrap the defined stretch ending at release
    s = k
    while s - 1 >= 0 and ok[s - 1]:
        s -= 1
    if not ok[k]:
        res.notes.append("upper-arm angle missing at release")
        return
    th = np.degrees(np.unwrap(np.radians(theta[s:k + 1])))
    fl = flex[s:k + 1]
    if np.isnan(fl).any():
        res.notes.append("elbow flexion has gaps between arm-horizontal and release")
        return
    res.metrics["shoulder_theta_at_release_deg"] = float(th[-1] % 360)
    res.metrics["elbow_flexion_at_release_deg"] = float(fl[-1])
    for mode in ("endpoint", "max"):
        ext = elbow_extension_deg(th, fl, len(th) - 1, "last_before_release", mode)
        res.metrics[f"elbow_extension_{mode}_deg"] = float("nan") if ext is None else ext
    from envs.legality import find_horizontal_crossings
    cr = [c for c in find_horizontal_crossings(th) if c <= len(th) - 1]
    if cr:
        h = cr[-1]
        res.metrics["arm_horizontal_frame"] = a + s + h
        res.metrics["elbow_flexion_at_horizontal_deg"] = float(fl[h])
        res.metrics["horizontal_to_release_ms"] = float((len(th) - 1 - h) / res.rate * 1000)
    else:
        res.notes.append("no arm-horizontal crossing inside the tracked stretch before release")


def _contact_metrics(res, plants):
    if res.release < 0 or not res.release_usable:
        return
    def rel_ms(p):
        return (p[0] - res.release) / res.rate * 1000
    ffc = [p for p in plants if FFC_WINDOW_MS[0] <= rel_ms(p) <= FFC_WINDOW_MS[1]]
    bfc = [p for p in plants if BFC_WINDOW_MS[0] <= rel_ms(p) < BFC_WINDOW_MS[1]]
    if ffc:
        res.metrics["ffc_frame"] = ffc[-1][0]
        res.metrics["ffc_to_release_ms"] = -rel_ms(ffc[-1])
    if bfc:
        res.metrics["bfc_frame"] = bfc[-1][0]
        res.metrics["bfc_to_release_ms"] = -rel_ms(bfc[-1])
    if ffc and bfc and abs(ffc[-1][1] - bfc[-1][1]) > 0.4:
        res.metrics["bfc_to_ffc_ms"] = (ffc[-1][0] - bfc[-1][0]) / res.rate * 1000
        res.metrics["stride_length_m"] = abs(ffc[-1][1] - bfc[-1][1])
    if not ffc and not bfc:
        res.notes.append("no foot plant tracked in the FFC/BFC timing windows")
