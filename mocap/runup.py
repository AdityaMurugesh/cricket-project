"""Label-free run-up: whole-body travel, vertical bob, running steps, the jump.

No foot or pelvis marker can be identified through the run-up (most were
left unlabelled by AIM, so they are not in the export), but 10-25 markers
on the body are tracked at any instant. Their motion is summarised per
frame by a robust whole-body velocity: the median over every visible track
of its central-difference velocity, recomputed over the inliers (within
INLIER_MS of that median) so the swinging arms and legs drop out. Velocity,
not position, is averaged because markers join and leave mid-run: a median
of positions jumps whenever the set of visible markers changes, a median of
velocities does not. Integrating it gives forward travel and height change.

The integral drifts slowly (up to ~20 cm over a 6 m run-up), so forward
travel is pinned to where the markers actually are: the same inlier
tracks' median POSITION has no drift but jumps by ~10 cm whenever the
visible set changes, so the slowly varying difference between the two
(Gaussian, DRIFT_SMOOTH_S) is added back to the integral. The result keeps
the integral's step-by-step detail and the markers' level.

Events, all from those two signals:

  run start  last frame before the jump at which forward speed is below
             RUN_START_MS. The bowler starts every run standing still at
             one end of the room; if they are already moving when first
             tracked, the start is flagged as not captured.
  steps      local minima of height between run start and the jump. A
             runner's body is lowest at mid-stance, so each minimum is one
             foot under the body. Which foot is not measured.
  jump       highest point of the height curve JUMP_WINDOW_MS before
             release; take-off = the step (minimum) before it.

Height change is NOT pelvis height: near release the trunk flexes forward
and sideways, which lowers the upper-body markers without the pelvis
dropping as much. It is trustworthy during the upright run and the jump,
not through the delivery stride.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from mocap import signal

HALF = 2                    # frames either side for the per-track velocity
MIN_TRACKS = 3
INLIER_MS = 1.0             # m/s from the median: arms/legs swing faster than this
VEL_CUTOFF_HZ = 10.0        # the bob is ~3 Hz; this removes median-switching jitter
DRIFT_SMOOTH_S = 0.3        # sd of the Gaussian that keeps only the drift of the integral
RUN_START_MS = 0.25         # m/s, 0.9 km/h: the bowler eases into the run
SEARCH_BACK_S = 4.0
POST = 40                   # frames after release, as mocap.delivery.POST
STEP_PROMINENCE_M = 0.015
STEP_MIN_SEP_S = 0.18
JUMP_WINDOW_MS = (-800, -200)
JUMP_MIN_RISE_M = 0.08
KMH = 3.6


@dataclass
class RunupResult:
    name: str
    rate: float
    frame0: int                 # first frame of the series (absolute)
    release: int                # release frame (absolute)
    release_source: str         # "hand-speed peak" or "video estimate"
    direction: int
    x: np.ndarray = None        # forward travel, m, 0 at release
    x_lab: np.ndarray = None    # the same, direction-signed lab x (x = x_lab - x_lab at release)
    h: np.ndarray = None        # height change, m, 0 at run start
    v_fwd: np.ndarray = None    # m/s
    n_tracks: np.ndarray = None
    start: int = -1             # run start (absolute frame); see notes if not captured
    steps: list = field(default_factory=list)   # mid-stance frames (absolute), take-off last
    takeoff: int = -1
    jump_peak: int = -1
    metrics: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def ms(self, f):
        return (f - self.release) / self.rate * 1000

    def i(self, f):
        return f - self.frame0


def body_velocity(tracks, a, b, rate):
    """Robust whole-body motion for frames a..b: (n, 3) velocity, m/s, the
    number of inlier tracks used, and (n, 3) median position of those inlier
    tracks, lab frame. NaN where fewer than MIN_TRACKS tracks are visible."""
    n = b - a + 1
    v = np.full((n, 3), np.nan)
    p = np.full((n, 3), np.nan)
    used = np.zeros(n, int)
    for i, f in enumerate(range(a, b + 1)):
        vis = [t for t in tracks if t.start <= f - HALF and f + HALF <= t.end]
        if len(vis) < MIN_TRACKS:
            continue
        vs = np.array([(t.xyz[f + HALF - t.start] - t.xyz[f - HALF - t.start]) * rate / (2 * HALF)
                       for t in vis])
        med = np.median(vs, axis=0)
        inl = np.linalg.norm(vs - med, axis=1) < INLIER_MS
        if inl.sum() >= MIN_TRACKS:
            v[i] = np.median(vs[inl], axis=0)
            p[i] = np.median([t.xyz[f - t.start] for t, k in zip(vis, inl) if k], axis=0)
            used[i] = inl.sum()
    return v, used, p


def analyse(name, trial, tracks, release, direction, release_source):
    rate = trial.rate
    a = max(HALF, release - int(SEARCH_BACK_S * rate))
    b = min(trial.n - 1 - HALF, release + POST)
    res = RunupResult(name, rate, a, release, release_source, direction)
    v, used, pos = body_velocity(tracks, a, b, rate)
    missing = np.isnan(v[:, 0])
    if missing.all():
        raise ValueError(f"{name}: no frames with {MIN_TRACKS}+ tracks before release")
    fwd_raw = direction * v[:, 0]
    # Before the bowler is first tracked they wait at the edge of the volume,
    # mostly out of view: that stretch is taken as standing still (zero
    # velocity). Later dropouts are interpolated across.
    first_seen = int(np.argmax(~missing))
    v[:first_seen] = 0.0
    k = np.arange(len(v))
    ok = ~np.isnan(v[:, 0])
    for j in range(3):
        v[:, j] = np.interp(k, k[ok], v[ok, j])
    res.metrics["interpolated_frames"] = int(missing[first_seen:].sum())
    v = signal.lowpass(v, rate, VEL_CUTOFF_HZ)
    fwd, up = direction * v[:, 0], v[:, 1]
    x = np.cumsum(fwd) / rate
    drift = direction * pos[:, 0] - x                 # NaN where the body is not tracked
    seen = ~np.isnan(drift)
    drift = gaussian_filter1d(np.interp(k, k[seen], drift[seen]), DRIFT_SMOOTH_S * rate, mode="nearest")
    res.x_lab = x + drift
    res.x = res.x_lab - res.x_lab[res.i(release)]
    res.v_fwd, res.n_tracks = fwd, used
    h = np.cumsum(up) / rate

    k_rel = res.i(release)
    lo, hi = (k_rel + int(m / 1000 * rate) for m in JUMP_WINDOW_MS)
    kp = lo + int(np.argmax(h[lo:hi + 1]))
    jump_ok = lo < kp < hi
    # run start: last near-standstill before the jump (or before release)
    k_end = kp if jump_ok else k_rel - 60
    slow = np.where(fwd[:k_end] < RUN_START_MS)[0]
    if np.nanmean(fwd_raw[first_seen:first_seen + 5]) >= RUN_START_MS:
        ks = first_seen
        res.notes.append(f"bowler already moving at {fwd_raw[first_seen] * KMH:.1f} km/h when "
                         "first tracked: run start not captured, start time and distance are "
                         "lower bounds")
    else:
        ks = int(slow[-1]) if len(slow) else first_seen
    res.start = a + ks
    res.h = h - h[ks]

    if jump_ok:
        mins, _ = find_peaks(-res.h[ks:kp], prominence=STEP_PROMINENCE_M,
                             distance=int(STEP_MIN_SEP_S * rate))
        res.steps = [a + ks + int(m) for m in mins]
    if jump_ok and res.steps:
        res.takeoff = res.steps[-1]
        rise = res.h[kp] - res.h[res.i(res.takeoff)]
        if rise >= JUMP_MIN_RISE_M:
            res.jump_peak = a + kp
        else:
            res.notes.append(f"no clear jump: highest point {rise:.2f} m above the last step")
            res.takeoff = -1
    elif not jump_ok:
        res.notes.append("no clear jump: height peaks at the edge of the "
                         f"{JUMP_WINDOW_MS[0]} to {JUMP_WINDOW_MS[1]} ms window")

    m = res.metrics
    k0 = ks
    m["release_timing"] = release_source
    m["runup_start_ms"] = res.ms(res.start)
    m["runup_distance_m"] = float(-res.x[k0])
    m["runup_peak_speed_kmh"] = float(np.max(fwd[k0:k_rel + 1]) * KMH)
    m["speed_300ms_before_release_kmh"] = float(fwd[k_rel - int(0.3 * rate)] * KMH)
    m["steps_detected"] = len(res.steps)
    if len(res.steps) >= 3:
        m["step_rate_hz"] = float(rate / np.median(np.diff(res.steps)))
    if res.jump_peak >= 0:
        m["takeoff_ms"] = res.ms(res.takeoff)
        m["jump_peak_ms"] = res.ms(res.jump_peak)
        m["jump_rise_m"] = float(res.h[res.i(res.jump_peak)] - res.h[res.i(res.takeoff)])
        m["takeoff_speed_kmh"] = float(fwd[res.i(res.takeoff)] * KMH)
    return res
