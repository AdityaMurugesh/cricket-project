"""Gap filling, zero-lag low-pass filtering and differentiation, NaN-aware.

Filtering choice: 4th-order (2nd-order run forwards and backwards)
Butterworth low-pass at 15 Hz. The marker data are sampled at only 100 Hz,
so the Nyquist limit is 50 Hz; 15 Hz is within the 10-20 Hz band used for
marker trajectories in cricket bowling studies, removes the frame-to-frame
reconstruction jitter that differentiation would otherwise amplify, and
leaves the ~2-5 Hz content of the arm circle intact. It does round off the
sharpest peak of the hand-speed curve slightly; peak speeds are reported
from the filtered signal, as is standard, and the raw value is kept next to
it so the size of that effect is visible.
"""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt

CUTOFF_HZ = 15.0
ORDER = 2  # per pass; filtfilt doubles it


def runs(mask):
    """[(start, end_inclusive)] of True runs in a boolean array."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(int))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0] - 1
    return list(zip(starts, ends))


def fill_gaps(x, max_gap):
    """Cubic-spline fill of interior NaN gaps no longer than max_gap frames.

    x: (n, k). Longer gaps stay NaN. Returns (filled, filled_mask)."""
    x = np.array(x, float)
    n = x.shape[0]
    ok = ~np.isnan(x[:, 0])
    filled = np.zeros(n, bool)
    if ok.sum() < 4:
        return x, filled
    t = np.arange(n)
    for s, e in runs(~ok):
        if s == 0 or e == n - 1 or e - s + 1 > max_gap:
            continue
        # fit on up to 10 good samples either side
        left = t[:s][ok[:s]][-10:]
        right = t[e + 1:][ok[e + 1:]][:10]
        if len(left) < 2 or len(right) < 2:
            continue
        idx = np.concatenate([left, right])
        cs = CubicSpline(idx, x[idx], axis=0)
        x[s:e + 1] = cs(t[s:e + 1])
        filled[s:e + 1] = True
    return x, filled


def lowpass(x, rate, cutoff=CUTOFF_HZ):
    """Zero-lag Butterworth applied separately to each contiguous non-NaN run.

    Runs shorter than 12 samples are left unfiltered (too short to filter
    without the edge transient dominating)."""
    x = np.array(x, float)
    b, a = butter(ORDER, cutoff / (rate / 2.0))
    ok = ~np.isnan(x[:, 0])
    for s, e in runs(ok):
        seg = x[s:e + 1]
        if len(seg) < 12:
            continue
        x[s:e + 1] = filtfilt(b, a, seg, axis=0, padtype="odd", padlen=min(9, len(seg) - 1))
    return x


def velocity(x, rate):
    """Central-difference velocity, NaN where a neighbour is missing."""
    x = np.asarray(x, float)
    v = np.full_like(x, np.nan)
    v[1:-1] = (x[2:] - x[:-2]) * rate / 2.0
    return v
