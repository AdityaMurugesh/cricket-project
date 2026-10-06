"""Elbow extension metric: joint angle series in, degrees out. No sim dependency."""
import numpy as np


def find_horizontal_crossings(shoulder_angles_deg):
    """Sample indices where the shoulder crosses a multiple of 180 deg."""
    theta = np.radians(np.asarray(shoulder_angles_deg, dtype=np.float64))
    s = np.sin(theta)
    crossings = []
    for i in range(len(s) - 1):
        if s[i] == 0.0:
            crossings.append(i)
        elif s[i] * s[i + 1] < 0.0:
            crossings.append(i if abs(s[i]) < abs(s[i + 1]) else i + 1)
    if s[-1] == 0.0 and (not crossings or crossings[-1] != len(s) - 1):
        crossings.append(len(s) - 1)
    return crossings


def elbow_extension_deg(shoulder_angles_deg, elbow_angles_deg, release_idx,
                         horizontal_selector="last_before_release", mode="endpoint"):
    """Extension between arm-horizontal and release, or None if no crossing."""
    elbow = np.asarray(elbow_angles_deg, dtype=np.float64)
    crossings = [c for c in find_horizontal_crossings(shoulder_angles_deg) if c <= release_idx]
    if not crossings:
        return None

    if horizontal_selector == "first":
        h_idx = crossings[0]
    elif horizontal_selector == "last_before_release":
        h_idx = crossings[-1]
    else:
        raise ValueError(f"unknown horizontal_selector: {horizontal_selector!r}")

    release_angle = elbow[release_idx]
    if mode == "endpoint":
        return float(elbow[h_idx] - release_angle)
    elif mode == "max":
        return float(np.max(elbow[h_idx:release_idx + 1]) - release_angle)
    else:
        raise ValueError(f"unknown mode: {mode!r}")
