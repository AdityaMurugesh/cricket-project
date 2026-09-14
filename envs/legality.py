"""Pure elbow-extension metric -- joint angle time series in, a number out.

No simulator dependency, so it works the same whether the angles came from
throw_env's 2-joint arm or (later) the full bowler humanoid.

elbow_angle convention (throw_arm.xml / bowler.xml): 0 deg = fully extended
(straight arm), increasing = more flexed. So a positive extension value
means the arm straightened between the horizontal instant and release;
negative/zero means it flexed further or held still -- not a legality
concern, since ICC only restricts extension, not flexion.

shoulder_angle convention: the shoulder hinge rotates about the Y axis
starting from a local +X direction at angle 0, so the upper arm is
horizontal whenever the shoulder angle is a multiple of 180 deg (that's
where sin(theta) == 0), regardless of which way it's pointing.

Two questions about the ICC definition are still unresolved pending expert
reply (see elbow-legality-design-decision memory) and are exposed as
parameters here rather than hardcoded:
  - which horizontal crossing counts as "the" horizontal instant
  - whether the reported value is the endpoint difference or the maximum
    extension observed across the whole horizontal-to-release window
"""
import numpy as np


def find_horizontal_crossings(shoulder_angles_deg):
    """Indices where the upper arm passes through horizontal (shoulder
    angle crosses a multiple of 180 deg). Returns the sample closer to the
    exact crossing on each side, not an interpolated sub-step index."""
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
    """Returns the extension in degrees, or None if no horizontal crossing
    is found before release_idx.

    horizontal_selector:
      "first"               -- first horizontal crossing in the whole series
      "last_before_release" -- horizontal crossing closest to (and before) release
    mode:
      "endpoint" -- elbow_angle[horizontal] - elbow_angle[release]
      "max"      -- max(elbow_angle[horizontal:release+1]) - elbow_angle[release]
    """
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
