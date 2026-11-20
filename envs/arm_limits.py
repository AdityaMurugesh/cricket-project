"""Actuator and joint limits for the bowling arm (supervisor, 20 Nov: "two sets of actuator and joint limits: one
human-like, one from a real humanoid robot").

The arm has 7 motors, in this order: shoulder about its local x (flexion/extension), z (abduction/adduction) and y
(internal/external rotation), elbow flexion, forearm pronation/supination, wrist flexion, wrist deviation
(envs/humanoid_bowler.py BALL_GEAR / HINGE_GEAR). Until now every result used the skeleton model's strengths:
250 N m at the shoulder, elbow and forearm and 50 at the wrist, with linear damping that makes each motor a
DC-motor-like curve with a 40 rad/s no-load speed. That is several times human strength (the forearm 25x).

Per profile, per motor: peak torque (N m, the same both ways: the mean of the two directions where they differ) and
the speed at which the motor can no longer push (rad/s), with the torque-speed curve
  hill  muscle (Hill's force-velocity relation, as KinematicBowlEnv2 actuator="hill", k = 0.25)
  dc    electric motor: torque falls linearly to zero at the no-load speed
Braking (the joint moving against the command) always gets the full peak torque.
Joint ranges (degrees) replace the model's where given; "shoulder_cone" limits the shoulder ball joint's rotation
from its rest pose.

human  adult male. Holzbaur et al. 2007 (J Biomech 40:2442, Table 1, the 5 men): shoulder abduction 74.4 /
       adduction 93.7, elbow flexion 79.5 / extension 60.5, wrist flexion 25.6 / extension 14.0 N m. Shoulder
       flexion 77 N m at 90 deg elevation (14 men, J Funct Morphol Kinesiol 11(1):20, 2026). Shoulder rotation
       (45), forearm rotation (10) and wrist deviation (12) are typical literature values, not from one study.
       Hill curve with 40 rad/s: muscle shortens at most ~10 fibre lengths/s, with ~0.1 m fibres on ~0.02-0.03 m
       moment arms. Ranges: the elbow's -15 deg hyperextension stop is kept (the v3 model's); forearm +/-85 deg;
       wrist flexion/extension +/-75 deg; the shoulder cone is 170 deg, just past the 168 deg this bowler's own
       captured shoulder reached (T3a).
g1     Unitree G1 (29-DoF version, 7-joint arm), its URDF (unitree_ros g1_29dof.urdf) <limit effort velocity>:
       shoulder pitch/roll/yaw, elbow, wrist roll 25 N m and 37 rad/s; wrist pitch and yaw 5 N m and 22 rad/s.
       Its joint ranges are all at least as wide as the human ones (shoulder pitch -177..153 deg, roll -129..91,
       yaw +/-150, elbow 180 deg of travel, wrist roll +/-113, pitch/yaw +/-92.5), so the human ranges bind.
h1     Unitree H1 (h1.urdf): shoulder pitch and roll 40 N m at 9 rad/s, shoulder yaw and elbow 18 N m at
       20 rad/s. Its arm has no forearm or wrist joints: those three are given the G1's.
model  the skeleton's own (as every result before 21 Nov), for reference.
"""
import numpy as np

MOTORS = ("shoulder_x", "shoulder_z", "shoulder_y", "elbow", "pro_sup", "wrist_flex", "wrist_dev")

PROFILES = {
    "human": {"torque": (77.0, 84.0, 45.0, 70.0, 10.0, 19.8, 12.0),
              "omega_max": (40.0,) * 7, "curve": "hill",
              "range_deg": {"elbow_flex_r": (-15.0, 150.0), "pro_sup_r": (-85.0, 85.0),
                            "wrist_flex_r": (-75.0, 75.0), "wrist_dev_r": (-30.0, 40.0)},
              "shoulder_cone_deg": 170.0},
    "g1": {"torque": (25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0),
           "omega_max": (37.0, 37.0, 37.0, 37.0, 37.0, 22.0, 22.0), "curve": "dc",
           "range_deg": {"elbow_flex_r": (-15.0, 150.0), "pro_sup_r": (-85.0, 85.0),
                         "wrist_flex_r": (-75.0, 75.0), "wrist_dev_r": (-30.0, 40.0)},
           "shoulder_cone_deg": 170.0},
    "h1": {"torque": (40.0, 40.0, 18.0, 18.0, 25.0, 5.0, 5.0),
           "omega_max": (9.0, 9.0, 20.0, 20.0, 37.0, 22.0, 22.0), "curve": "dc",
           "range_deg": {"elbow_flex_r": (-15.0, 150.0), "pro_sup_r": (-85.0, 85.0),
                         "wrist_flex_r": (-75.0, 75.0), "wrist_dev_r": (-30.0, 40.0)},
           "shoulder_cone_deg": 170.0},
    "model": {"torque": (250.0, 250.0, 250.0, 250.0, 250.0, 50.0, 50.0),
              "omega_max": (40.0,) * 7, "curve": "damping"},
}
HILL_K = 0.25


def available(profile, w, u):
    """Fraction of the commanded torque the motor can give at joint speed w (rad/s, along its axis)."""
    p = PROFILES[profile]
    s = np.abs(w) / np.asarray(p["omega_max"])
    conc = np.sign(w) == np.sign(u)            # pushing the way the joint already moves
    if p["curve"] == "hill":
        f = np.clip(1.0 - s, 0.0, 1.0) / (1.0 + s / HILL_K)
    elif p["curve"] == "dc":
        f = np.clip(1.0 - s, 0.0, 1.0)
    else:
        return np.ones_like(u)
    return np.where(conc, f, 1.0)
