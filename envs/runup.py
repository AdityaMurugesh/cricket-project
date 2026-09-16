"""Pure kinematic run-up trajectory generator.

The run-up is scripted, not RL-trained -- see the run-up-method decision
in the project scope. These are plain functions of elapsed time with no
simulator dependency: given t (seconds into the run-up), they return the
root position and leg joint angles to write directly into qpos. The RL
policy only ever controls the bowling arm, during the dynamic phase that
starts once the run-up hands off via BowlerEnv.start_delivery().
"""
import numpy as np

LEG_JOINT_NAMES = ["left_hip", "left_knee", "left_ankle", "right_hip", "right_knee", "right_ankle"]

CARRY_ARM_DEG = (100.0, 0.0)   # shoulder, elbow -- arm held cocked while running in

# Where the swing ENDS, which is not where the ball leaves. The swing used
# to stop at shoulder 290 and the ball was released partway along it, which
# put release at shoulder 212 deg -- arm pointing back and barely above
# shoulder height -- and lobbed the ball. That is a sling, not a bowling
# action, and it is the same failure the RL policy had (see throw_env.py).
#
# Two things have to be true at release and the old profile could not make
# both true at once. The arm must be near vertical (270 deg: geometry is
# 90 down, 180 horizontal backward, 270 straight up, 360 horizontal
# forward, and release velocity is tangential, so 270 sends the ball
# forward from maximum height). And the hand must be at maximum speed --
# but a smoothstep DECELERATES to zero at the end of its range, so
# releasing near the end of the old swing released at the slowest instant
# of the whole delivery.
#
# Carrying the swing through to a follow-through past the front horizontal
# satisfies both. Smoothstep peaks in velocity at the midpoint, and the
# midpoint of 100 -> 430 is 265 deg -- near vertical. So the default
# release_frac of 0.5 now releases near vertical AND at peak hand speed,
# and the arm carries on down and across the body afterwards, which is what
# a real follow-through looks like.
FOLLOW_THROUGH_ARM_DEG = (430.0, 20.0)

DELIVERY_STRIDE_DEG = {
    "left_hip": 25.0,    # front leg planted forward
    "left_knee": -10.0,
    "left_ankle": 0.0,
    "right_hip": -25.0,  # back leg trailing
    "right_knee": -25.0,
    "right_ankle": 10.0,
}


def run_cycle_leg_angles_deg(t, stride_freq, swing_amp=28.0, knee_lift=35.0):
    """Simple sagittal-plane running cycle, both legs, in degrees."""
    phase = 2 * np.pi * stride_freq * t
    left_hip = swing_amp * np.sin(phase)
    right_hip = swing_amp * np.sin(phase + np.pi)
    left_knee = -knee_lift * max(0.0, np.sin(phase)) ** 2
    right_knee = -knee_lift * max(0.0, np.sin(phase + np.pi)) ** 2
    return {
        "left_hip": left_hip, "left_knee": left_knee, "left_ankle": 0.0,
        "right_hip": right_hip, "right_knee": right_knee, "right_ankle": 0.0,
    }


def runup_pose(t, run_duration, blend_duration, run_speed, start_x, pelvis_height,
                bob_amplitude=0.03, stride_freq=1.8):
    """Return (root_xyz, leg_angles_deg) at time t seconds into the run-up.

    Runs at constant forward speed with a simple leg-swing cycle, then
    blends into DELIVERY_STRIDE_DEG over the final blend_duration seconds
    so there's no pop at handoff to dynamics. Holds the final pose for
    t >= run_duration.
    """
    t_run = min(t, run_duration)
    x = start_x + run_speed * t_run
    z = pelvis_height + bob_amplitude * abs(np.sin(2 * np.pi * stride_freq * t_run))
    cycle = run_cycle_leg_angles_deg(t_run, stride_freq)

    blend_start = run_duration - blend_duration
    if t_run >= blend_start:
        alpha = np.clip((t_run - blend_start) / blend_duration, 0.0, 1.0)
        legs = {k: (1 - alpha) * cycle[k] + alpha * DELIVERY_STRIDE_DEG[k] for k in cycle}
    else:
        legs = cycle

    if t >= run_duration:
        z = pelvis_height
        legs = dict(DELIVERY_STRIDE_DEG)

    return (x, 0.0, z), legs


def delivery_arm_angles_deg(t, duration):
    """Shoulder/elbow angles (degrees) during the scripted bowling swing,
    t seconds after the run-up hands off. Eases from CARRY_ARM_DEG all the
    way to FOLLOW_THROUGH_ARM_DEG with a smoothstep, holding the
    follow-through pose once t >= duration.

    The ball leaves partway along this, not at the end -- see
    FOLLOW_THROUGH_ARM_DEG for why the swing has to continue past release.
    """
    alpha = np.clip(t / duration, 0.0, 1.0)
    ease = 3 * alpha ** 2 - 2 * alpha ** 3
    shoulder = CARRY_ARM_DEG[0] + (FOLLOW_THROUGH_ARM_DEG[0] - CARRY_ARM_DEG[0]) * ease
    elbow = CARRY_ARM_DEG[1] + (FOLLOW_THROUGH_ARM_DEG[1] - CARRY_ARM_DEG[1]) * ease
    return shoulder, elbow
