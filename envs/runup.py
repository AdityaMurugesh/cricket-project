"""Scripted run-up and swing trajectories, pure functions of time."""
import numpy as np

LEG_JOINT_NAMES = ["left_hip", "left_knee", "left_ankle", "right_hip", "right_knee", "right_ankle"]

CARRY_ARM_DEG = (100.0, 0.0)   # shoulder, elbow while running in

# swing carries on past release so the ball leaves at peak hand speed
FOLLOW_THROUGH_ARM_DEG = (430.0, 20.0)

DELIVERY_STRIDE_DEG = {
    "left_hip": 25.0,    # front leg
    "left_knee": -10.0,
    "left_ankle": 0.0,
    "right_hip": -25.0,  # back leg
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
    """(root_xyz, leg_angles_deg) at time t; blends into the delivery stride."""
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
    """Smoothstep from carry pose to follow-through, in degrees."""
    alpha = np.clip(t / duration, 0.0, 1.0)
    ease = 3 * alpha ** 2 - 2 * alpha ** 3
    shoulder = CARRY_ARM_DEG[0] + (FOLLOW_THROUGH_ARM_DEG[0] - CARRY_ARM_DEG[0]) * ease
    elbow = CARRY_ARM_DEG[1] + (FOLLOW_THROUGH_ARM_DEG[1] - CARRY_ARM_DEG[1]) * ease
    return shoulder, elbow
