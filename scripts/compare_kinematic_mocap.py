"""The kinematic bowler's arm vs the captured bowler's (T3a), around release.

Three sources, never mixed up:
  measured   the raw marker series (data/mocap/series_T3a.npz, scripts/mocap_process.py):
             hand-marker speed, and the elbow angle from the arm markers (it carries a
             ~50 deg marker offset, so only its change is used, as for the 5.3 deg extension)
  fitted     the skeleton fitted to those markers (ref_T3a.npz): what the kinematic
             body replays, and the human's arm in the simulator's own angle convention
  simulated  the learned arm (KinematicBowlEnv, deterministic, start 1.0 s before release)
The body below the arm is the fitted capture in every case; only the arm is learned.

    python scripts/compare_kinematic_mocap.py --out results/kinematic/vs_mocap
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import PPO

from envs.humanoid_bowler import MOCAP_DIR
from envs.kinematic_bowl_env import KinematicBowlEnv
from envs.legality import elbow_extension_deg
from scripts.kin_equifinality import Recorder

ROOT = Path(__file__).resolve().parent.parent
POLICIES = [("kin_bowl_torque_a250_T3a_0", 250, "a250 s0"),
            ("kin_bowl_torque_a270_T3a_0", 270, "a270 s0"),
            ("kin_bowl_torque_a290_T3a_1", 290, "a290 s1")]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]          # categorical slots 1-3 (validated)
INK, INK2, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e4e3de", "#fcfcfb"


def horizontal_to_release_ms(arm, rel_idx, dt):
    """Time from the last arm-horizontal (a multiple of 180 deg) to release."""
    s = np.sin(np.radians(arm[: rel_idx + 1]))
    cross = [i for i in range(len(s) - 1) if s[i] * s[i + 1] < 0 or s[i] == 0]
    return (rel_idx - cross[-1]) * dt * 1000 if cross else None


def simulate(run, angle, before_s=1.0):
    env = KinematicBowlEnv(rsi=False, release_angle_deg=angle, arm_control="torque")
    model = PPO.load(ROOT / "checkpoints" / f"{run}.zip", device="cpu")
    rec = Recorder(env)
    obs, _ = env.reset(seed=0, options={"before_s": before_s})
    rec.rows.clear()
    while True:
        obs, _, te, tr, info = env.step(model.predict(obs, deterministic=True)[0])
        if te or tr:
            break
    h = env.m.opt.timestep
    t = env._start_t + np.arange(len(env._arm_hist)) * h - env.t_ref_release
    speed = np.r_[np.nan, [np.linalg.norm(v) * 3.6 for _, v, _ in rec.rows]]
    arm, elbow = np.array(env._arm_hist), np.array(env._elbow_hist)
    o, rel = info["outcome"], len(arm) - 1
    return {"t": t, "arm": arm, "elbow": elbow, "speed": speed,
            "row": {"source": "simulated", "who": run, "speed_kmh": o["speed_m_s"] * 3.6,
                    "release_height_m": o["release_height_m"], "release_vs_capture_s": float(t[rel]),
                    "arm_rate_rad_s": float(np.radians(arm[rel] - arm[rel - 5]) / (5 * h)),
                    "arm_angle_at_release_deg": float(arm[rel] % 360),
                    "horizontal_to_release_ms": horizontal_to_release_ms(arm, rel, h),
                    "elbow_extension_deg": o["extension_deg"], "land_x": o["land_x"], "good": o["good"]}}


def fitted(env):
    """The fitted skeleton's arm, in the simulator's convention, per mocap frame."""
    ref, d = env.ref, mujoco.MjData(env.m)
    grip, elbow = [], []
    for q in ref["qpos"]:
        d.qpos[:] = q
        mujoco.mj_kinematics(env.m, d)
        R = d.xmat[env._hand].reshape(3, 3)
        grip.append(d.xpos[env._hand] + R @ np.array([0.035, -0.09, 0.0]))
        elbow.append(np.degrees(q[env._elbow_q]))
    rate, ri = ref["rate"], ref["release_index"]
    t = (np.arange(len(grip)) - ri) / rate
    speed = np.r_[np.nan, np.linalg.norm(np.diff(grip, axis=0), axis=1) * rate * 3.6]
    arm, elbow = env._ref_arm, np.array(elbow)
    return {"t": t, "arm": arm, "elbow": elbow, "speed": speed,
            "row": {"source": "fitted skeleton", "who": "T3a", "speed_kmh": float(speed[ri]),
                    "release_height_m": float(grip[ri][2]), "release_vs_capture_s": 0.0,
                    "arm_rate_rad_s": float(np.radians(arm[ri] - arm[ri - 1]) * rate),
                    "arm_angle_at_release_deg": float(arm[ri] % 360),
                    "horizontal_to_release_ms": horizontal_to_release_ms(arm, ri, 1 / rate),
                    "elbow_extension_deg": elbow_extension_deg(arm, elbow, ri)}}


def measured():
    s = np.load(MOCAP_DIR / "series_T3a.npz")
    t = s["t"] - s["release_frame"] / s["rate"]
    return {"t": t, "speed": s["distal_speed"] * 3.6, "elbow": s["elbow_flexion"],     # stored in m/s
            "row": {"source": "measured markers", "who": "T3a", "speed_kmh": 68.5,
                    "release_height_m": 1.95, "release_vs_capture_s": 0.0, "horizontal_to_release_ms": 80.0,
                    "elbow_extension_deg": 5.3, "note": "hand marker on the back of the hand: a lower bound "
                                                        "on ball speed (data/mocap/REPORT.md)"}}


def style(ax, ylabel):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=10)
    ax.set_ylabel(ylabel, color=INK, fontsize=11)
    ax.axvline(0, color=INK2, lw=1, ls=":")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT / "results" / "kinematic" / "vs_mocap"))
    a = p.parse_args()
    env = KinematicBowlEnv(rsi=False)
    fit, mea = fitted(env), measured()
    sims = [simulate(run, ang) for run, ang, _ in POLICIES]
    lo, hi = -0.45, 0.05

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=SURFACE)
    # (a) arm angle: fitted human vs simulated (the markers' own upper-arm angle uses another convention)
    ax = axes[0]
    style(ax, "bowling-arm angle (deg)\n180 = back, 270 = up")
    m = (fit["t"] >= lo) & (fit["t"] <= hi)
    ax.plot(fit["t"][m], fit["arm"][m], color=INK, lw=2.5)
    ax.text(fit["t"][m][-1] + 0.005, fit["arm"][m][-1], "human (fit)", color=INK, fontsize=10, va="center")
    # label offsets (s, deg) chosen so no label sits on a line
    label_at = {"a250 s0": (0.012, -14, "left"), "a270 s0": (-0.015, 0, "right"), "a290 s1": (-0.015, 18, "right")}
    for s, c, (_, ang, lab) in zip(sims, COLORS, POLICIES):
        m = (s["t"] >= lo) & (s["t"] <= hi)
        ax.plot(s["t"][m], s["arm"][m], color=c, lw=2)
        ax.plot(s["t"][-1], s["arm"][-1], "o", ms=8, color=c, mec=SURFACE, mew=2)
        dx, dy, ha = label_at[lab]
        ax.text(s["t"][-1] + dx, s["arm"][-1] + dy, lab, color=INK, fontsize=10, ha=ha, va="center")
    ax.set_title("Arm circle (dot = release)", color=INK, fontsize=12, loc="left")

    # (b) hand speed: measured marker, fitted grip, simulated grip
    ax = axes[1]
    style(ax, "hand speed (km/h)")
    m = (mea["t"] >= lo) & (mea["t"] <= hi)
    ax.plot(mea["t"][m], mea["speed"][m], "s", ms=6, color=INK, mfc=SURFACE, mew=1.5, label="human, hand marker")
    m = (fit["t"] >= lo) & (fit["t"] <= hi)
    ax.plot(fit["t"][m], fit["speed"][m], color=INK, lw=2.5, label="human, fitted skeleton")
    for s, c, (_, _, lab) in zip(sims, COLORS, POLICIES):
        m = (s["t"] >= lo) & (s["t"] <= hi)
        ax.plot(s["t"][m], s["speed"][m], color=c, lw=2, label=f"learned arm {lab}")
        ax.plot(s["t"][-1], s["speed"][-1], "o", ms=8, color=c, mec=SURFACE, mew=2)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    ax.set_title("Hand speed into release", color=INK, fontsize=12, loc="left")

    # (c) elbow: change in flexion relative to release (markers carry a constant offset)
    ax = axes[2]
    style(ax, "elbow flexion minus flexion at release (deg)")
    rm = int(np.argmin(np.abs(mea["t"])))
    m = (mea["t"] >= lo) & (mea["t"] <= 0.0)
    ax.plot(mea["t"][m], mea["elbow"][m] - mea["elbow"][rm], "s", ms=6, color=INK, mfc=SURFACE, mew=1.5)
    ax.text(mea["t"][m][0], mea["elbow"][m][0] - mea["elbow"][rm] + 1.2, "human, markers", color=INK, fontsize=10)
    for s, c, (_, _, lab) in zip(sims, COLORS, POLICIES):
        m = (s["t"] >= lo)
        ax.plot(s["t"][m], s["elbow"][m] - s["elbow"][-1], color=c, lw=2)
    ax.axhspan(15, 30, color=GRID, zorder=0)
    ax.text(lo + 0.01, 15.8, "over 15 deg of straightening = illegal", color=INK2, fontsize=9)
    ax.set_ylim(-8, 30)
    ax.set_title("Elbow straightening into release", color=INK, fontsize=12, loc="left")
    for ax in axes:
        ax.set_xlim(lo, hi + 0.04)
        ax.set_xlabel("time from the captured release (s)", color=INK, fontsize=11)
    fig.tight_layout()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=160, facecolor=SURFACE)
    rows = [mea["row"], fit["row"]] + [s["row"] for s in sims]
    json.dump(rows, open(out.with_suffix(".json"), "w"), indent=1,
              default=lambda x: bool(x) if isinstance(x, np.bool_) else float(x))
    for r in rows:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "note"})


if __name__ == "__main__":
    main()
