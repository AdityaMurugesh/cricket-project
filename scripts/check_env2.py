"""Checks of envs/kinematic_bowl_env2.py against KinematicBowlEnv and kin_equifinality (single process, ~1 min).

1. all options off: the same outcomes as KinematicBowlEnv(arm_control="torque"), PPO policy, 5 starts
2. tolerance reward: the env's p_jit at release equals kin_equifinality's p_jitter_12ms
3. Hill actuator and torque scale: random actions stay finite; outcomes differ from the damping model
4. learned release with the release command at the fixed-angle crossing: a sensible delivery

    cricket/Scripts/python.exe scripts/check_env2.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from stable_baselines3 import PPO

from envs.kinematic_bowl_env import KinematicBowlEnv
from envs.kinematic_bowl_env2 import KinematicBowlEnv2
from scripts.kin_equifinality import delivery

RUN, ANG = "kin_bowl_torque_b4096_t12_a270_T3a_0", 270.0


def play(env, model, before, extra=None):
    obs, _ = env.reset(options={"before_s": before})
    while True:
        a, _ = model.predict(obs, deterministic=True)
        if extra is not None:
            a = np.append(a, extra(env))
        obs, r, te, tr, info = env.step(a)
        if te or tr:
            return r, info["outcome"]


model = PPO.load(ROOT / "checkpoints" / f"{RUN}.zip", device="cpu")
old = KinematicBowlEnv(rsi=False, release_angle_deg=ANG, arm_control="torque")
new = KinematicBowlEnv2(rsi=False, release_angle_deg=ANG)
print("1. options off vs KinematicBowlEnv")
for before in np.linspace(0.8, 1.2, 5):
    r0, o0 = play(old, model, before)
    r1, o1 = play(new, model, before)
    same = abs(r0 - r1) < 1e-9 and o0["good"] == o1["good"] and abs(o0["land_x"] - o1["land_x"]) < 1e-9
    print(f"   before {before:.2f}: old r {r0:8.3f} land {o0['land_x']:.4f} | new r {r1:8.3f} land {o1['land_x']:.4f}  {'SAME' if same else 'DIFFERENT'}")
    assert same

print("2. tolerance reward vs kin_equifinality p_jitter_12ms")
tol = KinematicBowlEnv2(rsi=False, release_angle_deg=ANG, tolerance_reward_sd_ms=12.0)
for before in (0.9, 1.0, 1.1):
    r, o = play(tol, model, before)
    d, _ = delivery(old, model, before, 0)
    print(f"   before {before:.2f}: env p_jit {o['p_jit']:.4f}, expected reward {r:8.3f} | scorer {d['p_jitter_12ms']:.4f}")
    assert abs(o["p_jit"] - d["p_jitter_12ms"]) < 0.02, "curves disagree"

print("3. Hill actuator / torque scale")
rng = np.random.default_rng(0)
for kw in ({"actuator": "hill", "omega_max": 40.0}, {"torque_scale": 0.75}, {"torque_scale": 1.5}):
    e = KinematicBowlEnv2(rsi=False, release_angle_deg=ANG, **kw)
    for ep in range(3):
        obs, _ = e.reset(options={"before_s": 1.0})
        while True:
            obs, r, te, tr, info = e.step(rng.uniform(-1, 1, e.action_space.shape))
            assert np.isfinite(obs).all()
            if te or tr:
                break
    r, o = play(e, model, 1.0)
    print(f"   {kw}: PPO policy -> released {o['released']} {o.get('why', '')} land {o.get('land_x')} "
          f"{(o.get('speed_m_s') or 0) * 3.6:.1f} km/h")

print("4. learned release, commanded when the arm passes the angle")
lr = KinematicBowlEnv2(rsi=False, release_angle_deg=ANG, learned_release=True)
r, o = play(lr, model, 1.0, extra=lambda env: 1.0 if env._arm_hist[-1] >= ANG - 25 else -1.0)
print(f"   released {o['released']} {o.get('why', '')} land {o.get('land_x')} {(o.get('speed_m_s') or 0) * 3.6:.1f} km/h good {o.get('good')}")

print("5. drag + stiff elbow (20 Nov): env tolerance reward vs the scorer with the same options; drag shortens")
from envs.ballistics import CRICKET_DRAG_K  # noqa: E402
v3 = dict(drag_k=CRICKET_DRAG_K, elbow_solref=[0.004, 1.0])
tol3 = KinematicBowlEnv2(rsi=False, release_angle_deg=ANG, tolerance_reward_sd_ms=12.0, **v3)
sc3 = KinematicBowlEnv2(rsi=False, release_angle_deg=ANG, **v3)
for before in (0.9, 1.0, 1.1):
    r, o = play(tol3, model, before)
    d, _ = delivery(sc3, model, before, 0)
    _, o0 = play(new, model, before)
    print(f"   before {before:.2f}: env p_jit {o['p_jit']:.4f} | scorer {d['p_jitter_12ms']:.4f}; land {d['land_x']:.3f} "
          f"(old model {o0['land_x']:.3f})")
    assert abs(o["p_jit"] - d["p_jitter_12ms"]) < 0.02, "curves disagree"
print("all checks done")
