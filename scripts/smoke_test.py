"""Headless sanity check: confirms mujoco + throw_env work in the current
Python environment. Used to validate the HPC job-submission pipeline
(module load, venv activation, PBS scheduling) independently of whether
the actual RL training code works yet.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from envs.throw_env import ThrowEnv

env = ThrowEnv()
obs = env.reset()

released_at = None
for i in range(env.max_steps):
    action = [1.0, 0.3, 1.0 if i > 50 else 0.0]
    obs, reward, done, info = env.step(action)
    if info["released"] and released_at is None:
        released_at = i
    if done:
        print("SMOKE TEST RESULT")
        print("steps:", i)
        print("released_at_step:", released_at)
        print("release_speed_m_s:", info["release_speed"])
        print("landing_pos:", info["landing_pos"])
        print("timeout:", info["timeout"])
        break
else:
    print("SMOKE TEST: never terminated within max_steps")

print("OK: mujoco + throw_env ran headlessly in this environment")
