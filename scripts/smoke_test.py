"""Headless check that mujoco + throw_env import and run (for HPC smoke tests)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from envs.throw_env import ThrowEnv

env = ThrowEnv()
obs = env.reset()

# release action = target shoulder angle, so just ask for 265 deg
release_action = env.release_action_for_angle(265.0)

released_at = None
for i in range(env.max_steps):
    action = [1.0, 0.3, release_action]
    obs, reward, done, info = env.step(action)
    if info["released"] and released_at is None:
        released_at = i
    if done:
        print("SMOKE TEST RESULT")
        print("steps:", i)
        print("released_at_step:", released_at)
        print("release_speed_m_s:", info["release_speed"])
        print("shoulder_at_release_deg:", info["shoulder_at_release_deg"])
        print("elbow_at_release_deg:", info["elbow_at_release_deg"])
        print("release_height_m:", info["release_height_m"])
        print("landing_pos:", info["landing_pos"])
        print("timeout:", info["timeout"])
        break
else:
    print("SMOKE TEST: never terminated within max_steps")

print("OK: mujoco + throw_env ran headlessly in this environment")
