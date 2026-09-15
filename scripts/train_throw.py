"""PPO training loop for ThrowEnvGym.

Logs release speed, landing position, and elbow extension for every
episode from the start (see workflow-constraints memory) -- this becomes
the results section later, so it isn't deferred as a later addition.

Reward stays exactly ThrowEnv._reward() (landing-zone only, see
throw_env.py) -- v1 scope is pure task-reward, no legality/style term
folded into the reward itself yet. Elbow extension is recorded for
analysis, not currently penalized.
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.throw_env_gym import ThrowEnvGym

ROOT = Path(__file__).resolve().parent.parent


class EpisodeLogger(BaseCallback):
    """Writes one CSV row per finished episode: release speed, landing
    position, elbow extension, reward, episode length."""

    def __init__(self, csv_path, verbose=0):
        super().__init__(verbose)
        self.csv_path = Path(csv_path)
        self._file = None
        self._writer = None
        self._episode_count = 0

    def _on_training_start(self):
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.csv_path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow([
            "episode", "timesteps", "reward", "release_speed_m_s",
            "landing_x", "landing_y", "elbow_extension_deg", "timeout",
        ])

    def _on_step(self):
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if not done:
                continue
            self._episode_count += 1
            landing = info.get("landing_pos")
            self._writer.writerow([
                self._episode_count,
                self.num_timesteps,
                info.get("episode_reward"),
                info.get("release_speed"),
                landing[0] if landing else None,
                landing[1] if landing else None,
                info.get("elbow_extension_deg"),
                info.get("timeout"),
            ])
            self._file.flush()
        return True

    def _on_training_end(self):
        if self._file is not None:
            self._file.close()


def make_env():
    return ThrowEnvGym()


def run(total_timesteps, n_envs, log_dir, model_path, seed, ent_coef):
    vec_env = DummyVecEnv([make_env for _ in range(n_envs)])
    model = PPO("MlpPolicy", vec_env, verbose=1, seed=seed, ent_coef=ent_coef)

    csv_path = Path(log_dir) / "episodes.csv"
    callback = EpisodeLogger(csv_path)
    model.learn(total_timesteps=total_timesteps, callback=callback)

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    print(f"saved model to {model_path}")
    print(f"episode log at {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--log-dir", default=str(ROOT / "logs" / "throw_ppo"))
    parser.add_argument("--model-path", default=str(ROOT / "checkpoints" / "throw_ppo"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ent-coef", type=float, default=0.01,
                         help="PPO entropy bonus coefficient. SB3 default is 0.0, which let run2's "
                              "policy collapse onto a low-effort near-zero-release strategy instead "
                              "of continuing to explore toward the target zone -- see the commit that "
                              "added this flag.")
    args = parser.parse_args()
    run(args.timesteps, args.n_envs, args.log_dir, args.model_path, args.seed, args.ent_coef)
