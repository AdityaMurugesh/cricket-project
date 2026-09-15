"""PPO training loop for ThrowEnvGym.

Logs release speed, landing position, and elbow extension for every
episode from the start (see workflow-constraints memory) -- this becomes
the results section later, so it isn't deferred as a later addition.

Reward is exactly ThrowEnv._reward(), which now gates the accuracy/speed
bonus on ICC legality as well as landing accuracy -- both "subject to"
clauses of the research question, so this is still pure task-reward and
v1's "no imitation/style term" scope is unchanged. Elbow extension is
still logged per episode either way, since the legal/illegal split is
itself a result.
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


def make_env(swing_weight):
    def _init():
        return ThrowEnvGym(swing_weight=swing_weight)
    return _init


def run(total_timesteps, n_envs, log_dir, model_path, seed, ent_coef, resume_from,
        swing_weight):
    vec_env = DummyVecEnv([make_env(swing_weight) for _ in range(n_envs)])
    if resume_from:
        # warm-start from an existing checkpoint instead of a fresh random
        # policy -- reset_num_timesteps=False keeps model.num_timesteps
        # (and its learning-rate/clip-range schedules) continuing on from
        # wherever the checkpoint left off, rather than restarting at 0.
        model = PPO.load(resume_from, env=vec_env)
        model.ent_coef = ent_coef
        print(f"resumed from {resume_from} at num_timesteps={model.num_timesteps}")
    else:
        model = PPO("MlpPolicy", vec_env, verbose=1, seed=seed, ent_coef=ent_coef)

    csv_path = Path(log_dir) / "episodes.csv"
    callback = EpisodeLogger(csv_path)
    model.learn(total_timesteps=total_timesteps, callback=callback,
                reset_num_timesteps=(resume_from is None))

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
    parser.add_argument("--resume-from", default=None,
                         help="path to an existing .zip checkpoint to warm-start from instead of a "
                              "fresh random policy -- --timesteps is additional steps beyond wherever "
                              "the checkpoint left off, not a new total. NOTE: do not use this to "
                              "introduce a changed reward/shaping. A policy that already converged "
                              "under the old reward keeps its collapsed exploration and stays in the "
                              "old local optimum -- that is what made the first legality run go from "
                              "16 horizontal crossings to zero after resuming.")
    parser.add_argument("--swing-weight", type=float, default=5.0,
                         help="scale of the shaping term rewarding progress of the shoulder around "
                              "the swing arc toward the release orientation. 0 disables it, leaving "
                              "the landing-distance potential alone. See ThrowEnvGym._potential().")
    args = parser.parse_args()
    run(args.timesteps, args.n_envs, args.log_dir, args.model_path, args.seed, args.ent_coef,
        args.resume_from, args.swing_weight)
