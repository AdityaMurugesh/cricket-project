"""PPO training loop for ThrowEnvGym, logging every episode to a CSV."""
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


class ActionStdLogger(BaseCallback):
    """Log per-dimension action std (SB3 only logs the mean across dims)."""

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        import numpy as np
        std = np.exp(self.model.policy.log_std.detach().cpu().numpy())
        for name, value in zip(["shoulder", "elbow", "release"], std):
            self.logger.record(f"action_std/{name}", float(value))
        if len(std) < 3:
            return  # fixed_angle: no release dimension
        env = self.training_env.envs[0]._env
        span = ((env.release_window_max - env.RELEASE_TARGET_INSET_DEG)
                - env.release_window_min)
        # release std in degrees of shoulder angle
        self.logger.record("action_std/release_deg", float(std[2] * span / 2.0))


class EpisodeLogger(BaseCallback):
    """One CSV row per finished episode."""

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
            "max_shoulder_deg", "shoulder_at_release_deg",
            "elbow_at_release_deg", "release_height_m",
            "release_target_deg",
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
                info.get("max_shoulder_deg"),
                info.get("shoulder_at_release_deg"),
                info.get("elbow_at_release_deg"),
                info.get("release_height_m"),
                info.get("release_target_deg"),
            ])
            self._file.flush()
        return True

    def _on_training_end(self):
        if self._file is not None:
            self._file.close()


def make_env(swing_weight, straight_arm_weight, release_window, max_release_elbow,
             speed_weight, gamma, illegal_penalty, release_mode, release_latch,
             release_angle, actuator_scale):
    def _init():
        # shaping gamma must equal PPO gamma
        return ThrowEnvGym(swing_weight=swing_weight,
                           straight_arm_weight=straight_arm_weight,
                           release_window_deg=tuple(release_window),
                           max_release_elbow_deg=max_release_elbow,
                           speed_weight=speed_weight,
                           illegal_penalty=illegal_penalty,
                           release_mode=release_mode,
                           release_latch=release_latch,
                           release_angle_deg=release_angle,
                           actuator_scale=actuator_scale,
                           shaping_gamma=gamma)
    return _init


def run(total_timesteps, n_envs, log_dir, model_path, seed, ent_coef, resume_from,
        swing_weight, straight_arm_weight, release_window, max_release_elbow,
        speed_weight, gamma, illegal_penalty, release_mode, release_latch,
        release_angle, actuator_scale):
    vec_env = DummyVecEnv([
        make_env(swing_weight, straight_arm_weight, release_window, max_release_elbow,
                 speed_weight, gamma, illegal_penalty, release_mode, release_latch,
                 release_angle, actuator_scale)
        for _ in range(n_envs)])
    if resume_from:
        # warm start; timestep counter continues from the checkpoint
        model = PPO.load(resume_from, env=vec_env)
        model.ent_coef = ent_coef
        print(f"resumed from {resume_from} at num_timesteps={model.num_timesteps}")
    else:
        model = PPO("MlpPolicy", vec_env, verbose=1, seed=seed, ent_coef=ent_coef,
                    gamma=gamma)

    csv_path = Path(log_dir) / "episodes.csv"
    callback = [EpisodeLogger(csv_path), ActionStdLogger()]
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
                         help="PPO entropy bonus coefficient")
    parser.add_argument("--resume-from", default=None,
                         help="checkpoint .zip to warm-start from; --timesteps is then additional")
    parser.add_argument("--speed-weight", type=float, default=0.1,
                         help="weight on release speed, applied only to legal in-zone throws")
    parser.add_argument("--swing-weight", type=float, default=30.0,
                         help="shaping weight on swing progress toward release; 0 disables")
    parser.add_argument("--straight-arm-weight", type=float, default=6.0,
                         help="shaping weight on straightening the arm near release; 0 disables")
    parser.add_argument("--release-window", type=float, nargs=2, default=[230.0, 310.0],
                         metavar=("MIN_DEG", "MAX_DEG"),
                         help="shoulder-angle window (deg) in which release is allowed; 270 is up")
    parser.add_argument("--max-release-elbow", type=float, default=40.0,
                         help="max elbow flexion (deg) allowed at release; not the ICC metric")
    parser.add_argument("--gamma", type=float, default=0.9999,
                         help="PPO discount factor, also used by the shaping; keep high")
    parser.add_argument("--illegal-penalty", type=float, default=2.0,
                         help="flat penalty added to an illegal delivery")
    parser.add_argument("--release-mode", default="target_angle",
                         choices=["target_angle", "threshold", "fixed_angle"],
                         help="target_angle: action[2] is a release angle; threshold: fire when > 0; "
                              "fixed_angle: release at --release-angle, torques-only policy")
    parser.add_argument("--release-angle", type=float, default=None,
                         help="shoulder angle (deg) to release at; fixed_angle mode only")
    parser.add_argument("--actuator-scale", type=float, default=1.0,
                         help="multiplier on every motor's gear (40/30 at 1.0), i.e. the actuator budget")
    parser.add_argument("--no-release-latch", action="store_true",
                         help="re-sample the release target every step instead of latching it")
    args = parser.parse_args()
    run(args.timesteps, args.n_envs, args.log_dir, args.model_path, args.seed, args.ent_coef,
        args.resume_from, args.swing_weight, args.straight_arm_weight, args.release_window,
        args.max_release_elbow, args.speed_weight, args.gamma, args.illegal_penalty,
        args.release_mode, not args.no_release_latch, args.release_angle, args.actuator_scale)
