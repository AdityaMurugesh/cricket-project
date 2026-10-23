"""PPO (or SAC) training loop for ThrowEnvGym, logging every episode to a CSV."""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.throw_env_gym import ThrowEnvGym

ROOT = Path(__file__).resolve().parent.parent


class ActionStdLogger(BaseCallback):
    """Log per-dimension action std (SB3 only logs the mean across dims)."""

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        import numpy as np
        if not hasattr(self.model.policy, "log_std"):
            return  # SAC: state-dependent std, logged by SB3 as train/ent_coef instead
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
             release_angle, actuator_scale, target_range=(6.0, 8.0), use_shaping=True,
             resolve_at_release=False, timeout_terminal=False, release_timing_sd_ms=0.0,
             torque_noise_sd=0.0):
    def _init():
        # shaping gamma must equal PPO gamma
        return ThrowEnvGym(use_shaping=use_shaping, resolve_at_release=resolve_at_release,
                           timeout_terminal=timeout_terminal,
                           release_timing_sd_ms=release_timing_sd_ms,
                           torque_noise_sd=torque_noise_sd,
                           swing_weight=swing_weight,
                           straight_arm_weight=straight_arm_weight,
                           release_window_deg=tuple(release_window),
                           max_release_elbow_deg=max_release_elbow,
                           speed_weight=speed_weight,
                           illegal_penalty=illegal_penalty,
                           release_mode=release_mode,
                           release_latch=release_latch,
                           release_angle_deg=release_angle,
                           actuator_scale=actuator_scale,
                           target_range=tuple(target_range),
                           shaping_gamma=gamma)
    return _init


def run(total_timesteps, n_envs, log_dir, model_path, seed, ent_coef, resume_from,
        swing_weight, straight_arm_weight, release_window, max_release_elbow,
        speed_weight, gamma, illegal_penalty, release_mode, release_latch,
        release_angle, actuator_scale, checkpoint_every=0, target_range=(6.0, 8.0),
        algo="ppo", use_shaping=True, lr=3e-4, lr_decay=False, target_kl=None,
        freeze_log_std=False, resolve_at_release=False,
        timeout_terminal=False, release_timing_sd_ms=0.0, torque_noise_sd=0.0):
    vec_env = DummyVecEnv([
        make_env(swing_weight, straight_arm_weight, release_window, max_release_elbow,
                 speed_weight, gamma, illegal_penalty, release_mode, release_latch,
                 release_angle, actuator_scale, target_range, use_shaping, resolve_at_release,
                 timeout_terminal, release_timing_sd_ms, torque_noise_sd)
        for _ in range(n_envs)])
    Algo = SAC if algo == "sac" else PPO
    if resume_from:
        # warm start; timestep counter continues from the checkpoint
        model = Algo.load(resume_from, env=vec_env)
        model.set_random_seed(seed)
        if algo == "ppo":
            model.ent_coef = ent_coef
        print(f"resumed from {resume_from} at num_timesteps={model.num_timesteps}")
    elif algo == "sac":
        # ent_coef stays "auto" (SAC tunes its own entropy target); one gradient
        # step per transition collected
        model = SAC("MlpPolicy", vec_env, verbose=1, seed=seed, gamma=gamma,
                    learning_rate=lr, gradient_steps=-1)
    else:
        model = PPO("MlpPolicy", vec_env, verbose=1, seed=seed, ent_coef=ent_coef,
                    gamma=gamma, learning_rate=lr)
    if lr_decay:
        # linear to 0 over this call's timesteps, starting at lr even when resumed
        start, end = model.num_timesteps, model.num_timesteps + total_timesteps
        p0 = 1.0 - start / end
        model.learning_rate = lambda p: lr * max(p, 0.0) / p0
        model._setup_lr_schedule()
    elif resume_from:
        model.learning_rate = lr
        model._setup_lr_schedule()
    if target_kl is not None and algo == "ppo":
        model.target_kl = target_kl
    if freeze_log_std and algo == "ppo":
        # exploration noise stays where it starts (1.0 fresh, or the checkpoint's)
        model.policy.log_std.requires_grad_(False)
    print(f"algo={algo} shaping={use_shaping} lr={lr} lr_decay={lr_decay} "
          f"target_kl={target_kl} freeze_log_std={freeze_log_std} "
          f"resolve_at_release={resolve_at_release} timeout_terminal={timeout_terminal} "
          f"release_timing_sd_ms={release_timing_sd_ms} torque_noise_sd={torque_noise_sd}")

    csv_path = Path(log_dir) / "episodes.csv"
    callback = [EpisodeLogger(csv_path), ActionStdLogger()]
    if checkpoint_every > 0:
        # run10 policies reached the zone and then drifted out of it, so the
        # final checkpoint alone can understate what training found
        callback.append(CheckpointCallback(
            save_freq=max(1, checkpoint_every // n_envs),
            save_path=str(Path(model_path).parent / (Path(model_path).name + "_ckpts")),
            name_prefix="step"))
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
    parser.add_argument("--checkpoint-every", type=int, default=1_000_000,
                         help="also save the policy every N timesteps (0 disables)")
    parser.add_argument("--target-range", type=float, nargs=2, default=[6.0, 8.0],
                         metavar=("MIN_M", "MAX_M"),
                         help="landing zone (m from release); 10.9 12.9 is a real good length")
    parser.add_argument("--algo", default="ppo", choices=["ppo", "sac"])
    parser.add_argument("--no-shaping", action="store_true",
                         help="task reward only; shaping is policy-invariant, so this is safe on --resume-from")
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate (SB3 default 3e-4)")
    parser.add_argument("--lr-decay", action="store_true", help="decay the learning rate linearly to 0")
    parser.add_argument("--target-kl", type=float, default=None,
                         help="PPO: stop an update's epochs early once KL to the old policy passes this")
    parser.add_argument("--freeze-log-std", action="store_true",
                         help="PPO: keep exploration noise fixed instead of learning it")
    parser.add_argument("--resolve-at-release", action="store_true",
                         help="simulate the ball's flight inside the releasing step and end the episode there")
    parser.add_argument("--timeout-terminal", action="store_true",
                         help="treat the time limit as a failed delivery (terminal), not a bootstrapped cut-off")
    parser.add_argument("--release-timing-sd-ms", type=float, default=0.0,
                         help="fixed_angle: SD (ms) of release-timing jitter, unseen by the policy (0 = exact)")
    parser.add_argument("--torque-noise-sd", type=float, default=0.0,
                         help="signal-dependent torque noise: torque * (1 + sd * N(0,1)) per decision")
    args = parser.parse_args()
    run(args.timesteps, args.n_envs, args.log_dir, args.model_path, args.seed, args.ent_coef,
        args.resume_from, args.swing_weight, args.straight_arm_weight, args.release_window,
        args.max_release_elbow, args.speed_weight, args.gamma, args.illegal_penalty,
        args.release_mode, not args.no_release_latch, args.release_angle, args.actuator_scale,
        args.checkpoint_every, args.target_range, args.algo, not args.no_shaping, args.lr,
        args.lr_decay, args.target_kl, args.freeze_log_std, args.resolve_at_release,
        args.timeout_terminal, args.release_timing_sd_ms, args.torque_noise_sd)
