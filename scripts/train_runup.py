"""PPO on the run-up envs. One CSV row per episode.

  --task imitate  RunupImitationEnv: learn the captured run-up and trunk motion (stage 1)
  --task bowl     RunupBowlEnv: the same, plus bowling with the right arm (stage 3)
  --task kinematic  KinematicBowlEnv: the captured body exactly, only the bowling arm learned

    python scripts/train_runup.py --delivery T3a --timesteps 50000000 --n-envs 16 \
        --log-dir logs/runup_ppo_r1 --model-path checkpoints/runup_ppo_r1
    python scripts/train_runup.py --task bowl --release-angle 250 ...

Needs data/mocap/ref_<D>.npz (envs/humanoid_bowler.export_reference).
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from envs.kinematic_bowl_env import KinematicBowlEnv
from envs.runup_bowl_env import RunupBowlEnv
from envs.runup_imitation_env import RunupImitationEnv

ROOT = Path(__file__).resolve().parent.parent
TERMS = ("pose", "vel", "ends", "root")


class EpisodeLogger(BaseCallback):
    """Per episode: where it started, how far it got, how well it tracked, and the delivery."""

    def __init__(self, csv_path):
        super().__init__()
        self.csv_path = Path(csv_path)
        self.acc = {}

    def _on_training_start(self):
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.csv_path, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(["episode", "timesteps", "start_t", "end_t", "length", "return",
                         "reached_release", "fell", *(f"mean_{k}" for k in TERMS),
                         "task_reward", "released", "why", "speed_kmh", "land_x", "lateral_m",
                         "extension_deg", "legal", "good"])
        self.n = 0

    @staticmethod
    def _delivery(info):
        o = info.get("outcome")
        if o is None:
            return [""] * 9
        r = lambda v, k=3: "" if v is None else round(float(v), k)
        return [r(info.get("task_reward")), int(o["released"]), o.get("why", ""),
                r(o["speed_m_s"] * 3.6, 2), r(o["land_x"]), r(o.get("lateral_m")),
                r(o["extension_deg"], 2), int(o["legal"]), int(o["good"])]

    def _on_step(self):
        for i, (info, r, done) in enumerate(zip(self.locals["infos"], self.locals["rewards"],
                                                self.locals["dones"])):
            a = self.acc.setdefault(i, {"R": 0.0, "n": 0, **dict.fromkeys(TERMS, 0.0)})
            a["R"] += float(r)
            a["n"] += 1
            for k in TERMS:
                a[k] += float(info["terms"][k])
            if done:
                self.n += 1
                self.w.writerow([self.n, self.num_timesteps, round(info["start_t"], 3), round(info["t"], 3),
                                 a["n"], round(a["R"], 3), int(info["reached_release"]), int(info["fell"]),
                                 *(round(a[k] / a["n"], 4) for k in TERMS), *self._delivery(info)])
                self.acc[i] = {"R": 0.0, "n": 0, **dict.fromkeys(TERMS, 0.0)}
        if self.n % 200 == 0:
            self.f.flush()
        return True

    def _on_training_end(self):
        self.f.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="imitate", choices=["imitate", "bowl", "kinematic"])
    p.add_argument("--delivery", default="T3a")
    p.add_argument("--release-angle", type=float, default=250.0, help="bowl: arm angle at release (deg)")
    p.add_argument("--target-range", type=float, nargs=2, default=[10.9, 12.9],
                   help="bowl: landing zone, metres from the release point")
    p.add_argument("--task-weight", type=float, default=10.0, help="bowl: weight of the delivery reward")
    p.add_argument("--arm-control", default="pd", choices=["pd", "torque"],
                   help="kinematic: the arm action is PD targets or (ThrowEnv's) torques")
    p.add_argument("--max-pelvis-drift", type=float, default=None,
                   help="bowl: drifting this far (m) from the captured pelvis path counts as a fall")
    p.add_argument("--timesteps", type=int, default=50_000_000)
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--log-dir", default=str(ROOT / "logs" / "runup_ppo"))
    p.add_argument("--model-path", default=str(ROOT / "checkpoints" / "runup_ppo"))
    p.add_argument("--checkpoint-every", type=int, default=5_000_000)
    p.add_argument("--resume-from", default=None,
                   help="warm start, e.g. a stage-1 run-up checkpoint for --task bowl (same spaces)")
    p.add_argument("--dummy", action="store_true", help="single-process envs (debugging)")
    a = p.parse_args()

    def make(rank):
        if a.task == "kinematic":
            return lambda: KinematicBowlEnv(delivery=a.delivery, seed=a.seed * 1000 + rank,
                                            release_angle_deg=a.release_angle,
                                            target_range=tuple(a.target_range), task_weight=a.task_weight,
                                            arm_control=a.arm_control)
        if a.task == "bowl":
            return lambda: RunupBowlEnv(delivery=a.delivery, seed=a.seed * 1000 + rank,
                                        release_angle_deg=a.release_angle,
                                        target_range=tuple(a.target_range), task_weight=a.task_weight,
                                        max_pelvis_drift_m=a.max_pelvis_drift)
        return lambda: RunupImitationEnv(delivery=a.delivery, seed=a.seed * 1000 + rank)

    Vec = DummyVecEnv if a.dummy else SubprocVecEnv
    venv = Vec([make(i) for i in range(a.n_envs)])
    if a.resume_from:
        model = PPO.load(a.resume_from, env=venv)
        model.set_random_seed(a.seed)
    elif a.task == "kinematic":
        # ThrowEnv's settings (run10-run15): one sparse reward per delivery, so no discounting
        # (the policy sets the episode length: episodic_discounting_trap), entropy 0.003
        model = PPO("MlpPolicy", venv, verbose=1, seed=a.seed, learning_rate=3e-4, gamma=0.9999,
                    ent_coef=0.003, target_kl=0.03)
    else:
        # DeepMimic-like settings: short horizon (gamma 0.95), wide nets, small initial noise
        model = PPO("MlpPolicy", venv, verbose=1, seed=a.seed, learning_rate=a.lr,
                    n_steps=256, batch_size=1024, n_epochs=10, gamma=0.95, gae_lambda=0.95,
                    clip_range=0.2, ent_coef=0.0,
                    policy_kwargs=dict(net_arch=dict(pi=[512, 256], vf=[512, 256]),
                                       activation_fn=torch.nn.ReLU, log_std_init=-1.0))
    callbacks = [EpisodeLogger(Path(a.log_dir) / "episodes.csv")]
    if a.checkpoint_every > 0:
        callbacks.append(CheckpointCallback(save_freq=max(1, a.checkpoint_every // a.n_envs),
                                            save_path=a.model_path + "_ckpts", name_prefix="step"))
    print(f"task={a.task} delivery={a.delivery} envs={a.n_envs} timesteps={a.timesteps} lr={a.lr} "
          f"release_angle={a.release_angle} zone={a.target_range} task_weight={a.task_weight} "
          f"max_pelvis_drift={a.max_pelvis_drift} resume_from={a.resume_from}", flush=True)
    model.learn(total_timesteps=a.timesteps, callback=callbacks, reset_num_timesteps=a.resume_from is None)
    Path(a.model_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(a.model_path)
    print("saved", a.model_path)


if __name__ == "__main__":
    main()
