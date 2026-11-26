"""Train the whole-body tracking controller for the dynamic G1 (envs/g1_track_env.py) with PPO, on the HPC.

Every --eval-every steps the current policy runs the whole clip deterministically from its first frame (the
real test: start mid run-up, get through release and the follow-through without falling) and the result is
printed and appended to logs/<run>/eval.csv; the policy is checkpointed with its observation normaliser.

    python scripts/train_g1_track.py --run g1track_s0 --n-envs 44 --timesteps 300000000 [--action-scale 0.25]
  -> checkpoints/<run>.zip + <run>_vecnorm.pkl (+ _ckpts/), logs/<run>/{progress.csv,eval.csv}
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def make(delivery, action_scale, seed, t_from, t_to):
    def f():
        from envs.g1_track_env import G1TrackEnv
        return G1TrackEnv(delivery=delivery, action_scale=action_scale, seed=seed, t_from=t_from, t_to=t_to)
    return f


def evaluate(model, vecnorm, delivery, action_scale, t_from, t_to, record=False):
    """The whole clip from its first frame, deterministic. Returns stats (and the qpos per step if record)."""
    from envs.g1_track_env import G1TrackEnv
    env = G1TrackEnv(delivery=delivery, action_scale=action_scale, rsi=False, t_from=t_from, t_to=t_to)
    obs, _ = env.reset(options={"k": 0})
    frames, R, n, info = [env.d.qpos.copy()], 0.0, 0, {}
    je, re_, ee = [], [], []
    while True:
        a, _ = model.predict(vecnorm.normalize_obs(obs[None]) if vecnorm is not None else obs[None], deterministic=True)
        obs, r, te, tr, info = env.step(a[0])
        R, n = R + r, n + 1
        je.append(info["joint_err_rad"]), re_.append(info["root_err_m"]), ee.append(info["ee_err_m"])
        if record:
            frames.append(env.d.qpos.copy())
        if te or tr:
            break
    out = {"completed": bool(info.get("completed")), "fell": bool(info.get("fell")), "steps": n,
           "of_steps": env.n_grid - 1, "reached_t_rel": float(info["t_rel_release"]), "return": float(R),
           "joint_err_deg": float(np.degrees(np.mean(je))), "root_err_m": float(np.mean(re_)),
           "ee_err_m": float(np.mean(ee))}
    if record:
        out["qpos"] = np.array(frames)
        out["dt"] = env.dt
        out["t_start_rel"] = env.t0 - env.t_rel
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--delivery", default="T3aLf")
    p.add_argument("--n-envs", type=int, default=44)
    p.add_argument("--timesteps", type=int, default=300_000_000)
    p.add_argument("--action-scale", type=float, default=0.25)
    p.add_argument("--from", dest="t_from", type=float, default=-1.0)
    p.add_argument("--to", dest="t_to", type=float, default=0.35)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-steps", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--target-kl", type=float, default=None, help="PPO early stop per update (e.g. 0.03)")
    p.add_argument("--log-std", type=float, default=-1.0)
    p.add_argument("--eval-every", type=int, default=5_000_000)
    p.add_argument("--ckpt-every", type=int, default=25_000_000)
    p.add_argument("--max-hours", type=float, default=None, help="stop cleanly (and save) after this long")
    p.add_argument("--torch-threads", type=int, default=4)
    p.add_argument("--dummy", action="store_true", help="DummyVecEnv (smoke tests)")
    a = p.parse_args()

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
    torch.set_num_threads(a.torch_threads)
    log = ROOT / "logs" / a.run
    log.mkdir(parents=True, exist_ok=True)
    ck = ROOT / "checkpoints"
    (ck / f"{a.run}_ckpts").mkdir(parents=True, exist_ok=True)
    Vec = DummyVecEnv if a.dummy else SubprocVecEnv
    venv = Vec([make(a.delivery, a.action_scale, a.seed * 1000 + i, a.t_from, a.t_to) for i in range(a.n_envs)])
    venv = VecNormalize(VecMonitor(venv), norm_obs=True, norm_reward=False, clip_obs=10.0)
    batch = a.n_envs * a.n_steps
    model = PPO("MlpPolicy", venv, seed=a.seed, learning_rate=a.lr, n_steps=a.n_steps, batch_size=batch // 4,
                n_epochs=5, gamma=0.95, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, max_grad_norm=0.5,
                target_kl=a.target_kl,
                policy_kwargs=dict(net_arch=dict(pi=[512, 256], vf=[512, 256]), log_std_init=a.log_std,
                                   activation_fn=torch.nn.ELU), verbose=0, device="cpu")
    model.set_logger(configure(str(log), ["csv", "stdout"]))
    t_start = time.time()
    ev_file = log / "eval.csv"
    print(f"run {a.run}: {a.n_envs} envs, rollout {batch}, delivery {a.delivery}, action scale {a.action_scale}, "
          f"clip {a.t_from}..{a.t_to} s", flush=True)

    class Cb(BaseCallback):
        def __init__(self):
            super().__init__()
            self.next_ev, self.next_ck = a.eval_every, a.ckpt_every
            self.done_ep, self.completed = 0, 0

        def _on_step(self):
            for info, done in zip(self.locals["infos"], self.locals["dones"]):
                if done:
                    self.done_ep += 1
                    self.completed += bool(info.get("completed"))
            if self.num_timesteps >= self.next_ev:
                self.next_ev += a.eval_every
                e = evaluate(self.model, venv, a.delivery, a.action_scale, a.t_from, a.t_to)
                e.update(timesteps=self.num_timesteps, hours=(time.time() - t_start) / 3600,
                         rsi_episodes=self.done_ep, rsi_completed_pct=100 * self.completed / max(self.done_ep, 1))
                self.done_ep = self.completed = 0
                new = not ev_file.exists()
                with open(ev_file, "a", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(e))
                    if new:
                        w.writeheader()
                    w.writerow(e)
                print(f"[eval {self.num_timesteps / 1e6:.0f}M, {e['hours']:.2f} h] full clip: "
                      f"{'COMPLETED' if e['completed'] else 'fell'} at {e['reached_t_rel']:+.2f} s "
                      f"({e['steps']}/{e['of_steps']} steps), joint err {e['joint_err_deg']:.1f} deg, "
                      f"pelvis err {e['root_err_m']:.3f} m | RSI episodes completed {e['rsi_completed_pct']:.0f}%",
                      flush=True)
            if self.num_timesteps >= self.next_ck:
                self.next_ck += a.ckpt_every
                self.model.save(ck / f"{a.run}_ckpts" / f"step_{self.num_timesteps}")
                venv.save(str(ck / f"{a.run}_ckpts" / f"step_{self.num_timesteps}_vecnorm.pkl"))
                self.model.save(ck / a.run)
                venv.save(str(ck / f"{a.run}_vecnorm.pkl"))
            if a.max_hours and time.time() - t_start > a.max_hours * 3600:
                print("time limit reached", flush=True)
                return False
            return True

    model.learn(total_timesteps=a.timesteps, callback=Cb())
    model.save(ck / a.run)
    venv.save(str(ck / f"{a.run}_vecnorm.pkl"))
    e = evaluate(model, venv, a.delivery, a.action_scale, a.t_from, a.t_to, record=True)
    np.savez(log / "final_rollout.npz", qpos=e.pop("qpos"), dt=e.pop("dt"), t_start_rel=e.pop("t_start_rel"))
    print("final:", e, flush=True)


if __name__ == "__main__":
    main()
