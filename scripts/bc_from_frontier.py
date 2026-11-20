"""Start PPO from an optimised delivery instead of from random: behaviour cloning onto the frontier.

scripts/kin_frontier.py finds, per release angle and timing tolerance tau, open-loop arm torques
that bowl far faster than anything PPO found. This turns one of them into a PPO policy that
scripts/train_runup.py --resume-from can fine-tune (across starts, with timing/torque noise):

  1. the plan as a function of time relative to the captured release (so it applies from any
     start in 0.8-1.2 s; before the first knot it holds the first knot's torques)
  2. rollouts from --starts start times; each decision is executed with Gaussian action noise
     (--explore-sd) so the data covers states near the plan, and labelled with the plan's action
     at that time (the plan is open loop, so its label depends only on time)
  3. the actor mean is regressed onto the labels (MSE); log_std is set to --log-std
  4. the critic is regressed onto the delivery reward each state's rollout ended with
  5. the cloned policy is checked deterministically over 21 starts, as kin_equifinality does

    cricket/Scripts/python.exe scripts/bc_from_frontier.py --frontier results/kinematic/frontier/frontier_a270.json \
        --tau 0.5 --out checkpoints/kin_bowl_torque_bc_a270_tau0.5_init.zip
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.kinematic_bowl_env import KinematicBowlEnv


class Plan:
    def __init__(self, rec):
        self.t = np.array(rec["knot_times_s"])
        self.k = np.array(rec["knots"])

    def __call__(self, t_rel):
        return np.clip([np.interp(t_rel, self.t, self.k[:, j]) for j in range(self.k.shape[1])], -1.0, 1.0)


def make_env(angle, handover=None, seed=0, env2=None):
    """The plan's own model: a frontier record's env2 (drag, stiff elbow, ...) selects KinematicBowlEnv2, whose
    observation and actions are KinematicBowlEnv's, so train_runup --task kinematic2 can resume the clone."""
    if env2:
        from envs.kinematic_bowl_env2 import KinematicBowlEnv2
        kw = dict(env2)
        if kw.get("elbow_solref") is not None:
            kw["elbow_solref"] = list(kw["elbow_solref"])
        return KinematicBowlEnv2(delivery="T3a", rsi=True, release_angle_deg=angle, arm_handover_s=handover,
                                 seed=seed, **kw)
    return KinematicBowlEnv(delivery="T3a", rsi=True, release_angle_deg=angle, arm_control="torque",
                            arm_handover_s=handover, seed=seed)


def collect(env, plan, starts, explore_sd, reps, rng):
    X, Y, G = [], [], []
    for before in starts:
        for rep in range(reps):
            obs, _ = env.reset(options={"before_s": float(before)})
            xs, ys = [], []
            while True:
                y = plan(env.t - env.t_ref_release)
                a = np.clip(y + (explore_sd * rng.standard_normal(y.shape) if rep > 0 else 0.0), -1, 1)
                xs.append(obs)
                ys.append(y)
                obs, r, te, tr, info = env.step(a)
                if te or tr:
                    break
            X += xs
            Y += ys
            G += [r] * len(xs)            # one sparse reward per delivery, gamma ~ 1
    return np.array(X, np.float32), np.array(Y, np.float32), np.array(G, np.float32)


def check(env, model, n=21):
    out = []
    for before in np.linspace(0.8, 1.2, n):
        obs, _ = env.reset(options={"before_s": float(before)})
        while True:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, te, tr, info = env.step(a)
            if te or tr:
                break
        out.append(info["outcome"])
    good = sum(o["good"] for o in out)
    rel = [o for o in out if o["released"]]
    kmh = np.mean([o["speed_m_s"] * 3.6 for o in rel]) if rel else 0.0
    return good, kmh, out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frontier", required=True)
    p.add_argument("--tau", type=float, required=True)
    p.add_argument("--starts", type=int, default=41)
    p.add_argument("--reps", type=int, default=8, help="rollouts per start (the first without exploration noise)")
    p.add_argument("--explore-sd", type=float, default=0.15)
    p.add_argument("--log-std", type=float, default=-1.6, help="initial PPO exploration (std = exp)")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=4096, help="PPO minibatch for the fine-tuning")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    rec = next(r for r in json.load(open(a.frontier)) if abs(r["tau"] - a.tau) < 1e-9)
    angle, ho, env2 = rec["angle"], rec.get("handover_s"), rec.get("env2") or {}
    print(f"plan: a{int(angle)} tau {a.tau}: {rec['speed_kmh']:.1f} km/h, p_jit {rec['p_jitter_12ms']:.2f}")
    plan = Plan(rec)
    env = make_env(angle, ho, env2=env2)
    rng = np.random.default_rng(0)
    X, Y, G = collect(env, plan, np.linspace(0.8, 1.2, a.starts), a.explore_sd, a.reps, rng)
    print(f"{len(X)} labelled decisions; mean delivery reward {G.mean():.2f}")

    # the same PPO settings as train_runup.py --task kinematic, so --resume-from continues as usual
    model = PPO("MlpPolicy", DummyVecEnv([lambda: make_env(angle, ho, env2=env2)]), seed=0, learning_rate=3e-4, gamma=0.9999,
                ent_coef=0.003, target_kl=0.03, batch_size=a.batch_size, device="cpu")
    pol = model.policy
    Xt, Yt, Gt = torch.tensor(X), torch.tensor(Y), torch.tensor(G)
    actor = list(pol.mlp_extractor.policy_net.parameters()) + list(pol.action_net.parameters())
    critic = list(pol.mlp_extractor.value_net.parameters()) + list(pol.value_net.parameters())
    opt_a, opt_c = torch.optim.Adam(actor, 1e-3), torch.optim.Adam(critic, 1e-3)
    n = len(Xt)
    for ep in range(a.epochs):
        perm = torch.randperm(n)
        la = lc = 0.0
        for i in range(0, n, 1024):
            idx = perm[i:i + 1024]
            feats = pol.extract_features(Xt[idx], pol.pi_features_extractor)
            lat_pi, lat_vf = pol.mlp_extractor(feats)
            loss_a = ((pol.action_net(lat_pi) - Yt[idx]) ** 2).mean()
            loss_c = ((pol.value_net(lat_vf).squeeze(-1) - Gt[idx]) ** 2).mean()
            opt_a.zero_grad()
            loss_a.backward(retain_graph=True)
            opt_a.step()
            opt_c.zero_grad()
            loss_c.backward()
            opt_c.step()
            la, lc = la + loss_a.item() * len(idx), lc + loss_c.item() * len(idx)
        if ep % 50 == 0 or ep == a.epochs - 1:
            print(f"epoch {ep:4d}  actor mse {la / n:.5f}  critic mse {lc / n:.3f}", flush=True)
    with torch.no_grad():
        pol.log_std.fill_(a.log_std)
    good, kmh, _ = check(make_env(angle, ho, env2=env2), model)
    print(f"cloned policy, deterministic over 21 starts: good {good}/21 at {kmh:.1f} km/h")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    model.save(a.out)
    json.dump({"frontier": a.frontier, "tau": a.tau, "angle": angle, "handover_s": ho, "env2": env2, "plan_kmh": rec["speed_kmh"],
               "plan_p_jit": rec["p_jitter_12ms"], "clone_good_of_21": good, "clone_kmh": kmh},
              open(Path(a.out).with_suffix(".json"), "w"), indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
