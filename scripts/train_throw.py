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


class ActionStdLogger(BaseCallback):
    """Log the policy's per-dimension action standard deviation.

    SB3's own `train/std` is the mean across dimensions, which hides exactly
    the failure that mattered here: the release dimension's std reached 15.09
    while the torque dimensions sat near 1.2, and the averaged figure looked
    unremarkable. A dimension whose reward is insensitive to its mean
    collects the entropy bonus for free, and its samples stop meaning
    anything -- so it needs to be visible per dimension, in the job output,
    not recoverable only by probing the saved checkpoint afterwards.

    For the release channel the useful unit is degrees of shoulder angle
    rather than raw action units, since that is what the number does.
    """

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        import numpy as np
        std = np.exp(self.model.policy.log_std.detach().cpu().numpy())
        for i, name in enumerate(["shoulder", "elbow", "release"]):
            self.logger.record(f"action_std/{name}", float(std[i]))
        env = self.training_env.envs[0]._env
        span = ((env.release_window_max - env.RELEASE_TARGET_INSET_DEG)
                - env.release_window_min)
        # half the action range spans the whole window, so this is the
        # +/- spread of commanded release angles the policy is sampling
        self.logger.record("action_std/release_deg", float(std[2] * span / 2.0))


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
            "max_shoulder_deg", "shoulder_at_release_deg",
            "elbow_at_release_deg", "release_height_m",
            # what the policy ASKED for vs where the ball actually left.
            # Release timing is a controlled variable now, not an accident,
            # so it belongs in the results rather than only in diagnostics.
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
             speed_weight, gamma, illegal_penalty, release_mode, release_latch):
    def _init():
        # shaping_gamma must match PPO's gamma for the shaping to be the
        # correct potential-based form -- see ThrowEnvGym.__init__.
        return ThrowEnvGym(swing_weight=swing_weight,
                           straight_arm_weight=straight_arm_weight,
                           release_window_deg=tuple(release_window),
                           max_release_elbow_deg=max_release_elbow,
                           speed_weight=speed_weight,
                           illegal_penalty=illegal_penalty,
                           release_mode=release_mode,
                           release_latch=release_latch,
                           shaping_gamma=gamma)
    return _init


def run(total_timesteps, n_envs, log_dir, model_path, seed, ent_coef, resume_from,
        swing_weight, straight_arm_weight, release_window, max_release_elbow,
        speed_weight, gamma, illegal_penalty, release_mode, release_latch):
    vec_env = DummyVecEnv([
        make_env(swing_weight, straight_arm_weight, release_window, max_release_elbow,
                 speed_weight, gamma, illegal_penalty, release_mode, release_latch)
        for _ in range(n_envs)])
    if resume_from:
        # warm-start from an existing checkpoint instead of a fresh random
        # policy -- reset_num_timesteps=False keeps model.num_timesteps
        # (and its learning-rate/clip-range schedules) continuing on from
        # wherever the checkpoint left off, rather than restarting at 0.
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
    parser.add_argument("--speed-weight", type=float, default=0.1,
                         help="coefficient on release speed in ThrowEnv's reward, applied only "
                              "once the throw is both accurate and legal. This is the research "
                              "question's objective, so it is a reported config value, not a "
                              "tuning knob to be changed casually between runs. "
                              "WARNING: 0.1 is measured to be too weak -- at 0.1 the policy never "
                              "commits to releasing, relies on exploration noise to fire the ball, "
                              "and fails deterministic evaluation entirely (0 releases). At 0.3 it "
                              # NB: literal percent signs must be escaped as %% -- argparse formats
                              # help strings with `help % params`, and Python 3.14 raises
                              # "badly formed help string" at add_argument() time. An
                              # unescaped % here killed all three run7 jobs instantly.
                              "commits, and the same 900k-step budget reaches 62%% of throws in the "
                              "zone at mean reward +1.63 instead of 33%% at -0.11. Raising it is a "
                              "research-objective decision, so the default is left alone pending "
                              "that call -- see context.md, 'Open problem: release timing'.")
    parser.add_argument("--swing-weight", type=float, default=30.0,
                         help="scale of the shaping term rewarding progress of the shoulder around "
                              "the swing arc toward the release orientation. 0 disables it, leaving "
                              "the landing-distance potential alone. Needs to be >~17 to overcome the "
                              "distance term's penalty on the backswing -- at 5.0 the policy collapses "
                              "to standing still and dropping the ball. See ThrowEnvGym._potential().")
    parser.add_argument("--straight-arm-weight", type=float, default=6.0,
                         help="scale of the shaping term rewarding elbow extension as the release "
                              "window approaches, so PPO can find the hard elbow-at-release gate. "
                              "0 disables it. Like --swing-weight this is potential-based and "
                              "therefore cannot change the optimal policy, only how fast it is "
                              "found. See ThrowEnvGym._potential().")
    parser.add_argument("--release-window", type=float, nargs=2, default=[230.0, 310.0],
                         metavar=("MIN_DEG", "MAX_DEG"),
                         help="shoulder-angle sector in which the ball may leave the hand (ICC "
                              "'overarm, not underarm'). 270 is straight up, the true overarm "
                              "release point. Widening this toward 180 re-admits the sling-from-"
                              "behind-the-back action that run6 converged on.")
    parser.add_argument("--max-release-elbow", type=float, default=40.0,
                         help="maximum absolute elbow flexion permitted at release, degrees. A "
                              "bowling action releases with a near-straight arm; run6 released at "
                              "113 deg, which is a shot-put. This is an action-validity filter, "
                              "NOT the ICC legality metric -- that stays extension-based and is "
                              "reported separately.")
    parser.add_argument("--gamma", type=float, default=0.9999,
                         help="PPO discount factor, also used for the potential-based shaping so "
                              "the two stay consistent. NOT SB3's 0.99 default: the entire task "
                              "reward arrives in one lump at episode end, and episode length is "
                              "something the policy controls by choosing when (or whether) to "
                              "release. At 0.99 a terminal reward 240 policy steps out is worth "
                              "0.09 of its value, so stalling to timeout (-6.1 -> -0.55) beat a "
                              "bad throw (-4.0 -> -1.46) and the policy collapsed to standing "
                              "still. 0.999 was not enough either -- it ranked a good throw above "
                              "stalling but left swinging-without-releasing below it, so the path out "
                              "of the do-nothing policy still ran downhill first. 0.9999 makes the "
                              "ordering monotone: good throw > bad throw > failed swing > stand still.")
    parser.add_argument("--illegal-penalty", type=float, default=2.0,
                         help="flat penalty added to an illegal delivery's accuracy score. Kept "
                              "separate from the accuracy term so accuracy stays monotone for "
                              "illegal throws too -- the previous form made an illegal "
                              "dead-centre throw (-1.00) score worse than an illegal throw "
                              "landing on the zone boundary (-0.00). Sets the size of the "
                              "legality incentive, so it is a reported config value.")
    parser.add_argument("--release-mode", default="target_angle",
                         choices=["target_angle", "threshold"],
                         help="how action[2] is read. target_angle: it names the shoulder angle to "
                              "release at, one continuous decision. threshold: fire on any step "
                              "inside the window where it exceeds zero -- the original, kept only "
                              "to reproduce run7. threshold is badly posed: the policy only has to "
                              "clear zero once in ~12 decisions, so the dimension gets almost no "
                              "gradient and the entropy bonus inflated its std to 9-19 while the "
                              "torque dims stayed near 1.5, making release timing random.")
    parser.add_argument("--no-release-latch", action="store_true",
                         help="re-decide the release target every policy step instead of latching "
                              "it once at window entry. This is what run8 did, and it does not "
                              "work: only one of the ~12 decisions inside the window has to say "
                              "go, so the ball leaves on the MINIMUM of the sampled targets. A "
                              "policy intending 265 deg released at 236.5 deg on average. Kept "
                              "only to reproduce run8.")
    args = parser.parse_args()
    run(args.timesteps, args.n_envs, args.log_dir, args.model_path, args.seed, args.ent_coef,
        args.resume_from, args.swing_weight, args.straight_arm_weight, args.release_window,
        args.max_release_elbow, args.speed_weight, args.gamma, args.illegal_penalty,
        args.release_mode, not args.no_release_latch)
