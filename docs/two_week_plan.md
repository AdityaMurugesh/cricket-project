# Two-week plan and log: 23 Oct – 6 Nov 2026

The plan comes first, then a dated log underneath. Each day's log entry records what ran, where its
output is, and what it showed. Job IDs are PBS jobs on the IITD-AD cluster (`cpuq`).

## Where things stand (23 Oct)

- **The main problem is find-then-lose.** Policies reach the target zone during training and then
  drift out of it.
  - run10 (6–8 m zone): 21 runs lost the zone.
  - run11 (10.9–12.9 m, a real good length): **0 of 24** final policies are good, even though 14 runs
    peaked at ≥10% good deliveries during training (best: 80%).
- **The collapse is not driven by pace** (`results/run10/run10_reward_terms.png`). The policies slow
  down before they lose the zone: 28.9 → 25.1 km/h in run10 and 38.0 → 28.4 km/h in run11. Then
  they land short, or they turn illegal at 1.5x.
- **PPO's own stats** (`results/run11/run11_ppo_stats.png`):
  - Exploration noise shrinks steadily over training (std 0.65 → 0.19), and the policy loses the zone
    while the noise is falling.
  - The size of each policy update (approx KL) roughly doubles (0.007 → 0.012).
  - The value function stays accurate (explained variance about 0.98).
  - Leading suspect: unstable updates once the policy is near-deterministic. The run has a constant
    learning rate and no `target_kl`, so the same step size moves a narrow policy further.
  - The "noise makes slow deliveries safer" idea is weaker now, because the noise was falling, not
    rising, while the policy slowed.
- Supervisor's thread (meeting 23 Oct): equifinal trajectories, noise-dependent actions, functional
  variability. Papers read: Andrews 2024, Müller & Loosch 1999, Reiser 2011, Lindsay 2024.
  - Current policies are not equifinal: the zone holds for only 4–22 ms of release timing.
  - The simulator has no motor noise and no sensor noise at all.

## Week 1 (24–30 Oct): fix the collapse

| When | What | Output |
|---|---|---|
| 23 Oct (running) | run11 checkpoint eval: finds the best checkpoint of each run (job 27040) | `logs/throw_ppo_run11_*/ckpt_eval.json` |
| 23 Oct (running) | SAC pilot: 240° 1.5x and 230° 1.0x, 2 seeds each, 1.5M steps (jobs 27045–27048) | `logs/throw_sac_run12_*` |
| 23 Oct (running) | run12 resume diagnosis: 4 arms (control / no shaping / stable lr + target_kl / frozen noise) x 3 run11 peak checkpoints, 4M steps each (jobs 27049�27051, about 40 min) | `logs/throw_ppo_run12{arm}_*` |
| 24–25 Oct | Read run12 and SAC. Whichever arm keeps the zone names the cause. | figures in `results/run12/` |
| 26–27 Oct | run13: a full sweep with the fix, from scratch, on the good length (and on 6–8 m, for comparison with run10) | `logs/throw_*_run13_*` |
| 28–29 Oct | Equifinality script (`scripts/equifinality.py`): Reiser's ±40 ms window and the in-zone release window for every checkpoint. Measures whether windows widen or narrow over training. | `results/*/equifinality.csv` |
| 30 Oct | **A100 reservation ends.** This doesn't matter, because everything runs on CPU. | |

Decision for the user in week 1: if SAC keeps the zone and PPO-with-fix doesn't, switch algorithms
for run13.

## Week 2 (31 Oct – 6 Nov): noise and equifinality (supervisor's thread)

| When | What | Output |
|---|---|---|
| 31 Oct – 1 Nov | Add motor noise to `ThrowEnv`, off by default. (a) Release-timing jitter with SD 10–15 ms (Müller & Loosch). (b) Signal-dependent torque noise, with SD proportional to the torque (Harris & Wolpert 1998). | env option and unit check |
| 1 Nov | **Review the noise design with the user before training**, since it changes the task. | |
| 2–4 Nov | run14: train with noise on vs off. Does the learned trajectory become equifinal, so that the in-zone window widens? Compute Δcovar (Müller & Loosch) and CV(timing) vs CV(speed) (Andrews). | `results/run14/` |
| 5 Nov | Compare with the 11 mocap deliveries (weak: n = 3–11, one bowler), as a limitation and not a finding | figure |
| 6 Nov | Supervisor meeting pack: collapse cause + fix, equifinality over training, noise result | slides / figures |

## Open items, not scheduled

- Known reward edge case: an illegal delivery that lands in the zone is scored by its distance to
  the nearest edge, so accuracy is punished once a delivery is illegal. Deferred; decide before any
  reward change.
- TensorBoard is not installed on the HPC. SB3 already prints the same stats into each `job.log`, and
  `scripts/ppo_stats.py` plots them, so it isn't needed for now.
- A wrist or trunk joint: the 2-DOF arm has little redundancy for equifinality to use.
- `data/mocap/` stays out of git (participant video; the repo is public).

## Log

### 23 Oct
- `scripts/reward_terms.py`: run10 reward split into hit/pace/miss/illegal. Pace is the biggest
  reward term, and the policies give it up.
- run11 finished (jobs 26992–26997). In training, up to 13.8k legal good-length deliveries per run;
  at the end, 0/24 policies are good.
- `scripts/ppo_stats.py`: noise shrinks and update size grows while the policies lose the zone.
- `scripts/train_throw.py`: added `--algo sac`, `--no-shaping`, `--lr`, `--lr-decay`, `--target-kl`,
  `--freeze-log-std`. `eval_ckpts.py` now works for any sweep. Added `hpc/sweep_post.pbs`.
- Submitted: 27040 (run11 checkpoint eval) and 27045–27048 (SAC pilot). run12 follows once the
  peak checkpoints are known.
- **run12 result (first seed): the reward shaping is what drives the collapse.** Each arm was
  restarted from 3 peak checkpoints. The table gives good deliveries (%) under normal noise at the
  end, and where the deterministic delivery lands.

  | arm | a230 1.0x | a230 1.5x | a250 1.5x |
  |---|---|---|---|
  | control (unchanged) | 0% (8.5 m, short) | 34% (10.1 m) | 0% (7.6 m) |
  | stable lr + target_kl | 0% | 0% | 0% |
  | frozen noise | 0% | 0% | 0% |
  | **no shaping** | **74% (11.7 m, good, 36.7 km/h)** | **80%** | 1% (16.0 m, long, 50 km/h) |

  - Neither stabilising PPO nor freezing the noise helps, so those explanations are out.
  - Without shaping, two of the three runs keep the zone. The third goes long, chasing pace.
  - Likely mechanism, not yet proven: the potential depends on the ball's position and velocity
    and on whether the ball has been released, but the policy only observes the arm's two angles
    and two velocities. Shaping is policy-invariant only if the potential is a function of the
    state the learner sees.
  - There is also a large drop in the potential at landing (up to −36: swing term 30 + straight term 6),
    at a time the critic can't see coming.
  - Fix candidates for run13:
    - (a) shaping to find the zone, then switch it off (what run12 did);
    - (b) end the shaping at release, not at landing;
    - (c) add the ball state and a released flag to the observation.
- Replication submitted: no-shaping vs control on all 10 rescued run11 checkpoints, seed 1
  (jobs 27060–27067, plus 2 more queued behind the per-user job limit).
- **Replication (seed 1, all 10 rescued checkpoints, jobs 27060–27067, 27216, 27217): the no-shaping
  fix does not hold up.** The table gives good deliveries (%) under normal noise at the end, and
  where the deterministic delivery lands.

  | source | control | no shaping |
  |---|---|---|
  | a230 1.0x s2 | 0% (9.0 m) | 0% (3.4 m, 18 km/h) |
  | a230 1.5x s0 | 0% (5.0 m) | 0% (no landing) |
  | a230 1.5x s1 | 0% | 0% (no landing) |
  | a230 1.5x s2 | 0.5% (9.4 m) | 0% (14.6 m) |
  | a240 1.5x s0 | 5.5% (10.4 m) | 1% (15.7 m) |
  | a240 1.5x s2 | 0% (6.0 m) | 0% (18.2 m) |
  | a250 1.5x s0 | 0% (7.3 m) | 9.5% (14.0 m) |
  | a250 1.5x s1 | 0% (7.6 m) | 4.5% (14.2 m) |
  | a250 1.5x s2 | 0% (6.8 m) | 36.5% (10.6 m) |
  | a260 1.5x s1 | 0% (5.6 m) | 74.5% (13.5 m, 53 km/h) |

  - **Revised reading:** the shaping sets the *direction* of the drift but is not the whole cause.
    - With shaping (control), runs drift slow and short: 10 out of 10, again.
    - Without it, they mostly drift fast and long (14–18 m, 41–53 km/h) and still leave the zone.
  - The seed-0 result (2 of 3 kept) was partly luck. Either way, PPO isn't holding a narrow target.
  - Next suspects:
    - the reward cliff at the zone edges (+1 + pace inside, −distance just outside);
    - PPO itself, which the SAC pilot tests.
  - Not yet checked: what "no landing" means in the two no-shaping runs (no release, or still in
    the air at timeout).

### 23 Oct (evening)
- **New lead: most decisions happen after the ball has gone.** I replayed three run11 checkpoints:
  - Release comes at decision 27–48, but the episode runs to decision 151–183.
  - So 104–156 decisions (75–85% of the episode, 1.0–1.6 s of flight) come after release, when no
    action can change the outcome.
  - With GAE (γλ ≈ 0.95), the landing reward reaches the release-time decisions with weight
    3e-4 to 5e-3. The rest has to be carried by the critic, which only sees the arm and not the ball.
  - So nearly all of the policy-gradient signal lands on irrelevant post-release arm motion. That
    fits a drift whose direction depends on the shaping: the shaping terms keep paying out on that
    post-release arm motion.
- **Fix: `--resolve-at-release`** (`envs/throw_env_gym.py`). When the ball is released, the flight
  is simulated inside that step and the episode ends.
  - Same physics, same reward.
  - Checked on 6 scripted deliveries: identical landing, reward and legality, with 24–26 decisions
    instead of 101–174.
  - It is not a reward change, and it also ends the shaping at release.
- **run13 pilot** = run11 + `--resolve-at-release`, 2M steps. 1.0x and 1.5x × 4 angles × seeds 0–1,
  16 runs (jobs 27336–27339). It runs at about 1,800 steps/s, so about 20 min per run.
  - If run13 keeps the zone where run11 lost it, this becomes the default and run11/run12 are
    explained.
- **run13 result: all 16 runs stalled.** By 25% of training, 100% of episodes time out with the arm
  hanging at about 105°. The task reward is −13, and not one ball is released.
  - Likely reason: with the episode ending at release, the shaping's whole drop to 0 (up to −36)
    now lands on the decision that triggers the release.
  - A timeout is treated as a cut-off, so SB3 bootstraps it with V and the stall escapes that drop.
    This is the same trap family as the earlier episodic-discounting stall.
  - So resolve-at-release can't simply be added on top of the current shaping.
- **run13b** (jobs 27374–27381): resolve at release plus one of two changes, 1.0x and 1.5x ×
  4 angles × 2 seeds each, 2M steps:
  - `noshape`: shaping off. With episodes now about 25 decisions long, the shaping may not be
    needed.
  - `toterm`: shaping on, with the timeout counted as a terminal failed delivery
    (`--timeout-terminal`).
- **run13b result: resolve-at-release removes the find-then-lose collapse.** Every run that found
  the good length kept it to the end, at 90–100% good. There was no drift in any of them.

  | variant | final policy bowls a good length | examples |
  |---|---|---|
  | `noshape` | **9 / 16** | 260° 1.5x: 98.5% good at 50.4 km/h; 240° 1.5x: 92.5% at 38.2 km/h |
  | `toterm` | 5 / 16 | 230° 1.0x and 240° 1.0x: 100% good; 260° 1.5x: 96% at 51.2 km/h |
  | run11 (for comparison) | 0 / 24 | |

  - What remains is exploration: the failed runs never learn to release at all.
  - Several `noshape` runs only found the zone in the last 25% of training and were still
    climbing, so run14 trains longer.
- Equifinality of three run13b policies over training:
  - The in-zone release window holds steady: 30–32 ms at 37–38 km/h (230–240°), but only 4 ms for
    the 50 km/h delivery at 260°.
  - The window does not widen over training. That's expected, since nothing in the simulator is
    noisy, and it's the baseline for the week-2 noise experiment. Faster means less
    timing-tolerant, as in Müller & Loosch.
- **run14** (jobs 27470–27473): resolve + no shaping, 6M steps, 1.0x and 1.5x × 4 angles, seeds 2–3.
- Plan change: the collapse fix took 1 day instead of the week, so the motor-noise work (week 2)
  can start earlier. The noise design still needs the user's review first.

### 24 Oct (early hours)
- **Before/after figure** for the supervisor: `results/run13b/collapse_fix.png`.
- **Motor noise implemented, off by default** (`--release-timing-sd-ms`, `--torque-noise-sd`).
  - Verified: the actual release shift follows the sampled jitter to within about 1 ms (the
    physics step is 2 ms). The torque noise SD is 0.150 × the command.
  - With noise off, the six showcase replays are unchanged.
  - **No training with noise yet:** `docs/noise_design.md` is still waiting for review.
- **Noise-robustness test (evaluation only, job 27526):** the 14 good run13b policies, which were
  trained without noise, scored with noise on, 200 deliveries each.
  See `results/run13b/noise_robustness.png`.

  | noise-free delivery | good, 12 ms timing jitter | landing SD under jitter |
  |---|---|---|
  | 37–39 km/h (230–240°) | 64–80% | 1.1–1.3 m |
  | 40–42 km/h | 33–47% | 1.8–2.2 m |
  | 50–53 km/h (260°) | 6–10% | 4.1–4.9 m |

  - Timing tolerance falls steeply with pace: the speed–accuracy trade-off from the equifinality
    papers, measured on our own policies.
  - Torque noise (15%) sorts them by budget instead:
    - 1.5x policies hold at 88–100%, because they have spare torque.
    - 1.0x policies drop to 2–76%. They swing at the torque limit, where noise can only weaken the
      swing.
  - So the noise-training experiment (run15) has a clear question: when trained under jitter, do
    the policies widen the window (equifinal), or slow down?
