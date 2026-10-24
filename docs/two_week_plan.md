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
| 23 Oct (running) | run12 resume diagnosis: 4 arms (control / no shaping / stable lr + target_kl / frozen noise) x 3 run11 peak checkpoints, 4M steps each (jobs 27049–27051, about 40 min) | `logs/throw_ppo_run12{arm}_*` |
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

### 24 Oct: training under noise, and the run-up
The user asked for two things, with the next supervisor meeting in about 2 weeks: train with the
noise now, and start learning the full run-up with the mocap as the run-up and trunk reference.

- `scripts/check_regressions.py` now runs before any sweep. It passes locally and on the HPC:
  - recorded deliveries replay identically through every code path;
  - resolve-at-release equals the full flight;
  - timeout handling is correct;
  - noise statistics, seeding and defaults are as specified;
  - a PPO smoke run with every flag trains and saves;
  - the physics bowler builds, scores 1 on its own reference, and falls under open-loop PD.
- **run15: training under motor noise** (`hpc/submit_run15.sh`, jobs from 27554). Setup: run13b
  no-shaping, 4M steps, 1.0x and 1.5x × 4 angles × seeds 0–1. Four conditions:
  - no noise
  - timing jitter only (12 ms)
  - torque noise only (0.15)
  - both

  16 jobs, submitted as queue slots free up. Analysis:
  - good % with and without noise;
  - equifinal window over training (`equifinality.py`);
  - noise-trained vs noise-free policies both scored under noise (`noise_robustness.py`,
    FILTER=released).
- **Run-up imitation, stage 1** (`envs/humanoid_bowler.py`, `envs/runup_imitation_env.py`,
  `hpc/train_runup.pbs`):
  - **Physics bowler:** the mocap-fitted Hamner skeleton (89 kg) with motors on every joint, using
    LocoMuJoCo skeleton_torque strengths. Feet collide with the floor, PD control. The root is
    unactuated, so balance has to be learned.
  - **Feasibility:**
    - open-loop PD on the reference falls after 0.3 s of the run-up, or about 1 s standing;
    - 650 decisions/s per core.
  - **Reward:** DeepMimic-style.
    - Tracks the pelvis path, legs, trunk (2× weight), head and front arm.
    - The bowling arm is neither tracked nor driven from the reference, per the legality rule.
  - **Training:** PPO, 16 envs, 60M steps, references T3a and T4b. Two jobs, queued behind run15.
  - The reference is joint angles only (`data/mocap/ref_<D>.npz`), copied to the HPC by hand.
    `data/mocap/` is now ignored whole in git.
- **Next stages:**
  - (2) Check the learned run-up visually against the reference and the camera-3 video, before
    trusting any score.
  - (3) Add the delivery: ball in the right hand, release, ICC legality from the simulated elbow,
    landing zone and pace. That is the full bowler.
  - (4) Noise on the full bowler.
- **SAC pilot result** (jobs 27045–27048, run with the *old* long episodes and shaping, 1.5M steps,
  about 10 h each):
  - 1 of 4 runs succeeded: 240° 1.5x seed 0 reached 100% good by 1.5M steps with no collapse
    (checkpoints: 48% → 96% → 100%).
  - The other 3 never found the zone.
  - PPO with resolve-at-release gets 9 of 16 in about 25 min per run.
  - Decision: stay with PPO. SAC was a hedge against the collapse, and the collapse now has a cause
    and a fix.
- Run-up T3a started (job 27778): 1,715 steps/s with 16 envs, so 60M steps take about 10 h.
  After 0.4M steps, episodes last about 52 decisions before a fall (open-loop PD: 10–45).
  T4b is queued.
- **run14 result** (resolve + no shaping, 6M steps, seeds 2–3, jobs 27470–27473): **10 / 16 final
  policies bowl a legal good length.** Every run that found the zone kept it.
  - 1.5x: 7 / 8. Several only found it after 3.5–5M steps, so the longer training mattered.
    - The 8th (230° s2) is 88% good with exploration noise, but its deterministic policy never
      releases.
    - Fastest: 260° s3 at **58.4 km/h**, 94% good under exploration noise.
  - 1.0x: 3 / 8. The two 260° runs can't reach: the scripted reach check gave 12 m at 1.0x only
    for 230–250°. The 230°/240° failures never learn to release (exploration).
  - Together with run13b, the remaining problem is getting the first releases at 1.0x, not
    keeping the zone.
- **Stage 3, the full bowler** (`envs/runup_bowl_env.py`), built after the user saw the stage-1
  viewer: "the solid simulation is bad, it's not even doing the full arm rotation". That's right: in
  stage 1 the bowling arm has no target at all (the legality rule keeps the mocap off the arm).
  - The arm now has the ThrowEnv job: ball at the right hand's grip, release at a set upper-arm
    angle, ICC legality from the simulated elbow, landing zone, pace.
  - Checks:
    - the captured arm reads 290° at release in ThrowEnv's convention;
    - not bowling never scores better than any release;
    - episodes start at least 1 s before release, so the swing is always the policy's own;
    - the arm length is the participant's (arm_scale 0.903).
  - Scripted over-the-top swing on the 5M stage-1 body: legal at 31 km/h but landing at 2.4–2.8 m.
    The weak body collapses and releases from 1.3 m, so the oracle is not yet conclusive.
  - Training: from scratch at release angles 250° and 270°, T3a, 60M steps (queued). A warm start
    from a strong stage-1 checkpoint follows later.

### 2026-10-24 (overnight, autonomous)
- **run15 finished** (training under motor noise; 64 runs = 4 conditions x 4 angles x 2 scales x
  2 seeds). Deterministic final deliveries, legal good length:
  - none: 1.5x 8/8, 1.0x 4/8
  - timing: 1.5x 5/8, 1.0x 3/8
  - torque: 1.5x 5/8, 1.0x 5/8
  - both: 1.5x 5/8, 1.0x 1/8
  - These evals run in each run's own (noisy) env, so one deterministic episode is one noise draw.
    The fair comparison is the noise-robustness scoring (200 episodes per condition), queued as
    jobs 28738–28741 with FILTER=released (results/run15_<cond>/noise_robustness/).
- **Stage 3, first results** (from scratch, jobs 28059 / 28073, at 30M of 60M steps):
  - Training episodes, legal and good: 250° 61%, 270° 47%, up from 0% at 7.5M steps. Falls dropped
    from 99% to 16–21%.
  - Deterministic replays (`scripts/eval_runup.py`, 8 start times): 250° 6/8 good at 45–54 km/h,
    270° 5/8 at 44–55 km/h. Released 1.5–1.9 m up; the arm turns 165–215° from hanging, over the
    top to release; elbow extension within ±18°, all legal.
  - **The loophole: it gave up the run-up.** The pelvis ends 1.5–4.5 m behind the captured path,
    and the pose score is 0.04–0.34 (stage 1 alone: 0.74). The fall test only checked height and
    tilt, so bowling from a near-standstill cost nothing but imitation reward.
  - Fix: `--max-pelvis-drift` (drifting that far from the captured pelvis path is a fall, so a no
    ball). Regression check added (section 8). The stage-1 35M policy, unchanged, reaches about
    1.0 m drift by 2–2.5 s, so 1.0 m is tight.
  - Launched (jobs 28742–28745): warm start from stage-1 35M at 250° and 270° with drift 1.5 m;
    warm 250° with drift 1.0 m; from scratch 250° with drift 1.5 m.
- Stage 1 (27778 T3a, 27880 T4b): at about 39M of 60M steps; pose 0.73–0.74.
- **run15 noise-robustness result** (jobs 28738–28741; the 45 final policies that release, each
  scored over 200 episodes per noise condition; `results/run15/noise_robustness/`, figure
  `results/run15/noise_training.png`). Matched pairs (same angle, budget, seed), each scored under
  the noise it was trained with versus the noise-free-trained twin:
  - Timing jitter: **+12.6 points good** on average (median +5.2), 6 better, 1 worse of 8.
  - Both noises: **+12.4** (median +12.5), 7 better, 0 worse of 7.
  - Torque noise: no gain (median 0, 1 better, 4 worse of 9). Torque noise barely hurts the
    noise-free 1.5x policies to begin with.
  - The biggest gains are on the fast, fragile deliveries (250°/260°): 7 → 35%, 27 → 64%,
    10 → 30% good under timing jitter. The already-robust ~37 km/h policies stay where they were.
  - Speed is about 1 km/h lower when trained with timing noise. That is the expected direction of the
    speed–tolerance trade-off, but small.
  - Caveats: 2 seeds per cell, only pairs where both runs release, and 1.0x is thin (1–3 pairs).
  - Next: equifinality windows of the noise-trained policies (do they widen the in-zone release
    window, or only aim at the zone centre?), and more seeds if the meeting wants a firm claim.
- **Equifinality explains the noise robustness** (`scripts/equifinality.py` on every run15
  checkpoint, `hpc/equifinality.pbs` jobs 28759/28774/28787; figure
  `results/run15/window_vs_noise.png`):
  - New measure `p_jitter_12ms`: from the *noise-free* deterministic delivery, the share of
    release instants within ±40 ms that land in the zone, weighted by N(0, 12 ms). It predicts the
    measured good % under 12 ms jitter for all 45 released final policies with **r = 0.98, mean
    error 4.5 points**.
  - The older "unbroken window" measure gave r = 0.95 for policies with a good noise-free
    delivery, but under-predicted several noise-trained ones (in-zone release instants need not
    be one stretch around the release).
  - Noise training does not widen the window across the board (median margins unchanged). At the
    fast angles it **flattens the landing curve at release**, |dLanding/dt| in m per 10 ms:
    260°: 4.5 → 2.5 (timing) and 4.5 → 3.0 (both); 250°: 1.7 → 1.5. At 230–240° the curve is already
    flat (~0.5) and nothing changes.
  - For the meeting: the policy's robustness to motor noise is a property of its noise-free
    trajectory, its equifinality (Müller & Loosch), and training under noise selects flatter
    trajectories where that matters. Caveat: legality is taken from the actual release, not each
    hypothetical one.
- **Drift-limited stage-3 runs at ~50M** (warm runs started from stage-1 35M, so +15M):
  - warm 250° drift 1.5 m: 31% good in training, pose 0.47 (from scratch without the limit: 61%
    good but pose 0.20). Deterministic at 45M: 3/8 starts bowl, 1/8 good, at 38–40 km/h,
    extension 5–8°. It releases about 0.25 s before the captured release.
  - warm 250° drift 1.0 m: 16% good and climbing fast (0.2% two million steps earlier).
  - warm 270° drift 1.5 m: about 10% released, no good yet; holding the ball is creeping up (13–18%).
  - from scratch 250° drift 1.5 m (15M): 7% good, but pose 0.18. The drift limit alone doesn't
    bring the run-up back, so the warm start does the work.
  - Stage 1 alone is nearly done: T3a 57M, pose 0.71; T4b 56M, pose 0.63.
  - Local viewer copies: `checkpoints/runup_bowl_a250_T3a_0_ckpts/step_45000000_steps.zip`
    (no run-up, fast) and `..._warm_d1.5_ckpts/step_45000000_steps.zip` (keeps the run-up).
- **Stage-3 at 60M (warm runs +25M), deterministic, 8 start times** (`scripts/eval_runup.py`,
  outcomes in `results/stage3/`):
  - warm 250° drift 1.5 m: **4/8 good, 5/8 released, 40–44 km/h**, extension 2–7°, pose 0.26–0.57
    (45M: 1/8 good). Training: 44% good at 60M. The 61M bucket dipped (45% released, falls 45%),
    so it's worth watching for a collapse.
  - warm 250° drift 1.0 m: **4/8 good, 37–39 km/h**, extension 8–9°, pose 0.25–0.55. Training 33% good.
  - warm 270° drift 1.5 m: 1/8 released (28 km/h, 4.3 m), none good. Training: 0% good, falls ~50%.
    270° doesn't come out of the warm start; it would need its own fix (curriculum from 250°?).
  - from scratch 250° drift 1.5 m (25M): **4/8 good, 7/8 released, 40–47 km/h**, but pose 0.05–0.30.
    It bowls more reliably than the warm runs but doesn't look like the captured bowler.
  - Every policy fell within 0.22 s when started at t = 0. Cause: the marker fit converges from its
    initial guess over the first ~4 frames (3 markers, residual 7 cm → 0.4 cm), so the reference
    velocity there is bogus (pelvis 3.8 m/s, 13 rad/s). Episodes now start at frame 5 or later
    (`FIRST_START_INDEX`, commit 5b0a935; reference times unchanged). That cost about 2% of training
    starts. Re-run with starts shifted 30–40 ms, warm d1.5 60M gives 3/8 good instead of 4/8, with
    landings near the zone edges (10.7–13.2 m), so the delivery is still sensitive to start time.
  - The warm runs end at 95M (35M + 60M) in about 6 h; the scratch d1.5 run ends at 60M.
  - Stage 1 finished at 60M: T3a pose 0.66, reaches release 68%; T4b pose 0.61, 47%.
    Final models are `checkpoints/runup_ppo_{T3a,T4b}_0.zip` (copied locally).
- **Stage-3 from scratch without the drift limit, final at 60M** (the loophole runs; starts from 0.05 s):
  250°: 6/8 good at 47–49 km/h; 270°: 6/8 good at 51–60 km/h. But pelvis error 1.1–4.4 m and pose
  0.04–0.54: they bowl well by leaving the captured run-up. That is the case for the drift limit, not a
  result. Finals copied to `checkpoints/runup_bowl_a{250,270}_T3a_0.zip`.
- **Correction (2026-10-24, after watching the replays): the stage-3 "good" counts above are not real
  deliveries.** The user watched them: no proper release, the bowler falls over and a trajectory appears.
  Measured at the release instant (8 starts, deterministic):
  - warm d1.5 65M: releases at 2.78–2.82 s, **0.3 s before the captured release (3.12 s)**, while
    toppling: pelvis tilted 35–52° from the mocap, 12–21 cm low, trunk lean 31–58° (mocap 17°). The
    bowling arm is **52–75° out of the bowling plane** (a sideways sling).
  - scratch d1.5 final: releases at the right time (3.08–3.12 s) with the arm in plane (4–20°), but
    crouched (pelvis 20 cm low, tilt 23–46°).
  - Line: median |lateral| of the "good" training deliveries is 1.0–1.3 m; replays miss by up to 4 m.
  - Causes, all in `envs/runup_bowl_env.py`:
    1. Release is automatic when the arm's x-z projection passes 250°. A sideways arm can cross that.
    2. The episode ends at release and the fall test comes after it, so a release mid-fall counts in
       full and nothing after it is simulated.
    3. "good" and the task reward ignore line.
    4. The run-up itself is a hunched shuffle (pose 0.4), not a run with a gather jump.
  - Fixes proposed to the user before any retraining.
- **Stage-3 release rules fixed** (`envs/runup_bowl_env.py`, regression checks in section 8):
  - Release window ±0.15 s around the captured release (was 0.6 s before to 0.3 s after).
  - A release is a no ball unless the bowler is in a bowling position: pelvis within 20° and 15 cm of
    the capture, and the upper arm within 50° of the bowling plane. The captured arm is at 41–44°
    as it passes 250–270°, leaning away; the toppling policies were at 52–75°.
  - Follow-through: the episode runs 0.3 s past release, tracking the captured follow-through. A fall
    there voids the delivery. The task reward is paid at its end.
  - Line: "good" needs |lateral| ≤ 0.5 m from the stump line, and the task distance is to the nearest
    point of the length × line box.
  - Re-scored under these rules, the old "good" policies give **0/8**: warm 65M falls, scratch final is
    "off balance at release" every time.
- **Stage 1 can't run the run-up either** (`checkpoints/runup_ppo_T3a_0.zip`, 10 starts): it falls
  0.5–1.5 s after every start before 2.4 s. It only reaches release from the last 0.7 s. The 68%
  "reached" in training came from those late starts.
  - Cause found in the reference (T3a): feet penetrate the floor in 51–63% of frames (up to 5.4 cm),
    and while in contact they slide at a median 1.1–1.4 m/s. A physics body can't follow a skating
    reference, which matches the 1–4 m root drift.
  - Options: clean the reference (lock stance feet, lift penetration), add a residual root force, or a
    kinematic run-up with only the bowling arm physical (the v1 design). Decision with the user.
- **Decision (user): kinematic run-up.** `envs/kinematic_bowl_env.py`, commit 2f2354c:
  - The whole body except the right arm follows the capture exactly. Each non-arm DoF has armature
    1e6 and force 1e6·q̈_ref, reset to the capture every 2 ms.
  - The arm feels the body's acceleration exactly: checked against the prescribed-motion solution,
    relative error 2e-6, and full arm torque leaves the body on the capture. This is unlike the v1
    weld/teleport, which distorted the arm.
  - Obs 21 (phase, arm angles/velocities, hand vs shoulder), action 7 (the arm's PD targets), reward
    only the delivery. PPO as in ThrowEnv (γ 0.9999, entropy 0.003, target_kl 0.03).
  - Release rules, legality from the simulated elbow, line and arm plane are all RunupBowlEnv's.
  - Runs 29243–29248: release angle 250 / 270 / 290° (the capture releases at 290°) × seeds 0, 1,
    30M steps each. Before believing any number: render the delivery and check the state at release.
- **Runs 29243–29248 cancelled at ~8M steps: two bugs, found by replaying the 5M checkpoint per start
  time** (commit 9a93b9f):
  - Arm angle on the wrong branch. Between 0.8 and 0.95 s before release, the captured arm is raised
    in front (unwrapped −40 to 42°). The tracker started it at 320–400°, already past the release
    angle, so it "released" the moment the window opened, with no arm circle. All the 2–3% "good"
    came from these starts; from the others (61%) the arm never circled ("still holding"). Starts now
    take the capture's unwrapped arm angle at that time.
  - Arm too damped to bowl. The skeleton's PD damping (0.2 × peak torque = 50 N·m·s/rad at the
    shoulder) caps the arm near 5 rad/s. Even PD-tracking the captured arm path only got to 124°.
    Arm damping is now peak torque / 40 rad/s; the captured path then bowls at 83 km/h. Random
    actions reach a release 4% (PD) / 10% (torque) of the time, instead of 0/150.
    0.3 (ThrowEnv's value) spun the light arm to NaN.
  - The same damping is in stage-3 RunupBowlEnv: its bowling arm was capped too.
  - New runs 29271–29279, 30M steps: `arm_control=torque` (ThrowEnv's action) at 250/270/290 × seeds
    0, 1, and `pd` at 250/270/290 × seed 0.
