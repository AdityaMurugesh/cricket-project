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
| 23 Oct (running) | run12 resume diagnosis: from run11 peak checkpoints, 4 arms (control / no shaping / stable lr + target_kl / frozen noise) | `logs/throw_ppo_run12{arm}_*` |
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
