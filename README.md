# RL cricket bowling under the ICC elbow law

ACOD310 mini project (supervisor: Dr Alap Kshirsagar). A simulated bowler, built from a motion-captured
delivery, learns to bowl as fast as it can while staying **legal** (elbow extension at most 15° between the arm
reaching horizontal and release, measured on the simulated elbow) and **accurate** (a good length and line).
The deliverable is the speed / legality / accuracy trade-off, not one speed number.

The running project log is [`context.md`](context.md). The latest write-ups are in [`docs/`](docs/).

## How it works

| Piece | What it does | Code |
|---|---|---|
| Captured bowler | Qualisys motion capture of a real delivery, fitted to a 3D skeleton. The whole body replays the capture; only the bowling arm is simulated (MuJoCo). | `mocap/`, `scripts/fit_mocap3d.py`, `envs/humanoid_bowler.py` |
| Bowling environment | The arm's 7 motors are driven every 20 ms. The ball leaves at a fixed arm angle; legality, length, line and speed are scored. v3 adds air drag on the ball and a hard elbow stop. | `envs/kinematic_bowl_env.py`, `envs/kinematic_bowl_env2.py`, `envs/legality.py`, `envs/ballistics.py` |
| Reward | Good ball: 1 + 0.5 × speed (m/s); a miss: minus the distance; illegal: a further −2; all × 10. Optionally the exact expectation over release-timing error. | `envs/runup_bowl_env.py` (`_task_reward`), `KinematicBowlEnv2._expected` |
| CMA-ES trajectory optimiser | The fastest legal, on-length delivery the arm can bowl, for a required accuracy under timing error (the ceiling for RL). | `scripts/kin_frontier.py` |
| CMA-ES → PPO | Behaviour cloning copies an optimised delivery into PPO's network (a warm start, not a reward term); PPO then fine-tunes it with the task reward. | `scripts/bc_from_frontier.py`, `scripts/train_runup.py --resume-from` |
| RL training | PPO (Stable-Baselines3), on the university HPC (PBS). | `scripts/train_runup.py`, `hpc/` |
| Scoring | Each delivery's landing curve over release-timing error: exact good rate at any timing skill. | `scripts/kin_equifinality.py`, `scripts/curves_at_sd.py` |
| CMA-ES alone vs CMA-ES + RL | Plans and policies bowled in the same model under the same noise draws. | `scripts/compare_cmaes_rl.py` |
| Arm limits | Human-like and real-robot (Unitree G1, H1) actuator and joint limits. | `envs/arm_limits.py` |
| Leg clean-up | Feet planted at the contacts seen in the lab video (the legs were not tracked after the jump). | `scripts/fix_legs.py` |
| Constrained RL | PPO-Lagrangian: the elbow law as a cost with a target illegal rate, its price λ adjusted during training. | `KinematicBowlEnv2(lagrangian=True)`, `scripts/train_runup.py --lagrangian` |
| Body vs robot | The captured legs and trunk against the Unitree G1/H1 joint ranges and speed limits. | `scripts/body_vs_robot.py` |
| Humanoid robot (Unitree G1) | The whole bowler as a G1 (MuJoCo Menagerie, 29 DoF): the capture retargeted onto it by per-frame IK, its own arm and motors; CMA-ES, BC + PPO and scoring run on it with `--robot g1`. | `assets/unitree_g1/`, `scripts/retarget_g1.py`, `envs/g1_bowl_env.py`, `scripts/g1_feasibility.py` |
| Dynamic whole-body G1 | The whole G1 physically simulated (floor contact, Unitree's servos and torque limits) tracking the delivery; a foot-pinned, Froude-scaled retarget (`--feet --froude`). | `scripts/g1_dynamic.py` |
| Whole-body tracking RL | DeepMimic-style PPO controller that keeps the dynamic G1 up through the delivery (corrections on the servo targets). | `envs/g1_track_env.py`, `scripts/train_g1_track.py`, `scripts/render_g1_track.py` |

A one-slide summary of the CMA-ES → PPO link: [`docs/slide_cmaes_to_ppo.png`](docs/slide_cmaes_to_ppo.png).

## Results

- [`docs/meeting_2026-11-27.md`](docs/meeting_2026-11-27.md): the Unitree G1 humanoid: its frontier, CMA-ES vs RL under the same noise on the robot (`results/kinematic/cmaes_vs_rl/g1_cmaes_vs_rl.png`), and the first dynamic whole-body test (`results/kinematic/g1/dynamic_T3aLf.png`). Plain-language guide to the whole project: [`docs/study_guide.pdf`](docs/study_guide.pdf).
- [`docs/progress_since_20nov.pdf`](docs/progress_since_20nov.pdf): the 20 Nov meeting's actions (CMA-ES vs RL under the same noise, legs, arm limits, constrained RL).
- [`docs/progress_since_6nov.pdf`](docs/progress_since_6nov.pdf): what was built and found from 6 to 20 Nov (a short version: [`docs/progress_simple.pdf`](docs/progress_simple.pdf)).
- `results/kinematic/frontier/`: CMA-ES frontiers (speed against accuracy per timing skill), e.g. `speed_vs_accuracy.png`.
- `results/kinematic/equifinality/`: every RL bowler's landing curves and scores (`curves_at_sd.json`: good rate at 1-12 ms timing error).
- `results/kinematic/cmaes_vs_rl/`, `results/kinematic/legfix/`: the same-noise comparison and the leg clean-up.
- Meeting notes: `docs/meeting_*.md`. Plan and build log: `docs/roadmap_2026-11-18.md`.

## Running it

```
python -m venv cricket && cricket/Scripts/pip install mujoco gymnasium stable-baselines3 cma scipy matplotlib
python scripts/kin_frontier.py --angle 270 --handover 0.5 --knot-dt 0.02 --diagonal --p-cont --drag --stiff-elbow \
    --jitter-sd-ms 4 --tau 0 0.5 0.75 --warm-frontier <json>:<tau> --out results/kinematic/frontier
python scripts/train_runup.py --task kinematic2 --drag --stiff-elbow --release-angle 270 --arm-handover 0.5 ...
```

Training runs go to the HPC as packs (`hpc/packs/*.txt`, submitted with `hpc/*.pbs`).

## Data

The motion capture (raw `.c3d`, the camera video, and everything fitted from it, `data/mocap/`) is participant
data and is **not** in this public repository. Scripts that need the reference motion expect
`data/mocap/ref_<delivery>.npz`.
