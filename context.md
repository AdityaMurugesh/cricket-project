# Project context

## What this is
ACOD310 mini project, supervised by Dr Alap Kshirsagar.
Training a simulated humanoid to bowl a cricket ball using
reinforcement learning, with ICC legality as a hard constraint.

## Research question
On a fixed actuator budget, what is the maximum ball release
speed an RL policy can achieve subject to:
  1. ICC legality: elbow extension <= 15 degrees between the
     bowling arm reaching horizontal and ball release
  2. Landing in a target line-and-length zone

The deliverable is not a single speed number. It is the
speed / legality / accuracy tradeoff curve, reported across
two or three actuator budgets.

## Key design decision (settled, do not revisit)
The legality constraint is evaluated on the SIMULATED robot's
elbow joint, where it is exact. It is NOT derived from measured
human data.

Reason: the only published validation of markerless motion
capture for cricket bowling (Abraham, Feros & Fox 2025,
Int J Sports Sci Coach, DOI 10.1177/17479541251348081) reports
elbow flexion RMSE of 22.71 degrees against a 15 degree
threshold. The authors state it is unsuitable for legality
testing. Their system also produced a phantom elbow flexion
mid-delivery caused by torso occlusion, which the marker-based
reference did not record.

Confirmed directly by Aaron Fox (senior author) via email.

## Scope
v1: pure task-reward policy. Maximize release speed subject to
elbow constraint and target zone. NO imitation / style term.

Later, if reference motion is captured: add a style term and
compare. Reference motion would supply gross body style only
(trunk, lower limb) where markerless error is 8-13 degrees.
Never the elbow.

Stretch goal, gated on a working delivery by mid-semester:
a batting policy, to test whether a batter trained against a
scripted ball launcher transfers to a humanoid bowler.

## Stack
- MuJoCo 3.13.0, Python venv at ./cricket
- Windows, RTX 4060 laptop GPU (8GB VRAM)
- AWS access pending; assume local-only for now
- MJX for scaled training later if needed
- Isaac Lab / MimicKit considered but not in use: without
  reference motion there is no imitation layer to run

## Constraints on my workflow
- ~1 serious training run per day on this hardware.
  Reward design must be right before launch, not debugged
  through iteration.
- Keep episodes short. Start the robot at the crease in a
  delivery-stride pose. Do NOT simulate the full run-up in v1.
- Log elbow extension, release speed, and landing position on
  every episode from the very first run. These become the
  results section.

## ICC rules to encode
Soft penalty (continuous, in the reward):
- Elbow extension <= 15 deg between arm-horizontal and release.
  Measured as EXTENSION (change), not absolute bend. A
  permanently bent but rigid arm is legal.

Hard episode filters (binary, evaluated post-hoc, excluded
from speed statistics rather than penalized):
- Front foot: some part behind the popping crease at landing
- Back foot: within and not touching the return crease, checked
  at first ground contact only
- Ball bounces at most once, on the pitch
- Overarm, not underarm

Out of scope for v1: head-height rule, beamer rule.

## Open question, pending expert reply
Operational definition of elbow extension. Two ambiguities:
  1. How is the "arm horizontal" instant determined?
  2. Is the reported value the endpoint difference
     (angle_at_horizontal - angle_at_release) or the MAXIMUM
     extension anywhere across the window?
Implement BOTH, named clearly, and make it a config flag.
Emailed Dr Paul Felton (Nottingham Trent, ICC Suspect Bowling
Actions Panel) about this.

## Current task
Two tracks now exist side by side:

- `envs/throw_env.py` (ThrowEnv): the original minimal two-joint
  arm, ball welded to the end effector, actuator/torque-driven,
  a release action, target zone on the ground, reward for
  landing in it. Remains the environment actually used for RL
  training -- validates ball physics, release mechanism, scoring.
- `envs/bowler_env.py` (BowlerEnv) + `assets/bowler.xml`: a full
  humanoid (torso, two legs, the same bowling arm) that runs in
  and bowls. Added 2026-09-11 at the user's request, overriding
  "no humanoid, no run-up in v1" below for this piece only. Both
  the run-up and the bowling arm swing are SCRIPTED/KINEMATIC
  (qpos set directly every frame, no actuators) -- not RL-trained.
  Only the ball is dynamically simulated, launched at the hand's
  velocity (finite difference) at a scripted release instant.
  This is a visual demo, not a training environment; RL training
  still targets a torque-driven arm. See `scripts/demo_bowler.py`.

Rationale for kinematic-only: an actuator-driven arm on a
free-jointed root, held in place by weld-equality constraints,
was tried first and produced badly wrong throws (release speed
and direction both off) because the weld's finite compliance let
the arm's reaction torque perturb the root enough to corrupt the
throw. Scripting the whole body sidesteps this, reusing only
already-validated pieces (kinematic qpos puppeteering, weld-based
ball carrying, real ball free-flight physics).

## Conventions
- git commit before every training run, with the config
- Keep the elbow-extension function pure: takes joint angle
  time series, returns a number. No simulator dependency.