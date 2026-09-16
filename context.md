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

## Settled: what makes the action a bowling action
(2026-09-16, after run6.) Legality alone cannot produce a bowling
action, and this cost several runs to learn. A rigidly flexed arm has
zero elbow EXTENSION and is legal by construction -- which is correct
ICC, and stated above -- so a policy maximising speed subject to
legality + accuracy is free to shot-put. run6 did exactly that: released
at shoulder 207 deg with the elbow bent 113 deg, ball leaving from
behind the body at 1.6 m, slung forward and up at 28 deg.

Also settled, and worth not rediscovering: potential-based reward
shaping (throw_env_gym.py's swing_weight / straight_arm_weight) is
provably policy-INVARIANT (Ng, Harada & Russell 1999). It changes how
fast PPO finds the optimum, never which optimum it is. Three runs were
spent tuning swing_weight against a failure mode it mathematically
cannot affect. Anything the task must actually REQUIRE belongs in
ThrowEnv as a reward term or a hard gate. Shaping is a learning-speed
knob, nothing more.

So "Overarm, not underarm" -- already listed below as a required hard
episode filter -- is now implemented, as a gate on the action space
rather than a reward penalty, so it cannot be traded away against speed:
  - release_window_deg=(230, 310): the ball may only leave the hand with
    the arm up near vertical (270 = straight up, the true overarm
    release point). Reaching the window from the 100 deg start pose
    requires swinging up through 180, since the shoulder's lower joint
    limit is 95 -- so entering it also guarantees the arm-horizontal
    crossing the ICC metric is measured from. One-shot in practice: the
    upper joint limit is 380, so an arm that sails past 310 without
    releasing cannot come round again.
  - max_release_elbow_deg=40: near-straight arm at release.

max_release_elbow_deg is an "is this a bowling action at all" filter and
is explicitly NOT part of the legality metric, which stays exactly as
settled above. Both gates are config, so their cost in achievable speed
is reportable rather than hidden.

This makes the ICC metric bind for the first time. Under run6 every
episode was trivially legal because the arm was rigid; a scripted
attempt to reproduce that action now releases at 235 deg with 63.5 deg
of extension and is correctly scored illegal. The three failure modes
are now separated by three different mechanisms: sling-from-behind by
the release window, folded-arm shot-put by the elbow cap, and genuine
chucking by the extension metric.

Scripted reference points on the current actuator budget (gear 40/30),
measured 2026-09-16 -- the constrained optimum beats run6's learned
sling on every axis:
  - release at 270 deg, straight arm: 41.5 km/h, lands 7.82 m,
    extension -0.02 deg, release height 1.84 m, reward +2.15
  - release at 250 deg, straight arm: 38.9 km/h, lands 11.09 m (over)
  - release at 300 deg, straight arm: 46.2 km/h, lands 3.06 m (short)
  - run6's learned policy, for comparison: 31.5 km/h, lands 8.48 m
    (a miss), release height 1.60 m, reward -0.48
The speed/accuracy tension across that window is the tradeoff curve the
research question asks for, so it exists in the environment as built.

## On LocoMuJoCo
Checked 2026-09-16, before trying to use it to fix the bowling action.
It cannot help here, for two independent reasons:
  - Its datasets are locomotion and general movement only: walk, run,
    sprint, jumps, dance, fight, fallAndGetUp (LAFAN1), plus AMASS.
    There is no cricket bowling and no overarm throw of any kind.
  - It retargets onto full humanoid embodiments (UnitreeH1/G1, Atlas,
    Talos, MyoSkeleton...). RL here targets a 2-joint arm; there is
    nothing to retarget onto.
Even with ideal mocap it would not have applied: per the scope section
below, a style term supplies GROSS BODY style only (trunk, lower limb)
and never the arm -- and the arm swing is precisely what was broken.
LocoMuJoCo's role in this project is unchanged: a future style term for
the humanoid's trunk/legs, gated on reference motion being captured.

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