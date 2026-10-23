# Motor-noise design (draft for review, 23 Oct 2026)

**Status: proposal. Nothing is implemented or trained yet.** It changes the task, so it needs the
user's and supervisor's sign-off first.

## Why

The supervisor's point (meeting 23 Oct): the learned action should depend on noise, and skilled
throwers use *equifinal* trajectories that tolerate it (Müller & Loosch 1999; Reiser et al. 2011).

Right now `ThrowEnv` has no noise at all. The reset is deterministic, the actuators are exact, and
the release fires at an exact angle. So the policy has no reason to become equifinal. That fits
what we measured:
- The in-zone release window stays flat over training: 30–32 ms at 37–38 km/h, and 4 ms at 50 km/h.
- In run11 the window shrank from 32 to 0 ms as the policy drifted.

## What to add (all off by default, one flag each)

1. **Release-timing jitter.** This is the noise the equifinality literature is about.
   - Each delivery samples a timing error δ ~ N(0, σ_t). The release fires δ after the arm reaches
     the set angle. If δ < 0, it fires early, once the predicted time to reach that angle is under
     |δ|.
   - Proposed σ_t = **12 ms**, inside Müller & Loosch's 10–15 ms release-time SD.
   - The policy does not observe δ.
   - Because δ is set in time, the angle error grows with arm speed, as it does in a real arm.
2. **Signal-dependent torque noise** (Harris & Wolpert 1998).
   - At each decision: torque = u · (1 + σ_s · ε), with ε ~ N(0, 1) per joint.
   - Proposed σ_s = **0.15**. A full-effort swing is noisier than a gentle one, so effort has a
     cost in accuracy.
3. (Optional, later) a small constant noise on the observed joint angles and velocities.

## Experiment (run15)

Base it on the run14 configuration (resolve at release, no shaping) and its result. Four conditions:
no noise / timing only / torque only / both. Each on 1.0x and 1.5x × 4 angles × 2 seeds.

Measured per checkpoint, with noise on:
- good % and release speed (km/h);
- equifinal window (ms) and the landing SD for release ±40 ms (`scripts/equifinality.py`);
- covariation of release parameters, Müller & Loosch's Δcovar: do release height, speed and angle
  co-vary so that landing stays put?
- functional variability: CV of timing vs CV of release speed, the Andrews et al. 2024 comparison.

## Predictions to test

- With timing noise, policies either **widen the equifinal window** (reshape the trajectory) or
  **give up pace** for tolerance (slower, lower release).
  - Widening would be the equifinality result.
  - Giving up pace would be a speed–accuracy trade-off.
- With torque noise, policies use less than full effort where they can.
- Noise-free policies, scored under noise, should do worse than noise-trained ones.

## Open questions for review

- **Is σ_t = 12 ms and σ_s = 0.15 sensible for fast bowling?** The 12 ms figure comes from darts.
- **Legality under timing noise.** The elbow-extension metric runs from arm-horizontal to release,
  so a late release lengthens that interval. Option A: measure legality at the real (noisy)
  release, which is stricter and realistic. Option B: measure it at the intended release.
  Option A is proposed.
- **Should evaluation always be with noise on?** Proposed: report both.
- The 2-DOF arm has little redundancy for equifinality to use. A wrist joint may be needed for a
  convincing result. That's a later step.
