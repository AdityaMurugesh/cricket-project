"""Regression checks for the throwing env: run before launching any sweep.

Exits non-zero if anything fails. No torch needed unless --train is given.
  1. the six recorded run10 deliveries replay to their recorded outcome, both
     physics-step by physics-step (ThrowEnv) and through ThrowEnvGym with and
     without --resolve-at-release
  2. resolve-at-release gives the same outcome and reward as the full flight
  3. --timeout-terminal makes a stall terminal, and is a truncation without it
  4. release-timing noise: sampled SD and mean as asked; the release really
     moves by the sampled delta; same seed -> same delta
  5. torque noise: applied SD = sd x command; off means exactly the command
  6. (--train) a short PPO run with every flag on trains and saves
  7. physics bowler (run-up imitation): the model builds with every joint
     actuated; reward is 1 on the reference; open-loop PD falls (skipped
     without data/mocap/ref_T3a.npz)
  8. full bowler: captured arm angle at release, no ball never beats a release,
     episodes start >= 1 s before release, the pelvis drift limit

    python scripts/check_regressions.py          # ~1 min
    python scripts/check_regressions.py --train  # + PPO smoke run
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from envs.throw_env import ThrowEnv
from envs.throw_env_gym import ThrowEnvGym
from scripts.showcase import ORDER, load, replay

ROOT = Path(__file__).resolve().parent.parent
FAILS = []


def check(ok, what):
    print(("  ok    " if ok else "  FAIL  ") + what)
    if not ok:
        FAILS.append(what)


def close(a, b, tol):
    if a is None or b is None:
        return a is b
    return abs(a - b) <= tol


def gym_replay(rec, resolve):
    env = ThrowEnvGym(speed_weight=0.5, release_mode="fixed_angle", release_angle_deg=rec["angle"],
                      actuator_scale=rec["scale"], resolve_at_release=resolve)
    assert env.frame_skip == rec["frame_skip"]
    env.reset()
    for i in range(10_000):
        a = rec["actions"][i] if i < len(rec["actions"]) else rec["actions"][-1]
        _, _, te, tr, info = env.step(a)
        if te or tr:
            return info


def outcome(info):
    land = info.get("landing_pos")
    return (info["release_speed"] * 3.6 if info.get("release_speed") else None,
            land[0] if land else None, bool(info.get("legal")))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", action="store_true")
    a = p.parse_args()
    replay_dir = ROOT / "results" / "replays"

    print("1. recorded deliveries replay to their recorded outcome")
    for name in ORDER:
        if not (replay_dir / f"{name}.npz").exists():
            continue
        rec = load(name, replay_dir)
        want = rec["recorded"]
        for label, info in [("ThrowEnv", replay(rec)), ("gym", gym_replay(rec, False)),
                            ("gym+resolve", gym_replay(rec, True))]:
            sp, land, legal = outcome(info)
            check(close(sp, want["speed_kmh"], 0.01) and close(land, want["land_x"], 0.001)
                  and legal == want["legal"],
                  f"{name} [{label}] {sp:.2f} km/h {land:.3f} m legal={legal}")

    print("2. resolve-at-release = full flight (scripted swings)")
    rng = np.random.default_rng(0)
    for k in range(6):
        base = np.array([1.0, -0.2]) + 0.3 * rng.standard_normal((400, 2))
        res = []
        for resolve in (False, True):
            env = ThrowEnvGym(speed_weight=0.5, release_mode="fixed_angle", release_angle_deg=240 + 5 * k,
                              actuator_scale=1.5, target_range=(10.9, 12.9), resolve_at_release=resolve)
            env.reset(seed=0)
            n, held = 0, None
            while True:
                act = held if held is not None else np.clip(base[n], -1, 1)
                _, _, te, tr, info = env.step(act)
                n += 1
                if env._env.released and held is None:
                    held = act
                if te or tr:
                    break
            res.append((info["episode_reward"], outcome(info)))
        check(res[0] == res[1], f"angle {240 + 5 * k}: reward {res[0][0]:+.3f} both ways")

    print("3. timeout handling")
    for flag in (False, True):
        env = ThrowEnvGym(release_mode="fixed_angle", release_angle_deg=240, timeout_terminal=flag)
        env.reset(seed=0)
        while True:
            _, _, te, tr, info = env.step(np.zeros(2))
            if te or tr:
                break
        check(info["timeout"] and te == flag and tr == (not flag),
              f"stall with timeout_terminal={flag}: terminated={te} truncated={tr}")

    print("4. release-timing noise")
    act = np.array([1.0, -0.1])

    def release_step(env, seed):
        env.reset(seed=seed)
        n = 0
        while not env.released and not env.spent and n < 1200:
            env.step(act)
            n += 1
        return n

    clean = ThrowEnv(release_mode="fixed_angle", release_angle_deg=245, actuator_scale=1.5)
    r0 = release_step(clean, 0)
    noisy = ThrowEnv(release_mode="fixed_angle", release_angle_deg=245, actuator_scale=1.5,
                     release_timing_sd_ms=12.0)
    got, want = [], []
    for s in range(1000):
        r = release_step(noisy, s)
        if not noisy.spent:
            got.append((r - r0) * 2.0)
            want.append(noisy.release_delay_ms)
    got, want = np.array(got), np.array(want)
    check(abs(want.std() - 12.0) < 1.0 and abs(want.mean()) < 1.5,
          f"sampled delta: mean {want.mean():+.2f} ms, SD {want.std():.2f} ms (asked 12)")
    check(np.abs(got - want).mean() < 1.5, f"realised shift tracks delta: mean error {np.abs(got - want).mean():.2f} ms")
    noisy.reset(seed=7)
    d1 = noisy.release_delay_ms
    noisy.reset(seed=7)
    check(noisy.release_delay_ms == d1, "same seed -> same delta")
    check(ThrowEnv(release_mode="fixed_angle", release_angle_deg=245).release_delay_ms == 0.0,
          "noise off by default")

    print("5. torque noise")
    for sd in (0.0, 0.15):
        g = ThrowEnvGym(release_mode="fixed_angle", release_angle_deg=245, torque_noise_sd=sd)
        g.reset(seed=3)
        c = []
        for _ in range(2000):
            g._env.reset()
            g.step(np.array([0.5, 0.5]))
            c.append(g._env.data.ctrl[0])
        c = np.array(c)
        check(abs(c.std() - sd * 0.5) < 0.01 and abs(c.mean() - 0.5) < 0.01,
              f"sd={sd}: command 0.5 -> applied mean {c.mean():.3f}, SD {c.std():.3f}")

    if a.train:
        print("6. PPO smoke run with every flag")
        with tempfile.TemporaryDirectory() as tmp:
            cmd = [sys.executable, str(ROOT / "scripts" / "train_throw.py"), "--timesteps", "16384",
                   "--n-envs", "8", "--speed-weight", "0.5", "--release-mode", "fixed_angle",
                   "--release-angle", "240", "--actuator-scale", "1.5", "--target-range", "10.9", "12.9",
                   "--checkpoint-every", "8192", "--resolve-at-release", "--no-shaping",
                   "--release-timing-sd-ms", "12", "--torque-noise-sd", "0.15",
                   "--log-dir", tmp, "--model-path", str(Path(tmp) / "m")]
            r = subprocess.run(cmd, capture_output=True, text=True)
            check(r.returncode == 0 and (Path(tmp) / "m.zip").exists()
                  and (Path(tmp) / "episodes.csv").stat().st_size > 0,
                  f"train_throw.py exit {r.returncode}, model + episode log written")
            if r.returncode:
                print(r.stderr[-1500:])

    print("7. physics bowler")
    from envs.humanoid_bowler import MOCAP_DIR, build_model
    hm = build_model()
    check(hm.nu == 34 and hm.nq == 47 and abs(hm.body_mass.sum() - 89.3) < 0.5,
          f"model: nq {hm.nq}, {hm.nu} motors, {hm.body_mass.sum():.1f} kg")
    if (MOCAP_DIR / "ref_T3a.npz").exists():
        from envs.runup_imitation_env import FIRST_START_INDEX, RunupImitationEnv
        env = RunupImitationEnv(rsi=False, seed=0)
        env.reset(seed=0)
        r0 = env._reward(env.d.qpos.copy(), env.ref["qvel"][FIRST_START_INDEX])[0]
        check(abs(r0 - 1.0) < 1e-6, f"reward is 1 on the reference itself ({r0:.6f})")
        v0, vs = (np.linalg.norm(env.ref["qvel"][i, :3]) for i in (0, FIRST_START_INDEX))
        check(vs < 1.0 < v0, f"episodes start after the fit transient (pelvis {v0:.1f} -> {vs:.1f} m/s)")
        n = 0
        while True:
            _, r, te, tr, info = env.step(np.zeros(env.n_act))
            n += 1
            if te or tr:
                break
        check(info["fell"] and 5 < n < 100, f"open-loop PD falls, as it must ({n} steps)")
        print("8. full bowler (run-up + bowling arm)")
        import mujoco
        from envs.runup_bowl_env import RunupBowlEnv
        be = RunupBowlEnv(rsi=False)
        ref = be.ref
        be.reset(seed=0)
        be._arm_hist, be._elbow_hist = [], []
        for i in range(ref["release_index"] + 1):
            be.d.qpos[:] = ref["qpos"][i]
            mujoco.mj_kinematics(be.m, be.d)
            if be._arm_hist:
                be._track_arm()
            else:
                be.start_arm_tracking()
        a_rel = be._arm_hist[-1]
        check(260 <= a_rel <= 320, f"captured bowling arm at release: {a_rel:.0f} deg (ThrowEnv convention; ~300 measured)")
        be._outcome = {"speed_m_s": 20.0}
        held = be._task_reward(None, False)
        worst = min(be._task_reward(x, lg) for x in (-30.0, 0.0, 5.0, 30.0, 60.0) for lg in (True, False))
        check(held <= worst, f"no ball ({held:.1f}) never beats a release (worst {worst:.1f})")
        be.reset(seed=0)
        R = 0.0
        while True:
            _, r, te, tr, info = be.step(np.zeros(be.n_act))
            R += r
            if te or tr:
                break
        o = info["outcome"]
        check(not o["released"] and abs(info["task_reward"] - held) < 1e-9,
              f"open-loop: no ball ({o['why']}), scored {info['task_reward']:.1f} x {be.task_weight:g}")
        check(be.rsi_last_index / ref["rate"] <= be.t_ref_release - 0.99,
              f"episodes start >= 1 s before the captured release ({be.rsi_last_index / ref['rate']:.2f} s)")
        # falling metres behind the captured run-up must end the episode as a no ball
        for drift_limit, want in ((1.0, "left the run-up"), (None, None)):
            de = RunupBowlEnv(rsi=False, max_pelvis_drift_m=drift_limit)
            de.reset(seed=0)
            de.d.qpos[0] -= 1.5
            mujoco.mj_forward(de.m, de.d)
            _, r, te, tr, info = de.step(np.zeros(de.n_act))
            why = info["outcome"]["why"] if te else None
            check(why == want and (not te or abs(info["task_reward"] - held) < 1e-9),
                  f"drift limit {drift_limit}: 1.5 m behind -> {why or 'carries on'} ({info['drift_m']:.2f} m)")
    else:
        print("  skip  no data/mocap/ref_T3a.npz here")

    print(f"\n{'ALL PASSED' if not FAILS else f'{len(FAILS)} FAILED'}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
