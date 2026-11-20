"""The pace-vs-timing-tolerance frontier of the kinematic bowler, by trajectory optimisation (no RL).

Every PPO batch so far asked "what does PPO find with this knob?". This asks what the
arm can physically do: for a required timing tolerance tau, the fastest delivery that
is legal, lands a good length and line, and stays good under 12 ms release jitter with
probability >= tau. If PPO's bowlers sit on this curve, the 45-60 km/h of the tolerant
ones is the physics of this arm, not a training failure; if they sit well below it, the
training is the bottleneck.

Same simulator and the same scoring as the RL results: KinematicBowlEnv with torque
control from 1.0 s before the captured release, and kin_equifinality.delivery() for
the outcome and p_jitter_12ms (exact under N(0, 12 ms) release jitter, from the
noise-free landing curve). The controller is open loop: each of the arm's 7 torque
commands is piecewise linear between knots (coarse during the run-up, every 40 ms
through the arm circle), held for each 20 ms decision as PPO's would be. With the body
prescribed and no noise, one start's open-loop optimum is the best any policy can do
from that start, so the curve is an upper bound for that start.

Optimiser: CMA-ES (pip package `cma`), warm-started from a trained PPO policy's own
action sequence (least-squares fitted onto the knots), in parallel worker processes.
Fitness (minimised), lexicographic: any delivery meeting every constraint beats any
that misses one.
  feasible:   -speed_kmh
  infeasible: 100 + 30 * violation - 0.1 * speed_kmh, violation = zone distance (m)
              + (elbow extension over 15 deg) / 5 + 10 * max(0, tau - p_jitter)
  no release: 400

    cricket/Scripts/python.exe scripts/kin_frontier.py --angle 270 --tau 0 0.5 0.75 \
        --warm kin_bowl_torque_a270_T3a_0 kin_bowl_torque_b4096_t12_a270_T3a_0 --out results/kinematic/frontier
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BEFORE_S = 1.0
# knot times relative to the captured release (s): the run-up, then every 40 ms through the circle
KNOTS = np.concatenate([np.linspace(-1.0, -0.5, 5), np.linspace(-0.44, 0.16, 16)])
# with --handover h the arm is the capture's until h before release (KinematicBowlEnv.arm_handover_s),
# so the knots only need to cover the rest: every ~40 ms
HANDOVER_KNOTS = lambda h: np.linspace(-h, 0.16, int(round((h + 0.16) / 0.04)) + 1)

_env = None


class OpenLoop:
    """A 'model' for kin_equifinality.delivery(): returns the planned action for the env's current decision."""

    def __init__(self, env, table):
        self.env, self.table = env, table

    def predict(self, obs, deterministic=True):
        k = int(round((self.env.t - self.env._start_t) / self.env.dt))
        return self.table[min(k, len(self.table) - 1)], None


def decision_times(env):
    """Each decision's start, relative to the captured release, for a start BEFORE_S ahead of it."""
    i0 = int(round(env.ref["release_index"] - BEFORE_S * env.ref["rate"]))
    start = i0 / env.ref["rate"]
    n = int(np.ceil((env.t_end - start) / env.dt)) + 1
    return start + np.arange(n) * env.dt - env.t_ref_release


def interp_matrix(t):
    """M (len(t) x len(KNOTS)): actions = M @ knot_values, piecewise linear, flat outside the knots."""
    M = np.zeros((len(t), len(KNOTS)))
    for i, ti in enumerate(t):
        j = np.clip(np.searchsorted(KNOTS, ti) - 1, 0, len(KNOTS) - 2)
        w = np.clip((ti - KNOTS[j]) / (KNOTS[j + 1] - KNOTS[j]), 0.0, 1.0)
        M[i, j], M[i, j + 1] = 1.0 - w, w
    return M


ENV2 = {}      # --torque-scale / --actuator / --omega-max / --drag / --stiff-elbow: KinematicBowlEnv2
DELIVERY = "T3a"   # --delivery: the captured body the arm rides on


def make_env(angle, target_range, handover=None, env2=None):
    env2 = ENV2 if env2 is None else env2
    kw = dict(delivery=DELIVERY, rsi=False, release_angle_deg=angle, target_range=target_range, arm_handover_s=handover)
    if env2:
        from envs.kinematic_bowl_env2 import KinematicBowlEnv2
        return KinematicBowlEnv2(**kw, **env2)
    from envs.kinematic_bowl_env import KinematicBowlEnv
    return KinematicBowlEnv(arm_control="torque", **kw)


def set_knots(handover, knot_dt=None):
    """knot_dt = 0.02 puts a knot on every decision, so a PPO policy's actions are copied exactly."""
    global KNOTS
    if knot_dt is not None:
        first = -handover if handover is not None else -BEFORE_S
        KNOTS = np.round(first + np.arange(int(round((0.16 - first) / knot_dt)) + 1) * knot_dt, 6)
    elif handover is not None:
        KNOTS = HANDOVER_KNOTS(handover)


P_CONT = False     # True: tau refers to the continuous (interpolated) good rate, not the 2 ms grid's


def _init(angle, target_range, handover=None, knot_dt=None, env2=None, jitter_sd_ms=12.0, max_ext_deg=15.0,
          p_cont=False, delivery="T3a", torque_noise=(0.0, 0)):
    global _env, _M, ENV2, P_CONT, DELIVERY, TORQUE_NOISE
    P_CONT = p_cont
    DELIVERY = delivery
    TORQUE_NOISE = tuple(torque_noise)
    import scripts.kin_equifinality as ke
    ke.JITTER_SD_S = jitter_sd_ms / 1000.0     # "p_jitter_12ms" is then the good rate under N(0, this)
    import envs.legality as legality
    legality.LEGAL_EXTENSION_DEG = max_ext_deg     # the law itself (15 = ICC); larger = what a throw buys
    ENV2 = env2 or {}
    set_knots(handover, knot_dt)
    _env = make_env(angle, target_range, handover)
    _M = interp_matrix(decision_times(_env))


TORQUE_NOISE = (0.0, 0)   # --torque-noise-sd, --noise-seeds: tau then refers to the good rate under motor noise too


def evaluate(x):
    """x: knot values (len(KNOTS) x 7, flattened) -> the delivery's metrics.

    With --torque-noise-sd s the plan is also bowled with signal-dependent motor noise (each decision's command
    u -> u (1 + s N(0, 1)), the env's torque_noise_sd) for seeds 1..N, the same N noise draws for every
    candidate (common random numbers). p_jitter_12ms is then the mean good rate under motor noise AND release
    timing (a ball that is not released counts 0); the noise-free delivery's own rate is p_jitter_nominal."""
    from scripts.kin_equifinality import delivery
    table = np.clip(_M @ x.reshape(len(KNOTS), -1), -1.0, 1.0)
    _env.torque_noise_sd = 0.0
    try:
        r, curve = delivery(_env, OpenLoop(_env, table), BEFORE_S, 0)
    except AssertionError as e:       # the scorer's own consistency checks: count it as no delivery
        return {"released": False, "why": f"scorer check: {e}"}
    out = {k: r.get(k) for k in ("released", "why", "speed_m_s", "land_x", "lateral_m", "extension_deg", "legal",
                                 "good", "p_jitter_12ms", "p_jitter_cont", "window_ms", "slope_m_per_10ms",
                                 "elevation_deg")}
    if P_CONT and out.get("p_jitter_cont") is not None:
        out["p_jitter_grid"], out["p_jitter_12ms"] = out["p_jitter_12ms"], out["p_jitter_cont"]
    sd, n = TORQUE_NOISE
    if sd > 0 and out["released"]:
        key = "p_jitter_cont" if P_CONT else "p_jitter_12ms"
        ps, speeds = [], []
        _env.torque_noise_sd = sd
        try:
            for s in range(1, n + 1):
                try:
                    rn, _ = delivery(_env, OpenLoop(_env, table), BEFORE_S, s)
                except AssertionError:
                    rn = {"released": False}
                ps.append(float(rn.get(key) or 0.0) if rn.get("released") else 0.0)
                if rn.get("released"):
                    speeds.append(rn["speed_m_s"] * 3.6)
        finally:
            _env.torque_noise_sd = 0.0
        out["p_jitter_nominal"] = out["p_jitter_12ms"]
        out["p_jitter_12ms"] = float(np.mean(ps))
        out["noisy_mean_kmh"] = float(np.mean(speeds)) if speeds else None
        out["noisy_released"] = len(speeds)
    return out


def LEGAL():
    import envs.legality as legality
    return legality.LEGAL_EXTENSION_DEG


def violation(r, tau, target_range):
    lo, hi = target_range
    x, y = r["land_x"], r["lateral_m"]
    dx = 0.0 if lo <= x <= hi else min(abs(x - lo), abs(x - hi))
    dy = max(abs(y) - 0.5, 0.0)
    ext = r["extension_deg"] if r["extension_deg"] is not None else 90.0
    p = r.get("p_jitter_12ms") or 0.0
    return float(np.hypot(dx, dy) + max(0.0, ext - LEGAL()) / 5.0 + 10.0 * max(0.0, tau - p))


def fitness(r, tau, target_range):
    if not r["released"]:
        return 400.0
    kmh = r["speed_m_s"] * 3.6
    v = violation(r, tau, target_range)
    if v == 0.0 and r["good"]:
        return -kmh
    return 100.0 + 30.0 * max(v, 1e-3) - 0.1 * kmh


def warm_start(run, angle, target_range, handover=None):
    """A PPO policy's deterministic actions from BEFORE_S, fitted onto the knots."""
    from stable_baselines3 import PPO
    model = PPO.load(ROOT / "checkpoints" / f"{run}.zip", device="cpu")
    # the policy's own env: budget 1 (budget frontiers start from budget-1 policies), same actuator model
    env = make_env(angle, target_range, handover, env2={k: v for k, v in ENV2.items() if k != "torque_scale"})
    obs, _ = env.reset(options={"before_s": BEFORE_S})
    acts = []
    while True:
        a, _ = model.predict(obs, deterministic=True)
        acts.append(np.clip(a, -1, 1))
        obs, _, te, tr, _ = env.step(a)
        if te or tr:
            break
    t = decision_times(env)
    A = np.array(acts)
    # the episode ends at the release: hold the last action after it
    A = np.vstack([A, np.repeat(A[-1:], len(t) - len(A), axis=0)])
    knots, *_ = np.linalg.lstsq(interp_matrix(t), A, rcond=None)
    # another actuator budget: ask for the same torque the policy did (saturating when the budget is smaller)
    knots = knots / ENV2.get("torque_scale", 1.0)
    return np.clip(knots, -1.0, 1.0).ravel()


def optimise(pool, x0, tau, target_range, sigma, gens, popsize, seed, log, diagonal=False):
    import cma
    es = cma.CMAEvolutionStrategy(x0, sigma, {"bounds": [-1.0, 1.0], "popsize": popsize, "seed": seed,
                                              "verbose": -9, "maxiter": gens,
                                              **({"CMA_diagonal": True} if diagonal else {})})
    # elitist: the start itself counts, so a large sigma can never return something worse than where it began
    # (job 38912: sigma 0.25 from the 127.9 km/h optimum returned 120.3)
    r0 = pool.map(evaluate, [np.asarray(x0, dtype=float)])[0]
    best = (fitness(r0, tau, target_range), np.asarray(x0, dtype=float), r0)
    t0 = time.time()
    while not es.stop():
        X = es.ask()
        R = pool.map(evaluate, X)
        F = [fitness(r, tau, target_range) for r in R]
        es.tell(X, F)
        i = int(np.argmin(F))
        if F[i] < best[0]:
            best = (F[i], np.array(X[i]), R[i])
        if es.countiter % 25 == 0:
            r = best[2]
            log(f"    gen {es.countiter:4d}  best f {best[0]:8.2f}  {(r.get('speed_m_s') or 0) * 3.6:5.1f} km/h  "
                f"good {r.get('good')}  p_jit {r.get('p_jitter_12ms') or 0:.2f}  ({time.time() - t0:.0f} s)")
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--angle", type=float, required=True, help="release angle (deg), as the RL runs")
    p.add_argument("--tau", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 0.9])
    p.add_argument("--warm", nargs="*", default=[], help="PPO runs (checkpoints/<run>.zip) to start from")
    p.add_argument("--knot-dt", type=float, default=None, help="uniform knots this far apart (0.02 = every decision)")
    p.add_argument("--diagonal", action="store_true", help="separable CMA (for many knots)")
    p.add_argument("--handover", type=float, default=None, help="arm_handover_s (s before release); tag _h<ms>")
    p.add_argument("--target-range", type=float, nargs=2, default=[10.9, 12.9])
    p.add_argument("--gens", type=int, default=300)
    p.add_argument("--popsize", type=int, default=28)
    p.add_argument("--sigma", type=float, default=0.15)
    p.add_argument("--workers", type=int, default=14)
    p.add_argument("--torque-scale", type=float, default=1.0, help="actuator budget (x the arm's gears); tag _q<s>")
    p.add_argument("--actuator", default="damping", choices=["damping", "hill"], help="hill: Hill force-velocity")
    p.add_argument("--omega-max", type=float, default=40.0, help="Hill curve's zero-torque joint speed (rad/s)")
    p.add_argument("--jitter-sd-ms", type=float, default=12.0,
                   help="release-timing SD that tau refers to (the bowler's skill); tag _j<ms>. The records keep "
                        "the key p_jitter_12ms for the good rate under it")
    p.add_argument("--max-extension-deg", type=float, default=15.0,
                   help="elbow-extension limit counted as legal (15 = ICC; 1000 = no law); tag _x<deg>")
    p.add_argument("--p-cont", action="store_true",
                   help="tau refers to the good rate with the landing curve interpolated between 2 ms physics "
                        "samples (needed for timing SD <= ~4 ms); tag _pc")
    p.add_argument("--warm-frontier", nargs="*", default=[],
                   help="also start from frontier optima: <json>:<tau> (same knots, angle and handover)")
    p.add_argument("--drag", action="store_true", help="air drag on the ball (envs/ballistics.py); tag _drag")
    p.add_argument("--stiff-elbow", action="store_true",
                   help="hard elbow hyperextension stop (KinematicBowlEnv elbow_solref 0.004); tag _se")
    p.add_argument("--delivery", default="T3a", help="captured body (data/mocap/ref_<D>.npz); tag _<D> if not T3a")
    p.add_argument("--torque-noise-sd", type=float, default=0.0,
                   help="tau refers to the good rate under signal-dependent motor noise of this SD as well "
                        "(the env's torque_noise_sd; kin_equifinality scores RL bowlers at 0.15); tag _tn<sd>")
    p.add_argument("--noise-seeds", type=int, default=16, help="motor-noise draws per candidate (common to all)")
    p.add_argument("--limits", default=None, choices=["human", "g1", "h1"],
                   help="the arm's actuator and joint limits (envs/arm_limits.py); tag _lim<name>")
    p.add_argument("--warm-rescale", action="store_true",
                   help="--warm-frontier starts made with other limits: ask for the torques they asked for "
                        "(commands x old peak / new peak, saturating at +/-1)")
    p.add_argument("--out", default=str(ROOT / "results" / "kinematic" / "frontier"))
    a = p.parse_args()
    global DELIVERY
    DELIVERY = a.delivery
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tr = tuple(a.target_range)
    global ENV2
    ENV2 = {}
    if a.torque_scale != 1.0:
        ENV2["torque_scale"] = a.torque_scale
    if a.actuator != "damping":
        ENV2.update(actuator=a.actuator, omega_max=a.omega_max)
    if a.drag:
        from envs.ballistics import CRICKET_DRAG_K
        ENV2["drag_k"] = CRICKET_DRAG_K
    if a.stiff_elbow:
        ENV2["elbow_solref"] = [0.004, 1.0]
    if a.limits:
        ENV2["limits"] = a.limits
    set_knots(a.handover, a.knot_dt)
    tag = f"a{int(a.angle)}" + (f"_h{int(round(a.handover * 1000))}" if a.handover is not None else "") \
        + (f"_k{int(round(a.knot_dt * 1000))}" if a.knot_dt is not None else "") \
        + (f"_q{a.torque_scale:g}".replace(".", "p") if a.torque_scale != 1.0 else "") \
        + (f"_hill{a.omega_max:g}" if a.actuator == "hill" else "") \
        + (f"_j{a.jitter_sd_ms:g}".replace(".", "p") if a.jitter_sd_ms != 12.0 else "") \
        + (f"_x{a.max_extension_deg:g}" if a.max_extension_deg != 15.0 else "") \
        + ("_pc" if a.p_cont else "") + ("_drag" if a.drag else "") + ("_se" if a.stiff_elbow else "") \
        + (f"_{a.delivery}" if a.delivery != "T3a" else "") \
        + (f"_tn{a.torque_noise_sd:g}".replace(".", "p") if a.torque_noise_sd > 0 else "") \
        + (f"_lim{a.limits}" if a.limits else "")
    import envs.legality as legality
    legality.LEGAL_EXTENSION_DEG = a.max_extension_deg    # the main process scores too (fitness/violation)
    global P_CONT
    P_CONT = a.p_cont
    logf = open(out / f"frontier_{tag}.log", "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    with mp.Pool(a.workers, initializer=_init, initargs=(a.angle, tr, a.handover, a.knot_dt, ENV2,
                                                               a.jitter_sd_ms, a.max_extension_deg,
                                                               a.p_cont, a.delivery,
                                                               (a.torque_noise_sd, a.noise_seeds))) as pool:
        starts = {}
        for run in a.warm:
            x = warm_start(run, a.angle, tr, a.handover)
            r = pool.map(evaluate, [x])[0]
            starts[run] = x
            log(f"warm start {run}: open-loop fit {(r.get('speed_m_s') or 0) * 3.6:.1f} km/h good {r.get('good')} "
                f"p_jit {r.get('p_jitter_12ms') or 0:.2f} land {r.get('land_x')}")
        for spec in a.warm_frontier:
            path, t = spec.rsplit(":", 1)
            rec = next(q for q in json.load(open(path)) if abs(q["tau"] - float(t)) < 1e-9)
            assert np.allclose(rec["knot_times_s"], KNOTS), f"{spec}: other knots"
            x = np.array(rec["knots"])
            if a.warm_rescale:
                from envs.arm_limits import PROFILES
                old = np.asarray(PROFILES[(rec.get("env2") or {}).get("limits") or "model"]["torque"])
                old = old * (rec.get("env2") or {}).get("torque_scale", 1.0)
                new = np.asarray(PROFILES[a.limits or "model"]["torque"])
                x = np.clip(x * old / new, -1.0, 1.0)
            x = x.ravel()
            r = pool.map(evaluate, [x])[0]
            starts[f"{Path(path).stem}@{t}"] = x
            log(f"warm start {spec}: {(r.get('speed_m_s') or 0) * 3.6:.1f} km/h good {r.get('good')} "
                f"p_jit({a.jitter_sd_ms:g} ms) {r.get('p_jitter_12ms') or 0:.2f}")
        results = []
        prev = None
        for tau in a.tau:
            best = None
            # continuation along the frontier: the previous tau's optimum is a start too
            cands = dict(starts, **({"previous_tau": prev} if prev is not None else {}))
            for k, (run, x0) in enumerate(cands.items()):
                log(f"tau {tau}: from {run}")
                b = optimise(pool, x0, tau, tr, a.sigma, a.gens, a.popsize, 1 + k, log, a.diagonal)
                if best is None or b[0] < best[0]:
                    best = (*b, run)
            f, x, r, run = best
            prev = x
            rec = {"angle": a.angle, "handover_s": a.handover, "env2": ENV2, "jitter_sd_ms": a.jitter_sd_ms, "max_extension_deg": a.max_extension_deg, "p_cont": a.p_cont, "delivery": a.delivery, "torque_noise_sd": a.torque_noise_sd, "noise_seeds": a.noise_seeds if a.torque_noise_sd > 0 else 0, "tau": tau, "fitness": f, "warm": run, "feasible": f < 0,
                   "speed_kmh": (r.get("speed_m_s") or 0) * 3.6, **r, "knots": x.reshape(len(KNOTS), -1).tolist(),
                   "knot_times_s": KNOTS.tolist()}
            results.append(rec)
            log(f"== a{int(a.angle)} tau {tau}: {'FEASIBLE' if f < 0 else 'infeasible'} {rec['speed_kmh']:.1f} km/h "
                f"p_jit {r.get('p_jitter_12ms') or 0:.2f} land {r.get('land_x')} ext {r.get('extension_deg')}")
            json.dump(results, open(out / f"frontier_{tag}.json", "w"), indent=1,
                      default=lambda v: bool(v) if isinstance(v, np.bool_) else float(v))


if __name__ == "__main__":
    main()
