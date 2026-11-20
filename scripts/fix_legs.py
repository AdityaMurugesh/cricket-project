"""Clean the captured bowler's legs after the jump, at the reference-motion level (supervisor, 20 Nov meeting).

Why: after ~600 ms before release no leg or foot marker of T3a is tracked (fit3d support), so the fit draws the
legs from its posture prior. The back foot then lands ~60 ms late and slides on the floor at 2-5 m/s, and the
front foot slides at 1-2 m/s and lifts 7 cm just after release. The pelvis is measured to -210 ms and the trunk
throughout, so only the legs are wrong.

The data the fix anchors on, besides the fit: the foot contacts, read off the synchronised Miqus videos
(camera 11, side view of the landing area; camera 9, from behind; 25 fps, so +/- 20 ms):
  back (right) foot   down -320 ms, heel off -140 ms, toe off -100 ms, then swings through
  front (left) foot   down  -80 ms, planted through release to the end of the capture (+400 ms)
Method (footskate clean-up, as Kovar, Schreiner & Gleicher 2002):
  1. foot targets. In a stance the foot is planted where the fit's foot is on average over that stance
     (least-squares plant: mean ball position and mean heading, flat on the floor). In the toe-off phase only the
     ball of the foot stays planted. Between stances the target is the fit's own foot path shifted by an offset
     that moves smoothly (minimum jerk) from one plant's correction to the next: interpolation between the
     instants that have data (the last leg markers at -600 ms, the video contacts). Never below the floor.
  2. per frame, damped least-squares IK on that leg's joints (hip ball joint, knee, ankle, subtalar, toes) to the
     heel and ball targets, regularised towards the fit's own angles. Pelvis, trunk and arms are not touched, so
     the bowling arm's simulation is unchanged (the body is prescribed; the legs are separate branches).
  3. a zero-lag low-pass on the leg joints (2nd-order Butterworth, --cutoff Hz).
Writes data/mocap/ref_<D>L.npz (delivery "<D>L", the layout of ref_<D>.npz; participant data, not in the repo:
copy it to the HPC by hand) and results/kinematic/legfix/legfix_<D>.json (+ .png): foot slide and floor
penetration in the video's stance windows, before and after, and the IK's residual.

    python scripts/fix_legs.py --delivery T3a
"""
import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from envs.humanoid_bowler import MOCAP_DIR, build_model, load_reference  # noqa: E402

OUT = ROOT / "results/kinematic/legfix"
# per delivery and side: stances, each a list of phases (start s, end s (None = end of capture), sites held), times
# from release. The delivery stride's are read off the video. The front foot stays down past +220 ms in the video,
# but the fit's pelvis (unmeasured after -210 ms) runs on at ~3 m/s and rises to 0.95 m, so no leg can keep that foot
# planted beyond ~+150 ms (the simulated delivery ends at most 40 ms after release). The follow-through steps (+0.28 s
# on) are not clear in the video and come from the fit's own foot heights (within 1 cm of the floor).
CONTACTS = {"T3a": {"data_end_s": -0.60,          # last frame with any leg marker (fit3d support)
                    "r": [[(-0.32, -0.14, ("heel", "ball")), (-0.14, -0.10, ("ball",))],
                          [(0.31, None, ("heel", "ball"))]],
                    "l": [[(-0.08, 0.10, ("heel", "ball")), (0.10, 0.18, ("ball",))],
                          [(0.28, None, ("heel", "ball"))]]}}
CLEARANCE_M = 0.06           # between two plants the swinging foot's sole stays above a 6 cm-high arch
REACH_M = {"ball": 0.955, "heel": 0.90}     # hip centre to the sole point, leg straight (thigh 0.366 + shank 0.480 +
                                            # ankle-to-ball 0.132 / ankle-to-heel 0.070), less 2 cm
LEG = ("hip", "knee_angle", "ankle_angle", "subtalar_angle", "mtp_angle")
SITES = ("heel", "ball")
BLEND_OUT_S = 0.20           # after a last toe-off, the correction fades back to the fit's swing over this long
FLOOR_Z = 0.0


def minjerk(u):
    u = np.clip(u, 0.0, 1.0)
    return u ** 3 * (10 - 15 * u + 6 * u ** 2)


class Legs:
    def __init__(self, m):
        self.m, self.d = m, mujoco.MjData(m)
        self.dofs, self.joints = {}, {}
        for s in "rl":
            js = [m.joint(f"{n}_{s}").id for n in LEG]
            self.joints[s] = js
            self.dofs[s] = np.concatenate([np.arange(m.jnt_dofadr[j], m.jnt_dofadr[j] + (3 if m.jnt_type[j] == 1 else 1))
                                           for j in js])
        self.site = {(s, k): m.site(f"sole_{k}_{s}").id for s in "rl" for k in SITES}
        # column in the side's dof list -> (joint, range) for the limited hinges
        self.hinge_cols = {}
        for s in "rl":
            cols, c = {}, 0
            for j in self.joints[s]:
                if m.jnt_type[j] == 3 and m.jnt_limited[j]:
                    cols[c] = (j, m.jnt_range[j].copy())
                c += 3 if m.jnt_type[j] == 1 else 1
            self.hinge_cols[s] = cols

    def feet(self, q):
        self.d.qpos[:] = q
        mujoco.mj_kinematics(self.m, self.d)
        return {key: self.d.site_xpos[i].copy() for key, i in self.site.items()}

    def hip(self, q, side):
        self.d.qpos[:] = q
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.xpos[self.m.body(f"femur_{side}").id].copy()

    def ik(self, q, q_reg, side, targets, tol=0.005):
        """A straight knee is a singularity that traps the solver with the leg locked straight (a flat foot under a
        high hip then misses by up to 6 cm), so a miss over tol is retried from the knee bent 35 and 60 degrees."""
        best = self._solve(q, q_reg, side, targets)
        kn = self.m.jnt_qposadr[self.m.joint(f"knee_angle_{side}").id]
        for k0 in (-0.6, -1.0):
            if best[1] <= tol:
                break
            q0 = q.copy()
            q0[kn] = k0
            r = self._solve(q0, q_reg, side, targets)
            if r[1] < best[1]:
                best = r
        return best

    def _solve(self, q, q_reg, side, targets, iters=120, mu=1e-3, lam=1e-3, max_step=0.15):
        """targets: {site: xyz}; damped least squares (step norm <= max_step rad), regularised towards q_reg.
        Returns the solved q and the worst residual (m): non-zero where the target is out of the leg's reach."""
        m, d = self.m, self.d
        dofs = self.dofs[side]
        q = q.copy()
        res = 0.0
        for _ in range(iters):
            d.qpos[:] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            J, e = [], []
            for k, x in targets.items():
                jp = np.zeros((3, m.nv))
                mujoco.mj_jacSite(m, d, jp, None, self.site[(side, k)])
                J.append(jp[:, dofs])
                e.append(x - d.site_xpos[self.site[(side, k)]])
            J, e = np.vstack(J), np.concatenate(e)
            res = float(np.abs(e).max())
            dreg = np.zeros(m.nv)
            mujoco.mj_differentiatePos(m, dreg, 1.0, q, q_reg)
            # active set: a hinge at its limit that the step would push further is held this iteration (otherwise
            # the clamp undoes its part of the step and the rest of the leg drifts off)
            free = np.ones(len(dofs), bool)
            for _a in range(len(dofs)):
                Jf = J * free
                A = Jf.T @ Jf + (mu + lam) * np.eye(len(dofs))
                dq = np.linalg.solve(A, Jf.T @ e + mu * dreg[dofs]) * free
                hit = False
                for c, (j, lim) in self.hinge_cols[side].items():
                    if free[c] and ((q[m.jnt_qposadr[j]] >= lim[1] - 1e-6 and dq[c] > 0) or
                                    (q[m.jnt_qposadr[j]] <= lim[0] + 1e-6 and dq[c] < 0)):
                        free[c] = False
                        hit = True
                if not hit:
                    break
            nd = np.linalg.norm(dq)
            if nd > max_step:
                dq *= max_step / nd
            v = np.zeros(m.nv)
            v[dofs] = dq
            mujoco.mj_integratePos(m, q, v, 1.0)
            for j in self.joints[side]:
                if m.jnt_type[j] == 3 and m.jnt_limited[j]:
                    a = m.jnt_qposadr[j]
                    q[a] = np.clip(q[a], *m.jnt_range[j])
            if res < 1e-4:
                break
        return q, res


def lowpass_legs(m, Q, rate, cutoff, legs):
    from scipy.signal import butter, filtfilt
    Q = Q.copy()
    b, a = butter(2, cutoff / (rate / 2))
    for s in "rl":
        for j in legs.joints[s]:
            qa = m.jnt_qposadr[j]
            n = 4 if m.jnt_type[j] == 1 else 1
            X = Q[:, qa:qa + n].copy()
            if n == 4:
                for f in range(1, len(X)):
                    if X[f] @ X[f - 1] < 0:
                        X[f] *= -1
            X = filtfilt(b, a, X, axis=0, padtype="odd", padlen=min(15, len(X) - 1))
            if n == 4:
                X /= np.linalg.norm(X, axis=1, keepdims=True)
            Q[:, qa:qa + n] = X
    return Q


def foot_metrics(P, t, rate, windows):
    """P[key]: (n, 3) site paths. Per foot and stance window: mean / max horizontal speed, lowest point."""
    out = {}
    for s, phases in windows.items():
        for (t0, t1, sites) in phases:
            t1 = t[-1] if t1 is None else t1
            w = (t >= t0 - 1e-9) & (t <= t1 + 1e-9)
            for k in sites:
                X = P[(s, k)]
                v = np.hypot(*np.gradient(X[:, :2], 1.0 / rate, axis=0).T)
                out[f"{s}_{k}_{t0:+.2f}"] = {"mean_slide_m_s": float(v[w].mean()), "max_slide_m_s": float(v[w].max()),
                                            "travel_m": float(np.hypot(*(X[w][-1, :2] - X[w][0, :2]))),
                                            "lowest_z_cm": float(X[w, 2].min() * 100)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delivery", default="T3a")
    ap.add_argument("--cutoff", type=float, default=10.0, help="leg low-pass (Hz)")
    a = ap.parse_args()
    C = CONTACTS[a.delivery]
    arm = float(np.load(MOCAP_DIR / f"ref_{a.delivery}.npz")["arm_scale"])
    m = build_model(arm_scale=arm)
    ref = load_reference(m, a.delivery)
    Q0, rate, rel = ref["qpos"], ref["rate"], ref["release_index"]
    n = len(Q0)
    t = (np.arange(n) - rel) / rate
    L = Legs(m)
    P0 = {key: np.zeros((n, 3)) for key in L.site}
    for i in range(n):
        for key, x in L.feet(Q0[i]).items():
            P0[key][i] = x

    # 1. targets per side and frame
    targets = {s: [dict() for _ in range(n)] for s in "rl"}
    i_data = int(np.searchsorted(t, C["data_end_s"]))
    plants, stances = {"r": [], "l": []}, {"r": {}, "l": {}}

    def plan(s, phases):
        """One stance: its plant and, per frame, which sole points are held (trimmed to what the leg can reach)."""
        t_down, t_off = phases[0][0], (phases[-1][1] if phases[-1][1] is not None else t[-1])
        W = np.where((t >= t_down - 1e-9) & (t <= t_off + 1e-9))[0]
        given = [next(ph[2] for ph in phases if t[i] <= (ph[1] if ph[1] is not None else t[-1]) + 1e-9) for i in W]
        ball, heel = P0[(s, "ball")][W], P0[(s, "heel")][W]
        fwd = ball[:, :2] - heel[:, :2]
        psi = np.arctan2(fwd[:, 1].mean(), fwd[:, 0].mean())
        u_f = np.array([np.cos(psi), np.sin(psi), 0.0])
        f0 = L.feet(Q0[W[0]])
        length = float(np.linalg.norm(f0[(s, "ball")] - f0[(s, "heel")]))     # the rigid foot's heel-to-ball
        H = np.array([L.hip(Q0[i], s) for i in W])
        # plant: the ball position (1 cm grid around the fit's foot and the hips) that lets the most stance frames be
        # reached as given (a flat frame the heel cannot reach counts half if the ball can: the foot lands or leaves
        # on its ball there); ties go to the least-squares spot (the fit's mean ball position)
        b_ls = np.array([*ball[:, :2].mean(0), FLOOR_Z])
        lo = np.minimum(ball[:, :2].min(0), H[:, :2].min(0)) - 0.3
        hi = np.maximum(ball[:, :2].max(0), H[:, :2].max(0)) + 0.3
        gx, gy = np.meshgrid(np.arange(lo[0], hi[0], 0.01), np.arange(lo[1], hi[1], 0.01))
        Bc = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, FLOOR_Z)], 1)
        rb = np.linalg.norm(Bc[:, None, :] - H[None], axis=2) <= REACH_M["ball"]
        rh = np.linalg.norm((Bc - length * u_f)[:, None, :] - H[None], axis=2) <= REACH_M["heel"]
        wants_heel = np.array(["heel" in v for v in given])
        score = np.where(wants_heel, np.where(rb & rh, 1.0, np.where(rb, 0.5, 0.0)), rb.astype(float)).sum(1)
        score -= 1e-3 * np.linalg.norm(Bc - b_ls, axis=1)
        k_best = int(np.argmax(score))
        b = Bc[k_best]
        plant = {"ball": b, "heel": b - length * u_f}
        kind = [("heel", "ball") if (wh and rb[k_best, j] and rh[k_best, j]) else ("ball",) if rb[k_best, j] else ()
                for j, wh in enumerate(wants_heel)]
        runs, cur = [], []
        for j, kd in enumerate(kind):
            if kd:
                cur.append(j)
            elif cur:
                runs.append(cur)
                cur = []
        if cur:
            runs.append(cur)
        run = max(runs, key=len)
        flat_t = [t[W[j]] for j in run if len(kind[j]) == 2]
        info = {"ball": b.tolist(), "heel": plant["heel"].tolist(), "least_squares_ball": b_ls.tolist(),
                "moved_m": float(np.linalg.norm(b - b_ls)), "given_stance_s": [float(t_down), float(t_off)],
                "used_stance_s": [float(t[W[run[0]]]), float(t[W[run[-1]]])],
                "used_flat_s": [float(min(flat_t)), float(max(flat_t))] if flat_t else None}
        return plant, {int(W[j]): kind[j] for j in run}, info

    for s in "rl":
        segs = []                       # (first frame, last frame, plant) per stance, in time order
        for phases in C[s]:
            plant, st, info = plan(s, phases)
            plants[s].append(info)
            stances[s].update(st)
            segs.append((min(st), max(st), plant))
            for i, sites in st.items():
                targets[s][i] = {k: plant[k].copy() for k in sites}
        # swings: the fit's whole foot shifted by one offset (the ball's correction, so heel and ball stay a rigid
        # pair) that moves smoothly from the last plant's correction (0 at the last marker data) to the next one's;
        # between two plants the sole also clears a CLEARANCE_M arch; after the last plant it fades back to the fit
        prev_i, prev_c, prev_is_plant = i_data, np.zeros(3), False
        for (i_a, i_b, plant) in segs + [(None, None, None)]:
            if i_a is not None:
                c_next = plant["ball"] - P0[(s, "ball")][i_a]
                for i in range(prev_i + (1 if prev_is_plant else 0), i_a):
                    u = (i - prev_i) / max(i_a - prev_i, 1)
                    c = prev_c + minjerk(u) * (c_next - prev_c)
                    targets[s][i] = {k: P0[(s, k)][i] + c for k in SITES}
                    if prev_is_plant:
                        for k in SITES:
                            targets[s][i][k][2] = max(targets[s][i][k][2], CLEARANCE_M * 4 * u * (1 - u))
                prev_i, prev_c, prev_is_plant = i_b, plant["ball"] - P0[(s, "ball")][i_b], True
            else:
                for i in range(prev_i + 1, n):
                    u = minjerk((t[i] - t[prev_i]) / BLEND_OUT_S)
                    targets[s][i] = {k: P0[(s, k)][i] + (1 - u) * prev_c for k in SITES}
        for i in range(i_data, n):
            for k in targets[s][i]:
                targets[s][i][k][2] = max(targets[s][i][k][2], FLOOR_Z)

    # 2. IK, frame by frame
    Q1 = Q0.copy()
    worst = {"r": 0.0, "l": 0.0}
    for i in range(i_data, n):
        q = Q0[i].copy()
        if i > i_data:                    # warm start: the previous frame's leg correction carried onto this frame
            for s in "rl":
                for j in L.joints[s]:
                    qa, nn = m.jnt_qposadr[j], (4 if m.jnt_type[j] == 1 else 1)
                    q[qa:qa + nn] = Q1[i - 1][qa:qa + nn]
        for s in "rl":
            if targets[s][i]:
                q, r = L.ik(q, Q0[i], s, targets[s][i])
                hz = L.feet(q)[(s, "heel")]
                if "heel" not in targets[s][i] and hz[2] < FLOOR_Z:      # a free heel may not go through the floor
                    q, r = L.ik(q, Q0[i], s, dict(targets[s][i], heel=np.array([hz[0], hz[1], FLOOR_Z])))
                worst[s] = max(worst[s], r)
        Q1[i] = q
    # 3. low-pass the legs, then keep the marker-data frames exactly as fitted (crossfade 60 ms)
    Q2 = lowpass_legs(m, Q1, rate, a.cutoff, L)
    # the filter blurs the plants a little: re-plant the stance frames exactly, starting from the filtered pose
    for s in "rl":
        for i, sites in stances[s].items():
            tg = {k: targets[s][i][k] for k in sites}
            q, _ = L.ik(Q2[i], Q2[i], s, tg)
            hz = L.feet(q)[(s, "heel")]
            if "heel" not in tg and hz[2] < FLOOR_Z:
                q, _ = L.ik(q, Q2[i], s, dict(tg, heel=np.array([hz[0], hz[1], FLOOR_Z])))
            Q2[i] = q
    k = int(round(0.06 * rate))
    for i in range(n):
        u = minjerk((i - (i_data - k)) / k)
        if u <= 0:
            Q2[i] = Q0[i]
        elif u < 1:
            v = np.zeros(m.nv)
            mujoco.mj_differentiatePos(m, v, 1.0, Q0[i], Q2[i])
            q = Q0[i].copy()
            mujoco.mj_integratePos(m, q, v, u)
            Q2[i] = q
    P2 = {key: np.zeros((n, 3)) for key in L.site}
    for i in range(n):
        for key, x in L.feet(Q2[i]).items():
            P2[key][i] = x
    windows = {}
    for s in "rl":                        # the stance actually used, split into its flat and ball-only stretches
        ii = sorted(stances[s])
        segs, cur = [], [ii[0]]
        for i in ii[1:]:
            if stances[s][i] == stances[s][cur[-1]] and i == cur[-1] + 1:
                cur.append(i)
            else:
                segs.append(cur)
                cur = [i]
        segs.append(cur)
        windows[s] = [(float(t[g[0]]), float(t[g[-1]]), stances[s][g[0]]) for g in segs]
    before, after = foot_metrics(P0, t, rate, windows), foot_metrics(P2, t, rate, windows)

    def skate(P):
        """Over the corrected frames: total horizontal distance a sole point covers while within 1 cm of the floor
        (footskate), and the lowest any sole point goes."""
        tot, low = 0.0, 9.0
        for key, X in P.items():
            Xc = X[i_data:]
            dz = np.minimum(Xc[:-1, 2], Xc[1:, 2])
            step = np.hypot(*(Xc[1:, :2] - Xc[:-1, :2]).T)
            tot += float(step[dz < 0.01].sum())
            low = min(low, float(Xc[:, 2].min()))
        return {"skate_m": tot, "lowest_cm": low * 100}
    before["all_feet"], after["all_feet"] = skate(P0), skate(P2)
    others = np.setdiff1d(np.arange(m.nq), np.concatenate([np.arange(m.jnt_qposadr[j], m.jnt_qposadr[j] + (4 if m.jnt_type[j] == 1 else 1))
                                                          for s in "rl" for j in L.joints[s]]))
    assert np.allclose(Q2[:, others], Q0[:, others]), "the fix must touch only the legs"
    dmax = {}
    for s in "rl":
        for j in L.joints[s]:
            qa, nn = m.jnt_qposadr[j], (4 if m.jnt_type[j] == 1 else 1)
            if nn == 1:
                dmax[m.joint(j).name] = float(np.degrees(np.abs(Q2[:, qa] - Q0[:, qa]).max()))
            else:
                dots = np.abs(np.sum(Q2[:, qa:qa + 4] * Q0[:, qa:qa + 4], axis=1)).clip(0, 1)
                dmax[m.joint(j).name] = float(np.degrees(2 * np.arccos(dots)).max())
    out_ref = MOCAP_DIR / f"ref_{a.delivery}L.npz"
    np.savez_compressed(out_ref, qpos=Q2, rate=rate, release_index=rel, nq=m.nq, delivery=f"{a.delivery}L",
                        arm_scale=arm)
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {"delivery": a.delivery, "contacts_from_video": {s: C[s] for s in "rl"}, "data_end_s": C["data_end_s"],
           "cutoff_hz": a.cutoff, "plants": plants, "ik_worst_residual_cm": {s: worst[s] * 100 for s in "rl"},
           "max_joint_change_deg": dmax, "before": before, "after": after, "ref": str(out_ref.relative_to(ROOT))}
    json.dump(rep, open(OUT / f"legfix_{a.delivery}.json", "w"), indent=1)
    print(f"wrote {out_ref}")
    print(f"  all feet, -600 ms on: footskate {before['all_feet']['skate_m']:.2f} -> {after['all_feet']['skate_m']:.2f} m, "
          f"lowest point {before['all_feet']['lowest_cm']:.1f} -> {after['all_feet']['lowest_cm']:.1f} cm")
    for key in [k for k in before if k != "all_feet"]:
        b_, a_ = before[key], after[key]
        print(f"  {key:16s} slide {b_['mean_slide_m_s']:4.2f} -> {a_['mean_slide_m_s']:4.2f} m/s (max {b_['max_slide_m_s']:4.2f} -> "
              f"{a_['max_slide_m_s']:4.2f}), travel {b_['travel_m']:4.2f} -> {a_['travel_m']:4.2f} m, "
              f"lowest {b_['lowest_z_cm']:5.1f} -> {a_['lowest_z_cm']:5.1f} cm")
    for s_, pls in plants.items():
        for pl in pls:
            print(f"  plant {s_}: {pl['moved_m']:.2f} m from the least-squares spot; given stance {pl['given_stance_s']}, "
                  f"used {[round(x, 2) for x in pl['used_stance_s']]}, flat {pl['used_flat_s'] and [round(x, 2) for x in pl['used_flat_s']]}")
    print("  IK worst residual (cm):", {s: round(worst[s] * 100, 2) for s in "rl"})
    print("  largest leg-joint change (deg):", {k: round(v, 1) for k, v in dmax.items()})
    # the foot paths are the participant's motion: kept with the reference, out of the repo (plot_legfix.py reads them)
    np.savez_compressed(MOCAP_DIR / f"legfix_{a.delivery}_feet.npz", t=t, **{f"before_{s}_{k}": P0[(s, k)] for s, k in L.site},
                        **{f"after_{s}_{k}": P2[(s, k)] for s, k in L.site})


if __name__ == "__main__":
    main()
