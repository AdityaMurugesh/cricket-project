"""Identify which physical tracks are which body landmarks, around one delivery.

The AIM labels in these files are unreliable (see tracks.py), so identity
is established from physics instead, and every assignment records the
evidence it rests on:

  * The bowling hand is the fastest-moving point near release -- nothing
    else on the body comes close (typically 15-22 m/s vs <12 m/s).
  * Markers on one rigid segment keep a constant distance while the
    segment rotates through large angles. During the delivery the forearm
    and hand sweep through ~180 deg, so a distance that stays within ~1 cm
    over that sweep is strong evidence, not coincidence.
  * The IOR marker set fixes the expected spacings: wrist styloids ~7 cm
    apart, wrist to lateral epicondyle ~25 cm, hand marker (2nd metacarpal)
    ~12 cm beyond the wrist, acromion ~30 cm above the epicondyle.
  * Order along the arm: proximal markers sit on the far side of the elbow
    from the wrist, and move more slowly.

Nothing here uses the AIM label except to report it alongside, so any
agreement or disagreement with QTM is visible in the output.
"""
from dataclasses import dataclass, field

import numpy as np

from mocap import signal
from mocap.tracks import Track


@dataclass
class PairStat:
    n: int
    d: float      # mean distance, m
    sd: float     # distance standard deviation, m
    rot: float    # largest angle (deg) the connecting vector swings from its mean


def pair_stat(A: Track, B: Track, a=None, b=None):
    s = max(A.start, B.start, a if a is not None else -10**9)
    e = min(A.end, B.end, b if b is not None else 10**9)
    if e - s + 1 < 3:
        return None
    pa = A.xyz[s - A.start:e - A.start + 1]
    pb = B.xyz[s - B.start:e - B.start + 1]
    v = pb - pa
    d = np.linalg.norm(v, axis=1)
    if np.any(d < 1e-6):
        return None   # two columns carrying the same physical marker
    u = v / d[:, None]
    m = u.mean(0)
    m /= np.linalg.norm(m)
    rot = np.degrees(np.arccos(np.clip(u @ m, -1, 1))).max()
    return PairStat(e - s + 1, float(d.mean()), float(d.std()), float(rot))


def track_speed(t: Track, rate):
    """Speed of a track after the standard low-pass filter (m/s), cached on the track."""
    sp = getattr(t, "_speed", None)
    if sp is None or len(sp) != t.n:
        x = signal.lowpass(t.xyz, rate)
        v = signal.velocity(x, rate)
        sp = np.linalg.norm(v, axis=1)
        t._speed = sp
    return sp


@dataclass
class ArmChain:
    side: str                                      # "R" (bowling) or "L"
    hand: list = field(default_factory=list)       # track ids on the hand (HM2)
    wrist: list = field(default_factory=list)      # track ids on wrist styloids
    elbow: list = field(default_factory=list)      # track ids at the elbow
    proximal: list = field(default_factory=list)   # track ids on upper arm / acromion
    proximal_d: float = float("nan")               # elbow -> proximal distance, m
    evidence: list = field(default_factory=list)   # human-readable lines

    def ok_for_elbow_angle(self):
        return bool((self.wrist or self.hand) and self.elbow and self.proximal)

    def all_ids(self):
        return self.hand + self.wrist + self.elbow + self.proximal


def _peak(t, rate, a, b):
    sp = track_speed(t, rate)
    s, e = max(a, t.start), min(b, t.end)
    if s > e:
        return 0.0, None
    seg = sp[s - t.start:e - t.start + 1]
    if np.all(np.isnan(seg)):
        return 0.0, None
    k = int(np.nanargmax(seg)) + s
    return float(sp[k - t.start]), k


def _angle_at(c, p, w):
    u, v = p - c, w - c
    return np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)))


def identify_arm(tracks, rate, r, seed=None, exclude=(), side="R", win=(-60, 30)):
    """Build an arm chain around frame r.

    seed: the distal (hand/wrist) track to start from; default is the
    fastest track above 1.3 m within [r-30, r+15]."""
    a, b = r + win[0], r + win[1]
    W = [t for t in tracks if t.tid not in exclude and min(t.end, b) - max(t.start, a) >= 5]
    pk = {t.tid: _peak(t, rate, r - 30, r + 15) for t in W}
    byid = {t.tid: t for t in W}
    chain = ArmChain(side)

    if seed is None:
        best = None
        for t in W:
            sp = track_speed(t, rate)
            fr = np.arange(t.start, t.end + 1)
            m = (fr >= r - 40) & (fr <= r + 20) & (t.xyz[:, 1] > 1.4)
            if not m.any():
                continue
            k = int(np.nanargmax(np.where(m, sp, -1)))
            if best is None or sp[k] > best[0]:
                best = (float(sp[k]), t, t.start + k)
        if best is None:
            chain.evidence.append("no candidate distal track above 1.4 m")
            return chain
        seed = best[1]
        pk[seed.tid] = (best[0], best[2])
    F = seed
    chain.evidence.append(f"seed T{F.tid}: peak {pk[F.tid][0]:.1f} m/s at f{pk[F.tid][1]} "
                          f"(AIM '{F.aim_label}')")

    # distal cluster: rigid with the seed and within 16 cm
    D = [F]
    for t in W:
        if t is F:
            continue
        ps = pair_stat(F, t, a, b)
        if ps and ps.n >= 6 and ps.sd <= 0.010 and 0.04 <= ps.d <= 0.16:
            D.append(t)
            chain.evidence.append(f"T{t.tid} rigid with seed: d={ps.d:.3f} sd={ps.sd:.4f} "
                                  f"n={ps.n} swing={ps.rot:.0f}deg (AIM '{t.aim_label}')")

    # elbow: rigid with at least one distal marker at forearm length (the
    # wrist), and within reach of the rest (the hand, whose distance to the
    # elbow changes with wrist flexion), and slower than the seed
    Dids = {t.tid for t in D}
    ecands = []
    for t in W:
        if t.tid in Dids or pk[t.tid][0] >= pk[F.tid][0]:
            continue
        rel = [pair_stat(x, t, a, b) for x in D]
        rel = [p for p in rel if p and p.n >= 6]
        if not rel or any(p.sd > 0.04 or not (0.15 <= p.d <= 0.45) for p in rel):
            continue
        tight = [p for p in rel if p.sd <= 0.015 and 0.19 <= p.d <= 0.31]
        if tight:
            ecands.append((min(p.sd for p in tight), min(p.d for p in tight), t))
    ecands.sort(key=lambda c: c[0])
    E = []
    if ecands:
        E = [ecands[0][2]]
        for sd, dmin, t in ecands[1:]:
            ps = pair_stat(E[0], t, a, b)
            if ps and ps.n >= 6 and ps.sd <= 0.010 and 0.05 <= ps.d <= 0.13:
                E.append(t)   # second marker in the elbow region
                chain.evidence.append(f"second elbow-region marker T{t.tid}: {ps.d:.3f} m from "
                                      f"T{E[0].tid}, sd {ps.sd:.4f}")
                break
        for sd, dmin, t in ecands:
            tag = "elbow" if t in E else "elbow cand (unused)"
            chain.evidence.append(f"{tag} T{t.tid}: {dmin:.3f} m from distal cluster, "
                                  f"worst sd {sd:.4f} (AIM '{t.aim_label}')")
    if not E:
        chain.hand = [t.tid for t in D]
        chain.evidence.append("no elbow track found")
        return chain

    # classify distal markers by distance from the elbow
    for t in D:
        ds = [pair_stat(e, t, a, b) for e in E]
        ds = [p.d for p in ds if p]
        if not ds:
            continue
        (chain.wrist if min(ds) <= 0.30 else chain.hand).append(t.tid)
    chain.elbow = [t.tid for t in E]

    # proximal: rigid-ish with the elbow, on the far side from the wrist
    Eids = {t.tid for t in E}
    epk = max(pk[t.tid][0] for t in E)
    distal = [byid[i] for i in chain.wrist + chain.hand]
    best = None
    for t in W:
        if t.tid in Dids or t.tid in Eids or pk[t.tid][0] > epk * 1.05:
            continue
        ps = pair_stat(E[0], t, a, b)
        if not ps or ps.n < 6 or ps.sd > 0.02 or not (0.10 <= ps.d <= 0.40):
            continue
        ang = []
        for dt in distal:
            s = max(t.start, E[0].start, dt.start, a)
            e = min(t.end, E[0].end, dt.end, b)
            ang += [_angle_at(E[0].at(f), t.at(f), dt.at(f)) for f in range(s, e + 1)]
        if len(ang) < 4 or np.median(ang) < 110:
            continue
        chain.evidence.append(f"proximal cand T{t.tid}: d={ps.d:.3f} sd={ps.sd:.4f} "
                              f"angle at elbow {np.median(ang):.0f}deg (AIM '{t.aim_label}')")
        if best is None or ps.sd < best[2].sd:
            best = (t.tid, t, ps)
    if best:
        chain.proximal = [best[0]]
        chain.proximal_d = best[2].d
    return chain


@dataclass
class SpineChain:
    ids: list = field(default_factory=list)        # track ids, top to bottom
    evidence: list = field(default_factory=list)


def identify_spine(tracks, rate, r, exclude=(), win=(-45, 10)):
    """Find the column of back markers (C7 / thoracic / lumbar) near frame r.

    Signature: >= 2 tracks, mutually rigid (the frames used span the last
    stride and the delivery, during which the trunk itself rotates, so a
    constant spacing is not trivial), 4-25 cm apart, stacked near-vertically
    while the bowler is still upright at the start of the window, at trunk
    height, and moving at trunk speed (< 7 m/s) rather than with a limb."""
    a, b = r + win[0], r + win[1]
    f0 = a + 3
    C = []
    for t in tracks:
        if t.tid in exclude or t.start > f0 or t.end < b - 5:
            continue
        p = t.at(f0)
        sp = track_speed(t, rate)[max(a, t.start) - t.start:min(b, t.end) - t.start + 1]
        if 0.95 <= p[1] <= 1.65 and np.nanmax(sp) < 7.0:
            C.append(t)
    chain = SpineChain()
    # body midline (lateral coordinate) at f0: median over everything visible
    # at trunk/arm height. Back markers sit on it; arm markers are ~0.2 m off
    # to either side, which is what separates a hanging upper arm (also a
    # rigid, near-vertical pair) from the spine.
    zs = [t.at(f0)[2] for t in tracks if t.at(f0) is not None and 0.9 <= t.at(f0)[1] <= 1.7]
    if len(zs) < 4:
        chain.evidence.append("too few markers visible to locate the body midline")
        return chain
    mid = float(np.median(zs))
    C = [t for t in C if abs(t.at(f0)[2] - mid) < 0.08]
    best = None
    for i, A in enumerate(C):
        for B in C[i + 1:]:
            ps = pair_stat(A, B, a, b)
            if not ps or ps.n < 20 or ps.sd > 0.010 or not (0.04 <= ps.d <= 0.25):
                continue
            v = B.at(f0) - A.at(f0)
            tilt = np.degrees(np.arccos(abs(v[1]) / np.linalg.norm(v)))
            if tilt > 30:
                continue
            top, bot = (A, B) if A.at(f0)[1] > B.at(f0)[1] else (B, A)
            # try to extend with a third collinear marker
            members = [top, bot]
            for Cc in C:
                if Cc in members:
                    continue
                p1, p2 = pair_stat(top, Cc, a, b), pair_stat(bot, Cc, a, b)
                if not p1 or not p2 or max(p1.sd, p2.sd) > 0.012:
                    continue
                w = Cc.at(f0) - bot.at(f0)
                vv = top.at(f0) - bot.at(f0)
                off = np.linalg.norm(np.cross(w, vv)) / np.linalg.norm(vv)
                if off < 0.03 and 0.03 <= min(p1.d, p2.d) and max(p1.d, p2.d) <= 0.35:
                    members.append(Cc)
            members.sort(key=lambda t: -t.at(f0)[1])
            extent = members[0].at(f0)[1] - members[-1].at(f0)[1]
            if best is None or (len(members), extent) > (len(best[0]), best[1]):
                best = (members, extent, tilt)
    if best and len(best[0]) < 3:
        chain.evidence.append(f"only a 2-marker column found ({['T%d' % t.tid for t in best[0]]}); "
                              "too easily confused with a hanging upper arm, not used")
        best = None
    if best:
        chain.ids = [t.tid for t in best[0]]
        chain.evidence.append(
            f"spine column {['T%d' % i for i in chain.ids]} (AIM {[t.aim_label for t in best[0]]}), "
            f"vertical extent {best[1]:.3f} m, tilt from vertical {best[2]:.0f} deg at f{f0}")
    else:
        chain.evidence.append("no rigid vertical marker column found at trunk height")
    return chain
