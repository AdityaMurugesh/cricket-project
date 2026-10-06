"""Rebuild physical marker tracks from a .c3d whose labels cannot be trusted.

QTM's AIM model was only "partially applied" to every dynamic trial of this
session (see Messages 2026-10-01.log) and failed outright on the static
trial. In practice the exported label columns are stitched together from
pieces of DIFFERENT physical markers: at an AIM re-identification boundary a
column jumps from one marker to another without a gap, and the marker it was
following carries on in some other column. Inter-marker distances that should
be rigid (e.g. right wrist to right hand) hold in as little as 3-11 % of
frames in some trials.

So labels are treated as hints only. A column is cut into pieces wherever it
has a gap or a jump that no real marker could make between two 10 ms frames,
and pieces in different columns are re-joined when one starts exactly where
another one's motion predicts it should be. Each resulting Track is one
physical marker for a continuous stretch of time, whatever it was called.
"""
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

# Constant-velocity prediction error that splits a column into two pieces.
# 0.12 m per 10 ms frame is 1200 m/s^2 of unexplained acceleration -- above
# anything a bowling hand produces (~500 m/s^2 near release) and far below
# the 0.2-1.0 m jumps seen at label switches.
JUMP_M = 0.12
# Two pieces are the same marker if the second starts within this distance
# of where the first one's last velocity says it should be.
LINK_M = 0.04


@dataclass
class Track:
    tid: int
    start: int                    # first frame (inclusive)
    end: int                      # last frame (inclusive)
    xyz: np.ndarray               # (end-start+1, 3) metres
    label_frames: Counter = field(default_factory=Counter)  # AIM label -> frames carried

    @property
    def n(self):
        return self.end - self.start + 1

    def at(self, f):
        return self.xyz[f - self.start] if self.start <= f <= self.end else None

    def window(self, a, b):
        """(frames, xyz) of the part of the track inside [a, b]."""
        s, e = max(a, self.start), min(b, self.end)
        if s > e:
            return np.array([], int), np.zeros((0, 3))
        return np.arange(s, e + 1), self.xyz[s - self.start:e - self.start + 1]

    @property
    def aim_label(self):
        return self.label_frames.most_common(1)[0][0] if self.label_frames else None


def _pieces(pos):
    n, m, _ = pos.shape
    out = []
    for j in range(m):
        ok = ~np.isnan(pos[:, j, 0])
        f = 0
        while f < n:
            if not ok[f]:
                f += 1
                continue
            s = f
            while f + 1 < n and ok[f + 1]:
                if f - 1 >= s:
                    pred = 2 * pos[f, j] - pos[f - 1, j]
                    if np.linalg.norm(pos[f + 1, j] - pred) > JUMP_M:
                        break
                elif np.linalg.norm(pos[f + 1, j] - pos[f, j]) > JUMP_M * 1.25:
                    break
                f += 1
            out.append((j, s, f))
            f += 1
    return out


def build_tracks(trial, min_frames=3):
    pos = trial.pos
    pcs = _pieces(pos)
    starts = {}
    for i, (j, s, e) in enumerate(pcs):
        starts.setdefault(s, []).append(i)
    nxt, prv = {}, {}
    for i, (j, s, e) in enumerate(pcs):
        if e - s < 1:
            continue
        pred = 2 * pos[e, j] - pos[e - 1, j]
        best, bd = None, LINK_M
        for k in starts.get(e + 1, []):
            if k in prv:
                continue
            d = np.linalg.norm(pos[e + 1, pcs[k][0]] - pred)
            if d < bd:
                best, bd = k, d
        if best is not None:
            nxt[i] = best
            prv[best] = i
    tracks = []
    for i in range(len(pcs)):
        if i in prv:
            continue
        chain = [i]
        while chain[-1] in nxt:
            chain.append(nxt[chain[-1]])
        s, e = pcs[chain[0]][1], pcs[chain[-1]][2]
        if e - s + 1 < min_frames:
            continue
        xyz = np.concatenate([pos[pcs[c][1]:pcs[c][2] + 1, pcs[c][0]] for c in chain])
        lab = Counter()
        for c in chain:
            lab[trial.labels[pcs[c][0]]] += pcs[c][2] - pcs[c][1] + 1
        tracks.append(Track(len(tracks), s, e, xyz, lab))
    return tracks


def speed(track, rate):
    """Per-frame speed (m/s) by central difference; ends use one-sided."""
    if track.n < 2:
        return np.zeros(track.n)
    v = np.gradient(track.xyz, axis=0) * rate
    return np.linalg.norm(v, axis=1)


def visible_at(tracks, f):
    return [t for t in tracks if t.start <= f <= t.end]
