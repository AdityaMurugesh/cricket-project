"""Load a Qualisys .c3d export into plain numpy arrays.

Missing samples (residual < 0 in the file) become NaN, so gaps are
explicit everywhere downstream. Positions are converted to metres.
"""
from dataclasses import dataclass, field
from pathlib import Path

import c3d
import numpy as np

RAW_DIR = Path(__file__).resolve().parent.parent / (
    "Aditya Bowling/Aditya Bowling/Aditya_Bowling/Data/"
    "IITD Abu Dhabi_Aditya_2026-10-01_001/2026-10-01/IOR - Full body_unspecified")


@dataclass
class Trial:
    name: str
    rate: float                 # marker sample rate, Hz
    labels: list                # marker labels, in column order
    pos: np.ndarray             # (n_frames, n_markers, 3), metres, NaN = missing
    analog_rate: float = 0.0
    analog_labels: list = field(default_factory=list)
    analog: np.ndarray = None   # (n_analog_samples, n_channels)
    plates: list = field(default_factory=list)  # dicts: corners (4,3) m, origin (3,) m, channels

    def marker(self, label):
        """(n, 3) trajectory for one marker, or all-NaN if the label is absent."""
        if label in self.labels:
            return self.pos[:, self.labels.index(label), :]
        return np.full((self.pos.shape[0], 3), np.nan)

    @property
    def n(self):
        return self.pos.shape[0]

    @property
    def t(self):
        return np.arange(self.n) / self.rate


def load(path):
    path = Path(path)
    with open(path, "rb") as h:
        r = c3d.Reader(h)
        lab_p = r.get("POINT:LABELS")
        labels = [l.strip() for l in lab_p.string_array] if lab_p is not None and r.point_used else []
        units = r.get("POINT:UNITS")
        scale = 0.001 if units is None or units.string_value.strip() == "mm" else 1.0
        pts, ans = [], []
        for _, p, a in r.read_frames():
            pts.append(p.copy())
            ans.append(a.copy())
        P = np.array(pts) if labels else np.zeros((len(pts), 0, 5))
        pos = P[:, :, :3].astype(float) * scale
        if labels:
            pos[P[:, :, 3] < 0] = np.nan
        analog = None
        alabels = []
        if r.analog_used:
            # read_frames yields (channels, samples_per_frame) per frame
            A = np.concatenate([a.T for a in ans], axis=0)
            analog = A.astype(float)
            alabels = [l.strip() for l in r.analog_labels]
        # Force plates: geometry is read but, in this session, is QTM's
        # unconfigured default (see REPORT.md), so nothing downstream uses it.
        plates = []
        try:
            n_used = int(r.get("FORCE_PLATFORM:USED").int16_value)
            corners = np.array(r.get("FORCE_PLATFORM:CORNERS").float_array)  # (n, 4, 3)
            origin = np.array(r.get("FORCE_PLATFORM:ORIGIN").float_array)    # (n, 3)
            chans = np.array(r.get("FORCE_PLATFORM:CHANNEL").int16_array)    # (n, 6)
            types = np.array(r.get("FORCE_PLATFORM:TYPE").int16_array).ravel()
            for i in range(n_used):
                plates.append(dict(corners=corners[i] * scale, origin=origin[i] * scale,
                                   channels=chans[i] - 1, type=int(types[i])))
        except Exception:
            pass
        # analog scaling (offset/scale/gen_scale) -- c3d.Reader returns raw counts
        if analog is not None:
            try:
                gen = float(r.get("ANALOG:GEN_SCALE").float_value)
                sc = np.array(r.get("ANALOG:SCALE").float_array, dtype=float).ravel()
                off = np.array(r.get("ANALOG:OFFSET").int16_array, dtype=float).ravel()
                analog = (analog - off[:analog.shape[1]]) * sc[:analog.shape[1]] * gen
            except Exception:
                pass
        return Trial(name=path.stem, rate=float(r.point_rate), labels=labels, pos=pos,
                     analog_rate=float(r.analog_rate) if r.analog_used else 0.0,
                     analog_labels=alabels, analog=analog, plates=plates)


def dynamic_trials():
    return sorted(RAW_DIR.glob("Gait FB - IOR *.c3d"), key=lambda p: int(p.stem.split()[-1]))


def static_trial():
    return RAW_DIR / "Static FB Anterior - IOR 1.c3d"
