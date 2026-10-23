"""Make a labelling sheet for one delivery: identify markers by eye, no QTM needed.

    python scripts/label_mocap.py --delivery T3a

For a few moments chosen so that every marker the 3D fit could not place is
visible in at least one, it shows the camera-3 frame beside the same instant
from the same end of the room in 3D: the fitted skeleton as a stick figure,
every marker numbered. Markers the fit did not place are large and red; the
ones it did are small and grey (correct those too if they look wrong).

Answer with phrases in data/mocap/labels.json, e.g.
    {"T3a": {"18": "left elbow", "67": "head", "11": "left shoulder"}}
(parts understood: scripts/fit_mocap3d.py PART_BONES; "left"/"right" for
limbs), then re-run scripts/fit_mocap3d.py: labelled markers are then pinned
to those bones. Writes data/mocap/plots/labels_<D>.png.
"""
import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import fit_mocap3d as fm  # noqa: E402
from mocap import c3dio, delivery as dl, fit3d, tracks as tk  # noqa: E402

LOCO_PY = ROOT / "loco-venv" / "Scripts" / "python.exe"
MESHES = ROOT / "assets" / "skeleton3d" / "meshes"
READER = r"""
import sys, cv2
path, out = sys.argv[1], sys.argv[2]
cap = cv2.VideoCapture(path)
fps = cap.get(cv2.CAP_PROP_FPS)
for f in sys.argv[3:]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(int(f) / 100 * fps)))
    ok, img = cap.read()
    cv2.imwrite(f"{out}/f{f}.png", img)
"""
STICK = [("pelvis", "femur_r"), ("femur_r", "tibia_r"), ("tibia_r", "talus_r"), ("talus_r", "toes_r"),
         ("pelvis", "femur_l"), ("femur_l", "tibia_l"), ("tibia_l", "talus_l"), ("talus_l", "toes_l"),
         ("pelvis", "torso"), ("torso", "humerus_r"), ("humerus_r", "ulna_r"), ("ulna_r", "hand_r"),
         ("torso", "humerus_l"), ("humerus_l", "ulna_l"), ("ulna_l", "hand_l")]
MIN_FRAMES = 30   # shorter tracks are not worth anyone's time
MOMENTS = 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delivery", default="T3a")
    args = ap.parse_args()
    D = args.delivery
    n_trial = next(n for name, n, _ in dl.DELIVERIES if name == D)
    T = c3dio.load(c3dio.dynamic_trials()[n_trial - 1])
    p = fm.prepare(D, {n_trial: (T, tk.build_tracks(T))})
    z = np.load(ROOT / "data" / "mocap" / f"fit3d_{D}.npz")
    attached = set(int(t) for t in z["attach_tid"])
    wt = fit3d.world_tracks(p["tracks"], p["first"], p["last"], p["direction"], p["x_offset"], p["z_centre"])
    todo = [t for t in wt if t.tid not in attached and len(t.P) >= MIN_FRAMES]
    k_rel = int(z["release_index"])

    # moments: greedy cover of the unplaced tracks, near the middle of each
    moments, left = [], list(todo)
    while left and len(moments) < MOMENTS:
        cands = range(0, len(z["qpos"]), 5)
        best = max(cands, key=lambda f: sum(t.f0 + 5 <= f <= t.f1 - 5 for t in left))
        moments.append(best)
        left = [t for t in left if not (t.f0 + 5 <= best <= t.f1 - 5)]
    moments.sort()

    xml = re.sub(r'meshdir="[^"]*"', f'meshdir="{MESHES.as_posix()}"', str(z["xml"]))
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    video = next(c3dio.RAW_DIR.glob(f"Gait FB - IOR {n_trial}_Miqus_3_*.avi"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    frames_abs = [int(z["first_frame"]) + f for f in moments]
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "read.py"
        script.write_text(READER)
        subprocess.run([str(LOCO_PY), "-I", str(script), str(video), tmp, *map(str, frames_abs)], check=True)
        imgs = [mpimg.imread(Path(tmp) / f"f{f}.png") for f in frames_abs]

    fig = plt.figure(figsize=(18, 8 * len(moments)))
    azim = 0 if p["direction"] > 0 else 180     # camera 3: ahead of 'a' bowlers, behind 'b'
    for row, (f, img) in enumerate(zip(moments, imgs)):
        ax = fig.add_subplot(len(moments), 2, 2 * row + 1)
        ax.imshow(img[150:1050, 360:1560])
        ax.set_title(f"camera 3, {(f - k_rel) * 10:+d} ms from release", fontsize=12)
        ax.set_axis_off()
        ax3 = fig.add_subplot(len(moments), 2, 2 * row + 2, projection="3d")
        d.qpos[:] = z["qpos"][f]
        mujoco.mj_kinematics(m, d)
        for a, b in STICK:
            pa, pb = d.xpos[m.body(a).id], d.xpos[m.body(b).id]
            ax3.plot(*zip(pa, pb), color="0.55", lw=2)
        head = d.xpos[m.body("torso").id] + d.xmat[m.body("torso").id].reshape(3, 3) @ np.array([0, 0.6, 0])
        ax3.scatter(*head, s=500, facecolors="none", edgecolors="0.55")
        for tr in wt:
            if not tr.f0 <= f <= tr.f1:
                continue
            q = tr.P[f - tr.f0]
            new = tr.tid not in attached and len(tr.P) >= MIN_FRAMES
            ax3.scatter(*q, s=70 if new else 12, color="red" if new else "0.6")
            if new:
                ax3.text(*q, f"  {tr.tid}", fontsize=15, color="red", weight="bold")
        c = d.xpos[m.body("pelvis").id]
        for setlim, i in ((ax3.set_xlim, 0), (ax3.set_ylim, 1)):
            setlim(c[i] - 0.7, c[i] + 0.7)
        ax3.set_zlim(0, 2.0)
        ax3.view_init(elev=8, azim=azim)
        ax3.set_box_aspect((1, 1, 1.15))
        ax3.set_xticks([]), ax3.set_yticks([]), ax3.set_zticks([])
        ax3.set_title("same instant in 3D, seen from camera 3's end (red = not yet identified)",
                      fontsize=11)
    fig.tight_layout()
    out = ROOT / "data" / "mocap" / "plots" / f"labels_{D}.png"
    fig.savefig(out, dpi=70)
    print(f"-> {out}")
    print("not yet identified (number, when visible, median height):")
    for t in sorted(todo, key=lambda t: t.tid):
        print(f"  {t.tid:3d}: {(t.f0 - k_rel) * 10:+6d} .. {(t.f1 - k_rel) * 10:+5d} ms, "
              f"{np.median(t.P[:, 2]):.2f} m")


if __name__ == "__main__":
    main()
