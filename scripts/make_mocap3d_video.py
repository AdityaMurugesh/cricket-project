"""Side-by-side video: camera footage | the fitted 3D skeleton from the same
viewpoint | the skeleton from the side, synchronised, in slow motion.

    python scripts/make_mocap3d_video.py --delivery T3a            # 4x slow motion
    python scripts/make_mocap3d_video.py --delivery T4b --slow 2

Every 100 Hz mocap frame becomes one 25 fps video frame (4x slow motion by
default; --slow 2 drops every other frame). The camera video is 25 fps, so
its frame is repeated. Markers are drawn as in scripts/demo_mocap3d.py
(orange = used by the fit, grey = not; translucent bones = no marker on
them). The ball is released at the measured instant with the fitted hand's
velocity where the hand was tracked through release, and not shown elsewhere.
Rendering uses this interpreter (MuJoCo); reading the camera video and writing
the mp4 use OpenCV in loco-venv. Writes data/mocap/plots/video3d_<D>.mp4.
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import demo_mocap3d as dm  # noqa: E402
from mocap import c3dio, delivery as dl  # noqa: E402

LOCO_PY = ROOT / "loco-venv" / "Scripts" / "python.exe"
W, H = 640, 540
COMPOSE = r"""
import sys, cv2, numpy as np
video, tmp, out, n, fps_out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), float(sys.argv[5])
labels = sys.argv[6].split("|")
note = sys.argv[7]
frames = [int(x) for x in open(f"{tmp}/frames.txt").read().split()]
times = [x for x in open(f"{tmp}/times.txt").read().split()]
cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS)
writer = None
last_v, img = None, None
for i in range(n):
    v = int(round(frames[i] / 100 * fps))
    if v != last_v:
        cap.set(cv2.CAP_PROP_POS_FRAMES, v)
        ok, img = cap.read()
        img = cv2.resize(img[150:1050, 360:1560], (720, 540))
        last_v = v
    front = cv2.imread(f"{tmp}/front_{i:04d}.png")
    side = cv2.imread(f"{tmp}/side_{i:04d}.png")
    row = np.hstack([img, front, side])
    for x, text in zip((10, 730, 1370), labels):
        cv2.putText(row, text, (x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(row, times[i], (10, 525), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2, cv2.LINE_AA)
    for j, line in enumerate(note.split("|")):
        cv2.putText(row, line, (730, 505 + 20 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
                    cv2.LINE_AA)
    if writer is None:
        writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps_out, (row.shape[1], row.shape[0]))
    writer.write(row)
writer.release()
print("wrote", out)
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delivery", default="T3a")
    ap.add_argument("--slow", type=float, default=4.0, help="slow-motion factor (4 = every mocap frame)")
    args = ap.parse_args()
    step = max(1, int(round(4 / args.slow)))       # mocap frames per output frame at 25 fps
    path = dm.DATA / f"fit3d_{args.delivery}.npz"
    rp = dm.Replay(path)
    first = int(np.load(path)["first_frame"])
    trial = next(n for name, n, _ in dl.DELIVERIES if name == args.delivery)
    video = next(c3dio.RAW_DIR.glob(f"Gait FB - IOR {trial}_Miqus_3_*.avi"))
    m, d = rp.m, rp.d
    r = mujoco.Renderer(m, H, W)
    dt = m.opt.timestep
    per_frame = int(round(1 / rp.rate / dt))          # sim steps per mocap frame
    out_frames, out_times = [], []

    with tempfile.TemporaryDirectory() as tmp:
        count = 0

        def shoot(t):
            nonlocal count
            rp.shade(min(t, rp.t1))
            for name, az in (("front", 180 if rp.direction > 0 else 0), ("side", 90)):
                cam = mujoco.MjvCamera()
                cam.lookat[:] = [d.qpos[0], d.qpos[1], 1.0]
                cam.distance, cam.azimuth, cam.elevation = 3.3, az, -10
                r.update_scene(d, camera=cam)
                rp.draw_markers(r.scene, min(t, rp.t1))
                img = r.render()
                import matplotlib.image as mpimg
                mpimg.imsave(f"{tmp}/{name}_{count:04d}.png", img)
            out_frames.append(first + rp.frame(min(t, rp.t1)))
            out_times.append(f"{t * 1000:+.0f}ms")
            count += 1

        steps = 0

        def on_frame(t, released):
            nonlocal steps
            if steps % (per_frame * step) == 0:
                shoot(t)
            steps += 1

        dm.play(rp, on_frame)
        Path(tmp, "frames.txt").write_text(" ".join(map(str, out_frames)))
        Path(tmp, "times.txt").write_text(" ".join(out_times))
        script = Path(tmp, "compose.py")
        script.write_text(COMPOSE)
        out = dm.DATA / "plots" / f"video3d_{args.delivery}.mp4"
        labels = "camera 3|fitted skeleton, same view|fitted skeleton, side"
        note = (("bowling elbow held straight around release for display - its absolute angle is not "
                 "measurable with this marker set") if rp.hand_tracked else
                "hand not tracked at release: no ball shown")
        note += "|translucent = no marker on that bone: posed like the deliveries where it was tracked"
        subprocess.run([str(LOCO_PY), "-I", str(script), str(video), tmp, str(out), str(count), "25",
                        labels, note], check=True)
    print(f"{count} frames, {count / 25:.1f} s at {args.slow:g}x slow motion -> {out}")


if __name__ == "__main__":
    main()
