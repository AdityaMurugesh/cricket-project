"""Check the 3D fit against the synchronised video: camera 3 above, the fitted
skeleton seen from the same end of the room below, at the same instants.

    python scripts/compare_mocap3d_video.py --delivery T3a
    python scripts/compare_mocap3d_video.py --delivery T4b --times=-1500,-720,-500,0

Camera 3 looks down the room along the run-up: a front view for deliveries
bowled towards it ('a'), a back view for the others. The video is 25 fps and
starts with the capture, so mocap frame f is video frame f/4. OpenCV lives in
loco-venv only, so the frames are read by that interpreter in a subprocess.
Writes data/mocap/plots/video3d_<D>.png.
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

import demo_mocap3d  # noqa: E402
from mocap import c3dio, delivery as dl  # noqa: E402

LOCO_PY = ROOT / "loco-venv" / "Scripts" / "python.exe"
CROP = (slice(150, 1050), slice(360, 1560))   # rows, cols of the 1920x1088 frame around the run-up
READER = """
import sys, cv2
path, out = sys.argv[1], sys.argv[2]
cap = cv2.VideoCapture(path)
fps = cap.get(cv2.CAP_PROP_FPS)
for f in sys.argv[3:]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(int(f) / 100 * fps)))
    ok, img = cap.read()
    cv2.imwrite(f"{out}/f{f}.png", img)
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delivery", default="T3a")
    ap.add_argument("--times", default="-1500,-1000,-740,-500,-200,0", help="ms from release")
    args = ap.parse_args()
    times = [int(t) for t in args.times.split(",")]
    trial = next(n for name, n, _ in dl.DELIVERIES if name == args.delivery)
    video = next(c3dio.RAW_DIR.glob(f"Gait FB - IOR {trial}_Miqus_3_*.avi"))
    rp = demo_mocap3d.Replay(demo_mocap3d.DATA / f"fit3d_{args.delivery}.npz")
    first = int(np.load(demo_mocap3d.DATA / f"fit3d_{args.delivery}.npz")["first_frame"])
    frames = [first + rp.frame(t / 1000) for t in times]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "read_frames.py"
        script.write_text(READER)
        subprocess.run([str(LOCO_PY), "-I", str(script), str(video), tmp, *map(str, frames)], check=True)
        imgs = [mpimg.imread(Path(tmp) / f"f{f}.png") for f in frames]
    r = mujoco.Renderer(rp.m, 450, 360)
    fig, axes = plt.subplots(2, len(times), figsize=(3.4 * len(times), 6.4))
    for i, (t, img) in enumerate(zip(times, imgs)):
        axes[0, i].imshow(img[CROP])
        axes[0, i].set_title(f"{t:+d} ms", fontsize=9)
        rp.pose(t / 1000)
        rp.shade(t / 1000)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [rp.d.qpos[0], rp.d.qpos[1], 1.0]
        # world x always points down the run: camera 3 is ahead of 'a' bowlers, behind 'b'
        towards = rp.direction > 0
        cam.distance, cam.azimuth, cam.elevation = 3.3, 180 if towards else 0, -12
        r.update_scene(rp.d, camera=cam)
        rp.draw_markers(r.scene, t / 1000)
        axes[1, i].imshow(r.render())
    for ax in axes.ravel():
        ax.set_axis_off()
    fig.suptitle(f"{args.delivery}: camera 3 (top) vs the fitted skeleton from the same end (bottom). "
                 "Orange = markers used, grey = not; translucent bones = no marker on them", fontsize=9)
    fig.tight_layout()
    out = demo_mocap3d.DATA / "plots" / f"video3d_{args.delivery}.png"
    fig.savefig(out, dpi=80)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
