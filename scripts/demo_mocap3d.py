"""Replay a captured delivery on the full-body 3D skeleton fitted to the markers.

Loads data/mocap/fit3d_<D>.npz (made by scripts/fit_mocap3d.py). What you see:

  * the skeleton, posed in every frame by the fit to the markers (mocap/fit3d.py);
  * every exported marker as a dot: orange if the fit used it, grey if not;
  * bones drawn solid carry a marker in that frame; translucent ones do not,
    so their pose comes from the fit's priors (gap filling, the floor and the
    measured steps, the initial guess) rather than from a marker;
  * the ball, carried in the bowling hand and let go at the release instant
    with the fitted hand's velocity, then physically simulated -- only where
    the hand was tracked through release (T1a, T2b, T3a, T4b); elsewhere the
    arm at release is the fit's estimate, so no ball is shown.

    python scripts/demo_mocap3d.py --delivery T3a --view      # MuJoCo viewer
    python scripts/demo_mocap3d.py --delivery T3a --frames    # saves data/mocap/plots/demo3d_T3a.png
"""
import argparse
import re
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "mocap"
MESHES = ROOT / "assets" / "skeleton3d" / "meshes"
USED_RGBA = np.array([1.0, 0.55, 0.0, 1.0], np.float32)
UNUSED_RGBA = np.array([0.6, 0.6, 0.6, 1.0], np.float32)
MARKER_RADIUS = 0.016
GHOST_ALPHA = 0.22          # bones with no marker on them in this frame
GRIP = np.array([0.035, -0.09, 0.0])   # ball centre in the right hand's frame (scaled): palm, fingers
FLIGHT_S = 1.5              # longest ball flight simulated after release
KMH = 3.6


class Replay:
    def __init__(self, path):
        z = np.load(path)
        xml = str(z["xml"])
        xml = re.sub(r'meshdir="[^"]*"', f'meshdir="{MESHES.as_posix()}"', xml)
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.d = mujoco.MjData(self.m)
        self.Q = z["qpos"]
        self.k_rel = int(z["release_index"])
        self.rate = float(z["rate"])
        self.markers, self.used = z["markers"], z["marker_used"]
        self.support = z["support"]
        self.hand_tracked = bool(z["hand_tracked"])
        self.release_source = str(z["release_source"])
        self.scale = float(z["scale"])
        self.resid_med = z["resid_med"]
        self.direction = int(z["direction"])     # +1 bowled towards lab +x (camera 3), -1 away
        m = self.m
        self.nq = m.joint("ball_free").qposadr[0]
        self.ball_q = m.joint("ball_free").qposadr[0]
        self.ball_v = m.joint("ball_free").dofadr[0]
        self.hand = m.body("hand_r").id
        if not self.hand_tracked:   # no measured release: no ball
            m.geom_rgba[m.geom("ball_geom").id, 3] = 0
        self.rgba0 = m.geom_rgba.copy()
        self.body_geoms = {b: np.where(m.geom_bodyid == b)[0] for b in range(m.nbody)}
        self.quat_adr = [m.jnt_qposadr[j] + (3 if m.jnt_type[j] == 0 else 0)
                         for j in range(m.njnt) if m.jnt_type[j] in (0, 1) and m.joint(j).name != "ball_free"]
        for a in self.quat_adr:   # sign-continuous quaternions, for interpolation
            for f in range(1, len(self.Q)):
                if self.Q[f, a:a + 4] @ self.Q[f - 1, a:a + 4] < 0:
                    self.Q[f, a:a + 4] *= -1

    @property
    def t0(self):
        return -self.k_rel / self.rate

    @property
    def t1(self):
        return (len(self.Q) - 1 - self.k_rel) / self.rate

    def frame(self, t):
        return int(np.clip(round(t * self.rate) + self.k_rel, 0, len(self.Q) - 1))

    def body_q(self, t):
        """Body coordinates at time t (s from release), interpolated between frames."""
        x = np.clip(t * self.rate + self.k_rel, 0, len(self.Q) - 1)
        i = min(int(x), len(self.Q) - 2)
        u = x - i
        q = (1 - u) * self.Q[i, :self.nq] + u * self.Q[i + 1, :self.nq]
        for a in self.quat_adr:
            q[a:a + 4] /= np.linalg.norm(q[a:a + 4])
        return q

    def grip(self):
        R = self.d.xmat[self.hand].reshape(3, 3)
        return self.d.xpos[self.hand] + R @ (GRIP * self.scale)

    def pose(self, t, carry=True):
        d = self.d
        d.qpos[:self.nq] = self.body_q(t)
        d.qvel[:self.ball_v] = 0
        mujoco.mj_kinematics(self.m, d)
        if carry:
            d.qpos[self.ball_q:self.ball_q + 3] = self.grip()
            d.qpos[self.ball_q + 3:self.ball_q + 7] = [1, 0, 0, 0]
            d.qvel[self.ball_v:self.ball_v + 6] = 0
        mujoco.mj_forward(self.m, d)

    def shade(self, t):
        """Solid bones carry a marker in this frame; the rest are translucent."""
        f = self.frame(t)
        rgba = self.rgba0.copy()
        for b, geoms in self.body_geoms.items():
            if b < self.support.shape[1] and not self.support[f, b] and len(geoms):
                rgba[geoms, 3] = np.minimum(rgba[geoms, 3], GHOST_ALPHA)
        rgba[self.m.geom("floor").id] = self.rgba0[self.m.geom("floor").id]
        rgba[self.m.geom("ball_geom").id] = self.rgba0[self.m.geom("ball_geom").id]
        self.m.geom_rgba[:] = rgba

    def draw_markers(self, scene, t):
        f = self.frame(t)
        for p, u in zip(self.markers[f], self.used[f]):
            if np.isnan(p[0]) or scene.ngeom >= scene.maxgeom:
                continue
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.array([MARKER_RADIUS, 0, 0]), p.astype(np.float64), np.eye(3).ravel(),
                                USED_RGBA if u else UNUSED_RGBA)
            scene.ngeom += 1


class Clock:
    """Paces the viewer by wall time: sleeping per 2 ms sim step overshoots on
    Windows (~15 ms per sleep) and played the delivery several times too slow."""

    def __init__(self, speed):
        self.speed, self.start = speed, None

    def wait(self, t):
        if self.start is None:
            self.start = (time.perf_counter(), t)
        ahead = (t - self.start[1]) / self.speed - (time.perf_counter() - self.start[0])
        if ahead > 0.005:
            time.sleep(ahead)


def play(rp, on_frame, real_time=False):
    """Run the replay; on_frame(t, released) is called every sim step. Returns
    (release speed m/s, landing (x, y) or None)."""
    m, d = rp.m, rp.d
    dt = m.opt.timestep
    clock = Clock(real_time) if real_time else None
    prev = None
    n_pre = int(round(-rp.t0 / dt))     # integer steps: a float sum drifts onto t = 0
    for i in range(n_pre):
        t = rp.t0 + i * dt
        rp.pose(t)
        prev = rp.grip()
        on_frame(t, False)
        if clock:
            clock.wait(t)
    if not rp.hand_tracked:   # play the rest of the delivery; the ball stays hidden
        for i in range(int(rp.t1 / dt) + 1):
            rp.pose(i * dt)
            on_frame(i * dt, True)
            if clock:
                clock.wait(i * dt)
        return None, None
    # release: the ball leaves with the hand's finite-difference velocity
    rp.pose(0.0)
    v = (rp.grip() - prev) / dt
    d.qpos[rp.ball_q:rp.ball_q + 3] = rp.grip()
    d.qvel[rp.ball_v:rp.ball_v + 3] = v
    landing = None
    t = 0.0
    floor = m.geom("floor").id
    ball = m.geom("ball_geom").id
    while t < FLIGHT_S:
        mujoco.mj_step(m, d)
        t += dt
        ball_qpos = d.qpos[rp.ball_q:rp.ball_q + 7].copy()
        ball_qvel = d.qvel[rp.ball_v:rp.ball_v + 6].copy()
        rp.pose(min(t, rp.t1), carry=False)   # the body stays kinematic
        d.qpos[rp.ball_q:rp.ball_q + 7] = ball_qpos
        d.qvel[rp.ball_v:rp.ball_v + 6] = ball_qvel
        mujoco.mj_forward(m, d)
        if landing is None:
            for i in range(d.ncon):
                c = d.contact[i]
                if {c.geom1, c.geom2} == {floor, ball}:
                    landing = tuple(d.qpos[rp.ball_q:rp.ball_q + 2])
        on_frame(t, True)
        if clock:
            clock.wait(t)
        if landing is not None and t > rp.t1:
            break
    return float(np.linalg.norm(v)), landing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delivery", default="T3a")
    ap.add_argument("--view", action="store_true", help="open the MuJoCo viewer (loops on Enter)")
    ap.add_argument("--frames", action="store_true", help="save rendered frames to data/mocap/plots/")
    ap.add_argument("--slow", type=float, default=1.0, help="playback speed in the viewer, e.g. 0.25")
    args = ap.parse_args()
    rp = Replay(DATA / f"fit3d_{args.delivery}.npz")
    print(f"[{args.delivery}] full-body fit: {np.median(np.isfinite(rp.markers[:, :, 0]).sum(1)):.0f} "
          f"markers visible and {np.median(rp.used.sum(1)):.0f} used per frame (median), median "
          f"marker-to-skeleton distance {np.nanmedian(rp.resid_med) * 100:.1f} cm; skeleton scale "
          f"{rp.scale:.2f}")
    print("  orange dots: markers the fit uses; grey: markers it could not place; translucent "
          "bones: no marker on them in that frame")

    if args.frames:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        times = [-2.0, -1.5, -1.0, -0.74, -0.5, -0.3, -0.2, -0.1, 0.0, 0.2]
        times = [t for t in times if rp.t0 <= t <= rp.t1]
        r = mujoco.Renderer(rp.m, 480, 400)
        shots = {}
        # the fit's world frame always runs along +x, so these hold for both ends of the room
        for az, label in ((90, "side"), (180, "front")):
            want = list(times)

            def grab(t, released, az=az, label=label):
                while want and t >= want[0] - 1e-9:
                    rp.shade(want[0])
                    cam = mujoco.MjvCamera()
                    cam.lookat[:] = [rp.d.qpos[0], rp.d.qpos[1], 1.0]
                    cam.distance, cam.azimuth, cam.elevation = 3.3, az, -8
                    r.update_scene(rp.d, camera=cam)
                    rp.draw_markers(r.scene, want[0])
                    shots[(label, want[0])] = r.render().copy()
                    want.pop(0)
            play(rp, grab)
        fig, axes = plt.subplots(2, len(times), figsize=(3.2 * len(times), 6.0))
        for i, t in enumerate(times):
            for j, label in enumerate(("side", "front")):
                axes[j, i].imshow(shots[(label, t)])
                axes[j, i].set_axis_off()
            axes[0, i].set_title(f"{t * 1000:+.0f} ms", fontsize=9)
        fig.suptitle(f"{args.delivery}: full-body skeleton fitted to the captured markers (release at "
                     f"0 ms). Orange = markers used, grey = not; translucent bones = no marker on "
                     f"them", fontsize=9)
        fig.tight_layout()
        out = DATA / "plots" / f"demo3d_{args.delivery}.png"
        fig.savefig(out, dpi=80)
        plt.close(fig)
        print(f"frames -> {out}")

    viewer = None
    if args.view:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(rp.m, rp.d)
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 4.5, 90, -10

    def show(t, released):
        if viewer is None:
            return
        with viewer.lock():
            rp.shade(t)
            if not released:
                viewer.cam.lookat[:] = [rp.d.qpos[0], rp.d.qpos[1], 1.0]
            viewer.user_scn.ngeom = 0
            rp.draw_markers(viewer.user_scn, min(t, rp.t1))
        viewer.sync()

    while True:
        speed, landing = play(rp, show, real_time=args.slow if viewer is not None else False)
        if speed is None:
            print(f"no ball: the hand was not tracked through release in {args.delivery}, so the arm "
                  f"there is the fit's estimate; release timed by {rp.release_source}")
        else:
            print(f"ball release speed (fitted hand, tracked through release): {speed * KMH:.1f} km/h")
        if landing is not None:
            print(f"ball lands {landing[0]:.2f} m along the pitch from the trunk at release")
        if viewer is None or not viewer.is_running():
            break
        try:
            if input("Enter to replay, q to quit > ").strip().lower() == "q":
                break
        except EOFError:
            break
    if viewer is not None:
        viewer.close()


if __name__ == "__main__":
    main()
