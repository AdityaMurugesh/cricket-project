"""Ball flight from the release state to the ground: where the ball pitches (length) and its line.

Until 20 Nov 2026 every result used a drag-free parabola. A cricket ball does slow down in the air:
quadratic drag F = 1/2 rho Cd A |v| v gives a deceleration k |v| v with

    k = rho * Cd * A / (2 m) = 1.2 * 0.45 * (pi * 0.036^2) / (2 * 0.156) = 0.00705 per metre

(air 1.2 kg/m^3; Cd ~0.4-0.5 for a cricket ball below the drag crisis, Mehta 1985 / 2005; diameter 72 mm,
mass 156 g, Laws of Cricket 4.1). Over a 12 m flight that takes ~8% off the speed and shortens a
100 km/h ball's length by ~0.5-1 m. Spin and seam forces (swing, Magnus) are still left out.

land(p, v, k): k = 0 is the closed-form parabola, bit-for-bit the old RunupBowlEnv._release; k > 0
integrates the flight (RK4, 2 ms steps, pure Python floats, ~0.3 ms per call).
"""
import math

BALL_R = 0.036
G = 9.81
CRICKET_DRAG_K = 1.2 * 0.45 * (math.pi * BALL_R ** 2) / (2 * 0.156)    # 0.00705 1/m


def land(p, v, k=0.0, dt=0.002):
    """Length (m beyond the release point, along x) and line (y) where the ball's centre is BALL_R above
    the ground. p, v: release position and velocity (x along the pitch, z up)."""
    if k <= 0.0:
        z0 = p[2] - BALL_R
        t_fl = (v[2] + math.sqrt(max(v[2] ** 2 + 2 * G * z0, 0.0))) / G
        return float(v[0] * t_fl), float(p[1] + v[1] * t_fl)
    x, y, z = 0.0, float(p[1]), float(p[2])
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])

    def acc(ux, uy, uz):
        s = k * math.sqrt(ux * ux + uy * uy + uz * uz)
        return -s * ux, -s * uy, -G - s * uz

    for _ in range(20000):                      # 40 s cap: never reached by a delivery
        if z <= BALL_R:
            break
        ax1, ay1, az1 = acc(vx, vy, vz)
        h2 = 0.5 * dt
        vx2, vy2, vz2 = vx + h2 * ax1, vy + h2 * ay1, vz + h2 * az1
        ax2, ay2, az2 = acc(vx2, vy2, vz2)
        vx3, vy3, vz3 = vx + h2 * ax2, vy + h2 * ay2, vz + h2 * az2
        ax3, ay3, az3 = acc(vx3, vy3, vz3)
        vx4, vy4, vz4 = vx + dt * ax3, vy + dt * ay3, vz + dt * az3
        ax4, ay4, az4 = acc(vx4, vy4, vz4)
        nx = x + dt / 6 * (vx + 2 * vx2 + 2 * vx3 + vx4)
        ny = y + dt / 6 * (vy + 2 * vy2 + 2 * vy3 + vy4)
        nz = z + dt / 6 * (vz + 2 * vz2 + 2 * vz3 + vz4)
        if nz <= BALL_R:                         # crossed in this step: interpolate (error ~ g dt^2 / 8)
            f = (z - BALL_R) / (z - nz)
            return float(x + f * (nx - x)), float(y + f * (ny - y))
        x, y, z = nx, ny, nz
        vx += dt / 6 * (ax1 + 2 * ax2 + 2 * ax3 + ax4)
        vy += dt / 6 * (ay1 + 2 * ay2 + 2 * ay3 + ay4)
        vz += dt / 6 * (az1 + 2 * az2 + 2 * az3 + az4)
    return float(x), float(y)
