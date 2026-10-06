"""Figures for checking identification and for the report."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def identification_check(res, tracks, path, offsets=(-30, -20, -10, -5, 0, 5, 15)):
    """Side and front views of every tracked point around release, with the
    identified bowling-arm chain and spine column drawn in."""
    by = {t.tid: t for t in tracks}
    r = res.release
    X = np.array([res.direction, 0, 0]); Z = np.cross(X, [0, 1, 0])
    fig, axes = plt.subplots(2, len(offsets), figsize=(2.6 * len(offsets), 6.4), sharey=True)
    arm = res.arm
    groups = [("proximal", arm.proximal, "tab:purple"), ("elbow", arm.elbow, "tab:red"),
              ("wrist", arm.wrist, "tab:orange"), ("hand", arm.hand, "gold")]
    spine = res.spine.ids if res.spine else []
    for k, off in enumerate(offsets):
        f = r + off
        vis = [t for t in tracks if t.start <= f <= t.end]
        if not vis:
            continue
        P = np.array([t.at(f) for t in vis])
        c = np.median(P, axis=0)
        for row, axis in enumerate((X, Z)):
            ax = axes[row, k]
            h = (P - c) @ axis
            ax.scatter(h, P[:, 1], s=8, c="0.6")
            pts = []
            for name, ids, col in groups:
                q = [by[i].at(f) for i in ids if by[i].at(f) is not None]
                if q:
                    q = np.mean(q, axis=0)
                    pts.append(q)
                    ax.scatter((q - c) @ axis, q[1], s=30, c=col, zorder=3)
            if len(pts) > 1:
                pts = np.array(pts)
                ax.plot((pts - c) @ axis, pts[:, 1], c="tab:red", lw=1.5)
            q = [by[i].at(f) for i in spine if by[i].at(f) is not None]
            if q:
                q = np.array(q)
                ax.plot((q - c) @ axis, q[:, 1], "o-", c="tab:blue", ms=4)
            ax.set_xlim(-1.0, 1.0); ax.set_ylim(0, 2.2); ax.set_aspect("equal"); ax.grid(alpha=.3)
            ax.set_title(f"{'side' if row == 0 else 'front'} {off:+d} fr", fontsize=8)
            ax.tick_params(labelsize=6)
    fig.suptitle(f"{res.name}: identified bowling arm (purple upper arm, red elbow, orange wrist, "
                 f"gold hand) and spine column (blue); release proxy f{r}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)
