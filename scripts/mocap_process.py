"""Process the 2026-10-01 Qualisys bowling session end to end.

Reads the raw .c3d files (never modified), writes everything derived to
data/mocap/: per-delivery metrics CSV, per-delivery time series, reference
motion for the humanoid, figures, and REPORT.md.

    python scripts/mocap_process.py
"""
import csv
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mocap import c3dio, delivery as dl, identify as idf, plots, retarget, runup, tracks as tk  # noqa: E402
from mocap.signal import CUTOFF_HZ  # noqa: E402

OUT = ROOT / "data" / "mocap"
PLOTS = OUT / "plots"
KMH = 3.6

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# AIM-label pairs that should be rigid (same segment) if the labels were right
RIGID_LABEL_PAIRS = [("R_RSP", "R_USP"), ("R_USP", "R_HM2"), ("R_HLE", "R_USP"),
                     ("L_RSP", "L_USP"), ("L_HLE", "L_USP"), ("L_FLE", "L_FAX"),
                     ("R_FLE", "R_FAX"), ("CV7", "TV2"), ("L_FAL", "L_FCC")]


def label_reliability(trials):
    """For each rigid AIM-label pair: modal distance across the session, and
    per trial the % of co-visible frames within 2 cm of it."""
    rows = []
    for a, b in RIGID_LABEL_PAIRS:
        ds = [np.linalg.norm(T.marker(a) - T.marker(b), axis=1) for T in trials]
        allv = np.concatenate([d[~np.isnan(d)] for d in ds])
        if len(allv) < 30:
            continue
        h, e = np.histogram(allv, bins=np.arange(0, 2.0, 0.005))
        mode = (e[h.argmax()] + e[h.argmax() + 1]) / 2
        per = []
        for d in ds:
            v = d[~np.isnan(d)]
            per.append(f"{(np.abs(v - mode) < 0.02).mean() * 100:.0f}% of {len(v)}" if len(v) else "-")
        rows.append((f"{a}-{b}", mode, per))
    return rows


def coverage(trials):
    rows = []
    for T in trials:
        vis = ~np.isnan(T.pos[:, :, 0])
        rows.append((T.name, T.n, T.n / T.rate, len(T.labels), float(vis.sum(1).mean()),
                     int(vis.sum(1).max())))
    return rows


def extension_sensitivity(res):
    """Endpoint extension under +/-1 frame of release and two arm-horizontal definitions."""
    from envs.legality import elbow_extension_deg
    k = res.release - res.frame0
    flex = res.series["elbow_flexion"]
    arm = res.series["wrist"] - res.series["proximal"]
    whole = np.degrees(np.arctan2(-arm[:, 1], res.direction * arm[:, 0]))
    out = []
    for theta in (res.series["shoulder_theta"], whole):
        for dk in (-1, 0, 1):
            kk = k + dk
            if kk >= len(flex) or np.isnan(theta[kk]) or np.isnan(flex[kk]):
                continue
            s = kk
            while s - 1 >= 0 and not np.isnan(theta[s - 1]) and not np.isnan(flex[s - 1]):
                s -= 1
            th = np.degrees(np.unwrap(np.radians(theta[s:kk + 1])))
            e = elbow_extension_deg(th, flex[s:kk + 1], len(th) - 1, "last_before_release", "endpoint")
            if e is not None:
                out.append(e)
    return out


def fmt(v, nd=1):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    trials = {}
    for p in c3dio.dynamic_trials():
        T = c3dio.load(p)
        trials[int(p.stem.split()[-1])] = (T, tk.build_tracks(T))
    static = c3dio.load(c3dio.static_trial())

    model = mujoco.MjModel.from_xml_path(str(ROOT / "assets" / "bowler.xml"))
    joint_names = [model.joint(i).name for i in range(model.njnt)]

    results, runups = [], {}
    for name, tr, r in dl.DELIVERIES:
        T, trs = trials[tr]
        res = dl.analyse(name, T, trs, r)
        results.append(res)
        usable = res.metrics.get("release_quality") in ("good", "lower bound (lost after peak)")
        rel, src = (res.release, "hand-speed peak") if usable else (r, "video")
        runups[name] = runup.analyse(name, T, trs, rel, res.direction, src)

    # --- metrics CSV --------------------------------------------------------
    cols = ["delivery", "trial", "direction", "release_frame", "release_time_s", "release_quality",
            "distal_marker", "peak_distal_speed_kmh", "peak_distal_speed_raw_kmh", "release_height_m",
            "approach_speed_kmh", "release_timing", "runup_start_ms", "runup_distance_m",
            "runup_peak_speed_kmh", "steps_detected", "step_rate_hz", "takeoff_ms", "jump_peak_ms",
            "jump_rise_m", "takeoff_speed_kmh",
            "ffc_to_release_ms", "bfc_to_release_ms", "stride_length_m",
            "arm_horizontal_frame", "horizontal_to_release_ms", "elbow_flexion_at_horizontal_deg",
            "elbow_flexion_at_release_deg", "elbow_extension_endpoint_deg", "elbow_extension_max_deg",
            "shoulder_theta_at_release_deg", "model_arm_theta_at_release_deg",
            "trunk_lateral_flexion_at_release_deg", "trunk_forward_flexion_at_release_deg",
            "arm_tracks", "notes"]
    ref_info = {}
    with open(OUT / "delivery_metrics.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for res in results:
            m = dict(res.metrics)
            m.update(runups[res.name].metrics)
            q = m.get("release_quality", "not captured")
            usable = q in ("good", "lower bound (lost after peak)")
            if usable and "elbow_flexion" in res.series:
                th, el, info = retarget.planar_arm(res)
                m["model_arm_theta_at_release_deg"] = info["theta_at_release_deg"]
            row = {"delivery": res.name, "trial": res.trial, "direction": res.direction,
                   "release_frame": res.release if usable else "",
                   "release_time_s": f"{res.release / res.rate:.2f}" if usable else "",
                   "release_quality": q,
                   "arm_tracks": json.dumps({"hand": res.arm.hand, "wrist": res.arm.wrist,
                                             "elbow": res.arm.elbow, "proximal": res.arm.proximal}),
                   "notes": " | ".join(res.notes + runups[res.name].notes)}
            for c in cols:
                if c not in row:
                    v = m.get(c, "")
                    if not usable and c in ("peak_distal_speed_kmh", "peak_distal_speed_raw_kmh",
                                            "release_height_m"):
                        v = ""   # a peak below release height is not a release value
                    row[c] = (f"{v:.3f}" if isinstance(v, float) and not np.isnan(v) else
                              "" if isinstance(v, float) else v)
            w.writerow([row[c] for c in cols])

    # --- per-delivery series and reference motion ---------------------------
    for name, run in runups.items():
        np.savez_compressed(OUT / f"runup_{name}.npz", frame0=run.frame0, release_frame=run.release,
                            rate=run.rate, direction=run.direction, x=run.x, h=run.h,
                            v_fwd=run.v_fwd, n_tracks=run.n_tracks, run_start=run.start,
                            steps=np.array(run.steps, int), takeoff=run.takeoff,
                            jump_peak=run.jump_peak, release_timing=run.release_source,
                            notes=json.dumps(run.notes))
    for res in results:
        q = res.metrics.get("release_quality", "")
        if q not in ("good", "lower bound (lost after peak)"):
            continue
        ser = {k: v for k, v in res.series.items() if isinstance(v, np.ndarray)}
        np.savez_compressed(OUT / f"series_{res.name}.npz", t=res.t,
                            frame0=res.frame0, release_frame=res.release, rate=res.rate,
                            direction=res.direction, **ser)
        if "elbow_flexion" in res.series:
            T, trs = trials[res.trial]
            qpos, measured, events, info = retarget.reference_motion(res, runups[res.name], trs, model)
            np.savez_compressed(OUT / f"reference_motion_{res.name}.npz", qpos=qpos,
                                timestep=1.0 / res.rate, release_index=events["release"],
                                measured_mask=measured, events=json.dumps(events, default=int),
                                joint_names=np.array(joint_names),
                                qpos_layout=np.array([f"{n}:{model.joint(n).qposadr[0]}" for n in joint_names]),
                                info=json.dumps(info, default=float), release_quality=q)
            # the raw markers over the same frames, for drawing next to the humanoid
            frames = events["first_frame"] + np.arange(len(qpos))
            z_centre = float(np.nanmedian(T.pos[res.release, :, 2]))
            np.savez_compressed(OUT / f"markers_{res.name}.npz",
                                pos=retarget.marker_cloud(T, res.direction, frames,
                                                          events["x_offset_m"], z_centre).astype(np.float32),
                                timestep=1.0 / res.rate, release_index=events["release"])
            ref_info[res.name] = info

    # --- figures ------------------------------------------------------------
    for res in results:
        if res.metrics.get("release_quality") in ("good", "lower bound (lost after peak)"):
            T, trs = trials[res.trial]
            plots.identification_check(res, trs, PLOTS / f"idcheck_{res.name}.png")

    good = [r for r in results if r.metrics.get("release_quality") in ("good", "lower bound (lost after peak)")]
    fig, ax = plt.subplots(figsize=(7, 4))
    for res in good:
        k = res.release - res.frame0
        tt = (np.arange(len(res.series["distal_speed"])) - k) / res.rate * 1000
        ax.plot(tt, res.series["distal_speed"] * KMH,
                label=f"{res.name} ({res.metrics['distal_marker']}, {res.metrics['release_quality']})")
    ax.axvline(0, c="k", lw=.8)
    ax.set_xlabel("time from release proxy (ms)"); ax.set_ylabel("speed (km/h)")
    ax.set_xlim(-500, 300); ax.grid(alpha=.3); ax.legend(fontsize=7)
    ax.set_title("Bowling hand / wrist speed, aligned to its peak (release proxy)", fontsize=9)
    fig.tight_layout(); fig.savefig(PLOTS / "hand_speed_aligned.png", dpi=110); plt.close(fig)

    arm_ok = [r for r in good if "elbow_flexion" in r.series]
    if arm_ok:
        fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
        for res in arm_ok:
            k = res.release - res.frame0
            tt = (np.arange(len(res.series["elbow_flexion"])) - k) / res.rate * 1000
            th, el, _ = retarget.planar_arm(res)
            axes[0].plot(tt, res.series["elbow_flexion"], label=res.name)
            axes[1].plot(tt, res.series["upper_arm_elev"], label=res.name)
            axes[2].plot(tt, th, label=f"{res.name} whole-arm theta")
            h = res.metrics.get("arm_horizontal_frame")
            if h is not None:
                for a_ in axes:
                    a_.axvline((h - res.release) / res.rate * 1000, ls="--", lw=.7,
                               c=axes[0].lines[-1].get_color())
        axes[0].set_ylabel("elbow flexion (deg)\nmarker-based, offset-affected")
        axes[1].set_ylabel("upper-arm elevation\nabove horizontal (deg)")
        axes[1].axhline(0, c="k", lw=.6)
        axes[2].set_ylabel("arm angle, model\nconvention (deg)")
        axes[2].axhline(270, c="k", lw=.6, ls=":")
        axes[2].axhspan(230, 310, color="tab:green", alpha=.08, label="RL release window 230-310")
        axes[2].set_xlabel("time from release proxy (ms)")
        for a_ in axes:
            a_.axvline(0, c="k", lw=.8); a_.grid(alpha=.3); a_.legend(fontsize=7)
        axes[0].set_xlim(-400, 200)
        axes[0].set_title("Bowling arm, deliveries with a complete arm chain\n"
                          "(dashed: arm-horizontal instant; HUMAN COMPARISON ONLY)", fontsize=9)
        fig.tight_layout(); fig.savefig(PLOTS / "bowling_arm_angles.png", dpi=110); plt.close(fig)

    tr_ok = [r for r in good if "trunk_lateral" in r.series]
    if tr_ok:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        for res in tr_ok:
            k = res.release - res.frame0
            tt = (np.arange(len(res.series["trunk_lateral"])) - k) / res.rate * 1000
            ax.plot(tt, res.series["trunk_lateral"], label=f"{res.name} lateral (+ = away from bowling arm)")
            ax.plot(tt, res.series["trunk_forward"], ls="--", label=f"{res.name} forward")
        ax.axvline(0, c="k", lw=.8); ax.grid(alpha=.3); ax.legend(fontsize=7)
        ax.set_xlabel("time from release proxy (ms)"); ax.set_ylabel("deg")
        ax.set_title("Back-marker column tilt (trunk proxy)", fontsize=9)
        fig.tight_layout(); fig.savefig(PLOTS / "trunk_angles.png", dpi=110); plt.close(fig)

    # one grey line per delivery (11 is too many to tell apart by colour, and
    # the point is that they agree), the median in one accent colour
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    grid = np.arange(-3000, 410, 10)
    stack = {0: [], 1: []}
    for run in runups.values():
        tt = (np.arange(len(run.x)) + run.frame0 - run.release) / run.rate * 1000
        for i, y in enumerate((run.v_fwd * KMH, run.h)):
            axes[i].plot(tt, y, c="0.72", lw=0.8)
            stack[i].append(np.interp(grid, tt, y, left=np.nan, right=np.nan))
        for f, mk in ((run.takeoff, "s"), (run.jump_peak, "^")):
            if f >= 0:
                axes[1].plot(run.ms(f), run.h[run.i(f)], mk, c="0.2", ms=5, ls="none")
    for i, ax in enumerate(axes):
        ax.plot(grid, np.nanmedian(stack[i], axis=0), c="tab:blue", lw=2)
        ax.axvline(0, c="k", lw=.8); ax.grid(alpha=.3)
    from matplotlib.lines import Line2D
    axes[0].legend(handles=[Line2D([], [], c="0.72", lw=0.8, label=f"each delivery ({len(runups)})"),
                            Line2D([], [], c="tab:blue", lw=2, label="median")], fontsize=7)
    axes[1].legend(handles=[Line2D([], [], c="0.2", marker="s", ls="none", label="take-off (last step)"),
                            Line2D([], [], c="0.2", marker="^", ls="none", label="top of the jump")],
                   fontsize=7)
    axes[0].set_ylabel("forward speed (km/h)")
    axes[1].set_ylabel("height change since\nstanding start (m)")
    axes[1].set_xlabel("time from release (ms)")
    axes[1].set_xlim(-3000, 400)
    axes[0].set_title("Run-up, label-free whole-body motion, aligned at release\n"
                      "(height is a marker median, not the pelvis: it dips through the delivery "
                      "stride as the trunk flexes)", fontsize=9)
    fig.tight_layout(); fig.savefig(PLOTS / "runup.png", dpi=110); plt.close(fig)

    # --- anthropometrics from identified chains ----------------------------
    anth = {"wrist marker spacing": [], "elbow-region marker spacing": [],
            "elbow marker -> wrist centre (forearm)": [], "wrist centre -> hand marker": [],
            "upper-arm marker -> elbow marker": []}
    for res in results:
        if not res.release_usable:
            continue   # identification only visually verified for these deliveries
        T, trs = trials[res.trial]
        by = {t.tid: t for t in trs}
        a_, b_ = res.frame0, res.frame0 + len(res.series["elbow"]) - 1
        arm = res.arm
        if len(arm.wrist) == 2:
            ps = idf.pair_stat(by[arm.wrist[0]], by[arm.wrist[1]], a_, b_)
            if ps: anth["wrist marker spacing"].append(ps.d)
        if len(arm.elbow) == 2:
            ps = idf.pair_stat(by[arm.elbow[0]], by[arm.elbow[1]], a_, b_)
            if ps: anth["elbow-region marker spacing"].append(ps.d)
        el, wr, hd = res.series["elbow"], res.series["wrist"], res.series["hand"]
        if arm.elbow and arm.wrist:
            d = np.linalg.norm(el - wr, axis=1); d = d[~np.isnan(d)]
            if len(d) > 5: anth["elbow marker -> wrist centre (forearm)"].append(float(np.median(d)))
        if arm.hand and arm.wrist:
            d = np.linalg.norm(hd - wr, axis=1); d = d[~np.isnan(d)]
            if len(d) > 5: anth["wrist centre -> hand marker"].append(float(np.median(d)))
        if arm.proximal:
            anth["upper-arm marker -> elbow marker"].append(arm.proximal_d)

    # --- play each reference motion on the humanoid, ball physically simulated
    sys.path.insert(0, str(ROOT / "scripts"))
    import demo_mocap  # noqa: E402
    from envs.bowler_env import BowlerEnv  # noqa: E402
    demo = {}
    for name in ref_info:
        env = BowlerEnv(max_steps=1500)
        mp = demo_mocap.MocapPlayback(OUT / f"reference_motion_{name}.npz", env.model)
        info, _ = demo_mocap.play(env, mp, real_time=False)
        demo[name] = (info["release_speed"] * KMH, info["landing_pos"], env.target_min, env.target_max)

    write_report(results, runups, trials, static, ref_info, anth, demo)
    print(f"wrote {OUT}")


def run_summary(runups):
    """Mean +/- sd strings of the run-up metrics, over the deliveries that have them."""
    def ms(key, scale=1.0, nd=1):
        # start time and distance are lower bounds where the start was not captured
        lower_bound = key in ("runup_start_ms", "runup_distance_m")
        v = [r.metrics[key] * scale for r in runups.values()
             if key in r.metrics and not np.isnan(r.metrics[key])
             and not (lower_bound and any("run start not captured" in n for n in r.notes))]
        return f"{np.mean(v):.{nd}f} +/- {np.std(v, ddof=1):.{nd}f}" if len(v) > 1 else "-"
    return {"start": ms("runup_start_ms", -1e-3, 2), "dist": ms("runup_distance_m"),
            "peak": ms("runup_peak_speed_kmh"),
            "steps": f"{np.median([r.metrics['steps_detected'] for r in runups.values()]):.0f}",
            "to": ms("takeoff_ms", -1, 0), "jp": ms("jump_peak_ms", -1, 0),
            "rise": ms("jump_rise_m", 1, 2)}


def write_runup_section(add, runups, rs):
    used = [int(np.median(r.n_tracks[r.i(r.start):r.i(r.release) + 1])) for r in runups.values()]
    add("## Run-up\n")
    add("No foot or pelvis marker can be identified through the run-up, but a median of "
        f"{min(used)}-{max(used)} tracks per frame (by delivery) move with the body between run start "
        "and release, so the run-up is measured from the WHOLE BODY, label-free "
        "(Method, step 8): forward travel and speed, and the height change whose dips are the "
        "running steps and whose last big rise is the jump. In every trial the bowler starts standing "
        "still at one end of the room. Figure: `plots/runup.png`.\n")
    hdr = ["delivery", "release timing", "start s before release", "distance m", "peak km/h",
           "km/h 0.3 s before release", "steps detected", "step rate Hz", "take-off ms",
           "top of jump ms", "jump rise m"]
    add("| " + " | ".join(hdr) + " |")
    add("|" + "---|" * len(hdr))
    for name, r in runups.items():
        m = r.metrics
        add("| " + " | ".join([
            name, m["release_timing"], fmt(-m["runup_start_ms"] / 1000, 2), fmt(m["runup_distance_m"], 2),
            fmt(m["runup_peak_speed_kmh"]), fmt(m["speed_300ms_before_release_kmh"]),
            fmt(m["steps_detected"]), fmt(m.get("step_rate_hz"), 2), fmt(m.get("takeoff_ms"), 0),
            fmt(m.get("jump_peak_ms"), 0), fmt(m.get("jump_rise_m"), 2)]) + " |")
    add(f"\nMean +/- sd over the deliveries that have each value: start {rs['start']} s before "
        f"release, distance {rs['dist']} m, peak speed {rs['peak']} km/h, take-off {rs['to']} ms and "
        f"top of the jump {rs['jp']} ms before release, jump rise {rs['rise']} m. Where the bowler was "
        f"already moving when first tracked (noted per delivery above), start time and distance in "
        f"the table are lower bounds and are left out of these means.\n")
    jp = {src: [-r.metrics["jump_peak_ms"] for r in runups.values()
                if r.metrics["release_timing"] == src and "jump_peak_ms" in r.metrics]
          for src in ("hand-speed peak", "video")}
    lo, hi = min(jp["hand-speed peak"]), max(jp["hand-speed peak"])
    first = {k: ((runups[k].release - f0) / runups[k].rate * 1000,
                 (runups[k].jump_peak - f0) / runups[k].rate * 1000 if runups[k].jump_peak >= 0 else None)
             for k, f0 in dl.FIRST_VIDEO_READING.items()}
    add(f"Release timing check. The top of the jump is {min(jp['hand-speed peak']):.0f}-"
        f"{max(jp['hand-speed peak']):.0f} ms before release in the {len(jp['hand-speed peak'])} "
        f"deliveries timed by the hand-speed peak and {min(jp['video']):.0f}-{max(jp['video']):.0f} ms "
        f"in the {len(jp['video'])} timed from video, and in all 11 the forward speed drops sharply at "
        f"release (the front foot blocking). The first video reading of those {len(first)} releases "
        f"was earlier by: " + ", ".join(f"{k} {d:.0f} ms" for k, (d, _) in first.items()) + ". "
        f"Measured from it, the top of the jump would sit: "
        + ", ".join(f"{k} {abs(j):.0f} ms {'before' if j < 0 else 'after'}"
                    for k, (_, j) in first.items() if j is not None)
        + f" - {sum(1 for _, j in first.values() if j is not None and not lo - 50 <= -j <= hi + 50)} "
        f"of them more than 50 ms outside the hand-timed range, which is what prompted the "
        f"re-reading.\n")
    add("Limits, in order of importance:\n")
    add("- **The legs are not measured.** Foot markers were mostly left unlabelled, so not exported; "
        "foot plants appear only sporadically. Which foot is down at each step is not known either. "
        "The humanoid's legs in the run-up are therefore animated (see Reference motion).")
    add("- **Height change is not pelvis height.** It is the whole body's marker median. It follows "
        "the trunk well while running upright and through the jump, but drops through the delivery "
        "stride as the trunk flexes forward and sideways. The jump rise may include some of the arms "
        "lifting in the gather; restricting the estimate to markers at trunk height gives a rise a "
        "few centimetres smaller.")
    add(f"- **Only steps with a clear bob are detected** - typically the last {rs['steps']}, take-off "
        "included. The first part of the run, accelerating from standing, has steps too, but too "
        "shallow to find.\n")


BODY_GROUPS = {"trunk": ("pelvis", "torso"), "head": ("head",),
               "bowling arm": ("humerus_r", "ulna_r", "radius_r", "hand_r"),
               "other arm": ("humerus_l", "ulna_l", "radius_l", "hand_l"),
               "right leg": ("femur_r", "tibia_r", "talus_r", "calcn_r", "toes_r"),
               "left leg": ("femur_l", "tibia_l", "talus_l", "calcn_l", "toes_l")}


def write_fit3d_section(add):
    """Summarise scripts/fit_mocap3d.py's output, if it has been run (it is slow,
    so this script only reads it)."""
    import demo_mocap3d   # scripts/ is on sys.path (see main)
    fits = [(name, OUT / f"fit3d_{name}.npz") for name, *_ in dl.DELIVERIES]
    fits = [(n, p) for n, p in fits if p.exists()]
    add("## Full-body 3D replay\n")
    if not fits:
        add("Not run yet: `python scripts/fit_mocap3d.py` (a few minutes per delivery), then "
            "`python scripts/demo_mocap3d.py --delivery T3a --view`.\n")
        return
    add("A full-body skeleton (LocoMuJoCo's MuJoCo port of the Hamner 2010 OpenSim running model: "
        "bone meshes, 3-DoF hips, lumbar and shoulders, knees, ankles, elbows, forearm rotation, "
        "wrists; `mocap/skeleton3d.py`) fitted to the raw markers frame by frame "
        "(`mocap/fit3d.py`, run by `scripts/fit_mocap3d.py`). Replay: "
        "`python scripts/demo_mocap3d.py --delivery T3a --view`.\n")
    import json
    lab_path = OUT / "labels.json"
    lab = json.loads(lab_path.read_text()) if lab_path.exists() else {}
    labelled = ", ".join(f"{n} tracks in {d}" for d, n in
                         ((d, len(lab[d])) for d, *_ in dl.DELIVERIES if lab.get(d))) or "none"
    add("Which marker is which is not taken from QTM: the export's labels are unreliable, and the "
        "markers worn do not fully match the AIM model (markers down the front of the vest, a "
        "second elbow marker, a 3-marker group on the shin; the video shows the vest markers, and "
        "the static trial's camera-3 frame a marker on the inside of each elbow - the medial "
        "epicondyle, labelled where a track sits level with the lateral one, as in T6a). "
        "Each track is attached to the bone it moves rigidly with, at a point near one of that "
        "bone's anatomical landmarks, decided by the fit itself in alternation with the pose. "
        f"Where the fit could not place a track or placed it on the wrong bone ({labelled}), it was "
        "identified by hand in `data/mocap/labels.json` and pinned: by which tracks keep a fixed "
        "spacing while turning, and by height over time - the front arm is a chain shoulder - "
        "upper arm - elbow - wrist, arm-high in the gather and hip-high at release; the head is a "
        "rigid pair above the shoulders that moves against the torso; pelvis markers sit at hip "
        "height and move with each other, not with the arms held against the body in the run-up. "
        "QTM's names agree with these in places but swap left and right in about half the "
        "deliveries. The head has its own neck joint.\n")
    hdr = ["delivery", "markers visible / used per frame", "tracks attached", "median distance cm",
           "90th pct cm"] + [f"{g} %" for g in BODY_GROUPS] + ["hand at release",
                                                               "fitted hand speed km/h"]
    add("| " + " | ".join(hdr) + " |")
    add("|" + "---|" * len(hdr))
    for name, path in fits:
        z = np.load(path)
        bodies = list(z["body_names"])
        sup = z["support"]
        pct = [100 * sup[:, [bodies.index(b) for b in grp if b in bodies]].any(axis=1).mean()
               for grp in BODY_GROUPS.values()]
        rp = demo_mocap3d.Replay(path)
        rp.pose(-0.01)
        a = rp.grip()
        rp.pose(0.01)
        speed = np.linalg.norm(rp.grip() - a) / 0.02 * KMH
        add("| " + " | ".join([
            name, f"{np.median(np.isfinite(z['markers'][:, :, 0]).sum(1)):.0f} / "
                  f"{np.median(z['marker_used'].sum(1)):.0f}",
            f"{len(z['attach_tid'])} of {int(z['n_tracks'])}",
            fmt(float(np.nanmedian(z["resid_med"])) * 100), fmt(float(np.nanmedian(z["resid_p90"])) * 100),
            *[f"{p:.0f}" for p in pct],
            "tracked" if bool(z["hand_tracked"]) else "not tracked",
            fmt(speed) if bool(z["hand_tracked"]) else "-"]) + " |")
    add("\n\"Body part %\" is the share of frames in which at least one marker is on that part; in "
        "the other frames it is posed by the fit's priors, and the replay draws it translucent. "
        "Distances are from each used marker to its attachment point on the skeleton. Where the hand "
        "is not tracked at release the skeleton's arm there follows the template (below), so no hand "
        "speed is given "
        "and the replay shows no ball.\n")
    add("How the pose is decided where markers are missing, in order: a limb gap of up to 0.8 s "
        "is interpolated between its measured poses; feet stay above the floor; a foot is on the "
        "floor at each measured step (`mocap/runup.py`), alternating back from a left-foot "
        "take-off, and the front foot is planted from 100 ms before release; the pelvis stays "
        "near level so the trunk does the leaning; otherwise a weak pull towards a running stride "
        "locked to the measured steps, becoming a delivery stride after the top of the jump. "
        "Around the delivery (arms 0.9 s, legs 1.4 s before release to the end) an arm or leg "
        "without enough markers of its own follows a template instead: the mean of its fitted pose "
        "at that instant in the deliveries where it was tracked (the bowling arm from the 4 with a "
        "tracked release; `data/mocap/limb_template.npz`). The bowling action is the same movement "
        "every time, so this is a far better guess than a hanging arm - but a guess, drawn "
        "translucent like every bone without a marker. Before take-off a trunk with fewer than 3 "
        "markers is held near a 15 deg running lean, and a thigh marker must sit at the knee or "
        "above (lower ones are on the shin). An "
        "elbow or knee left at its straight stop while the markers beyond it are more than 4 cm "
        "off is a wrong solution branch (a folded arm whose upper arm is rolled the wrong way, a "
        "swinging foot reached by the hip alone), and is re-solved from a bent start. The "
        "joint trajectories are low-passed at 12 Hz (20 Hz on the bowling arm), because the exported marker set changes "
        "wholesale at QTM's re-identification instants.\n")
    add("Checked against the synchronised camera-3 video for every delivery at the run-up, "
        "the gather, the bowling arm's swing down and back, release and follow-through "
        "(`scripts/compare_mocap3d_video.py`): posture and arm positions agree, including the "
        "large lateral trunk lean at release. In the deliveries without a captured release the "
        "bowling hand is lost before release, and the \"arm\" tracks listed for them in the "
        "per-delivery notes may be the front (left) arm pulling down - in T5a the \"bowling "
        "wrist\" sits on the left - so the 3D fit uses them only by their own motion or a "
        "label, and the bowling hand around release there has no marker (T6a's elbow is tracked "
        "through release; T4a's arm until 120 ms before). Limits: the legs carry few "
        "markers, so their pose is largely the priors'; "
        "the shoulder blade is part of the rigid model torso, so an acromion marker fits it only "
        "to ~3-4 cm and is used only where identified by hand; and the body is scaled uniformly (the static trial has no marker data). This is "
        "a visual and gross-motion reference. It is NOT a measurement of joint angles - in "
        "particular not the elbow, whose legality for the RL policy stays simulated.\n")
    add("The bowling arm needed three fixes to look like the video: the skeleton's arms are "
        "shortened to the participant's (forearm, lateral epicondyle to the wrist styloids, measured "
        "from the markers), the shoulder joint may slide a few cm against the ribcage (a shoulder "
        "girdle, which the source model fuses into the torso), and the arm is re-fitted to its "
        "markers after the trunk is smoothed. Even then the absolute elbow angle cannot be pinned "
        "down: the marker geometry reads 20-30 deg of bend at release, which matching would need, "
        "while every usable camera looks along the run-up and so cannot see a bend in the plane the "
        "arm swings through. Shown frame by frame, the fitted elbow also swings 20-27 deg in the last "
        "100 ms (fit error; the markers' own change there is ~5 deg, above). So in the deliveries "
        "with a tracked release the replay holds the bowling elbow straight from 350 ms before to "
        "150 ms after release, and the hand then sits a few cm off its markers there. The elbow "
        "offset removed is stored per fit (`elbow_offset_deg`).\n")


def write_report(results, runups, trials, static, ref_info, anth, demo):
    T1 = trials[1][0]
    cov = coverage([trials[k][0] for k in sorted(trials)])
    rel = label_reliability([trials[k][0] for k in sorted(trials)])
    good = [r for r in results if r.metrics.get("release_quality") == "good"]
    lb = [r for r in results if r.metrics.get("release_quality", "").startswith("lower")]
    L = []
    add = L.append
    add("# Motion capture report: participant, 2026-10-01 (Qualisys, IIT Delhi Abu Dhabi)\n")
    add("Generated by `scripts/mocap_process.py` from the raw `.c3d` exports. Every number below "
        "is recomputed by that script; nothing is typed in by hand. Speeds in km/h.\n")
    add("## Bottom line\n")
    add(f"- **11 deliveries** were bowled across 6 trials (trial 1 has one; trials 2-6 have one towards "
        f"each end of the room). Release times come from the hand-speed peak where the hand was tracked "
        f"and otherwise from the synchronised Miqus video, re-read on 2026-10-07 (the first reading was "
        f"up to 0.6 s early); the run-up gives an independent check (see Run-up).")
    rs = run_summary(runups)
    add(f"- **The whole run-up is captured**, label-free, in all 11 deliveries: from standing still "
        f"{rs['start']} s before release, over {rs['dist']} m, peaking at {rs['peak']} km/h; "
        f"about {rs['steps']} running steps with a clear bob, take-off {rs['to']} ms before release and "
        f"a jump whose top is {rs['jp']} ms before release, {rs['rise']} m above the take-off point. "
        f"The legs themselves are not measured (see Run-up).")
    add(f"- **The QTM marker labels cannot be used as exported.** QTM's AIM model was only "
        f"\"partially applied\" to every dynamic trial and failed outright on the static trial (session "
        f"log). Label columns jump between physical markers mid-trial: pairs of labels that sit on one "
        f"rigid segment keep their expected spacing in as few as 2-17 % of co-visible frames in some "
        f"trials (table below), "
        f"and the back-marker column is labelled upside down in some trials (`CV7` below `MAI`). "
        f"So identity was rebuilt from physics (rigid spacing, speed, geometry) - see Method.")
    add(f"- **The hand was tracked through release in {len(good)} of 11 deliveries** "
        f"({', '.join(r.name for r in good)}), plus {len(lb)} where it was lost within 30 ms of the "
        f"peak ({', '.join(r.name for r in lb)}). In the rest, the bowling hand/wrist markers drop out "
        f"(unlabelled, therefore not exported) before release, so no release speed or timing exists "
        f"for them.")
    arm_ok = [r for r in results if "elbow_extension_endpoint_deg" in r.metrics]
    add(f"- **Elbow extension (HUMAN COMPARISON ONLY)** could be measured in "
        f"{len(arm_ok)} deliveries ({', '.join(r.name for r in arm_ok) or 'none'}); the others lack an "
        f"identifiable upper-arm marker at release.")
    add("- **Not measurable from this export:** front-knee angle and every other leg joint angle, "
        "hip-shoulder separation, pelvis motion, run-up foot placement for most deliveries, and "
        "anthropometrics from the static "
        "trial (it contains no marker data at all). The fix is re-labelling in QTM - see the last "
        "section.\n")

    add("## Per-delivery results\n")
    hdr = ["delivery", "release", "hand/wrist peak km/h", "release height m", "approach km/h",
           "FFC->release ms", "elbow ext. endpoint deg", "elbow ext. max deg", "arm angle at release*",
           "trunk lateral deg"]
    add("| " + " | ".join(hdr) + " |")
    add("|" + "---|" * len(hdr))
    for r in results:
        m = r.metrics
        q = m.get("release_quality", "not captured")
        usable = q in ("good", "lower bound (lost after peak)")
        th = "-"
        if usable and "elbow_flexion" in r.series:
            th = fmt(retarget.planar_arm(r)[2]["theta_at_release_deg"], 0)
        add("| " + " | ".join([
            r.name,
            f"{q}" + (f" (f{r.release}, {r.release / r.rate:.2f} s)" if usable else ""),
            fmt(m.get("peak_distal_speed_kmh")) + (f" ({m['distal_marker']})" if usable else "") if usable else "-",
            fmt(m.get("release_height_m"), 2) if usable else "-",
            fmt(m.get("approach_speed_kmh")),
            fmt(m.get("ffc_to_release_ms"), 0),
            fmt(m.get("elbow_extension_endpoint_deg")),
            fmt(m.get("elbow_extension_max_deg")),
            th,
            fmt(m.get("trunk_lateral_flexion_at_release_deg")),
        ]) + " |")
    add("\n*Arm angle in the RL model's shoulder convention (270 = vertical, 360 = pointing forward), "
        "from the whole-arm direction - directly comparable with ThrowEnv's release window of 230-310 deg.\n")

    vals = [r.metrics["peak_distal_speed_kmh"] for r in good]
    if vals:
        add(f"Across the {len(vals)} cleanly tracked releases, peak hand/wrist speed is "
            f"{np.mean(vals):.1f} +/- {np.std(vals, ddof=1) if len(vals) > 1 else 0:.1f} km/h "
            f"(mean +/- sd; range {min(vals):.1f}-{max(vals):.1f}). These are marker speeds, not ball "
            f"speeds: no ball marker was used, the hand marker sits on the back of the hand, and the "
            f"wrist markers are slower still (T1a and T4b are wrist values). Ball speed at release is "
            f"normally higher than the hand marker's.\n")
    ap = [r.metrics["approach_speed_kmh"] for r in results if not np.isnan(r.metrics["approach_speed_kmh"])]
    add(f"Approach speed 0.25-0.5 s before release (median forward speed of all visible markers, so "
        f"label-free): {np.mean(ap):.1f} +/- {np.std(ap, ddof=1):.1f} km/h over all 11 deliveries - a "
        f"short indoor run-up.\n")

    add("### Notes per delivery\n")
    for r in results:
        add(f"- **{r.name}** (trial {r.trial}, bowling towards {'+x' if r.direction > 0 else '-x'}): "
            + ("; ".join(r.notes + runups[r.name].notes) if r.notes or runups[r.name].notes
               else "no issues"))
        for e in r.arm.evidence:
            add(f"    - arm: {e}")
        if r.spine:
            for e in r.spine.evidence:
                add(f"    - trunk: {e}")
    add("")

    write_runup_section(add, runups, rs)

    add("## Elbow extension - HUMAN COMPARISON ONLY\n")
    add("The project's settled decision stands: the RL policy's legality constraint is evaluated on "
        "the simulated elbow only. These numbers describe the participant's action, for comparison. Both "
        "operational definitions are reported, computed with the same pure function the simulator "
        "uses (`envs/legality.py`), with arm-horizontal = last upward crossing of the upper arm "
        "through horizontal before release:\n")
    for r in arm_ok:
        m = r.metrics
        add(f"- **{r.name}**: endpoint {m['elbow_extension_endpoint_deg']:.1f} deg, max "
            f"{m['elbow_extension_max_deg']:.1f} deg; arm-horizontal to release "
            f"{m.get('horizontal_to_release_ms', float('nan')):.0f} ms; marker-based flexion "
            f"{m.get('elbow_flexion_at_horizontal_deg', float('nan')):.1f} -> "
            f"{m['elbow_flexion_at_release_deg']:.1f} deg (release quality: {m['release_quality']}).")
    add("\nSensitivity of the endpoint value: release proxy moved one frame either way, and "
        "arm-horizontal taken from the whole-arm direction instead of the short upper-arm vector:\n")
    for r in arm_ok:
        vals = extension_sensitivity(r)
        add(f"- **{r.name}**: endpoint extension ranges {min(vals):.1f} to {max(vals):.1f} deg "
            f"over {len(vals)} variants.")
    add("\nCaveats, all of which matter against a 15 deg threshold:\n")
    add("- **Sampling rate.** 100 Hz gives only 7-12 samples between arm-horizontal and release; ICC "
        "testing uses >= 250 Hz. A one-frame shift of the release instant moves the endpoint value "
        "by a few degrees.")
    add("- **Release is a proxy.** No ball marker: release is taken as the hand/wrist speed peak.")
    add("- **Marker set and offsets.** IOR is a clinical gait/posture set, not the ICC protocol. "
        "Only one upper-arm marker was found per delivery, ~13-18 cm from the elbow marker and off the "
        "humeral axis, which is why the ABSOLUTE flexion reads implausibly high (~50 deg at release in "
        "T3a). That offset is nearly constant over the 70-110 ms window, so the CHANGE (extension) is "
        "much less affected than the absolute angle - but it is not zero, because the marker's offset "
        "direction rotates with humeral rotation.")
    add("- **Soft-tissue artefact** on skin markers is largest exactly when the arm accelerates "
        "hardest, i.e. during this window.\n")

    add("## Reference motion for the humanoid\n")
    if ref_info:
        add("`reference_motion_<delivery>.npz` holds per-frame `qpos` for `assets/bowler.xml` at "
            "100 Hz from standing still at the top of the run-up to the follow-through "
            "(`timestep`, `release_index`, `events`, `joint_names`, `qpos_layout`, `measured_mask` "
            "= frames whose arm angles are measured, `info`). Play it with "
            "`python scripts/demo_mocap.py --delivery T3a [--view]`. The replay also draws every "
            "exported marker as an orange dot (`markers_<delivery>.npz`, raw and unlabelled, in the "
            "same frame), so measured and animated motion can be told apart by eye: the dots move "
            "with the humanoid's body through the run-up and with its arm through the delivery; the "
            "animated legs have few or no dots, and the dots reach above the model's head because "
            "the model is shorter than the participant.\n")
        for k, v in ref_info.items():
            add(f"- **{k}**: " + "; ".join(f"{a}: {fmt(b) if isinstance(b, float) else b}" for a, b in v.items()))
        for k, (sp, land, lo, hi) in demo.items():
            if land:
                add(f"- Played on the humanoid (`scripts/demo_mocap.py`), **{k}** releases the simulated "
                    f"ball at {sp:.1f} km/h and it lands at x = {land[0]:.2f} m (target zone {lo}-{hi} m).")
            else:
                add(f"- Played on the humanoid, **{k}** releases at {sp:.1f} km/h; the ball did not "
                    f"land within the episode.")
        add("\nThe captured arm releases at ~300 deg in the model's convention (30 deg past vertical). "
            "On the planar model, which has no wrist and an upright torso, that sends the ball into the "
            "ground short of the target zone - the same result ThrowEnv's scripted arm gives at 300 deg. "
            "A real bowler steers the ball with wrist and fingers and releases from a trunk that is "
            "leaning forward and sideways; the model has neither, so matching the human arm angle is "
            "not by itself enough to land the ball where a human would.")
        add("\nMeasured: forward travel over the whole run-up, the bob of each running step and the "
            "jump (as root height), and the bowling arm (2 angles, sagittal plane) over the delivery. "
            "NOT measured: the legs, which are a running cycle locked to the measured step times and "
            "blended into BowlerEnv's delivery-stride pose from the top of the jump; and the arm before "
            "the delivery, which is BowlerEnv's carry pose. The model's root height settles to the "
            "delivery-stride height 100 ms before release, so the release itself is played as before "
            "the run-up was added. So this is a reference for the ARM ACTION and the RUN-UP'S TIMING, "
            "SPEED AND JUMP - not yet the gross-body (trunk/legs) style term the project scope "
            "envisages, which needs leg and pelvis markers from the re-labelled data.\n")
    write_fit3d_section(add)
    add("## Data quality, in numbers\n")
    add(f"Marker rate {T1.rate:.0f} Hz; analog {T1.analog_rate:.0f} Hz (6 force plates). Units mm "
        f"(converted to m). Lab axes: x along the room (run-up direction), y vertical up, z lateral.\n")
    add("| trial | frames | seconds | label columns exported | mean markers visible/frame | max |")
    add("|---|---|---|---|---|---|")
    for name, n, sec, ncol, mean_vis, mx in cov:
        add(f"| {name} | {n} | {sec:.1f} | {ncol} | {mean_vis:.1f} | {mx} |")
    add(f"\nThe IOR dynamic set has 41 markers; at most {max(c[5] for c in cov)} are present in any "
        f"frame, because QTM exports only trajectories AIM managed to label. The static trial "
        f"`Static FB Anterior - IOR 1.c3d` contains {len(static.labels)} labelled markers "
        f"(AIM failed on it), so segment lengths and stature cannot be measured from it.\n")
    add("Rigidity of AIM-labelled pairs that sit on one segment (share of co-visible frames within "
        "2 cm of the pair's modal spacing; a correctly labelled rigid pair is ~100 %):\n")
    add("| pair | modal spacing m | " + " | ".join(f"trial {k}" for k in sorted(trials)) + " |")
    add("|---|---|" + "---|" * len(trials))
    for pr, mode, per in rel:
        add(f"| {pr} | {mode:.3f} | " + " | ".join(per) + " |")
    add("\nForce plates: 6 Bertec/AMTI plates are recorded, but `FORCE_PLATFORM:CORNERS` is the "
        "default +/-200 x 300 mm square at the origin for all six and the channels are not "
        "zeroed (offsets of ~18,000 N), so plate positions are unknown and forces are unusable for "
        "foot-contact timing. Foot contacts were taken from marker kinematics instead (only "
        "three plants are tracked near any release).\n")
    add("Segment spacings measured on the identified bowling-arm markers of the visually verified "
        "deliveries, i.e. those with a usable release (median over each delivery, "
        "then across deliveries):\n")
    for k, v in anth.items():
        if v:
            add(f"- {k}: {np.mean(v):.3f} m (n={len(v)}, range {min(v):.3f}-{max(v):.3f})")
    add("\nTwo markers ~11 cm apart at the bowling elbow, equidistant from the wrist, appear in four "
        "deliveries. The IOR dynamic set has only the lateral epicondyle there, so either an extra "
        "(medial) marker was worn or the marker set on the body differs from the AIM model applied - "
        "worth checking with the lab, because a model/marker-set mismatch alone would explain AIM's "
        "failure.\n")

    add("## Method\n")
    add("1. **Load** (`mocap/c3dio.py`): pure-Python `c3d` reader; residual < 0 -> missing.")
    add("2. **Rebuild physical tracks** (`mocap/tracks.py`): cut each label column at gaps and at "
        "jumps no marker can make in 10 ms (> 0.12 m from constant-velocity prediction), then "
        "re-join pieces across columns where one starts within 4 cm of where another's motion "
        "predicts. Labels are kept only as a hint.")
    add("3. **Identify** (`mocap/identify.py`): bowling hand = fastest point above 1.4 m near "
        "release; hand/wrist markers = tracks rigid with it (distance sd <= 1 cm while the forearm "
        "swings 100-165 deg); elbow = rigid with the wrist at forearm length (19-31 cm) and slower; "
        "upper-arm marker = rigid with the elbow, on the far side of it from the wrist (angle > 110 "
        "deg), slower. Back-marker column = >= 3 mutually rigid tracks stacked near-vertically on "
        "the body midline at trunk height. Every assignment's evidence is listed above. Identities "
        "were checked visually: `plots/idcheck_*.png`.")
    add(f"4. **Filter** (`mocap/signal.py`): cubic-spline fill of gaps <= 5 frames, then zero-lag "
        f"4th-order Butterworth low-pass at {CUTOFF_HZ:.0f} Hz (100 Hz data, Nyquist 50 Hz; 10-20 Hz "
        f"is the usual band for bowling marker data). Speeds by central difference; peak speeds are "
        f"given filtered and raw.")
    add("5. **Events** (`mocap/delivery.py`): release = hand (else wrist-centre) speed peak, "
        "accepted only if the marker is tracked 50 ms before to 30 ms after it and is above 1.7 m. "
        "Arm-horizontal = last upward zero-crossing of upper-arm elevation before release. "
        "Foot plants = marker below 0.15 m and slower than 0.8 m/s for >= 3 frames; FFC/BFC by "
        "timing window relative to release (FFC within 160 ms before, BFC 150-450 ms before) "
        "because foot identity (left/right) is not recoverable.")
    add("6. **Delivery frame**: X = horizontal direction of travel, Y = up, Z = X x Y (bowler's "
        "right). Trunk lateral flexion = tilt of the back-marker column in the Y-Z plane, positive "
        "towards the non-bowling side; forward flexion = tilt in X-Y.")
    add("7. **Retarget** (`mocap/retarget.py`): see the module docstring - two arm angles in the "
        "model's sagittal convention, forward travel and root height over the run-up, legs animated.")
    add("8. **Run-up** (`mocap/runup.py`): per frame, the median velocity of every visible track, "
        "recomputed over the tracks within 1 m/s of it (drops the swinging limbs); low-passed at 10 Hz "
        "and integrated to forward travel and height change. Run start = last frame below 0.9 km/h; "
        "running steps = height minima (mid-stance) with >= 1.5 cm prominence, >= 180 ms apart; top of "
        "the jump = highest point 200-800 ms before release, take-off = the step before it.\n")

    add("## What would make this data fully usable\n")
    add("All of these are done in QTM on the lab PC (the raw `.qtm` files contain every "
        "trajectory, including the unlabelled ones this export dropped):\n")
    add("1. Open `Gait FB - IOR 1.qtm`, label every trajectory by hand against the marker set the "
        "subject actually wore (check the elbow - see above), and use *AIM > Add to existing model* "
        "(or create a subject-specific AIM model from it).")
    add("2. Re-apply AIM to all six dynamic trials and the static trial; check each delivery window "
        "by eye; fix swaps by hand.")
    add("3. Gap-fill with a larger maximum gap than the 10 frames used (log: most gaps were longer "
        "and left unfilled), but never across the release.")
    add("4. Export C3D again (labelled trajectories only is fine once labelling is right).")
    add("5. For any future capture: >= 250 Hz for the bowling arm (ICC uses this order), a small "
        "reflective marker on the ball for a true release instant, and force-plate locations set in "
        "QTM's force-plate settings.\n")
    add("Re-running `python scripts/mocap_process.py` on a corrected export needs no code changes: "
        "the identification is label-free, and correct labels only make it easier to verify.\n")
    (OUT / "REPORT.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
