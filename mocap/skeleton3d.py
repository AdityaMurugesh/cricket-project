"""Full-body 3D skeleton for replaying the captured bowling, with the IOR markers on it.

Source: LocoMuJoCo's `skeleton_torque.xml` (loco-mujoco-models, MIT), itself a
MuJoCo port of the Hamner et al. (2010) OpenSim running model: bone meshes,
3-DoF lumbar, hips and shoulders, knees, ankles, subtalar, toes, elbows,
forearm rotation and wrists. The head is part of the torso (no neck joint).

What this module changes, and why:

  * Each shoulder gets three short slides (GIRDLE_M) before its ball joint:
    a shoulder girdle, which the source fuses into the torso.
  * Hips, shoulders and the lumbar joint become BALL joints. The source models
    each as three hinges (flexion, adduction, rotation), which gimbal-lock when
    the arm is overhead -- exactly where a bowling arm spends its delivery.
  * Hinge ranges are widened to anatomical limits (the source's +/-45 deg hip
    flexion cannot reach a delivery stride; its forearm cannot fully pronate).
  * The body is kinematic: no actuators, no collisions. A free ball that
    collides only with the floor is added, as in BowlerEnv.
  * One site per IOR marker ("m_<LABEL>"), placed from the bone meshes
    themselves -- e.g. the lateral malleolus is the most lateral point of the
    bottom 4 cm of the fibula -- then pushed out ~1 cm for skin and marker
    radius (LANDMARKS below). Two medial epicondyle sites (HME) are added
    beyond the AIM model: the capture shows a second elbow marker ~11 cm from
    the lateral one (REPORT.md).

Every body frame keeps the source's OpenSim orientation in the default pose:
x forward, y along the segment (up), z to the right. The world is z-up and
the skeleton faces +x, like assets/bowler.xml.

    python -m mocap.skeleton3d            # writes assets/skeleton3d/ (needs loco-venv once)
"""
import copy
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "assets" / "skeleton3d"
LOCO_DIR = ROOT / "loco-venv" / "Lib" / "site-packages" / "loco_mujoco_models" / "skeleton"
# write_assets copies the source model and meshes here; after that nothing
# needs the loco-venv installation
SRC_DIR = OUT_DIR / "source" if (OUT_DIR / "source" / "skeleton_torque.xml").exists() else LOCO_DIR

BALL_JOINTS = {   # body: (new ball joint, the three source hinges it replaces)
    "femur_r": ("hip_r", ("hip_flexion_r", "hip_adduction_r", "hip_rotation_r")),
    "femur_l": ("hip_l", ("hip_flexion_l", "hip_adduction_l", "hip_rotation_l")),
    "torso": ("lumbar", ("lumbar_extension", "lumbar_bending", "lumbar_rotation")),
    "humerus_r": ("shoulder_r", ("arm_flex_r", "arm_add_r", "arm_rot_r")),
    "humerus_l": ("shoulder_l", ("arm_flex_l", "arm_add_l", "arm_rot_l")),
}
GIRDLE_M = 0.06    # shoulder joint travel relative to the ribcage (scapula/clavicle), each axis
BALL_LIMIT_DEG = {"hip_r": 125, "hip_l": 125, "lumbar": 70, "neck": 60}   # cone half-angle; shoulders free
NECK_POS = (-0.04, 0.50, 0.0)   # neck pivot in the torso frame (scale 1): mid-cervical, in front of the spinous processes
HINGE_RANGE_DEG = {   # widened to anatomical limits
    "knee_angle": (-145, 10), "elbow_flex": (-15, 155), "pro_sup": (-95, 95),
    "wrist_flex": (-80, 80), "wrist_dev": (-30, 40), "ankle_angle": (-55, 55),
    "subtalar_angle": (-35, 35), "mtp_angle": (-60, 60),
}
BONE_RGBA = "0.88 0.84 0.76 1"
SKIN_M = 0.01   # marker centre outside the bone: skin, soft tissue, marker radius

# (label, body, geoms, region, pick, push) -- all in the body frame, sizes in
# metres at scale 1. region: which vertices to consider; pick: which extreme.
#   lat = lateral coordinate (+z on the right side, -z on the left)
#   regions: "top"/"bottom" = within d of the max/min y; "y" = within 1.2 cm
#            of a height, on the midline; "above" = y > v; "yband" = lo < y < hi;
#            "upper"/"lower" = above/below the y median
LANDMARKS = [
    ("IAS", "pelvis", "hemi", ("upper",), "+x", "+x"),        # anterior superior iliac spine
    ("IPS", "pelvis", "hemi", ("upper",), "-x", "-x"),        # posterior superior iliac spine
    ("FTC", "femur", "femur", ("top", 0.10), "+lat", "+lat"),   # greater trochanter
    ("FLE", "femur", "femur", ("bottom", 0.06), "+lat", "+lat"),  # lateral femoral epicondyle
    ("FAX", "tibia", "fibula", ("top", 0.02), "+lat", "+lat"),    # head of the fibula
    ("TTC", "tibia", "tibia", ("band", 0.09, 0.04), "+x", "+x"),  # tibial tuberosity
    ("FAL", "tibia", "fibula", ("bottom", 0.04), "+lat", "+lat"),  # lateral malleolus
    ("FCC", "calcn", "foot", ("all",), "-x", "-x"),              # calcaneus
    ("FM1", "calcn", "foot", ("medial",), "+x", "-lat"),         # 1st metatarsal head
    ("FM5", "calcn", "foot", ("lateral",), "+x", "+lat"),        # 5th metatarsal head
    ("SAE", "torso", "hat_ribs_scap", ("above", 0.40), "+lat", "+y"),  # acromion: outermost, above the shoulder joint
    ("HUM", "humerus", "humerus", ("yband", -0.17, -0.11), "+lat", "+lat"),  # lateral upper arm
    ("HLE", "humerus", "humerus", ("bottom", 0.04), "+lat", "+lat"),  # lateral epicondyle
    ("HME", "humerus", "humerus", ("bottom", 0.04), "-lat", "-lat"),  # medial epicondyle
    ("USP", "ulna", "ulna", ("bottom", 0.03), "-lat", "-lat"),   # ulnar styloid
    ("RSP", "radius", "radius", ("bottom", 0.03), "+lat", "+lat"),  # radial styloid
    ("HM2", "hand", "metacarpal2", ("bottom", 0.015), "-x", "-x"),  # 2nd metacarpal head, dorsal
    ("HEAD", "head", "hat_skull", ("band", 0.10, 0.04), "+lat", "+lat"),  # temple (front half of the skull)
]
# midline markers: (label, geom, region, pick, push)
MIDLINE = [
    ("SGL", "hat_skull", ("band", 0.12, 0.07), "+x", "+x"),     # glabella
    # Not in the AIM model, but the video shows markers down the front of the
    # vest, so the fit allows the sternum: jugular notch, body, xiphoid.
    ("SJN", "hat_ribs_scap", ("midband", 0.37, 0.44), "+x", "+x"),
    ("STN", "hat_ribs_scap", ("midband", 0.27, 0.36), "+x", "+x"),
    ("SXS", "hat_ribs_scap", ("midband", 0.18, 0.26), "+x", "+x"),
    ("CV7", "hat_spine", ("y", 0.450), "-x", "-x"),              # spinous processes, by height
    ("TV2", "hat_spine", ("y", 0.405), "-x", "-x"),              # above the lumbosacral joint
    ("MAI", "hat_spine", ("y", 0.290), "-x", "-x"),
    ("LV1", "hat_spine", ("y", 0.155), "-x", "-x"),
    ("LV3", "hat_spine", ("y", 0.085), "-x", "-x"),
    ("LV5", "hat_spine", ("y", 0.030), "-x", "-x"),
]
SIDE_GEOM = {("pelvis", "r", "hemi"): "pelvis", ("pelvis", "l", "hemi"): "l_pelvis",
             ("calcn", "r", "foot"): "r_foot", ("calcn", "l", "foot"): "l_foot"}


def marker_names():
    names = [f"{s}_{lab}" for s in ("L", "R") for lab, *_ in LANDMARKS]
    return names + [lab for lab, *_ in MIDLINE]


def _strip(root):
    """Source XML -> kinematic skeleton without sites, actuators or includes."""
    for tag in ("include", "actuator", "tendon", "contact", "equality", "sensor", "keyframe", "size"):
        for el in root.findall(tag):
            root.remove(el)
    comp = root.find("compiler")
    for el in list(comp):
        comp.remove(el)
    default = root.find("default")
    for el in default.findall("motor"):
        default.remove(el)
    for el in default.findall("joint"):
        el.attrib = {"limited": "true", "damping": "0", "armature": "0.01"}
    for body in root.iter("body"):
        for el in list(body):
            if el.tag in ("site", "light", "camera"):
                body.remove(el)
            elif el.tag == "geom":
                el.set("contype", "0"); el.set("conaffinity", "0"); el.set("rgba", BONE_RGBA)
            elif el.tag == "joint" and el.get("type") == "hinge":
                base = el.get("name").rsplit("_", 1)[0]
                if base in HINGE_RANGE_DEG:
                    lo, hi = np.radians(HINGE_RANGE_DEG[base])
                    el.set("range", f"{lo:.4f} {hi:.4f}")
                for k in ("stiffness", "damping", "armature"):
                    el.attrib.pop(k, None)
        if body.get("name") in BALL_JOINTS:
            new, old = BALL_JOINTS[body.get("name")]
            joints = [el for el in body if el.tag == "joint" and el.get("name") in old]
            at = list(body).index(joints[0])
            for el in joints:
                body.remove(el)
            ball = ET.Element("joint", name=new, type="ball", pos="0 0 0")
            if new in BALL_LIMIT_DEG:
                ball.set("range", f"0 {np.radians(BALL_LIMIT_DEG[new]):.4f}")
            else:
                ball.set("limited", "false")
            body.insert(at, ball)
            if new.startswith("shoulder"):
                # shoulder girdle: the source fuses scapula and clavicle into the
                # torso, but in a delivery they lift and push the shoulder joint
                # several cm; without this the arm cannot reach its markers
                side = new[-1]
                for i, axis in enumerate(("1 0 0", "0 1 0", "0 0 1")):
                    body.insert(at + i, ET.Element(
                        "joint", name=f"girdle_{side}_{'xyz'[i]}", type="slide", axis=axis,
                        range=f"{-GIRDLE_M} {GIRDLE_M}"))
    _add_neck(root)
    return root


def _add_neck(root):
    """Give the head its own body on a ball joint. The source welds skull and
    jaw to the torso, so head markers could not follow the head tilting
    against the trunk -- which it does a lot in a delivery."""
    torso = next(b for b in root.iter("body") if b.get("name") == "torso")
    head = ET.Element("body", name="head", pos=" ".join(f"{v:.6g}" for v in NECK_POS))
    ET.SubElement(head, "joint", name="neck", type="ball", pos="0 0 0",
                  range=f"0 {np.radians(BALL_LIMIT_DEG['neck']):.4f}")
    for g in [g for g in torso.findall("geom") if g.get("name") in ("hat_skull", "hat_jaw")]:
        torso.remove(g)
        g.set("pos", " ".join(f"{-v:.6g}" for v in NECK_POS))   # meshes stay where they were
        head.append(g)
    torso.append(head)


def _scale(root, s):
    if s == 1.0:
        return
    for body in root.iter("body"):
        if body.get("name") in ("ball",):
            continue
        p = np.array(body.get("pos", "0 0 0").split(), float) * s
        body.set("pos", " ".join(f"{v:.6g}" for v in p))
        for g in body.findall("geom"):
            if g.get("pos"):
                q = np.array(g.get("pos").split(), float) * s
                g.set("pos", " ".join(f"{v:.6g}" for v in q))
        for inert in body.findall("inertial"):
            q = np.array(inert.get("pos", "0 0 0").split(), float) * s
            inert.set("pos", " ".join(f"{v:.6g}" for v in q))
    for mesh in root.iter("mesh"):
        sc = np.array(mesh.get("scale", "1 1 1").split(), float) * s
        mesh.set("scale", " ".join(f"{v:.6g}" for v in sc))


ARM_BODIES = ("ulna", "radius", "hand")   # placed within the arm, so scaled with it


def _scale_arms(root, s):
    """Both arms' segment lengths (upper arm, forearm, hand) by s, on top of the
    uniform scale. The humerus stays where it is on the torso; everything below
    it moves in, and the arm bones' meshes shrink with it."""
    if s == 1.0:
        return
    arm_meshes = set()
    for body in root.iter("body"):
        name = body.get("name", "")
        if name.startswith("humerus") or name.split("_")[0] in ARM_BODIES:
            if name.split("_")[0] in ARM_BODIES:
                p = np.array(body.get("pos", "0 0 0").split(), float) * s
                body.set("pos", " ".join(f"{v:.6g}" for v in p))
            for g in body.findall("geom"):
                if g.get("mesh"):
                    arm_meshes.add(g.get("mesh"))
    for mesh in root.iter("mesh"):
        if mesh.get("name") in arm_meshes:
            sc = np.array(mesh.get("scale", "1 1 1").split(), float) * s
            mesh.set("scale", " ".join(f"{v:.6g}" for v in sc))


def _add_world(root):
    vis = ET.SubElement(root, "visual")
    ET.SubElement(vis, "headlight", diffuse="0.6 0.6 0.6", ambient="0.35 0.35 0.35", specular="0 0 0")
    ET.SubElement(vis, "global", offwidth="1280", offheight="960")
    asset = root.find("asset")
    ET.SubElement(asset, "texture", name="grid", type="2d", builtin="checker", rgb1="0.32 0.42 0.30",
                  rgb2="0.28 0.37 0.26", width="300", height="300")
    ET.SubElement(asset, "material", name="grid", texture="grid", texrepeat="1 1", texuniform="true")
    wb = root.find("worldbody")
    ET.SubElement(wb, "light", pos="0 0 6", dir="0 0 -1", diffuse="0.7 0.7 0.7", castshadow="true")
    ET.SubElement(wb, "light", pos="4 -4 4", dir="-1 1 -1", diffuse="0.3 0.3 0.3", castshadow="false")
    wb.insert(0, ET.Element("geom", name="floor", type="plane", size="20 6 0.1", material="grid",
                            contype="2", conaffinity="2"))
    ball = ET.SubElement(wb, "body", name="ball", pos="0 0 2")
    ET.SubElement(ball, "freejoint", name="ball_free")
    ET.SubElement(ball, "geom", name="ball_geom", type="sphere", size="0.036", mass="0.16",
                  rgba="0.75 0.1 0.1 1", contype="2", conaffinity="2")


def _body_vertices(m, body, geom):
    """Vertices of one mesh geom, in its body's frame."""
    g = m.geom(geom)
    assert m.body(g.bodyid[0]).name == body, (geom, body)
    mid = g.dataid[0]
    v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    R = np.empty(9)
    mujoco.mju_quat2Mat(R, g.quat)
    return v @ R.reshape(3, 3).T + g.pos


def _select(V, region, lat_sign):
    y = V[:, 1]
    lat = V[:, 2] * lat_sign
    kind = region[0]
    if kind == "all":
        return V
    if kind == "top":
        return V[y > y.max() - region[1]]
    if kind == "bottom":
        return V[y < y.min() + region[1]]
    if kind == "band":   # between d_far and d_near below the top
        return V[(y < y.max() - region[2]) & (y > y.max() - region[1])]
    if kind == "y":
        sel = V[(np.abs(y - region[1]) < 0.012) & (np.abs(V[:, 2]) < 0.03)]
        return sel if len(sel) else V[np.argsort(np.abs(y - region[1]))[:20]]
    if kind in ("upper", "lower"):
        return V[y > np.median(y)] if kind == "upper" else V[y < np.median(y)]
    if kind == "above":
        return V[y > region[1]]
    if kind == "yband":
        return V[(y > region[1]) & (y < region[2])]
    if kind == "midband":
        return V[(y > region[1]) & (y < region[2]) & (np.abs(V[:, 2]) < 0.03)]
    if kind == "medial":
        return V[lat < np.quantile(lat, 0.25)]
    if kind == "lateral":
        return V[lat > np.quantile(lat, 0.75)]
    raise ValueError(region)


def _direction(code, lat_sign):
    d = {"x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0]), "lat": np.array([0, 0, lat_sign])}
    return (-1 if code[0] == "-" else 1) * d[code[1:]]


def _landmark(V, region, pick, push, lat_sign, s):
    region = tuple(r * s if isinstance(r, float) else r for r in region)
    sel = _select(V, region, lat_sign)
    p = sel[np.argmax(sel @ _direction(pick, lat_sign))]
    return p + SKIN_M * _direction(push, lat_sign)


def build_xml(scale=1.0, site_offsets=None, src_dir=SRC_DIR, meshdir=None, arm_scale=1.0):
    """The skeleton as an XML string (meshes from `meshdir`, default the
    absolute path of OUT_DIR/meshes so the string compiles from anywhere),
    scaled uniformly by `scale` and its arms further by `arm_scale`, with one
    site per IOR marker. site_offsets: {marker: (3,) body-frame correction,
    metres} added to the mesh landmark."""
    src = ET.parse(src_dir / "skeleton_torque.xml").getroot()
    root = _strip(src)
    _scale(root, scale)
    _scale_arms(root, arm_scale)
    _add_world(root)
    root.find("compiler").attrib.update({"angle": "radian", "inertiafromgeom": "auto",
                                         "meshdir": str(OUT_DIR / "meshes")})
    for mesh in root.iter("mesh"):
        mesh.set("file", Path(mesh.get("file")).name)
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    bodies = {b.get("name"): b for b in root.iter("body")}
    site_offsets = site_offsets or {}

    def add_site(name, body, pos):
        pos = pos + np.asarray(site_offsets.get(name, np.zeros(3)))
        ET.SubElement(bodies[body], "site", name=f"m_{name}", pos=" ".join(f"{v:.5f}" for v in pos),
                      size="0.012", rgba="0.1 0.5 1 1", group="3")

    for side, sg in (("L", -1.0), ("R", 1.0)):
        sx = side.lower()
        for lab, body, geom, region, pick, push in LANDMARKS:
            b = body if body in ("torso", "pelvis", "head") else f"{body}_{sx}"
            g = SIDE_GEOM.get((body, sx, geom))
            if g is None:
                g = geom if geom.startswith("hat_") else (
                    f"{geom}_{sx}v" if body in ("humerus", "ulna", "radius") else
                    f"{geom}_{sx}vs" if body == "hand" else f"{geom}_{sx}")
            V = _body_vertices(m, b, g)
            if geom in ("hat_ribs_scap", "hat_skull"):   # bilateral meshes: this side only
                V = V[V[:, 2] * sg > 0]
            if lab == "HEAD":
                V = V[V[:, 0] > np.median(V[:, 0])]
            add_site(f"{side}_{lab}", b, _landmark(V, region, pick, push, sg, scale))
    for lab, geom, region, pick, push in MIDLINE:
        b = "head" if geom == "hat_skull" else "torso"
        V = _body_vertices(m, b, geom)
        add_site(lab, b, _landmark(V, region, pick, push, 1.0, scale))
    for s in ("l", "r"):   # sole contact points: heel and ball of the foot (lowest vertices)
        V = _body_vertices(m, f"calcn_{s}", f"{s}_foot")
        for name, part in (("heel", V[V[:, 0] < np.quantile(V[:, 0], 0.25)]),
                           ("ball", V[V[:, 0] > np.quantile(V[:, 0], 0.75)])):
            p = part[np.argmin(part[:, 1])]
            ET.SubElement(bodies[f"calcn_{s}"], "site", name=f"sole_{name}_{s}",
                          pos=" ".join(f"{v:.5f}" for v in p), size="0.01", group="4")
    if meshdir is not None:
        root.find("compiler").set("meshdir", meshdir)
    return ET.tostring(root, encoding="unicode")


def write_assets(scale=1.0, src_dir=LOCO_DIR, out_dir=OUT_DIR):
    (out_dir / "meshes").mkdir(parents=True, exist_ok=True)
    (out_dir / "source").mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_dir / "skeleton_torque.xml", out_dir / "source" / "skeleton_torque.xml")
    for f in (src_dir / "meshes").glob("*.stl"):
        dst = out_dir / "meshes" / f.name
        if not dst.exists():
            shutil.copy2(f, dst)
    xml = build_xml(scale, src_dir=src_dir, meshdir="meshes")
    (out_dir / "skeleton3d.xml").write_text(xml, encoding="utf-8")
    return out_dir / "skeleton3d.xml"


if __name__ == "__main__":
    print("wrote", write_assets())
