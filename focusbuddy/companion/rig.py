"""Lightweight, data-driven 2D skeletal rig: bones, forward kinematics and two-bone IK.

Every character is described by a ``SkeletonDef`` - a list of bones (name,
parent, length, attach offset, rest angle, joint limits) plus which bone
chains are legs / arms / tail.  Proportions differ per character (a fox has
longer legs than a rabbit's front legs; the robot has short stiff shins) and
can be overridden without touching code:

    assets/characters/<character id>/skeleton.json
    {"legs": {"FL": {"upper": 12.5, "lower": 12}}, "tail": {"lengths": [9, 9, 9]}}

Coordinates are renderer units (140 x 140 canvas, feet at (0, 0), +x is the
direction the character faces, y points down).  Angles are degrees clockwise
in screen space: 0 = forward (+x), 90 = straight down.

Pipeline position::

    ... -> smoothed Pose -> layout (torso, head, hips) -> foot targets (gait.py)
        -> pose overrides (sit, lie, paw, press, groom...) -> IK -> RigPose -> renderer

The rig is pure Python (no Qt) so it is unit-tested directly.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

# ------------------------------------------------------------------- math
def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def smoothstep(x: float) -> float:
    x = _clamp(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def rotate(px: float, py: float, deg: float, pivot: tuple[float, float] = (0.0, 0.0)) -> tuple[float, float]:
    """Rotate a point clockwise (screen space) around a pivot."""
    if not deg:
        return px, py
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    x, y = px - pivot[0], py - pivot[1]
    return pivot[0] + x * c - y * s, pivot[1] + x * s + y * c


@dataclass
class IKResult:
    root: tuple[float, float]
    joint: tuple[float, float]
    end: tuple[float, float]
    target: tuple[float, float]
    upper_angle: float      # world (deg)
    lower_angle: float      # world (deg)
    rel: float              # lower relative to upper (deg) - the knee / elbow bend
    reached: bool


def two_bone_ik(root: tuple[float, float], target: tuple[float, float], l1: float, l2: float,
                bend: int = 1, max_bend: float = 165.0) -> IKResult:
    """Analytic two-bone IK (law of cosines).

    ``bend`` +1 puts the middle joint on the forward side (human knee), -1 on the
    backward side (elbow, animal hock).  Unreachable targets give a straight limb
    pointing at the target; the bend never reverses and never exceeds ``max_bend``.
    """
    ox, oy = root
    tx, ty = target
    dx, dy = tx - ox, ty - oy
    d = math.hypot(dx, dy)
    reach = l1 + l2
    if not (d == d) or d < 1e-6:        # NaN guard / degenerate
        dx, dy, d = 0.0, 1.0, 1e-6
    base = math.atan2(dy, dx)
    lo = abs(l1 - l2) + 1e-4
    dc = _clamp(d, lo, reach * 0.99995)
    cos_a = _clamp((l1 * l1 + dc * dc - l2 * l2) / (2 * l1 * dc), -1.0, 1.0)
    a = math.acos(cos_a)
    a1 = base - bend * a
    jx, jy = ox + l1 * math.cos(a1), oy + l1 * math.sin(a1)
    ex, ey = ox + dc * math.cos(base), oy + dc * math.sin(base)
    a2 = math.atan2(ey - jy, ex - jx)
    rel = math.degrees(a2 - a1)
    rel = (rel + 180.0) % 360.0 - 180.0
    # joint limit: never bend backwards, never fold flatter than max_bend.
    # bend=+1 (forward knee): the lower bone turns clockwise relative to the upper -> rel >= 0
    lim_lo, lim_hi = (0.0, max_bend) if bend > 0 else (-max_bend, 0.0)
    rel_c = _clamp(rel, lim_lo, lim_hi)
    if abs(rel_c - rel) > 1e-6:
        a2 = a1 + math.radians(rel_c)
        ex, ey = jx + l2 * math.cos(a2), jy + l2 * math.sin(a2)
        rel = rel_c
    return IKResult((ox, oy), (jx, jy), (ex, ey), (tx, ty), math.degrees(a1), math.degrees(a2), rel,
                    d <= reach + 1e-6)


def two_bone_ik_3d(root, target, l1: float, l2: float, pole=(0.0, 0.0, 1.0), max_bend: float = 165.0):
    """Two-bone IK in 3D: the middle joint bends toward ``pole`` (a direction).

    Returns (joint, end, reached).  Used for limbs that leave the side plane - hands meeting in a
    clap / high five, paws reaching across the body.  Same guarantees as the 2D solver: the
    bend never exceeds ``max_bend`` and unreachable targets give a straight limb aimed at them.
    """
    ox, oy, oz = root
    dx, dy, dz = target[0] - ox, target[1] - oy, target[2] - oz
    d = math.sqrt(dx * dx + dy * dy + dz * dz)
    if not (d == d) or d < 1e-6:
        dx, dy, dz, d = 0.0, 1.0, 0.0, 1e-6
    ux, uy, uz = dx / d, dy / d, dz / d
    reach = l1 + l2
    lo = max(abs(l1 - l2) + 1e-4, math.sqrt(max(0.0, l1 * l1 + l2 * l2 - 2 * l1 * l2 *
                                                  math.cos(math.radians(180.0 - max_bend)))))
    dc = _clamp(d, lo, reach * 0.99995)
    cos_a = _clamp((l1 * l1 + dc * dc - l2 * l2) / (2 * l1 * dc), -1.0, 1.0)
    a = math.acos(cos_a)
    # pole direction made perpendicular to the root->target axis
    px, py, pz = pole
    dot = px * ux + py * uy + pz * uz
    px, py, pz = px - dot * ux, py - dot * uy, pz - dot * uz
    pn = math.sqrt(px * px + py * py + pz * pz)
    if pn < 1e-6:                     # pole parallel to the limb: pick any perpendicular
        px, py, pz = (-uy, ux, 0.0) if abs(uz) < 0.9 else (0.0, -uz, uy)
        pn = math.sqrt(px * px + py * py + pz * pz)
    px, py, pz = px / pn, py / pn, pz / pn
    c, s_ = math.cos(a), math.sin(a)
    joint = (ox + l1 * (ux * c + px * s_), oy + l1 * (uy * c + py * s_), oz + l1 * (uz * c + pz * s_))
    end = (ox + ux * dc, oy + uy * dc, oz + uz * dc)
    return joint, end, d <= reach + 1e-6


# ------------------------------------------------------------- definitions
@dataclass(frozen=True)
class BoneDef:
    name: str
    parent: Optional[str]
    length: float
    rest: float = 0.0                  # rest angle relative to the parent (deg)
    limits: Optional[tuple[float, float]] = None
    width: float = 0.0


@dataclass(frozen=True)
class LimbDef:
    name: str                # FL FR BL BR | L R | arm_L arm_R
    upper: float
    lower: float
    bend: int                # +1 forward joint, -1 backward joint
    width: float = 8.6
    side: str = "near"       # near (drawn in front of the body) | far
    max_bend: float = 160.0


@dataclass(frozen=True)
class TailDef:
    rest: tuple[float, ...] = ()       # first = base angle relative to torso, rest = relative per segment
    lengths: tuple[float, ...] = ()
    widths: tuple[float, ...] = ()
    stiffness: tuple[float, ...] = ()  # spring frequency per segment (Hz)


@dataclass(frozen=True)
class SkeletonDef:
    id: str
    kind: str                          # quadruped | biped
    legs: tuple[LimbDef, ...]
    arms: tuple[LimbDef, ...] = ()
    tail: TailDef = TailDef()
    ankle: float = 4.0                 # foot joint height above the ground
    pivot: tuple[float, float] = (0.0, -18.0)
    head_r: float = 21.0
    bones: tuple[BoneDef, ...] = ()    # full hierarchy (built from the above, used for FK + debug)

    def limb(self, name: str) -> LimbDef:
        for l in self.legs + self.arms:
            if l.name == name:
                return l
        raise KeyError(name)


def _quad(cid: str, upper: float, lower: float, width: float, tail: TailDef,
          front: tuple[float, float] | None = None, head_r: float = 21.0) -> SkeletonDef:
    fu, fl = front or (upper, lower)
    legs = (LimbDef("BL", upper, lower, -1, width, "far"), LimbDef("FL", fu, fl, -1, width, "far"),
            LimbDef("BR", upper, lower, -1, width, "near"), LimbDef("FR", fu, fl, -1, width, "near"))
    return _with_bones(SkeletonDef(cid, "quadruped", legs, (), tail, ankle=width * 0.45, pivot=(0.0, -18.0),
                                   head_r=head_r))


def _biped(cid: str, upper: float, lower: float, arm: tuple[float, float], width: float = 8.5) -> SkeletonDef:
    legs = (LimbDef("L", upper, lower, 1, width, "far"), LimbDef("R", upper, lower, 1, width, "near"))
    arms = (LimbDef("arm_L", arm[0], arm[1], -1, 7.6, "far", 150.0),
            LimbDef("arm_R", arm[0], arm[1], -1, 7.6, "near", 150.0))
    return _with_bones(SkeletonDef(cid, "biped", legs, arms, TailDef(), ankle=width * 0.45, pivot=(0.0, -20.0),
                                   head_r=23.0))


def _with_bones(sk: SkeletonDef) -> SkeletonDef:
    """Expand the compact description into an explicit, species-specific bone hierarchy.

    quadruped:  root > pelvis > spine > chest > neck > head > jaw / eye_L / eye_R / ear_L / ear_R
                back legs hang from the pelvis, front legs from the chest:
                hip_X > upper_X > lower_X > foot_X ;  tail_0.. from the pelvis
    biped:      root > pelvis > spine > chest > neck > head > jaw / eyes (/ ears or antenna)
                legs from the pelvis (hip > upper > lower > foot), arms from the chest
                (shoulder > upper (arm) > lower (forearm) > hand)
    """
    quad = sk.kind == "quadruped"
    C = CONSTRAINTS.get(sk.id) or CONSTRAINTS["quad" if quad else "biped"]
    b: list[BoneDef] = [BoneDef("root", None, 0.0), BoneDef("pelvis", "root", 0.0, 0.0, C["pelvis_roll"])]
    if quad:
        b += [BoneDef("spine", "pelvis", 14.0, 0.0, C["spine"]), BoneDef("chest", "spine", 13.0, 0.0, C["spine"]),
              BoneDef("neck", "chest", 22.0, -55.0, (-95.0, -10.0)), BoneDef("head", "neck", 0.0, 0.0, C["head_pitch"])]
    else:
        b += [BoneDef("spine", "pelvis", 14.0, -90.0, (-110.0, -70.0)), BoneDef("chest", "spine", 14.0, 0.0, C["spine"]),
              BoneDef("neck", "chest", 6.0, 0.0, (-30.0, 30.0)), BoneDef("head", "neck", 0.0, 0.0, C["head_pitch"])]
    b += [BoneDef("jaw", "head", 7.0, 20.0, C["jaw"]), BoneDef("eye_L", "head", 0.0, 0.0, C["eye_yaw"]),
          BoneDef("eye_R", "head", 0.0, 0.0, C["eye_yaw"])]
    if sk.id != "buddy":
        b += [BoneDef("ear_L", "head", 8.0, -100.0, C["ear"]), BoneDef("ear_R", "head", 8.0, -100.0, C["ear"])]
    for l in sk.legs:
        parent = ("chest" if l.name[0] == "F" else "pelvis") if quad else "pelvis"
        b += [BoneDef(f"hip_{l.name}", parent, 0.0),
              BoneDef(f"upper_{l.name}", f"hip_{l.name}", l.upper, 90.0, None, l.width),
              BoneDef(f"lower_{l.name}", f"upper_{l.name}", l.lower, 0.0,
                      (0.0, l.max_bend) if l.bend > 0 else (-l.max_bend, 0.0), l.width * 0.92),
              BoneDef(f"foot_{l.name}", f"lower_{l.name}", l.width * 0.9, -90.0, C["foot"], l.width * 0.6)]
    for a in sk.arms:
        b += [BoneDef(f"shoulder_{a.name}", "chest", 0.0),
              BoneDef(f"upper_{a.name}", f"shoulder_{a.name}", a.upper, 90.0, None, a.width),
              BoneDef(f"lower_{a.name}", f"upper_{a.name}", a.lower, 0.0, (-a.max_bend, 0.0), a.width),
              BoneDef(f"hand_{a.name}", f"lower_{a.name}", 4.5, 0.0, (-60.0, 60.0), a.width)]
    prev = "pelvis"
    for i, ln in enumerate(sk.tail.lengths):
        b.append(BoneDef(f"tail_{i}", prev, ln, sk.tail.rest[i] if i < len(sk.tail.rest) else 0.0,
                         None if i == 0 else C["tail"], sk.tail.widths[i] if i < len(sk.tail.widths) else 6))
        prev = f"tail_{i}"
    return replace(sk, bones=tuple(b))


# ------------------------------------------------------------- constraints
# Joint limits per species (degrees unless noted).  Knee / elbow limits live on the LimbDefs
# (two_bone_ik never bends a joint backwards); everything else is clamped by ``apply_constraints``.
CONSTRAINTS: dict[str, dict] = {
    "quad": {"head_pitch": (-30.0, 30.0), "head_yaw": (-1.0, 1.0), "neck_yaw": (-1.0, 1.0),
             "eye_yaw": (-35.0, 35.0), "eye_pitch": (-25.0, 25.0), "jaw": (0.0, 28.0), "ear": (-60.0, 60.0),
             "tail": (-75.0, 95.0), "foot": (-35.0, 40.0), "spine": (-12.0, 12.0), "pelvis_roll": (-8.0, 8.0),
             "torso_twist": (-15.0, 15.0)},
    "biped": {"head_pitch": (-25.0, 25.0), "head_yaw": (-1.0, 1.0), "neck_yaw": (-1.0, 1.0),
              "eye_yaw": (-35.0, 35.0), "eye_pitch": (-25.0, 25.0), "jaw": (0.0, 24.0), "ear": (-20.0, 20.0),
              "tail": (-75.0, 95.0), "foot": (-30.0, 35.0), "spine": (-15.0, 15.0), "pelvis_roll": (-8.0, 8.0),
              "torso_twist": (-15.0, 15.0)},
}
CONSTRAINTS["rabbit"] = {**CONSTRAINTS["quad"], "ear": (-90.0, 40.0)}      # long ears droop a long way
CONSTRAINTS["dog"] = {**CONSTRAINTS["quad"], "ear": (-50.0, 70.0)}
CONSTRAINTS["robot"] = {**CONSTRAINTS["biped"], "head_pitch": (-20.0, 20.0), "jaw": (0.0, 0.0),
                        "foot": (-10.0, 12.0), "pelvis_roll": (-3.0, 3.0), "torso_twist": (-6.0, 6.0)}


def constraints_for(species: str, kind: str = "quadruped") -> dict:
    return CONSTRAINTS.get(species) or CONSTRAINTS["biped" if kind == "biped" else "quad"]


def apply_constraints(p, species: str, kind: str = "quadruped") -> None:
    """Clamp the pose channels that map to constrained joints (head, neck, eyes, jaw, ears, spine)."""
    C = constraints_for(species, kind)

    def cl(name, lo, hi):
        v = getattr(p, name, None)
        if v is None:
            return
        if not (v == v):              # NaN guard
            v = 0.0
        setattr(p, name, lo if v < lo else hi if v > hi else v)
    cl("head_tilt", *C["head_pitch"])
    cl("head_yaw", *C["head_yaw"])
    cl("neck_yaw", *C["neck_yaw"])
    cl("pelvis_roll", *C["pelvis_roll"])
    cl("torso_twist", *C["torso_twist"])
    cl("jaw", 0.0, 1.0 if C["jaw"][1] > 0 else 0.0)
    cl("look_x", -1.0, 1.0)
    cl("look_y", -1.0, 1.0)
    cl("eye_conv", 0.0, 0.8)
    lo, hi = C["ear"]
    cl("ear_flop", lo, hi)
    cl("ear_l", lo, hi)
    cl("ear_r", lo, hi)


SKELETONS: dict[str, SkeletonDef] = {
    "cat": _quad("cat", 8.5, 12.5, 8.6, TailDef((196, 39, 35, 30), (10, 10, 10, 10), (6.5, 6.5, 6.2, 6.0),
                                                   (3.6, 3.0, 2.5, 2.1))),
    "dog": _quad("dog", 8.8, 12.8, 8.6, TailDef((205, 40, 35), (8, 8, 8), (6.5, 6.5, 6.0), (3.4, 2.8, 2.3))),
    "fox": _quad("fox", 9.0, 13.0, 8.6, TailDef((192, 28, 32, 34), (10, 10, 10, 9), (7, 11, 12, 8),
                                                   (2.8, 2.3, 1.9, 1.6))),
    "rabbit": _quad("rabbit", 9.2, 12.6, 9.5, TailDef((200,), (3,), (7,), (4.0,)), front=(8.2, 12.2),
                    head_r=20.0),
    "robot": _biped("robot", 10.3, 10.1, (9.5, 9.5)),
    "buddy": _biped("buddy", 10.3, 10.1, (9.5, 9.5)),
}


def _merge(sk: SkeletonDef, data: dict) -> SkeletonDef:
    legs = []
    for l in sk.legs:
        o = (data.get("legs") or {}).get(l.name) or {}
        legs.append(replace(l, upper=float(o.get("upper", l.upper)), lower=float(o.get("lower", l.lower)),
                            width=float(o.get("width", l.width))))
    arms = []
    for a in sk.arms:
        o = (data.get("arms") or {}).get(a.name) or {}
        arms.append(replace(a, upper=float(o.get("upper", a.upper)), lower=float(o.get("lower", a.lower))))
    tail = sk.tail
    t = data.get("tail") or {}
    if t:
        tail = TailDef(tuple(t.get("rest", tail.rest)), tuple(t.get("lengths", tail.lengths)),
                       tuple(t.get("widths", tail.widths)), tuple(t.get("stiffness", tail.stiffness)))
    return _with_bones(replace(sk, legs=tuple(legs), arms=tuple(arms), tail=tail))


def load_skeleton(char_id: str, species: str = "", kind: str = "quadruped", assets_dir: Path | None = None,
                  extra: dict | None = None) -> SkeletonDef:
    """Skeleton for a character: built-in species data + optional JSON / spec overrides."""
    base = SKELETONS.get(char_id) or SKELETONS.get(species)
    if base is None:
        base = SKELETONS["buddy" if kind == "biped" else "cat"]
    base = replace(base, id=char_id)
    data: dict = {}
    if assets_dir is None:
        try:
            from ..config import ASSETS_DIR
            assets_dir = ASSETS_DIR
        except Exception:  # noqa: BLE001
            assets_dir = None
    if assets_dir is not None:
        f = Path(assets_dir) / "characters" / char_id / "skeleton.json"
        if f.is_file():
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                data = {}
    if extra:
        data = {**data, **extra}
    return _merge(base, data) if data else base


def skeleton_for_profile(profile_id: str, gait: str) -> SkeletonDef:
    if profile_id in SKELETONS:
        return SKELETONS[profile_id]
    return SKELETONS["robot" if gait == "mech" else "buddy" if gait == "biped" else "cat"]


TARGET_NAMES = {"FL": "front_left_foot_target", "FR": "front_right_foot_target",
                "BL": "back_left_foot_target", "BR": "back_right_foot_target",
                "L": "left_foot_target", "R": "right_foot_target",
                "arm_L": "left_hand_target", "arm_R": "right_hand_target"}
TARGET_LIMB = {v: k for k, v in TARGET_NAMES.items()}


# ------------------------------------------------------------------ output
@dataclass
class LimbSolve:
    name: str
    root: tuple[float, float]
    joint: tuple[float, float]
    end: tuple[float, float]
    target: tuple[float, float]
    reached: bool
    rel: float
    width: float
    side: str
    contact: bool = False
    phase: str = ""
    foot_angle: float = 0.0           # foot / paw pitch (deg, + = toe down)
    foot: tuple = (0.0, 0.0)          # toe point (world units)
    splay: float = 0.0                # 0..1 elbow / knee pushed out sideways (3D only: hands on hips, arms open)
    hand_z: float = 0.0               # lateral hand offset (3D only): - toward the body centre, + outward


@dataclass
class RigPose:
    kind: str
    skeleton: str
    layout: dict
    tilt: float
    pivot: tuple[float, float]
    joints: dict = field(default_factory=dict)        # name -> (x, y) world units
    legs: dict = field(default_factory=dict)          # name -> LimbSolve
    arms: dict = field(default_factory=dict)          # name -> LimbSolve
    tail: list = field(default_factory=list)          # [(x, y)...] base -> tip
    tail_widths: tuple = ()
    ground: float = 0.0                               # ground height (units below the feet origin, >= 0)
    targets: dict = field(default_factory=dict)       # named IK targets: left_foot_target ... -> (x, y)

    def bones(self) -> list[tuple[str, tuple[float, float], tuple[float, float]]]:
        """Flat list of drawn bones (for the debug skeleton)."""
        out = []
        j = self.joints
        if "pelvis" in j and "chest" in j:
            out.append(("spine", j["pelvis"], j["chest"]))
            if "neck_base" in j:
                out.append(("chest", j["chest"], j["neck_base"]))
        elif "torso" in j and "neck_base" in j:
            out.append(("spine", j["tail_root"] if "tail_root" in j else j["torso"], j["neck_base"]))
        if "neck_base" in j and "head" in j:
            out.append(("neck", j["neck_base"], j["head"]))
        if "head" in j and "jaw" in j:
            out.append(("jaw", j["head"], j["jaw"]))
        for d in (self.legs, self.arms):
            for n, s in d.items():
                out.append((f"upper_{n}", s.root, s.joint))
                out.append((f"lower_{n}", s.joint, s.end))
                if s.foot != (0.0, 0.0):
                    out.append((("foot_" if d is self.legs else "hand_") + n, s.end, s.foot))
        for i, (a, b) in enumerate(zip(self.tail, self.tail[1:])):
            out.append((f"tail_{i}", a, b))
        return out


# ------------------------------------------------------------------- the rig
class Rig:
    """Turns a Pose (+ optional planted-foot targets) into a solved skeleton."""

    def __init__(self, skeleton: SkeletonDef):
        self.sk = skeleton

    @property
    def kind(self) -> str:
        return self.sk.kind

    # ---------------------------------------------------------- layout
    def layout(self, p) -> dict:
        if self.sk.kind == "biped":
            return self._layout_biped(p)
        return self._layout_quad(p)

    def _layout_quad(self, p) -> dict:
        s = _clamp(p.sit, 0.0, 1.0)
        l = _clamp(p.lie, 0.0, 1.0) * (1 - s * 0.5)
        stand = max(0.0, 1 - s - l)
        st = _clamp(p.stretch, 0.0, 1.0)
        fl = p.flex
        shift = getattr(p, "shift", 0.0)
        breath = getattr(p, "breath", 0.0)

        def mix(a, b, c):
            return a * stand + b * s + c * l

        y0 = -p.lift + p.bob
        bx = mix(-2, -7, -1) + shift * 1.6
        by = mix(-27, -27, -14) + y0 - getattr(p, "crouch_body", 0.0) * 5.0
        brx = mix(27, 20, 30) + fl * 3.0
        bry = (mix(17, 21, 13.5) - fl * 1.0) * (1 + 0.025 * breath + 0.06 * getattr(p, "puff", 0.0))
        hx = mix(21, 9, 21) + st * 6 + p.head_dx + fl * 1.5 + shift * 1.2
        hy = mix(-53, -61, -26) + y0 + st * 16 + p.head_dy + p.groom * 4 + p.yawn * -2 \
            - getattr(p, "crouch_body", 0.0) * 4.0
        leg_top = by + bry * 0.35
        front_top = leg_top * (1 - s) + (by + bry * 0.12) * s       # continuous: no pop when sitting down
        back_x = bx - fl * 2.5
        front_x = bx + fl * 2.5
        br_y = leg_top * (1 - s) + (-12 + y0 * 0.3) * s
        r = math.sin(math.radians(getattr(p, "pelvis_roll", 0.0))) * 4.5 * stand
        hips = {"BL": (back_x - 16 - 2 * st, leg_top - r), "BR": (back_x - 10 - 2 * st, br_y + r),
                "FL": (front_x + 12 - 4 * s, front_top + r * 0.6), "FR": (front_x + 18 - 4 * s, front_top - r * 0.6)}
        return {"s": s, "l": l, "stand": stand, "st": st, "y0": y0, "bx": bx, "by": by, "brx": brx, "bry": bry,
                "hx": hx, "hy": hy, "hr": self.sk.head_r, "leg_top": leg_top, "front_top": front_top,
                "hips_body": hips, "tilt": p.tilt + st * 10, "tail_base": (bx - brx * 0.88, by - bry * 0.15 - st * 8),
                "neck_base": (bx + brx * 0.55, by - bry * 0.45)}

    def _layout_biped(self, p) -> dict:
        s = _clamp(max(p.sit, p.lie), 0.0, 1.0)
        sleeping = p.lie > 0.5
        y0 = -p.lift + p.bob
        shift = getattr(p, "shift", 0.0)
        hip_y = -22 * (1 - s) - 9 * s + y0 - getattr(p, "crouch_body", 0.0) * 4.0
        body_top = hip_y - 28
        hx, hy = 1.0 + p.head_dx + shift * 0.8, body_top - 20 + p.head_dy
        if sleeping:
            hx += 3
            hy += 3
        r = math.sin(math.radians(getattr(p, "pelvis_roll", 0.0))) * 5.0 * (1 - s)
        hips = {"L": (-5 + shift, hip_y - r), "R": (5 + shift, hip_y + r)}
        sr = r * 0.5                                   # shoulders counter-roll a little
        shy = -3.5 * _clamp(getattr(p, "shoulders", 0.0), -1.0, 1.0)      # raised (tense) / drooped (sad)
        shoulders = {"arm_L": (-11 + shift, body_top + 5 + sr + shy), "arm_R": (11 + shift, body_top + 5 - sr + shy)}
        return {"s": s, "l": 0.0, "stand": 1 - s, "st": 0.0, "y0": y0, "hip_y": hip_y, "body_top": body_top,
                "hx": hx, "hy": hy, "hr": self.sk.head_r, "hips_body": hips, "shoulders_body": shoulders,
                "tilt": p.tilt, "sleeping": sleeping, "shift": shift}

    def to_world(self, lay: dict, pt: tuple[float, float]) -> tuple[float, float]:
        return rotate(pt[0], pt[1], lay["tilt"], self.sk.pivot)

    def hips(self, p, lay: dict | None = None) -> dict:
        lay = lay or self.layout(p)
        return {k: self.to_world(lay, v) for k, v in lay["hips_body"].items()}

    def leg_reach(self, name: str) -> float:
        l = self.sk.limb(name)
        return l.upper + l.lower

    def neutral_feet(self, p, lay: dict | None = None) -> dict:
        """Stance position of every foot (directly below its hip, on the ground)."""
        hips = self.hips(p, lay)
        st = _clamp(getattr(p, "stance", 0.0), -1.0, 1.0)
        out = {}
        for k, v in hips.items():
            if self.sk.kind == "biped":
                dx = (3.0 if k == "R" else -3.0) * st            # feet apart (front-on stance)
            else:
                dx = (4.5 if k[0] == "F" else -4.5) * st        # front paws forward, hind paws back: planted
            out[k] = (v[0] + dx, -self.sk.ankle)
        return out

    def stance_reach(self, p, lay: dict | None = None) -> dict:
        """How far (units) each planted foot may drift fore/aft of its hip before the leg is straight."""
        out = {}
        for k, (hx, hy) in self.hips(p, lay).items():
            h = max(0.0, -self.sk.ankle - hy)
            r = self.leg_reach(k) * 0.97
            out[k] = math.sqrt(max(0.0, r * r - h * h))
        return out

    # --------------------------------------------------------- solving
    def solve(self, p, feet: dict | None = None, contacts: dict | None = None,
              phases: dict | None = None, foot_pitch: dict | None = None,
              ik_targets: dict | None = None) -> RigPose:
        """Solve the skeleton.

        ``feet``        planted / swinging foot positions from the foot-phase system
        ``foot_pitch``  per-foot pitch (deg, + = toe down) - clamped by the foot constraint
        ``ik_targets``  named IK targets that override the pose, e.g.
                        {"right_hand_target": (x, y, weight)} or {"left_foot_target": (x, y)}
        """
        lay = self.layout(p)
        self._contacts = contacts or {}
        self._ik_over = {}
        for name, v in (ik_targets or {}).items():
            limb = TARGET_LIMB.get(name, name)
            w = float(v[2]) if len(v) > 2 else 1.0
            self._ik_over[limb] = (float(v[0]), float(v[1]), _clamp(w, 0.0, 1.0))
        if self.sk.kind == "biped":
            rig = self._solve_biped(p, lay, feet or {})
        else:
            rig = self._solve_quad(p, lay, feet or {})
        C = constraints_for(self.sk.id, self.sk.kind)
        flo, fhi = C["foot"]
        mech = self.sk.id == "robot"
        for k, s in rig.legs.items():
            s.contact = bool((contacts or {}).get(k, False))
            s.phase = (phases or {}).get(k, "")
            ang = _clamp(float((foot_pitch or {}).get(k, 0.0)), flo, fhi)
            if s.contact and s.phase not in ("LIFT",):
                ang = 0.0                      # planted feet stay flat on the ground
            elif mech:
                ang *= 0.5                     # robot feet stay stiff
            s.foot_angle = ang
            fl = self.sk.limb(k).width * (0.95 if self.sk.kind != "biped" else 0.85)
            a = math.radians(ang)
            s.foot = (s.end[0] + fl * math.cos(a), s.end[1] + fl * math.sin(a))
        for k, s in rig.arms.items():
            d = (s.end[0] - s.joint[0], s.end[1] - s.joint[1])
            n = math.hypot(*d) or 1.0
            s.foot = (s.end[0] + d[0] / n * 4.0, s.end[1] + d[1] / n * 4.0)      # hand tip
        rig.targets = {TARGET_NAMES[k]: s.target for k, s in list(rig.legs.items()) + list(rig.arms.items())
                       if k in TARGET_NAMES}
        self._add_joints(p, lay, rig)
        rig.ground = max(0.0, getattr(p, "ground", 0.0))
        return rig

    def _override(self, k: str, tx: float, ty: float) -> tuple[float, float]:
        o = getattr(self, "_ik_over", None)
        if not o or k not in o:
            return tx, ty
        x, y, w = o[k]
        return tx + (x - tx) * w, ty + (y - ty) * w

    def _add_joints(self, p, lay, rig: RigPose) -> None:
        """Extra named joints (pelvis, chest, jaw, eyes) for the debug view and the renderers."""
        j = rig.joints
        head = j.get("head")
        if head is None:
            return
        hr = lay["hr"]
        tilt = lay["tilt"] + getattr(p, "head_tilt", 0.0)
        jaw = _clamp(getattr(p, "jaw", 0.0), 0.0, 1.0) * constraints_for(self.sk.id, self.sk.kind)["jaw"][1]
        j["jaw"] = rotate(head[0] + hr * 0.62, head[1] + hr * 0.62, tilt + jaw * 0.5, head)
        for i, side in enumerate((-1, 1)):
            j["eye_L" if i == 0 else "eye_R"] = rotate(head[0] + hr * 0.6, head[1] - hr * 0.05 + side * 0.01,
                                                       tilt, head)
        if "pelvis" not in j:
            if self.sk.kind == "biped":
                j["pelvis"] = j.get("torso", (0.0, -22.0))
            else:
                j["pelvis"] = self.to_world(lay, (lay["bx"] - lay["brx"] * 0.55, lay["by"]))
        if "chest" not in j:
            if self.sk.kind == "biped":
                j["chest"] = self.to_world(lay, (lay["shift"], lay["body_top"] + 10.0))
            else:
                j["chest"] = self.to_world(lay, (lay["bx"] + lay["brx"] * 0.45, lay["by"]))

    def _paw_target(self, root, angle_deg: float, length: float) -> tuple[float, float]:
        r = math.radians(angle_deg)
        return root[0] - length * math.sin(r), root[1] + length * math.cos(r)

    def _blend(self, a, b, w):
        return a[0] + (b[0] - a[0]) * w, a[1] + (b[1] - a[1]) * w

    def _solve_quad(self, p, lay, feet) -> RigPose:
        sk = self.sk
        hips = {k: self.to_world(lay, v) for k, v in lay["hips_body"].items()}
        s, l, stand, st = lay["s"], lay["l"], lay["stand"], lay["st"]
        ank = sk.ankle
        # legs tuck while airborne (lift beyond the gait's own bounce); continuous -> no foot pops
        tuck = max(_clamp(p.hop, 0.0, 1.0), smoothstep((p.lift - 4.0) / 8.0)) * (1 - s) * (1 - l)
        legs = {}
        press_pt = self.press_point()
        for limb in sk.legs:
            k = limb.name
            hx, hy = hips[k]
            front = k[0] == "F"
            gx, gy = feet.get(k, (hx, -ank))
            sit_t = (hx + (1.5 if front else 7.0), -ank)
            lie_t = (hx + (13.0 if front else 10.0), -ank)
            tx = gx * stand + sit_t[0] * s + lie_t[0] * l
            ty = gy * stand + sit_t[1] * s + lie_t[1] * l
            if front and st > 0:
                tx += st * 8.0
            if tuck > 0:
                tt = (hx + (-4.0 if front else 5.0), hy + 12.5)
                tx, ty = self._blend((tx, ty), tt, tuck)
            reach = limb.upper + limb.lower
            # ------------- action overrides on individual limbs
            if k == "FR" and (p.paw > 0 or p.press != 0 or p.ripple > 0):
                a = -115 * p.paw + p.paw_wave
                if p.press != 0 or p.ripple > 0:
                    aim = -88 - 30 * max(0.0, -p.press)
                    a = a * (1 - _clamp(p.paw, 0, 1)) + (aim + p.paw_wave * 0.3) * _clamp(p.paw, 0, 1)
                pt = self._paw_target((hx, hy), a, reach * 0.93)
                if p.press > 0:
                    pt = self._blend(pt, press_pt, _clamp(p.press, 0.0, 1.0))
                tx, ty = self._blend((tx, ty), pt, _clamp(max(p.paw, abs(p.press)), 0.0, 1.0))
            if k == "FL" and getattr(p, "paw2", 0.0) > 0:
                pt = self._paw_target((hx, hy), -115 * p.paw2 - p.paw_wave * 0.6, reach * 0.93)
                tx, ty = self._blend((tx, ty), pt, _clamp(p.paw2, 0.0, 1.0))
            if k == "FR" and p.groom > 0:
                hwx, hwy = self.to_world(lay, (lay["hx"] + 3, lay["hy"] + 13))
                pt = (hwx + p.paw_wave * 0.15, hwy)
                tx, ty = self._blend((tx, ty), pt, _clamp(p.groom, 0.0, 1.0))
            cover = getattr(p, "cover", 0.0)
            if front and cover > 0:            # paws over the face
                fx_, fy_ = self.to_world(lay, (lay["hx"] + (6.0 if k == "FR" else 3.0), lay["hy"] + 3.0))
                tx, ty = self._blend((tx, ty), (fx_, fy_), _clamp(cover, 0.0, 1.0))
            stomp = getattr(p, "stomp", 0.0)
            if k == "FR" and stomp > 0:        # angry stomp: lift the paw, then slam it down
                tx, ty = self._blend((tx, ty), (hx + 3.0, -sk.ankle - 9.0), _clamp(stomp, 0.0, 1.0))
            if k == "BR" and p.scratch > 0:
                a = -100 + 14 * math.sin(p.mark_phase * 32)
                pt = self._paw_target((hx, hy), a, 17.0)
                tx, ty = self._blend((tx, ty), pt, _clamp(p.scratch, 0.0, 1.0))
            tx, ty = self._override(k, tx, ty)
            ik = two_bone_ik((hx, hy), (tx, ty), limb.upper, limb.lower, limb.bend, limb.max_bend)
            legs[k] = LimbSolve(k, ik.root, ik.joint, ik.end, (tx, ty), ik.reached, ik.rel, limb.width, limb.side)
        joints = {"root": (0.0, 0.0), "torso": self.to_world(lay, (lay["bx"], lay["by"])),
                  "neck_base": self.to_world(lay, lay["neck_base"]), "head": self.to_world(lay, (lay["hx"], lay["hy"])),
                  "tail_root": self.to_world(lay, lay["tail_base"])}
        for k, v in hips.items():
            joints[f"hip_{k}"] = v
        rig = RigPose("quadruped", sk.id, lay, lay["tilt"], sk.pivot, joints, legs)
        rig.tail, rig.tail_widths = self._tail(p, lay)
        return rig

    def _tail(self, p, lay) -> tuple[list, tuple]:
        t = self.sk.tail
        if not t.lengths:
            return [], ()
        chain = getattr(p, "tail_chain", None) or ()
        base = self.to_world(lay, lay["tail_base"])
        ang = lay["tilt"] + t.rest[0] - p.tail
        pts = [base]
        x, y = base
        for i, ln in enumerate(t.lengths):
            if i > 0:
                ang += chain[i] if i < len(chain) else t.rest[i]
            elif chain:
                ang += chain[0]       # base follow-through offset
            x += ln * math.cos(math.radians(ang))
            y += ln * math.sin(math.radians(ang))
            pts.append((x, y))
        return pts, t.widths

    def press_point(self) -> tuple[float, float]:
        """Paw / hand tip while pressing a button (units, facing right). Reachable by IK."""
        return (29.0, -46.0) if self.sk.kind == "biped" else (38.0, -24.0)

    def _solve_biped(self, p, lay, feet) -> RigPose:
        sk = self.sk
        hips = {k: self.to_world(lay, v) for k, v in lay["hips_body"].items()}
        sh = {k: self.to_world(lay, v) for k, v in lay["shoulders_body"].items()}
        s = lay["s"]
        ank = sk.ankle
        legs, arms = {}, {}
        tuck = _clamp(p.hop, 0.0, 1.0) * (1 - s)
        for limb in sk.legs:
            k = limb.name
            hx, hy = hips[k]
            gx, gy = feet.get(k, (hx, -ank))
            sit_t = (hx + 19.0, -ank - 1.0)
            tx, ty = self._blend((gx, gy), sit_t, s)
            stomp = getattr(p, "stomp", 0.0)
            if k == "R" and stomp > 0:         # angry stomp: knee up, then slam
                tx, ty = self._blend((tx, ty), (hx + 4.0, hy + 12.0), _clamp(stomp, 0.0, 1.0))
            if tuck > 0 or p.lift > 4:
                w = max(tuck, smoothstep((p.lift - 4.0) / 8.0) * (1 - s))
                tt = (hx + (5.0 if k == "L" else -4.0), hy + 16.0)
                tx, ty = self._blend((tx, ty), tt, w)
            tx, ty = self._override(k, tx, ty)
            ik = two_bone_ik((hx, hy), (tx, ty), limb.upper, limb.lower, limb.bend, limb.max_bend)
            legs[k] = LimbSolve(k, ik.root, ik.joint, ik.end, (tx, ty), ik.reached, ik.rel, limb.width, limb.side)
        # ---- arms: legacy swing angles become IK targets, actions override
        amp = p.leg_amp * 24 * 1.15
        ph = p.leg_phase
        rb = _clamp(p.gait_run, 0.0, 1.0)
        swing = amp * (0.9 + 0.7 * rb)
        far_a = -math.sin(ph) * swing + 8 + rb * -18
        far_a = far_a * (1 - p.arms_up) + 155 * p.arms_up
        if p.read > 0:
            far_a = far_a * (1 - p.read) + (-58) * p.read
        near_a = math.sin(ph) * swing - 8 + rb * -18
        if p.paw > 0:
            near_a = near_a * (1 - p.paw) + (-118 * p.paw + p.paw_wave)
        if p.press != 0 or p.ripple > 0:
            aim = -88 - 30 * max(0.0, -p.press)
            k_ = _clamp(p.paw, 0.0, 1.0)
            near_a = near_a * (1 - k_) + aim * k_
        if p.scratch > 0:
            near_a = near_a * (1 - p.scratch) + (-168 + 10 * math.sin(p.mark_phase * 24)) * p.scratch
        if p.read > 0:
            near_a = near_a * (1 - p.read) + (-62) * p.read
        near_a = near_a * (1 - p.arms_up) + (-155 + p.paw_wave * 0.5) * p.arms_up
        if getattr(p, "paw2", 0.0) > 0:
            far_a = far_a * (1 - p.paw2) + (-150 - p.paw_wave * 0.5) * p.paw2
        head_w = self.to_world(lay, (lay["hx"], lay["hy"]))
        chest = self.to_world(lay, (14.0 + lay["shift"], lay["body_top"] + 14.0))
        for limb in sk.arms:
            k = limb.name
            ox, oy = sh[k]
            reach = limb.upper + limb.lower
            a = near_a if k == "arm_R" else far_a
            bend_room = 0.9 if k == "arm_R" else 0.93
            tx, ty = self._paw_target((ox, oy), a, reach * bend_room)
            if k == "arm_R":
                if p.press > 0:
                    tx, ty = self._blend((tx, ty), self.press_point(), _clamp(p.press, 0.0, 1.0))
                point = getattr(p, "point", 0.0)
                if point > 0:
                    tx, ty = self._blend((tx, ty), (ox + reach * 0.98, oy - 3.0), point)
                chin = getattr(p, "chin", 0.0)
                if chin > 0:
                    tx, ty = self._blend((tx, ty), (head_w[0] + 7.0, head_w[1] + 21.0), chin)
                write = getattr(p, "write", 0.0)
                if write > 0:
                    wt = (chest[0] + 8 + 3 * math.sin(p.mark_phase * 9), chest[1] + 6 + 1.2 * math.sin(p.mark_phase * 23))
                    tx, ty = self._blend((tx, ty), wt, write)
                thumb = getattr(p, "thumb", 0.0)
                if thumb > 0:
                    tx, ty = self._blend((tx, ty), (ox + 13.0, oy - 9.0), thumb)
            cover = getattr(p, "cover", 0.0)
            if cover > 0:                      # hands over the face
                tx, ty = self._blend((tx, ty), (head_w[0] + (9.0 if k == "arm_R" else 3.0), head_w[1] + 6.0), cover)
            cross = getattr(p, "cross", 0.0)
            if cross > 0:                      # arms crossed in front of the chest
                tgt = (chest[0] - 4.0, chest[1] + 2.0) if k == "arm_R" else (chest[0] + 5.0, chest[1] + 0.5)
                tx, ty = self._blend((tx, ty), tgt, cross)
            belly = getattr(p, "belly", 0.0)
            if belly > 0:                      # holding the belly (laughing)
                tgt = (chest[0] + (2.0 if k == "arm_R" else -2.0), chest[1] + 10.0)
                tx, ty = self._blend((tx, ty), tgt, belly)
            splay = 0.0
            hand_z = 2.0 * (1.0 - _clamp(p.arms_up, 0.0, 1.0))       # hands hang just clear of the body
            fists = _clamp(getattr(p, "fists", 0.0), 0.0, 1.0)
            if fists > 0:                      # clenched, elbows bent and out, fists braced at the hips
                tx, ty = self._blend((tx, ty), (ox + 5.0, oy + 15.0), fists * 0.85)
                splay = max(splay, 0.45 * fists)
                hand_z -= 2.0 * fists
            akimbo = _clamp(getattr(p, "akimbo", 0.0), 0.0, 1.0)
            if akimbo > 0:                     # hands on hips, elbows out
                tx, ty = self._blend((tx, ty), (ox + 1.0, lay["hip_y"] - 12.0), akimbo)
                splay = max(splay, akimbo)
                hand_z -= 7.0 * akimbo
            hug = _clamp(getattr(p, "hug", 0.0), 0.0, 1.0)
            if hug > 0:                        # arms pulled in, hands to the chest / opposite shoulder
                tgt = (chest[0] + 1.0, chest[1] - 5.0) if k == "arm_R" else (chest[0] + 3.0, chest[1] - 7.0)
                tx, ty = self._blend((tx, ty), tgt, hug)
                hand_z -= 13.0 * hug
            arms_out = _clamp(getattr(p, "arms_out", 0.0), 0.0, 1.0)
            if arms_out > 0:                   # open arms (excited / surprised / shrug)
                pt = self._paw_target((ox, oy), -40.0 if k == "arm_R" else -32.0, reach * 0.9)
                tx, ty = self._blend((tx, ty), pt, arms_out * 0.8)
                splay = max(splay, 0.6 * arms_out)
                hand_z += 16.0 * arms_out
            clap = getattr(p, "clap", 0.0)
            if clap > 0:
                gap = 3.0 + 6.0 * abs(math.sin(p.mark_phase * 9))
                cx, cy = chest[0] + 6.0, chest[1] - 6.0
                tx, ty = self._blend((tx, ty), (cx + (gap if k == "arm_R" else -gap) * 0.5, cy), clap)
            tx, ty = self._override(k, tx, ty)
            ik = two_bone_ik((ox, oy), (tx, ty), limb.upper, limb.lower, limb.bend, limb.max_bend)
            arms[k] = LimbSolve(k, ik.root, ik.joint, ik.end, (tx, ty), ik.reached, ik.rel, limb.width, limb.side)
            arms[k].splay = splay
            arms[k].hand_z = hand_z
        joints = {"root": (0.0, 0.0), "pelvis": self.to_world(lay, (lay["shift"], lay["hip_y"])),
                  "neck_base": self.to_world(lay, (lay["shift"], lay["body_top"])), "head": head_w,
                  "torso": self.to_world(lay, (lay["shift"], lay["hip_y"]))}
        for k, v in hips.items():
            joints[f"hip_{k}"] = v
        for k, v in sh.items():
            joints[f"shoulder_{k}"] = v
        return RigPose("biped", sk.id, lay, lay["tilt"], sk.pivot, joints, legs, arms)
