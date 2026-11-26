"""Rig-driven skeletal animation for mesh characters (production GLB assets).

The animation system still produces one ``Pose`` + solved ``RigPose`` per frame (planted
feet, IK, gaits, emotions, secondary springs).  ``RigDriver`` turns that into world
transforms for *any* skeleton that uses the canonical bone names:

1. targets     pelvis / chest / neck / head / tail points / every limb (root, knee or
               elbow, ankle or wrist, toe) in internal 3D units, plus the head frame,
               ear, jaw and twist angles
2. rotations   each bone is aimed from its bind direction at its target direction
               (minimal arc + roll for the pelvis / chest twist); the head copies the
               head frame; ears / jaw rotate by their channels; extra bones follow
               their parent
3. FK          positions come from the asset's *own* bone offsets (proportions are
               preserved - this is rotation retargeting)
4. leg / arm IK the limbs of the asset are re-solved with 3D two-bone IK toward the
               rig's foot / hand targets with the rig's knee / elbow as the pole, so
               feet stay planted on the desktop even when the asset's proportions differ
5. clips       an optional GLB animation clip (retargeted through world-space deltas)
               is blended on top per bone
6. skinning    linear blend skinning (4 influences) + morph targets, vectorised NumPy

Everything is plain NumPy so the CPU and GPU raster paths share it.
"""

from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np

from .character_asset import ASSET_TO_INTERNAL, LIMB_BONES, CharacterAsset
from .gltf import quat_to_mat, sample_channel
from .rig import RigPose, two_bone_ik_3d


def rot_z(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_x(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])


def axis_angle(axis, ang: float) -> np.ndarray:
    # scalar math: called many times per frame on 3-vectors, where NumPy's per-call overhead dominates
    x, y, z = float(axis[0]), float(axis[1]), float(axis[2])
    n = math.sqrt(x * x + y * y + z * z)
    if n < 1e-9 or abs(ang) < 1e-12:
        return np.eye(3)
    x, y, z = x / n, y / n, z / n
    c, s = math.cos(ang), math.sin(ang)
    C = 1 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def rotation_between(a, b) -> np.ndarray:
    """Minimal rotation taking direction ``a`` onto direction ``b``."""
    ax, ay, az = float(a[0]), float(a[1]), float(a[2])
    bx, by, bz = float(b[0]), float(b[1]), float(b[2])
    na = math.sqrt(ax * ax + ay * ay + az * az)
    nb = math.sqrt(bx * bx + by * by + bz * bz)
    if na < 1e-9 or nb < 1e-9:
        return np.eye(3)
    ax, ay, az = ax / na, ay / na, az / na
    bx, by, bz = bx / nb, by / nb, bz / nb
    vx, vy, vz = ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx
    c = ax * bx + ay * by + az * bz
    s = math.sqrt(vx * vx + vy * vy + vz * vz)
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        perp = np.cross([ax, ay, az], [1.0, 0.0, 0.0]) if abs(ax) < 0.9 else np.cross([ax, ay, az], [0.0, 1.0, 0.0])
        return axis_angle(perp, math.pi)
    return axis_angle((vx, vy, vz), math.atan2(s, c))


def planar_rotation(a, b) -> np.ndarray:
    """Rotation about the side axis (z) taking ``a``'s side-view direction onto ``b``'s. The rig's
    tail lives in the side view: the minimal 3D rotation flips its axis when a tail swings close to
    the opposite of its rest direction (the tip jumped through the body); this one never does."""
    ax, ay, bx, by = float(a[0]), float(a[1]), float(b[0]), float(b[1])
    if ax * ax + ay * ay < 1e-12 or bx * bx + by * by < 1e-12:
        return rotation_between(a, b)
    return rot_z(math.degrees(math.atan2(ax * by - ay * bx, ax * bx + ay * by)))


def blend_rot(a: np.ndarray, b: np.ndarray, w: float) -> np.ndarray:
    if w <= 0:
        return a
    if w >= 1:
        return b
    from .gltf import mat_to_quat, slerp
    return quat_to_mat(slerp(mat_to_quat(a), mat_to_quat(b), w))


# --------------------------------------------------------------- rig targets
LIMB_Z_QUAD = {"FL": -9.0, "FR": 9.0, "BL": -9.0, "BR": 9.0}
LIMB_Z_BIPED = {"L": -5.5, "R": 5.5, "arm_L": -16.0, "arm_R": 16.0}
XSHIFT_BIPED = {"L": 5.0, "R": -5.0, "arm_L": 11.0, "arm_R": -11.0}


def limb_offsets(kind: str, name: str, layout: dict) -> tuple[float, float]:
    """(x shift, lateral z) of a limb in 3D - the same placement the procedural 3D models use."""
    if kind == "biped":
        return XSHIFT_BIPED.get(name, 0.0), LIMB_Z_BIPED.get(name, 0.0)
    brz = layout.get("bry", 17.0) * 0.9
    return 0.0, LIMB_Z_QUAD.get(name, 0.0) * (brz / 15.3)


def rig_targets(rp: RigPose, kind: str) -> dict:
    """Internal 3D target points for the canonical skeleton from a solved rig."""
    j = rp.joints
    lay = rp.layout
    t: dict = {}

    def p3(pt, z=0.0, dx=0.0):
        return np.array([pt[0] + dx, pt[1], z], dtype=float)
    if kind == "biped":
        sx = lay.get("shift", 0.0)
        t["Pelvis"] = p3((sx, lay["hip_y"]))
        t["Chest"] = p3((sx, lay["body_top"] + 12.0))
        t["Neck"] = p3((sx + 0.5, lay["body_top"] + 1.0))
    else:
        t["Pelvis"] = p3(j.get("pelvis", j["torso"]))
        t["Chest"] = p3(j.get("chest", j["torso"]))
        t["Neck"] = p3(j["neck_base"])
    t["Spine"] = (t["Pelvis"] + t["Chest"]) * 0.5
    t["Head"] = p3(j["head"])
    for d in (rp.legs, rp.arms):
        for name, ls in d.items():
            bones = LIMB_BONES[kind].get(name)
            if not bones:
                continue
            dx, z = limb_offsets(kind, name, lay)
            up, lo, end = bones
            side = 1.0 if z >= 0 else -1.0
            sp = getattr(ls, "splay", 0.0) * side
            hz = getattr(ls, "hand_z", 0.0) * side
            t[up] = p3(ls.root, z, dx)
            spl = abs(sp)
            jx = ls.joint[0] + ((ls.root[0] + ls.end[0]) * 0.5 - ls.joint[0]) * spl     # bend sideways, not back
            jy = ls.joint[1] + ((ls.root[1] + ls.end[1]) * 0.5 - ls.joint[1]) * spl
            t[lo] = p3((jx, jy), z + sp * 14.0 + hz * 0.4, dx)      # elbows / knees pushed outward
            t[end] = p3(ls.end, z + hz, dx)
            t[end + "_tip"] = p3(ls.foot if ls.foot != (0.0, 0.0) else ls.end, z + hz, dx)
    zt = -lay.get("bry", 17.0) * 0.9 * 0.15 if kind != "biped" else 0.0
    for i, pt in enumerate(rp.tail):
        t[f"Tail{i}"] = p3(pt, zt)
    return t


# ------------------------------------------------------------------ driver
_AIM_CHILD = {"Pelvis": "Chest", "Spine": "Chest", "Chest": "Neck", "Neck": "Head",
              "LeftUpperArm": "LeftLowerArm", "LeftLowerArm": "LeftHand", "RightUpperArm": "RightLowerArm",
              "RightLowerArm": "RightHand", "LeftUpperLeg": "LeftLowerLeg", "LeftLowerLeg": "LeftFoot",
              "RightUpperLeg": "RightLowerLeg", "RightLowerLeg": "RightFoot",
              "LeftHand": "LeftHand_tip", "RightHand": "RightHand_tip", "LeftFoot": "LeftFoot_tip",
              "RightFoot": "RightFoot_tip"}


class RigDriver:
    """Poses an asset skeleton from the rig every frame and skins its meshes."""

    def __init__(self, asset: CharacterAsset, kind: str, species: str = ""):
        self.asset = asset
        self.kind = kind
        self.species = species
        self._ground_odo = 0.0          # distance walked (internal units), integrated from the leg phase
        self._last_leg_phase: Optional[float] = None
        self._foot_lock: dict = {}      # foot joint -> ground-frame x where it was planted
        self._leg_state: dict = {}      # rig-driven feet of assets with their own proportions (_leg_goals)
        J = asset.joints
        self.n = len(J)
        self.B = np.array([j.bind_world[:3, 3] for j in J]) if J else np.zeros((0, 3))
        self.parent = [j.parent for j in J]
        self.canon = {j.canonical: i for i, j in enumerate(J) if j.canonical}
        self.order = self._topo()
        self.children = {i: [k for k in range(self.n) if self.parent[k] == i] for i in range(self.n)}
        # bind aim directions
        self.aim0: dict = {}
        tails = sorted((int(c[4:]), i) for c, i in self.canon.items() if c and c.startswith("Tail"))
        self.tail_idx = [i for _, i in tails]
        for c, i in self.canon.items():
            child = _AIM_CHILD.get(c)
            if child and child in self.canon:
                self.aim0[i] = self.B[self.canon[child]] - self.B[i]
            elif child and child.endswith("_tip"):
                kids = self.children.get(i, [])
                if kids:
                    self.aim0[i] = self.B[kids[0]] - self.B[i]
                else:
                    self.aim0[i] = np.array([1.0, 0.25, 0.0]) if self.kind != "biped" or "Foot" in c \
                        else self.B[i] - self.B[self.parent[i]]
        for k, i in enumerate(self.tail_idx):
            if k + 1 < len(self.tail_idx):
                self.aim0[i] = self.B[self.tail_idx[k + 1]] - self.B[i]
            else:
                kids = self.children.get(i, [])
                self.aim0[i] = (self.B[kids[0]] - self.B[i]) if kids else (self.B[i] - self.B[self.parent[i]])
        self.world_R = np.repeat(np.eye(3)[None], self.n, axis=0)
        self.world_T = self.B.copy()
        self.clip: Optional[tuple] = None          # (Clip, time, weight, mask)
        self.last_ms = 0.0
        # skeletal clips (artist-made GLB animation): evaluated first, then IK / look-at / offsets on top
        self.animator = None
        self.clip_morphs: dict = {}
        self.root_delta = np.zeros(3)              # ROOT_MOTION policy: internal units this frame
        self.clip_weight = 0.0
        self.clip_info = ("", "")
        from .ground import GroundSurface
        self.surface = GroundSurface()             # what the feet stand on (ground.py)
        self.use_clip_animation = True
        if any(getattr(c, "channels", None) for c in getattr(asset, "clips", {}).values()) and \
                all(j.rest_r is not None for j in asset.joints):
            try:
                from .clip_anim import ClipAnimator
                self.animator = ClipAnimator(asset)
                if not self.animator.clips:
                    self.animator = None
            except Exception:  # noqa: BLE001 - clips are optional; the procedural rig always works
                import logging
                logging.getLogger("focusbuddy.companion").exception("clip animator unavailable")
                self.animator = None
        self._subtree_cache: dict = {}
        self.rig = None                            # set by the renderer (neutral pose for clip placement)
        # metadata "attach": {"<node>": "<joint>"} - a node outside a limb chain (e.g. a paw driven by
        # an IK target in the source rig) rigidly follows a chain joint, so IK / foot locks / procedural
        # poses never pull the paw away from its leg
        by_name = {j.name: i for i, j in enumerate(J)}
        self.attach = []
        for child, par in ((getattr(asset, "meta", None) or {}).get("attach") or {}).items():
            c, p = by_name.get(child, -1), by_name.get(par, -1)
            if c >= 0 and p >= 0 and c != p:
                self.attach.append((c, p))
        # metadata "head_scale": a cuter, more stylised (chibi) look - the head and everything attached
        # to it (ears, eyes, muzzle) is drawn this much bigger, uniformly about the head joint, in any pose
        try:
            self.head_scale = float((getattr(asset, "meta", None) or {}).get("head_scale", 1.0) or 1.0)
        except (TypeError, ValueError):
            self.head_scale = 1.0
        self.head_scale = max(0.7, min(1.6, self.head_scale))
        # the final pose never jumps between frames (inertia.py). Left out: the eyes (a saccade is
        # instant) and everything that stands on the ground - the pelvis and the leg chains (all four
        # for animals), whose feet the IK keeps planted; an offset there would make them slide
        from .inertia import PoseInertializer, joint_scale, optional_exclude
        excl = set(optional_exclude(self.canon, ("LeftEye", "RightEye", "Pelvis", "Root")))
        for name, bones in LIMB_BONES[kind if kind in LIMB_BONES else "quadruped"].items():
            if kind == "biped" and name.startswith("arm"):
                continue                                  # a person's arms are free: filtered
            for b in bones:
                i = self.canon.get(b, -1)
                if i >= 0:
                    excl.update(self._subtree(i))
        self.inertia = PoseInertializer(self.parent, sorted(excl), joint_scale(self.B))
        self._last_clock: Optional[float] = None

    def _topo(self) -> list:
        out, seen = [], set()

        def visit(i):
            if i in seen or i < 0:
                return
            p = self.parent[i]
            if p >= 0:
                visit(p)
            seen.add(i)
            out.append(i)
        for i in range(self.n):
            visit(i)
        return out

    # ---------------------------------------------------------------- pose
    def pose(self, pose, rp: RigPose, head_R: np.ndarray) -> np.ndarray:
        """Returns the (J, 4, 4) skinning matrices (bind -> posed, internal units)."""
        t0 = time.perf_counter()
        try:
            return self._pose(pose, rp, head_R)
        finally:
            self.last_ms = (time.perf_counter() - t0) * 1000

    def _pose(self, pose, rp: RigPose, head_R: np.ndarray) -> np.ndarray:
        if self.n == 0:
            return np.zeros((0, 4, 4))
        tg = rig_targets(rp, self.kind)
        n = self.n
        res = None
        if self.animator is not None and self.use_clip_animation:
            res = self.animator.update(pose)
        full_clip = False
        if res is not None and res.base:
            full_clip = float(res.weight) * (1.0 - float(res.proc_full)) >= 0.999 and res.proc_upper <= 0
        if full_clip:
            # the clip covers every joint: the rig's own pose would be blended away - skip it
            D = np.repeat(np.eye(3)[None], n, axis=0)
            T = self.B.copy()
        else:
            D, T = self._rig_pose(pose, rp, head_R, tg)
        self.clip_weight = 0.0
        self.clip_morphs = {}
        self.root_delta = np.zeros(3)
        if res is not None and res.base:
            Dc, Tc = self._clip_pose(res, pose, rp, head_R, tg)
            w = float(max(0.0, min(1.0, res.weight))) * (1.0 - float(res.proc_full))
            wj = np.full(n, w)
            if res.proc_upper > 0:
                ch = self.canon.get("Chest", -1)
                if ch >= 0:
                    wj[self._subtree(ch)] *= (1.0 - float(res.proc_upper))
            if full_clip:
                D, T = Dc, Tc
            else:
                D = np.array([blend_rot(D[i], Dc[i], float(wj[i])) for i in range(n)])
                T = T + (Tc - T) * wj[:, None]
            self.clip_weight = w
            self.clip_morphs = res.morphs
            self.root_delta = self._C3 @ res.root_delta if res.root_delta is not None else np.zeros(3)
            self.clip_info = (res.base, res.action)
        else:
            if res is not None:
                self.clip_morphs = res.morphs
            self.clip_info = ("", res.action if res is not None else "")
            if self.clip is not None:
                D, T = self._apply_clip(D, T)
        clock = getattr(pose, "clock", None)
        if clock is not None:
            clock = float(clock)
            dt = 0.0 if self._last_clock is None else clock - self._last_clock
            if dt < 0.0:                                 # the clock went back (a new scene): start over
                self.inertia.reset()
                dt = 0.0
            self._last_clock = clock
            D, T = self.inertia.apply(D, T, dt)
        self._eyes(D, T, pose)
        M = np.zeros((n, 4, 4))
        M[:, :3, :3] = D
        M[:, :3, 3] = T - np.einsum("nij,nj->ni", D, self.B)
        M[:, 3, 3] = 1.0
        hs = self.head_scale
        hi = self.canon.get("Head", -1)
        if abs(hs - 1.0) > 1e-3 and hi >= 0:
            # uniform scale of the posed head subtree about the posed head joint: p' = P + s (p - P)
            idx = self._subtree_idx(hi)
            P = T[hi].copy()
            M[idx, :3, :] *= hs
            M[idx, :3, 3] += (1.0 - hs) * P
            T = T.copy()
            T[idx] = P + hs * (T[idx] - P)
        self.world_R, self.world_T = D, T
        return M

    def _rig_pose(self, pose, rp: RigPose, head_R: np.ndarray, tg: dict):
        """The procedural rig's pose on the asset's bones: FK from the rig targets + limb IK."""
        n = self.n
        D = np.repeat(np.eye(3)[None], n, axis=0)        # world-space delta rotations
        T = np.zeros((n, 3))
        known = set()
        canon_of = {i: c for c, i in self.canon.items()}
        # metadata "pelvis_lift": "auto" - a model whose hips sit higher / lower than the rig's (long /
        # short legs for its height) stands at its own hip height instead of crouching or floating;
        # fades out when sitting / lying
        up_lift = np.array([0.0, self._pelvis_lift() * float(rp.layout.get("stand", 1.0)), 0.0])
        twist = float(getattr(pose, "torso_twist", 0.0))
        roll = float(getattr(pose, "pelvis_roll", 0.0))
        for i in self.order:
            c = canon_of.get(i)
            p = self.parent[i]
            # ---- position: FK from the parent (asset proportions), pelvis / root placed by the rig
            if c == "Root" or p < 0:
                T[i] = self.B[i] if c != "Pelvis" else tg.get("Pelvis", self.B[i])
                if c == "Pelvis":
                    T[i] = tg["Pelvis"] - up_lift
            elif c == "Pelvis":
                T[i] = tg["Pelvis"] - up_lift
            else:
                T[i] = T[p] + D[p] @ (self.B[i] - self.B[p])
            # ---- rotation
            if c is None:
                D[i] = D[p] if p >= 0 else np.eye(3)
                continue
            if c == "Root":
                continue
            if c == "Head":
                D[i] = head_R
            elif c in ("Pelvis", "Spine", "Chest", "Neck"):
                child = {"Pelvis": "Chest", "Spine": "Chest", "Chest": "Neck", "Neck": "Head"}[c]
                d_now = tg[child] - tg[c] if c != "Pelvis" else tg["Chest"] - tg["Pelvis"]
                R = rotation_between(self.aim0.get(i, d_now), d_now)
                if c == "Pelvis":
                    R = axis_angle(d_now, math.radians(roll)) @ R
                elif c == "Chest":
                    R = axis_angle(d_now, math.radians(-roll * 0.5 + twist * (1 if self.kind == "biped" else 0.6))) @ R
                D[i] = R
            elif c.startswith("Tail"):
                k = self.tail_idx.index(i) if i in self.tail_idx else -1
                if k >= 0 and k + 1 < len(rp.tail) and f"Tail{k + 1}" in tg:
                    D[i] = planar_rotation(self.aim0[i], tg[f"Tail{k + 1}"] - tg[f"Tail{k}"])
                elif p >= 0:
                    D[i] = D[p]
            elif c in ("LeftEar", "RightEar"):
                D[i] = D[p] @ self._ear(pose, c == "RightEar")
            elif c == "Jaw":
                jaw = max(0.0, min(1.0, float(getattr(pose, "jaw", 0.0))))
                D[i] = D[p] @ rot_z(22.0 * jaw)
            elif i in self.aim0:
                D[i] = D[p] if p >= 0 else np.eye(3)     # limbs are re-solved with IK below
            else:
                D[i] = D[p] if p >= 0 else np.eye(3)
            known.add(i)
        sit = self._sit_posture(D, T, rp, tg) if self.kind != "biped" else 0.0
        bow = self._bow_posture(D, T, pose, sit) if self.kind != "biped" and self._lift_mode() else 0.0
        if self.kind == "biped" and self.animator is None:
            self._posture(D, T, pose)
        leg_goals = self._leg_goals(T, tg, pose) if self._lift_mode() else {}
        air, air_u = self._air_posture(D, T, pose) if self.kind != "biped" and self._lift_mode() else (0.0, 0.0)
        wave_w, wave_leg = 0.0, ""
        if self.kind != "biped" and self._lift_mode() and air <= 0.0:
            wave_w = max(0.0, min(1.0, (float(getattr(pose, "paw", 0.0)) - 0.15) / 0.5))
            if wave_w > 0.0 and "LeftHand" in tg and "RightHand" in tg:
                # which paw waves is decided once, when the wave starts, and kept until it ends:
                # re-deciding every frame (both paws still near the ground) made the paw flicker
                # (the rig lifts its FR paw for ``paw`` and FL for ``paw2``)
                prev = getattr(self, "_wave_leg", "")
                if not prev:
                    prev = "FR" if float(getattr(pose, "paw", 0.0)) >= float(getattr(pose, "paw2", 0.0)) else "FL"
                wave_leg = self._wave_leg = prev
        if wave_w <= 0.0:
            self._wave_leg = ""
        # ---- limbs: two-bone IK on the asset's own bones toward the rig's targets
        wave_pole = None
        if self.kind == "biped":
            shz = [float(T[self.canon[b]][2]) for b in ("LeftUpperArm", "RightUpperArm") if self.canon.get(b, -1) >= 0]
            mid_z = sum(shz) / len(shz) if shz else 0.0
        for name, bones in LIMB_BONES[self.kind].items():
            up, lo, end = (self.canon.get(b, -1) for b in bones)
            if min(up, lo, end) < 0 or bones[0] not in tg:
                continue
            root = T[up]
            l1 = float(np.linalg.norm(self.B[lo] - self.B[up]))
            l2 = float(np.linalg.norm(self.B[end] - self.B[lo]))
            goal = tg[bones[2]]
            if self._lift_mode():
                # proportions differ from the rig's (hips / shoulders elsewhere, other limb lengths):
                # the rig's foot / hand offset from its hip / shoulder is carried over to the asset's own
                # hip / shoulder (scaled to the asset's limb), feet keep the rig's ground height
                if self.kind == "biped" and name.startswith("arm"):
                    goal = root + (tg[bones[2]] - tg[bones[0]]) * self._limb_scale(name, l1 + l2)
                    # keep the hands clear of the body so they read (metadata "arm_out" / "arm_fwd":
                    # fractions of the arm's length, sideways / forward)
                    meta = getattr(self.asset, "meta", None) or {}
                    out_k, fwd_k = float(meta.get("arm_out", 0.0)), float(meta.get("arm_fwd", 0.0))
                    side = 1.0 if float(self.B[up][2]) >= 0.0 else -1.0
                    if out_k or fwd_k:
                        goal = goal + np.array([fwd_k, 0.0, side * out_k]) * (l1 + l2)
                    # a real wave (metadata "wave_up"): the waving hand goes up beside the head and
                    # swings from the elbow, instead of the animals' raised front paw
                    wave = float(getattr(pose, "paw", 0.0)) if meta.get("wave_up") else 0.0
                    fists_ = float(getattr(pose, "fists", 0.0))
                    pump = wave if fists_ > 0.5 else 0.0             # a fist pump, not a wave
                    if pump:
                        wave = 0.0
                    # expressive hand poses read in the 3/4 view: in front of / out from the body, at
                    # chest or shoulder height (the side-view rig's targets point straight at the camera
                    # or were pinned down by the low-arm limit below)
                    L = l1 + l2
                    expr, eg = self._expressive_arm(name, pose, root, side, L, mid_z, pump)
                    if expr > 0.0:
                        goal = goal + (eg - goal) * min(1.0, expr)
                        wave_pole = np.array([0.0, 1.0, side * 1.0])
                    # which hand waves is decided when the wave starts and kept until it ends (re-deciding
                    # every frame by the hands' heights swapped the arms mid-wave: a hand teleported)
                    if wave > 0.05:
                        if not getattr(self, "_wave_arm", ""):
                            self._wave_arm = "arm_L" if tg["LeftHand"][1] <= tg["RightHand"][1] else "arm_R"
                    else:
                        self._wave_arm = ""
                    if wave > 0.05 and name == self._wave_arm:
                        rel = np.array([0.08 * L, -0.8 * L, side * 0.6 * L])
                        rel = axis_angle((1.0, 0.0, 0.0), math.radians(0.6 * float(getattr(pose, "paw_wave", 0.0)))) @ rel
                        goal = goal + (root + rel - goal) * min(1.0, wave)
                        wave_pole = np.array([0.0, 1.0, side * 1.2])
                    # an arm reconstructed against the body stretches the side of the torso when it
                    # swings up: keep the hand at most ~100 deg away from hanging down (a wave about shoulder height)
                    rest = self.B[end] - self.B[up]
                    v = goal - root
                    nv, nr = float(np.linalg.norm(v)), float(np.linalg.norm(rest))
                    if nv > 1e-6 and nr > 1e-6:
                        ang = math.acos(max(-1.0, min(1.0, float(v @ rest) / (nv * nr))))
                        lim_deg = float((getattr(self.asset, "meta", None) or {}).get("arm_limit", 100.0))
                        # arms go high only to wave or cheer: a run's arm swing stays low (a stiff arm
                        # held out level reads as a sleepwalker)
                        lim = math.radians(lim_deg)
                        if ang > lim:
                            axis = np.cross(rest, v)
                            if float(np.linalg.norm(axis)) > 1e-9:
                                goal = root + axis_angle(axis, lim) @ (rest / nr) * nv
                        # arms go high only to wave or cheer: otherwise (a run's arm swing) the hand
                        # stays within ~50 deg of hanging straight down - an arm held out level reads
                        # as a sleepwalker
                        high = max(wave, float(getattr(pose, "arms_up", 0.0)), 0.5 * expr,
                                   0.6 * float(getattr(pose, "arms_out", 0.0)))
                        if high < 0.5:
                            down = np.array([0.0, 1.0, 0.0])            # internal +y is down
                            v = goal - root
                            lim2 = math.radians(50.0 + 130.0 * high * 2.0)
                            a2 = math.acos(max(-1.0, min(1.0, float(v @ down) / max(1e-9, float(np.linalg.norm(v))))))
                            if a2 > lim2:
                                axis = np.cross(down, v)
                                if float(np.linalg.norm(axis)) > 1e-9:
                                    goal = root + axis_angle(axis, lim2) @ down * float(np.linalg.norm(v))
                else:
                    goal = leg_goals[name]
            if bow > 0.0 and name in ("FL", "FR"):
                # the stretch / play bow: front paws reach far forward on the ground
                goal = goal.copy()
                goal[0] += 0.5 * (l1 + l2) * bow
            if air > 0.0 and name in ("FL", "FR", "BL", "BR"):
                goal = goal + (self._air_goal(name, root, l1 + l2, air_u) - goal) * air
            if self.kind != "biped" and self._lift_mode() and name == wave_leg:
                # a proper paw wave: the near front paw comes up to the chin and waves side to side
                ph = math.radians(float(getattr(pose, "paw_wave", 0.0)))
                L_ = l1 + l2
                wg = root + np.array([(0.72 + 0.12 * math.sin(ph * 2.2)) * L_, (-0.1 - 0.12 * math.cos(ph * 2.2)) * L_,
                                      0.08 * L_ * (1.0 if name == "FR" else -1.0)])
                goal = goal + (wg - goal) * wave_w
            if sit > 0.0 and name in ("FL", "FR"):
                # sitting: front legs straight down from the shoulders, paws just in front of them
                goal = goal.copy()
                goal[0] += (root[0] + 0.12 * (l1 + l2) - goal[0]) * sit
            pole = tg[bones[1]] - (tg[bones[0]] + tg[bones[2]]) * 0.5
            if wave_pole is not None:
                pole, wave_pole = wave_pole, None
            if self._lift_mode() and not name.startswith("arm"):
                pole = np.array([pole[0], min(0.0, pole[1]), pole[2]])   # knees / elbows never bend into the floor
                # a nearly straight rig leg gives a tiny, noisy pole whose sign can flip from one frame
                # to the next (the elbow / knee snapped to the other side): bias it toward the asset's
                # own bind-pose bend so the joint always bends the same way
                bind_pole = self.B[lo] - (self.B[up] + self.B[end]) * 0.5
                bind_pole = np.array([bind_pole[0], 0.0, bind_pole[2]])
                bpn = float(np.linalg.norm(bind_pole))
                if bpn > 1e-6:
                    pole = pole + bind_pole / bpn * 0.35 * (l1 + l2)
            jnt, eff, _ = two_bone_ik_3d(tuple(root), tuple(goal), l1, l2, tuple(pole), 170.0)
            jnt, eff = np.array(jnt), np.array(eff)
            D[up] = rotation_between(self.B[lo] - self.B[up], jnt - root)
            D[lo] = rotation_between(self.B[end] - self.B[lo], eff - jnt)
            T[lo] = jnt
            T[end] = eff
            tip = tg.get(bones[2] + "_tip")
            aim = self.aim0.get(end)
            if tip is not None and aim is not None and np.linalg.norm(tip - goal) > 1e-6:
                D[end] = rotation_between(aim, tip - goal)
                if self._lift_mode():
                    # a reconstructed shoe / paw is one solid lump: bending it far from its shin
                    # squashes it, so the ankle turns at most ~50 deg (flat on the ground when standing)
                    rel = D[lo].T @ D[end]
                    ang = math.acos(max(-1.0, min(1.0, (float(np.trace(rel)) - 1.0) * 0.5)))
                    lim = math.radians(50.0)
                    if ang > lim:
                        D[end] = blend_rot(D[lo], D[end], lim / ang)
                    if not name.startswith("arm"):
                        # a planted foot / paw lies flat as in the bind pose (the model stands on its
                        # paws there): aimed like the rig's, a long paw tips its toes into the floor
                        ank_h = float(getattr(getattr(self.rig, "sk", None), "ankle", 0.0) or 0.0)
                        above = max(0.0, -float(tg[bones[2]][1]) - ank_h)
                        flat = max(0.0, min(1.0, 1.0 - above / (2.0 * ank_h + 1.0)))
                        if flat > 0.0:
                            D[end] = blend_rot(D[end], np.eye(3), flat)
            else:
                D[end] = D[lo]
            fists = float(getattr(pose, "fists", 0.0))
            if fists > 0 and "Hand" in bones[2]:
                D[end] = D[end] * (1.0 - 0.2 * min(1.0, fists))          # clenched: a smaller, rounder hand
            # everything below the limb (toes, fingers) follows rigidly
            stack = list(self.children.get(end, []))
            while stack:
                k = stack.pop()
                D[k] = D[self.parent[k]]
                T[k] = T[self.parent[k]] + D[self.parent[k]] @ (self.B[k] - self.B[self.parent[k]])
                stack.extend(self.children.get(k, []))
        self._tail_floor(D, T)
        # descendants of re-solved bones that aren't limbs (e.g. ear tips) were placed during FK already
        for c, p in self.attach:
            self._follow(D, T, c, p, np.eye(3), self.B[c] - self.B[p])
        return D, T

    def _sit_posture(self, D, T, rp: RigPose, tg: dict) -> float:
        """Quadruped sit on an artist rig: the spine pitches up about the hips until the front legs
        (the asset's own, often long, legs) can stand nearly straight, like a real sitting dog or cat.
        Returns the sit weight (0 = not sitting)."""
        s = float(rp.layout.get("s", 0.0)) * (1.0 - float(rp.layout.get("l", 0.0)))
        pel = self.canon.get("Pelvis", -1)
        sh = self.canon.get("LeftUpperArm", -1)
        lo = self.canon.get("LeftLowerArm", -1)
        hand = self.canon.get("LeftHand", -1)
        if s < 0.05 or min(pel, sh, lo, hand) < 0:
            return 0.0
        L = float(np.linalg.norm(self.B[lo] - self.B[sh]) + np.linalg.norm(self.B[hand] - self.B[lo]))
        ground = float(tg.get("LeftHand", np.array([0.0, 0.0, 0.0]))[1])
        want = 0.95 * L                                       # shoulder height over the paw: front legs straight
        # the rump goes down onto the haunches (the rig's own sit barely lowers a generated model's
        # high hips): the pelvis drops to about 45 % of its standing height, the hind legs fold
        hip_h = max(0.0, ground - float(T[pel][1]))
        # metadata "sit_drop": how far the rump goes down (a cat sits right down on its haunches,
        # which pitches its long body up more - the classic upright cat sit)
        k_drop = float((getattr(self.asset, "meta", None) or {}).get("sit_drop", 0.55))
        drop = hip_h * k_drop * s
        if drop > 0.0:
            for k in self._subtree(pel):
                T[k] = T[k] + np.array([0.0, drop, 0.0])
        piv = T[pel].copy()
        v = T[sh] - piv

        def height(deg: float) -> float:
            return ground - float((piv + rot_z(deg) @ v)[1])  # internal +y is down
        if height(0.0) >= want:
            return s
        sign = 1.0 if height(10.0) > height(-10.0) else -1.0
        lo_a, hi_a = 0.0, 55.0
        if height(sign * hi_a) < want:
            ang = hi_a
        else:
            for _ in range(14):
                mid = 0.5 * (lo_a + hi_a)
                if height(sign * mid) >= want:
                    hi_a = mid
                else:
                    lo_a = mid
            ang = hi_a
        ang *= sign * s
        hd = self.canon.get("Head", -1)
        head_D = D[hd].copy() if hd >= 0 else None
        R = rot_z(ang)
        self._rotate(D, T, pel, R)
        if hd >= 0:                                           # the head keeps its own aim (look-at)
            self._rotate(D, T, hd, head_D @ np.linalg.inv(D[hd]))
        return s

    def _bow_posture(self, D, T, pose, sit: float) -> float:
        """Quadruped stretch / play bow on a generated model: the chest goes down toward the floor
        (the spine pitches nose-down about the hips, rump up), the head keeps its aim. The front paws
        reach forward (see the limb goals). Returns the bow weight."""
        w = max(0.0, min(1.0, float(getattr(pose, "stretch", 0.0)))) * (1.0 - sit)
        pel, sh = self.canon.get("Pelvis", -1), self.canon.get("LeftUpperArm", -1)
        if w < 0.03 or min(pel, sh) < 0:
            return 0.0
        piv = T[pel].copy()
        v = T[sh] - piv
        down = 1.0 if float((rot_z(10.0) @ v)[1]) > float((rot_z(-10.0) @ v)[1]) else -1.0
        hd = self.canon.get("Head", -1)
        head_D = D[hd].copy() if hd >= 0 else None
        self._rotate(D, T, pel, rot_z(down * 24.0 * w))
        if hd >= 0:
            self._rotate(D, T, hd, rot_z(-down * 10.0 * w) @ head_D @ np.linalg.inv(D[hd]))
        return w

    def _air_posture(self, D, T, pose) -> tuple:
        """A quadruped's leap: nose up on the way up (front paws reaching, hind legs pushing off
        behind), legs gathered at the top, nose down on the way down with the front paws reaching
        for the ground. Returns (air weight, flight position -1 takeoff .. 0 apex .. +1 landing)."""
        air = max(0.0, min(1.0, (float(getattr(pose, "hop", 0.0)) - 0.2) / 0.6))
        st = getattr(self, "_air_state", None)
        if st is None:
            st = self._air_state = {"v0": 1.0}
        vy = float(getattr(pose, "jump_vy", 0.0))
        clock = float(getattr(pose, "clock", 0.0))
        dt = clock - st.get("t", clock)
        st["t"] = clock
        if air <= 0.0:
            st["v0"] = 1.0
            st["u"] = 0.0
            return 0.0, 0.0
        st["v0"] = max(st["v0"], abs(vy))
        u_raw = max(-1.0, min(1.0, vy / st["v0"]))
        # the flight position eases: the vertical speed snaps at take-off (0 -> full) and at touch-down
        # (full -> 0), which pitched the whole body (and swung the tail / ears) in a single frame
        if 0.0 < dt < 0.25:
            st["u"] = st.get("u", u_raw) + (u_raw - st.get("u", u_raw)) * (1.0 - math.exp(-dt / 0.06))
        elif dt != 0.0 or "u" not in st:
            st["u"] = u_raw
        u = st["u"]
        pel, sh = self.canon.get("Pelvis", -1), self.canon.get("LeftUpperArm", -1)
        if min(pel, sh) >= 0:
            piv = T[pel].copy()
            v = T[sh] - piv
            down = 1.0 if float((rot_z(10.0) @ v)[1]) > float((rot_z(-10.0) @ v)[1]) else -1.0
            hd = self.canon.get("Head", -1)
            head_D = D[hd].copy() if hd >= 0 else None
            pitch = -18.0 * u * air                   # + = nose up
            self._rotate(D, T, pel, rot_z(-down * pitch))
            if hd >= 0:                              # the head stays nearer level (eyes on the landing)
                self._rotate(D, T, hd, rot_z(down * 0.6 * pitch) @ head_D @ np.linalg.inv(D[hd]))
        return air, u

    @staticmethod
    def _air_goal(name: str, root: np.ndarray, L: float, u: float) -> np.ndarray:
        """Paw target in the air (internal frame: +x forward, +y down), from the leg's root."""
        front = name in ("FL", "FR")
        if front:
            rise, top, fall = (0.5, 0.62), (0.3, 0.52), (0.42, 0.88)
        else:
            rise, top, fall = (-0.55, 0.78), (0.12, 0.55), (0.02, 0.8)
        a, b = (rise, top) if u < 0 else (fall, top)
        k = abs(u)
        x = b[0] + (a[0] - b[0]) * k
        y = b[1] + (a[1] - b[1]) * k
        return root + np.array([x * L, y * L, 0.0])

    def _ear(self, pose, right: bool) -> np.ndarray:
        perk = float(getattr(pose, "ear", 0.0))
        flop = float(getattr(pose, "ear_flop", 0.0)) + float(getattr(pose, "ear_r" if right else "ear_l", 0.0))
        side = 1.0 if right else -1.0
        sp = self.species
        if sp == "rabbit":                     # long ears fold back along the body when they droop
            droop = 70.0 * max(0.0, -perk)
            return rot_x(side * 10.0 * max(0.0, -perk)) @ rot_z(flop * 0.8 - droop - 6.0 * max(0.0, perk))
        if sp == "dog":                        # floppy ears lift outward when perked, swing with motion
            return rot_x(side * (-25.0 * max(0.0, perk) + 10.0 * max(0.0, -perk))) @ rot_z(flop * 0.8)
        droop = (65.0 if sp == "rabbit" else 32.0) * max(0.0, -perk)
        return rot_x(side * droop) @ rot_z(flop * 0.8 - 8.0 * max(0.0, perk) + 0.35 * droop)

    # ---------------------------------------------------------------- skeletal clips
    @property
    def _C3(self) -> np.ndarray:
        C = getattr(self.asset, "_C", None)
        return C[:3, :3] if C is not None else ASSET_TO_INTERNAL

    def _subtree(self, i: int) -> list:
        st = self._subtree_cache.get(i)
        if st is None:
            st, stack = [], [i]
            while stack:
                k = stack.pop()
                st.append(k)
                stack.extend(self.children.get(k, []))
            self._subtree_cache[i] = st
        return st

    def _rotate(self, D, T, i: int, R: np.ndarray, pivot: Optional[np.ndarray] = None) -> None:
        """Rotate bone ``i`` and everything below it by world rotation ``R`` about its joint."""
        if i < 0:
            return
        piv = T[i].copy() if pivot is None else pivot
        sub = self._subtree_idx(i)
        if len(sub) <= 2:
            for k in sub:
                D[k] = R @ D[k]
                T[k] = piv + R @ (T[k] - piv)
            return
        D[sub] = np.matmul(R, D[sub])
        T[sub] = piv + (T[sub] - piv) @ R.T

    def _subtree_idx(self, i: int) -> np.ndarray:
        c = getattr(self, "_subtree_arr", None)
        if c is None:
            c = self._subtree_arr = {}
        a = c.get(i)
        if a is None:
            a = c[i] = np.array(self._subtree(i), dtype=np.int64)
        return a

    def _clip_pose(self, res, pose, rp: RigPose, head_R: np.ndarray, tg: dict):
        """FK of the clip pose (asset space -> internal), then the procedural layers on top:
        placement / lean / crouch -> IK (feet on the ground) -> look-at -> tail / ears / jaw."""
        D, T = self._clip_fk(res)
        # attached nodes: their clip pose relative to the joint they follow (asset rotation incl. scale)
        att = []
        for c, p in self.attach:
            Pi = np.linalg.inv(D[p])
            att.append((c, p, Pi @ D[c], Pi @ (T[c] - T[p])))
        foot_ground = {}
        for name, bones in LIMB_BONES[self.kind].items():
            e = self.canon.get(bones[2], -1)
            if e >= 0:                                                 # planted in the clip itself
                foot_ground[e] = T[e][1] > self.surface.height_at(float(T[e][0])) - self._bind_foot_height(e) - 1.4
        moving = float(getattr(pose, "moving", 0.0))
        # a motion-capture posture / gesture owns the body: the rig adds no crouch, lean or arm pose
        owns = bool(getattr(res, "posture", False))
        gesture = bool(getattr(res, "action", "")) and float(getattr(pose, "action_w", 0.0)) > 0.05
        # ---- placement: jumps / crouch / body lean from the rig (not its walk bob - the clip has one)
        pel = self.canon.get("Pelvis", self.canon.get("Root", 0))
        dy = (float(tg["Pelvis"][1]) - float(rig_neutral_pelvis(self))) * (1.0 - 0.85 * moving)
        # a captured gesture (crying, laughing ...) brings its own crouch / lean: the rig's fades out
        g_w = float(getattr(pose, "action_w", 0.0)) if gesture and not getattr(res, "upper_only", False) else 0.0
        if owns:
            dy = 0.0
        dy *= 1.0 - g_w
        lift = float(getattr(pose, "lift", 0.0))
        root = self.canon.get("Root", pel)
        top = root if root >= 0 else pel
        dy += self._stance_drop()                  # metadata "stance_drop": a lower, crouched stance
        for k in self._subtree(top):
            T[k][1] += dy
        still = (1.0 - 0.6 * moving) * (0.0 if owns else 1.0 - g_w)
        tilt = float(rp.tilt) * still * 1.3
        if abs(tilt) > 0.05 and pel >= 0:
            self._rotate(D, T, pel, rot_z(tilt))
        # posture from the body-language layer: chest slump / hunch, head drop & reach (neck), shoulders
        ch = self.canon.get("Chest", self.canon.get("Spine", -1))
        slump = float(getattr(pose, "slump", 0.0)) + 0.3 * max(0.0, float(getattr(pose, "shoulders", 0.0)))
        if owns:
            slump = 0.0
        slump *= 1.0 - g_w
        if ch >= 0 and abs(slump) > 0.01:
            # a tall generated model (metadata "pelvis_lift": "auto") has a long, straight torso: it
            # needs a deeper bend for the same slump to read
            self._rotate(D, T, ch, rot_z((20.0 if self._lift_mode() else 14.0) * slump))
            sp = self.canon.get("Spine", -1)
            if self._lift_mode() and sp >= 0 and sp != ch and slump > 0:
                self._rotate(D, T, sp, rot_z(8.0 * slump))
        nk0 = self.canon.get("Neck", -1)
        # living layers on top of the capture: breathing in the chest, the look chain's torso twist
        # (eyes -> head -> neck -> chest) and small head motion (talking nods, head drift). A captured
        # posture keeps most of its own motion, so these are softer there; a sleeping body only breathes.
        base_name = getattr(res, "base", "")
        asleep = base_name in ("sleep", "lie_down")
        soft = 0.4 if owns else 1.0
        if ch >= 0:
            br = float(getattr(pose, "breath", 0.0)) + 0.6 * float(getattr(pose, "puff", 0.0))
            if abs(br) > 0.01:
                self._rotate(D, T, ch, rot_z(-1.3 * br * (1.6 if asleep else 1.0)))
            tw = 0.0 if asleep else float(getattr(pose, "torso_twist", 0.0)) * soft * (1.0 - g_w)
            if abs(tw) > 0.05 and nk0 >= 0:
                self._rotate(D, T, ch, axis_angle(T[nk0] - T[ch], math.radians(tw * (1 if self.kind == "biped" else 0.6))))
        if asleep:
            hdy = hdx = 0.0
        elif owns:                                  # only the face-life part (nods, drift), not the state pose
            hdy = float(getattr(pose, "nod", 0.0))
            hdx = float(getattr(pose, "sway", 0.0))
        else:
            hdy = float(getattr(pose, "head_dy", 0.0))
            hdx = float(getattr(pose, "head_dx", 0.0))
        if nk0 >= 0 and (abs(hdy) > 0.05 or abs(hdx) > 0.05):
            kneck = 2.4 if self.kind == "biped" else 3.2
            self._rotate(D, T, nk0, rot_z(max(-40.0, min(40.0, kneck * hdy + 1.2 * hdx))))
        self._raised = {}
        if not (owns or gesture):
            self._clip_arms(D, T, pose, rp, tg)
        # ---- IK: feet / paws back onto the ground (or kept planted) after the offsets
        # a locomotion clip owns the foot contacts: the procedural hop gait (rabbit) is not "airborne"
        clip_loco = getattr(res, "base", "") == "loco"
        hop_air = float(getattr(pose, "hop", 0.0)) > 0.3 and not clip_loco
        lift_air = lift - (float(getattr(pose, "gait_lift", 0.0)) if clip_loco else 0.0)
        airborne = lift_air > 2.0 or hop_air or self.surface.kind == "air"
        # ground odometer: the rig's leg phase advances 2*pi per stride of travel
        lp = float(getattr(pose, "leg_phase", 0.0))
        if self._last_leg_phase is not None:
            dphi = (lp - self._last_leg_phase + math.pi) % (2 * math.pi) - math.pi
            self._ground_odo += dphi / (2 * math.pi) * float(getattr(pose, "stride_u", 0.0))
        self._last_leg_phase = lp
        locking = moving > 0.5 and not airborne and float(getattr(pose, "stride_u", 0.0)) > 0
        for name, bones in LIMB_BONES[self.kind].items():
            up, lo, end = (self.canon.get(b, -1) for b in bones)
            if min(up, lo, end) < 0 or (self.kind == "biped" and name.startswith("arm")) or owns:
                continue
            goal = T[end].copy()
            goal[1] -= self._stance_drop()                   # the crouch lowers the body, not the feet
            rest_y = self.surface.height_at(float(goal[0])) - self._bind_foot_height(end)   # standing height
            planted = not airborne and (foot_ground.get(end) or goal[1] > rest_y)
            lifted = self._raised.get(end, 0.0)
            if lifted > 0.999:
                continue                                             # held up by the pose: keep it there
            if planted:
                goal[1] = rest_y + (goal[1] - rest_y) * lifted       # planted feet stay on the ground
            # foot lock: a planted foot keeps its spot on the ground while the body moves over it
            # (hides the stride mismatch between the clip and the desktop movement)
            if planted and locking:
                reach = 0.9 * (float(np.linalg.norm(self.B[lo] - self.B[up])) +
                                float(np.linalg.norm(self.B[end] - self.B[lo])))
                anchor = self._foot_lock.get(end)
                if anchor is None:
                    self._foot_lock[end] = goal[0] + self._ground_odo
                else:
                    gx = anchor - self._ground_odo
                    if abs(gx - goal[0]) > reach:
                        # stretched too far: the foot slides along at the reach limit (no pop)
                        gx = goal[0] + math.copysign(reach, gx - goal[0])
                        self._foot_lock[end] = gx + self._ground_odo
                    goal[0] = gx
            else:
                self._foot_lock.pop(end, None)
            self._ik_limb(D, T, up, lo, end, goal)
        # ---- look-at / viewer turn: the head frame without the body tilt, split neck 30 % / head 70 %
        hd = self.canon.get("Head", -1)
        nk = self.canon.get("Neck", -1)
        F = rot_z(-rp.tilt) @ head_R
        if owns:
            # sitting / lying / mid-gesture: the head keeps the captured motion, only a soft look-at
            F = blend_rot(np.eye(3), F, 0.35 if getattr(res, "base", "") not in ("sleep", "lie_down", "wake") else 0.0)
        if hd >= 0:
            if nk >= 0:
                Fn = blend_rot(np.eye(3), F, 0.3)
                self._rotate(D, T, nk, Fn)
                F = Fn.T @ F
            self._rotate(D, T, hd, F)
        # ---- secondary / emotion on tail and ears, jaw
        t0 = self.canon.get("Tail0", -1)
        if t0 >= 0:
            # Pose.tail: negative = raised (it already contains tail_pose, wag and spring motion);
            # the asset's tail points backward, where a positive z-rotation lifts it
            gain = float((getattr(self.asset, "meta", None) or {}).get("tail_gain", 1.0))
            ang = max(-70.0, min(70.0, -0.8 * gain * float(getattr(pose, "tail", 0.0))))
            self._rotate(D, T, t0, rot_z(ang))
            t1 = self.canon.get("Tail1", -1)
            if t1 >= 0:
                self._rotate(D, T, t1, rot_z(ang * 0.4))
        for ear, right in (("LeftEar", False), ("RightEar", True)):
            e = self.canon.get(ear, -1)
            if e >= 0:
                Rl = self._ear(pose, right)
                Ph = D[self.parent[e]] if self.parent[e] >= 0 else np.eye(3)
                Ph = Ph / max(1e-9, np.cbrt(abs(np.linalg.det(Ph))))
                self._rotate(D, T, e, Ph @ Rl @ Ph.T)
        jw = self.canon.get("Jaw", -1)
        if jw >= 0:
            jaw = max(0.0, min(1.0, float(getattr(pose, "jaw", 0.0))))
            if jaw > 0.01:
                Ph = D[jw] / max(1e-9, np.cbrt(abs(np.linalg.det(D[jw]))))
                self._rotate(D, T, jw, Ph @ rot_z(22.0 * jaw) @ Ph.T)
        self._tail_floor(D, T)
        for c, p, Rrel, off in att:
            self._follow(D, T, c, p, Rrel, off)
        return D, T

    def _tail_floor(self, D, T) -> None:
        """A tail never goes through the floor: a tail joint that would end up below the ground (a
        sitting animal pitches its rump down, a long tail resting on the ground in the bind pose)
        bends up at the joint before it, so the tail lies along the floor instead of lifting the
        whole body (the renderer's ground clamp)."""
        if not self.tail_idx:
            return
        ground = float(getattr(getattr(self, "surface", None), "height_at", lambda _x: 0.0)(0.0))
        meta = getattr(self.asset, "meta", None) or {}
        r = float(meta.get("tail_floor", 0.0) or 0.0)
        frac = float(meta.get("tail_floor_frac", 0.0) or 0.0)
        if frac > 0.0:                              # a fraction of the model's height (ground to the
            r = max(r, frac * float(self.B[:, 1].max() - self.B[:, 1].min()))   # highest joint, an ear tip)
        if r <= 0.0:
            return
        chain = list(self.tail_idx)
        tip = [k for k in self.children.get(chain[-1], [])]
        if tip:
            chain.append(tip[0])
        for a, b in zip(chain[:-1], chain[1:]):
            lim = ground - r                         # internal +y is down: the joint's lowest allowed y
            if T[b][1] <= lim:
                continue
            v = T[b] - T[a]
            L = float(np.linalg.norm(v))
            if L < 1e-6:
                continue
            # rotate about the side axis through a so that b rises to the limit (or as far as it can)
            h = lim - T[a][1]                       # wanted vertical offset (negative = up)
            h = max(-L, min(L, h))
            hv = np.array([v[0], 0.0, v[2]])
            hn = float(np.linalg.norm(hv))
            # the tail is laid down behind the animal (away from the chest): choosing "along the
            # segment" flipped it through the body to the front whenever a near-vertical segment
            # leaned a little forward (a sitting fox's tail jumped back and forth)
            pel_, ch_ = self.canon.get("Pelvis", -1), self.canon.get("Chest", self.canon.get("Spine", -1))
            bk = (T[pel_] - T[ch_]) if pel_ >= 0 and ch_ >= 0 else np.array([-1.0, 0.0, 0.0])
            bk = np.array([bk[0], 0.0, bk[2]])
            bn = float(np.linalg.norm(bk))
            bk = bk / bn if bn > 1e-9 else np.array([-1.0, 0.0, 0.0])
            # ... and the choice is a blend, not a switch: the direction turns from "behind" to "along
            # the segment" gradually as the segment leans back (a hard switch swung the tail ~90 deg
            # in one frame whenever the lean crossed the vertical)
            if hn > 1e-9:
                hd = hv / hn
                w = max(0.0, min(1.0, float(np.dot(hd, bk)) / 0.5))
                w = w * w * (3.0 - 2.0 * w)
                mix = hd * w + bk * (1.0 - w)
                mn = float(np.linalg.norm(mix))
                hv = mix / mn if mn > 1e-9 else bk
            else:
                hv = bk
            want = hv * math.sqrt(max(L * L - h * h, 0.0)) + np.array([0.0, h, 0.0])
            Rm = rotation_between(v, want)
            self._rotate(D, T, a, Rm)

    def _fk_setup(self) -> None:
        """Per-skeleton constants for the vectorised clip FK (depth levels, rest TRS, bases, IBMs)."""
        J = self.asset.joints
        n = self.n
        depth = [0] * n
        for i in self.order:
            p = self.parent[i]
            depth[i] = depth[p] + 1 if p >= 0 else 0
        self._levels = []
        for d in range(max(depth) + 1 if n else 0):
            idx = np.array([i for i in self.order if depth[i] == d], dtype=np.int64)
            self._levels.append((idx, np.array([self.parent[i] for i in idx], dtype=np.int64)))
        self._rest_q = np.array([j.rest_r if j.rest_r is not None else (0, 0, 0, 1.0) for j in J], dtype=float)
        self._rest_t = np.array([j.rest_t if j.rest_t is not None else (0, 0, 0) for j in J], dtype=float)
        self._rest_s = np.array([j.rest_s if j.rest_s is not None else (1, 1, 1) for j in J], dtype=float)
        self._base = np.array([j.base if j.base is not None else np.eye(4) for j in J], dtype=float)
        self._ibm = np.array([j.ibm if j.ibm is not None else np.eye(4) for j in J], dtype=float)
        C = self.asset._C
        self._fk_C = C
        self._fk_post = self._ibm @ np.linalg.inv(C)            # (n, 4, 4): ibm @ C^-1

    def _clip_fk(self, res):
        """FK of the clip pose for all joints at once (level by level); returns world delta
        rotations D (n, 3, 3) and positions T (n, 3) in internal units."""
        if getattr(self, "_levels", None) is None:
            self._fk_setup()
        n = self.n
        q = self._rest_q.copy()
        t = self._rest_t.copy()
        sc = self._rest_s.copy()
        for i, v in res.rot.items():
            q[i] = v
        for i, v in res.trans.items():
            t[i] = v
        for i, v in res.scale.items():
            sc[i] = v
        x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        L = np.zeros((n, 4, 4))
        L[:, 0, 0] = 1 - 2 * (y * y + z * z)
        L[:, 0, 1] = 2 * (x * y - z * w)
        L[:, 0, 2] = 2 * (x * z + y * w)
        L[:, 1, 0] = 2 * (x * y + z * w)
        L[:, 1, 1] = 1 - 2 * (x * x + z * z)
        L[:, 1, 2] = 2 * (y * z - x * w)
        L[:, 2, 0] = 2 * (x * z - y * w)
        L[:, 2, 1] = 2 * (y * z + x * w)
        L[:, 2, 2] = 1 - 2 * (x * x + y * y)
        L[:, :3, :3] *= sc[:, None, :]
        L[:, :3, 3] = t
        L[:, 3, 3] = 1.0
        W = np.empty((n, 4, 4))
        for d, (idx, par) in enumerate(self._levels):
            if d == 0:
                W[idx] = self._base[idx] @ L[idx]
            else:
                W[idx] = W[par] @ L[idx]
        CW = self._fk_C @ W
        M = CW @ self._fk_post
        return M[:, :3, :3].copy(), CW[:, :3, 3].copy()

    def _follow(self, D, T, c: int, p: int, Rrel: np.ndarray, off: np.ndarray) -> None:
        """Move node ``c`` and its subtree rigidly so it keeps the relation (Rrel, off) to ``p``."""
        Dn = D[p] @ Rrel
        Tn = T[p] + D[p] @ off
        if np.allclose(Dn, D[c]) and np.allclose(Tn, T[c]):
            return
        X = Dn @ np.linalg.inv(D[c])
        piv = T[c].copy()
        sub = self._subtree_idx(c)
        T[sub] = Tn + (T[sub] - piv) @ X.T
        D[sub] = np.einsum("ij,njk->nik", X, D[sub])

    def _eyes(self, D, T, pose) -> None:
        """Eye bones (LeftEye / RightEye) follow the look target: pupils first (the attention chain
        then turns the head and the torso), with convergence on near targets."""
        lx = max(-1.0, min(1.0, float(getattr(pose, "look_x", 0.0))))
        ly = max(-1.0, min(1.0, float(getattr(pose, "look_y", 0.0))))
        conv = max(0.0, min(0.8, float(getattr(pose, "eye_conv", 0.0))))
        hd = self.canon.get("Head", -1)
        if hd < 0:
            return
        Hd = D[hd] / max(1e-9, np.cbrt(abs(np.linalg.det(D[hd]))))
        for name, side in (("LeftEye", -1.0), ("RightEye", 1.0)):
            e = self.canon.get(name, -1)
            if e < 0:
                continue
            yaw = 24.0 * lx - side * 14.0 * conv
            pitch = 18.0 * ly
            R = Hd @ (axis_angle((0.0, 1.0, 0.0), math.radians(-yaw)) @ rot_z(-pitch)) @ Hd.T
            self._rotate(D, T, e, R)

    def _expressive_arm(self, name: str, pose, root, side: float, L: float, mid_z: float, pump: float):
        """(weight, goal) of a gesture for one arm of a generated biped (internal frame: +x forward,
        +y down, z sideways). The rig's arm_R is the gesturing hand (the one nearer the camera)."""
        g = lambda f: max(0.0, min(1.0, float(getattr(pose, f, 0.0))))   # noqa: E731
        ph = float(getattr(pose, "mark_phase", 0.0))
        cands = []
        clap = g("clap")
        if clap > 0:                           # both hands meet in front of the chest
            gap = (0.05 + 0.08 * abs(math.sin(ph * 9))) * L
            cands.append((clap, np.array([root[0] + 0.5 * L, root[1] + 0.42 * L, mid_z + side * gap])))
        if name == "arm_R":
            point = g("point")
            if point > 0:                      # straight out, forward and to the side
                cands.append((point, root + np.array([0.32 * L, -0.12 * L, side * 0.9 * L])))
            thumb = g("thumb")
            if thumb > 0:                      # thumbs up at shoulder height, out in front
                cands.append((thumb, root + np.array([0.35 * L, 0.18 * L, side * 0.55 * L])))
            beckon = g("beckon")
            if beckon > 0:                     # palm up in front, the forearm curling in and out
                curl = 0.12 * L * math.sin(math.radians(float(getattr(pose, "paw_wave", 0.0))) * 3.0)
                cands.append((beckon, root + np.array([0.42 * L - curl, 0.18 * L - curl, side * 0.5 * L])))
            if pump > 0:                       # fist pump: elbow bent, fist punching up beside the head
                pw = max(0.0, min(1.0, -float(getattr(pose, "paw_wave", 0.0)) / 60.0))
                cands.append((pump, root + np.array([0.25 * L, (0.1 - 0.55 * pw) * L, side * 0.38 * L])))
        elif pump > 0:                         # the other fist braced at the hip
            cands.append((pump, root + np.array([0.12 * L, 0.78 * L, side * 0.22 * L])))
        if not cands:
            return 0.0, None
        w, goal = max(cands, key=lambda c: c[0])
        return w, goal

    ARM_FIELDS = ("fists", "akimbo", "hug", "arms_out", "arms_up", "cover", "cross", "clap", "point", "chin",
                  "thumb", "belly", "paw", "paw2", "write", "read")

    def _clip_gestures(self, D, T, pose) -> set:
        """Hand gestures without a captured clip (clap, point, thumbs up, beckon, fist pump) on a
        motion-capture biped: the same goals as the rig path, reached from the clip's own shoulders.
        Returns the fields it handled."""
        g = lambda f: max(0.0, min(1.0, float(getattr(pose, f, 0.0))))   # noqa: E731
        pump = g("paw") if g("fists") > 0.5 else 0.0
        if max(g("clap"), g("point"), g("thumb"), g("beckon"), pump) <= 0.02:
            return set()
        sh = [T[self.canon[b]][2] for b in ("LeftUpperArm", "RightUpperArm") if b in self.canon]
        mid_z = sum(sh) / len(sh) if sh else 0.0
        for name in ("arm_L", "arm_R"):
            up, lo, end = (self.canon.get(b, -1) for b in LIMB_BONES["biped"][name])
            if min(up, lo, end) < 0:
                continue
            L = float(np.linalg.norm(self.B[lo] - self.B[up]) + np.linalg.norm(self.B[end] - self.B[lo]))
            side = 1.0 if float(self.B[up][2]) >= 0.0 else -1.0
            w, goal = self._expressive_arm(name, pose, T[up].copy(), side, L, mid_z, pump)
            if goal is not None and w > 0.0:
                self._ik_limb(D, T, up, lo, end, T[end] + (goal - T[end]) * min(1.0, w * 1.2))
        return {"clap", "point", "thumb", "beckon"}

    def _clip_arms(self, D, T, pose, rp: RigPose, tg: dict) -> None:
        """Expressive arm / front-paw poses of the procedural rig (hands on hips, hugging, fists,
        arms out, covering the face, raised paws ...) blended onto the clip's arms with IK."""
        fields = self.ARM_FIELDS
        gest = self.kind == "biped" and getattr(self.animator, "mocap", False)
        if gest:
            fields = tuple(f for f in fields if f not in ("clap", "point", "thumb", "beckon"))
            try:
                self._clip_arm_fields(D, T, pose, rp, tg, fields)
            finally:
                self._clip_gestures(D, T, pose)       # the gesture wins over the generic arm pose
            return
        self._clip_arm_fields(D, T, pose, rp, tg, fields)

    def _clip_arm_fields(self, D, T, pose, rp: RigPose, tg: dict, fields) -> None:
        strength = max(float(getattr(pose, f, 0.0)) for f in fields)
        if strength <= 0.02:
            return
        raised = self._raised = getattr(self, "_raised", {})
        w = max(0.0, min(1.0, strength * 1.4))
        limbs = ("arm_L", "arm_R") if self.kind == "biped" else ("FL", "FR")
        for name in limbs:
            bones = LIMB_BONES[self.kind].get(name)
            if not bones or bones[0] not in tg or bones[2] not in tg:
                continue
            up, lo, end = (self.canon.get(b, -1) for b in bones)
            if min(up, lo, end) < 0:
                continue
            l_asset = float(np.linalg.norm(self.B[lo] - self.B[up]) + np.linalg.norm(self.B[end] - self.B[lo]))
            l_rig = float(np.linalg.norm(tg[bones[1]] - tg[bones[0]]) + np.linalg.norm(tg[bones[2]] - tg[bones[1]]))
            k = l_asset / max(1e-6, l_rig)
            goal_rig = T[up] + (tg[bones[2]] - tg[bones[0]]) * k
            goal = T[end] + (goal_rig - T[end]) * w
            self._ik_limb(D, T, up, lo, end, goal)
            if self.kind != "biped":
                # a paw lifted by the pose (wave, give paw, cover the face ...) must not be pulled
                # back onto the ground by the foot planting below - that made raised paws flicker
                ground = self.surface.height_at(float(goal[0])) - self._bind_foot_height(end)
                h = ground - float(goal[1])
                raised[end] = w * max(0.0, min(1.0, (h - 0.5) / 4.0))
        fists = float(getattr(pose, "fists", 0.0))
        if fists > 0.05 and self.kind == "biped":
            for hb in ("LeftHand", "RightHand"):
                h = self.canon.get(hb, -1)
                if h >= 0:
                    s_ = 1.0 - 0.2 * min(1.0, fists)
                    piv = T[h].copy()
                    for kk in self._subtree(h):
                        D[kk] = D[kk] * s_
                        T[kk] = piv + (T[kk] - piv) * s_

    def _bind_foot_height(self, end: int) -> float:
        """How far above the ground the foot joint sits in the bind pose (internal units, >= 0)."""
        cache = getattr(self, "_foot_h", None)
        if cache is None:
            cache = self._foot_h = {}
        if end not in cache:
            cache[end] = max(0.0, -float(self.B[end][1]))
        return cache[end]

    def _posture(self, D, T, pose) -> None:
        """Body-language posture on a clip-less biped (the same layer the clip path adds): the chest
        slumps / hunches and the head drops or reaches - a short-spined, big-headed model barely
        changes its silhouette from the spine targets alone."""
        ch = self.canon.get("Chest", self.canon.get("Spine", -1))
        slump = float(getattr(pose, "slump", 0.0)) + 0.3 * max(0.0, float(getattr(pose, "shoulders", 0.0)))
        if ch >= 0 and abs(slump) > 0.01:
            # a tall generated model (metadata "pelvis_lift": "auto") has a long, straight torso: it
            # needs a deeper bend for the same slump to read
            self._rotate(D, T, ch, rot_z((20.0 if self._lift_mode() else 14.0) * slump))
            sp = self.canon.get("Spine", -1)
            if self._lift_mode() and sp >= 0 and sp != ch and slump > 0:
                self._rotate(D, T, sp, rot_z(8.0 * slump))
        nk = self.canon.get("Neck", -1)
        hdy = float(getattr(pose, "head_dy", 0.0))
        hdx = float(getattr(pose, "head_dx", 0.0))
        if nk >= 0 and (abs(hdy) > 0.05 or abs(hdx) > 0.05):
            self._rotate(D, T, nk, rot_z(max(-40.0, min(40.0, 2.4 * hdy + 1.2 * hdx))))

    def _leg_goals(self, T, tg, pose) -> dict:
        """Foot / paw goals for an asset with its own proportions (metadata "pelvis_lift": "auto"):
        the rig's foot offset from its hip, carried over to the asset's hip (stride scaled to the
        asset's leg), at the asset's own ankle height on the ground. When a leg can't reach its
        planted goal the whole body is lowered (T is shifted in place) so no foot hovers."""
        ank = float(getattr(getattr(self.rig, "sk", None), "ankle", 0.0) or 0.0)
        goals, need = {}, 0.0
        # ground odometer (as on the clip path): the rig's leg phase advances 2*pi per stride of travel
        lp = float(getattr(pose, "leg_phase", 0.0))
        if self._last_leg_phase is not None:
            dphi = (lp - self._last_leg_phase + math.pi) % (2 * math.pi) - math.pi
            self._ground_odo += dphi / (2 * math.pi) * float(getattr(pose, "stride_u", 0.0))
        self._last_leg_phase = lp
        odo = self._ground_odo
        for name, bones in LIMB_BONES[self.kind].items():
            if self.kind == "biped" and name.startswith("arm"):
                continue
            up, lo, end = (self.canon.get(b, -1) for b in bones)
            if min(up, lo, end) < 0 or bones[0] not in tg:
                continue
            L = float(np.linalg.norm(self.B[lo] - self.B[up]) + np.linalg.norm(self.B[end] - self.B[lo]))
            root = T[up]
            rel = tg[bones[2]] - tg[bones[0]]
            k = self._limb_scale(name, L)
            gy = tg[bones[2]][1] + ank - self._bind_foot_height(end)
            # planted: the asset's foot moves exactly like the rig's planted foot (fixed on the ground,
            # whatever the asset's spine does); swinging: the rig's offset from the hip, scaled to the
            # asset's leg, with the lift-off offset fading out
            free = root[0] + rel[0] * k
            planted = tg[bones[2]][1] >= -ank - 0.3
            st = self._leg_state.setdefault(name, {"planted": False, "last": free, "off": 0.0, "g0": free,
                                                   "t0": float(tg[bones[2]][0])})
            if k >= 0.999:
                # the asset's leg is at least as long as the rig's: the rig's own foot (already
                # planted without sliding by the foot-phase system) carries over 1:1, shifted by the
                # asset hip's offset from the rig hip. The offset is frozen while the foot is down
                # (so a planted paw stays put even when the asset's spine flexes differently) and
                # follows the hip while it swings. (A separate ground lock used before drifted from
                # the rig's feet a little every step, until the feet trailed far behind and the body
                # sank: the stretched-out "seal" walk.)
                offx = float(root[0] - tg[bones[0]][0])
                if not (planted and st["planted"]) or "offx" not in st:
                    st["offx"] = offx
                gx = float(tg[bones[2]][0]) + st["offx"]
            elif planted:
                if not st["planted"]:
                    st["g0"], st["t0"] = st["last"], odo
                gx = st["g0"] - (odo - st["t0"])                 # fixed on the ground as the body travels
            else:
                st["off"] = (st["last"] - free) if st["planted"] else st["off"] * 0.75
                gx = free + st["off"]
            st["planted"], st["last"] = planted, gx
            g = np.array([gx, gy, root[2] + rel[2]])
            goals[name] = g
            if planted:
                h2 = float((g[0] - root[0]) ** 2 + (g[2] - root[2]) ** 2)
                reach = 0.97 * L
                if h2 < reach * reach:
                    need = max(need, float(g[1] - root[1]) - math.sqrt(reach * reach - h2))
        if need > 0.0:
            T[:, 1] += min(need, 0.35 * max(1.0, abs(float(self.B[self.canon.get("Pelvis", 0)][1]))))
        return goals

    def _lift_mode(self) -> bool:
        return str((getattr(self.asset, "meta", None) or {}).get("pelvis_lift", "")).lower() == "auto"

    def _limb_scale(self, name: str, asset_len: float) -> float:
        """Asset limb length / rig limb length (clamped), for carrying the rig's limb offsets over."""
        cache = getattr(self, "_limb_k", None)
        if cache is None:
            cache = self._limb_k = {}
        if name not in cache:
            k = 1.0
            try:
                lim = self.rig.sk.limb(name)
                k = asset_len / max(1e-6, lim.upper + lim.lower)
            except (AttributeError, KeyError):
                pass
            cache[name] = max(0.5, min(1.0 if not name.startswith("arm") else 1.4, k))
        return cache[name]

    def _pelvis_lift(self) -> float:
        """How much higher (+) or lower (-) the asset's hips are than the rig's neutral hips
        (internal units)."""
        v = getattr(self, "_lift_cache", None)
        if v is None:
            v = 0.0
            if str((getattr(self.asset, "meta", None) or {}).get("pelvis_lift", "")).lower() == "auto":
                pel = self.canon.get("Pelvis", -1)
                if pel >= 0 and getattr(self, "rig", None) is not None:
                    v = float(rig_neutral_pelvis(self)) - float(self.B[pel][1])
                    # a little knee bend (the leg IK never locks) and a hair of clearance under a low
                    # body; metadata "pelvis_lift_k" overrides it (1.0: exactly the asset's hip height)
                    try:
                        k = float((self.asset.meta or {}).get("pelvis_lift_k", 0.94))
                    except (TypeError, ValueError):
                        k = 0.94
                    v *= max(0.7, min(1.05, k))
                    self._lift_cache = v
                return v
            self._lift_cache = v
        return v

    def _stance_drop(self) -> float:
        """Body lowered by a fraction of the hip height (internal +y is down); the leg IK then bends
        the legs so the feet stay on the ground - a crouched, compact stance (e.g. a rabbit)."""
        v = getattr(self, "_drop_cache", None)
        if v is None:
            v = 0.0
            f = float((getattr(self.asset, "meta", None) or {}).get("stance_drop", 0.0) or 0.0)
            up, ft = self.canon.get("LeftUpperLeg", -1), self.canon.get("LeftFoot", -1)
            if f > 0 and up >= 0 and ft >= 0:
                v = f * abs(float(self.B[ft][1] - self.B[up][1]))
            self._drop_cache = v
        return v

    def _ik_limb(self, D, T, up: int, lo: int, end: int, goal: np.ndarray) -> None:
        """Two-bone IK that keeps the animated twist: rotate the upper bone, then the lower bone,
        with the current knee / elbow as the pole."""
        root = T[up].copy()
        l1 = float(np.linalg.norm(self.B[lo] - self.B[up]))
        l2 = float(np.linalg.norm(self.B[end] - self.B[lo]))
        if l1 < 1e-6 or l2 < 1e-6 or float(np.linalg.norm(goal - T[end])) < 1e-4:
            return
        pole = T[lo] - (root + T[end]) * 0.5
        jnt, eff, _ = two_bone_ik_3d(tuple(root), tuple(goal), l1, l2, tuple(pole), 175.0)
        jnt, eff = np.array(jnt), np.array(eff)
        R1 = rotation_between(T[lo] - root, jnt - root)
        self._rotate(D, T, up, R1)
        R2 = rotation_between(T[end] - T[lo], eff - T[lo])
        self._rotate(D, T, lo, R2)

    # ---------------------------------------------------------------- clips
    def set_clip(self, clip, t: float, weight: float, mask: str = "full", rest: Optional[dict] = None) -> None:
        """Blend a (retargeted) animation clip on top of the rig pose. ``rest`` = per-joint
        (local rotation quat, local translation) of the skeleton the clip was authored for."""
        self.clip = None if clip is None or weight <= 0 else (clip, t, weight, mask, rest or {})

    def _apply_clip(self, D, T):
        clip, t, w, mask, rest = self.clip
        A = ASSET_TO_INTERNAL
        rots = {}
        for (ji, path, times, values, interp) in clip.channels:
            if path != "rotation":
                continue

            class _C:
                pass
            ch = _C()
            ch.times, ch.values, ch.interpolation, ch.path = times, values, interp, path
            rots[ji] = sample_channel(ch, t % max(1e-6, clip.duration or 1.0))
        if not rots:
            return D, T
        # world rotations of the clip and of its rest pose (asset space), then world-space deltas
        Wc, Wr = {}, {}
        for i in self.order:
            p = self.parent[i]
            r_loc = rest.get(i, (np.array([0, 0, 0, 1.0]), None))[0]
            c_loc = rots.get(i, r_loc)
            Rr = quat_to_mat(r_loc)
            Rc = quat_to_mat(c_loc)
            Wr[i] = Rr if p < 0 else Wr[p] @ Rr
            Wc[i] = Rc if p < 0 else Wc[p] @ Rc
        legs = {self.canon.get(b) for b in ("LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "RightUpperLeg",
                                            "RightLowerLeg", "RightFoot")}
        D2, T2 = D.copy(), T.copy()
        for i in self.order:
            p = self.parent[i]
            if i in rots and not (mask == "upper" and i in legs):
                delta = A @ (Wc[i] @ Wr[i].T) @ A.T
                D2[i] = blend_rot(D[i], delta, w)
            if p >= 0 and (i in rots or p in rots):
                T2[i] = T2[p] + D2[p] @ (self.B[i] - self.B[p])
        return D2, T2


# ------------------------------------------------------------------ skinning
def rig_neutral_pelvis(driver) -> float:
    """Pelvis height of the rig's neutral standing pose (cached on the driver)."""
    v = getattr(driver, "_rig_neutral_y", None)
    if v is None:
        v = 0.0
        rig = getattr(driver, "rig", None)
        if rig is not None:
            from .animation import Pose
            v = float(rig_targets(rig.solve(Pose()), driver.kind)["Pelvis"][1])
        driver._rig_neutral_y = v
    return v


def skin(mesh, M: np.ndarray, morph_weights: Optional[dict] = None, tangents: bool = False):
    """Linear blend skinning (+ morph targets) on the CPU. Returns posed (positions, normals)
    or (positions, normals, tangents) with ``tangents=True``. The GPU path does the same in the
    vertex shader (``render3d_gl.GLSkinRaster``)."""
    pos = mesh.positions
    nor = mesh.normals
    if morph_weights:
        pos = pos.copy()
        nor = nor.copy()
        for name, w in morph_weights.items():
            if abs(w) < 1e-4 or name not in mesh.morphs:
                continue
            pos += mesh.morphs[name] * w
            if name in mesh.morph_normals:
                nor += mesh.morph_normals[name] * w
    tan = getattr(mesh, "tangents", None) if tangents else None
    if len(M) == 0:
        return (pos, nor, tan) if tangents else (pos, nor)
    W = mesh.weights
    J = mesh.joints
    # blended 3x4 matrix per vertex
    Mv = np.einsum("nk,nkij->nij", W, M[J][:, :, :3, :])
    p = np.einsum("nij,nj->ni", Mv[:, :, :3], pos) + Mv[:, :, 3]
    n = np.einsum("nij,nj->ni", Mv[:, :, :3], nor)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    if not tangents:
        return p, n
    if tan is None:
        return p, n, np.zeros((len(p), 4))
    t = np.einsum("nij,nj->ni", Mv[:, :, :3], tan[:, :3])
    return p, n, np.concatenate([t, tan[:, 3:4]], axis=1)
