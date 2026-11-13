"""Motion-capture clips for the people (and the robot): real human movement on our own skeletons.

``assets/mocap/library.npz`` holds clips cut from the CMU Graphics Lab Motion Capture Database
(see ``assets/mocap/LICENSE.md``), prepared by ``tools/mocap/build_library.py``: trimmed, turned to
face forward, made to loop seamlessly, and stored skeleton-independently as

* ``<clip>.q``    (F, B, 4) world rotation of each body segment relative to the capture skeleton's
                  rest pose (x, y, z, w; asset axes: +Y up, +Z forward, +X = the actor's left),
* ``<clip>.dir``  (B, 3)    that skeleton's rest direction of each segment,
* ``<clip>.hips`` (F, 3)    hip position over the floor, in leg lengths.

``retarget(asset)`` turns them into ordinary :class:`~.character_asset.Clip` objects for one model
(local joint rotations + a pelvis translation track) by *bone direction*: every segment of the
model is first turned from its own bind direction onto the capture skeleton's rest direction, then
follows the captured rotation. That works for any proportions (a chibi head, short legs, a long
saree) and keeps the captured twist of arms and legs. The clips then play through the normal
``ClipAnimator`` (crossfades, phase-synced locomotion, foot IK, look-at).
"""

from __future__ import annotations

import json
import logging
import math
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("focusbuddy.companion")

# segment order in the library
BONES = ("Pelvis", "Spine", "Chest", "Neck", "Head",
         "LeftUpperArm", "LeftLowerArm", "LeftHand", "RightUpperArm", "RightLowerArm", "RightHand",
         "LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "RightUpperLeg", "RightLowerLeg", "RightFoot")
# the model joint whose position gives each segment's direction (canonical name, or a tip node name)
TARGET_CHILD = {"Pelvis": ("Spine",), "Spine": ("Chest",), "Chest": ("Neck",), "Neck": ("Head",),
                "Head": ("HeadTop",),
                "LeftUpperArm": ("LeftLowerArm",), "LeftLowerArm": ("LeftHand",), "LeftHand": ("LeftHandTip",),
                "RightUpperArm": ("RightLowerArm",), "RightLowerArm": ("RightHand",),
                "RightHand": ("RightHandTip",),
                "LeftUpperLeg": ("LeftLowerLeg",), "LeftLowerLeg": ("LeftFoot",), "LeftFoot": ("LeftFootTip",),
                "RightUpperLeg": ("RightLowerLeg",), "RightLowerLeg": ("RightFoot",),
                "RightFoot": ("RightFootTip",)}

try:
    from ..config import ASSETS_DIR as _ASSETS          # the installed app's resource folder too
    LIBRARY_DIR = Path(_ASSETS) / "mocap"
except Exception:  # noqa: BLE001
    LIBRARY_DIR = Path(__file__).resolve().parents[2] / "assets" / "mocap"


@lru_cache(maxsize=2)
def load_library(directory: Optional[str] = None) -> Optional[dict]:
    """{clip name: {"q", "dir", "hips", "fps", "loop", ...}} or None when the library is missing."""
    d = Path(directory) if directory else LIBRARY_DIR
    f = d / "library.npz"
    m = d / "library.json"
    if not f.is_file() or not m.is_file():
        return None
    try:
        meta = json.loads(m.read_text(encoding="utf-8"))
        out = {}
        with np.load(f) as z:
            for name, info in (meta.get("clips") or {}).items():
                if f"{name}.q" not in z:
                    continue
                out[name] = dict(info, q=z[f"{name}.q"].astype(float), dir=z[f"{name}.dir"].astype(float),
                                 hips=z[f"{name}.hips"].astype(float))
        return out
    except Exception:  # noqa: BLE001 - motion capture is optional, the procedural rig always works
        log.exception("mocap library could not be read")
        return None


# ------------------------------------------------------------------ math (vectorised)
def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a) or 1.0)
    b = b / (np.linalg.norm(b) or 1.0)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        p = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(p) < 1e-6:
            p = np.cross(a, [0.0, 1.0, 0.0])
        p /= np.linalg.norm(p)
        return 2.0 * np.outer(p, p) - np.eye(3)
    k = v / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    ang = math.atan2(s, c)
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)


def quats_to_mats(q: np.ndarray) -> np.ndarray:
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    m = np.empty(q.shape[:-1] + (3, 3))
    m[..., 0, 0] = 1 - 2 * (y * y + z * z)
    m[..., 0, 1] = 2 * (x * y - z * w)
    m[..., 0, 2] = 2 * (x * z + y * w)
    m[..., 1, 0] = 2 * (x * y + z * w)
    m[..., 1, 1] = 1 - 2 * (x * x + z * z)
    m[..., 1, 2] = 2 * (y * z - x * w)
    m[..., 2, 0] = 2 * (x * z - y * w)
    m[..., 2, 1] = 2 * (y * z + x * w)
    m[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return m


def mats_to_quats(m: np.ndarray) -> np.ndarray:
    """(…, 3, 3) rotation matrices -> (…, 4) quaternions (x, y, z, w), w >= 0 hemisphere-free."""
    m = np.asarray(m, dtype=float)
    tr = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    cand = np.stack([tr, m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]], axis=-1)
    k = np.argmax(cand, axis=-1)
    q = np.empty(m.shape[:-2] + (4,))
    # w largest
    s = np.sqrt(np.maximum(tr + 1.0, 1e-12)) * 2
    qw = np.stack([(m[..., 2, 1] - m[..., 1, 2]) / s, (m[..., 0, 2] - m[..., 2, 0]) / s,
                   (m[..., 1, 0] - m[..., 0, 1]) / s, 0.25 * s], axis=-1)
    s = np.sqrt(np.maximum(1.0 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2], 1e-12)) * 2
    qx = np.stack([0.25 * s, (m[..., 0, 1] + m[..., 1, 0]) / s, (m[..., 0, 2] + m[..., 2, 0]) / s,
                   (m[..., 2, 1] - m[..., 1, 2]) / s], axis=-1)
    s = np.sqrt(np.maximum(1.0 + m[..., 1, 1] - m[..., 0, 0] - m[..., 2, 2], 1e-12)) * 2
    qy = np.stack([(m[..., 0, 1] + m[..., 1, 0]) / s, 0.25 * s, (m[..., 1, 2] + m[..., 2, 1]) / s,
                   (m[..., 0, 2] - m[..., 2, 0]) / s], axis=-1)
    s = np.sqrt(np.maximum(1.0 + m[..., 2, 2] - m[..., 0, 0] - m[..., 1, 1], 1e-12)) * 2
    qz = np.stack([(m[..., 0, 2] + m[..., 2, 0]) / s, (m[..., 1, 2] + m[..., 2, 1]) / s, 0.25 * s,
                   (m[..., 1, 0] - m[..., 0, 1]) / s], axis=-1)
    for i, qq in enumerate((qw, qx, qy, qz)):
        sel = k == i
        q[sel] = qq[sel]
    q /= np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    return q


def _unit_rot(m: np.ndarray) -> np.ndarray:
    """Rotation part of an (affine, possibly uniformly scaled) 3x3."""
    u, _s, vt = np.linalg.svd(m)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


# ------------------------------------------------------------------ retarget
def _bind_worlds(asset) -> list:
    """World 4x4 of every joint in the rest pose (asset space)."""
    from .gltf import trs_matrix
    J = asset.joints
    W: list = [None] * len(J)

    def world(i):
        if W[i] is not None:
            return W[i]
        j = J[i]
        L = trs_matrix(j.rest_t if j.rest_t is not None else (0, 0, 0),
                       j.rest_r if j.rest_r is not None else (0, 0, 0, 1),
                       j.rest_s if j.rest_s is not None else (1, 1, 1))
        if j.parent >= 0:
            W[i] = world(j.parent) @ L
        else:
            W[i] = (j.base if j.base is not None else np.eye(4)) @ L
        return W[i]
    for i in range(len(J)):
        world(i)
    return W


def leg_length(pos: dict) -> float:
    """Hip drop + thigh + shin + ankle height (a straight leg, from the pelvis to the floor)."""
    total = 0.0
    for side in ("Left", "Right"):
        up, lo, ft = pos.get(side + "UpperLeg"), pos.get(side + "LowerLeg"), pos.get(side + "Foot")
        pel = pos.get("Pelvis")
        if up is None or lo is None or ft is None or pel is None:
            return 0.0
        total += (pel[1] - up[1]) + np.linalg.norm(lo - up) + np.linalg.norm(ft - lo) + (ft[1] - pos["_ground"])
    return total / 2.0


def retarget(asset, library: Optional[dict] = None, names: Optional[list] = None) -> dict:
    """{clip name: Clip} for ``asset`` (an empty dict when it can't be done)."""
    from .character_asset import Clip
    lib = library if library is not None else load_library()
    if not lib:
        return {}
    J = asset.joints
    if not J or any(j.rest_r is None for j in J):
        return {}
    canon = {j.canonical: i for i, j in enumerate(J) if j.canonical}
    by_name = {j.name: i for i, j in enumerate(J)}
    if any(b not in canon for b in BONES):
        return {}
    Wb = _bind_worlds(asset)
    P = {i: Wb[i][:3, 3] for i in range(len(J))}
    Rb = {i: _unit_rot(Wb[i][:3, :3]) for i in range(len(J))}
    children = {i: [k for k in range(len(J)) if J[k].parent == i] for i in range(len(J))}

    def child_pos(bone: str) -> Optional[np.ndarray]:
        for c in TARGET_CHILD[bone]:
            if c in canon:
                return P[canon[c]]
            if c in by_name:
                return P[by_name[c]]
        kids = children.get(canon[bone], [])
        return P[kids[0]] if kids else None

    tdir = {}
    for b in BONES:
        cp = child_pos(b)
        if cp is not None and np.linalg.norm(cp - P[canon[b]]) > 1e-6:
            tdir[b] = cp - P[canon[b]]
    root = canon.get("Root", -1)
    pel = canon["Pelvis"]
    ground = float(P[root][1]) if root >= 0 else 0.0
    named = {b: P[canon[b]] for b in BONES}
    named["_ground"] = ground
    L_t = leg_length(named)
    if L_t <= 1e-6:
        return {}
    order = sorted(range(len(J)), key=lambda i: _depth(J, i))
    meta = asset.meta or {}
    tweak = meta.get("mocap") if isinstance(meta.get("mocap"), dict) else {}
    out = {}
    for name, c in lib.items():
        if names is not None and name not in names:
            continue
        try:
            out[name] = _one(name, c, J, canon, order, Wb, Rb, P, tdir, pel, root, ground, L_t, tweak, Clip)
        except Exception:  # noqa: BLE001 - one odd clip must not cost the others
            log.exception("mocap clip %s could not be retargeted onto %s", name, getattr(asset, "id", "?"))
    return out


def _nlerp(a: np.ndarray, b: np.ndarray, u: np.ndarray) -> np.ndarray:
    d = np.sum(a * b, axis=-1, keepdims=True)
    b = np.where(d < 0, -b, b)
    r = a + (b - a) * u[..., None]
    return r / np.maximum(np.linalg.norm(r, axis=-1, keepdims=True), 1e-12)


def _damp_legs(q: np.ndarray, hips: np.ndarray, k: float) -> np.ndarray:
    """Smaller steps for a long skirt / saree (metadata "mocap": {"leg_damp": k}): while standing or
    walking, each leg segment turns only part of the way from its parent's orientation (thigh from
    the pelvis, shin from the thigh, foot from the shin) - narrow, gliding steps that keep the hem
    together. Sitting and lying (hips low) keep the full captured legs."""
    q = q.copy()
    w = np.clip((hips[:, 1] - 0.6) / 0.25, 0.0, 1.0) * k          # (F,)
    keep = 1.0 - w
    ix = {b: i for i, b in enumerate(BONES)}
    for side in ("Left", "Right"):
        par = q[:, ix["Pelvis"]]
        for seg in ("UpperLeg", "LowerLeg", "Foot"):
            i = ix[side + seg]
            q[:, i] = _nlerp(par, q[:, i], keep)
            par = q[:, i]
    return q


def _depth(J, i: int) -> int:
    d = 0
    while J[i].parent >= 0 and d < 64:
        i = J[i].parent
        d += 1
    return d


def _one(name, c, J, canon, order, Wb, Rb, P, tdir, pel, root, ground, L_t, tweak, Clip):
    q = c["q"]                                   # (F, B, 4)
    h = c["hips"]
    if c.get("loop"):
        # a cycle wraps from its last frame back to the first: repeat it at the end so the wrap
        # interpolates over one frame instead of jumping
        q = np.concatenate([q, q[:1]], axis=0)
        h = np.concatenate([h, h[:1]], axis=0)
    F = q.shape[0]
    damp = float(tweak.get("leg_damp", 0.0))
    if damp > 0.0:
        q = _damp_legs(q, h, damp)
    Q = quats_to_mats(q)                          # (F, B, 3, 3)
    n = len(J)
    W = np.zeros((F, n, 3, 3))
    done = np.zeros(n, dtype=bool)
    bone_ix = {b: k for k, b in enumerate(BONES)}
    # per-model corrections (metadata "mocap": {"arm_out": deg, ...}) applied in the bind frame
    arm_out = math.radians(float(tweak.get("arm_out", 0.0)))
    for i in order:
        j = J[i]
        b = j.canonical
        if b in bone_ix and b in tdir:
            k = bone_ix[b]
            R_al = _rotation_between(tdir[b], c["dir"][k])
            Wk = Q[:, k]
            if arm_out and b in ("LeftUpperArm", "RightUpperArm"):
                # swing the captured upper arm a little further out from the body (about the
                # forward axis, after the capture) - wide hips / a draped saree would swallow the hand
                s = 1.0 if b.startswith("Left") else -1.0
                ca, sa = math.cos(s * arm_out), math.sin(s * arm_out)
                Rz = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
                Wk = Wk @ Rz
            W[:, i] = Wk @ (R_al @ Rb[i])
        else:
            p = j.parent
            Lr = _unit_rot(Rb[p].T @ Rb[i]) if p >= 0 else Rb[i]
            W[:, i] = (W[:, p] if p >= 0 else Rb[i][None]) @ (Lr if p >= 0 else np.eye(3))
        done[i] = True
    chans = []
    times = np.arange(F, dtype=float) / float(c.get("fps", 30.0))
    for i in order:
        j = J[i]
        if j.canonical not in bone_ix:
            continue
        p = j.parent
        if p >= 0:
            Lm = np.einsum("fji,fjk->fik", W[:, p], W[:, i])        # parent^T @ child
        else:
            base = _unit_rot(j.base[:3, :3]) if j.base is not None else np.eye(3)
            Lm = np.einsum("ji,fjk->fik", base, W[:, i])
        lq = mats_to_quats(Lm)
        for f in range(1, F):                    # one hemisphere along time
            if float(np.dot(lq[f], lq[f - 1])) < 0:
                lq[f] = -lq[f]
        chans.append((i, "rotation", times, lq, "LINEAR"))
    # pelvis translation: hips over the floor, in leg lengths -> this model's leg length
    Pw = np.empty((F, 3))
    Pw[:, 0] = P[pel][0] + h[:, 0] * L_t
    Pw[:, 1] = ground + h[:, 1] * L_t
    Pw[:, 2] = P[pel][2] + h[:, 2] * L_t
    par = J[pel].parent
    M = Wb[par] if par >= 0 else np.eye(4)
    Mi = np.linalg.inv(M)
    loc = (Mi[:3, :3] @ Pw.T).T + Mi[:3, 3]
    chans.append((pel, "translation", times, loc, "LINEAR"))
    clip = Clip(name, float(times[-1]) if F > 1 else 0.0, chans, {}, "mocap")
    clip.own = True
    clip.mocap = c
    return clip
