"""Build the *demonstration* GLB characters (cat, dog, fox, rabbit, robot, study buddy).

    python tools/build_demo_assets.py [ids...] [--ultra] [--out DIR]   (needs scikit-image)

Output goes to ``DIR/<id>`` (default ``build/demo_assets/<id>``) - the built-in characters in
``assets/characters`` are artist-made GLBs now and are never overwritten. The test suite keeps two
of these demo assets in ``tests/assets/demo_cat`` and ``demo_rabbit``.

These are procedurally *sculpted* assets, not artist-made production models:

1. every character is described as a signed-distance-field sculpture on top of its rig's
   bind pose (rounded cones, ellipsoids, rounded boxes, tori) combined with smooth
   unions, so the body is one continuous organic surface - not separate primitives
2. colour regions (belly, muzzle, stripes, socks, inner ears, visor ...) are painted
   volumes; each region carries a stylised shading type (fur, cloth, skin, metal, plastic,
   emissive)
3. the surface is extracted with marching cubes at four resolutions (LOD low / medium /
   high / ultra) and lightly smoothed (Taubin)
4. every vertex is skinned to the canonical skeleton (up to 4 bones, distance-based
   weights that blend smoothly across joints)
5. morph targets: ``breath`` (chest expansion) and ``cheeks`` (cheek raise)
6. a ``Wave`` clip is baked from the procedural action library to exercise the
   GLB animation / retargeting path
7. written as glTF 2.0 binary (+Y up, +Z forward, ground at Y = 0) with metadata.json

``metadata.json`` says ``"production": false``.  Drop artist-made GLBs into the same
folders (or the user data folder) to replace them - see README "Character assets".
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from focusbuddy.companion.animation import Pose  # noqa: E402
from focusbuddy.companion.character_asset import INTERNAL_TO_ASSET  # noqa: E402
from focusbuddy.companion.characters import BUILTIN  # noqa: E402
from focusbuddy.companion.gltf import GLBWriter, mat_to_quat  # noqa: E402
from focusbuddy.companion.rig import SKELETONS, Rig  # noqa: E402
from focusbuddy.companion.skinning import rig_targets  # noqa: E402

SCALE = 200.0                       # internal units per metre (a cat is ~0.4 m tall)
LODS = {"low": 2.3, "medium": 1.55, "high": 1.1, "ultra": 0.8}       # marching-cubes cell size (units)


def hexrgb(h: str) -> tuple:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


# =========================================================== SDF primitives
def _len(v):
    return np.sqrt((v * v).sum(axis=-1))


class Prim:
    def __init__(self, kind, bone, color, mat="fur", k=4.0, paint=False, **p):
        self.kind, self.bone, self.color, self.mat, self.k, self.paint, self.p = kind, bone, color, mat, k, paint, p
        self.tag = p.pop("tag", "")

    def sd(self, P: np.ndarray) -> np.ndarray:
        p = self.p
        if self.kind == "ell":
            q = P - p["c"]
            R = p.get("R")
            if R is not None:
                q = q @ R
            r = np.asarray(p["r"], dtype=float)
            k0 = _len(q / r)
            k1 = _len(q / (r * r))
            return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)
        if self.kind == "cone":                       # rounded cone a -> b, radii r1 -> r2, optional squash
            a, b = p["a"], p["b"]
            r1, r2 = p["r1"], p["r2"]
            q = P - a
            sq = p.get("squash")                      # (axis, factor): flatten across an axis (ears)
            if sq is not None:
                ax = np.asarray(sq[0], dtype=float)
                ax /= np.linalg.norm(ax)
                q = q + np.outer(q @ ax, ax) * (sq[1] - 1.0)
            ba = b - a
            l2 = float(ba @ ba)
            rr = r1 - r2
            a2 = l2 - rr * rr
            il2 = 1.0 / l2
            y = q @ ba
            z = y - l2
            xv = q * l2 - np.outer(y, ba)
            x2 = (xv * xv).sum(axis=1)
            y2 = y * y * l2
            z2 = z * z * l2
            k = np.sign(rr) * rr * rr * x2
            out = np.empty(len(P))
            m1 = np.sign(z) * a2 * z2 > k
            m2 = np.sign(y) * a2 * y2 < k
            out[m1] = np.sqrt(x2[m1] + z2[m1]) * il2 - r2
            out[m2 & ~m1] = np.sqrt(x2[m2 & ~m1] + y2[m2 & ~m1]) * il2 - r1
            m3 = ~(m1 | m2)
            out[m3] = (np.sqrt(x2[m3] * a2 * il2) + y[m3] * rr) * il2 - r1
            return out
        if self.kind == "box":
            q = P - p["c"]
            R = p.get("R")
            if R is not None:
                q = q @ R
            h = np.asarray(p["h"], dtype=float)
            r = p.get("rad", 2.0)
            d = np.abs(q) - (h - r)
            return _len(np.maximum(d, 0.0)) + np.minimum(d.max(axis=1), 0.0) - r
        if self.kind == "torus":                      # ring around axis n through c
            q = P - p["c"]
            n = np.asarray(p["n"], dtype=float)
            n /= np.linalg.norm(n)
            h = q @ n
            radial = _len(q - np.outer(h, n))
            return np.sqrt((radial - p["R"]) ** 2 + h * h) - p["r"]
        raise ValueError(self.kind)


def smin(a, b, k):
    if k <= 1e-6:
        return np.minimum(a, b)
    h = np.maximum(k - np.abs(a - b), 0.0) / k
    return np.minimum(a, b) - h * h * k * 0.25


def V(*a):
    return np.array(a, dtype=float)


def rotz(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def rotx(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0, 0], [0, c, s], [0, -s, c]])


# ================================================================ characters
def bones_quad(t, H, hr, extras=()):
    """Canonical joint heads (internal units) + parents for a quadruped bind pose."""
    j = {"Root": V(0, 0, 0), "Pelvis": t["Pelvis"], "Spine": t["Spine"], "Chest": t["Chest"], "Neck": t["Neck"],
         "Head": H, "Jaw": H + V(hr * 0.35, hr * 0.42, 0)}
    par = {"Pelvis": "Root", "Spine": "Pelvis", "Chest": "Spine", "Neck": "Chest", "Head": "Neck", "Jaw": "Head"}
    for side, s in (("Left", -1), ("Right", 1)):
        for limb, parent in (("Arm", "Chest"), ("Leg", "Pelvis")):
            up, lo = f"{side}Upper{limb}", f"{side}Lower{limb}"
            end = f"{side}Hand" if limb == "Arm" else f"{side}Foot"
            j[up], j[lo], j[end] = t[up], t[lo], t[end]
            par[up], par[lo], par[end] = parent, up, lo
        j[f"{side}Ear"] = H + V(-1.5, -hr * 0.72, s * hr * 0.48)
        par[f"{side}Ear"] = "Head"
    prev = "Pelvis"
    i = 0
    while f"Tail{i}" in t:
        j[f"Tail{i}"] = t[f"Tail{i}"]
        par[f"Tail{i}"] = prev
        prev = f"Tail{i}"
        i += 1
    for name, pos, parent in extras:
        j[name], par[name] = pos, parent
    return j, par


def bones_biped(t, H, hr, extras=()):
    j = {"Root": V(0, 0, 0), "Pelvis": t["Pelvis"], "Spine": t["Spine"], "Chest": t["Chest"], "Neck": t["Neck"],
         "Head": H, "Jaw": H + V(hr * 0.4, hr * 0.5, 0)}
    par = {"Pelvis": "Root", "Spine": "Pelvis", "Chest": "Spine", "Neck": "Chest", "Head": "Neck", "Jaw": "Head"}
    for side, s in (("Left", -1), ("Right", 1)):
        for limb, parent in (("Arm", "Chest"), ("Leg", "Pelvis")):
            up, lo = f"{side}Upper{limb}", f"{side}Lower{limb}"
            end = f"{side}Hand" if limb == "Arm" else f"{side}Foot"
            j[up], j[lo], j[end] = t[up], t[lo], t[end]
            par[up], par[lo], par[end] = parent, up, lo
        j[f"{side}Ear"] = H + V(-1.0, 2.0, s * (hr - 1.0))
        par[f"{side}Ear"] = "Head"
    for name, pos, parent in extras:
        j[name], par[name] = pos, parent
    return j, par


def leg_prims(t, side, limb, color, r_up=(5.2, 4.2), r_lo=(4.0, 3.3), paw=(5.6, 3.5, 4.8), mat="fur",
              thigh=None, paw_color=None, sock=None, k=3.0):
    up, lo = f"{side}Upper{limb}", f"{side}Lower{limb}"
    end = f"{side}Hand" if limb == "Arm" else f"{side}Foot"
    A, B, Cc, tip = t[up], t[lo], t[end], t[end + "_tip"]
    fwd = tip - Cc
    fwd = fwd / (np.linalg.norm(fwd) or 1.0)
    out = [Prim("cone", up, color, mat, k, a=A, b=B, r1=r_up[0], r2=r_up[1]),
           Prim("cone", lo, color, mat, k, a=B, b=Cc + V(0, -0.5, 0), r1=r_lo[0], r2=r_lo[1]),
           Prim("ell", end, paw_color or color, mat, 2.5, c=Cc + fwd * 1.6 + V(0, 0.6, 0), r=paw,
                R=np.array([[fwd[0], -fwd[1], 0], [fwd[1], fwd[0], 0], [0, 0, 1.0]]))]
    if thigh:
        out.append(Prim("ell", up, color, mat, 5.0, c=A + V(thigh[3], thigh[4], 0), r=thigh[:3]))
    if sock:
        out.append(Prim("cone", lo, sock, mat, 0, paint=True, a=B + (Cc - B) * 0.45, b=Cc + V(0, 4, 0), r1=6.5, r2=7.5))
        out.append(Prim("ell", end, sock, mat, 0, paint=True, c=Cc + fwd * 1.6, r=(paw[0] + 1.5, paw[1] + 2, paw[2] + 1.5)))
    return out


def tail_prims(t, color, radii, tip_color=None, mat="fur", k=3.0):
    pts = []
    i = 0
    while f"Tail{i}" in t:
        pts.append(t[f"Tail{i}"])
        i += 1
    out = []
    for i in range(len(pts) - 1):
        r1 = radii[min(i, len(radii) - 1)]
        r2 = radii[min(i + 1, len(radii) - 1)]
        out.append(Prim("cone", f"Tail{i}", color, mat, k, a=pts[i], b=pts[i + 1], r1=r1, r2=r2))
    if tip_color is not None and len(pts) >= 2:
        out.append(Prim("ell", f"Tail{len(pts) - 2}", tip_color, mat, 0, paint=True, c=pts[-1],
                        r=(radii[-1] * 3.2,) * 3))
    return out


def ear_cone(H, hr, side, length, r1, r2, color, inner, lean=-12.0, splay=0.42, base_up=0.72, flat=0.45,
             mat="fur", bone=None, tip=None):
    s = 1 if side == "Right" else -1
    base = H + V(-1.5, -hr * base_up, s * hr * 0.48)
    d = rotz(-lean * 0.6) @ V(0.0, -1.0, s * splay)
    d = d / np.linalg.norm(d)
    b = base + d * length
    fwd = V(1.0, 0.0, 0.0)
    out = [Prim("cone", bone or f"{side}Ear", color, mat, 2.0, a=base - d * 2.0, b=b, r1=r1, r2=r2,
                squash=(fwd, flat))]
    if inner:
        out.append(Prim("cone", bone or f"{side}Ear", inner, mat, 0, paint=True, a=base + d * 2 + fwd * 1.5,
                        b=b - d * 3.0 + fwd * 1.2, r1=r1 * 0.62, r2=max(0.5, r2 * 0.5), squash=(fwd, 0.6)))
    if tip:
        out.append(Prim("ell", bone or f"{side}Ear", tip, mat, 0, paint=True, c=b, r=(r1 * 0.9, length * 0.28,
                                                                                    r1 * 0.9)))
    return out


def build_cat(sk, rp, t, pal):
    main, light, extra, accent = hexrgb(pal.main), hexrgb(pal.light), hexrgb(pal.extra), hexrgb(pal.accent)
    lay = rp.layout
    H = t["Head"]
    hr = lay["hr"]
    P = []
    P += [Prim("ell", "Pelvis", main, k=7, c=t["Pelvis"] + V(1, 0, 0), r=(15, 15, 13.5), tag="torso"),
          Prim("ell", "Spine", main, k=7, c=t["Spine"] + V(0, -0.5, 0), r=(13, 14, 13), tag="torso"),
          Prim("ell", "Chest", main, k=7, c=t["Chest"] + V(-1, 0.5, 0), r=(15, 15.5, 14), tag="torso"),
          Prim("cone", "Neck", main, k=6, a=t["Neck"] + V(-2, 4, 0), b=H + V(-3, 9, 0), r1=10, r2=9),
          Prim("ell", "Head", main, k=4, c=H, r=(hr * 0.98, hr * 0.92, hr * 1.04)),
          Prim("ell", "Head", main, k=5, c=H + V(4, 7, -9.5), r=(8.5, 7, 8), tag="cheek"),
          Prim("ell", "Head", main, k=5, c=H + V(4, 7, 9.5), r=(8.5, 7, 8), tag="cheek"),
          Prim("ell", "Head", light, k=3, c=H + V(hr * 0.66, hr * 0.33, 0), r=(7.5, 5.8, 9.4)),
          Prim("ell", "Jaw", light, k=2.5, c=H + V(hr * 0.6, hr * 0.6, 0), r=(5.2, 3.2, 5.6)),
          Prim("ell", "Head", accent, "plastic", 1.0, c=H + V(hr * 0.98, hr * 0.2, 0), r=(2.0, 1.6, 2.6))]
    for side in ("Left", "Right"):
        P += ear_cone(H, hr, side, 16.0, 7.0, 1.2, main, accent)
    P += [Prim("ell", "Chest", light, k=0, paint=True, c=t["Spine"] + V(4, 11, 0), r=(24, 7.5, 12)),
          Prim("ell", "Chest", light, k=0, paint=True, c=t["Chest"] + V(9, 4, 0), r=(7, 10, 9))]
    for i in range(3):
        P.append(Prim("ell", "Spine", extra, k=0, paint=True, c=t["Spine"] + V(-10 + i * 9.5, -13, 0),
                      r=(2.3, 6, 16), R=rotz(-12)))
    for i in range(2):
        P.append(Prim("ell", "Head", extra, k=0, paint=True, c=H + V(-3 + i * 5, -hr * 0.88, 0), r=(1.6, 4, 5.5)))
    for side in ("Left", "Right"):
        P += leg_prims(t, side, "Leg", main, (5.6, 4.3), (4.1, 3.3), (5.8, 3.4, 4.8), thigh=(8.5, 9.5, 6.5, -1, 2),
                       paw_color=light)
        P += leg_prims(t, side, "Arm", main, (5.2, 4.2), (4.0, 3.3), (5.4, 3.4, 4.6), paw_color=light)
    P += tail_prims(t, main, (3.8, 3.6, 3.3, 3.0, 2.6), tip_color=extra)
    return P


def build_dog(sk, rp, t, pal):
    main, light, extra, accent = hexrgb(pal.main), hexrgb(pal.light), hexrgb(pal.extra), hexrgb(pal.accent)
    H = t["Head"]
    hr = rp.layout["hr"]
    P = [Prim("ell", "Pelvis", main, k=7, c=t["Pelvis"] + V(1, 0, 0), r=(15.5, 15.5, 14.5), tag="torso"),
         Prim("ell", "Spine", main, k=7, c=t["Spine"], r=(13, 14.5, 14), tag="torso"),
         Prim("ell", "Chest", main, k=7, c=t["Chest"] + V(0, 1, 0), r=(16, 16.5, 15), tag="torso"),
         Prim("cone", "Neck", main, k=6, a=t["Neck"] + V(-2, 4, 0), b=H + V(-3, 10, 0), r1=11, r2=10),
         Prim("ell", "Head", main, k=4, c=H, r=(hr * 0.95, hr * 0.9, hr * 0.94)),
         Prim("ell", "Head", main, k=5, c=H + V(5, 6, -8), r=(7, 6, 6), tag="cheek"),
         Prim("ell", "Head", main, k=5, c=H + V(5, 6, 8), r=(7, 6, 6), tag="cheek"),
         Prim("cone", "Head", light, k=4, a=H + V(9, 5, 0), b=H + V(hr * 1.08, hr * 0.38, 0), r1=9.0, r2=7.2),
         Prim("ell", "Jaw", light, k=3, c=H + V(hr * 0.78, hr * 0.62, 0), r=(7, 3.5, 6.5)),
         Prim("ell", "Head", hexrgb("#2A1E2E"), "plastic", 1.0, c=H + V(hr * 1.38, hr * 0.26, 0), r=(3.0, 2.4, 3.6)),
         Prim("ell", "Head", extra, k=0, paint=True, c=H + V(-4, -hr * 0.5, 7), r=(9, 9, 7)),        # eye patch
         Prim("torus", "Neck", hexrgb("#E5484D"), "plastic", 1.0, c=t["Neck"] + V(3, -3, 0), n=V(0.8, -0.6, 0),
              R=10.3, r=2.2)]
    for side, s in (("Left", -1), ("Right", 1)):
        base = H + V(-2.5, -hr * 0.55, s * hr * 0.8)
        P.append(Prim("ell", f"{side}Ear", extra, k=2.5, c=base + V(1.5, 9.5, s * 3), r=(6.0, 11.5, 2.8),
                      R=rotx(-s * 14) @ rotz(-8)))
    P += [Prim("ell", "Chest", light, k=0, paint=True, c=t["Chest"] + V(10, 6, 0), r=(8, 12, 11)),
          Prim("ell", "Spine", light, k=0, paint=True, c=t["Spine"] + V(2, 12, 0), r=(20, 6, 11))]
    for side in ("Left", "Right"):
        P += leg_prims(t, side, "Leg", main, (6.4, 5.0), (4.8, 4.0), (6.2, 3.8, 5.2), thigh=(9.5, 10, 7.5, -1, 2))
        P += leg_prims(t, side, "Arm", main, (6.0, 5.0), (4.8, 4.0), (6.0, 3.8, 5.0), paw_color=light)
    P += tail_prims(t, main, (4.0, 3.8, 3.3, 2.6))
    return P


def build_fox(sk, rp, t, pal):
    main, light, extra, accent = hexrgb(pal.main), hexrgb(pal.light), hexrgb(pal.extra), hexrgb(pal.accent)
    H = t["Head"]
    hr = rp.layout["hr"]
    P = [Prim("ell", "Pelvis", main, k=7, c=t["Pelvis"] + V(1, -0.5, 0), r=(15, 13.5, 12), tag="torso"),
         Prim("ell", "Spine", main, k=7, c=t["Spine"] + V(0, -0.5, 0), r=(13, 12.5, 11.5), tag="torso"),
         Prim("ell", "Chest", main, k=7, c=t["Chest"], r=(15, 14.5, 12.5), tag="torso"),
         Prim("cone", "Neck", main, k=6, a=t["Neck"] + V(-2, 4, 0), b=H + V(-3, 9, 0), r1=9.5, r2=8.5),
         Prim("ell", "Head", main, k=4, c=H, r=(hr * 0.98, hr * 0.86, hr * 1.0)),
         Prim("ell", "Head", light, k=5, c=H + V(6, 8, -9), r=(9, 6.5, 7.5), tag="cheek"),
         Prim("ell", "Head", light, k=5, c=H + V(6, 8, 9), r=(9, 6.5, 7.5), tag="cheek"),
         Prim("cone", "Head", light, k=4, a=H + V(8, 6, 0), b=H + V(hr * 1.28, hr * 0.36, 0), r1=8.5, r2=2.6),
         Prim("ell", "Jaw", light, k=2, c=H + V(hr * 0.85, hr * 0.55, 0), r=(6, 2.5, 4.5)),
         Prim("ell", "Head", hexrgb("#2A1E2E"), "plastic", 1.0, c=H + V(hr * 1.32, hr * 0.3, 0), r=(2.4, 2.0, 2.6)),
         Prim("ell", "Chest", light, k=0, paint=True, c=t["Chest"] + V(10, 6, 0), r=(8, 13, 10))]
    for side in ("Left", "Right"):
        P += ear_cone(H, hr, side, 21.0, 8.0, 1.0, main, light, lean=-6, splay=0.38, tip=extra)
    for side in ("Left", "Right"):
        P += leg_prims(t, side, "Leg", main, (5.4, 4.0), (3.6, 3.0), (5.2, 3.2, 4.2), thigh=(8, 9, 6, -1, 2),
                       sock=extra)
        P += leg_prims(t, side, "Arm", main, (5.0, 3.9), (3.6, 3.0), (5.0, 3.2, 4.0), sock=extra)
    P += tail_prims(t, main, (4.5, 7.5, 8.5, 7.0, 3.8), tip_color=light, k=5)
    return P


def build_rabbit(sk, rp, t, pal):
    main, light, extra, accent = hexrgb(pal.main), hexrgb(pal.light), hexrgb(pal.extra), hexrgb(pal.accent)
    H = t["Head"] + V(-1, 2, 0)
    hr = rp.layout["hr"]
    P = [Prim("ell", "Pelvis", main, k=8, c=t["Pelvis"] + V(2, -1, 0), r=(17, 17.5, 15.5), tag="torso"),
         Prim("ell", "Spine", main, k=8, c=t["Spine"] + V(0, 0.5, 0), r=(13, 14, 13), tag="torso"),
         Prim("ell", "Chest", main, k=8, c=t["Chest"] + V(-2, 1, 0), r=(13, 13.5, 12.5), tag="torso"),
         Prim("cone", "Neck", main, k=7, a=t["Neck"] + V(-3, 4, 0), b=H + V(-3, 9, 0), r1=10, r2=9),
         Prim("ell", "Head", main, k=4, c=H, r=(hr * 0.98, hr * 0.92, hr * 1.0)),
         Prim("ell", "Head", main, k=5, c=H + V(4, 8, -9), r=(8.5, 7, 7.5), tag="cheek"),
         Prim("ell", "Head", main, k=5, c=H + V(4, 8, 9), r=(8.5, 7, 7.5), tag="cheek"),
         Prim("ell", "Head", light, k=3, c=H + V(hr * 0.68, hr * 0.4, 0), r=(6, 5, 8)),
         Prim("ell", "Jaw", light, k=2, c=H + V(hr * 0.62, hr * 0.62, 0), r=(4, 2.6, 4.5)),
         Prim("ell", "Head", accent, "plastic", 1.0, c=H + V(hr * 0.96, hr * 0.24, 0), r=(1.8, 1.4, 2.4)),
         Prim("ell", "Tail0", light, k=3, c=t["Tail0"] + V(-2, -2, 0), r=(6.5, 6.5, 6.5))]
    for side, s in (("Left", -1), ("Right", 1)):
        base = H + V(-3, -hr * 0.8, s * hr * 0.32)
        d = rotz(8) @ V(-0.1, -1.0, s * 0.18)
        d = d / np.linalg.norm(d)
        P.append(Prim("cone", f"{side}Ear", main, k=3, a=base, b=base + d * 27, r1=4.6, r2=3.8,
                      squash=(V(1, 0, 0), 0.5)))
        P.append(Prim("cone", f"{side}Ear", accent, k=0, paint=True, a=base + d * 5 + V(2, 0, 0),
                      b=base + d * 23 + V(1.5, 0, 0), r1=2.6, r2=2.0, squash=(V(1, 0, 0), 0.6)))
    for side in ("Left", "Right"):
        P += leg_prims(t, side, "Leg", main, (6.0, 4.4), (4.2, 3.4), (8.8, 3.0, 4.6), thigh=(11, 11.5, 8, -1, 3))
        P += leg_prims(t, side, "Arm", main, (4.2, 3.2), (3.0, 2.6), (4.4, 2.8, 3.6))
    P += [Prim("ell", "Spine", light, k=0, paint=True, c=t["Spine"] + V(6, 11, 0), r=(20, 6, 11))]
    return P


def build_robot(sk, rp, t, pal):
    main, shade, light = hexrgb(pal.main), hexrgb(pal.shade), hexrgb(pal.light)
    accent, extra, outline = hexrgb(pal.accent), hexrgb(pal.extra), hexrgb(pal.outline)
    H = t["Head"]
    P = [Prim("box", "Chest", main, "metal", 1.2, c=t["Chest"] + V(0, 0.5, 0), h=(11, 13, 14.5), rad=5.5,
              tag="torso"),
         Prim("box", "Chest", light, "plastic", 0, paint=True, c=t["Chest"] + V(10, 0, 0), h=(3, 8, 9), rad=3),
         Prim("ell", "Chest", accent, "emissive", 0.5, c=t["Chest"] + V(10.5, -1, 0), r=(1.8, 3.0, 3.0)),
         Prim("box", "Pelvis", shade, "metal", 1.0, c=t["Pelvis"] + V(0, -1.5, 0), h=(8, 5, 11), rad=3),
         Prim("cone", "Neck", outline, "metal", 0.8, a=t["Neck"] + V(0, 2, 0), b=H + V(0, 16, 0), r1=4, r2=4),
         Prim("box", "Head", main, "metal", 0.6, c=H, h=(17, 16, 20), rad=8),
         Prim("box", "Head", extra, "plastic", 0.6, c=H + V(9.5, 1, 0), h=(8, 10.5, 14.5), rad=5),
         Prim("cone", "Head", outline, "metal", 0.5, a=H + V(0, -15, 0), b=H + V(1.5, -26, 0), r1=1.3, r2=1.1),
         Prim("ell", "Antenna", accent, "emissive", 0.5, c=H + V(1.8, -28.5, 0), r=(3.6, 3.6, 3.6))]
    for side, s in (("Left", -1), ("Right", 1)):
        P.append(Prim("cone", f"{side}Ear", shade, "metal", 0.6, a=H + V(0, 1, s * 18.5), b=H + V(0, 1, s * 22.5),
                      r1=5.6, r2=4.8))
        for limb in ("Arm", "Leg"):
            up, lo = f"{side}Upper{limb}", f"{side}Lower{limb}"
            end = f"{side}Hand" if limb == "Arm" else f"{side}Foot"
            r = 3.0 if limb == "Arm" else 3.4
            P += [Prim("cone", up, shade, "metal", 0.6, a=t[up], b=t[lo], r1=r, r2=r),
                  Prim("cone", lo, shade, "metal", 0.6, a=t[lo], b=t[end], r1=r * 0.95, r2=r * 0.9),
                  Prim("ell", up, outline, "metal", 0.5, c=t[up], r=(r + 1.6,) * 3),
                  Prim("ell", lo, outline, "metal", 0.5, c=t[lo], r=(r + 1.0,) * 3)]
            if limb == "Arm":
                P.append(Prim("ell", end, main, "metal", 0.8, c=t[end] + (t[end] - t[lo]) * 0.25, r=(4.4, 4.4, 4.4)))
            else:
                fwd = t[end + "_tip"] - t[end]
                fwd /= np.linalg.norm(fwd)
                P.append(Prim("box", end, main, "metal", 0.8, c=t[end] + fwd * 2.4 + V(0, 0.6, 0), h=(6.8, 3.2, 4.6),
                              rad=2.2))
    return P


def build_buddy(sk, rp, t, pal):
    hoodie, shade, skin, hair = hexrgb(pal.main), hexrgb(pal.shade), hexrgb(pal.light), hexrgb(pal.extra)
    pants, shoe, sole = hexrgb("#3E4A6E"), hexrgb("#FFFFFF"), hexrgb("#9AA3B5")
    H = t["Head"]
    hr = 21.5
    P = [Prim("ell", "Chest", hoodie, "cloth", 4, c=t["Chest"] + V(0, 1.5, 0), r=(11.5, 14, 14.5), tag="torso"),
         Prim("ell", "Spine", hoodie, "cloth", 4, c=t["Spine"] + V(0, 1, 0), r=(10.5, 9, 13), tag="torso"),
         Prim("ell", "Pelvis", pants, "cloth", 3, c=t["Pelvis"] + V(0, -1.5, 0), r=(8.5, 6, 11.5)),
         Prim("ell", "Pelvis", shade, "cloth", 0, paint=True, c=t["Pelvis"] + V(0, -5.5, 0), r=(12, 2.2, 16),
              tag="hem"),
         Prim("ell", "Chest", shade, "cloth", 3, c=t["Neck"] + V(-6, 4, 0), r=(5.5, 4.5, 11)),       # hood
         Prim("ell", "Chest", shade, "cloth", 0, paint=True, c=t["Chest"] + V(10.5, 7, 0), r=(3, 4, 8)),  # pocket
         Prim("cone", "Neck", skin, "skin", 2.5, a=t["Neck"] + V(0, 1, 0), b=H + V(0, 15, 0), r1=5, r2=5),
         Prim("ell", "Head", skin, "skin", 3, c=H, r=(hr * 0.96, hr, hr * 0.92)),
         Prim("ell", "Head", skin, "skin", 3, c=H + V(4, 9, -8.5), r=(7, 6, 6), tag="cheek"),
         Prim("ell", "Head", skin, "skin", 3, c=H + V(4, 9, 8.5), r=(7, 6, 6), tag="cheek"),
         Prim("ell", "Head", skin, "skin", 1.5, c=H + V(hr * 0.96, 3.5, 0), r=(2.2, 2.4, 2.0)),       # nose
         Prim("ell", "Jaw", skin, "skin", 2.5, c=H + V(hr * 0.55, hr * 0.62, 0), r=(5.5, 3.5, 7)),
         Prim("ell", "Head", hair, "fur", 2.5, c=H + V(-4, -5.5, 0), r=(hr * 0.98, hr * 0.86, hr * 0.98)),
         Prim("ell", "Head", hair, "fur", 2.5, c=H + V(9.5, -13, 0), r=(9.5, 5.5, 16), R=rotz(24)),
         Prim("cone", "Hair", hair, "fur", 2.5, a=H + V(-8, -16, 0), b=H + V(-13, -25, 0), r1=5, r2=2.2)]
    for side, s in (("Left", -1), ("Right", 1)):
        P.append(Prim("ell", f"{side}Ear", skin, "skin", 2, c=H + V(-1, 3, s * (hr - 1.5)), r=(3.4, 4.5, 2.4)))
        up, lo, hand = f"{side}UpperArm", f"{side}LowerArm", f"{side}Hand"
        P += [Prim("cone", up, hoodie, "cloth", 3, a=t[up] + V(0, 1.5, -s * 2.5), b=t[lo], r1=4.6, r2=3.8),
              Prim("cone", lo, hoodie, "cloth", 2.5, a=t[lo], b=t[hand], r1=3.8, r2=3.4),
              Prim("ell", hand, skin, "skin", 2, c=t[hand] + (t[hand] - t[lo]) * 0.32, r=(3.6, 4.3, 3.0)),
              Prim("ell", lo, shade, "cloth", 0, paint=True, c=t[hand] - (t[hand] - t[lo]) * 0.05, r=(4.4, 2.0, 4.4))]
        upl, lol, foot = f"{side}UpperLeg", f"{side}LowerLeg", f"{side}Foot"
        fwd = t[foot + "_tip"] - t[foot]
        fwd /= np.linalg.norm(fwd)
        P += [Prim("cone", upl, pants, "cloth", 3, a=t[upl] + V(0, -1, 0), b=t[lol], r1=5.0, r2=4.3),
              Prim("cone", lol, pants, "cloth", 2.5, a=t[lol], b=t[foot] + V(0, -2, 0), r1=4.2, r2=3.8),
              Prim("box", foot, shoe, "plastic", 1.5, c=t[foot] + fwd * 2.8 + V(0, 0.7, 0), h=(6.8, 3.3, 4.4), rad=2.6),
              Prim("box", foot, sole, "plastic", 0, paint=True, c=t[foot] + fwd * 2.8 + V(0, 3.4, 0),
                   h=(8, 1.2, 5.5), rad=0.5)]
    return P


BUILDERS = {"cat": build_cat, "dog": build_dog, "fox": build_fox, "rabbit": build_rabbit, "robot": build_robot,
            "buddy": build_buddy}
EXTRAS = {
    "dog": lambda t, H, hr: [("Tag", t["Neck"] + V(9, 6, 0), "Neck")],
    "robot": lambda t, H, hr: [("Antenna", H + V(0, -15, 0), "Head")],
    "buddy": lambda t, H, hr: [("Hair", H + V(-8, -16, 0), "Head"), ("Hem", t["Pelvis"] + V(0, -5, 0), "Pelvis")],
}
FACES = {   # same face anchors as the procedural models (head bone frame, internal units)
    "cat": {"eye_dirs": [[-27.0, 0.02], [27.0, 0.02]], "eye_size": 1.3, "head_radius": 21.0, "mouth": [0.98, 0.5, 0.0],
            "mouth_normal": [1.0, 0.45, 0.0], "whiskers": True},
    "dog": {"eye_dirs": [[-27.0, 0.0], [27.0, 0.0]], "eye_size": 1.3, "head_radius": 20.5, "mouth": [1.22, 0.52, 0.0],
            "mouth_normal": [1.0, 0.5, 0.0]},
    "fox": {"eye_dirs": [[-26.0, 0.02], [26.0, 0.02]], "eye_size": 1.25, "head_radius": 20.5,
            "mouth": [1.18, 0.5, 0.0], "mouth_normal": [1.0, 0.55, 0.0]},
    "rabbit": {"eye_dirs": [[-30.0, 0.04], [30.0, 0.04]], "eye_size": 1.3, "head_radius": 20.0,
               "mouth": [0.96, 0.46, 0.0], "mouth_normal": [1.0, 0.45, 0.0], "center": [-1.0, 2.0, 0.0]},
    "robot": {"eye_dirs": [[-24.0, 0.0], [24.0, 0.0]], "eye_size": 1.2, "head_radius": 22.6,
              "mouth": [22.5, 9.0, 0.0], "mouth_absolute": True, "mouth_normal": [1.0, 0.35, 0.0], "robot": True},
    "buddy": {"eye_dirs": [[-26.0, 0.12], [26.0, 0.12]], "eye_size": 1.2, "head_radius": 20.5,
              "mouth": [19.0, 10.5, 0.0], "mouth_absolute": True, "mouth_normal": [1.0, 0.55, 0.0]},
}


# ============================================================== mesh building
def evaluate(prims, P):
    field = None
    for pr in prims:
        if pr.paint:
            continue
        d = pr.sd(P)
        field = d if field is None else smin(field, d, pr.k)
    return field


def taubin(verts, faces, iters=4, lam=0.5, mu=-0.53):
    n = len(verts)
    nb = [[] for _ in range(n)]
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    deg = np.bincount(e.ravel(), minlength=n).astype(float)
    v = verts.copy()
    for _ in range(iters):
        for f in (lam, mu):
            acc = np.zeros_like(v)
            np.add.at(acc, e[:, 0], v[e[:, 1]])
            np.add.at(acc, e[:, 1], v[e[:, 0]])
            lap = acc / np.maximum(deg, 1)[:, None] - v
            v = v + f * lap
    del nb
    return v


def vertex_normals(v, f):
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    n = np.zeros_like(v)
    for k in range(3):
        np.add.at(n, f[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)


def build(cid: str, out_dir: Path, lods=("low", "medium", "high", "ultra")) -> dict:
    from skimage.measure import marching_cubes
    spec = BUILTIN[cid]
    sk = SKELETONS[cid]
    rig = Rig(sk)
    rp = rig.solve(Pose())
    kind = sk.kind
    t = rig_targets(rp, kind)
    if kind == "biped":
        # bind in an A-pose: arms angled away from the body and legs slightly apart, so arms, legs and
        # torso are separate surfaces (the rig re-poses them every frame anyway)
        for side, s_ in (("Left", -1), ("Right", 1)):
            sh = t[f"{side}UpperArm"]
            up = V(0.06, 0.86, 0.5 * s_)
            el = sh + up / np.linalg.norm(up) * 9.5
            lo = V(0.12, 0.88, 0.42 * s_)
            ha = el + lo / np.linalg.norm(lo) * 9.5
            t[f"{side}LowerArm"], t[f"{side}Hand"] = el, ha
            t[f"{side}Hand_tip"] = ha + lo / np.linalg.norm(lo) * 4.0
            for nm in ("UpperLeg", "LowerLeg", "Foot", "Foot_tip"):
                t[f"{side}{nm}"] = t[f"{side}{nm}"] + V(0, 0, s_ * 1.4)
    H = t["Head"] + (V(-1, 2, 0) if cid == "rabbit" else V(0, 0, 0))
    hr = rp.layout["hr"]
    extras = EXTRAS.get(cid, lambda *_: [])(t, H, hr)
    joints, parents = (bones_quad if kind != "biped" else bones_biped)(t, H, hr, extras)
    prims = BUILDERS[cid](sk, rp, t, spec.palette)
    names = list(joints)
    jidx = {n: i for i, n in enumerate(names)}
    stats = {}
    result = {}
    lo = np.full(3, 1e9)
    hi = np.full(3, -1e9)
    for pr in prims:            # bounds from a coarse probe of each primitive
        p = pr.p
        c = p.get("c") if "c" in p else (p["a"] + p["b"]) / 2
        ext = max(np.max(p.get("r", (0,))) if "r" in p else 0, np.max(p.get("h", (0,))) if "h" in p else 0,
                  (np.linalg.norm(p["b"] - p["a"]) / 2 + max(p["r1"], p["r2"])) if "a" in p else 0,
                  p.get("R", 0) + p.get("r", 0) if pr.kind == "torus" else 0)
        lo = np.minimum(lo, c - ext - 6)
        hi = np.maximum(hi, c + ext + 6)
    for lod in lods:
        h = LODS[lod]
        xs = [np.arange(lo[i], hi[i] + h, h) for i in range(3)]
        G = np.stack(np.meshgrid(*xs, indexing="ij"), axis=-1).reshape(-1, 3)
        field = evaluate(prims, G).reshape(len(xs[0]), len(xs[1]), len(xs[2]))
        verts, faces, _, _ = marching_cubes(field, 0.0, spacing=(h, h, h))
        verts = verts + lo
        verts = taubin(verts, faces, iters=3 if lod != "low" else 2)
        # attributes from the primitives
        geo = [p for p in prims if not p.paint]
        D = np.stack([p.sd(verts) for p in geo], axis=1)                  # (V, P)
        nearest = np.argmin(D, axis=1)
        col = np.array([geo[i].color for i in nearest], dtype=float)
        mat = np.array([geo[i].mat for i in nearest], dtype=object)
        for p in prims:
            if p.paint:
                inside = p.sd(verts) < 0
                col[inside] = p.color
                if p.mat in ("plastic", "emissive"):
                    mat[inside] = p.mat
        # skin weights: the nearest part's bone, blended only with its parent / children near the joint
        # between them (no bleeding between surfaces that merely touch, e.g. an arm resting on the body)
        sig = 2.6 if cid != "robot" else 1.2
        infl = np.exp(-np.maximum(D, 0.0) / sig) + 1e-9
        Wb = np.zeros((len(verts), len(names)))
        for k, p in enumerate(geo):
            Wb[:, jidx[p.bone]] = np.maximum(Wb[:, jidx[p.bone]], infl[:, k] ** 2)
        near_bone = np.array([jidx[geo[i].bone] for i in nearest])
        par_idx = np.array([jidx[parents[n]] if parents.get(n) else -1 for n in names])
        W = np.zeros_like(Wb)
        rows = np.arange(len(verts))
        W[rows, near_bone] = Wb[rows, near_bone]
        R_gate = 7.0 if cid != "robot" else 3.5
        jpos = np.array([joints[n] for n in names])
        for b in range(len(names)):
            sel = near_bone == b
            if not sel.any():
                continue
            pb = par_idx[b]
            if pb >= 0:                                # blend into the parent near this bone's head
                g_ = np.exp(-(np.linalg.norm(verts[sel] - jpos[b], axis=1) / R_gate) ** 2)
                W[sel, pb] = np.maximum(W[sel, pb], Wb[sel, pb] * g_)
            for c in np.flatnonzero(par_idx == b):     # and into children near their heads
                g_ = np.exp(-(np.linalg.norm(verts[sel] - jpos[c], axis=1) / R_gate) ** 2)
                W[sel, c] = np.maximum(W[sel, c], Wb[sel, c] * g_)
        top = np.argsort(-W, axis=1)[:, :4]
        tw = np.take_along_axis(W, top, axis=1)
        tw = tw / tw.sum(axis=1, keepdims=True)
        tw[tw < 0.02] = 0.0
        tw = tw / tw.sum(axis=1, keepdims=True)
        normals = vertex_normals(verts, faces)
        # morph targets
        torso = np.array([p.tag == "torso" for p in geo])
        cheek = np.array([p.tag == "cheek" for p in geo])
        tot = infl.sum(axis=1)
        wt = (infl[:, torso].sum(axis=1) / tot) if torso.any() else np.zeros(len(verts))
        wc = (infl[:, cheek].sum(axis=1) / tot) if cheek.any() else np.zeros(len(verts))
        morphs = {"breath": normals * (1.8 * wt)[:, None], "cheeks": normals * (1.6 * wc)[:, None]}
        result[lod] = (verts, faces, normals, col, mat, top, tw, morphs)
        stats[lod] = len(faces)
    out_dir.mkdir(parents=True, exist_ok=True)
    height = float(max(r[0][:, 1].max() for r in result.values()) - min(r[0][:, 1].min() for r in result.values()))
    for lod, (verts, faces, normals, col, mat, top, tw, morphs) in result.items():
        w = GLBWriter("FocusBuddy build_demo_assets.py")
        A = INTERNAL_TO_ASSET
        node_of = {}
        for nme in names:
            par = parents.get(nme)
            rel = joints[nme] - (joints[par] if par else V(0, 0, 0))
            node_of[nme] = w.add_node(nme, translation=(A @ rel) / SCALE, parent=node_of.get(par),
                                      root=par is None)
        ibm = np.repeat(np.eye(4)[None], len(names), axis=0)
        for i, nme in enumerate(names):
            ibm[i, :3, 3] = -(A @ joints[nme]) / SCALE
        skin = w.add_skin([node_of[n] for n in names], ibm, node_of["Root"])
        mats = {}
        for key in sorted(set(mat)):
            mats[key] = w.add_material(f"{cid}_{key}", (1, 1, 1, 1), 0.85 if key == "metal" else 0.0,
                                       0.35 if key in ("metal", "plastic") else 0.85,
                                       emissive=(0.9, 0.9, 0.9) if key == "emissive" else (0, 0, 0),
                                       extras={"shading": key})
        fmat = np.array([max(set(mat[f]), key=list(mat[f]).count) for f in faces], dtype=object)
        for key, mi in mats.items():
            fsel = faces[fmat == key]
            if not len(fsel):
                continue
            used, inv = np.unique(fsel.ravel(), return_inverse=True)
            sub = inv.reshape(-1, 3)
            pos_a = (verts[used] @ A.T) / SCALE
            nor_a = normals[used] @ A.T
            targets = [{"POSITION": (morphs[m][used] @ A.T) / SCALE} for m in ("breath", "cheeks")]
            mesh = w.add_mesh(f"{cid}_{key}", pos_a, sub, nor_a, col[used], top[used], tw[used], mi, targets,
                              ["breath", "cheeks"])
            w.add_node(f"{cid}_{key}_mesh", mesh=mesh, skin=skin, root=True)
        if lod == "high":
            bake_wave(w, cid, node_of, joints, names, parents)
        fname = {"high": "model.glb", "low": "model_low.glb", "medium": "model_medium.glb",
                 "ultra": "model_ultra.glb"}[lod]
        w.save(out_dir / fname, extras={"focusbuddy": {"id": cid, "lod": lod}})
    meta = {"id": cid, "kind": "biped" if kind == "biped" else "quadruped", "species": spec.species,
            "production": False,
            "vertex_colors": "display",
            "description": "Procedurally sculpted DEMONSTRATION asset (signed-distance sculpt -> marching cubes, "
                           "auto skin weights), generated by tools/build_demo_assets.py. Not artist-made: replace "
                           "model*.glb with a production model for final quality.",
            "rig_height": height, "face": dict(FACES[cid], mode="decal"),
            "morph_map": {"breath": {"param": "breath", "scale": 1.0}, "cheeks": {"param": "cheek", "scale": 1.0}},
            "clips": {"WAVE": "Wave"}, "clip_mask": {"Wave": "upper"},
            "triangles": stats}
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return stats


def bake_wave(w: GLBWriter, cid, node_of, joints, names, parents) -> None:
    """Bake the procedural WAVE action into a glTF clip (local rotations per bone)."""
    from focusbuddy.companion.actions import ActionPlayer
    from focusbuddy.companion.animation import AnimationController
    from focusbuddy.companion.character_asset import ASSET_TO_INTERNAL, CharacterAsset, Joint
    from focusbuddy.companion.profile import get_profile
    from focusbuddy.companion.skinning import RigDriver
    from focusbuddy.companion.states import CompanionState as S
    sk = SKELETONS[cid]
    asset = CharacterAsset(cid, "gltf", {})
    for n in names:
        bw = np.eye(4)
        bw[:3, 3] = joints[n]
        par = parents.get(n)
        asset.joints.append(Joint(n, n, names.index(par) if par else -1, bw, np.linalg.inv(bw)))
    drv = RigDriver(asset, sk.kind, cid)
    rig = Rig(sk)
    an = AnimationController(seed=1, profile=get_profile(cid))
    an.set_rig(rig)
    pl = ActionPlayer(cid)
    pl.play("WAVE", 0.0)
    times, rots = [], {n: [] for n in names}
    A = ASSET_TO_INTERNAL
    fps = 15
    for k in range(int(1.6 * fps) + 1):
        tt = k / fps
        pl.update(tt)
        p = an.update(S.WAVING if sk.kind == "biped" else S.IDLE, tt, 1 / fps, None, None, None, 1.0, pl.layers())
        drv.pose(p, p.rig, np.eye(3))
        Dw = drv.world_R
        times.append(tt)
        for i, n in enumerate(names):
            par = parents.get(n)
            loc = Dw[i] if par is None else Dw[names.index(par)].T @ Dw[i]
            rots[n].append(mat_to_quat(A.T @ loc @ A))
    chans = [(node_of[n], "rotation", np.array(times), np.array(rots[n])) for n in names
             if np.ptp(np.array(rots[n]), axis=0).max() > 1e-4]
    if chans:
        w.add_animation("Wave", chans)


def main(argv):
    ultra = "--ultra" in argv            # the ultra LOD (~60k triangles) is optional: it falls back to "high"
    out_root = ROOT / "build" / "demo_assets"
    if "--out" in argv:
        k = argv.index("--out")
        out_root = Path(argv[k + 1]).resolve()
        argv = argv[:k] + argv[k + 2:]
    ids = [a for a in argv if not a.startswith("--")] or list(BUILDERS)
    lods = ("low", "medium", "high", "ultra") if ultra else ("low", "medium", "high")
    for cid in ids:
        t0 = time.time()
        stats = build(cid, out_root / cid, lods)
        print(f"{cid}: {stats}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main(sys.argv[1:])
