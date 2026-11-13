"""Skeletons for the companion models: landmarks read off the annotation views (tools/rig/grid.py),
refined on the mesh, plus the parts found automatically (ear tips, head top, tail chain).

Landmarks are (Y, Z) in the rig frame (height 1, facing -Y, left = +X). Limb joints are then
centred inside the limb: the centroid of the mesh slice through the landmark's height near the
landmark. Spine joints sit on the body's midline at their Y. A head drawn turned toward the camera
(as the image-to-3D source pictures often are) is straightened first (see ``straighten_head``).
"""

from __future__ import annotations

import math

import numpy as np

from core import Model
from joints import parents_for

# ------------------------------------------------------------------ per-character landmarks
# quadruped legs: FL FR BL BR -> top (shoulder / hip), mid (elbow / hock), end (wrist / ankle), tip (toes)
LANDMARKS = {
    "fox": {
        "Pelvis": (0.16, 0.49), "Spine": (-0.10, 0.46), "Chest": (-0.34, 0.47), "Neck": (-0.50, 0.58),
        "Head": (-0.56, 0.68), "tail_base": (0.26, 0.55),
        "FL": {"top": (-0.40, 0.50), "mid": (-0.36, 0.30), "end": (-0.47, 0.05), "tip": (-0.56, 0.03)},
        "FR": {"top": (-0.42, 0.50), "mid": (-0.40, 0.30), "end": (-0.49, 0.05), "tip": (-0.59, 0.03)},
        "BL": {"top": (0.16, 0.47), "mid": (0.205, 0.28), "end": (0.17, 0.05), "tip": (0.08, 0.03)},
        "BR": {"top": (0.17, 0.47), "mid": (0.22, 0.28), "end": (0.155, 0.05), "tip": (0.04, 0.03)},
    },
    "cat": {
        "Pelvis": (0.25, 0.42), "Spine": (0.0, 0.40), "Chest": (-0.30, 0.40), "Neck": (-0.39, 0.49),
        "Head": (-0.455, 0.56), "tail_base": (0.335, 0.49), "tail_tip": (0.527, 0.85),
        "FL": {"top": (-0.36, 0.44), "mid": (-0.37, 0.25), "end": (-0.38, 0.06), "tip": (-0.44, 0.02)},
        "FR": {"top": (-0.37, 0.44), "mid": (-0.38, 0.25), "end": (-0.40, 0.06), "tip": (-0.46, 0.02)},
        "BL": {"top": (0.26, 0.41), "mid": (0.335, 0.18), "end": (0.27, 0.055), "tip": (0.19, 0.02)},
        "BR": {"top": (0.27, 0.41), "mid": (0.34, 0.18), "end": (0.28, 0.055), "tip": (0.20, 0.02)},
    },
    "dog": {
        "Pelvis": (0.30, 0.52), "Spine": (0.087, 0.49), "Chest": (-0.17, 0.46), "Neck": (-0.235, 0.54),
        "Head": (-0.275, 0.62), "tail_base": (0.37, 0.57), "tail_tip": (0.356, 0.87), "ears": False,
        "FL": {"top": (-0.18, 0.46), "mid": (-0.17, 0.26), "end": (-0.235, 0.065), "tip": (-0.315, 0.027)},
        "FR": {"top": (-0.19, 0.46), "mid": (-0.18, 0.26), "end": (-0.25, 0.065), "tip": (-0.33, 0.027)},
        "BL": {"top": (0.30, 0.46), "mid": (0.456, 0.195), "end": (0.40, 0.06), "tip": (0.31, 0.027)},
        "BR": {"top": (0.30, 0.46), "mid": (0.45, 0.195), "end": (0.39, 0.06), "tip": (0.30, 0.027)},
    },
    "rabbit": {
        "Pelvis": (0.31, 0.36), "Spine": (0.143, 0.35), "Chest": (-0.076, 0.34), "Neck": (-0.097, 0.40),
        "Head": (-0.128, 0.445), "tail_base": (0.356, 0.435), "tail_n": 1, "tail_tip": (0.51, 0.43),
        "FL": {"top": (-0.076, 0.32), "mid": (-0.066, 0.215), "end": (-0.128, 0.06), "tip": (-0.20, 0.026)},
        "FR": {"top": (-0.08, 0.32), "mid": (-0.07, 0.215), "end": (-0.138, 0.06), "tip": (-0.21, 0.026)},
        "BL": {"top": (0.30, 0.32), "mid": (0.434, 0.173), "end": (0.382, 0.055), "tip": (0.30, 0.02)},
        "BR": {"top": (0.30, 0.32), "mid": (0.43, 0.173), "end": (0.37, 0.055), "tip": (0.29, 0.02)},
    },
}


# bipeds: (X, Z) read on the front view, Y on the side view; arms L = the character's left (+X)
BIPEDS = {
    "buddy": {"Pelvis": (0.0, 0.0, 0.484), "Spine": (0.0, 0.01, 0.58), "Chest": (0.0, 0.016, 0.72),
              "Neck": (0.0, 0.0, 0.855), "Head": (0.0, -0.01, 0.895),
              "shoulder": (0.11, 0.0, 0.78), "elbow": (0.208, -0.02, 0.667), "wrist": (0.266, -0.026, 0.599),
              "handtip": (0.31, -0.03, 0.516), "hip": (0.085, 0.0, 0.47), "knee": (0.094, 0.016, 0.27),
              "ankle": (0.104, 0.026, 0.07), "toe": (0.104, -0.094, 0.02)},
    "study_girl": {"Pelvis": (0.0, 0.0, 0.50), "Spine": (0.0, 0.01, 0.60), "Chest": (0.0, 0.01, 0.73),
                   "Neck": (0.0, 0.0, 0.838), "Head": (0.0, -0.01, 0.875),
                   "shoulder": (0.10, 0.0, 0.79), "elbow": (0.177, -0.016, 0.68), "wrist": (0.247, -0.03, 0.615),
                   "handtip": (0.298, -0.036, 0.536), "hip": (0.075, 0.0, 0.48), "knee": (0.089, 0.016, 0.27),
                   "ankle": (0.099, 0.016, 0.07), "toe": (0.099, -0.094, 0.02), "hair_lum": 0.3},
    "saree_girl": {"Pelvis": (-0.005, -0.09, 0.52), "Spine": (-0.005, -0.095, 0.62), "Chest": (-0.005, -0.104, 0.73),
                   "Neck": (-0.005, -0.094, 0.84), "Head": (-0.005, -0.095, 0.875),
                   "shoulder": (0.10, -0.10, 0.79), "elbow": (0.172, -0.104, 0.687), "wrist": (0.25, -0.104, 0.615),
                   "handtip": (0.294, -0.11, 0.526), "hip": (0.07, -0.09, 0.50), "knee": (0.07, -0.09, 0.27),
                   "ankle": (0.075, -0.09, 0.05), "toe": (0.08, -0.15, 0.015), "hair_lum": 0.3, "skirt": True,
                   "centre_x": -0.005},
}
LANDMARKS.update(BIPEDS)


def section_centre(V, p, direction, half=0.008, radius=0.08):
    """Centre of the mesh's cross-section through ``p`` perpendicular to ``direction`` (the middle
    of a limb there)."""
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d)
    sel = V[(np.abs((V - p) @ d) < half) & (np.linalg.norm(V - p, axis=1) < radius)]
    if len(sel) < 8:
        return np.asarray(p, float)
    # extents along two axes perpendicular to d
    u = np.cross(d, [0, 0, 1.0]) if abs(d[2]) < 0.9 else np.cross(d, [1.0, 0, 0])
    u /= np.linalg.norm(u)
    w = np.cross(d, u)
    a, b = (sel - p) @ u, (sel - p) @ w
    return np.asarray(p, float) + u * 0.5 * (a.min() + a.max()) + w * 0.5 * (b.min() + b.max())


def biped_skeleton(model: Model, lm: dict) -> dict:
    V = model.V
    J: dict = {}
    for n in ("Pelvis", "Spine", "Chest", "Neck", "Head"):
        J[n] = np.array(lm[n], float)
    J["Root"] = np.array([J["Pelvis"][0], J["Pelvis"][1], 0.0])
    cx = lm.get("centre_x", 0.0)
    for side, sg in (("Left", 1.0), ("Right", -1.0)):
        def P(key, sg=sg):
            x, y, z = lm[key]
            return np.array([x if sg > 0 else 2 * cx - x, y, z], float)
        sh, el, wr, tip = P("shoulder"), P("elbow"), P("wrist"), P("handtip")
        el = section_centre(V, el, wr - sh, radius=0.07)
        wr = section_centre(V, wr, tip - el, radius=0.05)
        hip, kn, an, toe = P("hip"), P("knee"), P("ankle"), P("toe")
        if not lm.get("skirt"):
            kn = section_centre(V, kn, an - hip, radius=0.08)
            kn[0] = 0.5 * kn[0] + 0.5 * P("knee")[0]                # one leg, not both trouser legs
            an = section_centre(V, an, kn - an, radius=0.07)
        J[f"{side}UpperArm"], J[f"{side}LowerArm"], J[f"{side}Hand"] = sh, el, wr
        J[f"{side}HandTip"] = tip
        J[f"{side}UpperLeg"], J[f"{side}LowerLeg"], J[f"{side}Foot"] = hip, kn, an
        J[f"{side}FootTip"] = toe
    return J


def body_mid_x(V, y, z0, z1, half=0.012):
    s = V[(np.abs(V[:, 1] - y) < half) & (V[:, 2] > z0) & (V[:, 2] < z1)]
    if len(s) < 5:
        return 0.0
    return 0.5 * float(s[:, 0].min() + s[:, 0].max())


def limb_centre(V, x, y, z, radius=0.07, half=0.008):
    """Centroid of the mesh slice at height z near (x, y): the middle of the limb there."""
    s = V[(np.abs(V[:, 2] - z) < half)]
    for r in (radius, radius * 1.5):
        near = s[np.hypot(s[:, 0] - x, s[:, 1] - y) < r]
        if len(near) >= 6:
            # the slice outline of a limb: the middle of its extent
            return np.array([0.5 * (near[:, 0].min() + near[:, 0].max()), 0.5 * (near[:, 1].min() + near[:, 1].max()), z])
    return np.array([x, y, z])


def leg_x_guess(V, y, z, side):
    """The X of a leg at (y, z): the slice blob on that side."""
    s = V[(np.abs(V[:, 2] - z) < 0.008) & (np.abs(V[:, 1] - y) < 0.05)]
    s = s[s[:, 0] * side > -0.02] if side > 0 else s[s[:, 0] < 0.02]
    if not len(s):
        return 0.1 * side
    # the blob farthest to that side
    xs = np.sort(s[:, 0] * side)[::-1]
    return float(np.median(xs[:max(3, len(xs) // 3)])) * side


def quadruped_skeleton(model: Model, lm: dict) -> dict:
    V = model.V
    J: dict = {}
    for name in ("Pelvis", "Spine", "Chest"):
        y, z = lm[name]
        J[name] = np.array([body_mid_x(V, y, 0.3, 0.7), y, z])
    J["Root"] = np.array([J["Pelvis"][0], 0.5 * (J["Pelvis"][1] + J["Chest"][1]), 0.0])
    for name in ("Neck", "Head"):
        y, z = lm[name]
        J[name] = np.array([body_mid_x(V, y, z - 0.08, z + 0.08), y, z])
    names = {"FL": ("Left", "UpperArm", "LowerArm", "Hand"), "FR": ("Right", "UpperArm", "LowerArm", "Hand"),
             "BL": ("Left", "UpperLeg", "LowerLeg", "Foot"), "BR": ("Right", "UpperLeg", "LowerLeg", "Foot")}
    for leg, (side, up, lo, end) in names.items():
        sg = 1.0 if side == "Left" else -1.0
        L = lm[leg]
        pts = {}
        for key in ("end", "mid"):
            y, z = L[key]
            x0 = leg_x_guess(V, y, z, sg)
            pts[key] = limb_centre(V, x0, y, z)
        y, z = L["tip"]
        pts["tip"] = np.array([pts["end"][0], y, z])
        y, z = L["top"]
        root = J["Chest"] if up == "UpperArm" else J["Pelvis"]
        # the shoulder / hip sits inside the body, between the leg and the midline
        pts["top"] = np.array([root[0] + 0.7 * (pts["mid"][0] - root[0]), y, z])
        J[f"{side}{up}"], J[f"{side}{lo}"], J[f"{side}{end}"] = pts["top"], pts["mid"], pts["end"]
        J[f"{side}{end}Tip"] = pts["tip"]
    return J


# ------------------------------------------------------------------ head
def head_points(V, J):
    """The head: past the plane through the head joint (normal along neck -> head), in front of
    the chest (never the tail or the rump)."""
    a, b = J["Neck"], J["Head"]
    d = (b - a) / np.linalg.norm(b - a)
    quad = abs(float(J["Head"][1] - J["Spine"][1])) > 0.1        # (a biped's head is over its spine)
    front = (V[:, 1] < J["Spine"][1]) if quad else np.ones(len(V), bool)
    return np.flatnonzero(((V - b) @ d > -0.01) & front & (np.linalg.norm(V - b, axis=1) < 1.0))


def rot(axis, deg):
    a = math.radians(deg)
    x, y, z = np.asarray(axis, float) / np.linalg.norm(axis)
    c, s, C = math.cos(a), math.sin(a), 1 - math.cos(a)
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def head_turn(V, J) -> tuple:
    """(yaw, roll, plane) - the rotation (degrees) that makes the head mirror-symmetric with the
    snout toward -Y (the inverse of the turn the source picture has), and the X of the head's mirror
    plane after it (relative to the head joint). Uses the face and skull only (ears flop
    asymmetrically) and a robust (median) mirror distance."""
    from scipy.spatial import cKDTree
    idx = head_points(V, J)
    P0 = V[idx] - J["Head"]
    zmax = float(P0[:, 2].max())
    P0 = P0[P0[:, 2] < 0.55 * zmax] if zmax > 0.05 else P0
    rng = np.random.default_rng(0)
    P = P0[rng.choice(len(P0), min(3000, len(P0)), replace=False)]

    def err(yaw, roll, want_c=False):
        Q = P @ (rot((0, 1, 0), roll) @ rot((0, 0, 1), yaw)).T
        tree = cKDTree(Q)
        mid = 0.5 * (np.percentile(Q[:, 0], 2) + np.percentile(Q[:, 0], 98))
        best = None
        for c in mid + np.arange(-0.06, 0.0601, 0.01):
            d, _ = tree.query(Q * np.array([-1, 1, 1]) + np.array([2 * c, 0, 0]))
            e = float(np.median(d))
            if best is None or e < best[0]:
                best = (e, c)
        return best if want_c else best[0]
    best = min(((err(y, r), y, r) for y in range(-80, 81, 4) for r in range(-40, 41, 4)))
    _, by, br = best
    fine = min(((err(by + dy, br + dr), by + dy, br + dr) for dy in np.arange(-4, 4.1, 0.5)
                for dr in np.arange(-4, 4.1, 0.5)))
    _, by, br = fine
    _, c = err(by, br, True)
    return float(by), float(br), float(c)


def straighten_head(V, J, yaw, roll, blend=0.6, centre=True, plane=None):
    """Rotate the head (and part of the neck, blended) by the correction - like posing the head
    bone and applying the pose as the rest pose - and (``centre``) move it over the chest's
    midline. Updates J["Head"] / J["Neck"] in place."""
    R = rot((0, 1, 0), roll) @ rot((0, 0, 1), yaw)
    a, b = J["Neck"], J["Head"]
    d = (b - a) / np.linalg.norm(b - a)
    t = (V - a) @ d / np.linalg.norm(b - a)                  # 0 at the neck joint, 1 at the head joint
    w = np.clip((t - (1 - blend)) / blend, 0, 1)
    w = w * w * (3 - 2 * w)
    near = np.zeros(len(V), bool)                            # only the head and neck (never a tail
    near[head_points(V, J)] = True                           # curling up behind it)
    near |= (np.linalg.norm(V - a, axis=1) < 1.2 * np.linalg.norm(b - a)) & (V[:, 1] < J["Spine"][1])
    w = w * near
    out = V.copy()
    for k in np.flatnonzero(w > 0):
        if w[k] >= 1:
            out[k] = b + R @ (V[k] - b)
    part = np.flatnonzero((w > 0) & (w < 1))
    for k in part:                                          # partial: a fraction of the rotation angle
        Rk = rot((0, 1, 0), roll * w[k]) @ rot((0, 0, 1), yaw * w[k])
        out[k] = b + Rk @ (V[k] - b)
    if centre:
        if plane is None:
            hp = out[head_points(out, J)]
            mid = 0.5 * (np.percentile(hp[:, 0], 2) + np.percentile(hp[:, 0], 98))
        else:
            mid = float(b[0] + plane)
        dx = float(J["Chest"][0] - mid)
        out[:, 0] += dx * w
        J["Head"] = np.array([J["Chest"][0], J["Head"][1], J["Head"][2]])     # on the face's mid plane
        J["Neck"] = np.array([J["Chest"][0], J["Neck"][1], J["Neck"][2]])
    return out


def ears_and_top(V, J, ears=True):
    """Head top (the skull between the ears) and each ear's base and tip, on a straight head."""
    hx = J["Head"][0]
    idx = head_points(V, J)
    P = V[idx]
    mid = P[np.abs(P[:, 0] - hx) < 0.025]
    skull = float(np.percentile(mid[:, 2], 99)) if len(mid) else float(P[:, 2].max())
    yc = float(np.median(P[P[:, 2] > skull - 0.06][:, 1])) if len(P) else J["Head"][1]
    out = {"HeadTop": np.array([hx, yc, skull])}
    if not ears:
        return out
    for side, sg in (("Left", 1.0), ("Right", -1.0)):
        e = P[((P[:, 0] - hx) * sg > 0.03) & (P[:, 2] > skull - 0.03)]
        if len(e) < 30:
            continue
        # the root: the ring where the ear leaves the skull (just around the skull-top height);
        # the tip: the ear point farthest from it (a floppy ear's tip need not be the highest)
        ring = e[np.abs(e[:, 2] - skull) < 0.03]
        root = ring.mean(axis=0) if len(ring) > 8 else e[np.argsort(e[:, 2])[:20]].mean(axis=0)
        far = np.linalg.norm(e - root, axis=1)
        tip = e[far >= np.percentile(far, 99)].mean(axis=0)
        out[f"{side}Ear"] = root + 0.12 * (tip - root)
        out[f"{side}EarTip"] = tip
    return out


def tail_chain(model: Model, J, base_yz, n=4, tip_yz=None):
    """Tail joints along the tail's centre line: a geodesic from the tip to the rump, re-centred in
    the cross-section at every joint."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    V, F = model.V, model.F
    by, bz = base_yz
    base = np.array([body_mid_x(V, by, bz - 0.08, bz + 0.08), by, bz])
    if tip_yz is not None:
        ty, tz = tip_yz
        near = np.flatnonzero((np.abs(V[:, 1] - ty) < 0.04) & (np.abs(V[:, 2] - tz) < 0.04))
        tip = int(near[np.argmin(np.abs(V[near, 0] - base[0]))]) if len(near) else \
            int(np.argmin(np.linalg.norm(V[:, 1:] - np.array([ty, tz]), axis=1)))
    else:
        cand = np.flatnonzero(V[:, 1] > by)
        tip = int(cand[np.argmax(np.linalg.norm(V[cand] - base, axis=1))])
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    w = np.linalg.norm(V[e[:, 0]] - V[e[:, 1]], axis=1)
    G = coo_matrix((w, (e[:, 0], e[:, 1])), shape=(len(V), len(V))).tocsr()
    G = G.maximum(G.T)
    start = int(np.argmin(np.linalg.norm(V - base, axis=1)))
    _, pred = dijkstra(G, indices=tip, return_predecessors=True)
    path = [start]
    while path[-1] != tip and pred[path[-1]] >= 0:
        path.append(int(pred[path[-1]]))
    pts = V[np.array(path)]                                       # base surface -> tip
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    total = float(seg[-1])
    chain = [base]
    for t in np.linspace(0, total, n + 1)[1:-1]:
        k = int(np.searchsorted(seg, t))
        q = pts[min(k, len(pts) - 1)]
        tan = pts[min(k + 3, len(pts) - 1)] - pts[max(k - 3, 0)]
        tan /= max(np.linalg.norm(tan), 1e-9)
        # the cross-section through q, perpendicular to the path: its centre
        sec = V[(np.abs((V - q) @ tan) < 0.01) & (np.linalg.norm(V - q, axis=1) < 0.2)]
        chain.append(0.5 * (sec.min(0) + sec.max(0)) if len(sec) > 8 else q)
    out = {f"Tail{k}": p for k, p in enumerate(chain)}
    out["TailTip"] = V[tip]
    return out


def straighten_lbs(model, J, yaw, roll, plane, cid):
    """Straighten the head like posing it in Blender and applying the pose as the rest pose:
    the Head bone turns by the correction (and moves onto the body's mid plane), the Neck by half
    of it, and the mesh follows by linear blend skinning with Blender bone-heat weights computed
    on the original mesh - smooth over the surface, so nothing tears or stretches."""
    import weights as Wt
    from joints import parents_for
    body = {k: v for k, v in J.items() if k in ("Root", "Pelvis", "Spine", "Chest", "Neck", "Head") or
            "Arm" in k or "Hand" in k or "Leg" in k or "Foot" in k}
    V = model.V
    hp = V[head_points(V, J)]
    body["HeadTop"] = np.array([J["Head"][0], J["Head"][1] - 0.02, float(hp[:, 2].max())])
    names, W = Wt.bone_heat(cid + "_pre", V, model.F, body, parents_for(body))
    ix = {n: i for i, n in enumerate(names)}
    R = rot((0, 1, 0), roll) @ rot((0, 0, 1), yaw)
    Rh = rot((0, 1, 0), roll * 0.5) @ rot((0, 0, 1), yaw * 0.5)
    a, b = J["Neck"].copy(), J["Head"].copy()
    mid_after = b[0] + plane                      # the face's mid plane after the turn (about b)
    dx = float(J["Chest"][0] - mid_after)
    shift_n = np.array([0.5 * dx, 0, 0])
    b_new = a + Rh @ (b - a) + shift_n
    shift_h = np.array([dx, 0, 0]) - (b_new - (a + Rh @ (b - a)))
    out = V * (1.0 - W[:, ix["Neck"]] - W[:, ix["Head"]])[:, None]
    out += W[:, ix["Neck"], None] * (a + (V - a) @ Rh.T + shift_n)
    out += W[:, ix["Head"], None] * (b_new + (V - b) @ R.T + shift_h)
    J["Head"] = np.array([J["Chest"][0], b_new[1], b_new[2]])
    J["Neck"] = np.array([J["Chest"][0] + 0.0, a[1], a[2]])
    return out
