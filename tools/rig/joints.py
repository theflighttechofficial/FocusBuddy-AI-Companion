"""Skeleton placement for the companion models (rig frame, see core.py).

Joints are found from the shape (horizontal slices for the legs, vertical sections for the body,
the head mass, the ear tips, a geodesic path for the tail) and can be corrected per character in
``OVERRIDES`` (absolute positions or offsets, in the rig frame, height = 1). Every placement is
checked on overlay renders (front / side / top) before the skin weights are computed.
"""

from __future__ import annotations

import numpy as np

from core import ground_blobs

# canonical hierarchy (child -> parent); tips are not deforming bones (they give the last bone its
# direction); "Tail<k>" bones are added per model
PARENTS = {
    "Pelvis": "Root", "Spine": "Pelvis", "Chest": "Spine", "Neck": "Chest", "Head": "Neck", "HeadTop": "Head",
    "LeftEar": "Head", "LeftEarTip": "LeftEar", "RightEar": "Head", "RightEarTip": "RightEar",
    "LeftUpperArm": "Chest", "LeftLowerArm": "LeftUpperArm", "LeftHand": "LeftLowerArm", "LeftHandTip": "LeftHand",
    "RightUpperArm": "Chest", "RightLowerArm": "RightUpperArm", "RightHand": "RightLowerArm",
    "RightHandTip": "RightHand",
    "LeftUpperLeg": "Pelvis", "LeftLowerLeg": "LeftUpperLeg", "LeftFoot": "LeftLowerLeg", "LeftFootTip": "LeftFoot",
    "RightUpperLeg": "Pelvis", "RightLowerLeg": "RightUpperLeg", "RightFoot": "RightLowerLeg",
    "RightFootTip": "RightFoot",
}
TIPS = {"HeadTop", "LeftEarTip", "RightEarTip", "LeftHandTip", "RightHandTip", "LeftFootTip", "RightFootTip",
        "TailTip"}


def parents_for(J: dict) -> dict:
    par = {k: v for k, v in PARENTS.items() if k in J}
    tails = sorted(int(k[4:]) for k in J if k.startswith("Tail") and k[4:].isdigit())
    for i, t in enumerate(tails):
        par[f"Tail{t}"] = "Pelvis" if i == 0 else f"Tail{tails[i - 1]}"
    if "TailTip" in J and tails:
        par["TailTip"] = f"Tail{tails[-1]}"
    return par


def slice_pts(V, z, half=0.006):
    return V[np.abs(V[:, 2] - z) < half]


def trace_leg(V, paw_xy, z_from, z_to, step=0.01, radius=0.09):
    """Centre line of a leg: slice centroids from the paw upward while the slice around the line
    stays leg-sized; returns [(z, x, y, area-ish count)] and the height where it merges."""
    out = []
    cx, cy = paw_xy
    for z in np.arange(z_from, z_to, step):
        s = slice_pts(V, z)
        near = s[np.hypot(s[:, 0] - cx, s[:, 1] - cy) < radius]
        if len(near) < 5:
            break
        cx, cy = 0.6 * cx + 0.4 * near[:, 0].mean(), 0.6 * cy + 0.4 * near[:, 1].mean()
        out.append((z, near[:, 0].mean(), near[:, 1].mean(), np.ptp(near[:, 0]), np.ptp(near[:, 1]), len(near)))
    return np.array(out)


def quadruped(model, ov: dict) -> dict:
    V, H = model.V, 1.0
    J: dict = {}
    blobs = ground_blobs(V, 0.03)
    head_xy = V[V[:, 2] > 0.85][:, :2].mean(0)
    cand = [b for b in blobs if len(b) > 0.3 * len(blobs[0])]
    cents = sorted([b[:, :2].mean(0) for b in cand], key=lambda c: np.linalg.norm(c - head_xy))
    # four paws: the two pairs nearest the head (a tail tip on the ground is ignored)
    cents = sorted(cents[:4], key=lambda c: c[1])                     # front (-Y) first
    front, back = sorted(cents[:2], key=lambda c: -c[0]), sorted(cents[2:], key=lambda c: -c[0])  # left (+X) first
    paws = {"FL": front[0], "FR": front[1], "BL": back[0], "BR": back[1]}
    # body: the vertical extent at x = 0 along the length
    def body_at(y, half=0.02):
        s = V[(np.abs(V[:, 1] - y) < half) & (np.abs(V[:, 0]) < 0.05)]
        if len(s) < 4:
            return None
        zs = np.sort(s[:, 2])
        cuts = np.flatnonzero(np.diff(zs) > 0.03)
        runs = np.split(zs, cuts + 1)
        top = runs[-1]
        return float(runs[-1].min() if len(runs) > 1 else zs.min()), float(top.max())
    legs = {}
    for name, p in paws.items():
        legs[name] = trace_leg(V, p, 0.01, 0.8)
    model.debug = {"paws": paws, "legs": legs}
    y_front = float(np.mean([paws["FL"][1], paws["FR"][1]]))
    y_back = float(np.mean([paws["BL"][1], paws["BR"][1]]))
    # body bottom (belly) and top (back) above the front and back legs, between the legs
    def span(y):
        b = body_at(y)
        return b if b else (0.35, 0.6)
    fb, ft = span(y_front)
    bb, bt = span(y_back)
    model.debug.update(spans=((fb, ft), (bb, bt)))
    # legs merge into the body at the belly line
    for name, (lo_, top_) in (("FL", (fb, ft)), ("FR", (fb, ft)), ("BL", (bb, bt)), ("BR", (bb, bt))):
        pass
    return J
