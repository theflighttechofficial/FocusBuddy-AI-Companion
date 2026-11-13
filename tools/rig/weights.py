"""Skin weights: Blender's bone heat on the rig-frame mesh, then clean-up rules per body kind."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from core import BUILD
from joints import TIPS

AIM = {"Pelvis": "Spine", "Spine": "Chest", "Chest": "Neck", "Neck": "Head", "Head": "HeadTop",
       "LeftEar": "LeftEarTip", "RightEar": "RightEarTip",
       "LeftUpperArm": "LeftLowerArm", "LeftLowerArm": "LeftHand", "LeftHand": "LeftHandTip",
       "RightUpperArm": "RightLowerArm", "RightLowerArm": "RightHand", "RightHand": "RightHandTip",
       "LeftUpperLeg": "LeftLowerLeg", "LeftLowerLeg": "LeftFoot", "LeftFoot": "LeftFootTip",
       "RightUpperLeg": "RightLowerLeg", "RightLowerLeg": "RightFoot", "RightFoot": "RightFootTip",
       "Root": "Pelvis"}


def bone_table(J: dict, par: dict):
    """Blender bones for the deforming joints: head = joint, tail = its aim child."""
    names = [n for n in J if n not in TIPS]
    tails_ = []
    for n in names:
        if n in AIM and AIM[n] in J:
            t = J[AIM[n]]
        elif n.startswith("Tail"):
            k = int(n[4:])
            t = J.get(f"Tail{k + 1}", J.get("TailTip"))
        else:
            t = J[n] + np.array([0, 0, 0.05])
        if np.linalg.norm(t - J[n]) < 1e-4:
            t = J[n] + np.array([0, 0, 0.02])
        tails_.append(t)
    idx = {n: i for i, n in enumerate(names)}
    parents = [idx.get(par.get(n), -1) for n in names]
    deform = [n != "Root" for n in names]
    return names, np.array([J[n] for n in names]), np.array(tails_), np.array(parents), np.array(deform)


def bone_heat(cid: str, V, F, J, par) -> tuple[list, np.ndarray]:
    names, heads, tails_, parents, deform = bone_table(J, par)
    src = BUILD / f"{cid}_heat_in.npz"
    dst = BUILD / f"{cid}_heat_out.npz"
    np.savez(src, V=V, F=F, names=np.array(names), heads=heads, tails=tails_, parents=parents, deform=deform)
    r = subprocess.run([sys.executable, str(Path(__file__).with_name("blender_weights.py")), str(src), str(dst)],
                       capture_output=True, text=True)
    out = [l for l in (r.stdout + r.stderr).splitlines() if "weights:" in l or "rror" in l or "Heat" in l]
    print("\n".join(out))
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-2000:])
    return names, np.load(dst)["W"]


def smooth01(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def gate(W, names, i_from, mask_w, i_to):
    """Move (1 - mask_w) of bone i_from's weight to bone i_to."""
    moved = W[:, i_from] * (1.0 - mask_w)
    W[:, i_from] -= moved
    W[:, i_to] += moved


def fill_empty(V, W):
    """Vertices bone heat left without weights take their nearest weighted neighbours' weights."""
    from scipy.spatial import cKDTree
    empty = W.sum(1) < 1e-6
    if empty.any() and (~empty).any():
        d, idx = cKDTree(V[~empty]).query(V[empty], k=4)
        w = 1.0 / np.maximum(d, 1e-6)
        W[empty] = (W[~empty][idx] * w[..., None]).sum(1) / w.sum(1, keepdims=True)
    return W


def clean(V, F, W, names, J, kind, opts=None, col=None):
    W = fill_empty(V, W.copy())
    opts = opts or {}
    ix = {n: i for i, n in enumerate(names)}
    # ears: only beyond their root (never the face or the skull)
    for side in ("Left", "Right"):
        e = f"{side}Ear"
        if e in ix:
            a, b = J[e], J[f"{side}EarTip"]
            d = (b - a) / np.linalg.norm(b - a)
            t = (V - a) @ d
            gate(W, names, ix[e], smooth01((t + 0.01) / 0.03), ix["Head"])
    if kind == "quadruped":
        mid = float(J["Chest"][0])
        for side in ("Left", "Right"):
            for up, body in ((f"{side}UpperArm", "Chest"), (f"{side}UpperLeg", "Pelvis")):
                if up not in ix or body not in ix:
                    continue
                j = J[up]
                off = abs(j[0] - mid)
                # the upper leg owns the leg and the side of the body over it, not the chest front /
                # belly between the legs, nor the back above the joint
                lat = smooth01((np.abs(V[:, 0] - mid) - 0.35 * off) / (0.5 * off + 1e-6))
                high = 1.0 - smooth01((V[:, 2] - (j[2] + 0.02)) / 0.05)
                gate(W, names, ix[up], lat * high, ix[body])
    if kind == "biped":
        W = clean_biped(V, F, W, names, J, opts, col)
    W /= np.maximum(W.sum(1, keepdims=True), 1e-12)
    return W


def move_all(W, sel, frm: list, to: dict):
    """On the selected vertices, remove the weight of bones ``frm`` and give it to ``to``
    ({bone index: share})."""
    moved = W[np.ix_(sel, frm)].sum(1)
    W[np.ix_(sel, frm)] = 0.0
    for k, share in to.items():
        W[sel, k] += moved * share


def clean_biped(V, F, W, names, J, opts, col):
    ix = {n: i for i, n in enumerate(names)}
    limbs = [ix[n] for n in names if any(t in n for t in ("Arm", "Hand", "Leg", "Foot"))]
    arms = [ix[n] for n in names if "Arm" in n or "Hand" in n]
    legs = [ix[n] for n in names if "Leg" in n or "Foot" in n]
    hip_z = float(J["Pelvis"][2])
    # long hair (dark, behind the shoulders / below the head): head, neck and chest only
    if col is not None and opts.get("hair_lum"):
        lum = col @ np.array([0.299, 0.587, 0.114])
        hair = (lum < opts["hair_lum"]) & (V[:, 2] > hip_z) & (V[:, 1] > J["Chest"][1] - 0.02)
        move_all(W, np.flatnonzero(hair), limbs, {ix["Chest"]: 0.6, ix["Neck"]: 0.4})
    if opts.get("skirt"):
        # a saree: everything below the waist (skirt, pleats, the hanging pallu, the hem on the floor)
        # follows the hips and, more and more toward the hem, the thigh on its side - one smooth field,
        # so it sways instead of splitting; only the feet peeking out under the hem follow the feet
        below = np.flatnonzero(V[:, 2] < hip_z)
        t = np.clip((hip_z - V[below, 2]) / hip_z, 0, 1)
        side = 1.0 / (1.0 + np.exp(-(V[below, 0] - J["Pelvis"][0]) / 0.04))
        tgt = np.zeros((len(below), W.shape[1]))
        tgt[:, ix["Pelvis"]] = 1.0 - 0.55 * t
        # the pallu hanging behind the body follows the legs less (it would swing out)
        behind = smooth01((V[below, 1] - (J["Pelvis"][1] + 0.05)) / 0.05)
        legk = 0.55 * t * (1.0 - 0.7 * behind)
        tgt[:, ix["Pelvis"]] = 1.0 - legk
        tgt[:, ix["LeftUpperLeg"]] = legk * side
        tgt[:, ix["RightUpperLeg"]] = legk * (1 - side)
        for sd in (("Left", "Right") if opts.get("skirt_feet", False) else ()):
            a_, b_ = J[f"{sd}Foot"], J[f"{sd}FootTip"]
            ab = b_ - a_
            tt = np.clip(((V[below] - a_) @ ab) / float(ab @ ab), 0, 1)
            dist = np.linalg.norm(V[below] - (a_ + tt[:, None] * ab), axis=1)
            fw = 1.0 - smooth01((dist - 0.03) / 0.02)
            tgt *= (1 - fw)[:, None]
            tgt[:, ix[f"{sd}Foot"]] += fw
        blend = smooth01((hip_z - V[below, 2]) / 0.04)          # fade in under the waist
        W[below] = W[below] * (1 - blend)[:, None] + tgt * blend[:, None]
    # arms: only inside a capsule around each arm bone (the A-pose arms stand free of the body, so
    # anything farther - the pallu hanging beside the arm, a lock of hair, the side of the jacket -
    # stays with the body and never stretches into a sheet when the arm swings)
    caps = {"UpperArm": ("LowerArm", 0.055), "LowerArm": ("Hand", 0.045), "Hand": ("HandTip", opts.get("hand_r", 0.05))}
    for side in ("Left", "Right"):
        for part, (child, rad) in caps.items():
            n = f"{side}{part}"
            if n not in ix or f"{side}{child}" not in J:
                continue
            a, b = J[n], J[f"{side}{child}"]
            ab = b - a
            t = np.clip(((V - a) @ ab) / float(ab @ ab), -0.15, 1.15)
            d = np.linalg.norm(V - (a + t[:, None] * ab), axis=1)
            keep = 1.0 - smooth01((d - rad) / 0.015)
            gate(W, names, ix[n], keep, ix["Chest"])
    # arms never take the torso, legs never the torso above the hips
    torso = np.flatnonzero((np.abs(V[:, 0] - J["Chest"][0]) < 0.6 * abs(J["LeftUpperArm"][0] - J["Chest"][0])) &
                           (V[:, 2] > hip_z))
    move_all(W, torso, arms, {ix["Chest"]: 0.7, ix["Spine"]: 0.3})
    above = np.flatnonzero(V[:, 2] > hip_z + 0.04)
    move_all(W, above, legs, {ix["Pelvis"]: 1.0})
    return W


def smooth_weights(V, F, W, iters=3):
    """A few Laplacian passes over the surface (soft joints)."""
    from scipy.sparse import coo_matrix
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e = np.concatenate([e, e[:, ::-1]])
    A = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(V), len(V))).tocsr()
    deg = np.asarray(A.sum(1)).ravel()
    for _ in range(iters):
        W = 0.5 * W + 0.5 * (A @ W) / np.maximum(deg[:, None], 1)
    return W


def limit4(W):
    """Top four influences, renormalised; returns (joints (n, 4), weights (n, 4))."""
    top = np.argsort(-W, axis=1)[:, :4]
    tw = np.take_along_axis(W, top, axis=1)
    tw[tw < 0.01] = 0.0
    tw /= np.maximum(tw.sum(1, keepdims=True), 1e-12)
    return top, tw.astype(np.float32)
