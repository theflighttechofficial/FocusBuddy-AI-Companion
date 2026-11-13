"""Build ``assets/mocap/library.npz`` from CMU motion-capture BVH files.

    python -m tools.mocap.build_library --bvh DIR [--only clip ...]

DIR holds the CMU trials as ``SS_TT.bvh`` (the "MotionBuilder-friendly" BVH conversion by Bruce
Hahne, e.g. github.com/meshula/cmu-mocap ``data/0SS/SS_TT.bvh``). Every clip in ``clips.CLIPS`` is
cut out, resampled to 30 fps, turned to face +Z, given in-place root motion, looped seamlessly
when it is a cycle, and stored skeleton-independently (see ``focusbuddy/companion/mocap.py``).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from focusbuddy.companion.mocap import BONES, mats_to_quats, quats_to_mats  # noqa: E402
from tools.mocap import bvh  # noqa: E402
from tools.mocap.clips import CLIPS  # noqa: E402

FPS = 30.0
# library segment -> (capture joint whose rotation drives it, joint / "end" giving its rest direction)
SOURCE = {"Pelvis": ("LowerBack", "Spine"), "Spine": ("Spine", "Spine1"), "Chest": ("Spine1", "Neck1"),
          "Neck": ("Neck1", "Head"), "Head": ("Head", "end"),
          "LeftUpperArm": ("LeftArm", "LeftForeArm"), "LeftLowerArm": ("LeftForeArm", "LeftHand"),
          "LeftHand": ("LeftFingerBase", "LeftHandIndex1"),
          "RightUpperArm": ("RightArm", "RightForeArm"), "RightLowerArm": ("RightForeArm", "RightHand"),
          "RightHand": ("RightFingerBase", "RightHandIndex1"),
          "LeftUpperLeg": ("LeftUpLeg", "LeftLeg"), "LeftLowerLeg": ("LeftLeg", "LeftFoot"),
          "LeftFoot": ("LeftFoot", "LeftToeBase"),
          "RightUpperLeg": ("RightUpLeg", "RightLeg"), "RightLowerLeg": ("RightLeg", "RightFoot"),
          "RightFoot": ("RightFoot", "RightToeBase")}
MIRROR = {b: (b.replace("Left", "Right") if "Left" in b else b.replace("Right", "Left")) for b in BONES}


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def heading(Rh: np.ndarray) -> np.ndarray:
    """Yaw (rad) of the body per frame (0 = facing +Z): from the forward axis while it is level
    (standing, lying on the side) and from the left-right axis while that one is level (lying on
    the back or the front, where "forward" points at the ceiling)."""
    f = Rh[..., :, 2]
    lv = Rh[..., :, 0]
    yf = np.arctan2(f[..., 0], f[..., 2])
    yl = np.arctan2(-lv[..., 2], lv[..., 0])
    wf = np.hypot(f[..., 0], f[..., 2]) ** 4
    wl = np.hypot(lv[..., 0], lv[..., 2]) ** 4
    return np.arctan2(wf * np.sin(yf) + wl * np.sin(yl), wf * np.cos(yf) + wl * np.cos(yl))


def circ_mean(a: np.ndarray) -> float:
    return float(math.atan2(np.sin(a).mean(), np.cos(a).mean()))


def leg_length(m: bvh.Motion) -> float:
    off = {j.name: j.offset for j in m.joints}
    return float(-off["LeftUpLeg"][1] + np.linalg.norm(off["LeftLeg"]) + np.linalg.norm(off["LeftFoot"])
                 - off["LeftToeBase"][1])


def slerp_mats(A: np.ndarray, B: np.ndarray, u: np.ndarray) -> np.ndarray:
    qa, qb = mats_to_quats(A), mats_to_quats(B)
    d = np.sum(qa * qb, axis=-1, keepdims=True)
    qb = np.where(d < 0, -qb, qb)
    q = qa + (qb - qa) * u[..., None]
    return quats_to_mats(q)


def sample(m: bvh.Motion, t0: float, t1: float, speed: float):
    """FK of the trial between t0 and t1 (s), resampled to FPS after a speed change."""
    R, P = m.fk()
    n = m.frames
    dur = (t1 - t0) / speed
    F = max(2, int(round(dur * FPS)) + 1)
    src = t0 + np.arange(F) / FPS * speed
    fi = np.clip(src * m.fps, 0, n - 1)
    i0 = np.floor(fi).astype(int)
    i1 = np.minimum(i0 + 1, n - 1)
    u = fi - i0
    Ri = slerp_mats(R[i0], R[i1], np.repeat(u[:, None], R.shape[1], axis=1))
    Pi = P[i0] + (P[i1] - P[i0]) * u[:, None, None]
    return Ri, Pi


def pose_feature(m, R, P) -> np.ndarray:
    """Per-frame pose descriptor for loop matching: joint positions relative to the hips, in the
    hips' heading frame, plus their velocities."""
    h = m.index("Hips")
    yaw = heading(R[:, h])
    rel = P - P[:, h:h + 1]
    c, s = np.cos(-yaw), np.sin(-yaw)
    x = rel[..., 0] * c[:, None] + rel[..., 2] * s[:, None]
    z = -rel[..., 0] * s[:, None] + rel[..., 2] * c[:, None]
    pos = np.stack([x, rel[..., 1], z], -1).reshape(len(R), -1)
    vel = np.gradient(pos, axis=0) * 0.5
    return np.concatenate([pos, vel], axis=1)


def build_clip(spec: dict, bvh_dir: Path) -> dict:
    m = bvh.load(bvh_dir / f"{spec['src']}.bvh")
    m.data = m.data[1:]                       # frame 0 of the Hahne files is a T-pose reference
    speed = float(spec.get("speed", 1.0))
    t0, t1 = spec["t"]
    loop = bool(spec.get("loop", False))
    R, P = sample(m, t0, t1, speed)
    h = m.index("Hips")
    if loop:
        # find the end frame (within +-search s of t1) whose pose best matches the start
        search = int(round(float(spec.get("search", 0.25)) / speed * FPS))
        feat = pose_feature(m, R, P)
        F = len(R)
        lo = max(int(F * 0.5), F - 1 - search)
        best, bi = None, F - 1
        for e in range(lo, F):
            d = float(np.sum((feat[e] - feat[0]) ** 2))
            if best is None or d < best:
                best, bi = d, e
        R, P = R[:bi + 1], P[:bi + 1]
    F = len(R)
    # ---- heading: face +Z
    yaw = heading(R[:, h])
    ref_hips = None
    if spec.get("ref_t") is not None:
        # heading / origin taken from another moment of the same trial (clips that follow each
        # other - lie down, sleep, get up - share one frame of reference)
        Rr_, Pr_ = sample(m, float(spec["ref_t"]), float(spec["ref_t"]) + 0.3, 1.0)
        ref = circ_mean(heading(Rr_[:, h]))
        ref_hips = Pr_[0, h].copy()
    elif spec.get("face") == "start":
        ref = circ_mean(yaw[: max(1, int(0.3 * FPS))])
    elif spec.get("face") == "end":
        ref = circ_mean(yaw[-max(1, int(0.3 * FPS)):])
    else:
        ref = circ_mean(yaw)
    ref += math.radians(float(spec.get("yaw", 0.0)))
    Ry = rot_y(-ref)
    R = np.einsum("ik,fjkl->fjil", Ry, R)
    P = np.einsum("ik,fjk->fji", Ry, P)
    if ref_hips is not None:
        ref_hips = Ry @ ref_hips
    if spec.get("straighten"):
        # remove the slow heading drift of a long cycle (a walk along a curve)
        yaw = heading(R[:, h])
        drift = np.unwrap(yaw)
        drift = drift - drift.mean()
        for f in range(F):
            Rf = rot_y(-drift[f])
            R[f] = np.einsum("ik,jkl->jil", Rf, R[f])
            P[f] = (Rf @ (P[f] - P[f, h]).T).T + P[f, h]
    if spec.get("straight_legs"):
        # lying down: the legs stretch out along the body instead of the captured bent knees (a long
        # skirt fans up over bent knees; a sleeper looks calmer stretched out)
        k = float(spec["straight_legs"])
        lb = m.index("LowerBack")
        up = P[:, m.index("Spine1")] - P[:, h]
        up /= np.maximum(np.linalg.norm(up, axis=1, keepdims=True), 1e-9)
        w = k * np.clip((0.55 - up[:, 1]) / 0.35, 0.0, 1.0)
        R = R.copy()
        for name in ("LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase", "RightUpLeg", "RightLeg", "RightFoot",
                     "RightToeBase"):
            ji = m.index(name)
            R[:, ji] = slerp_mats(R[:, ji], R[:, lb], w)
    # ---- floor and leg length
    lt = m.end_positions(R, P, "LeftToeBase")
    rt = m.end_positions(R, P, "RightToeBase")
    floor = float(spec.get("floor", np.percentile(np.minimum(lt[:, 1], rt[:, 1]), 2)))
    floor -= 0.0
    L = leg_length(m)
    hips = P[:, h].copy()
    hips[:, 1] -= floor
    # ---- root motion
    root = spec.get("root", "inplace" if loop else "start")
    if root == "ref" and ref_hips is not None:
        hips[:, [0, 2]] -= ref_hips[[0, 2]]
    elif loop or root == "inplace":
        # remove the travel: a straight line from the first to the (looped) last frame
        trend = np.linspace(0.0, 1.0, F)[:, None] * (hips[-1] - hips[0])[None]
        hips[:, [0, 2]] -= trend[:, [0, 2]]
        hips[:, [0, 2]] -= hips[:, [0, 2]].mean(axis=0)
    elif root == "start":
        hips[:, [0, 2]] -= hips[0, [0, 2]]
    elif root == "end":
        hips[:, [0, 2]] -= hips[-1, [0, 2]]
    elif root == "feet":
        # keep the body over the feet: hips relative to the feet midpoint (standing gestures)
        mid = 0.5 * (P[:, m.index("LeftFoot")] + P[:, m.index("RightFoot")])
        hips[:, [0, 2]] -= mid[:, [0, 2]]
    if spec.get("center"):
        # a lying body is centred on its own length (head .. feet), not on the hips
        k = float(spec["center"])
        head = P[:, m.index("Head")]
        feet = 0.5 * (P[:, m.index("LeftFoot")] + P[:, m.index("RightFoot")])
        flat = np.clip((hips[:, 1] - 0.0) / max(1e-6, L), 0.0, 1.0)
        w = np.clip((0.75 - flat) / 0.5, 0.0, 1.0) * k            # 0 standing .. k lying
        mid = 0.5 * (head + feet)
        off = (mid[:, [0, 2]] - P[:, h][:, [0, 2]])
        hips[:, [0, 2]] -= off * w[:, None]
    if spec.get("lock_y") is not None:
        # jump in the air: the body's flight comes from the physics - keep the hips at most at
        # their standing height (the crouch and the landing squash stay)
        stand = float(spec["lock_y"]) * L
        hips[:, 1] = np.minimum(hips[:, 1], stand)
    events = {}
    if spec.get("events") == "jump":
        # takeoff / landing: both feet off the floor .. the first foot back on it
        toe = np.minimum(lt[:, 1], rt[:, 1]) - floor
        air = toe > 0.06 * L
        k = int(np.argmax(air)) if air.any() else 0
        e = k + int(np.argmax(~air[k:])) if air.any() else 0
        events = {"takeoff": k / FPS, "land": e / FPS}
    hips /= L
    # ---- segment rotations + rest directions
    Rr, Pr = m.fk(rest=True)
    q = np.zeros((F, len(BONES), 4))
    dirs = np.zeros((len(BONES), 3))
    for k, b in enumerate(BONES):
        j, c = SOURCE[b]
        ji = m.index(j)
        q[:, k] = mats_to_quats(R[:, ji])
        if c == "end":
            e = m.joints[ji].end
            dirs[k] = e if e is not None else np.array([0.0, 1.0, 0.0])
        else:
            dirs[k] = Pr[0, m.index(c)] - Pr[0, ji]
        dirs[k] /= np.linalg.norm(dirs[k]) or 1.0
    if spec.get("mirror"):
        q, dirs, hips = mirror(q, dirs, hips)
    if loop:
        q, hips = close_loop(q, hips)
    if spec.get("reverse"):
        q, hips = q[::-1].copy(), hips[::-1].copy()
    for f in range(1, len(q)):
        flip = np.sum(q[f] * q[f - 1], axis=-1) < 0
        q[f][flip] *= -1
    return {"q": q.astype(np.float32), "dir": dirs.astype(np.float32), "hips": hips.astype(np.float32),
            "fps": FPS, "loop": loop, "src": spec["src"], "t": [float(t0), float(t1)],
            "frames": int(len(q)), "events": events}


def mirror(q, dirs, hips):
    """Swap left and right (reflect across the body's mid plane, x -> -x)."""
    idx = [BONES.index(MIRROR[b]) for b in BONES]
    q2 = q[:, idx].copy()
    q2[..., 1] *= -1
    q2[..., 2] *= -1
    d2 = dirs[idx].copy()
    d2[:, 0] *= -1
    h2 = hips.copy()
    h2[:, 0] *= -1
    return q2, d2, h2


def close_loop(q, hips):
    """Spread the last-to-first frame mismatch over the cycle so it loops without a pop."""
    F = len(q)
    if F < 3:
        return q, hips
    Q = quats_to_mats(q)
    # residual rotation that takes the last frame onto the first, applied progressively
    res = np.einsum("bij,bkj->bik", Q[0], Q[-1])            # first @ last^T
    rq = mats_to_quats(res)
    ident = np.zeros_like(rq)
    ident[:, 3] = 1.0
    flip = np.sum(rq * ident, axis=-1) < 0
    rq[flip] *= -1
    u = np.linspace(0.0, 1.0, F)
    out = np.empty_like(Q)
    for f in range(F):
        r = quats_to_mats(ident + (rq - ident) * u[f])
        out[f] = r @ Q[f]
    hips = hips + np.linspace(0.0, 1.0, F)[:, None] * (hips[0] - hips[-1])[None]
    # the last frame now equals the first: drop it (the clip wraps from F-2 to 0)
    return mats_to_quats(out[:-1]), hips[:-1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bvh", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "assets" / "mocap")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    arrays, meta = {}, {}
    old = a.out / "library.npz"
    if a.only and old.is_file():
        with np.load(old) as z:
            arrays = {k: z[k] for k in z.files}
        meta = json.loads((a.out / "library.json").read_text(encoding="utf-8")).get("clips", {})
    for name, spec in CLIPS.items():
        if a.only and name not in a.only:
            continue
        c = build_clip(spec, a.bvh)
        for k in ("q", "dir", "hips"):
            arrays[f"{name}.{k}"] = c.pop(k)
        meta[name] = c
        print(f"{name:14s} {c['src']:7s} {c['frames']:4d} frames  loop={c['loop']}")
    np.savez_compressed(a.out / "library.npz", **arrays)
    doc = {"about": "Motion-capture clips from the CMU Graphics Lab Motion Capture Database "
                    "(mocap.cs.cmu.edu), BVH conversion by Bruce Hahne; see LICENSE.md.",
           "bones": list(BONES), "fps": FPS, "clips": dict(sorted(meta.items()))}
    (a.out / "library.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
