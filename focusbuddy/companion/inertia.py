"""Pose inertialization: no bone ever jumps from one frame to the next.

The procedural rig makes many small decisions per frame - which paw waves, when a planted foot
lets go, when the tail is lifted off the floor, which way an elbow bends - and a clip can start
or stop on top of it. Each such decision can move a bone a long way in a single frame: on screen
that is a "cut" (an ear tip, a hand or a foot teleports).

:class:`PoseInertializer` runs on the final skeleton every frame. It tracks every joint's
*local* rotation (relative to its parent) and local offset, and predicts the next frame from the
last two. Continuous motion - even fast motion - matches the prediction and passes through
unchanged. When a joint breaks that prediction by more than a threshold (a discontinuity, not a
movement), the difference is captured as an offset that is added to the new pose and then decays
smoothly to zero (critically damped, half-life ``HALF_LIFE``). The pose still arrives where the
animation wants it, a few frames later, along a smooth path - the technique games use to blend
between animations without cross-fading ("inertialization").

Working in local space keeps the skeleton rigid: bone lengths never change during a blend.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


# ----------------------------------------------------------------- quaternion helpers (w, x, y, z)
def mat_to_quat(R: np.ndarray) -> np.ndarray:
    """(N, 3, 3) rotation matrices -> (N, 4) unit quaternions (w, x, y, z)."""
    R = np.asarray(R, dtype=float)
    n = R.shape[0]
    q = np.empty((n, 4))
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    # Shepperd: pick the numerically largest of w, x, y, z
    cand = np.stack([tr, R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]], axis=1)
    k = np.argmax(cand, axis=1)
    m = k == 0
    if m.any():
        s = np.sqrt(np.maximum(tr[m] + 1.0, 1e-12)) * 2
        q[m, 0] = 0.25 * s
        q[m, 1] = (R[m, 2, 1] - R[m, 1, 2]) / s
        q[m, 2] = (R[m, 0, 2] - R[m, 2, 0]) / s
        q[m, 3] = (R[m, 1, 0] - R[m, 0, 1]) / s
    m = k == 1
    if m.any():
        s = np.sqrt(np.maximum(1.0 + R[m, 0, 0] - R[m, 1, 1] - R[m, 2, 2], 1e-12)) * 2
        q[m, 0] = (R[m, 2, 1] - R[m, 1, 2]) / s
        q[m, 1] = 0.25 * s
        q[m, 2] = (R[m, 0, 1] + R[m, 1, 0]) / s
        q[m, 3] = (R[m, 0, 2] + R[m, 2, 0]) / s
    m = k == 2
    if m.any():
        s = np.sqrt(np.maximum(1.0 + R[m, 1, 1] - R[m, 0, 0] - R[m, 2, 2], 1e-12)) * 2
        q[m, 0] = (R[m, 0, 2] - R[m, 2, 0]) / s
        q[m, 1] = (R[m, 0, 1] + R[m, 1, 0]) / s
        q[m, 2] = 0.25 * s
        q[m, 3] = (R[m, 1, 2] + R[m, 2, 1]) / s
    m = k == 3
    if m.any():
        s = np.sqrt(np.maximum(1.0 + R[m, 2, 2] - R[m, 0, 0] - R[m, 1, 1], 1e-12)) * 2
        q[m, 0] = (R[m, 1, 0] - R[m, 0, 1]) / s
        q[m, 1] = (R[m, 0, 2] + R[m, 2, 0]) / s
        q[m, 2] = (R[m, 1, 2] + R[m, 2, 1]) / s
        q[m, 3] = 0.25 * s
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    return q


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((q.shape[0], 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], axis=1)


def quat_conj(q: np.ndarray) -> np.ndarray:
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def quat_log(q: np.ndarray) -> np.ndarray:
    """Unit quaternions -> rotation vectors (axis * angle), shortest way."""
    q = np.where(q[:, :1] < 0, -q, q)
    v = q[:, 1:]
    s = np.linalg.norm(v, axis=1)
    ang = 2.0 * np.arctan2(s, np.clip(q[:, 0], -1.0, 1.0))
    k = np.where(s > 1e-9, ang / np.maximum(s, 1e-12), 2.0)
    return v * k[:, None]


def quat_exp(r: np.ndarray) -> np.ndarray:
    """Rotation vectors -> unit quaternions."""
    ang = np.linalg.norm(r, axis=1)
    half = 0.5 * ang
    k = np.where(ang > 1e-9, np.sin(half) / np.maximum(ang, 1e-12), 0.5)
    return np.concatenate([np.cos(half)[:, None], r * k[:, None]], axis=1)


# ----------------------------------------------------------------- the filter
class PoseInertializer:
    """Removes single-frame discontinuities from a skeleton pose (see the module docstring).

    ``parent[i]`` is joint i's parent index (-1 for roots). ``exclude`` joints pass through
    untouched (eyes: a saccade is meant to be instant). ``scale`` is the skeleton's size in its
    own units (offset thresholds scale with it)."""

    HALF_LIFE = 0.075          # s - how fast a captured jump fades (gone after ~0.3 s)
    ROT_THRESH = math.radians(9.0)    # rotation error vs. prediction that counts as a jump
    POS_THRESH = 0.035         # offset error (fraction of ``scale``) that counts as a jump
    MAX_GAP = 0.25             # s - after a longer pause (hidden, asleep) start fresh
    EXTRAPOLATE = False        # continue the old velocity through a captured jump (off: hold, then ease)

    def __init__(self, parent: Iterable[int], exclude: Iterable[int] = (), scale: float = 100.0):
        self.parent = np.asarray(list(parent), dtype=int)
        self.n = len(self.parent)
        self.order = self._topo()
        self.keep = np.ones(self.n, bool)
        for i in exclude:
            if 0 <= i < self.n:
                self.keep[i] = False
        self.scale = max(1e-6, float(scale))
        self.enabled = True
        self.reset()

    def _topo(self) -> list:
        seen, out = set(), []

        def visit(i):
            if i in seen:
                return
            p = int(self.parent[i])
            if p >= 0:
                visit(p)
            seen.add(i)
            out.append(i)
        for i in range(self.n):
            visit(i)
        return out

    def reset(self) -> None:
        self._q1 = self._q2 = None        # last two target local rotations
        self._o1 = self._o2 = None        # last two target local offsets
        self._dt1 = None
        self.dq = None                    # current rotation offsets (rotation vectors)
        self.do = None                    # current offset offsets
        self.events = 0                   # jumps captured so far (diagnostics / tests)

    # ------------------------------------------------------------ space conversion
    def _local(self, D: np.ndarray, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = self.parent
        has = p >= 0
        PD = np.where(has[:, None, None], D[np.maximum(p, 0)], np.eye(3)[None])
        PT = np.where(has[:, None], T[np.maximum(p, 0)], 0.0)
        L = np.einsum("nji,njk->nik", PD, D)          # PD^T @ D
        o = np.einsum("nji,nj->ni", PD, T - PT)       # PD^T @ (T - PT)
        return L, o

    def _world(self, L: np.ndarray, o: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        D = np.empty_like(L)
        T = np.empty_like(o)
        for i in self.order:
            p = int(self.parent[i])
            if p < 0:
                D[i] = L[i]
                T[i] = o[i]
            else:
                D[i] = D[p] @ L[i]
                T[i] = T[p] + D[p] @ o[i]
        return D, T

    # ------------------------------------------------------------ per frame
    def apply(self, D: np.ndarray, T: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Filter one frame (world rotations ``D`` (N,3,3) and positions ``T`` (N,3))."""
        if not self.enabled or self.n == 0 or D.shape[0] != self.n:
            return D, T
        dt = float(dt)
        if dt <= 0.0:
            # same moment again (a repaint): show what was shown, don't advance the filter
            if self.dq is None:
                return D, T
            dt = 0.0
        # bones may carry a uniform scale (a clenched fist): filter the rotation, keep the scale
        sc = np.cbrt(np.abs(np.linalg.det(D)))
        sc = np.where(sc > 1e-9, sc, 1.0)
        self._sc = sc
        L, o = self._local(D / sc[:, None, None], T)
        q = mat_to_quat(L)
        if self._q1 is not None:
            q = np.where((np.sum(q * self._q1, axis=1) < 0)[:, None], -q, q)   # same hemisphere
        if self._q1 is None or dt > self.MAX_GAP:
            self._q2, self._q1, self._o2, self._o1 = None, q, None, o
            self._dt1 = dt if dt > 0 else None
            self.dq = np.zeros((self.n, 3))
            self.do = np.zeros((self.n, 3))
            return D, T
        if dt == 0.0:
            return self._compose(q, o, D, T)
        # prediction from the last two frames (constant velocity)
        if self._q2 is not None and self._dt1:
            vr = quat_log(quat_mul(quat_conj(self._q2), self._q1)) * (dt / self._dt1)
            vo = (self._o1 - self._o2) * (dt / self._dt1)
        else:
            vr = np.zeros((self.n, 3))
            vo = np.zeros((self.n, 3))
        # a prediction is only as good as a plausible speed (never extrapolate a glitch)
        k = max(1.0, dt * 60.0)
        sp = np.linalg.norm(vr, axis=1)
        cap = 0.6 * k
        vr = np.where((sp > cap)[:, None], vr * (cap / np.maximum(sp, 1e-12))[:, None], vr)
        q_pred = quat_mul(self._q1, quat_exp(vr))
        o_pred = self._o1 + vo
        err_r = quat_log(quat_mul(quat_conj(q_pred), q))           # target vs prediction
        err_o = o - o_pred
        # (k: allow more change per frame at lower frame rates - the same motion covers more per frame)
        jump_r = (np.linalg.norm(err_r, axis=1) > self.ROT_THRESH * k) & self.keep
        jump_o = (np.linalg.norm(err_o, axis=1) > self.POS_THRESH * self.scale * k) & self.keep
        # decay the running offsets
        f = 0.5 ** (dt / self.HALF_LIFE)
        dq = self.dq * f
        do = self.do * f
        q2_next, o2_next = self._q1.copy(), self._o1.copy()
        if jump_r.any():
            # output continues from where it was heading: offset = (prediction incl. old offset) - target
            # the output holds where it was (not extrapolated: a fast oscillation such as a wag would
            # overshoot) and eases from there to the new pose
            base = q_pred[jump_r] if self.EXTRAPOLATE else self._q1[jump_r]
            prev_out = quat_mul(base, quat_exp(dq[jump_r]))
            dq[jump_r] = quat_log(quat_mul(quat_conj(q[jump_r]), prev_out))
            # the jump itself is not a velocity: the history behind the new target keeps the old one
            q2_next[jump_r] = quat_mul(q[jump_r], quat_exp(-vr[jump_r]))
            self.events += int(jump_r.sum())
        if jump_o.any():
            base_o = o_pred[jump_o] if self.EXTRAPOLATE else self._o1[jump_o]
            do[jump_o] = (base_o + do[jump_o]) - o[jump_o]
            o2_next[jump_o] = o[jump_o] - vo[jump_o]
            self.events += int(jump_o.sum())
        # excluded joints never carry an offset
        dq[~self.keep] = 0.0
        do[~self.keep] = 0.0
        self.dq, self.do = dq, do
        self._q2, self._q1 = q2_next, q
        self._o2, self._o1 = o2_next, o
        self._dt1 = dt
        return self._compose(q, o, D, T)

    def _compose(self, q: np.ndarray, o: np.ndarray, D: np.ndarray, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        active = (np.abs(self.dq).max(axis=1) > 1e-5) | (np.abs(self.do).max(axis=1) > 1e-5 * self.scale)
        if not active.any():
            return D, T                     # nothing settling: the pose exactly as animated
        L = quat_to_mat(q)
        L[active] = quat_to_mat(quat_mul(q[active], quat_exp(self.dq[active])))
        R, T2 = self._world(L, o + self.do)
        return R * self._sc[:, None, None], T2

    @property
    def settling(self) -> bool:
        """True while a captured jump is still fading out."""
        return self.dq is not None and (float(np.abs(self.dq).max(initial=0.0)) > 1e-4 or
                                        float(np.abs(self.do).max(initial=0.0)) > 1e-4 * self.scale)


def joint_scale(B: np.ndarray) -> float:
    """Size of a skeleton (its bind positions' extent)."""
    if B is None or len(B) == 0:
        return 100.0
    ext = np.asarray(B).max(axis=0) - np.asarray(B).min(axis=0)
    return float(max(1e-3, np.max(ext)))


def optional_exclude(canon: dict, names: Iterable[str]) -> list:
    return [canon[n] for n in names if canon.get(n, -1) >= 0]


__all__ = ["PoseInertializer", "joint_scale", "mat_to_quat", "quat_to_mat", "optional_exclude"]
