"""Minimal BVH reader with vectorised forward kinematics (CMU / Hahne "MotionBuilder-friendly" files).

``load(path)`` returns a :class:`Motion` whose ``fk()`` gives every joint's world rotation and
position for every frame. Axes are the file's own (CMU: +Y up, the actor faces +Z, left = +X),
which happen to be the companion assets' axes too.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class BvhJoint:
    name: str
    parent: int
    offset: np.ndarray
    channels: list = field(default_factory=list)
    end: np.ndarray | None = None          # End Site offset (leaf bones)


@dataclass
class Motion:
    joints: list
    frame_time: float
    data: np.ndarray                       # (F, total channels)
    source: str = ""

    @property
    def frames(self) -> int:
        return int(self.data.shape[0])

    @property
    def fps(self) -> float:
        return 1.0 / self.frame_time

    def index(self, name: str) -> int:
        for i, j in enumerate(self.joints):
            if j.name == name:
                return i
        raise KeyError(name)

    def fk(self, frames: slice | np.ndarray | None = None, rest: bool = False):
        """World rotations (F, J, 3, 3) and positions (F, J, 3). ``rest`` = the zero-rotation pose."""
        data = self.data if frames is None else self.data[frames]
        if rest:
            data = np.zeros((1, self.data.shape[1]))
        F = data.shape[0]
        J = len(self.joints)
        R = np.zeros((F, J, 3, 3))
        P = np.zeros((F, J, 3))
        col = 0
        for i, j in enumerate(self.joints):
            loc = np.repeat(np.eye(3)[None], F, axis=0)
            pos = np.repeat(j.offset[None], F, axis=0).astype(float)
            for ch in j.channels:
                v = data[:, col]
                col += 1
                if ch.endswith("position"):
                    if not rest:
                        pos = pos.copy()
                        pos[:, "XYZ".index(ch[0])] = v + (j.offset["XYZ".index(ch[0])] if j.parent >= 0 else 0.0)
                else:
                    loc = loc @ _axis_rot(ch[0], np.radians(v))
            if j.parent < 0:
                R[:, i] = loc
                P[:, i] = pos
            else:
                R[:, i] = R[:, j.parent] @ loc
                P[:, i] = P[:, j.parent] + np.einsum("fij,fj->fi", R[:, j.parent], pos)
        return R, P

    def end_positions(self, R: np.ndarray, P: np.ndarray, name: str) -> np.ndarray:
        """World position of a joint's End Site (or of the joint itself when it has none)."""
        i = self.index(name)
        e = self.joints[i].end
        if e is None:
            return P[:, i]
        return P[:, i] + np.einsum("fij,j->fi", R[:, i], e)


def _axis_rot(axis: str, a: np.ndarray) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    if axis == "X":
        m = [[o, z, z], [z, c, -s], [z, s, c]]
    elif axis == "Y":
        m = [[c, z, s], [z, o, z], [-s, z, c]]
    else:
        m = [[c, -s, z], [s, c, z], [z, z, o]]
    return np.moveaxis(np.array(m), (0, 1), (-2, -1))


def load(path) -> Motion:
    tokens = Path(path).read_text(encoding="utf-8", errors="replace").split()
    k = 0
    joints: list[BvhJoint] = []
    stack: list[int] = []
    pending_end = False
    while k < len(tokens):
        t = tokens[k]
        if t in ("ROOT", "JOINT"):
            joints.append(BvhJoint(tokens[k + 1], stack[-1] if stack else -1, np.zeros(3)))
            k += 2
        elif t == "End":
            pending_end = True
            k += 2
        elif t == "{":
            stack.append(len(joints) - 1 if not pending_end else -2)
            k += 1
        elif t == "}":
            stack.pop()
            pending_end = False
            k += 1
        elif t == "OFFSET":
            off = np.array([float(x) for x in tokens[k + 1:k + 4]])
            if pending_end:
                joints[[s for s in stack if s >= 0][-1]].end = off
            else:
                joints[-1].offset = off
            k += 4
        elif t == "CHANNELS":
            n = int(tokens[k + 1])
            joints[-1].channels = tokens[k + 2:k + 2 + n]
            k += 2 + n
        elif t == "MOTION":
            break
        else:
            k += 1
    # fix parents recorded while an End Site block was open (never: End Sites have no children)
    frames = int(tokens[k + 2])
    ft = float(tokens[k + 5])
    vals = np.array([float(x) for x in tokens[k + 6:]], dtype=float)
    nch = sum(len(j.channels) for j in joints)
    data = vals[: frames * nch].reshape(frames, nch)
    return Motion(joints, ft, data, str(path))
