"""Painted faces: blinking, squinting and an opening mouth drawn into the model's own texture.

The companion models have their eyes and mouth painted in the colour texture. A separate eyelid
mesh over a painted eye always looks stuck on, so the lids are painted instead: the build
(``tools/rig/face.py``) stores, for every texel of the eyes and the mouth, where it lies on the
face (its position in the eye's ellipse, or on the mouth), plus the skin / fur colour a closed lid
shows there (the face around the eye, painted over the eye). At runtime :class:`PaintedFace`
re-draws those texels for the current lid / squint / jaw values into the renderer's own copy of
the texture:

* the **upper lid** comes down over the eye from its top edge, with a dark lash line along its
  moving edge and a soft shadow just under it; fully closed, the lash line is the curve of the
  lower edge of the eye - a drawn closed eye;
* the **lower lid** (a happy squint) rises from the bottom in an upward arch (the "^^" of a
  cartoon smile);
* the **mouth** opens as a dark oval with a tongue under the lip line (the jaw morph also drops
  the chin).

Nothing changes while the values stay the same, so the texture is only rewritten (and re-uploaded
on the GPU path) during a blink or an expression change.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("focusbuddy.companion")


def _s(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class PaintedFace:
    STEP = 0.04      # values are quantised: a change smaller than this does not redraw
    R = 1.17         # the lid area: the fitted eye ellipse and its painted outline (data may override)
    LASH_R = 1.1     # the closed eye's lash line: the bottom of the eye (data may override)

    def __init__(self, path: Path):
        z = np.load(path)
        self.size = tuple(int(x) for x in z["size"])
        if "R" in z:
            self.R = float(z["R"])
        if "lash_r" in z:
            self.LASH_R = float(z["lash_r"])
        # "arch": a happy squint closes the eye completely and draws it as an upward arch (the
        # people's narrow painted eyes); "rise": the lower lid rises to a "^^" arch (the animals)
        self.squint_style = str(z["squint_style"]) if "squint_style" in z else "rise"
        self.e_idx = z["e_idx"].astype(np.int64)            # eye texels (flat index into the texture)
        self.e_xy = z["e_xy"].astype(np.float32)            # their place in the eye's ellipse (1 = the rim)
        self.e_fill = z["e_fill"].astype(np.float32)        # what a closed lid shows there
        self.e_eye = z["e_eye"].astype(np.int64)            # which eye
        self.lash = z["lash"].astype(np.float32)            # lash colour per eye
        # one texel in eye units, per eye: lines are never thinner than ~2 texels (no stair-steps)
        self.texel = z["texel"].astype(np.float32) if "texel" in z else np.zeros(len(self.lash), np.float32)
        self.m_idx = z["m_idx"].astype(np.int64) if "m_idx" in z else np.zeros(0, np.int64)
        self.m_xy = z["m_xy"].astype(np.float32) if "m_xy" in z else np.zeros((0, 2), np.float32)
        self.m_cols = z["m_cols"].astype(np.float32) if "m_cols" in z else np.zeros((3, 3), np.float32)
        # the open mouth's shape: half-width (mouth units) and how far below the lip line its middle
        # sits (fraction of its half-height: 1 = it hangs from the line, an animal's "o"; less = it
        # opens between the lips, a person's)
        self.m_shape = z["m_shape"].astype(np.float32) if "m_shape" in z else np.array([0.62, 1.0], np.float32)
        # 2.3, the people: "human" mouths open between the lips with soft edges and a hint of the
        # upper teeth; "skin" lids close with a soft crease above the lash line
        self.m_style = str(z["m_style"]) if "m_style" in z else "oval"
        self.m_open = float(z["m_open"]) if "m_open" in z else 0.55
        self.lid_style = str(z["lid_style"]) if "lid_style" in z else "fill"
        # "skin" lids: the eye opening itself (R_eye) closes; the ring of liner around it (out to R)
        # only fades into the lid skin as the eye shuts, so a closed eye has no outline
        self.R_eye = float(z["R_eye"]) if "R_eye" in z else None
        self.tex: Optional[np.ndarray] = None
        self.e_orig = self.m_orig = None
        self.state = (0.0, 0.0, 0.0)

    def bind(self, tex: np.ndarray) -> None:
        if tuple(tex.shape[:2]) != self.size:
            raise ValueError(f"painted face is for a {self.size} texture, got {tex.shape[:2]}")
        self.tex = tex
        flat = tex.reshape(-1, tex.shape[2])
        self.e_orig = self._living_eyes(flat[self.e_idx, :3].astype(np.float32))
        flat[self.e_idx, :3] = np.clip(self.e_orig + 0.5, 0, 255).astype(np.uint8)
        self.m_orig = flat[self.m_idx, :3].astype(np.float32)
        self.state = (0.0, 0.0, 0.0)

    # the key light's reflection in the eye (eye-ellipse units, +Y up): upper left of the viewer,
    # the same spot in both eyes, so the two reflections agree like real ones do
    CATCH = (-0.30, 0.34)
    CATCH2 = (0.28, -0.34)    # a small, faint second reflection (the fill light) low on the other side

    def _living_eyes(self, col: np.ndarray) -> np.ndarray:
        """Make the painted eyes read as wet, round eyeballs set into the head: a soft shadow from
        the upper lid across the top of the eye, a bright catchlight and a faint second reflection.
        Drawn once into the base colours, so a blinking lid covers them like the rest of the eye."""
        X, Y = self.e_xy[:, 0].astype(float), self.e_xy[:, 1].astype(float)
        out = col.astype(float).copy()
        inside = (X * X + Y * Y) < 1.0
        # the upper lid shades the top of the eyeball
        k = 0.16 * _s((Y - 0.30) / 0.60) * inside
        out *= (1.0 - k)[:, None]
        texel = self.texel[self.e_eye].astype(float)
        for (cx, cy), size, alpha in ((self.CATCH, 0.17, 0.88), (self.CATCH2, 0.09, 0.42)):
            rr = np.maximum(size, 1.6 * texel * size / 0.17)
            d = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) / rr
            a = alpha * _s((1.0 - d) / 0.5) * inside
            out = out + (255.0 - out) * a[:, None]
        return out.astype(np.float32)

    # ------------------------------------------------------------ drawing
    def eyes(self, blink: float, squint: float) -> np.ndarray:
        """Eye texels for these lid values. Coordinates are in units of the fitted eye ellipse.
        The lid area is the larger ellipse ``R`` (the eye with its painted outline; the stored fill
        meets the face seamlessly there). The upper lid's edge moves from the top of that area
        down to the bottom of the eye itself (``LASH_R``), where the closed eye's lash line lies;
        the outline band under it fades to the fill as the eye closes."""
        X, Y = self.e_xy[:, 0], self.e_xy[:, 1]
        R, Rl = self.R, self.LASH_R
        ring = None
        if self.lid_style == "skin" and self.R_eye:
            ring = ((X / R) ** 2 + (Y / R) ** 2 <= 1.0) & ((X / self.R_eye) ** 2 + (Y / self.R_eye) ** 2 > 1.0)
            R = self.R_eye
        if self.lid_style == "skin":
            # a person's resting smile does not lift the lower lids; a real laugh does
            squint = min(1.0, max(0.0, (squint - 0.25) / 0.75))
        top = R * np.sqrt(np.clip(1.0 - (X / R) ** 2, 0.0, 1.0))
        hb = np.sqrt(np.clip(1.0 - (X / Rl) ** 2, 0.0, 1.0))
        bot = -Rl * hb
        area = (X / R) ** 2 + (Y / R) ** 2 <= 1.0
        col = self.e_orig.copy()
        fill = self.e_fill
        # upper lid
        edge_u = top + (bot - top) * blink
        cover_u = area & (Y > edge_u)
        # lower lid (squint): rises from the bottom in an arch ("^^")
        if self.squint_style == "arch":
            arch = top                                   # the lower lid rises over the whole eye
        else:
            arch = Rl * (-0.25 + 0.75 * hb * hb)
        if self.lid_style == "skin":
            # a person's lower lid rises from the bottom of the eye itself (the wider lid area
            # only lets the upper lid cover the liner around the eye)
            low = bot - 0.3                              # starts just under the eye: a faint smile
            edge_l = low + squint * (arch - low)         # squint barely shows, a big one closes it
        else:
            edge_l = -top + squint * (arch + top)
        cover_l = area & (Y < edge_l) & (squint > 0.0)
        if self.lid_style == "skin":
            cover_l &= Y > bot - 0.32
        cover = cover_u | cover_l
        col[cover] = fill[cover]
        if ring is not None and blink > 0.4:
            k = _s((blink - 0.4) / 0.5)
            col[ring] = col[ring] * (1 - k) + fill[ring] * k
        # closing: the band of the eye's outline below the lash line fades into the lid / fur
        if blink > 0.5:
            band = area & ~cover & (Y < bot + 0.02)
            k = min(1.0, (blink - 0.5) * 2.0)
            col[band] = col[band] * (1 - k) + fill[band] * k
        lash = self.lash[self.e_eye]
        w = np.maximum(0.07 + 0.03 * hb, 1.1 * self.texel[self.e_eye])   # line half-width (eye units), tapered
        taper = _s(hb / 0.5)
        if blink > 0.02:
            # a soft shadow cast on the visible eye just under the lid, then the lash line
            sh = area & ~cover & (Y > edge_u - 0.35) & (np.abs(X) < Rl)
            k = (1.0 - (edge_u[sh] - Y[sh]) / 0.35) * 0.35 * min(1.0, blink * 4)
            col[sh] *= (1.0 - k)[:, None]
            d = np.abs(Y - edge_u)
            a = _s(1.0 - d / w) * taper * min(1.0, blink * 3)
            if self.lid_style == "skin":                # no lash line until the lid reaches the eye
                a = a * _s((Rl * hb + 0.15 - edge_u) / 0.3)
            col = col * (1 - a[:, None]) + lash * a[:, None]
        if self.lid_style == "skin" and blink > 0.35:
            # a closed eye has a soft crease in the lid, an arc above the lash line, and the lid
            # darkens a little toward the lashes (it curves away from the light)
            k = min(1.0, (blink - 0.35) / 0.5)
            crease = Rl * (0.05 + 0.42 * hb) + 0.12
            d = np.abs(Y - crease)
            a = _s(1.0 - d / (2.2 * w)) * _s(hb / 0.35) * 0.22 * k * cover_u
            col = col * (1 - a[:, None])
            near = cover_u & (Y < bot + 0.45) & (Y > bot)
            g = (1.0 - (Y[near] - bot[near]) / 0.45) * 0.10 * k
            col[near] *= (1.0 - g)[:, None]
        if squint > 0.02 and self.squint_style == "arch":
            # the closed happy eye: an upward arch through the eye's middle, its corners low
            curve = Rl * (-0.35 + 0.75 * hb ** 1.5)
            d = np.abs(Y - curve)
            k = min(1.0, max(0.0, (squint - 0.35) / 0.4))
            a = _s(1.0 - d / (1.15 * w)) * _s(hb / 0.3) * k * (Y < edge_l + 0.05)
            col = col * (1 - a[:, None]) + lash * a[:, None]
        elif squint > 0.02:
            d = np.abs(Y - edge_l) + (0.0 if self.lid_style != "skin" else 10.0 * (np.abs(X) > Rl))
            a = _s(1.0 - d / (0.7 * w)) * taper * min(1.0, squint * 3) * 0.75
            col = col * (1 - a[:, None]) + (lash * 0.8 + 30) * a[:, None]
        return col

    def mouth(self, jaw: float) -> np.ndarray:
        col = self.m_orig.copy()
        if jaw <= 0.02 or not len(self.m_idx):
            return col
        if self.m_style == "human":
            return self._human_mouth(col, jaw)
        X, Y = self.m_xy[:, 0], self.m_xy[:, 1]           # mouth units: X = 1 at the corners, Y = 1 a mouth-width down
        ry = 0.55 * jaw
        rx = float(self.m_shape[0]) * (0.55 + 0.45 * jaw)
        cy = ry * float(self.m_shape[1])                    # the oval hangs from / opens around the lip line
        r2 = (X / rx) ** 2 + ((Y - cy) / max(ry, 1e-3)) ** 2
        inside = r2 <= 1.0
        base, tongue, lip = self.m_cols[0], self.m_cols[1], self.m_cols[2]
        t = np.clip((Y - (cy - ry)) / max(2 * ry, 1e-3), 0, 1)
        tw = _s((t - 0.5) / 0.35) * _s((1.0 - np.abs(X / rx)) / 0.5)
        inner = base[None, :] * (1 - tw[:, None]) + tongue[None, :] * tw[:, None]
        col[inside] = inner[inside]
        ring = (r2 > 1.0) & (r2 < 1.35)
        a = _s(1.0 - (r2[ring] - 1.0) / 0.35) * 0.8
        col[ring] = col[ring] * (1 - a[:, None]) + lip[None, :] * a[:, None]
        return col

    def _human_mouth(self, col: np.ndarray, jaw: float) -> np.ndarray:
        """A person's open mouth: it parts along the lip line (which follows the smile), darkest
        at the back, the tongue low, a soft band of upper teeth, anti-aliased edges and a faint
        shadow of the lips around it."""
        X, Y = self.m_xy[:, 0].astype(np.float32), self.m_xy[:, 1].astype(np.float32)
        ry = self.m_open * jaw
        rx = float(self.m_shape[0]) * (0.6 + 0.4 * jaw)
        cy = ry * float(self.m_shape[1])
        r = np.sqrt((X / rx) ** 2 + ((Y - cy) / max(ry, 1e-3)) ** 2)
        inside = _s((1.0 - r) / 0.12)                                  # soft edge
        base, tongue = np.array([92.0, 34.0, 36.0]), np.array([196.0, 98.0, 100.0])
        t = np.clip((Y - (cy - ry)) / max(2 * ry, 1e-3), 0, 1)          # 0 top .. 1 bottom
        tw = _s((t - 0.55) / 0.35) * _s((1.0 - np.abs(X / rx)) / 0.6) * min(1.0, jaw * 1.5)
        inner = base[None, :] * (1 - tw[:, None]) + tongue[None, :] * tw[:, None]
        inner *= (0.85 + 0.3 * _s(np.abs(X / rx)))[:, None]            # deeper in the middle
        teeth = (1.0 - _s((t - 0.2) / 0.16)) * _s((0.8 - np.abs(X / rx)) / 0.3) * min(1.0, jaw * 2.5) * 0.9
        inner = inner * (1 - teeth[:, None]) + np.array([226.0, 216.0, 208.0])[None, :] * teeth[:, None]
        out = col * (1 - inside[:, None]) + inner * inside[:, None]
        ring = (r > 0.95) & (r < 1.35)
        a = _s(1.0 - (r[ring] - 0.95) / 0.4) * 0.18 * min(1.0, jaw * 2.0)
        out[ring] *= (1.0 - a)[:, None]
        return out

    def update(self, blink: float, squint: float, jaw: float) -> bool:
        """Redraw for these values (0..1); True when the texture changed."""
        if self.tex is None:
            return False
        q = tuple(round(float(np.clip(v, 0.0, 1.0)) / self.STEP) * self.STEP for v in (blink, squint, jaw))
        if q == self.state:
            return False
        flat = self.tex.reshape(-1, self.tex.shape[2])
        if q[0] != self.state[0] or q[1] != self.state[1]:
            flat[self.e_idx, :3] = np.clip(self.eyes(q[0], q[1]) + 0.5, 0, 255).astype(np.uint8)
        if q[2] != self.state[2] and len(self.m_idx):
            flat[self.m_idx, :3] = np.clip(self.mouth(q[2]) + 0.5, 0, 255).astype(np.uint8)
        self.state = q
        return True


def make_painted_face(asset, meta: dict) -> Optional[PaintedFace]:
    folder = getattr(asset, "path", None)
    if folder is None:
        return None
    folder = Path(folder)
    if folder.is_file():
        folder = folder.parent
    try:
        return PaintedFace(folder / meta.get("data", "face/painted.npz"))
    except Exception as e:  # noqa: BLE001 - a broken face never breaks the companion
        log.warning("Painted face for %s could not be loaded (%s)", getattr(asset, "id", "?"), e)
        return None
