"""3D renderer: shaded, outlined 3D models for every built-in character.

How it works
------------
* **Real 3D geometry.**  Every character is assembled from smooth 3D primitives
  (rotated ellipsoids: torso, head, muzzle, ears, two-segment limbs, paws, tail
  chain, collar, visor, hair ...).  Their placement comes from the *same*
  skeleton the 2D renderer uses (``Pose.rig``: layout + IK-solved limbs +
  spring tail), with a lateral depth axis added - so planted feet, gaits,
  squash & stretch, every action and every emotion carry over unchanged.
* **Camera.**  Orthographic, slightly above (pitch), at a 3/4 angle (yaw).
  Turning around really rotates the model in 3D (through a front view), and
  the head turns toward the viewer while the body walks sideways.
* **Rasteriser.**  A small vectorised software renderer (NumPy): analytic
  ray / ellipsoid intersection per pixel, z-buffer, per-pixel normals, toon
  lighting (key light + soft fill + rim + specular), silhouette and contour
  outlines from depth / part discontinuities, 2x super-sampling.  No GPU,
  driver or OpenGL dependency - it runs identically on every Windows PC, in
  remote desktop sessions and in headless tests.
* **Face decals.**  Eyes, brows, mouths, cheeks, tears, whiskers are drawn as
  vector decals projected onto the head surface (foreshortened with the
  surface normal and hidden when occluded by the z-buffer), so all 2D
  expressions (angry lids, spiral eyes, heart eyes, shouting mouth ...) work
  in 3D.
* **Cost.**  A few ms per frame at the default size.  The companion redraws
  only when the pose changes and the character window only covers the model.

The renderer is a drop-in ``BaseRenderer`` (same ``paint`` / ``press_point``
API); ``make_renderer(spec, mode="3d")`` selects it, with automatic fallback to
the 2D renderer if NumPy is unavailable or rendering fails.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen

from .animation import Pose
from .characters import CharacterSpec
from .renderer import GROUND_MARGIN, UNITS, BaseRenderer, _c, _clamp
from .rig import RigPose

BASE_YAW = 22.0          # degrees: walking right shows the near side + a bit of the face
PITCH = 11.0             # camera looks slightly down
XCOMP = 1.0 / math.cos(math.radians(BASE_YAW))   # keeps planted feet planted on screen at the base yaw
LIGHT = np.array([-0.45, -0.72, 0.53])
LIGHT = LIGHT / np.linalg.norm(LIGHT)
HALF = (LIGHT + np.array([0.0, 0.0, 1.0]))
HALF = HALF / np.linalg.norm(HALF)


def _rgb(hex_color: str) -> tuple[float, float, float]:
    c = QColor(hex_color)
    return (c.redF(), c.greenF(), c.blueF())


def rot_z(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_y(deg: float) -> np.ndarray:
    """Model -> camera yaw: +x (forward) gains +z (toward the viewer) for positive angles."""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])


def rot_x(deg: float) -> np.ndarray:
    """Camera pitch: points above the origin (negative y) come toward the viewer."""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])


def frame_along(d, up_hint=(0.0, 0.0, 1.0)) -> np.ndarray:
    """3x3 axes (columns) whose first axis is ``d`` (pure-Python math: called ~40x per frame)."""
    dx, dy, dz = float(d[0]), float(d[1]), float(d[2])
    n = math.sqrt(dx * dx + dy * dy + dz * dz)
    if n < 1e-9:
        dx, dy, dz, n = 1.0, 0.0, 0.0, 1.0
    dx, dy, dz = dx / n, dy / n, dz / n
    hx, hy, hz = float(up_hint[0]), float(up_hint[1]), float(up_hint[2])
    if abs(dx * hx + dy * hy + dz * hz) > 0.95:
        hx, hy, hz = (0.0, 1.0, 0.0) if abs(dy) < 0.9 else (1.0, 0.0, 0.0)
    k = dx * hx + dy * hy + dz * hz
    ax, ay, az = hx - dx * k, hy - dy * k, hz - dz * k
    m = math.sqrt(ax * ax + ay * ay + az * az) or 1.0
    ax, ay, az = ax / m, ay / m, az / m
    # a2 = a3 x d
    bx, by, bz = ay * dz - az * dy, az * dx - ax * dz, ax * dy - ay * dx
    return np.array([[dx, bx, ax], [dy, by, ay], [dz, bz, az]])


@dataclass
class Prim:
    c: np.ndarray            # centre (model space)
    ax: np.ndarray           # 3x3 axes (columns)
    r: np.ndarray            # radii along the axes
    col: tuple               # rgb 0..1
    grp: int                 # parts in one group don't outline against each other
    em: bool = False         # emissive (LEDs, glowing eyes)
    spec: float = 0.25       # specular strength


class Raster:
    """Vectorised ellipsoid rasteriser with z-buffer, toon shading and outlines."""

    def __init__(self, ss: int = 2):
        self.ss = ss
        self.depth = None
        self.origin = (0.0, 0.0)
        self.scale = 1.0

    def render(self, prims: list[Prim], cam: np.ndarray, scale: float, outline_rgb: tuple,
               outline_px: float = 1.15) -> tuple[np.ndarray, tuple[int, int]]:
        """Returns (premultiplied BGRA uint8 image HxWx4, (x0, y0) top-left in units*scale pixels)."""
        ss = self.ss
        S = scale * ss
        cams = []
        lo = np.array([1e9, 1e9])
        hi = np.array([-1e9, -1e9])
        for p in prims:
            C = cam @ p.c
            A = cam @ p.ax
            hx = math.sqrt(float(np.sum((A[0, :] * p.r) ** 2)))
            hy = math.sqrt(float(np.sum((A[1, :] * p.r) ** 2)))
            cams.append((C, A, hx, hy))
            lo = np.minimum(lo, [C[0] - hx, C[1] - hy])
            hi = np.maximum(hi, [C[0] + hx, C[1] + hy])
        pad = 3.0 / scale + outline_px * 2 / scale
        ox = math.floor((lo[0] - pad) * scale)
        oy = math.floor((lo[1] - pad) * scale)
        W = (math.ceil((hi[0] + pad) * scale) - ox) * ss
        H = (math.ceil((hi[1] + pad) * scale) - oy) * ss
        W, H = max(ss, W), max(ss, H)
        zb = np.full((H, W), -1e9, dtype=np.float32)
        idx = np.full((H, W), -1, dtype=np.int32)
        nxb = np.zeros((H, W), dtype=np.float32)
        nyb = np.zeros((H, W), dtype=np.float32)
        nzb = np.ones((H, W), dtype=np.float32)
        for i, (p, (C, A, hx, hy)) in enumerate(zip(prims, cams)):
            x0 = max(0, int((C[0] - hx) * S - ox * ss))
            x1 = min(W, int((C[0] + hx) * S - ox * ss) + 2)
            y0 = max(0, int((C[1] - hy) * S - oy * ss))
            y1 = min(H, int((C[1] + hy) * S - oy * ss) + 2)
            if x1 <= x0 or y1 <= y0:
                continue
            du = ((np.arange(x0, x1) + 0.5) / S + ox / scale - C[0])[None, :]
            dv = ((np.arange(y0, y1) + 0.5) / S + oy / scale - C[1])[:, None]
            M = A.T / p.r[:, None]                              # camera -> unit sphere
            m = M[:, 2]
            a = float(m @ m)
            Gm = M.T @ M                                        # quadratic form of the ellipsoid
            bu, bv = 2.0 * float(M[:, 0] @ m), 2.0 * float(M[:, 1] @ m)
            # b = bu*du + bv*dv ; c = q0.q0 - 1 with q0 = M0*du + M1*dv
            b = bu * du + bv * dv
            c = (Gm[0, 0] * du) * du + (2.0 * Gm[0, 1] * du) * dv + (Gm[1, 1] * dv) * dv - 1.0
            disc = b * b - (4.0 * a) * c
            hit = disc >= 0.0
            if not hit.any():
                continue
            w = (np.sqrt(np.maximum(disc, 0.0)) - b) * (0.5 / a)
            z = w + C[2]
            sub = zb[y0:y1, x0:x1]
            upd = hit & (z > sub)
            ys, xs = np.nonzero(upd)
            if ys.size == 0:
                continue
            ws = w[ys, xs]
            dus = du[0, xs]
            dvs = dv[ys, 0]
            nx = Gm[0, 0] * dus + Gm[0, 1] * dvs + Gm[0, 2] * ws
            ny = Gm[1, 0] * dus + Gm[1, 1] * dvs + Gm[1, 2] * ws
            nz = Gm[2, 0] * dus + Gm[2, 1] * dvs + Gm[2, 2] * ws
            inv = 1.0 / np.maximum(np.sqrt(nx * nx + ny * ny + nz * nz), 1e-6)
            Y, X = ys + y0, xs + x0
            zb[Y, X] = C[2] + ws
            idx[Y, X] = i
            nxb[Y, X] = nx * inv
            nyb[Y, X] = ny * inv
            nzb[Y, X] = nz * inv

        fg = idx >= 0
        cols = np.array([p.col for p in prims], dtype=np.float32)
        ems = np.array([p.em for p in prims])
        specs = np.array([p.spec for p in prims], dtype=np.float32)
        grps = np.array([p.grp for p in prims] + [-1], dtype=np.int32)
        # shade only the covered pixels (float32, gathered)
        sel = np.flatnonzero(fg)
        ii_s = idx.ravel()[sel]
        nx_, ny_, nz_ = nxb.ravel()[sel], nyb.ravel()[sel], nzb.ravel()[sel]
        L0, L1, L2 = float(LIGHT[0]), float(LIGHT[1]), float(LIGHT[2])
        H0, H1, H2 = float(HALF[0]), float(HALF[1]), float(HALF[2])
        ndl = nx_ * L0 + ny_ * L1 + nz_ * L2
        t = np.clip((ndl - 0.02) * (1 / 0.32), 0.0, 1.0)
        toon = t * t * (3.0 - 2.0 * t)
        shade = 0.6 + 0.3 * toon + 0.12 * np.clip(ndl, 0.0, 1.0) + np.clip(-ndl, 0.0, 1.0) * 0.08
        r1 = np.clip(1.0 - nz_, 0.0, 1.0)
        rim = r1 * r1 * r1 * 0.22
        ndh = np.clip(nx_ * H0 + ny_ * H1 + nz_ * H2, 0.0, 1.0)
        ndh2 = ndh * ndh
        ndh4 = ndh2 * ndh2
        ndh8 = ndh4 * ndh4
        spec = ndh8 * ndh8 * ndh8 * ndh4 * specs[ii_s] * (toon > 0.5)
        base = cols[ii_s]
        col_s = base * shade[:, None] + (rim + spec)[:, None]
        em = ems[ii_s]
        if em.any():
            col_s[em] = base[em]
        np.clip(col_s, 0.0, 1.0, out=col_s)
        rgb = np.zeros((H * W, 3), dtype=np.float32)
        rgb[sel] = col_s
        rgb = rgb.reshape(H, W, 3)
        ii = np.where(fg, idx, len(prims))

        # outlines: silhouette (outside) + contours where parts overlap with a depth step
        k = max(1, int(round(outline_px * ss)))
        g = grps[ii]
        edge = np.zeros_like(fg)
        dz = 2.2
        for sy, sx in ((0, 1), (1, 0), (1, 1), (1, -1)):
            if sx < 0:
                A_ = (slice(sy, H), slice(0, W + sx))
                B_ = (slice(0, H - sy), slice(-sx, W))
            else:
                A_ = (slice(sy, H), slice(sx, W))
                B_ = (slice(0, H - sy), slice(0, W - sx))
            a_g, b_g, a_z, b_z = g[A_], g[B_], zb[A_], zb[B_]
            both = (a_g >= 0) & (b_g >= 0) & (a_g != b_g) & (np.abs(a_z - b_z) > dz)
            if not both.any():
                continue
            # paint the contour on the farther side so the nearer part keeps its full shape
            edge[A_] |= both & (a_z < b_z)
            edge[B_] |= both & (b_z <= a_z)
        if k > 1:
            edge = _dilate(edge, k // 2) & fg
        sil = _dilate(fg, k) & ~fg
        oc = np.array(outline_rgb, dtype=np.float32)
        rgb[edge] = oc
        rgb[sil] = oc
        alpha = (fg | sil).astype(np.float32)
        # premultiply + downsample
        if ss > 1:
            Hs, Ws = H // ss, W // ss
            rgb = rgb[:Hs * ss, :Ws * ss]
            alpha = alpha[:Hs * ss, :Ws * ss]
            prem = rgb * alpha[..., None]
            acc = np.zeros((Hs, Ws, 3), dtype=np.float32)
            acca = np.zeros((Hs, Ws), dtype=np.float32)
            for oy_ in range(ss):
                for ox_ in range(ss):
                    acc += prem[oy_::ss, ox_::ss]
                    acca += alpha[oy_::ss, ox_::ss]
            inv_n = 1.0 / (ss * ss)
            prem = acc * inv_n
            alpha = acca * inv_n
            zs = zb[::ss, ::ss][:Hs, :Ws]
        else:
            prem = rgb * alpha[..., None]
            zs = zb
        out = np.empty(prem.shape[:2] + (4,), dtype=np.uint8)
        out[..., 0] = (prem[..., 2] * 255 + 0.5).astype(np.uint8)
        out[..., 1] = (prem[..., 1] * 255 + 0.5).astype(np.uint8)
        out[..., 2] = (prem[..., 0] * 255 + 0.5).astype(np.uint8)
        out[..., 3] = (alpha * 255 + 0.5).astype(np.uint8)
        self.depth = zs
        self.origin = (ox, oy)
        self.scale = scale
        return out, (ox, oy)

    def depth_at(self, u: float, v: float) -> float:
        """Front-most depth (camera space units) at a projected point, or -inf if empty."""
        if self.depth is None:
            return -1e9
        x = int(u * self.scale - self.origin[0])
        y = int(v * self.scale - self.origin[1])
        h, w = self.depth.shape
        if 0 <= x < w and 0 <= y < h:
            return float(self.depth[y, x])
        return -1e9


def _dilate(mask: np.ndarray, k: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(k):
        m = out.copy()
        m[1:, :] |= out[:-1, :]
        m[:-1, :] |= out[1:, :]
        m[:, 1:] |= out[:, :-1]
        m[:, :-1] |= out[:, 1:]
        out = m
    return out


# =========================================================================
class Renderer3D(BaseRenderer):
    """Shaded 3D models for quadrupeds (cat, dog, fox, rabbit) and bipeds (robot, study buddy)."""

    QUALITY_SS = {"low": 1, "medium": 2, "normal": 2, "high": 3, "ultra": 4}
    # LOW: simple lighting (no highlights, thin outline, no contact shadow)
    # MEDIUM: normal lighting + shadows          HIGH: sharper (3x supersampling) + bolder outlines
    # ULTRA: 4x supersampling, glossier highlights, soft double contact shadow
    TIERS = {
        "low": {"ss": 1, "spec": 0.0, "shadow": False, "outline": 0.85, "soft_shadow": False},
        "medium": {"ss": 2, "spec": 1.0, "shadow": True, "outline": 1.15, "soft_shadow": False},
        "high": {"ss": 3, "spec": 1.0, "shadow": True, "outline": 1.35, "soft_shadow": True},
        "ultra": {"ss": 4, "spec": 1.3, "shadow": True, "outline": 1.4, "soft_shadow": True},
    }

    @staticmethod
    def tier_name(quality: str) -> str:
        q = str(quality).lower()
        return "medium" if q in ("normal", "medium") else q if q in ("low", "high", "ultra") else "medium"

    def __init__(self, spec: CharacterSpec, quality: str = "normal", use_gpu: bool = True):
        super().__init__(spec)
        self.quality = quality
        self.tier = self.TIERS[self.tier_name(quality)]
        self.raster = Raster(self.tier["ss"])
        self.orbit: dict | None = None      # Playground orbit camera {"yaw", "pitch", "zoom"}; None = desktop view
        self.stats = {"total_ms": 0.0, "raster_ms": 0.0, "build_ms": 0.0, "decal_ms": 0.0, "prims": 0,
                      "draw_calls": 0, "backend": "cpu"}
        self.kind = self.rig.sk.kind
        sp = spec.species
        self.species = sp if sp in ("cat", "dog", "fox", "rabbit", "robot", "buddy") else (
            "buddy" if self.kind == "biped" else "cat")
        pal = spec.palette
        self.cols = {k: _rgb(getattr(pal, k)) for k in ("main", "shade", "light", "outline", "accent", "extra")}
        self.outline_rgb = tuple(v * 0.85 for v in self.cols["outline"])
        self._cache_key = None
        self._cache = None
        self.last_ms = 0.0
        self.backend = "cpu"
        self._gl = None
        self._cp: list = []
        if use_gpu:
            try:
                from .render3d_gl import GLRaster
                self._gl = GLRaster.create()
            except Exception:  # noqa: BLE001
                self._gl = None
            if self._gl is not None:
                self.backend = "gpu"

    def set_quality(self, quality: str) -> None:
        self.quality = quality
        self.tier = self.TIERS[self.tier_name(quality)]
        self.raster.ss = self.tier["ss"]
        self._cache_key = None

    def set_orbit(self, yaw: float = 0.0, pitch: float = 0.0, zoom: float = 1.0) -> None:
        """Playground-only orbit camera (degrees added to the normal view; zoom scales the model)."""
        self.orbit = {"yaw": float(yaw), "pitch": max(-60.0, min(70.0, float(pitch))),
                      "zoom": max(0.4, min(2.5, float(zoom)))}

    def reset_orbit(self) -> None:
        self.orbit = None

    # ------------------------------------------------------------- camera
    def camera(self, pose: Pose) -> tuple[np.ndarray, float]:
        t = max(-1.0, min(1.0, pose.turn))
        yaw = 90.0 - (90.0 - BASE_YAW) * t
        back = _clamp(getattr(pose, "back", 0.0))
        if back > 0:                       # sulking: rotate on past the profile until the back faces us
            yaw -= 112.0 * back * (1 if t >= 0 else -1)
        pitch = PITCH
        if self.orbit:
            yaw += self.orbit["yaw"]
            pitch += self.orbit["pitch"]
        cam = rot_x(pitch) @ rot_y(yaw)
        return cam, yaw

    def _model(self, pose: Pose):
        rp = self.rig_pose(pose)
        if self.kind == "biped":
            prims, face = self._build_biped(pose, rp)
        else:
            prims, face = self._build_quad(pose, rp)
        return rp, prims, face

    def _post(self, pose: Pose, prims: list[Prim], face: dict) -> np.ndarray:
        """Squash & stretch + whole-body roll applied to the model (around the feet / hips)."""
        sq = pose.squash
        S = np.diag([XCOMP * (1 + 0.08 * sq), 1 - 0.12 * sq, 1 + 0.08 * sq])
        R = np.eye(3)
        piv = np.array([0.0, -30.0 - pose.lift, 0.0])
        if pose.roll:
            R = rot_z(pose.roll)
        return S, R, piv

    def to_cam(self, pose: Pose, cam: np.ndarray, pts: np.ndarray, S, R, piv) -> np.ndarray:
        p = np.asarray(pts, dtype=float)
        if pose.roll:
            p = (R @ (p - piv).T).T + piv if p.ndim == 2 else R @ (p - piv) + piv
        p = (S @ p.T).T if p.ndim == 2 else S @ p
        return (cam @ p.T).T if p.ndim == 2 else cam @ p

    # --------------------------------------------------------------- paint
    def press_point(self) -> tuple[float, float]:
        """Projected paw / hand tip while pressing (default camera, facing right)."""
        x, y = self.rig.press_point()
        z = self._limb_z("FR" if self.kind != "biped" else "arm_R")
        p = Pose()
        cam, _ = self.camera(p)
        if self.kind == "biped":
            x += self._biped_xshift("arm_R")
        v = cam @ (np.diag([XCOMP, 1.0, 1.0]) @ np.array([x, y, z]))
        return (float(v[0]), float(v[1]))

    def paint(self, g: QPainter, rect: QRectF, pose: Pose, facing: float = 1, state: str | None = None,
              debug: dict | None = None) -> None:
        t0 = time.perf_counter()
        self._decal_count = 0
        g.save()
        g.setRenderHint(QPainter.Antialiasing, True)
        g.setRenderHint(QPainter.SmoothPixmapTransform, True)
        s = rect.width() / UNITS
        tier = self.tier
        if self.orbit:
            s *= self.orbit["zoom"]
            g.translate(rect.center().x(), rect.bottom() - GROUND_MARGIN * s + rect.height() * 0.42 *
                        (self.orbit["zoom"] - 1.0))
        else:
            g.translate(rect.center().x(), rect.bottom() - GROUND_MARGIN * s)
        g.scale(s, s)
        if tier["shadow"]:
            if tier["soft_shadow"]:          # ULTRA: soft penumbra under the contact shadow
                g.save()
                g.scale(1.35, 1.25)
                g.setOpacity(0.45)
                self._shadow(g, pose)
                g.restore()
            self._shadow(g, pose)
        t_build = time.perf_counter()
        rp, prims, face = self._model(pose)
        cam, yaw = self.camera(pose)
        S, R, piv = self._post(pose, prims, face)
        # transform prims into camera-ready model space (squash / roll) and render
        tprims = []
        for p in prims:
            c = p.c
            ax = p.ax
            if pose.roll:
                c = R @ (c - piv) + piv
                ax = R @ ax
            c = S @ c
            ax2 = S @ ax
            norms = np.linalg.norm(ax2, axis=0)
            tprims.append(Prim(c, ax2 / norms, p.r * norms, p.col, p.grp, p.em, p.spec))
        # camera-space primitives (also used for analytic decal occlusion)
        self._cp = []
        for p in tprims:
            C = cam @ p.c
            A = cam @ p.ax
            self._cp.append((C, A, p.r, p.col, p.em, min(1.0, p.spec * tier["spec"]), A.T / p.r[:, None]))
        t_raster = time.perf_counter()
        out = None
        if self._gl is not None:
            try:
                out = self._gl.render([c[:6] for c in self._cp], s, self.outline_rgb, LIGHT, HALF,
                                      tier["outline"] / s, max(2, self.raster.ss))
            except Exception:  # noqa: BLE001 - driver trouble: fall back to the CPU renderer for good
                import logging
                logging.getLogger("focusbuddy.companion").exception("GPU 3D render failed - using CPU")
                self._gl = None
                self.backend = "cpu"
                out = None
        if out is not None:
            qi, ox, oy = out
        else:
            if tier["spec"] != 1.0:
                tprims = [Prim(p.c, p.ax, p.r, p.col, p.grp, p.em, min(1.0, p.spec * tier["spec"])) for p in tprims]
            img, (ox, oy) = self.raster.render(tprims, cam, s, self.outline_rgb, tier["outline"])
            h, w = img.shape[:2]
            qi = QImage(img.tobytes(), w, h, w * 4, QImage.Format_ARGB32_Premultiplied)
        g.save()
        g.scale(1 / s, 1 / s)
        g.drawImage(QPointF(ox, oy), qi)
        g.restore()
        t_decal = time.perf_counter()
        facing_sign = 1 if math.cos(math.radians(yaw)) >= 0 else -1
        self._decals(g, pose, face, cam, S, R, piv, facing_sign)
        if pose.ripple > 0:
            self._ripple(g, pose)
        if debug:
            self._debug3d(g, pose, rp, cam, S, R, piv, debug)
        self._marks(g, pose, facing_sign)
        g.restore()
        t1 = time.perf_counter()
        self.last_ms = (t1 - t0) * 1000
        st = self.stats
        st["total_ms"] = self.last_ms
        st["build_ms"] = (t_raster - t_build) * 1000
        st["raster_ms"] = (t_decal - t_raster) * 1000
        st["decal_ms"] = (t1 - t_decal) * 1000
        st["prims"] = len(prims)
        st["draw_calls"] = (1 if self.backend == "gpu" else 0) + 1 + getattr(self, "_decal_count", 0)
        st["backend"] = self.backend
        st["quality"] = self.tier_name(self.quality)

    # ---------------------------------------------------------------- helpers
    def _limb_z(self, name: str) -> float:
        if self.kind == "biped":
            return {"L": -5.5, "R": 5.5, "arm_L": -16.0, "arm_R": 16.0}.get(name, 0.0)
        near = name in ("FR", "BR")
        return (1 if near else -1) * 9.0

    def _biped_xshift(self, name: str) -> float:
        return {"L": 5.0, "R": -5.0, "arm_L": 11.0, "arm_R": -11.0}.get(name, 0.0)

    @staticmethod
    def _seg(a, b, radius: float, col, grp: int, spec: float = 0.2, over: float = 0.82) -> Prim:
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        d = b - a
        ln = float(np.linalg.norm(d))
        ax = frame_along(d if ln > 1e-6 else (0.0, 1.0, 0.0))
        return Prim((a + b) / 2, ax, np.array([ln / 2 + radius * over, radius, radius]), col, grp, spec=spec)

    @staticmethod
    def _ell(c, radii, col, grp: int, ax=None, em: bool = False, spec: float = 0.25) -> Prim:
        return Prim(np.asarray(c, dtype=float), np.eye(3) if ax is None else np.asarray(ax, dtype=float),
                    np.asarray(radii, dtype=float), col, grp, em, spec)

    def _head_frame(self, pose: Pose, yaw: float) -> np.ndarray:
        """Head rotation in model space: tilt + a turn toward the viewer (cute 3/4 face)."""
        k = 0.55 + 0.45 * _clamp(pose.front)
        ft = getattr(self, "face_turn", None)
        if ft:                                          # sheet faces: show the face nearly square on
            k = max(k, float(ft))
        k *= 1.0 - _clamp(getattr(pose, "back", 0.0))
        lim = 80.0 if ft else 70.0
        yaw = (yaw + 180.0) % 360.0 - 180.0             # orbit camera: keep the turn on the short side
        delta = max(-lim, min(lim, (90.0 - yaw) * k))
        # look-at chain: the head (and neck) turn toward the target (+ = the facing direction)
        sgn = 1.0 if pose.turn >= 0 else -1.0
        delta -= sgn * (getattr(pose, "head_yaw", 0.0) * 26.0 + getattr(pose, "neck_yaw", 0.0) * 14.0) * \
            (1.0 - 0.6 * _clamp(pose.front))
        # rot_y is model->camera convention; the same sign turns the face toward +z (viewer at yaw 0)
        return rot_y(delta) @ rot_z(pose.head_tilt) @ rot_x(getattr(pose, "head_roll", 0.0))

    # ------------------------------------------------------------ quadrupeds
    def _build_quad(self, pose: Pose, rp: RigPose):
        sp = self.species
        C = self.cols
        lay = rp.layout
        prims: list[Prim] = []
        tilt = rp.tilt
        Rb = rot_z(tilt)
        bx, by = rp.joints["torso"]
        brx, bry = lay["brx"], lay["bry"]
        brz = bry * 0.9
        s = lay["s"]
        main, light, extra, accent = C["main"], C["light"], C["extra"], C["accent"]
        leg_col = extra if sp == "fox" else main
        G_BODY, G_FAR, G_NEAR, G_HEAD, G_TAIL, G_EARS, G_MUZ = 1, 2, 3, 4, 5, 6, 7
        # torso + belly (+ stripes)
        tw = getattr(pose, "torso_twist", 0.0)
        Rt = Rb @ rot_x(tw * 0.6) if tw else Rb
        prims.append(self._ell((bx, by, 0.0), (brx, bry, brz), main, G_BODY, Rt))
        prims.append(self._ell(np.array([bx, by, 0.0]) + Rb @ np.array([brx * 0.25, bry * 0.42, 0.0]),
                               (brx * 0.62, bry * 0.55, brz * 0.82), light, G_BODY, Rb, spec=0.15))
        if sp == "cat":
            for i in range(3):
                off = Rb @ np.array([-brx * 0.45 + i * 9.0, -bry * 0.86, 0.0])
                prims.append(self._ell(np.array([bx, by, 0.0]) + off, (2.6, 3.4, brz * 0.7), C["extra"], G_BODY,
                                       Rb @ rot_z(-12)))
        # haunch when sitting
        if s > 0.05:
            hx_, hy_ = rp.layout["bx"] - 9, -10 + lay["y0"] * 0.3
            prims.append(self._ell((hx_, hy_, 0.0), (13 * s + 0.1, 10.5 * s + 0.1, brz * 0.95 * s + 0.1), main,
                                   G_BODY))
        # legs (far side first doesn't matter: z-buffer)
        tuck = _clamp(getattr(pose, "tuck", 0.0))
        for name, ls in rp.legs.items():
            near = name in ("FR", "BR")
            z = self._limb_z(name) * (brz / 15.3)
            grp = G_NEAR if near else G_FAR
            if (name == "FL" and tuck > 0.6) or (name == "BL" and s > 0.7) or \
                    (name == "BR" and s > 0.7 and pose.scratch < 0.2):
                continue
            if name == "FR" and tuck > 0.6 and pose.paw < 0.2:
                prims.append(self._ell((ls.root[0] + 4, -3.2, z), (5.0, 3.2, 4.0), leg_col, grp))
                continue
            w = ls.width
            a = (ls.root[0], ls.root[1], z)
            j = (ls.joint[0], ls.joint[1], z)
            e = (ls.end[0], ls.end[1], z)
            prims.append(self._seg(a, j, w * 0.55, leg_col, grp))
            prims.append(self._seg(j, e, w * 0.48, leg_col, grp))
            fa = getattr(ls, "foot_angle", 0.0)
            fr = math.radians(fa)
            prims.append(self._ell((e[0] + 1.2 * math.cos(fr), e[1] + 0.6 + 1.2 * math.sin(fr), z),
                                   (w * 0.66, w * 0.48, w * 0.58), leg_col, grp, rot_z(fa) if fa else None))
        # tail
        pts = rp.tail
        if pts and sp != "rabbit":
            widths = list(rp.tail_widths) + [6.0] * len(pts)
            zt = -brz * 0.15
            for i, (p0, p1) in enumerate(zip(pts, pts[1:])):
                last = i == len(pts) - 2
                col = main
                if last and sp == "cat":
                    col = extra
                if last and sp == "fox":
                    col = light
                rad = widths[i] * (0.5 if sp != "fox" else 0.62)
                prims.append(self._seg((p0[0], p0[1], zt), (p1[0], p1[1], zt), rad, col, G_TAIL, over=0.9))
        elif pts:
            x, y = pts[0]
            prims.append(self._ell((x - 3, y - 3, 0.0), (7.5, 7.0, 7.0), light, G_TAIL))
        # neck + head
        hx, hy = rp.joints["head"]
        hr = lay["hr"]
        nb = rp.joints["neck_base"]
        prims.append(self._seg((nb[0], nb[1], 0.0), (hx, hy + hr * 0.3, 0.0), 8.5, main, G_BODY))
        cam, yaw = self.camera(pose)
        Rh = self._head_frame(pose, yaw)
        Rh = rot_z(tilt) @ Rh
        H = np.array([hx, hy, 0.0])
        head_r = (hr + 1, hr - 1, hr) if sp == "fox" else (hr, hr, hr * 0.98)
        prims.append(self._ell(H, head_r, main, G_HEAD, Rh))
        if sp == "dog":   # collar + tag
            cpos = H + Rh @ np.array([-hr * 0.2, hr * 0.85, 0.0])
            prims.append(self._ell(cpos, (4.0, 3.2, hr * 0.75), _rgb("#E5484D"), G_BODY, Rh @ rot_z(-20)))
            tag = cpos + Rh @ (rot_z(_clamp(pose.tag, -40, 40)) @ np.array([3.0, 5.0, 0.0])) + np.array([0, 0, 6.0])
            prims.append(self._ell(tag, (2.8, 2.8, 1.4), _rgb("#FFD84D"), G_BODY, spec=0.6))
        # muzzle / cheeks / nose
        if sp == "fox":
            for side in (-1, 1):
                prims.append(self._ell(H + Rh @ np.array([hr * 0.38, hr * 0.42, side * hr * 0.42]),
                                       (hr * 0.46, hr * 0.36, hr * 0.42), light, G_MUZ, Rh, spec=0.1))
            prims.append(self._ell(H + Rh @ np.array([hr * 0.78, hr * 0.28, 0.0]), (hr * 0.36, hr * 0.24, hr * 0.3),
                                   light, G_MUZ, Rh, spec=0.1))
            nose = H + Rh @ np.array([hr * 1.12, hr * 0.2, 0.0])
            prims.append(self._ell(nose, (2.6, 2.0, 2.4), _rgb("#2A1E2E"), G_MUZ, Rh, spec=0.9))
        else:
            mz = {"cat": (0.62, 0.36, 0.4, 0.3, 0.46), "dog": (0.6, 0.4, 0.5, 0.34, 0.52),
                  "rabbit": (0.66, 0.34, 0.34, 0.26, 0.4)}.get(sp, (0.62, 0.36, 0.4, 0.3, 0.46))
            mc = H + Rh @ np.array([hr * mz[0], hr * mz[1], 0.0])
            prims.append(self._ell(mc, (hr * mz[2], hr * mz[3], hr * mz[4]), light, G_MUZ, Rh, spec=0.12))
            nz = getattr(pose, "nose", 0.0) * 0.6
            nose = mc + Rh @ np.array([hr * mz[2] * 0.92, -hr * mz[3] * 0.35 + nz, 0.0])
            ncol = _rgb("#2A1E2E") if sp == "dog" else _rgb("#E9788A")
            prims.append(self._ell(nose, (2.4, 1.9, 2.6) if sp != "dog" else (3.2, 2.4, 3.6), ncol, G_MUZ, Rh,
                                   spec=0.8))
        # ears
        perk = pose.ear
        flop = pose.ear_flop
        if sp in ("cat", "fox"):
            hw, hh = (8.5, 16.0) if sp == "cat" else (9.5, 21.0)
            hh *= 1 + 0.08 * perk
            for side in (-1, 1):
                lean = -12 * (1 - 0.35 * perk) - 28 * max(0.0, -perk)
                fl_ = flop + (getattr(pose, "ear_l", 0.0) if side < 0 else getattr(pose, "ear_r", 0.0))
                d = rot_z(-lean * 0.6 + fl_ * 0.7) @ np.array([0.0, -1.0, side * (0.42 + 0.3 * max(0.0, -perk))])
                base = H + Rh @ np.array([-1.5, -hr * 0.72, side * hr * 0.48])
                dd = Rh @ d
                ax = frame_along(dd, Rh @ np.array([1.0, 0.0, 0.0]))
                ctr = base + dd / np.linalg.norm(dd) * hh * 0.42
                prims.append(Prim(ctr, ax, np.array([hh * 0.55, hw * 0.34, hw * 0.62]), main, G_EARS))
                prims.append(Prim(ctr + ax[:, 1] * hw * 0.2 + dd / np.linalg.norm(dd) * 1.0, ax,
                                  np.array([hh * 0.38, hw * 0.16, hw * 0.36]),
                                  extra if sp == "fox" else accent, G_EARS, spec=0.05))
        elif sp == "rabbit":
            for side in (-1, 1):
                droop = 55 * max(0.0, -perk) - 6 * perk + (flop + (getattr(pose, "ear_l", 0.0) if side < 0 else
                                                                   getattr(pose, "ear_r", 0.0))) * 1.1
                d = rot_z(-droop) @ np.array([-0.1, -1.0, side * 0.2])
                base = H + Rh @ np.array([-2.0, -hr * 0.78, side * hr * 0.32])
                dd = Rh @ d
                ax = frame_along(dd, Rh @ np.array([1.0, 0.0, 0.0]))
                ctr = base + dd / np.linalg.norm(dd) * 15.0
                prims.append(Prim(ctr, ax, np.array([18.0, 3.4, 6.8]), main, G_EARS))
                prims.append(Prim(ctr + ax[:, 1] * 1.6, ax, np.array([12.5, 1.9, 3.4]), accent, G_EARS, spec=0.05))
        elif sp == "dog":
            for side in (-1, 1):
                rot = 12 - 30 * max(0.0, perk) + (flop + (getattr(pose, "ear_l", 0.0) if side < 0 else
                                                          getattr(pose, "ear_r", 0.0))) * 0.9
                d = rot_z(side * 0 + rot * 0.5) @ np.array([0.12, 1.0, side * 0.35])
                base = H + Rh @ np.array([-2.0, -hr * 0.55, side * hr * 0.82])
                dd = Rh @ d
                ax = frame_along(dd, Rh @ np.array([0.0, 0.0, side]))
                ctr = base + dd / np.linalg.norm(dd) * 9.0
                prims.append(Prim(ctr, ax, np.array([11.5, 6.5, 2.8]), extra, G_EARS))
        face = {"H": H, "Rh": Rh, "hr": hr, "muzzle": True,
                "eye_dirs": [(-27.0, 0.02), (27.0, 0.02)], "eye_size": 1.3,
                "mouth": H + Rh @ np.array([hr * 0.98, hr * 0.5, 0.0]),
                "mouth_n": Rh @ np.array([1.0, 0.45, 0.0]),
                "whiskers": sp == "cat"}
        return prims, face

    # --------------------------------------------------------------- bipeds
    def _build_biped(self, pose: Pose, rp: RigPose):
        sp = self.species
        robot = sp == "robot"
        C = self.cols
        lay = rp.layout
        prims: list[Prim] = []
        G_BODY, G_LEGL, G_LEGR, G_ARML, G_ARMR, G_HEAD, G_HAIR, G_VISOR = 1, 2, 3, 4, 5, 6, 7, 8
        tilt = rp.tilt
        Rb = rot_z(tilt)
        sx = lay["shift"]
        body_top = lay["body_top"]
        pants = C["shade"] if robot else _rgb("#3E4A6E")
        shoe = C["outline"] if robot else _rgb("#FFFFFF")
        arm_col = C["shade"] if robot else C["main"]
        hand = C["accent"] if robot else C["light"]
        piv = np.array([rp.pivot[0], rp.pivot[1], 0.0])

        def tw(p):   # body-frame point -> tilted world
            p = np.asarray(p, dtype=float)
            return Rb @ (p - piv) + piv

        breath = 1 + 0.03 * getattr(pose, "breath", 0.0) + 0.07 * getattr(pose, "puff", 0.0)
        torso_c = tw((sx, body_top + 14.5, 0.0))
        if robot:
            prims.append(self._ell(torso_c, (12.0, 15.5 * breath, 16.5), C["main"], G_BODY, Rb, spec=0.55))
            prims.append(self._ell(torso_c + Rb @ np.array([9.0, -1.5, 0.0]), (4.5, 7.5, 10.0), C["light"], G_BODY,
                                   Rb, spec=0.5))
            prims.append(self._ell(torso_c + Rb @ np.array([12.0, -2.0, 0.0]), (1.6, 3.0, 3.0), C["accent"], G_BODY,
                                   Rb, em=True))
        else:
            prims.append(self._ell(torso_c, (11.5, 15.5 * breath, 15.5), C["main"], G_BODY, Rb, spec=0.12))
            hem = _clamp(getattr(pose, "hem", 0.0), -20.0, 20.0)
            prims.append(self._ell(tw((sx, body_top + 26.0, 0.0)), (10.0, 5.0, 13.0), C["shade"], G_BODY,
                                   Rb @ rot_z(hem) if hem else Rb, spec=0.1))
        # legs
        for name, ls in rp.legs.items():
            dx = self._biped_xshift(name)
            z = self._limb_z(name)
            grp = G_LEGR if name == "R" else G_LEGL
            a = (ls.root[0] + dx, ls.root[1], z)
            j = (ls.joint[0] + dx, ls.joint[1], z)
            e = (ls.end[0] + dx, ls.end[1], z)
            prims.append(self._seg(a, j, ls.width * 0.55, pants, grp))
            prims.append(self._seg(j, e, ls.width * 0.5, pants, grp))
            fa = getattr(ls, "foot_angle", 0.0)
            fr = math.radians(fa)
            prims.append(self._ell((e[0] + 2.6 * math.cos(fr) - 0.8 * math.sin(fr), e[1] + 0.8 + 2.6 * math.sin(fr), z),
                                   (6.2, 3.9, 4.6), shoe, grp, rot_z(fa) if fa else None, spec=0.4))
        # arms
        twist = math.radians(getattr(pose, "torso_twist", 0.0))
        for name, ls in rp.arms.items():
            dx = self._biped_xshift(name)
            z = self._limb_z(name) * (1.05 if robot else 0.95)
            if twist:                          # shoulders counter-rotate: near arm swings forward / back
                dx += -math.sin(twist) * z * 0.6
            grp = G_ARMR if name == "arm_R" else G_ARML
            a = (ls.root[0] + dx, ls.root[1], z)
            j = (ls.joint[0] + dx, ls.joint[1], z)
            e = (ls.end[0] + dx, ls.end[1], z)
            prims.append(self._seg(a, j, 3.9, arm_col, grp, spec=0.4 if robot else 0.12))
            prims.append(self._seg(j, e, 3.6, arm_col, grp, spec=0.4 if robot else 0.12))
            prims.append(self._ell(e, (4.7, 4.7, 4.7), hand, grp, em=False, spec=0.3))
        # head
        hx, hy = rp.joints["head"]
        H = np.array([hx, hy, 0.0])
        cam, yaw = self.camera(pose)
        Rh = rot_z(tilt) @ self._head_frame(pose, yaw)
        if lay.get("sleeping"):
            Rh = Rh @ rot_z(10)
        hr = 23.0
        if robot:
            prims.append(self._ell(H, (21.0, 19.5, 24.0), C["main"], G_HEAD, Rh, spec=0.6))
            prims.append(self._ell(H + Rh @ np.array([12.5, 1.0, 0.0]), (10.0, 13.5, 17.5), C["extra"], G_VISOR, Rh,
                                   spec=0.9))
            for side in (-1, 1):
                prims.append(self._ell(H + Rh @ np.array([0.0, 1.0, side * 23.5]), (5.0, 6.5, 3.0), C["shade"],
                                       G_HEAD, Rh, spec=0.5))
            top = H + Rh @ np.array([0.0, -19.0, 0.0])
            d = Rh @ (rot_z(_clamp(pose.antenna, -35, 35)) @ np.array([0.18, -1.0, 0.0]))
            tip = top + d / np.linalg.norm(d) * 11.0
            prims.append(self._seg(top, tip, 1.3, C["outline"], G_HEAD, spec=0.4))
            led = "#FF4D4D" if (pose.mark == "!" or pose.glitch > 0.2 or getattr(pose, "flush", 0.0) > 0.55) else self.spec.palette.accent
            on = int(pose.mark_phase * 2) % 2 == 0
            prims.append(self._ell(tip, (4.0, 4.0, 4.0), _rgb(led if on else "#FFB27A"), G_HEAD, em=True))
            eye_dirs = [(-24.0, 0.0), (24.0, 0.0)]
            mouth = H + Rh @ np.array([22.5, 9.0, 0.0])
            mouth_n = Rh @ np.array([1.0, 0.35, 0.0])
            eye_r = 22.6
        else:
            prims.append(self._ell(H + Rh @ np.array([1.0, 2.0, 0.0]), (hr - 2, hr - 3, hr - 2.5), C["light"],
                                   G_HEAD, Rh, spec=0.18))
            for side in (-1, 1):
                prims.append(self._ell(H + Rh @ np.array([-1.0, 4.0, side * (hr - 3.0)]), (3.2, 4.5, 2.2),
                                       C["light"], G_HEAD, Rh, spec=0.1))
            hair = C["extra"]
            # hair: a cap over the top and back of the head + a fringe; the face stays free
            prims.append(self._ell(H + Rh @ np.array([-6.5, -5.5, 0.0]), (hr - 3.0, hr - 3.5, hr + 0.3), hair, G_HAIR,
                                   Rh @ rot_z(-12), spec=0.35))
            prims.append(self._ell(H + Rh @ np.array([7.0, -14.0, 0.0]), (11.0, 6.0, hr - 3.0), hair, G_HAIR,
                                   Rh @ rot_z(24), spec=0.35))
            tuft = H + Rh @ np.array([-hr * 0.55, -hr * 0.85, 0.0])
            d = Rh @ (rot_z(-30 + _clamp(pose.hair, -30, 30)) @ np.array([-0.4, -1.0, 0.0]))
            prims.append(Prim(tuft + d * 3.0, frame_along(d), np.array([7.0, 4.5, 4.5]), hair, G_HAIR, spec=0.35))
            eye_dirs = [(-26.0, 0.12), (26.0, 0.12)]
            mouth = H + Rh @ np.array([19.0, 10.5, 0.0])
            mouth_n = Rh @ np.array([1.0, 0.55, 0.0])
            eye_r = 20.5
        face = {"H": H, "Rh": Rh, "hr": eye_r, "muzzle": False, "eye_dirs": eye_dirs, "eye_size": 1.2,
                "mouth": mouth, "mouth_n": mouth_n, "whiskers": False, "robot": robot,
                "chest": torso_c, "Rb": Rb}
        return prims, face

    # ---------------------------------------------------------------- decals
    def _project(self, pose, cam, S, R, piv, p):
        q = self.to_cam(pose, cam, np.asarray(p, dtype=float), S, R, piv)
        return float(q[0]), float(q[1]), float(q[2])

    def _decal(self, g: QPainter, pose, cam, S, R, piv, pos, normal, size: float, draw, min_facing: float = 0.12,
               occlusion: bool = True):
        u, v, z = self._project(pose, cam, S, R, piv, pos)
        n = cam @ (S @ np.asarray(normal, dtype=float))
        n = n / (np.linalg.norm(n) or 1.0)
        if n[2] < min_facing:
            return False
        if occlusion and self.depth_at(u, v) > z + 2.5:
            return False                   # something (a paw, a hand) is in front of it
        ang = math.degrees(math.atan2(n[1], n[0]))
        self._decal_count = getattr(self, "_decal_count", 0) + 1
        g.save()
        g.translate(u, v)
        g.rotate(ang)
        g.scale(max(0.15, float(n[2])), 1.0)
        g.rotate(-ang)
        g.scale(size, size)
        draw(g)
        g.restore()
        return True

    def _decals(self, g: QPainter, pose: Pose, face: dict, cam, S, R, piv, facing_sign: int) -> None:
        H, Rh, hr = face["H"], face["Rh"], face["hr"]
        robot = face.get("robot", False)
        glow = None
        color = "#2A1E2E"
        if robot:
            glow = self.spec.palette.accent if pose.mark != "!" and getattr(pose, "flush", 0.0) <= 0.55 else "#FF6B6B"
        lx = pose.look_x * 0.28
        ly = pose.look_y * 0.18
        # eyes - eye 0 is whichever eye is on the left of the screen (brows / lids are mirrored per eye)
        eyes = []
        for i, (deg, up) in enumerate(face["eye_dirs"]):
            a = math.radians(deg) + lx
            d = np.array([math.cos(a), up + ly, math.sin(a)])
            d /= np.linalg.norm(d)
            pos = H + Rh @ (d * hr)
            eyes.append((self._project(pose, cam, S, R, piv, pos)[0], d, pos))
        left_first = eyes[0][0] <= eyes[-1][0]
        for i, (_u, d, pos) in enumerate(eyes):
            idx = i if left_first else len(eyes) - 1 - i
            conv = _clamp(getattr(pose, "eye_conv", 0.0), 0.0, 0.8) * 1.2 * (1 if idx == 0 else -1)

            # on a 3D head the expression multipliers stack up (emotion x action); compress big
            # values so wide / shocked eyes stay on the face instead of covering it
            es = max(0.05, float(pose.eye_scale))
            soft = es if es <= 1.0 else min(1.28, 1.0 + (es - 1.0) * 0.55)
            if pose.eye_style == "shock":
                soft *= 0.82

            def draw_eye(gg, k=idx, cv=conv, sz=soft / es):
                if cv:
                    gg.translate(cv, 0)
                self.eye(gg, pose, k, sz, color, glow)
            self._decal(g, pose, cam, S, R, piv, pos, Rh @ d, face["eye_size"], draw_eye, 0.08)
        # cheeks / blush / flush
        if not robot and face.get("cheeks", True):
            for deg in (-44.0, 44.0):
                a = math.radians(deg)
                d = np.array([math.cos(a) * 0.9, 0.42, math.sin(a)])
                d /= np.linalg.norm(d)
                pos = H + Rh @ (d * hr)
                self._decal(g, pose, cam, S, R, piv, pos, Rh @ d, 1.0,
                            lambda gg: self._cheek(gg, pose), 0.15)
        # mouth (an asset with a modelled muzzle can turn the decal mouth off)
        mp = face["mouth"] + face["Rh"] @ np.array([0.0, 1.6 * _clamp(getattr(pose, "jaw", 0.0)), 0.0])
        if not face.get("mouth_on", True):
            mp = None
        mn = face["mouth_n"]

        def draw_mouth(gg):
            if robot:
                self._robot_mouth(gg, pose)
            elif self.species == "cat" and pose.mouth == "smile":
                self.mouth(gg, 0, 0, pose, style_override="w")
            elif self.species == "rabbit" and pose.mouth in ("grin", "smile"):
                gg.setPen(self.pen(self.pal.outline, 0.9))
                gg.setBrush(_c("#FFFFFF"))
                gg.drawRect(QRectF(-2, -0.4, 4, 3.2))
                gg.drawLine(QPointF(0, -0.4), QPointF(0, 2.8))
            else:
                self.mouth(gg, 0, 0, pose)
            if pose.tongue > 0.05:
                tl = 2.5 + 4.5 * _clamp(pose.tongue)
                gg.setPen(self.pen("#B0465A", 1.0))
                gg.setBrush(_c("#F58C9A"))
                gg.drawRoundedRect(QRectF(-2.4, 1.0, 4.8, tl), 2.4, 2.4)
        if mp is not None:
            self._decal(g, pose, cam, S, R, piv, mp, mn, 1.0, draw_mouth, 0.1)
        # whiskers
        if face.get("whiskers"):
            for side in (-1, 1):
                a = math.radians(side * 30)
                d = np.array([math.cos(a) * 0.85, 0.38, math.sin(a)])
                d /= np.linalg.norm(d)
                self._decal(g, pose, cam, S, R, piv, H + Rh @ (d * hr * 1.02), Rh @ d, 1.0,
                            lambda gg, sd=side: self._whiskers(gg, sd), 0.2)
        # headband (focus mode)
        if pose.headband:
            u, v, _ = self._project(pose, cam, S, R, piv, H)
            r_ = face["hr"] * 1.02
            g.save()
            pen = QPen(_c("#E5484D"), 5.0)
            pen.setCapStyle(Qt.FlatCap)
            g.setPen(pen)
            g.setBrush(Qt.NoBrush)
            g.drawArc(QRectF(u - r_, v - r_ * 0.98, 2 * r_, 2 * r_), 28 * 16, 124 * 16)
            g.restore()
        # props: book / notepad / pencil, scan beam, glitch
        if face.get("chest") is not None and max(pose.read, getattr(pose, "write", 0.0)) > 0.05:
            u, v, _ = self._project(pose, cam, S, R, piv, face["chest"] + np.array([16.0, 0.0, 0.0]))
            g.save()
            g.translate(u, v)
            self._prop(g, pose, robot)
            g.restore()
        if robot and pose.scan > 0.05:
            u, v, _ = self._project(pose, cam, S, R, piv, H + Rh @ np.array([22.0, 0.0, 0.0]))
            g.save()
            g.setPen(Qt.NoPen)
            a = math.radians(pose.look_y * 18) + (0 if facing_sign > 0 else math.pi)
            beam = QPainterPath(QPointF(u, v))
            for da in (-0.16, 0.16):
                beam.lineTo(QPointF(u + 70 * math.cos(a + da), v + 70 * math.sin(a + da)))
            beam.closeSubpath()
            g.setBrush(_c("#FF4D4D", int(70 * pose.scan)))
            g.drawPath(beam)
            g.restore()
        if robot and pose.glitch > 0.1:
            u, v, _ = self._project(pose, cam, S, R, piv, H)
            g.save()
            g.setPen(Qt.NoPen)
            t = pose.mark_phase
            for i in range(4):
                yy = v - 10 + ((i * 37 + int(t * 30) * 13) % 22)
                off = math.sin(t * 50 + i * 2) * 6 * pose.glitch
                g.setBrush(_c(["#FF4D4D", "#4DF2FF", "#FFD84D", "#A16BFF"][i], int(150 * pose.glitch)))
                g.drawRect(QRectF(u - 17 + off, yy, 34, 2.2))
            g.restore()
        # where marks (! zzz hearts steam ...) float
        u, v, _ = self._project(pose, cam, S, R, piv, H)
        self.head_pos = (u * facing_sign, v)   # _marks multiplies x by facing
        self.head_r = face["hr"]

    def depth_at(self, u: float, v: float) -> float:
        """Front-most model depth at a projected point (vectorised analytic ray cast against all parts)."""
        if not self._cp:
            return -1e9
        if getattr(self, "_cp_arr", None) is None or self._cp_arr[0] is not self._cp:
            Cs = np.array([c[0] for c in self._cp])
            Ms = np.array([c[6] for c in self._cp])
            self._cp_arr = (self._cp, Cs, Ms)
        _, Cs, Ms = self._cp_arr
        du = u - Cs[:, 0]
        dv = v - Cs[:, 1]
        q0 = Ms[:, :, 0] * du[:, None] + Ms[:, :, 1] * dv[:, None]
        m = Ms[:, :, 2]
        a = (m * m).sum(axis=1)
        b = 2.0 * (q0 * m).sum(axis=1)
        c = (q0 * q0).sum(axis=1) - 1.0
        disc = b * b - 4.0 * a * c
        ok = disc >= 0
        if not ok.any():
            return -1e9
        z = Cs[ok, 2] + (np.sqrt(disc[ok]) - b[ok]) / (2.0 * a[ok])
        return float(z.max())

    def _lid_color(self) -> str:
        if self.species == "buddy":
            return self.pal.light
        if self.species == "robot":
            return self.pal.extra
        return self.pal.main

    def _cheek(self, g: QPainter, pose: Pose) -> None:
        fl = _clamp(getattr(pose, "flush", 0.0))
        a = int(170 * _clamp(pose.blush))
        g.setPen(Qt.NoPen)
        if fl > 0.05:
            g.setBrush(_c("#FF4D5E", int(80 * fl)))
            g.drawEllipse(QPointF(0, -1.5), 6.5, 4.5)
            a = max(a, int(210 * fl))
        if a > 5:
            g.setBrush(_c(self.pal.accent if fl < 0.5 else "#FF6F7D", a))
            g.drawEllipse(QPointF(0, 0), 3.8, 2.4)

    def _whiskers(self, g: QPainter, side: int) -> None:
        g.setPen(self.pen("#7A4318", 0.8))
        for dy in (-0.8, 1.6):
            g.drawLine(QPointF(0, dy), QPointF(side * 8.5, dy + (dy - 0.4) * 0.8))

    def _robot_mouth(self, g: QPainter, pose: Pose) -> None:
        glow = self.spec.palette.accent if pose.mark != "!" and getattr(pose, "flush", 0.0) <= 0.55 else "#FF6B6B"
        g.setPen(self.pen(glow, 1.5))
        g.setBrush(Qt.NoBrush)
        m = pose.mouth
        if m in ("grin", "smile", "laugh"):
            p = QPainterPath(QPointF(-4, -1))
            p.quadTo(0, 2.5, 4, -1)
            g.drawPath(p)
        elif m in ("o", "open", "yawn", "shout"):
            r = 2 + 2.2 * max(pose.yawn, 1.0 if m == "shout" else 0.0)
            g.drawEllipse(QPointF(0, 0), r, r)
        elif m in ("frown", "wobble", "pout"):
            p = QPainterPath(QPointF(-4, 1.5))
            p.quadTo(0, -2, 4, 1.5)
            g.drawPath(p)
        elif m in ("teeth",):
            for k in range(5):
                g.drawLine(QPointF(-4 + k * 2, -1.2), QPointF(-4 + k * 2, 1.2))
        elif m == "w":
            p = QPainterPath(QPointF(-4, 0))
            p.lineTo(-2, -1.5)
            p.lineTo(0, 0)
            p.lineTo(2, -1.5)
            p.lineTo(4, 0)
            g.drawPath(p)
        else:
            g.drawLine(QPointF(-3, 0), QPointF(3, 0))

    def _prop(self, g: QPainter, pose: Pose, robot: bool) -> None:
        pal = self.pal
        g.setOpacity(_clamp(max(pose.read, pose.write) * 1.4))
        g.rotate(-14 if pose.read >= pose.write else 8)
        if robot:
            g.setPen(self.pen(pal.outline, 1.4))
            g.setBrush(_c(pal.extra))
            g.drawRoundedRect(QRectF(-11, -8, 22, 15), 3, 3)
            g.setPen(self.pen(pal.accent, 1.3))
            for i in range(3):
                g.drawLine(QPointF(-7, -3.5 + i * 3.4), QPointF(5 - (i % 2) * 4, -3.5 + i * 3.4))
        elif pose.write > pose.read:
            g.setPen(self.pen("#8C7A55", 1.2))
            g.setBrush(_c("#FFF6D8"))
            g.drawRoundedRect(QRectF(-11, -7, 22, 15), 2, 2)
            g.setPen(self.pen("#9DB3E8", 0.8))
            n = 1 + int(pose.mark_phase * 0.8) % 4
            for i in range(n):
                g.drawLine(QPointF(-8, -3 + i * 3.2), QPointF(7 - (i == n - 1) * 6, -3 + i * 3.2))
        else:
            g.setPen(self.pen("#8E2328", 1.3))
            g.setBrush(_c("#E5484D"))
            g.drawRoundedRect(QRectF(-12, -8, 24, 16), 2.5, 2.5)
            g.setPen(Qt.NoPen)
            g.setBrush(_c("#FFF8EC"))
            g.drawRect(QRectF(-10.5, -6.5, 10, 13))
            g.drawRect(QRectF(0.5, -6.5, 10, 13))
        g.setOpacity(1.0)

    # ---------------------------------------------------------------- debug
    def _debug3d(self, g, pose, rp: RigPose, cam, S, R, piv, opts) -> None:
        g.save()

        def P(p, z=0.0):
            u, v, _ = self._project(pose, cam, S, R, piv, (p[0], p[1], z))
            return QPointF(u, v)
        if opts.get("skeleton", True):
            g.setPen(QPen(QColor(91, 108, 255, 230), 1.4))
            for d in (rp.legs, rp.arms):
                for n, ls in d.items():
                    dx = self._biped_xshift(n) if self.kind == "biped" else 0.0
                    z = self._limb_z(n)
                    g.drawLine(P((ls.root[0] + dx, ls.root[1]), z), P((ls.joint[0] + dx, ls.joint[1]), z))
                    g.drawLine(P((ls.joint[0] + dx, ls.joint[1]), z), P((ls.end[0] + dx, ls.end[1]), z))
            for a, b in zip(rp.tail, rp.tail[1:]):
                g.drawLine(P(a), P(b))
            if "neck_base" in rp.joints and "head" in rp.joints:
                g.drawLine(P(rp.joints["neck_base"]), P(rp.joints["head"]))
        if opts.get("ik", True):
            g.setPen(QPen(QColor(229, 72, 77, 220), 1.0))
            for d in (rp.legs, rp.arms):
                for n, ls in d.items():
                    dx = self._biped_xshift(n) if self.kind == "biped" else 0.0
                    c = P((ls.target[0] + dx, ls.target[1]), self._limb_z(n))
                    g.drawLine(c + QPointF(-2.5, 0), c + QPointF(2.5, 0))
                    g.drawLine(c + QPointF(0, -2.5), c + QPointF(0, 2.5))
        if opts.get("contacts", True):
            for n, ls in rp.legs.items():
                dx = self._biped_xshift(n) if self.kind == "biped" else 0.0
                g.setPen(Qt.NoPen)
                g.setBrush(QColor(34, 176, 125, 230) if ls.contact else QColor(242, 153, 74, 200))
                g.drawEllipse(P((ls.end[0] + dx, ls.end[1] + 3.5), self._limb_z(n)), 2.2, 1.2)
        if opts.get("names", False):
            for d in (rp.legs, rp.arms):
                for n, ls in d.items():
                    dx = self._biped_xshift(n) if self.kind == "biped" else 0.0
                    z = self._limb_z(n)
                    for nm, a, b in ((f"upper_{n}", ls.root, ls.joint), (f"lower_{n}", ls.joint, ls.end)):
                        q = P(((a[0] + b[0]) / 2 + dx, (a[1] + b[1]) / 2), z)
                        self.bone_label(g, nm, (q.x(), q.y()))
            for nm in ("pelvis", "chest", "neck_base", "head", "jaw"):
                if nm in rp.joints:
                    q = P(rp.joints[nm])
                    self.bone_label(g, nm.replace("_base", ""), (q.x(), q.y()))
        g.restore()

    def _ripple(self, g: QPainter, pose: Pose) -> None:
        self._tip = self.press_point()
        super()._ripple(g, pose)

    def draw(self, g: QPainter, pose: Pose) -> None:   # pragma: no cover - paint() is overridden
        pass
