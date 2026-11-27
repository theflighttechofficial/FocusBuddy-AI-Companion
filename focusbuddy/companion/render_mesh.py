"""Production renderer: skinned GLB characters (``CharacterAsset``) with stylised materials.

Pipeline per frame (one call from the companion's single update loop):

    Pose + RigPose  ->  RigDriver (bone transforms, limb IK on the asset's own bones)
                    ->  morph targets (breath, cheeks, ARKit-style face shapes)
                    ->  linear blend skinning (NumPy)
                    ->  squash / roll / camera
                    ->  GPU skinning + raster (GLSkinRaster) or CPU skinning + the vectorised NumPy rasteriser
                    ->  stylised shading: soft key light + cool fill + hemispherical ambient +
                        rim, per-material response (fur sheen, cloth, skin with warm
                        subsurface tint, plastic, metal with environment, emissive LEDs)
                    ->  scale-aware outline (silhouette + depth contours)
                    ->  decal face (eyes / brows / mouth) or the asset's own morph face
                    ->  optional effects (marks), debug skeleton

It subclasses ``Renderer3D`` so the camera, the 3/4 head turn, the face decals, the
marks, the press point and the debug overlays are shared with the procedural models.
"""

from __future__ import annotations

import math
import time

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen

from .character_asset import ARKIT_MORPHS, CharacterAsset
from .render3d import HALF, LIGHT, XCOMP, Prim, Renderer3D, _dilate, rot_z
from .renderer import GROUND_MARGIN, UNITS, _clamp
from .skinning import RigDriver, skin

# --------------------------------------------------------------- materials
from .pbr import FILL, GROUND, KEY_COL, FILL_COL, MATERIALS, SKY, material_inputs, material_table, shade  # noqa: E402,F401
from . import pbr as _pbr  # noqa: E402


def auto_decal_face(asset: CharacterAsset, face: dict) -> dict:
    """Place the decal eyes over a model's own (small) eye meshes: ``"face": {"mode": "decal",
    "eyes_from": "<eye material>", "eye_yaw": 49}``. The head sphere is centred behind the two eye
    centroids so both eyes sit on it at +/- ``eye_yaw`` degrees; values are bind-pose offsets from
    the Head joint in the head's frame (internal units), like a hand-written face block."""
    try:
        names = [m.key for m in asset.materials]
        mi = names.index(str(face["eyes_from"]))
        mesh = asset.lod("high")
        P = mesh.positions[mesh.material == mi]
        hi = asset.joint_index("Head")
        if len(P) < 2 or hi < 0:
            return {}
        side = P[:, 2] > np.median(P[:, 2])
        e0, e1 = P[~side].mean(axis=0), P[side].mean(axis=0)
        mid = 0.5 * (e0 + e1)
        half = 0.5 * float(np.linalg.norm(e1 - e0))
        yaw = float(face.get("eye_yaw", 49.0))
        r = half / max(0.2, math.sin(math.radians(yaw)))
        centre = mid - np.array([r * math.cos(math.radians(yaw)), 0.0, 0.0])   # internal +x = forward
        H = asset.joints[hi].bind_world[:3, 3]
        out = {"center": [float(v) for v in centre - H], "head_radius": r, "eye_dirs": [[-yaw, 0.0], [yaw, 0.0]]}
        # eyelids in the fur colour around the eyes (display colours of the mesh near each eye)
        ring = np.zeros(len(mesh.positions), dtype=bool)
        for e in (e0, e1):
            d = np.linalg.norm(mesh.positions - e, axis=1)
            ring |= (d < 0.45 * r) & (mesh.material != mi)
        if ring.any() and "lid_color" not in face:
            c = np.clip(np.median(mesh.colors[ring], axis=0), 0, 1)
            out["lid_color"] = "#%02X%02X%02X" % tuple(int(round(v * 255)) for v in c)
        if "eye_size" not in face:
            out["eye_size"] = 0.09 * r
        return out
    except Exception:  # noqa: BLE001 - a face that can't be placed falls back to the defaults
        return {}


# ---------------------------------------------------------- CPU rasteriser
class MeshRaster:
    """Vectorised NumPy triangle rasteriser (z-buffer, per-pixel normals, outlines, supersampling)."""

    def __init__(self, ss: int = 2):
        self.ss = ss
        self.depth = None
        self.origin = (0.0, 0.0)
        self.scale = 1.0
        self.stats = {"tris": 0, "samples": 0}

    def render(self, V: np.ndarray, N: np.ndarray, C: np.ndarray, M: np.ndarray, F: np.ndarray, scale: float,
               table: np.ndarray, outline_rgb, outline_px: float = 1.0, lights: dict | None = None,
               contour_dz: float = 3.0, surf: dict | None = None):
        """``surf`` (optional): {"uv": (N,2), "tan": (N,4) camera space, "mats": [MaterialDesc],
        "inputs": material_inputs(...), "double": (n_mat,) bool} - per-pixel PBR textures."""
        ss = self.ss
        S = scale * ss
        pad = 3.0 / scale + outline_px * 2 / scale
        lo = V[:, :2].min(axis=0)
        hi = V[:, :2].max(axis=0)
        ox = math.floor((lo[0] - pad) * scale)
        oy = math.floor((lo[1] - pad) * scale)
        W = max(ss, (math.ceil((hi[0] + pad) * scale) - ox) * ss)
        H = max(ss, (math.ceil((hi[1] + pad) * scale) - oy) * ss)
        px = V[:, 0] * S - ox * ss
        py = V[:, 1] * S - oy * ss
        z = V[:, 2]
        # cull triangles facing away (normals point to +z toward the viewer); double-sided kept
        nzf = N[F, 2].sum(axis=1)
        keep_f = nzf > -0.6
        if surf is not None and surf.get("double") is not None and np.any(surf["double"]):
            keep_f |= surf["double"][M[F[:, 0]]]
        F = F[keep_f]
        mask_mats = None
        if surf is not None and surf.get("mask") is not None and np.any(surf["mask"] > 0):
            mask_mats = surf["mask"]
        x0, x1, x2 = px[F[:, 0]], px[F[:, 1]], px[F[:, 2]]
        y0, y1, y2 = py[F[:, 0]], py[F[:, 1]], py[F[:, 2]]
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        ok = np.abs(area) > 1e-9
        F, x0, x1, x2, y0, y1, y2, area = F[ok], x0[ok], x1[ok], x2[ok], y0[ok], y1[ok], y2[ok], area[ok]
        xmin = np.clip(np.floor(np.minimum(np.minimum(x0, x1), x2) - 0.5).astype(np.int64) + 1, 0, W - 1)
        xmax = np.clip(np.floor(np.maximum(np.maximum(x0, x1), x2) - 0.5).astype(np.int64), 0, W - 1)
        ymin = np.clip(np.floor(np.minimum(np.minimum(y0, y1), y2) - 0.5).astype(np.int64) + 1, 0, H - 1)
        ymax = np.clip(np.floor(np.maximum(np.maximum(y0, y1), y2) - 0.5).astype(np.int64), 0, H - 1)
        bw = np.maximum(0, xmax - xmin + 1)
        bh = np.maximum(0, ymax - ymin + 1)
        cnt = bw * bh
        keep = cnt > 0
        idx = np.flatnonzero(keep)
        cnt = cnt[keep]
        total = int(cnt.sum())
        self.stats = {"tris": int(len(idx)), "samples": total}
        depth = np.full(H * W, -1e9, dtype=np.float32)
        owner = np.full(H * W, -1, dtype=np.int64)
        bary = np.zeros((H * W, 2), dtype=np.float32)
        # chunk to bound memory
        starts = np.cumsum(cnt) - cnt
        chunk = 2_000_000
        lo_i = 0
        cum = np.cumsum(cnt)
        while lo_i < len(idx):
            # the chunk's end found by a binary search on the running total (a Python loop over every
            # triangle cost about a third of the frame)
            base_ = int(cum[lo_i - 1]) if lo_i > 0 else 0
            hi_i = int(np.searchsorted(cum, base_ + chunk, side="right"))
            hi_i = max(hi_i, lo_i + 1)
            acc = int(cum[hi_i - 1]) - base_
            sel = idx[lo_i:hi_i]
            c = cnt[lo_i:hi_i]
            tri = np.repeat(np.arange(lo_i, hi_i), c)
            local = np.arange(acc) - np.repeat(starts[lo_i:hi_i] - starts[lo_i], c)
            ti = idx[tri]
            w_ = bw[ti]
            cx = xmin[ti] + local % w_
            cy = ymin[ti] + local // w_
            pcx = cx + 0.5
            pcy = cy + 0.5
            ax, ay = x0[ti], y0[ti]
            d = area[ti]
            e1x, e1y = x1[ti] - ax, y1[ti] - ay
            e2x, e2y = x2[ti] - ax, y2[ti] - ay
            qx, qy = pcx - ax, pcy - ay
            b1 = (qx * e2y - qy * e2x) / d
            b2 = (e1x * qy - e1y * qx) / d
            b0 = 1.0 - b1 - b2
            inside = (b0 >= -1e-6) & (b1 >= -1e-6) & (b2 >= -1e-6)
            if mask_mats is not None and inside.any():
                # alpha-mask materials (hair cards, whiskers): drop transparent texels before the z test
                fm = F[ti]
                mm = M[fm[:, 0]]
                cut = mask_mats[mm]
                test = inside & (cut > 0)
                if test.any():
                    k = np.flatnonzero(test)
                    f_ = fm[k]
                    uvk = (surf["uv"][f_[:, 0]] * b0[k, None] + surf["uv"][f_[:, 1]] * b1[k, None] +
                           surf["uv"][f_[:, 2]] * b2[k, None])
                    a_ = np.ones(len(k))
                    for mi_ in np.unique(mm[k]):
                        tx = surf["mats"][int(mi_)].textures.get("baseColor")
                        if tx is not None:
                            s_ = mm[k] == mi_
                            a_[s_] = _pbr.sample(tx, uvk[s_])[:, 3]
                    inside[k[a_ < cut[k]]] = False
            if inside.any():
                ti, b1, b2, b0 = ti[inside], b1[inside], b2[inside], b0[inside]
                pid = (cy[inside] * W + cx[inside])
                f = F[ti]
                zz = (b0 * z[f[:, 0]] + b1 * z[f[:, 1]] + b2 * z[f[:, 2]]).astype(np.float32)
                # z-test: the nearest sample per pixel. A scatter-max into the depth buffer, then the
                # samples that equal it win (ties write the same depth) - about 30x faster than sorting
                # all samples by (pixel, depth), which was the most expensive step of a frame
                np.maximum.at(depth, pid, zz)
                o = np.flatnonzero(zz >= depth[pid])
                pw = pid[o]
                owner[pw] = ti[o]
                bary[pw, 0] = b1[o]
                bary[pw, 1] = b2[o]
            del sel
            lo_i = hi_i
        fg = owner >= 0
        sel = np.flatnonzero(fg)
        rgb = np.zeros((H * W, 3), dtype=np.float32)
        if len(sel):
            f = F[owner[sel]]
            b1 = bary[sel, 0:1]
            b2 = bary[sel, 1:2]
            b0 = 1.0 - b1 - b2
            n = N[f[:, 0]] * b0 + N[f[:, 1]] * b1 + N[f[:, 2]] * b2
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
            col = C[f[:, 0]] * b0 + C[f[:, 1]] * b1 + C[f[:, 2]] * b2
            mat = M[f[:, 0]]
            if surf is None:
                rgb[sel] = shade(n, col, table[mat], lights)
            else:
                uv = surf["uv"]
                uvp = uv[f[:, 0]] * b0 + uv[f[:, 1]] * b1 + uv[f[:, 2]] * b2
                T = surf.get("tan")
                tp = None
                if T is not None:
                    tp = T[f[:, 0]] * b0 + T[f[:, 1]] * b1 + T[f[:, 2]] * b2
                    tp[:, 3] = np.sign(T[f[:, 0], 3]) + (T[f[:, 0], 3] == 0)
                inp = surf["inputs"]
                base, n2, metal, rough, ao, emis, _a = _pbr.surface(surf["mats"], mat, uvp, n, tp, col, inp)
                rgb[sel] = shade(n2, base, table[mat], lights, metal, rough, ao, emis, inp["pbr"][mat])
                unl = inp["unlit"][mat] > 0.5
                if unl.any():
                    rgb[sel[unl]] = np.clip(base[unl], 0, 1)
        fg2 = fg.reshape(H, W)
        zb = depth.reshape(H, W)
        rgb = rgb.reshape(H, W, 3)
        k = max(1, int(round(outline_px * ss)))
        edge = np.zeros_like(fg2)
        for sy, sx in ((0, 1), (1, 0)):
            A_ = (slice(sy, H), slice(sx, W))
            B_ = (slice(0, H - sy), slice(0, W - sx))
            a_z, b_z = zb[A_], zb[B_]
            both = fg2[A_] & fg2[B_] & (np.abs(a_z - b_z) > contour_dz)
            if both.any():
                edge[A_] |= both & (a_z < b_z)
                edge[B_] |= both & (b_z <= a_z)
        if k > 1:
            edge = _dilate(edge, max(1, k // 2)) & fg2
        # 2.2: pin-holes between alpha-masked hair strands are closed (filled with the colour of
        # the strands around them) before the silhouette is traced - otherwise every gap got its
        # own dark outline ring and hair looked spiky and noisy instead of soft
        rc = max(1, int(round(1.5 * ss)))
        closed = ~_dilate(~_dilate(fg2, rc), rc)
        holes = closed & ~fg2
        if holes.any():
            near_h = _neighbour_colour(rgb, fg2, rc)
            rgb[holes] = near_h[holes] * 0.82
            fg2 = closed
        sil = _dilate(fg2, k) & ~fg2
        oc = np.array(outline_rgb, dtype=np.float32)
        # 2.2: lines in a deep shade of the surface's own colour (a ginger cat gets a deep ginger
        # edge, white fur a soft grey one) instead of one flat brown everywhere; inner contours are
        # only a light darkening
        if sil.any():
            near = _neighbour_colour(rgb, fg2, k + 1)
            rgb[sil] = near[sil] * 0.38 * 0.65 + oc * 0.35
        rgb[edge] = rgb[edge] * 0.73 + oc * 0.13
        alpha = (fg2 | sil).astype(np.float32)
        if ss > 1:
            Hs, Ws = H // ss, W // ss
            rgb = rgb[:Hs * ss, :Ws * ss]
            alpha = alpha[:Hs * ss, :Ws * ss]
            prem = (rgb * alpha[..., None]).reshape(Hs, ss, Ws, ss, 3).mean(axis=(1, 3))
            alpha = alpha.reshape(Hs, ss, Ws, ss).mean(axis=(1, 3))
            zs = zb[:Hs * ss:ss, :Ws * ss:ss]
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


def _neighbour_colour(rgb: np.ndarray, mask: np.ndarray, r: int) -> np.ndarray:
    """Average colour of the masked pixels within ``r`` of each pixel (for outline tinting)."""
    acc = np.zeros_like(rgb)
    cnt = np.zeros(mask.shape, dtype=np.float32)
    H, W = mask.shape
    m = mask.astype(np.float32)
    src = rgb * m[..., None]
    for dy in (-r, 0, r):
        for dx in (-r, 0, r):
            ys = slice(max(0, dy), H + min(0, dy))
            yd = slice(max(0, -dy), H + min(0, -dy))
            xs = slice(max(0, dx), W + min(0, dx))
            xd = slice(max(0, -dx), W + min(0, -dx))
            acc[yd, xd] += src[ys, xs]
            cnt[yd, xd] += m[ys, xs]
    return acc / np.maximum(cnt, 1e-6)[..., None]


# --------------------------------------------------------------- renderer
LOD_FOR_TIER = {"low": "low", "medium": "medium", "high": "high", "ultra": "ultra"}


class MeshRenderer3D(Renderer3D):
    """Renders a skinned ``CharacterAsset`` (GLB). Falls back to the procedural model if the asset
    has no usable mesh (``make_renderer`` handles that)."""

    def __init__(self, spec, asset: CharacterAsset, quality: str = "normal", use_gpu: bool = True):
        super().__init__(spec, quality, use_gpu=False)
        self.asset = asset
        self.mode = "production"
        self.kind = self.rig.sk.kind
        self.driver = RigDriver(asset, self.kind, self.species)
        self.driver.rig = self.rig
        self.table = material_table(asset.materials or asset.shading)
        self.inputs = material_inputs(asset.materials)
        self.mesh_raster = MeshRaster(max(1, self.tier["ss"]))
        self.skinning = "gpu"                 # "gpu" (vertex shader) | "cpu" (NumPy) - Playground switch
        self.lights = {"key": True, "fill": True, "rim": True}
        self.face = dict(asset.meta.get("face") or {})
        if self.face.get("eyes_from"):
            self.face.update(auto_decal_face(asset, self.face))
        # the materials this renderer draws with (a private copy of the face texture for sheet faces)
        self.materials = list(asset.materials)
        self.sheet_face = None
        self.expression_picker = None
        if self.face.get("mode") == "sheet":
            self._init_sheet_face()
            self.face_turn = float(self.face.get("turn", 0.9))
        self.painted = None
        if self.face.get("mode") == "painted":
            self._init_painted_face()
        self._mesh_cache = None
        self.backend = "cpu"
        self._glm = None
        if use_gpu:
            try:
                from .render3d_gl import GLSkinRaster
                self._glm = GLSkinRaster.create()
            except Exception:  # noqa: BLE001
                self._glm = None
            if self._glm is not None:
                self.backend = "gpu"
        self._joint_radius = None
        self._last_skin_ms = 0.0
        self._morph_map = self._build_morph_map()
        self._morph_names = set().union(*[set(m.morphs) for m in asset.lods.values()]) if asset.lods else set()
        self.use_clips = True

    # ------------------------------------------------------------ helpers
    def _init_sheet_face(self) -> None:
        """Facial expressions from the character sheet: paint into this renderer's own copy of the
        base-colour texture (the asset may be shared with other renderers)."""
        import copy
        from .face_sheet import ExpressionPicker, make_sheet_face
        sf = make_sheet_face(self.asset, self.face)
        if sf is None:
            return
        for i, md in enumerate(self.materials):
            tx = getattr(md, "textures", None) or {}
            img = tx.get("baseColor")
            if img is None or sf.size is None or tuple(img.shape[:2]) != sf.size:
                continue
            mine = copy.copy(md)
            mine.textures = dict(tx)
            mine.textures["baseColor"] = np.array(img, dtype=np.uint8, copy=True)
            mine.tex_version = 0
            self.materials[i] = mine
            try:
                sf.bind(mine.textures["baseColor"])
            except ValueError:
                return
            self.sheet_face = sf
            self._face_mat = mine
            self.expression_picker = ExpressionPicker(sf.names, sf.default)
            return

    def _own_base_texture(self):
        """A private copy of the base-colour texture of the main material (the asset may be shared
        with other renderers); returns (material copy, texture) or (None, None)."""
        import copy
        for i, md in enumerate(self.materials):
            tx = getattr(md, "textures", None) or {}
            img = tx.get("baseColor")
            if img is None:
                continue
            own = np.array(img, dtype=np.uint8, copy=True)
            first = None
            # every material drawn from the same picture (2.3: the people's skin material shares
            # the body's texture) gets the private copy, so a blink shows on all of them
            for j, mj in enumerate(self.materials):
                tj = getattr(mj, "textures", None) or {}
                if tj.get("baseColor") is not img:
                    continue
                mine = copy.copy(mj)
                mine.textures = dict(tj)
                mine.textures["baseColor"] = own
                mine.tex_version = 0
                self.materials[j] = mine
                self._face_mats = getattr(self, "_face_mats", []) + [mine]
                first = first or mine
            return first, own
        return None, None

    def _bump_face(self) -> None:
        # painted faces say which texture regions changed, so the GPU re-sends only those
        dirty = self.painted.dirty if getattr(self, "painted", None) is not None else None
        for md in getattr(self, "_face_mats", None) or [self._face_mat]:
            md.tex_version = getattr(md, "tex_version", 0) + 1
            md.tex_dirty = dirty

    def _init_painted_face(self) -> None:
        from .face_paint import make_painted_face
        pf = make_painted_face(self.asset, self.face)
        if pf is None:
            return
        mat, tex = self._own_base_texture()
        if tex is None:
            return
        try:
            pf.bind(tex)
        except ValueError as e:
            import logging
            logging.getLogger("focusbuddy.companion").warning("painted face: %s", e)
            return
        self.painted = pf
        self._face_mat = mat

    def set_expression(self, name: str) -> bool:
        """Show a sheet expression now (Playground / tests); True when the face changed."""
        if self.sheet_face is None:
            return False
        name = self.expression_picker.resolve(name)
        self.expression_picker.shown = name
        changed = self.sheet_face.show(name)
        if changed:
            self._bump_face()
        return changed

    def _update_face(self, pose, state) -> None:
        if self.painted is not None:
            fp = self.face_params(pose)
            if self.painted.update(fp["lid"], fp["lid_low"], fp["jaw"]):
                self._bump_face()
            return
        if self.sheet_face is None or getattr(self, "expression_lock", None):
            return
        name = self.expression_picker.pick(pose, state, time.perf_counter())
        if self.sheet_face.show(name):
            self._bump_face()

    @property
    def expression(self) -> str:
        return self.sheet_face.current if self.sheet_face is not None else ""

    def _build_morph_map(self) -> dict:
        """morph target name -> (face parameter, scale). Sources, strongest first:
        ``morph_map`` {"Breathe": {"param": "breath", "scale": 1}}, ``morphs`` {"happy": "mouthSmile"}
        (parameter -> target, the friendlier form), then the built-in ARKit / common names."""
        out = {}
        mm = self.asset.meta.get("morph_map") or {}
        alias = self.asset.meta.get("morphs") or {}
        names = set()
        for lod in self.asset.lods.values():
            names |= set(lod.morphs)
        inv = {}
        for param, target in alias.items():
            for t in ([target] if isinstance(target, str) else list(target)):
                inv[t] = param
        for n in names:
            key = "".join(ch for ch in n.lower() if ch.isalnum())
            if n in mm:
                out[n] = (mm[n].get("param", n), float(mm[n].get("scale", 1.0)))
            elif n in inv:
                out[n] = (inv[n], 1.0)
            elif key in ARKIT_MORPHS:
                out[n] = ARKIT_MORPHS[key]
        return out

    @staticmethod
    def face_params(pose) -> dict:
        """Face / body parameters (0..1) that morph targets are driven by."""
        g = lambda k, d=0.0: float(getattr(pose, k, d))  # noqa: E731
        smile, tilt, brow = g("smile"), g("lid_tilt"), g("brow")
        lid = max(g("lid"), 1.0 - g("eye_open", 1.0))
        wide = max(0.0, g("eye_scale", 1.0) - 1.0) * 3.0
        return {"breath": 0.5 + 0.5 * g("breath") + 0.9 * g("puff"),
                "cheek": max(g("lid_low"), max(0.0, smile) * 0.8),
                "lid": lid, "blink_left": lid, "blink_right": lid,
                "jaw": g("jaw"), "smile": smile, "happy": max(0.0, smile), "frown": max(0.0, -smile),
                "brow_raise": g("brow_raise"), "brow_up": max(0.0, g("brow_raise")) + max(0.0, brow) * 0.5,
                "brow_down": max(0.0, -g("brow_raise")) + max(0.0, -brow) * 0.6,
                "lid_low": g("lid_low"), "mouth_width": g("mouth_width", 1.0) - 1.0, "eye_wide": wide,
                "sad": max(0.0, tilt) * 0.7 + max(0.0, -smile) * 0.5,
                "angry": max(0.0, -tilt) * 0.8 + max(0.0, -brow) * 0.4,
                "surprised": min(1.0, wide + max(0.0, g("brow_raise")) * 0.6),
                "cheek_puff": g("puff")}

    def set_quality(self, quality: str) -> None:
        super().set_quality(quality)
        self.mesh_raster.ss = max(1, self.tier["ss"])

    @property
    def lod_name(self) -> str:
        return LOD_FOR_TIER.get(self.tier_name(self.quality), "medium")

    def current_mesh(self):
        return self.asset.lod(self.lod_name)

    def morph_weights(self, pose) -> dict:
        vals = self.face_params(pose)
        out = {}
        for name, (param, sc) in self._morph_map.items():
            v = vals.get(param, 0.0) * sc
            out[name] = max(0.0, min(1.6, v))
        for name, v in (getattr(self.driver, "clip_morphs", None) or {}).items():   # clip "weights" channels
            if name in out or name in self._morph_names:
                out[name] = max(out.get(name, 0.0), float(v))
        return out

    def head_rotation(self, pose, rp, yaw: float) -> np.ndarray:
        R = rot_z(rp.tilt) @ self._head_frame(pose, yaw)
        if rp.layout.get("sleeping"):
            R = R @ rot_z(10)
        return R

    # -------------------------------------------------------------- paint
    def paint(self, g: QPainter, rect: QRectF, pose, facing: float = 1, state: str | None = None,
              debug: dict | None = None) -> None:
        mesh = self.current_mesh()
        if mesh is None:
            return super().paint(g, rect, pose, facing, state, debug)
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
            if tier["soft_shadow"]:
                g.save()
                g.scale(1.35, 1.25)
                g.setOpacity(0.45)
                self._shadow(g, pose)
                g.restore()
            self._shadow(g, pose)
        t_build = time.perf_counter()
        rp = self.rig_pose(pose)
        cam, yaw = self.camera(pose)
        head_R = self.head_rotation(pose, rp, yaw)
        self._apply_clip(pose)
        self._update_face(pose, state)
        self.driver.view = cam                  # gestures are placed on screen (people's hands)
        Msk = self.driver.pose(pose, rp, head_R)
        self._ground_clamp(mesh, Msk)
        mw = self.morph_weights(pose)
        S, R, piv = self._post(pose, [], {})
        # model -> camera affine: cam . S . (R (p - piv) + piv)
        A = cam @ S @ R
        b = cam @ S @ (piv - R @ piv)
        Sinv = np.diag(1.0 / np.diag(S))
        An = cam @ Sinv @ R
        if tier["shadow"]:
            self._contact_shadows(g, A, b)
        t_raster = time.perf_counter()
        outline_px = tier["outline"] * max(0.75, min(1.25, s / 1.6))     # scale-aware, subtle
        out = None
        skin_ms = 0.0
        gpu_skin = self._glm is not None and self.skinning == "gpu" and len(Msk) <= getattr(self._glm, "max_bones", 0)
        if gpu_skin:
            try:
                out = self._glm.render(mesh, self.materials, self.table, self.inputs, Msk, mw, A, b, An,
                                       self._bounds(mesh, Msk, A, b), s, self.outline_rgb, outline_px * 0.7 / s,
                                       self.lights, max(2, self.mesh_raster.ss))
                if out is not None:
                    self.backend = "gpu-skinning"
            except Exception:  # noqa: BLE001 - driver trouble: CPU skinning from now on
                import logging
                logging.getLogger("focusbuddy.companion").exception("GPU skinning failed - using CPU skinning")
                self.skinning = "cpu"
                out = None
        if out is None:
            ts = time.perf_counter()
            verts, norms, tans = skin(mesh, Msk, mw, tangents=True)
            skin_ms = (time.perf_counter() - ts) * 1000
            if self._glm is not None:
                try:
                    lo, hi = self._vbounds(verts, A, b)
                    out = self._glm.render(mesh, self.materials, self.table, self.inputs, None, {}, A, b, An,
                                           (lo, hi), s, self.outline_rgb, outline_px * 0.7 / s, self.lights,
                                           max(2, self.mesh_raster.ss), skinned=(verts, norms, tans))
                    if out is not None:
                        self.backend = "gpu"
                except Exception:  # noqa: BLE001 - driver trouble: CPU from now on
                    import logging
                    logging.getLogger("focusbuddy.companion").exception("GPU mesh render failed - using CPU")
                    self._glm = None
                    out = None
            if out is None:
                self.backend = "cpu"
                Vc = verts @ A.T + b
                Nc = norms @ An.T
                Nc /= np.maximum(np.linalg.norm(Nc, axis=1, keepdims=True), 1e-9)
                surf = None
                if any(m.textures for m in self.materials) or np.any(self.inputs["pbr"] > 0.5):
                    Tc = np.concatenate([tans[:, :3] @ A.T, tans[:, 3:4]], axis=1) if tans is not None else None
                    mats = self.materials
                    surf = {"uv": mesh.uvs if mesh.uvs is not None else np.zeros((len(Vc), 2)), "tan": Tc,
                            "mats": mats, "inputs": self.inputs,
                            "double": np.array([m.double_sided for m in mats] or [False]),
                            "mask": np.array([m.alpha_cutoff if m.alpha_mode == "MASK" and "baseColor" in m.textures
                                              else 0.0 for m in mats] or [0.0])}
                # CPU fallback: big companions need less supersampling for the same smoothness - keep the
                # sample count (and the frame time) about level as the user scales the companion up
                tss = max(1, int(self.tier["ss"]))
                self.mesh_raster.ss = max(1, min(tss, int(tss * 200.0 / max(1.0, rect.width()))))
                img, (ox, oy) = self.mesh_raster.render(Vc, Nc, mesh.colors, mesh.material, mesh.indices, s,
                                                        self.table, self.outline_rgb, outline_px, self.lights,
                                                        surf=surf)
                h, w = img.shape[:2]
                out = (QImage(img.tobytes(), w, h, w * 4, QImage.Format_ARGB32_Premultiplied).copy(), ox, oy)
        qi, ox, oy = out
        self._last_skin_ms = skin_ms
        g.save()
        g.scale(1 / s, 1 / s)
        g.drawImage(QPointF(ox, oy), qi)
        g.restore()
        t_decal = time.perf_counter()
        # occluders for the face decals: hands / front paws / arms (analytic, same test as procedural)
        self._occluders(pose, cam, S, R, piv)
        facing_sign = 1 if math.cos(math.radians(yaw)) >= 0 else -1
        face = self._face_dict(head_R)
        if self.face.get("mode", "decal") == "decal":
            self._decals(g, pose, face, cam, S, R, piv, facing_sign)
        else:
            u, v, _ = self._project(pose, cam, S, R, piv, face["H"])
            self.head_pos = (u * facing_sign, v)
            self.head_r = face["hr"]
        if pose.ripple > 0:
            self._ripple(g, pose)
        if debug:
            self._debug_mesh(g, pose, rp, cam, S, R, piv, debug)
        if getattr(self, "effects", True):
            self._marks(g, pose, facing_sign)
        g.restore()
        t1 = time.perf_counter()
        self.last_ms = (t1 - t0) * 1000
        st = self.stats
        st.update(total_ms=self.last_ms, build_ms=(t_raster - t_build) * 1000, raster_ms=(t_decal - t_raster) * 1000,
                  decal_ms=(t1 - t_decal) * 1000, prims=int(len(mesh.indices)), triangles=int(len(mesh.indices)),
                  draw_calls=(1 + len(self.materials) if self.backend.startswith("gpu") else 0) + 1 +
                  self._decal_count, backend=self.backend, skinning=("gpu" if self.backend == "gpu-skinning" else "cpu"),
                  skin_ms=self._last_skin_ms, gpu_ms=((t_decal - t_raster) * 1000 - self._last_skin_ms
                                                      if self.backend.startswith("gpu") else 0.0),
                  rig_ms=self.driver.last_ms, quality=self.tier_name(self.quality), lod=self.lod_name,
                  mode="production")

    def _contact_shadows(self, g: QPainter, A: np.ndarray, b: np.ndarray) -> None:
        """Small dark soft spots under the feet / paws that touch the ground (grounding cue)."""
        from PySide6.QtGui import QBrush, QRadialGradient
        d = self.driver
        kind = "biped" if self.kind == "biped" else "quadruped"
        from .character_asset import LIMB_BONES
        g.save()
        g.setPen(Qt.NoPen)
        for name, bones in LIMB_BONES[kind].items():
            if kind == "biped" and name.startswith("arm"):
                continue
            e = d.canon.get(bones[2], -1)
            if e < 0 or e >= len(d.world_T):
                continue
            p = np.array(d.world_T[e], dtype=float)
            ground = d.surface.height_at(float(p[0])) if hasattr(d, "surface") else 0.0
            h = (ground - p[1]) - d._bind_foot_height(e)       # how far the foot is lifted
            k = max(0.0, 1.0 - h / 6.0)
            if k <= 0.02:
                continue
            c = A @ np.array([p[0], ground, p[2]]) + b
            rx, ry = 6.5, 1.8
            grad = QRadialGradient(QPointF(float(c[0]), float(c[1])), rx)
            col = QColor(20, 16, 30, int(70 * k))
            grad.setColorAt(0.0, col)
            col2 = QColor(20, 16, 30, 0)
            grad.setColorAt(1.0, col2)
            g.setBrush(QBrush(grad))
            g.drawEllipse(QPointF(float(c[0]), float(c[1])), rx, ry)
        g.restore()

    def root_motion_px(self, char_px: float) -> float:
        """Forward root motion of the last frame in screen pixels (ROOT_MOTION clips only)."""
        d = getattr(self.driver, "root_delta", None)
        if d is None:
            return 0.0
        return float(d[0]) * char_px / UNITS

    # ------------------------------------------------------------ bounds
    def _ground_clamp(self, mesh, Msk: np.ndarray) -> None:
        """Nothing goes through the floor: when a pose (sitting / lying on long-legged artist rigs)
        pushes part of the mesh below the ground line, the whole character is lifted onto it.
        Uses a fixed sample of vertices skinned on the CPU (cheap) - works for GPU skinning too."""
        if len(Msk) == 0:
            return
        if getattr(self, "_gc_mesh", None) is not mesh:
            n = len(mesh.positions)
            k = min(n, 8000)                                   # every vertex of a typical low-poly model
            idx = np.unique(np.r_[np.linspace(0, n - 1, k).astype(np.int64),
                                  np.argsort(mesh.positions[:, 1])[-64:]])        # + the lowest bind vertices
            # a tail (metadata "tail_floor_frac") bends up off the floor by itself (RigDriver._tail_floor);
            # its thick fur never lifts the whole body
            if (self.asset.meta or {}).get("tail_floor_frac") and self.driver.tail_idx:
                tails = set(self.driver.tail_idx)
                dom = mesh.joints[idx, 0].astype(np.int64)
                idx = idx[~np.isin(dom, list(tails))]
            self._gc_mesh = mesh
            Ph = np.concatenate([mesh.positions[idx], np.ones((len(idx), 1))], axis=1)
            self._gc = (Ph, mesh.joints[idx].astype(np.int64), mesh.weights[idx])
        Ph, Jx, Wt = self._gc
        Y = Ph @ Msk[:, 1, :].T                                    # (n, J): every vertex under every bone
        y = (np.take_along_axis(Y, Jx, axis=1) * Wt).sum(axis=1)
        low = float(y.max())                                       # internal +y is down; ground = 0
        ground = float(getattr(getattr(self.driver, "surface", None), "height_at", lambda _x: 0.0)(0.0))
        if low > ground + 0.25:
            dy = low - ground
            Msk[:, 1, 3] -= dy
            self.driver.world_T[:, 1] -= dy

    def _bounds(self, mesh, Msk: np.ndarray, A: np.ndarray, b: np.ndarray):
        """Camera-space rectangle around the skinned mesh without skinning it: each joint's posed
        position +- the radius of the vertices it carries (computed once per mesh)."""
        if self._joint_radius is None or self._joint_radius[0] is not mesh:
            J = len(self.asset.joints)
            B = np.array([j.bind_world[:3, 3] for j in self.asset.joints]) if J else np.zeros((1, 3))
            dom = mesh.joints[np.arange(len(mesh.joints)), np.argmax(mesh.weights, axis=1)]
            rad = np.zeros(max(1, J))
            d = np.linalg.norm(mesh.positions - B[dom], axis=1)
            np.maximum.at(rad, dom, d)
            used = np.zeros(max(1, J), bool)
            used[np.unique(dom)] = True
            hs = float(getattr(self.driver, "head_scale", 1.0))
            hi = self.driver.canon.get("Head", -1)
            if hs > 1.0 and hi >= 0:
                rad[self.driver._subtree_idx(hi)] *= hs        # the chibi head carries bigger geometry
            self._joint_radius = (mesh, rad * 1.08 + 1.5, used, B)
        _, rad, used, B = self._joint_radius
        pos = np.einsum("jab,jb->ja", Msk[:, :3, :3], B) + Msk[:, :3, 3]
        c = pos[used] @ A.T + b
        k = float(np.abs(A[:2]).sum(axis=1).max())
        r = rad[used] * k
        return (c[:, :2] - r[:, None]).min(axis=0), (c[:, :2] + r[:, None]).max(axis=0)

    @staticmethod
    def _vbounds(verts: np.ndarray, A: np.ndarray, b: np.ndarray):
        c = verts @ A.T + b
        return c[:, :2].min(axis=0), c[:, :2].max(axis=0)

    def _decal(self, g, pose, cam, S, R, piv, pos, normal, size: float, draw, min_facing: float = 0.12,
               occlusion: bool = True):
        # eyelids / lid shapes are painted in the head's colour *as lit at that point*, so they blend
        # into the shaded mesh instead of showing as flat patches
        n = cam @ (S @ np.asarray(normal, dtype=float))
        n = n / (np.linalg.norm(n) or 1.0)
        if n[2] < min_facing:
            return False                                    # facing away: nothing to shade or draw
        lid = self.face.get("lid_color") or super()._lid_color()
        ck = (lid, self.species)
        if getattr(self, "_lid_key", None) != ck:
            qc = QColor(lid)
            key = "skin" if self.species == "buddy" else ("plastic" if self.species == "robot" else "fur")
            self._lid_key = ck
            self._lid_base = np.array([[qc.redF(), qc.greenF(), qc.blueF()]])
            self._lid_table = material_table([key])
        c = _pbr.shade_one(n, self._lid_base[0], self._lid_table[0])
        self._lid_rgb = QColor.fromRgbF(c[0], c[1], c[2]).name()
        try:
            return super()._decal(g, pose, cam, S, R, piv, pos, normal, size, draw, min_facing, occlusion)
        finally:
            self._lid_rgb = None

    def _lid_color(self) -> str:
        return getattr(self, "_lid_rgb", None) or super()._lid_color()

    def clip_for(self, action: str):
        """GLB clip that replaces an action's procedural recipe (metadata "clips": {"WAVE": "Wave"})."""
        name = (self.asset.meta.get("clips") or {}).get((action or "").upper())
        return self.asset.clips.get(name) if name else None

    def _apply_clip(self, pose) -> None:
        self.driver.use_clip_animation = self.use_clips
        clip = self.clip_for(getattr(pose, "action", ""))
        if self.driver.animator is not None and self.driver.animator.has_base and self.use_clips:
            clip = None                       # the clip animator plays action clips as layers
        if clip is None or not self.use_clips:
            self.driver.set_clip(None, 0.0, 0.0)
            return
        mask = (self.asset.meta.get("clip_mask") or {}).get(clip.name, "full")
        self.driver.set_clip(clip, float(getattr(pose, "action_t", 0.0)), float(getattr(pose, "action_w", 0.0)),
                             mask, clip.rest)

    def _face_dict(self, head_R: np.ndarray) -> dict:
        f = self.face
        d = self.driver
        hi = d.canon.get("Head", -1)
        H = d.world_T[hi].copy() if hi >= 0 else np.array([21.0, -53.0, 0.0])
        hs = float(getattr(d, "head_scale", 1.0))
        if f.get("center"):
            H = H + head_R @ (hs * np.asarray(f["center"], dtype=float))
        hr = float(f.get("head_radius", 21.0)) * hs
        m = np.asarray(f.get("mouth", [0.98, 0.5, 0.0]), dtype=float)
        mouth = H + head_R @ (m * hs if f.get("mouth_absolute") else m * hr)
        return {"H": H, "Rh": head_R, "hr": hr, "muzzle": self.kind != "biped",
                "eye_dirs": [tuple(e) for e in f.get("eye_dirs", [(-27.0, 0.02), (27.0, 0.02)])],
                "eye_size": float(f.get("eye_size", 1.3)) * hs, "mouth": mouth,
                "mouth_n": head_R @ np.asarray(f.get("mouth_normal", [1.0, 0.45, 0.0]), dtype=float),
                "whiskers": bool(f.get("whiskers")), "robot": bool(f.get("robot")),
                "cheeks": bool(f.get("cheeks", True)), "mouth_on": bool(f.get("mouth_decal", True)),
                "chest": d.world_T[d.canon["Chest"]] if "Chest" in d.canon else None}

    def _occluders(self, pose, cam, S, R, piv) -> None:
        d = self.driver
        self._cp = []
        pairs = [("LeftLowerArm", "LeftHand"), ("RightLowerArm", "RightHand"), ("LeftUpperArm", "LeftLowerArm"),
                 ("RightUpperArm", "RightLowerArm")]
        for a, b in pairs:
            if a not in d.canon or b not in d.canon:
                continue
            pa, pb = d.world_T[d.canon[a]], d.world_T[d.canon[b]]
            tip = pb + (pb - pa) * 0.35
            p = self._seg(pa, tip, 4.0, (0, 0, 0), 0)
            c = p.c
            ax = p.ax
            if pose.roll:
                c = R @ (c - piv) + piv
                ax = R @ ax
            c = S @ c
            ax2 = S @ ax
            norms = np.linalg.norm(ax2, axis=0)
            pr = Prim(c, ax2 / norms, p.r * norms, (0, 0, 0), 0)
            C = cam @ pr.c
            A = cam @ pr.ax
            self._cp.append((C, A, pr.r, pr.col, False, 0.0, A.T / pr.r[:, None]))
        self._cp_arr = None

    def _debug_mesh(self, g, pose, rp, cam, S, R, piv, opts) -> None:
        d = self.driver
        g.save()

        def P(p):
            u, v, _ = self._project(pose, cam, S, R, piv, p)
            return QPointF(u, v)
        if opts.get("skeleton", True):
            g.setPen(QPen(QColor(91, 108, 255, 230), 1.2))
            for i, par in enumerate(d.parent):
                if par >= 0:
                    g.drawLine(P(d.world_T[par]), P(d.world_T[i]))
            g.setPen(Qt.NoPen)
            g.setBrush(QColor(255, 255, 255, 230))
            for i in range(d.n):
                g.drawEllipse(P(d.world_T[i]), 1.3, 1.3)
        if opts.get("names", False):
            for i, j in enumerate(self.asset.joints):
                q = P(d.world_T[i])
                self.bone_label(g, j.canonical or j.name, (q.x(), q.y()))
        g.restore()
        sub = {k: opts.get(k) for k in ("ik", "contacts")}
        if any(sub.values()):
            Renderer3D._debug3d(self, g, pose, rp, cam, S, R, piv, dict(sub, skeleton=False))


def asset_rig_height(rig) -> float:
    """Height of the rig's neutral pose (used to normalise assets without ``rig_height``)."""
    from .animation import Pose
    rp = rig.solve(Pose())
    hy = rp.joints["head"][1]
    return float(-(hy - rp.layout["hr"]))


__all__ = ["MeshRenderer3D", "MeshRaster", "MATERIALS", "shade", "material_table", "XCOMP", "_clamp"]
