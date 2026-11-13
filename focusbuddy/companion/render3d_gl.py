"""GPU backend for the 3D renderer (OpenGL through Qt, no extra dependency).

Each ellipsoid is drawn as a screen-space quad ("impostor"); the fragment
shader ray-casts the exact ellipsoid, writes the true depth and shades it with
the same toon lighting as the NumPy rasteriser.  Outlines use slightly
inflated, depth-biased "hull" ellipsoids drawn in the outline colour - they
show as silhouettes and as contours wherever a part lies more than a few
units in front of another one.  The frame is rendered super-sampled into an
offscreen framebuffer and read back as a small premultiplied QImage that is
painted into the companion window like any other image (the click-through
transparent overlay stays exactly the same).

If an OpenGL context cannot be created (remote sessions without GL, very old
drivers, headless test runs) ``GLRaster.create()`` returns ``None`` and the
renderer uses the NumPy backend instead.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QImage, QOffscreenSurface, QOpenGLContext, QSurfaceFormat, QVector3D, QVector4D
from PySide6.QtOpenGL import QOpenGLBuffer, QOpenGLFramebufferObject, QOpenGLShader, QOpenGLShaderProgram

log = logging.getLogger("focusbuddy.companion")

GL_FLOAT = 0x1406
GL_TRIANGLES = 0x0004
GL_DEPTH_TEST = 0x0B71
GL_BLEND = 0x0BE2
GL_LESS = 0x0201
GL_COLOR_BUFFER_BIT = 0x4000
GL_DEPTH_BUFFER_BIT = 0x0100

VERT = """
#version 120
attribute vec2 a_uv;
attribute vec3 a_c;
attribute vec3 a_m0;
attribute vec3 a_m1;
attribute vec3 a_m2;
attribute vec4 a_col;
attribute vec3 a_par;
attribute vec3 a_s;
uniform vec4 u_view;
varying vec2 v_uv;
varying vec3 v_c;
varying vec3 v_m0;
varying vec3 v_m1;
varying vec3 v_m2;
varying vec4 v_col;
varying vec3 v_par;
varying vec3 v_s;
void main() {
    v_uv = a_uv; v_c = a_c; v_m0 = a_m0; v_m1 = a_m1; v_m2 = a_m2; v_col = a_col; v_par = a_par; v_s = a_s;
    gl_Position = vec4((a_uv.x - u_view.x) * u_view.z - 1.0, 1.0 - (a_uv.y - u_view.y) * u_view.w, 0.0, 1.0);
}
"""

FRAG = """
#version 120
varying vec2 v_uv;
varying vec3 v_c;
varying vec3 v_m0;
varying vec3 v_m1;
varying vec3 v_m2;
varying vec4 v_col;
varying vec3 v_par;
varying vec3 v_s;
uniform vec3 u_light;
uniform vec3 u_half;
uniform vec3 u_outline;
void main() {
    vec2 d = v_uv - v_c.xy;
    vec3 c0 = vec3(v_m0.x, v_m1.x, v_m2.x);
    vec3 c1 = vec3(v_m0.y, v_m1.y, v_m2.y);
    vec3 m  = vec3(v_m0.z, v_m1.z, v_m2.z);
    vec3 q0 = c0 * d.x + c1 * d.y;
    float a = dot(m, m);
    float b = 2.0 * dot(q0, m);
    float cc = dot(q0, q0) - 1.0;
    float disc = b * b - 4.0 * a * cc;
    if (disc < 0.0) discard;
    if (v_par.x > 0.5) {
        // outline hull: only the ring outside the part's own silhouette, at the silhouette depth
        vec3 qi = q0 * v_s;
        vec3 mi = m * v_s;
        float ai = dot(mi, mi);
        float bi = 2.0 * dot(qi, mi);
        float ci = dot(qi, qi) - 1.0;
        if (bi * bi - 4.0 * ai * ci >= 0.0) discard;
        gl_FragDepth = clamp(0.5 - (v_c.z - v_par.z) / 2000.0, 0.0, 1.0);
        gl_FragColor = vec4(u_outline, 1.0);
        return;
    }
    float w = (sqrt(disc) - b) / (2.0 * a);
    float z = v_c.z + w;
    gl_FragDepth = clamp(0.5 - z / 2000.0, 0.0, 1.0);
    vec3 q = q0 + m * w;
    vec3 n = normalize(vec3(dot(c0, q), dot(c1, q), dot(m, q)));
    vec3 base = v_col.rgb;
    if (v_col.a > 0.5) { gl_FragColor = vec4(base, 1.0); return; }
    float ndl = dot(n, u_light);
    float t = clamp((ndl - 0.02) / 0.32, 0.0, 1.0);
    float toon = t * t * (3.0 - 2.0 * t);
    float shade = 0.6 + 0.3 * toon + 0.12 * clamp(ndl, 0.0, 1.0) + clamp(-ndl, 0.0, 1.0) * 0.08;
    float r1 = clamp(1.0 - n.z, 0.0, 1.0);
    float rim = r1 * r1 * r1 * 0.22;
    float ndh = clamp(dot(n, u_half), 0.0, 1.0);
    float spec = pow(ndh, 28.0) * v_par.y * (toon > 0.5 ? 1.0 : 0.0);
    gl_FragColor = vec4(clamp(base * shade + vec3(rim + spec), 0.0, 1.0), 1.0);
}
"""


class GLRaster:
    """OpenGL impostor renderer. Use ``GLRaster.create()``; it returns None if GL is unavailable."""

    _shared = None
    _failed = False

    def __init__(self):
        self.ctx = None
        self.surface = None
        self.program = None
        self.vbo = None
        self.fbo = None
        self.fbo_size = QSize(0, 0)
        self.ss = 2

    @classmethod
    def create(cls):
        if cls._failed:
            return None
        if cls._shared is not None:
            return cls._shared
        try:
            r = cls()
            if not r._init():
                cls._failed = True
                return None
            cls._shared = r
            return r
        except Exception:  # noqa: BLE001
            log.exception("OpenGL backend unavailable - using the CPU renderer")
            cls._failed = True
            return None

    def _init(self) -> bool:
        fmt = QSurfaceFormat()
        fmt.setRenderableType(QSurfaceFormat.OpenGL)
        fmt.setProfile(QSurfaceFormat.CompatibilityProfile)
        fmt.setVersion(2, 1)
        self.ctx = QOpenGLContext()
        self.ctx.setFormat(fmt)
        if not self.ctx.create():
            return False
        self.surface = QOffscreenSurface()
        self.surface.setFormat(self.ctx.format())
        self.surface.create()
        if not self.surface.isValid() or not self.ctx.makeCurrent(self.surface):
            return False
        try:
            p = QOpenGLShaderProgram()
            if not p.addShaderFromSourceCode(QOpenGLShader.Vertex, VERT):
                log.warning("GL vertex shader: %s", p.log())
                return False
            if not p.addShaderFromSourceCode(QOpenGLShader.Fragment, FRAG):
                log.warning("GL fragment shader: %s", p.log())
                return False
            if not p.link():
                log.warning("GL link: %s", p.log())
                return False
            self.program = p
            self.vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
            self.vbo.create()
            self.vbo.setUsagePattern(QOpenGLBuffer.StreamDraw)
            return True
        finally:
            self.ctx.doneCurrent()

    # ----------------------------------------------------------------------------
    def render(self, cprims: list, scale: float, outline_rgb: tuple, light, half, outline_units: float,
               ss: int = 2):
        """``cprims``: [(C (3,), A (3x3 axes as columns), r (3,), col, em, spec)] in camera space.
        Returns (QImage premultiplied, ox_px, oy_px) or None on failure."""
        if not cprims:
            return None
        C = np.array([c[0] for c in cprims], dtype=np.float64)              # (N, 3)
        A = np.array([c[1] for c in cprims], dtype=np.float64)              # (N, 3, 3) axes as columns
        R = np.array([c[2] for c in cprims], dtype=np.float64)              # (N, 3)
        col = np.array([c[3] for c in cprims], dtype=np.float64)
        em = np.array([1.0 if c[4] else 0.0 for c in cprims])
        spec = np.array([c[5] for c in cprims], dtype=np.float64)
        n = len(cprims)
        bias = 0.6
        blocks = []
        lo_u = lo_v = 1e9
        hi_u = hi_v = -1e9
        for hull in (1, 0):
            rr = R + outline_units if hull else R
            M = np.transpose(A, (0, 2, 1)) / rr[:, :, None]                 # rows of camera -> unit sphere
            hx = np.sqrt(((A[:, 0, :] * rr) ** 2).sum(axis=1))
            hy = np.sqrt(((A[:, 1, :] * rr) ** 2).sum(axis=1))
            u0, u1, v0, v1 = C[:, 0] - hx, C[:, 0] + hx, C[:, 1] - hy, C[:, 1] + hy
            lo_u, hi_u = min(lo_u, float(u0.min())), max(hi_u, float(u1.max()))
            lo_v, hi_v = min(lo_v, float(v0.min())), max(hi_v, float(v1.max()))
            sc = rr / R
            attrs = np.concatenate([C, M.reshape(n, 9), col, em[:, None], np.full((n, 1), float(hull)),
                                    spec[:, None], np.full((n, 1), bias if hull else 0.0), sc], axis=1)
            corners_u = np.stack([u0, u1, u1, u0, u1, u0], axis=1)            # two triangles per quad
            corners_v = np.stack([v0, v0, v1, v0, v1, v1], axis=1)
            uv = np.stack([corners_u, corners_v], axis=2).reshape(n * 6, 2)
            blocks.append(np.concatenate([uv, np.repeat(attrs, 6, axis=0)], axis=1))
        data = np.ascontiguousarray(np.concatenate(blocks, axis=0), dtype=np.float32)
        nverts = data.shape[0]
        pad = 2.0 / scale
        ox = math.floor((lo_u - pad) * scale)
        oy = math.floor((lo_v - pad) * scale)
        W = max(1, math.ceil((hi_u + pad) * scale) - ox)
        H = max(1, math.ceil((hi_v + pad) * scale) - oy)
        Ws, Hs = W * ss, H * ss
        if not self.ctx.makeCurrent(self.surface):
            return None
        try:
            if self.fbo is None or self.fbo_size.width() < Ws or self.fbo_size.height() < Hs:
                fw = max(64, (Ws + 63) // 64 * 64)
                fh = max(64, (Hs + 63) // 64 * 64)
                self.fbo = QOpenGLFramebufferObject(fw, fh, QOpenGLFramebufferObject.Depth)
                self.fbo_size = QSize(fw, fh)
            fbo = self.fbo
            fbo.bind()
            f = self.ctx.functions()
            f.glViewport(0, fbo.height() - Hs, Ws, Hs)
            f.glClearColor(0.0, 0.0, 0.0, 0.0)
            f.glClearDepthf(1.0) if hasattr(f, "glClearDepthf") else None
            f.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            f.glEnable(GL_DEPTH_TEST)
            f.glDepthFunc(GL_LESS)
            f.glDisable(GL_BLEND)
            p = self.program
            p.bind()
            p.setUniformValue("u_view", QVector4D(ox / scale, oy / scale, 2.0 * scale / W, 2.0 * scale / H))
            p.setUniformValue("u_light", QVector3D(*map(float, light)))
            p.setUniformValue("u_half", QVector3D(*map(float, half)))
            p.setUniformValue("u_outline", QVector3D(*map(float, outline_rgb)))
            self.vbo.bind()
            raw = data.tobytes()
            self.vbo.allocate(raw, len(raw))
            stride = data.shape[1] * 4
            layout = (("a_uv", 0, 2), ("a_c", 2, 3), ("a_m0", 5, 3), ("a_m1", 8, 3), ("a_m2", 11, 3),
                      ("a_col", 14, 4), ("a_par", 18, 3), ("a_s", 21, 3))
            for name, off, size in layout:
                p.enableAttributeArray(name)
                p.setAttributeBuffer(name, GL_FLOAT, off * 4, size, stride)
            f.glDrawArrays(GL_TRIANGLES, 0, nverts)
            for name, _, _ in layout:
                p.disableAttributeArray(name)
            self.vbo.release()
            p.release()
            fbo.release()
            img = fbo.toImage(True)
        finally:
            self.ctx.doneCurrent()
        img = img.copy(0, 0, Ws, Hs)
        if ss > 1:
            img = img.scaled(W, H, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        if img.format() != QImage.Format_ARGB32_Premultiplied:
            img = img.convertToFormat(QImage.Format_ARGB32_Premultiplied)
        return img, ox, oy


# ============================================================ skinned meshes (GPU skinning)
SKIN_VERT = """
#version 120
attribute vec3 a_pos;
attribute vec3 a_nrm;
attribute vec4 a_tan;
attribute vec2 a_uv;
attribute vec3 a_col;
attribute vec4 a_j;
attribute vec4 a_w;
attribute vec3 a_md0;
attribute vec3 a_md1;
attribute vec3 a_md2;
attribute vec3 a_md3;
attribute vec3 a_mn;
uniform vec4 u_bones[%(rows)d];
uniform float u_skin;
uniform vec4 u_mw;
uniform vec3 u_A0;
uniform vec3 u_A1;
uniform vec3 u_A2;
uniform vec3 u_b;
uniform vec3 u_N0;
uniform vec3 u_N1;
uniform vec3 u_N2;
uniform vec4 u_view;
uniform float u_hull;
uniform float u_outline;
varying vec3 v_nrm;
varying vec4 v_tan;
varying vec2 v_uv;
varying vec3 v_col;
void main() {
    vec3 p = a_pos + a_md0 * u_mw.x + a_md1 * u_mw.y + a_md2 * u_mw.z + a_md3 * u_mw.w;
    vec3 n = a_nrm + a_mn;
    vec3 t = a_tan.xyz;
    if (u_skin > 0.5) {
        int i0 = int(a_j.x + 0.5) * 3;
        int i1 = int(a_j.y + 0.5) * 3;
        int i2 = int(a_j.z + 0.5) * 3;
        int i3 = int(a_j.w + 0.5) * 3;
        vec4 r0 = u_bones[i0] * a_w.x + u_bones[i1] * a_w.y + u_bones[i2] * a_w.z + u_bones[i3] * a_w.w;
        vec4 r1 = u_bones[i0 + 1] * a_w.x + u_bones[i1 + 1] * a_w.y + u_bones[i2 + 1] * a_w.z + u_bones[i3 + 1] * a_w.w;
        vec4 r2 = u_bones[i0 + 2] * a_w.x + u_bones[i1 + 2] * a_w.y + u_bones[i2 + 2] * a_w.z + u_bones[i3 + 2] * a_w.w;
        vec4 P = vec4(p, 1.0);
        p = vec3(dot(r0, P), dot(r1, P), dot(r2, P));
        n = vec3(dot(r0.xyz, n), dot(r1.xyz, n), dot(r2.xyz, n));
        t = vec3(dot(r0.xyz, t), dot(r1.xyz, t), dot(r2.xyz, t));
    }
    vec3 c = vec3(dot(u_A0, p), dot(u_A1, p), dot(u_A2, p)) + u_b;
    vec3 cn = normalize(vec3(dot(u_N0, n), dot(u_N1, n), dot(u_N2, n)) + vec3(0.0, 0.0, 1e-6));
    vec3 ct = vec3(dot(u_A0, t), dot(u_A1, t), dot(u_A2, t));
    if (u_hull > 0.5) c += cn * u_outline;
    v_nrm = cn; v_tan = vec4(ct, a_tan.w); v_uv = a_uv; v_col = a_col;
    float z = clamp(0.5 - c.z / 2000.0, 0.0, 1.0) + (u_hull > 0.5 ? 0.0012 : 0.0);
    gl_Position = vec4((c.x - u_view.x) * u_view.z - 1.0, 1.0 - (c.y - u_view.y) * u_view.w, z * 2.0 - 1.0, 1.0);
}
"""

SKIN_FRAG = """
#version 120
varying vec3 v_nrm;
varying vec4 v_tan;
varying vec2 v_uv;
varying vec3 v_col;
uniform float u_hull;
uniform vec3 u_outline_col;
uniform vec3 u_light;
uniform vec3 u_half;
uniform vec3 u_fill;
uniform vec3 u_lights;
uniform vec4 u_m0;      // wrap, toon, spec, power
uniform vec4 u_m1;      // rim, sheen, sss, metal
uniform float u_m2;     // stylised emissive (unlit)
uniform vec4 u_pbr;     // pbr flag, metallic, roughness, unlit
uniform vec3 u_emis;
uniform vec4 u_has;     // base, normal, metallic-roughness, occlusion
uniform vec4 u_has2;    // emissive, normal scale, occlusion strength, alpha cutoff (0 = opaque)
uniform vec4 u_basef;   // base colour alpha factor (x)
uniform sampler2D t_base;
uniform sampler2D t_nrm;
uniform sampler2D t_mr;
uniform sampler2D t_occ;
uniform sampler2D t_em;
void main() {
    if (u_hull > 0.5) {
        // outline in a deep shade of the surface's own colour (not one flat brown everywhere)
        vec3 hb = v_col;
        if (u_has.x > 0.5) hb *= texture2D(t_base, v_uv).rgb;
        gl_FragColor = vec4(mix(hb * 0.38, u_outline_col, 0.35), 1.0);
        return;
    }
    vec3 base = v_col;
    float alpha = u_basef.x;
    if (u_has.x > 0.5) { vec4 tb = texture2D(t_base, v_uv); base *= tb.rgb; alpha *= tb.a; }
    if (u_has2.w > 0.0 && alpha < u_has2.w) discard;
    if (u_pbr.w > 0.5) { gl_FragColor = vec4(clamp(base, 0.0, 1.0), 1.0); return; }
    vec3 n = normalize(v_nrm);
    if (u_has.y > 0.5) {
        vec3 tn = texture2D(t_nrm, v_uv).xyz * 2.0 - 1.0;
        tn.xy *= u_has2.y;
        vec3 T = v_tan.xyz - n * dot(n, v_tan.xyz);
        if (dot(T, T) > 1e-12) {
            T = normalize(T);
            vec3 B = cross(n, T) * (v_tan.w < 0.0 ? -1.0 : 1.0);
            n = normalize(T * tn.x + B * tn.y + n * tn.z);
        }
    }
    float metal = u_pbr.x > 0.5 ? u_pbr.y : u_m1.w;
    float rough = u_pbr.z;
    if (u_has.z > 0.5) { vec4 mr = texture2D(t_mr, v_uv); rough *= mr.g; if (u_pbr.x > 0.5) metal *= mr.b; }
    float ao = 1.0;
    if (u_has.w > 0.5) ao = 1.0 + (texture2D(t_occ, v_uv).r - 1.0) * u_has2.z;
    vec3 emis = u_emis;
    if (u_has2.x > 0.5) emis *= texture2D(t_em, v_uv).rgb;
    float ndl = dot(n, u_light);
    float w = clamp((ndl + u_m0.x) / (1.0 + u_m0.x), 0.0, 1.0);
    float t = clamp(w / 0.6, 0.0, 1.0);
    float tn2 = t * t * (3.0 - 2.0 * t);
    float diff = (w + (tn2 - w) * u_m0.y) * u_lights.x;
    float hemi = 0.5 - 0.5 * n.y;
    vec3 amb = mix(vec3(0.72, 0.66, 0.66), vec3(1.0, 1.0, 1.04), hemi);
    float fdl = clamp(dot(n, u_fill), 0.0, 1.0) * u_lights.y;
    vec3 light = amb * 0.44 * ao + vec3(1.0, 0.97, 0.92) * diff * 0.70 * (0.55 + 0.45 * ao)
               + vec3(0.80, 0.86, 1.0) * fdl * 0.18 * ao;
    vec3 col = base * light;
    if (metal > 0.0) {
        vec3 env = (0.55 + 0.45 * hemi) * vec3(0.92, 0.95, 1.0);
        float avg = (base.r + base.g + base.b) / 3.0;
        vec3 mcol = base * (0.30 + 0.55 * diff) + env * 0.32 * avg;
        col = mix(col, mcol, metal);
    }
    col += base * (1.0 - diff) * u_m1.z * vec3(0.30, 0.08, 0.04);
    float edge = clamp(1.0 - n.z, 0.0, 1.0);
    col += base * u_m1.y * edge * edge * 0.6;
    float ndh = clamp(dot(n, u_half), 0.0, 1.0);
    float sp;
    if (u_pbr.x > 0.5) {
        float r = clamp(rough, 0.08, 1.0);
        float a2 = r * r * r * r;
        float d = ndh * ndh * (a2 - 1.0) + 1.0;
        float D = a2 / (3.14159265 * d * d);
        float vdh = max(1e-3, u_half.z);
        float fres = pow(1.0 - vdh, 5.0);
        float F0 = 0.04 + (dot(base, vec3(1.0 / 3.0)) - 0.04) * metal;
        float F = F0 + (1.0 - F0) * fres;
        sp = min(D * F * 0.25, 0.55) * clamp(ndl, 0.0, 1.0) * u_lights.x * (0.6 + 0.4 * ao);
    } else {
        sp = (w > 0.25 ? u_m0.z * pow(ndh, u_m0.w) : 0.0) * u_lights.x;
    }
    float rim = u_m1.x * edge * edge * edge * u_lights.z * (0.6 + 0.4 * hemi);
    col += vec3(sp) + rim * (vec3(1.0, 0.95, 0.88) * 0.6 + base * 0.4);
    col = mix(col, base, u_m2);
    col += emis;
    // gentle filmic grade (pbr.grade): soft S-curve, a touch more colour
    vec3 c = clamp(col, 0.0, 1.0);
    c = c + 0.10 * c * (1.0 - c) * (2.0 * c - 1.0) * 2.0;
    float lum = dot(c, vec3(0.299, 0.587, 0.114));
    gl_FragColor = vec4(clamp(vec3(lum) + (c - vec3(lum)) * 1.07, 0.0, 1.0), 1.0);
}
"""

GL_TEXTURE_SLOTS = (("baseColor", "t_base"), ("normal", "t_nrm"), ("metallicRoughness", "t_mr"),
                    ("occlusion", "t_occ"), ("emissive", "t_em"))
STATIC_LAYOUT = (("a_pos", 0, 3), ("a_nrm", 3, 3), ("a_tan", 6, 4), ("a_uv", 10, 2), ("a_col", 12, 3),
                 ("a_j", 15, 4), ("a_w", 19, 4))
STATIC_FLOATS = 23


class _GPUMesh:
    """Static GPU copy of one LOD: expanded triangles sorted by material + morph-target buffers."""

    def __init__(self):
        self.vbo = None
        self.count = 0
        self.ranges: list = []              # (material index, first vertex, vertex count)
        self.order = None                   # expanded vertex -> mesh vertex
        self.morphs: dict = {}              # name -> QOpenGLBuffer of (count, 3) deltas
        self.dyn = None                     # CPU-skinning stream buffer (pos, nrm, tan)
        self.zero = None                    # zero buffer for unused morph slots
        self.comb = None                    # combined morph deltas (more than four active morphs)
        self.ncomb = None                   # combined morph normal deltas
        self.ncomb_key = None


class GLSkinRaster(GLRaster):
    """OpenGL renderer for ``CharacterAsset`` meshes.

    * GPU skinning: the mesh lives in static vertex buffers; each frame only the bone matrices
      (3 x vec4 rows per joint, up to ``max_bones``), up to four active morph weights and the camera
      are sent. The vertex shader skins (4 influences), applies morphs and projects.
    * CPU skinning mode: the same program draws vertices skinned in NumPy (``u_skin = 0``) - used for
      comparison / as a fallback when a skeleton has more joints than the GPU allows.
    Materials: base colour, normal (tangent space), metallic-roughness, occlusion and emissive
    textures, alpha mask, unlit, the stylised toy shading of ``pbr.shade`` (mirrored in GLSL).
    """

    _shared = None
    _failed = False
    MAX_BONES = (128, 64)

    def _init(self) -> bool:
        fmt = QSurfaceFormat()
        fmt.setRenderableType(QSurfaceFormat.OpenGL)
        fmt.setProfile(QSurfaceFormat.CompatibilityProfile)
        fmt.setVersion(2, 1)
        self.ctx = QOpenGLContext()
        self.ctx.setFormat(fmt)
        if not self.ctx.create():
            return False
        self.surface = QOffscreenSurface()
        self.surface.setFormat(self.ctx.format())
        self.surface.create()
        if not self.surface.isValid() or not self.ctx.makeCurrent(self.surface):
            return False
        try:
            for mb in self.MAX_BONES:
                p = QOpenGLShaderProgram()
                ok = p.addShaderFromSourceCode(QOpenGLShader.Vertex, SKIN_VERT % {"rows": mb * 3}) and \
                    p.addShaderFromSourceCode(QOpenGLShader.Fragment, SKIN_FRAG) and p.link()
                if ok:
                    self.program = p
                    self.max_bones = mb
                    break
                log.info("GPU skinning with %d bones not available: %s", mb, p.log())
            if self.program is None:
                return False
            p = self.program
            self._bone_loc = [p.uniformLocation(f"u_bones[{i}]") for i in range(self.max_bones * 3)]
            if self._bone_loc[0] < 0:
                return False
            self.meshes: dict = {}
            self.textures: dict = {}
            self._tex_versions: dict = {}
            self.renderer_name = ""
            try:
                from PySide6.QtCore import QCoreApplication
                app = QCoreApplication.instance()
                if app is not None:
                    app.aboutToQuit.connect(self.release)
            except Exception:  # noqa: BLE001
                pass
            try:
                from PySide6.QtGui import QOpenGLFunctions  # noqa: F401
                f = self.ctx.functions()
                self.renderer_name = str(f.glGetString(0x1F01) or "")      # GL_RENDERER
            except Exception:  # noqa: BLE001
                pass
            return True
        finally:
            self.ctx.doneCurrent()

    def release(self) -> None:
        """Free textures / buffers while the context is current (at application exit)."""
        try:
            if self.ctx is not None and self.ctx.makeCurrent(self.surface):
                for t in self.textures.values():
                    t.destroy()
                for g in self.meshes.values():
                    for b in [g.vbo, g.dyn, g.zero, g.ncomb, *g.morphs.values()]:
                        if b is not None:
                            b.destroy()
                self.textures.clear()
                self.meshes.clear()
                self.ctx.doneCurrent()
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------ uploads
    def _buffer(self, arr: np.ndarray, usage=QOpenGLBuffer.StaticDraw):
        b = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        b.create()
        b.setUsagePattern(usage)
        b.bind()
        raw = np.ascontiguousarray(arr, dtype=np.float32).tobytes()
        b.allocate(raw, len(raw))
        b.release()
        return b

    def _upload(self, mesh) -> _GPUMesh:
        g = self.meshes.get(id(mesh))
        if g is not None and g.src is mesh:
            return g
        g = _GPUMesh()
        g.src = mesh
        F = np.asarray(mesh.indices, dtype=np.int64)
        mat = np.asarray(mesh.material, dtype=np.int64)[F[:, 0]]
        order_t = np.argsort(mat, kind="stable")
        F = F[order_t]
        mat = mat[order_t]
        g.order = F.reshape(-1)
        g.count = int(g.order.size)
        n = len(mesh.positions)
        uv = mesh.uvs if getattr(mesh, "uvs", None) is not None else np.zeros((n, 2))
        tan = mesh.tangents if getattr(mesh, "tangents", None) is not None else np.zeros((n, 4))
        data = np.concatenate([mesh.positions, mesh.normals, tan, uv, mesh.colors, mesh.joints.astype(np.float32),
                               mesh.weights], axis=1).astype(np.float32)
        g.vbo = self._buffer(data[g.order])
        starts = np.flatnonzero(np.r_[True, mat[1:] != mat[:-1]])
        for k, st in enumerate(starts):
            end = starts[k + 1] if k + 1 < len(starts) else len(mat)
            g.ranges.append((int(mat[st]), int(st) * 3, int(end - st) * 3))
        for name, d in mesh.morphs.items():
            g.morphs[name] = self._buffer(np.asarray(d, dtype=np.float32)[g.order])
        g.zero = self._buffer(np.zeros((g.count, 3), dtype=np.float32))
        self.meshes = {k: v for k, v in self.meshes.items() if v.src is not None and k != id(mesh)}
        self.meshes[id(mesh)] = g
        return g

    def _texture(self, key, img: np.ndarray, srgb_data: bool = True, version: int = 0):
        t = self.textures.get(key)
        src = getattr(self, "_tex_src", None)
        if src is None:
            src = self._tex_src = {}
        if t is not None and self._tex_versions.get(key, 0) == version and src.get(key) is img:
            return t
        if t is not None:                       # repainted on the CPU (sheet faces): upload again
            try:
                t.destroy()
            except Exception:  # noqa: BLE001
                pass
        from PySide6.QtGui import QImage
        from PySide6.QtOpenGL import QOpenGLTexture
        a = np.ascontiguousarray(img, dtype=np.uint8)
        h, w = a.shape[:2]
        qi = QImage(a.tobytes(), w, h, w * 4, QImage.Format_RGBA8888).copy()
        t = QOpenGLTexture(qi, QOpenGLTexture.GenerateMipMaps)
        t.setMinificationFilter(QOpenGLTexture.LinearMipMapLinear)
        t.setMagnificationFilter(QOpenGLTexture.Linear)
        t.setWrapMode(QOpenGLTexture.Repeat)
        self.textures[key] = t
        self._tex_versions[key] = version
        src[key] = img                          # held, so the key's id can never be reused meanwhile
        return t

    # ------------------------------------------------------------ render
    def render(self, mesh, mats: list, table: np.ndarray, inputs: dict, bones: Optional[np.ndarray],
               morph_weights: dict, A: np.ndarray, b: np.ndarray, An: np.ndarray, bounds: tuple, scale: float,
               outline_rgb, outline_units: float, lights: dict, ss: int = 2, skinned: Optional[tuple] = None):
        """Draw ``mesh``. ``bones`` (J, 4, 4) skinning matrices -> GPU skinning; or ``skinned`` =
        (V, N, T) already skinned in model space (CPU skinning) with ``bones=None``.
        ``A``, ``b`` map model -> camera space, ``An`` the normal matrix; ``bounds`` = (lo_xy, hi_xy)
        camera-space rectangle that contains the character. Returns (QImage, ox, oy) or None."""
        from .render3d import HALF, LIGHT
        from .pbr import FILL
        if mesh is None or len(mesh.indices) == 0:
            return None
        if bones is not None and len(bones) > self.max_bones:
            return None
        lo, hi = bounds
        pad = 2.0 / scale + outline_units * 2
        ox = math.floor((lo[0] - pad) * scale)
        oy = math.floor((lo[1] - pad) * scale)
        W = max(1, math.ceil((hi[0] + pad) * scale) - ox)
        H = max(1, math.ceil((hi[1] + pad) * scale) - oy)
        Ws, Hs = W * ss, H * ss
        if not self.ctx.makeCurrent(self.surface):
            return None
        try:
            g = self._upload(mesh)
            if self.fbo is None or self.fbo_size.width() < Ws or self.fbo_size.height() < Hs:
                fw = max(64, (Ws + 63) // 64 * 64)
                fh = max(64, (Hs + 63) // 64 * 64)
                self.fbo = QOpenGLFramebufferObject(fw, fh, QOpenGLFramebufferObject.Depth)
                self.fbo_size = QSize(fw, fh)
            fbo = self.fbo
            fbo.bind()
            f = self.ctx.functions()
            f.glViewport(0, fbo.height() - Hs, Ws, Hs)
            f.glClearColor(0.0, 0.0, 0.0, 0.0)
            f.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            f.glEnable(GL_DEPTH_TEST)
            f.glDepthFunc(GL_LESS)
            f.glDisable(GL_BLEND)
            p = self.program
            p.bind()
            p.setUniformValue("u_view", QVector4D(ox / scale, oy / scale, 2.0 * scale / W, 2.0 * scale / H))
            for nm, M3 in (("u_A", A), ("u_N", An)):
                for r in range(3):
                    p.setUniformValue(f"{nm}{r}", QVector3D(*map(float, M3[r])))
            p.setUniformValue("u_b", QVector3D(*map(float, b)))
            p.setUniformValue("u_light", QVector3D(*map(float, LIGHT)))
            p.setUniformValue("u_half", QVector3D(*map(float, HALF)))
            p.setUniformValue("u_fill", QVector3D(*map(float, FILL)))
            p.setUniformValue("u_lights", QVector3D(float(lights.get("key", True)), float(lights.get("fill", True)),
                                                    float(lights.get("rim", True))))
            p.setUniformValue("u_outline_col", QVector3D(*map(float, outline_rgb)))
            p.setUniformValue1f("u_outline", float(outline_units))
            # bones (rows of the 3x4 skinning matrices)
            if bones is not None:
                p.setUniformValue1f("u_skin", 1.0)
                rows = np.asarray(bones, dtype=np.float64)[:, :3, :]
                loc = self._bone_loc
                for j in range(len(rows)):
                    for r in range(3):
                        p.setUniformValue(loc[j * 3 + r], QVector4D(*map(float, rows[j, r])))
            else:
                p.setUniformValue1f("u_skin", 0.0)
            # morphs: the four strongest go to the shader; with more active (a face rig), all of
            # them are summed on the CPU into one combined delta buffer (one slot, weight 1)
            src_weights = morph_weights
            active = sorted(((abs(w), k) for k, w in (morph_weights or {}).items() if k in g.morphs and abs(w) > 1e-4),
                            reverse=True)
            if len(active) > 4 and skinned is None:
                src = getattr(mesh, "morphs", {})
                comb = None
                for _w, k in active:
                    d = np.asarray(src[k], dtype=np.float32) * float(morph_weights[k])
                    comb = d if comb is None else comb + d
                comb = np.ascontiguousarray(comb[g.order], dtype=np.float32)
                if getattr(g, "comb", None) is None:
                    g.comb = self._buffer(comb, QOpenGLBuffer.StreamDraw)
                else:
                    g.comb.bind()
                    raw = comb.tobytes()
                    g.comb.allocate(raw, len(raw))
                    g.comb.release()
                g.morphs["__combined__"] = g.comb
                morph_weights = dict(morph_weights, __combined__=1.0)
                active = [(1.0, "__combined__")]
            active = active[:4]
            mw = [0.0, 0.0, 0.0, 0.0]
            g.vbo.bind()
            stride = STATIC_FLOATS * 4
            for name, off, size in STATIC_LAYOUT:
                p.enableAttributeArray(name)
                p.setAttributeBuffer(name, GL_FLOAT, off * 4, size, stride)
            g.vbo.release()
            if skinned is not None:
                V, N, T = skinned
                dyn = np.concatenate([V, N, T], axis=1).astype(np.float32)[g.order]
                if g.dyn is None:
                    g.dyn = self._buffer(dyn, QOpenGLBuffer.StreamDraw)
                else:
                    g.dyn.bind()
                    raw = np.ascontiguousarray(dyn).tobytes()
                    g.dyn.allocate(raw, len(raw))
                    g.dyn.release()
                g.dyn.bind()
                for name, off, size in (("a_pos", 0, 3), ("a_nrm", 3, 3), ("a_tan", 6, 4)):
                    p.setAttributeBuffer(name, GL_FLOAT, off * 4, size, 10 * 4)
                g.dyn.release()
                active = []                                     # morphs already applied on the CPU
            for slot in range(4):
                name = f"a_md{slot}"
                buf = g.morphs[active[slot][1]] if slot < len(active) else g.zero
                if slot < len(active):
                    mw[slot] = float(morph_weights[active[slot][1]])
                buf.bind()
                p.enableAttributeArray(name)
                p.setAttributeBuffer(name, GL_FLOAT, 0, 3, 12)
                buf.release()
            p.setUniformValue("u_mw", QVector4D(*mw))
            # morph normal deltas (a morph that smooths a shape also smooths its shading): all
            # active ones summed on the CPU into one buffer, re-uploaded only when the weights change
            nbuf = g.zero
            mn = getattr(mesh, "morph_normals", None) or {}
            if skinned is None and mn:
                wts = tuple(sorted((k, round(float(w), 3)) for k, w in (src_weights or {}).items()
                                   if k in mn and abs(w) > 1e-4))
                if wts:
                    if wts != g.ncomb_key:
                        acc = None
                        for k, w in wts:
                            d = np.asarray(mn[k], dtype=np.float32) * w
                            acc = d if acc is None else acc + d
                        raw = np.ascontiguousarray(acc[g.order], dtype=np.float32).tobytes()
                        if g.ncomb is None:
                            g.ncomb = self._buffer(acc[g.order], QOpenGLBuffer.StreamDraw)
                        else:
                            g.ncomb.bind()
                            g.ncomb.allocate(raw, len(raw))
                            g.ncomb.release()
                        g.ncomb_key = wts
                    nbuf = g.ncomb
            nbuf.bind()
            p.enableAttributeArray("a_mn")
            p.setAttributeBuffer("a_mn", GL_FLOAT, 0, 3, 12)
            nbuf.release()
            n = g.count
            # outline hull (one draw), then the shaded mesh per material
            p.setUniformValue1f("u_hull", 1.0)
            f.glDrawArrays(GL_TRIANGLES, 0, n)
            p.setUniformValue1f("u_hull", 0.0)
            for unit, (_slot, sampler) in enumerate(GL_TEXTURE_SLOTS):
                p.setUniformValue1i(sampler, unit)
            for mi, first, count in g.ranges:
                md = mats[mi] if 0 <= mi < len(mats) else None
                row = [float(v) for v in table[min(mi, len(table) - 1)]]
                p.setUniformValue("u_m0", QVector4D(*row[0:4]))
                p.setUniformValue("u_m1", QVector4D(*row[4:8]))
                p.setUniformValue1f("u_m2", row[8])
                pbr = float(inputs["pbr"][mi]) if mi < len(inputs["pbr"]) else 0.0
                p.setUniformValue("u_pbr", QVector4D(pbr, float(inputs["metal"][mi]), float(inputs["rough"][mi]),
                                                     float(inputs["unlit"][mi])))
                p.setUniformValue("u_emis", QVector3D(*map(float, inputs["emis"][mi])))
                tx = getattr(md, "textures", {}) if md is not None else {}
                has = []
                for unit, (slot, _sampler) in enumerate(GL_TEXTURE_SLOTS):
                    if slot in tx:
                        self._texture((id(tx[slot]), slot), tx[slot],
                                      version=getattr(md, "tex_version", 0) if slot == "baseColor" else 0).bind(unit)
                        has.append(1.0)
                    else:
                        has.append(0.0)
                p.setUniformValue("u_has", QVector4D(has[0], has[1], has[2], has[3]))
                cut = float(md.alpha_cutoff) if md is not None and md.alpha_mode == "MASK" else 0.0
                p.setUniformValue("u_has2", QVector4D(has[4], float(getattr(md, "normal_scale", 1.0)),
                                                      float(getattr(md, "occlusion_strength", 1.0)), cut))
                p.setUniformValue("u_basef", QVector4D(float(md.base_color[3]) if md is not None and
                                                       len(md.base_color) > 3 else 1.0, 0.0, 0.0, 0.0))
                f.glDrawArrays(GL_TRIANGLES, first, count)
            for name, _, _ in STATIC_LAYOUT:
                p.disableAttributeArray(name)
            for slot in range(4):
                p.disableAttributeArray(f"a_md{slot}")
            p.disableAttributeArray("a_mn")
            p.release()
            fbo.release()
            img = fbo.toImage(True)
        finally:
            self.ctx.doneCurrent()
        img = img.copy(0, 0, Ws, Hs)
        if ss > 1:
            img = img.scaled(W, H, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        if img.format() != QImage.Format_ARGB32_Premultiplied:
            img = img.convertToFormat(QImage.Format_ARGB32_Premultiplied)
        return img, ox, oy
