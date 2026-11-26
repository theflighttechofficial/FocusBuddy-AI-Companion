"""Procedural vector renderers for the built-in characters.

Characters are drawn with QPainter from the solved skeleton in ``Pose.rig``
(rig.py): torso and head are placed by the rig layout, every leg / arm is a
two-segment limb whose middle joint comes from two-bone IK (so planted feet
stay planted while the body moves over them), and tails are drawn along a
spring-driven bone chain.  Everything is vector based - crisp at any size /
DPI.  Sprite packs (sprites.py) can still replace any character.

Coordinate system: a 140x140 unit canvas, feet at (0, 0), +x is the
direction the character faces, -y is up.
"""

from __future__ import annotations

import math
import threading

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen

from .animation import Pose
from .characters import CharacterSpec
from .rig import Rig, RigPose, load_skeleton, rotate

UNITS = 140.0
GROUND_MARGIN = 6.0


def _c(hex_or_color, alpha: int | None = None) -> QColor:
    c = QColor(hex_or_color)
    if alpha is not None:
        c.setAlpha(alpha)
    return c


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


class BaseRenderer:
    OUTLINE_W = 2.2

    def __init__(self, spec: CharacterSpec):
        self.spec = spec
        self.pal = spec.palette
        self.head_pos = (0.0, -60.0)
        self.head_r = 21.0
        self.shadows = True
        self.effects = True          # emotion symbols (!, ?, hearts, steam ...) - optional decoration
        kind = "biped" if spec.skeleton == "biped" else "quadruped"
        extra = spec.extra.get("skeleton") if isinstance(spec.extra, dict) else None
        self.rig = Rig(load_skeleton(spec.id, spec.species, kind, extra=extra))

    def rig_pose(self, pose: Pose) -> RigPose:
        """The solved skeleton for this pose (re-solved locally if the pose carries none)."""
        r = getattr(pose, "rig", None)
        if r is None or getattr(r, "skeleton", None) != self.rig.sk.id:
            r = self.rig.solve(pose)
        return r

    def limb2(self, g: QPainter, a, b, c, width: float, fill, foot_fill=None, foot: str = "paw"):
        """Two-segment limb (root a -> joint b -> end c) drawn as one bent capsule."""
        path = QPainterPath(QPointF(*a))
        path.lineTo(QPointF(*b))
        path.lineTo(QPointF(*c))
        ow = width + 2 * self.OUTLINE_W
        g.setBrush(Qt.NoBrush)
        g.setPen(self.pen(self.pal.outline, ow))
        g.drawPath(path)
        fx, fy = c
        dx = b[0] - c[0]
        if foot == "paw":
            g.save()
            g.setPen(self.pen())
            g.setBrush(_c(foot_fill or fill))
            g.drawEllipse(QPointF(fx + 1.0 - 0.0 * dx, fy + 0.6), width * 0.62, width * 0.48)
            g.restore()
        pen = QPen(_c(fill), width)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        g.setPen(pen)
        g.drawPath(path)
        if foot == "paw":
            g.setPen(Qt.NoPen)
            g.setBrush(_c(foot_fill or fill))
            g.drawEllipse(QPointF(fx + 1.0, fy + 0.6), width * 0.62 - 1.1, width * 0.48 - 1.1)

    def draw_debug(self, g: QPainter, pose: Pose, opts: dict) -> None:
        """Skeleton / IK targets / foot contacts overlay (same transform as ``draw``)."""
        r = self.rig_pose(pose)
        g.save()
        if opts.get("skeleton", True):
            g.setPen(QPen(QColor(91, 108, 255, 230), 1.4))
            for _, a, b in r.bones():
                g.drawLine(QPointF(*a), QPointF(*b))
            g.setPen(Qt.NoPen)
            g.setBrush(QColor(255, 255, 255, 240))
            for (x, y) in r.joints.values():
                g.drawEllipse(QPointF(x, y), 1.6, 1.6)
            for d in (r.legs, r.arms):
                for ls in d.values():
                    g.setBrush(QColor(91, 108, 255))
                    g.drawEllipse(QPointF(*ls.joint), 1.5, 1.5)
        if opts.get("ik", True):
            for d in (r.legs, r.arms):
                for ls in d.values():
                    g.setPen(QPen(QColor(229, 72, 77, 220), 1.0))
                    g.setBrush(Qt.NoBrush)
                    x, y = ls.target
                    g.drawLine(QPointF(x - 2.5, y), QPointF(x + 2.5, y))
                    g.drawLine(QPointF(x, y - 2.5), QPointF(x, y + 2.5))
        if opts.get("contacts", True):
            for ls in r.legs.values():
                g.setPen(Qt.NoPen)
                g.setBrush(QColor(34, 176, 125, 230) if ls.contact else QColor(242, 153, 74, 200))
                g.drawEllipse(QPointF(ls.end[0], ls.end[1] + 3.5), 2.2, 1.2)
        if opts.get("names", False):
            for name, a, b in r.bones():
                self.bone_label(g, name, ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2))
        g.restore()

    @staticmethod
    def bone_label(g: QPainter, text: str, pt: tuple[float, float]) -> None:
        """Small readable label in model units (un-mirrors itself when the character faces left)."""
        g.save()
        g.translate(QPointF(pt[0], pt[1]))
        if g.transform().m11() < 0:
            g.scale(-1, 1)
        f = QFont()
        f.setPixelSize(4)
        g.setFont(f)
        g.setPen(QColor(40, 48, 120, 230))
        g.drawText(QPointF(1.5, -1.0), text)
        g.restore()

    # --------------------------------------------------------------- helpers
    def pen(self, color=None, w: float | None = None) -> QPen:
        p = QPen(_c(color or self.pal.outline), w or self.OUTLINE_W)
        p.setCapStyle(Qt.RoundCap)
        p.setJoinStyle(Qt.RoundJoin)
        return p

    def ellipse(self, g: QPainter, cx, cy, rx, ry, fill, outline=True, rot: float = 0.0):
        g.save()
        g.translate(cx, cy)
        if rot:
            g.rotate(rot)
        g.setPen(self.pen() if outline else Qt.NoPen)
        g.setBrush(_c(fill))
        g.drawEllipse(QPointF(0, 0), rx, ry)
        g.restore()

    def limb(self, g: QPainter, x, y, length, width, angle, fill, outline=True, foot: str | None = None):
        """Rounded limb hanging down from (x, y), rotated by angle (deg)."""
        if length <= 0.5:
            return
        g.save()
        g.translate(x, y)
        g.rotate(angle)
        g.setPen(self.pen() if outline else Qt.NoPen)
        g.setBrush(_c(fill))
        g.drawRoundedRect(QRectF(-width / 2, -width / 2, width, length + width / 2), width / 2, width / 2)
        if foot:
            g.setBrush(_c(foot))
            g.drawEllipse(QPointF(width * 0.15, length), width * 0.62, width * 0.45)
        g.restore()

    # ---------------------------------------------------------------- public
    @staticmethod
    def x_scale(pose: Pose, facing: float = 1) -> float:
        """Horizontal scale for facing/turning: shrinks through ~0.25 mid-turn, then flips."""
        t = pose.turn * (1 if facing >= 0 else -1) * (1.0 - 2.0 * _clamp(getattr(pose, "back", 0.0)))
        sign = 1.0 if t >= 0 else -1.0
        return sign * (0.25 + 0.75 * min(1.0, abs(t)))

    def paint(self, g: QPainter, rect: QRectF, pose: Pose, facing: float = 1, state: str | None = None,
              debug: dict | None = None) -> None:
        g.save()
        g.setRenderHint(QPainter.Antialiasing, True)
        s = rect.width() / UNITS
        g.translate(rect.center().x(), rect.bottom() - GROUND_MARGIN * s)
        g.scale(s, s)
        self._shadow(g, pose)
        g.save()
        xs = self.x_scale(pose, facing)
        g.scale(xs, 1)
        sq = pose.squash
        g.scale(1 + 0.08 * sq, 1 - 0.12 * sq)
        if pose.roll:
            g.translate(0, -30 - pose.lift)
            g.rotate(pose.roll)
            g.translate(0, 30 + pose.lift)
        self.draw(g, pose)
        if pose.ripple > 0:
            self._ripple(g, pose)
        if debug:
            self.draw_debug(g, pose, debug)
        g.restore()
        self._marks(g, pose, 1 if xs >= 0 else -1)
        g.restore()

    # paw/hand tip while pressing (units, facing-right frame) - used to aim at buttons
    def press_point(self) -> tuple[float, float]:
        return (40.0, -24.0)

    def _ripple(self, g: QPainter, pose: Pose) -> None:
        x, y = self._tip if getattr(self, "_tip", None) else self.press_point()
        r = pose.ripple
        g.save()
        g.setBrush(Qt.NoBrush)
        for k, (base, w) in enumerate(((4, 2.4), (9, 1.6))):
            a = max(0.0, 1 - r * (1.1 + 0.3 * k))
            if a <= 0:
                continue
            g.setPen(QPen(_c("#5B6CFF", int(220 * a)), w))
            g.drawEllipse(QPointF(x, y), base + 16 * r, base + 16 * r)
        g.restore()

    def draw(self, g: QPainter, pose: Pose) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def _shadow(self, g: QPainter, pose: Pose):
        """Contact shadow on the ground (stays on the surface while airborne, shrinks & fades with height)."""
        if not getattr(self, "shadows", True):
            return
        ground = max(0.0, getattr(pose, "ground", 0.0))
        h = max(0.0, pose.lift) + ground
        k = _clamp(1 - h / 70.0) * (0.55 + 0.45 * min(1.0, abs(pose.turn)))
        if k <= 0.02:
            return
        y = min(ground, GROUND_MARGIN + 4.0)       # the window only extends a little below the feet
        g.setPen(Qt.NoPen)
        g.setBrush(QColor(0, 0, 0, int(48 * k)))
        rx = (24 + 6 * k) * (1 + 0.15 * max(0.0, pose.squash))
        g.drawEllipse(QPointF(0, y), rx * k, 4.5 * k)

    # ------------------------------------------------------------ face parts
    def eyes(self, g: QPainter, cx, cy, spacing, pose: Pose, size=1.0, color="#2A1E2E", glow=None):
        lx = pose.look_x * 1.8 * (1 - 0.6 * pose.front)
        ly = pose.look_y * 1.5
        spacing = spacing + 2.5 * pose.front
        conv = _clamp(getattr(pose, "eye_conv", 0.0), 0.0, 0.8) * 1.3
        for i, ex in enumerate((cx - spacing / 2, cx + spacing / 2)):
            g.save()
            g.translate(ex + lx + (conv if i == 0 else -conv), cy + ly)
            self.eye(g, pose, i, size, color, glow)
            g.restore()

    def eye(self, g: QPainter, pose: Pose, i: int, size: float = 1.0, color="#2A1E2E", glow=None) -> None:
        """One eye at the painter origin. ``i`` 0 = left (inner corner on the right), 1 = right."""
        style = pose.eye_style
        if pose.eye_open < 0.2 and style in ("normal", "wide", "focused", "angry", "shock"):
            style = "blink"
        size = size * pose.eye_scale
        inner = 1 if i == 0 else -1
        ink = glow or color
        g.save()
        g.scale(size, size)
        if style in ("normal", "wide", "focused", "angry", "half"):
            pup = _clamp(getattr(pose, "pupil", 1.0), 0.4, 1.6)
            psz = 1.0 if pup >= 0.8 else 0.75 + 0.25 * pup / 0.8      # tiny pupils shrink the eye (fear)
            w = (5.6 if style != "wide" else 6.4) * psz
            h = (7.2 if style != "wide" else 8.6) * _clamp(pose.eye_open, 0.15, 1.2) * psz
            if glow:
                g.setPen(Qt.NoPen)
                g.setBrush(_c(glow, 70))
                g.drawRoundedRect(QRectF(-w / 2 - 2, -h / 2 - 2, w + 4, h + 4), 3, 3)
                g.setBrush(_c(glow))
                g.drawRoundedRect(QRectF(-w / 2, -h / 2, w, h), 2, 2)
            else:
                g.setPen(Qt.NoPen)
                g.setBrush(_c(color))
                g.drawEllipse(QPointF(0, 0), w / 2, h / 2)
                if h > 3:
                    hl = 1.6 * (pup if pup > 1.0 else 1.0)              # big sparkly pupils (love / cute)
                    g.setBrush(QColor(255, 255, 255, 235))
                    g.drawEllipse(QPointF(-1.1, -h * 0.22), hl, hl)
                    g.drawEllipse(QPointF(1.3, h * 0.2), 0.7 * hl / 1.6, 0.7 * hl / 1.6)
            if style in ("angry", "half"):
                g.save()
                self._eye_clip(g, w, h)
            if style == "angry":           # lid slants down toward the nose
                g.setPen(Qt.NoPen)
                g.setBrush(_c(self._lid_color()))
                lid = QPainterPath(QPointF(-w, -h / 2 - 3))
                lid.lineTo(w, -h / 2 - 3)
                lid.lineTo(w * inner, -h * 0.05)
                lid.lineTo(-w * inner, -h / 2 + 0.2)
                lid.closeSubpath()
                g.drawPath(lid)
            elif style == "half":          # heavy upper lid (bored / smug / grumpy)
                g.setPen(Qt.NoPen)
                g.setBrush(_c(self._lid_color()))
                g.drawRect(QRectF(-w, -h / 2 - 3, 2 * w, h * 0.55 + 3))
                g.setPen(self.pen(ink, 1.4))
                g.drawLine(QPointF(-w / 2 - 0.5, -h * 0.05), QPointF(w / 2 + 0.5, -h * 0.05))
            if style in ("angry", "half"):
                g.restore()
            if style in ("normal", "wide", "focused") and not glow:
                self._param_lids(g, pose, w, h, inner)
            brow = pose.brow if abs(pose.brow) > 0.12 else (-0.6 if style in ("focused", "angry") else 0.0)
            if brow:
                g.setPen(self.pen(ink, 1.6))
                yb = -max(h, 5) / 2 - 3.0 - 1.4 * _clamp(getattr(pose, "brow_raise", 0.0), -1.0, 1.0)
                g.drawLine(QPointF(-3.4 * inner, yb + brow * 0.4), QPointF(3.4 * inner, yb - brow * 1.8))
        elif style == "shock":                 # big white eyes, tiny pupils
            g.setPen(self.pen(ink, 1.3))
            g.setBrush(_c("#FFFFFF"))
            g.drawEllipse(QPointF(0, 0), 3.7, 4.6)
            g.setPen(Qt.NoPen)
            g.setBrush(_c(ink))
            g.drawEllipse(QPointF(0.4, 0.3), 1.2, 1.3)
        elif style == "heart":
            self._heart_shape(g, 0, -0.4, 3.4, "#FF5C7A", ink)
        elif style == "spiral":
            g.setPen(self.pen(ink, 1.1))
            g.setBrush(Qt.NoBrush)
            path = QPainterPath()
            spin = pose.mark_phase * 6 * (1 if i else -1)
            for k in range(28):
                a_ = spin + k * 0.45
                r_ = 0.25 + k * 0.14
                pt = QPointF(math.cos(a_) * r_, math.sin(a_) * r_)
                if k == 0:
                    path.moveTo(pt)
                else:
                    path.lineTo(pt)
            g.drawPath(path)
        elif style in ("cry", "teary"):
            if style == "cry":
                g.setPen(self.pen(ink, 1.9))
                g.setBrush(Qt.NoBrush)
                path = QPainterPath(QPointF(-3.3, -0.2))
                path.quadTo(0, -2.6, 3.3, -0.2)
                g.drawPath(path)
            else:
                g.setPen(Qt.NoPen)
                g.setBrush(_c(ink))
                g.drawEllipse(QPointF(0, 0), 3.3, 4.2)
                g.setBrush(QColor(255, 255, 255, 240))
                g.drawEllipse(QPointF(-1.0, -1.5), 1.6, 1.6)
                g.drawEllipse(QPointF(1.3, 1.2), 0.9, 0.9)
            # tear stream
            k = (pose.mark_phase * 1.6 + i * 0.5) % 1.0
            g.setPen(Qt.NoPen)
            g.setBrush(_c("#7CC4FF", 220))
            g.drawRoundedRect(QRectF(-1.0 + 1.2 * inner, 1.5, 2.0, 3.5 + 5 * k), 1, 1)
            g.drawEllipse(QPointF(1.2 * inner, 6 + 8 * k), 1.4, 1.8)
        else:
            g.setPen(self.pen(ink, 1.9))
            g.setBrush(Qt.NoBrush)
            path = QPainterPath()
            if abs(pose.brow) > 0.3 and style == "closed":
                g.drawLine(QPointF(-3.2 * inner, -4.2 + pose.brow * 0.4), QPointF(3.2 * inner, -4.2 - pose.brow * 1.6))
            if style == "happy":
                path.moveTo(-3.2, 1.4)
                path.quadTo(0, -3.6, 3.2, 1.4)
            elif style == "closed":
                path.moveTo(-3.3, -0.6)
                path.quadTo(0, 2.6, 3.3, -0.6)
            elif style == "squeeze":         # >< laughing hard / tantrum
                path.moveTo(-3.0 * inner, -2.4)
                path.lineTo(2.6 * inner, 0)
                path.lineTo(-3.0 * inner, 2.4)
            else:  # blink
                path.moveTo(-3, 0.3)
                path.lineTo(3, 0.3)
            g.drawPath(path)
        g.restore()

    def _lid_color(self) -> str:
        return self.pal.main

    def _param_lids(self, g: QPainter, pose: Pose, w: float, h: float, inner: int) -> None:
        """Parametric eyelids: upper lid lowered (``lid``), slanted (``lid_tilt``: - angry, + sad)
        and a cheek-raised lower lid (``lid_low``). Nothing is drawn for a neutral face."""
        lid = _clamp(getattr(pose, "lid", 0.0), 0.0, 1.0)
        tilt = max(-1.0, min(1.0, getattr(pose, "lid_tilt", 0.0)))
        low = _clamp(getattr(pose, "lid_low", 0.0), 0.0, 1.0)
        col = _c(self._lid_color())
        if lid <= 0.04 and abs(tilt) <= 0.08 and low <= 0.05:
            return
        g.save()
        self._eye_clip(g, w, h)          # lids only cover the eye itself - no patches on the face
        if lid > 0.04 or abs(tilt) > 0.08:
            base = -h / 2 + h * lid * 0.9
            yi = base + h * 0.32 * max(0.0, -tilt)      # angry: inner corner pulled down
            yo = base + h * 0.32 * max(0.0, tilt)       # sad: outer corner droops
            g.setPen(Qt.NoPen)
            g.setBrush(col)
            path = QPainterPath(QPointF(-w, -h / 2 - 3))
            path.lineTo(w, -h / 2 - 3)
            a = QPointF(w * inner, yi if inner > 0 else yo)
            b = QPointF(-w * inner, yo if inner > 0 else yi)
            path.lineTo(a)
            path.lineTo(b)
            path.closeSubpath()
            g.drawPath(path)
            if lid > 0.2:                # the lid's edge line keeps a half-closed eye readable
                g.setPen(self.pen("#2A1E2E", 0.9))
                g.drawLine(a, b)
        if low > 0.05:
            top = h / 2 - h * low * 0.62
            g.setPen(Qt.NoPen)
            g.setBrush(col)
            path = QPainterPath(QPointF(-w, h / 2 + 3))
            path.lineTo(-w, h / 2)
            path.quadTo(0, top - h * 0.12, w, h / 2)
            path.lineTo(w, h / 2 + 3)
            path.closeSubpath()
            g.drawPath(path)
        g.restore()

    @staticmethod
    def _eye_clip(g: QPainter, w: float, h: float) -> None:
        clip = QPainterPath()
        clip.addEllipse(QPointF(0, 0), w / 2 + 0.5, h / 2 + 0.5)
        g.setClipPath(clip, Qt.IntersectClip)

    def _heart_shape(self, g: QPainter, x, y, r, fill, outline):
        p = QPainterPath(QPointF(x, y + r * 0.9))
        p.cubicTo(x - r * 1.6, y - r * 0.2, x - r * 0.7, y - r * 1.4, x, y - r * 0.5)
        p.cubicTo(x + r * 0.7, y - r * 1.4, x + r * 1.6, y - r * 0.2, x, y + r * 0.9)
        g.setPen(self.pen(outline, 0.9))
        g.setBrush(_c(fill))
        g.drawPath(p)

    def cheeks(self, g, cx, cy, spacing, pose: Pose):
        a = int(170 * _clamp(pose.blush))
        fl = _clamp(getattr(pose, "flush", 0.0))
        if fl > 0.05:                      # angry / embarrassed: a red flush across the face
            g.setPen(Qt.NoPen)
            g.setBrush(_c("#FF4D5E", int(70 * fl)))
            g.drawEllipse(QPointF(cx, cy - 3), spacing * 0.75, 5.5)
            a = max(a, int(200 * fl))
        if a <= 5:
            return
        g.setPen(Qt.NoPen)
        g.setBrush(_c(self.pal.accent if fl < 0.5 else "#FF6F7D", a))
        for x in (cx - spacing / 2, cx + spacing / 2):
            g.drawEllipse(QPointF(x, cy), 3.6, 2.3)

    def mouth(self, g, mx, my, pose: Pose, color=None, style_override=None):
        """Mouth = a base shape (string) shaped by the continuous face parameters
        (``smile`` curve, ``mouth_open``, ``mouth_width``). Neutral parameters draw the classic shapes."""
        style = style_override or pose.mouth
        sm = max(-1.0, min(1.0, getattr(pose, "smile", 0.0)))
        mo = _clamp(getattr(pose, "mouth_open", 0.0), 0.0, 1.0)
        mw = max(0.6, min(1.5, getattr(pose, "mouth_width", 1.0)))
        if abs(sm) < 0.05 and mo < 0.05 and abs(mw - 1.0) < 0.03:
            return self._mouth_shape(g, mx, my, pose, color, style_override)
        g.save()
        g.translate(mx, my)
        if style in ("smile", "flat", "frown"):
            g.scale(mw, 1.0)
            base = {"smile": 1.0, "flat": 0.0, "frown": -1.0}[style]
            c = 2.6 * max(-1.3, min(1.8, base + sm * 1.2))
            col = color or "#5B2A2A"
            g.setPen(self.pen(col, 1.5 / max(0.8, mw)))
            if mo > 0.12:                       # open mouth that keeps the smile / frown curve
                path = QPainterPath(QPointF(-3.6, -0.5))
                path.quadTo(0, -0.5 - c * 0.25, 3.6, -0.5)
                path.quadTo(0, c * 0.8 + 7.5 * mo, -3.6, -0.5)
                g.setBrush(_c("#7A2E35"))
                g.drawPath(path)
                if mo > 0.35:
                    g.setPen(Qt.NoPen)
                    g.setBrush(_c("#F58C9A"))
                    g.drawEllipse(QPointF(0, max(1.0, c * 0.45 + 4.5 * mo)), 2.0, 1.2)
            else:
                path = QPainterPath(QPointF(-3, -0.5 + max(0.0, -c) * 0.3))
                path.quadTo(0, -0.5 + c, 3, -0.5 + max(0.0, -c) * 0.3)
                g.setBrush(Qt.NoBrush)
                g.drawPath(path)
        elif style == "w":                  # cat mouth: the smile parameter deepens / flattens the curves
            g.scale(mw, 1.0)
            d = 2.4 * max(0.2, 1.0 + sm * 0.9)
            col = color or "#5B2A2A"
            g.setPen(self.pen(col, 1.5))
            g.setBrush(Qt.NoBrush)
            path = QPainterPath(QPointF(-4, 0))
            path.quadTo(-2, d, 0, 0)
            path.quadTo(2, d, 4, 0)
            g.drawPath(path)
            if mo > 0.12:
                g.setBrush(_c("#7A2E35"))
                g.drawEllipse(QPointF(0, 1.2 + 2.0 * mo), 1.6 + 1.2 * mo, 1.2 + 2.6 * mo)
        else:
            open_k = 1.0 + (0.45 * mo if style in ("open", "o", "shout", "laugh", "grin", "yawn") else 0.0)
            g.scale(mw, open_k)
            self._mouth_shape(g, 0, 0, pose, color, style_override)
        g.restore()

    def _mouth_shape(self, g, mx, my, pose: Pose, color=None, style_override=None):
        style = style_override or pose.mouth
        col = color or "#5B2A2A"
        g.setPen(self.pen(col, 1.5))
        g.setBrush(Qt.NoBrush)
        path = QPainterPath()
        if style == "smile":
            path.moveTo(-3, -0.5)
            path.quadTo(0, 2.6, 3, -0.5)
            g.drawPath(path)
        elif style == "w":
            path.moveTo(-4, 0)
            path.quadTo(-2, 2.4, 0, 0)
            path.quadTo(2, 2.4, 4, 0)
            g.save()
            g.translate(mx, my)
            g.drawPath(path)
            g.restore()
            return
        elif style == "grin":
            g.save()
            g.translate(mx, my)
            path.moveTo(-4.2, -0.8)
            path.quadTo(0, 6.5, 4.2, -0.8)
            path.closeSubpath()
            g.setBrush(_c("#7A2E35"))
            g.drawPath(path)
            g.setPen(Qt.NoPen)
            g.setBrush(_c("#F58C9A"))
            g.drawEllipse(QPointF(0, 2.6), 2.3, 1.4)
            g.restore()
            return
        elif style in ("open", "o"):
            g.save()
            g.translate(mx, my)
            g.setBrush(_c("#7A2E35"))
            r = (2.2, 2.8) if style == "open" else (1.7, 1.9)
            g.drawEllipse(QPointF(0, 1.2), r[0], r[1])
            g.restore()
            return
        elif style == "flat":
            path.moveTo(-2.8, 0.6)
            path.lineTo(2.8, 0.6)
            g.save()
            g.translate(mx, my)
            g.drawPath(path)
            g.restore()
            return
        elif style == "frown":
            path.moveTo(-3, 1.6)
            path.quadTo(0, -1.2, 3, 1.6)
        elif style == "yawn":
            g.save()
            g.translate(mx, my)
            k = max(0.35, pose.yawn)
            g.setBrush(_c("#7A2E35"))
            g.drawEllipse(QPointF(0, 2.2 * k), 3.0 * k + 0.6, 4.6 * k + 0.6)
            g.setPen(Qt.NoPen)
            g.setBrush(_c("#F58C9A"))
            g.drawEllipse(QPointF(0, 4.6 * k), 2.0 * k, 1.4 * k)
            g.restore()
            return
        elif style == "sleep":
            path.moveTo(-1.6, 0.8)
            path.quadTo(0, 1.8, 1.6, 0.8)
        elif style in ("shout", "laugh"):
            g.save()
            g.translate(mx, my)
            if style == "laugh":
                g.rotate(-6)
            k = 1.0 if style == "shout" else 0.8 + 0.2 * abs(math.sin(pose.mark_phase * 18))
            m = QPainterPath(QPointF(-5.2 * k, -0.6))
            m.quadTo(0, -2.2 if style == "shout" else -1.0, 5.2 * k, -0.6)
            m.quadTo(0, 10.5 * k, -5.2 * k, -0.6)
            g.setBrush(_c("#7A2E35"))
            g.drawPath(m)
            g.setPen(Qt.NoPen)
            g.setBrush(_c("#FFFFFF"))
            g.drawRect(QRectF(-3.6 * k, -0.6, 7.2 * k, 1.6))       # teeth
            g.setBrush(_c("#F58C9A"))
            g.drawEllipse(QPointF(0, 6.2 * k), 3.0 * k, 1.8 * k)    # tongue
            g.restore()
            return
        elif style == "teeth":           # gritted teeth (angry)
            g.save()
            g.translate(mx, my)
            g.setBrush(_c("#FFFFFF"))
            g.drawRoundedRect(QRectF(-4.6, -0.6, 9.2, 4.2), 1.6, 1.6)
            g.setPen(self.pen("#7A2E35", 0.8))
            g.drawLine(QPointF(-4.4, 1.5), QPointF(4.4, 1.5))
            for x in (-2.2, 0.0, 2.2):
                g.drawLine(QPointF(x, -0.4), QPointF(x, 3.4))
            g.restore()
            return
        elif style == "wobble":          # trembling worried / sad
            path.moveTo(-4.0, 1.0)
            for k in range(1, 9):
                path.lineTo(-4.0 + k, 1.0 + (0.9 if k % 2 else -0.3))
        elif style == "smirk":
            path.moveTo(-3.2, 0.6)
            path.quadTo(1.0, 1.4, 3.8, -1.4)
        elif style == "pout":
            path.moveTo(-2.4, 1.6)
            path.quadTo(0, -0.6, 2.4, 1.6)
            g.save()
            g.translate(mx, my)
            g.drawPath(path)
            g.drawLine(QPointF(-1.4, 2.6), QPointF(1.4, 2.6))
            g.restore()
            return
        elif style == "tongue":          # blep / yuck
            g.save()
            g.translate(mx, my)
            g.drawLine(QPointF(-3.6, 0.0), QPointF(3.6, 0.6))
            g.setBrush(_c("#F58C9A"))
            g.drawRoundedRect(QRectF(-0.4, 0.2, 3.6, 4.6), 1.8, 1.8)
            g.restore()
            return
        g.save()
        g.translate(mx, my)
        g.drawPath(path)
        g.restore()

    def headband(self, g: QPainter, hx, hy, r):
        g.save()
        pen = QPen(_c("#E5484D"), 5.2)
        pen.setCapStyle(Qt.FlatCap)
        g.setPen(pen)
        g.setBrush(Qt.NoBrush)
        rr = r - 2.2
        g.drawArc(QRectF(hx - rr, hy - rr, rr * 2, rr * 2), 25 * 16, 130 * 16)
        # knot + ribbons at the back of the head
        kx, ky = hx - rr * math.cos(math.radians(25)) - 1, hy - rr * math.sin(math.radians(25))
        g.setPen(self.pen("#9E2A2F", 1.4))
        g.setBrush(_c("#E5484D"))
        for ang in (200, 235):
            p = QPainterPath()
            p.moveTo(kx, ky)
            a = math.radians(ang)
            p.lineTo(kx + math.cos(a) * 11 - 3, ky - math.sin(a) * 11)
            p.lineTo(kx + math.cos(a) * 11 + 2, ky - math.sin(a) * 11 + 4)
            p.closeSubpath()
            g.drawPath(p)
        g.restore()

    # ------------------------------------------------------------------ marks
    def _text_path(self, g: QPainter, text: str, x: float, y: float, size: float, fill: str, outline: str):
        f = QFont("Segoe UI", 1)
        f.setPixelSize(int(size))
        f.setBold(True)
        path = QPainterPath()
        path.addText(0, 0, f, text)
        br = path.boundingRect()
        g.save()
        g.translate(x - br.center().x(), y - br.center().y())
        g.setPen(self.pen(outline, 3.2))
        g.setBrush(Qt.NoBrush)
        g.drawPath(path)
        g.setPen(Qt.NoPen)
        g.setBrush(_c(fill))
        g.drawPath(path)
        g.restore()

    def _marks(self, g: QPainter, pose: Pose, facing: int):
        if not pose.mark or not getattr(self, "effects", True):     # effects off: body language only
            return
        hx, hy = self.head_pos
        hx *= facing
        top = hy - self.head_r - 12 - pose.lift
        t = pose.mark_phase
        m = pose.mark
        if m == "!":
            pulse = 1 + 0.08 * math.sin(t * 10)
            g.save()
            g.translate(hx + 14 * facing, top)
            g.scale(pulse, pulse)
            g.setPen(self.pen("#7A5A00", 2))
            g.setBrush(_c("#FFD84D"))
            g.drawEllipse(QPointF(0, 0), 10.5, 10.5)
            g.restore()
            self._text_path(g, "!", hx + 14 * facing, top, 17, "#E5484D", "#FFFFFF")
        elif m == "?":
            g.save()
            g.translate(hx + 14 * facing, top + math.sin(t * 3) * 1.5)
            g.rotate(math.sin(t * 2.5) * 10)
            self._text_path(g, "?", 0, 0, 22, "#5B7CFA", "#FFFFFF")
            g.restore()
        elif m == "zzz":
            for i in range(3):
                k = ((t * 0.5) + i / 3.0) % 1.0
                alpha = int(255 * math.sin(math.pi * k))
                x = hx + facing * (10 + 12 * k + i * 2)
                y = top + 8 - 26 * k
                g.setOpacity(alpha / 255)
                self._text_path(g, "z", x, y, 9 + 6 * k, "#7C8CF8", "#FFFFFF")
            g.setOpacity(1.0)
        elif m == "dots":
            g.save()
            g.translate(hx + 16 * facing, top + 2)
            g.setPen(self.pen("#9AA3B5", 1.4))
            g.setBrush(QColor(255, 255, 255, 235))
            g.drawRoundedRect(QRectF(-13, -8, 26, 15), 7, 7)
            g.setPen(Qt.NoPen)
            for i in range(3):
                on = int(t * 3) % 3 >= i
                g.setBrush(_c("#5B6475" if on else "#C9CED8"))
                g.drawEllipse(QPointF(-7 + i * 7, -0.5), 2.1, 2.1)
            g.restore()
        elif m == "sparkle":
            for i in range(5):
                a = t * 1.5 + i * (2 * math.pi / 5)
                x = math.cos(a) * 46
                y = -48 + math.sin(a) * 34 - pose.lift
                sc = 0.6 + 0.4 * abs(math.sin(t * 4 + i))
                self._star(g, x, y, 6.5 * sc, ["#FFD84D", "#FF8FB1", "#7CE0C3", "#8FB2FF", "#FFB86B"][i])
        elif m == "speed":
            g.save()
            g.setPen(self.pen("#9AA3B5", 2.2))
            for i, (y, ln) in enumerate(((-42, 18), (-30, 26), (-18, 16))):
                off = (t * 120 + i * 9) % 12
                x0 = -facing * (40 + off)
                g.drawLine(QPointF(x0, y - pose.lift), QPointF(x0 - facing * ln, y - pose.lift))
            g.restore()
        elif m == "heart":
            for i in range(2):
                k = ((t * 0.6) + i * 0.5) % 1.0
                g.setOpacity(math.sin(math.pi * k))
                x = hx + facing * (8 + 10 * i + 4 * math.sin(t * 3 + i))
                y = top + 6 - 24 * k
                self._heart(g, x, y, 5.5 + 2 * k)
            g.setOpacity(1.0)
        elif m in ("battery_low", "battery_charge"):
            x, y = hx + 16 * facing, top + 2
            g.save()
            g.translate(x, y)
            g.setPen(self.pen("#3A4255", 1.6))
            g.setBrush(QColor(255, 255, 255, 235))
            g.drawRoundedRect(QRectF(-11, -6, 20, 12), 2.5, 2.5)
            g.setBrush(_c("#3A4255"))
            g.drawRect(QRectF(9, -2.5, 2.5, 5))
            g.setPen(Qt.NoPen)
            if m == "battery_low":
                on = int(t * 3) % 2 == 0
                g.setBrush(_c("#E5484D" if on else "#F5A3A3"))
                g.drawRect(QRectF(-9, -4, 4, 8))
            else:
                n = int(t * 2.5) % 4 + 1
                g.setBrush(_c("#22B07D"))
                for i in range(n):
                    g.drawRect(QRectF(-9 + i * 4.5, -4, 3.5, 8))
            g.restore()
        elif m == "anger":                 # cartoon anger vein
            pulse = 1 + 0.15 * abs(math.sin(t * 9))
            g.save()
            g.translate(hx + 16 * facing, top + 6)
            g.scale(pulse, pulse)
            pen = QPen(_c("#E5484D"), 2.6)
            pen.setCapStyle(Qt.RoundCap)
            g.setPen(pen)
            for a_ in (0, 90, 180, 270):
                g.save()
                g.rotate(a_ + 45)
                path = QPainterPath(QPointF(2.0, -1.5))
                path.quadTo(4.5, 0, 2.0, 1.5)
                g.drawPath(path)
                g.restore()
            g.restore()
        elif m == "steam":                 # puffs rising from both sides of the head
            for side in (-1, 1):
                for i in range(3):
                    k = ((t * 1.2) + i / 3.0) % 1.0
                    g.setOpacity(0.75 * math.sin(math.pi * k))
                    x = hx + side * (self.head_r * 0.8 + 6 * k)
                    y = top + 14 - 20 * k
                    g.setPen(Qt.NoPen)
                    g.setBrush(_c("#FFFFFF"))
                    g.drawEllipse(QPointF(x, y), 3 + 3 * k, 2.5 + 2.5 * k)
            g.setOpacity(1.0)
        elif m == "gloom":                 # little rain cloud + blue lines
            g.save()
            g.translate(hx, top - 2)
            g.setPen(Qt.NoPen)
            g.setBrush(_c("#8C96AD"))
            for dx, r in ((-6, 5), (0, 6.5), (6, 5)):
                g.drawEllipse(QPointF(dx, 0), r, r * 0.75)
            g.setPen(self.pen("#7CC4FF", 1.4))
            for i in range(3):
                k = ((t * 1.5) + i / 3.0) % 1.0
                g.drawLine(QPointF(-5 + i * 5, 5 + 8 * k), QPointF(-5 + i * 5, 8 + 8 * k))
            g.restore()
        elif m == "stars":                 # dizzy stars orbiting the head
            for i in range(3):
                a = t * 4 + i * 2 * math.pi / 3
                x = hx + math.cos(a) * (self.head_r + 4)
                y = top + 10 + math.sin(a) * 4
                self._star(g, x, y, 4.5, ["#FFD84D", "#FFB86B", "#FFF2A8"][i])
        elif m == "sigh":
            k = (t * 0.8) % 1.0
            g.setOpacity(math.sin(math.pi * k))
            g.setPen(self.pen("#C9D0E0", 1.2))
            g.setBrush(_c("#FFFFFF"))
            x = hx + facing * (self.head_r + 6 + 10 * k)
            g.drawEllipse(QPointF(x, top + 24 - 6 * k), 5 + 3 * k, 3.5 + 2 * k)
            g.setOpacity(1.0)
        elif m == "!!":
            for j, dx in enumerate((8, 20)):
                pulse = 1 + 0.1 * math.sin(t * 12 + j)
                g.save()
                g.translate(hx + dx * facing, top - j * 2)
                g.scale(pulse, pulse)
                self._text_path(g, "!", 0, 0, 20, "#E5484D", "#FFFFFF")
                g.restore()
        elif m == "haha":
            for i in range(2):
                k = ((t * 0.9) + i * 0.5) % 1.0
                g.setOpacity(math.sin(math.pi * k))
                self._text_path(g, "ha", hx + facing * (14 + 8 * i), top + 6 - 18 * k, 10 + 3 * k, "#FF9F43", "#FFFFFF")
            g.setOpacity(1.0)
        elif m == "shout":                 # radiating lines
            g.save()
            g.setPen(self.pen("#E5484D", 2.0))
            cx_, cy_ = hx + facing * (self.head_r * 0.9), top + 18
            for i in range(5):
                a = math.radians(-60 + i * 30)
                k = 0.8 + 0.2 * math.sin(t * 20 + i)
                r0, r1 = 6 * k, 13 * k
                g.drawLine(QPointF(cx_ + facing * math.cos(a) * r0, cy_ + math.sin(a) * r0),
                           QPointF(cx_ + facing * math.cos(a) * r1, cy_ + math.sin(a) * r1))
            g.restore()
        elif m == "note":
            for i in range(2):
                k = ((t * 0.7) + i * 0.5) % 1.0
                g.setOpacity(math.sin(math.pi * k))
                x = hx + facing * (12 + 8 * i + 5 * math.sin(t * 3 + i))
                y = top + 4 - 22 * k
                self._text_path(g, "\u266a", x, y, 15, "#A16BFF", "#FFFFFF")
            g.setOpacity(1.0)
        elif m == "sweat":
            x = hx - 18 * facing
            y = hy - 10 - pose.lift
            p = QPainterPath()
            p.moveTo(x, y - 6)
            p.quadTo(x + 5, y + 2, x, y + 3)
            p.quadTo(x - 5, y + 2, x, y - 6)
            g.setPen(self.pen("#3A7BD5", 1.2))
            g.setBrush(_c("#9CD3FF"))
            g.drawPath(p)

    def _heart(self, g: QPainter, x, y, r):
        p = QPainterPath(QPointF(x, y + r * 0.9))
        p.cubicTo(x - r * 1.6, y - r * 0.2, x - r * 0.7, y - r * 1.4, x, y - r * 0.5)
        p.cubicTo(x + r * 0.7, y - r * 1.4, x + r * 1.6, y - r * 0.2, x, y + r * 0.9)
        g.setPen(self.pen("#FFFFFF", 1.2))
        g.setBrush(_c("#FF6B8B"))
        g.drawPath(p)

    def _star(self, g: QPainter, x, y, r, color):
        p = QPainterPath()
        for k in range(8):
            a = k * math.pi / 4 - math.pi / 2
            rr = r if k % 2 == 0 else r * 0.38
            pt = QPointF(x + math.cos(a) * rr, y + math.sin(a) * rr)
            if k == 0:
                p.moveTo(pt)
            else:
                p.lineTo(pt)
        p.closeSubpath()
        g.setPen(self.pen("#FFFFFF", 1.2))
        g.setBrush(_c(color))
        g.drawPath(p)


# =========================================================================
class QuadrupedRenderer(BaseRenderer):
    """Cat, dog, fox and rabbit share one skeleton with species details."""

    # walk = trot (diagonal pairs); run = rotary gallop (pairs nearly together)
    WALK_OFFSETS = {"far_back": 0.0, "far_front": math.pi, "near_back": math.pi, "near_front": 0.0}
    RUN_OFFSETS = {"far_back": 0.0, "far_front": math.pi, "near_back": 0.25 * math.pi, "near_front": 1.25 * math.pi}

    def press_point(self) -> tuple[float, float]:
        return self.rig.press_point()

    def _tilted(self, g: QPainter, rp: RigPose):
        g.save()
        px, py = rp.pivot
        g.translate(px, py)
        g.rotate(rp.tilt)
        g.translate(-px, -py)

    def draw(self, g: QPainter, pose: Pose) -> None:
        pal = self.pal
        sp = self.spec.species
        rp = self.rig_pose(pose)
        lay = rp.layout
        s = lay["s"]
        bx, by, brx, bry = lay["bx"], lay["by"], lay["brx"], lay["bry"]
        hx, hy, hr = lay["hx"], lay["hy"], lay["hr"]
        y0 = lay["y0"]
        self.head_pos = rotate(hx, hy, rp.tilt, rp.pivot)
        self.head_r = hr
        leg_col = pal.extra if sp == "fox" else pal.main
        far_col = pal.extra if sp == "fox" else pal.shade
        legs = rp.legs
        tuck = _clamp(getattr(pose, "tuck", 0.0))

        # far legs (world frame: feet stay planted regardless of the body tilt)
        for k in ("BL", "FL"):
            ls = legs[k]
            if (k == "FL" and tuck > 0.6) or (k == "BL" and s > 0.7):
                continue
            self.limb2(g, ls.root, ls.joint, ls.end, ls.width, far_col)

        # tail (spring chain, behind the body)
        self._tail_chain(g, rp, pose)

        # body
        self._tilted(g, rp)
        body = QPainterPath()
        body.addEllipse(QPointF(bx, by), brx, bry)
        g.setPen(self.pen())
        g.setBrush(_c(pal.main))
        g.drawPath(body)
        g.save()
        g.setClipPath(body)
        g.setPen(Qt.NoPen)
        g.setBrush(_c(pal.light))
        g.drawEllipse(QPointF(bx + brx * 0.38, by + bry * 0.45), brx * 0.55, bry * 0.62)
        if sp == "cat":
            g.setPen(self.pen(pal.extra, 2.4))
            for i in range(3):
                x = bx - brx * 0.45 + i * 9
                g.drawLine(QPointF(x, by - bry - 1), QPointF(x - 2, by - bry * 0.45))
        g.restore()
        if s > 0.05:      # haunch when sitting
            self.ellipse(g, bx - 9, -10 + y0 * 0.3, 13 * s, 10.5 * s, pal.main)
            self.ellipse(g, bx + 1, -3 + y0 * 0.2, 8 * s, 3.8 * s, pal.light if sp != "fox" else pal.extra)
        g.restore()

        # near legs
        ls = legs["BR"]
        if not (s > 0.7 and pose.scratch < 0.2):
            self.limb2(g, ls.root, ls.joint, ls.end, ls.width, leg_col)
        ls = legs["FR"]
        if tuck > 0.6 and pose.paw < 0.2:
            self.ellipse(g, ls.root[0] + 4, -3.2, 5.0, 3.2, leg_col)      # tucked paw (loaf)
        else:
            self.limb2(g, ls.root, ls.joint, ls.end, ls.width, leg_col)
        dx, dy = ls.end[0] - ls.joint[0], ls.end[1] - ls.joint[1]
        d = math.hypot(dx, dy) or 1.0
        self._tip = (ls.end[0] + dx / d * 2.0, ls.end[1] + dy / d * 2.0)

        # collar (dog) + head
        self._tilted(g, rp)
        if sp == "dog":
            g.save()
            g.translate(hx - 6, hy + hr * 0.82)
            g.rotate(-20)
            g.setPen(self.pen("#7A1F23", 1.4))
            g.setBrush(_c("#E5484D"))
            g.drawRoundedRect(QRectF(-12, -3, 24, 6), 3, 3)
            g.translate(4, 2)
            g.rotate(20 + _clamp(pose.tag, -40, 40))      # tag swings like a pendulum
            g.setBrush(_c("#FFD84D"))
            g.drawEllipse(QPointF(0, 3.2), 3.2, 3.2)
            g.restore()
        g.save()
        g.translate(hx, hy)
        g.rotate(pose.head_tilt)
        g.translate(-hx, -hy)
        self._ears(g, hx, hy, hr, pose, before_head=True)
        head = QPainterPath()
        if sp == "fox":
            head.addEllipse(QPointF(hx, hy), hr + 1, hr - 1)
        else:
            head.addEllipse(QPointF(hx, hy), hr, hr)
        g.setPen(self.pen())
        g.setBrush(_c(pal.main))
        g.drawPath(head)
        g.save()
        g.setClipPath(head)
        fx = hx + 4 * (1 - pose.front)
        if sp == "fox":
            g.setPen(Qt.NoPen)
            g.setBrush(_c(pal.light))
            g.drawEllipse(QPointF(fx - 9, hy + 12), 12, 10)
            g.drawEllipse(QPointF(fx + 11, hy + 12), 12, 10)
        if sp == "cat":
            g.setPen(self.pen(pal.extra, 2.2))
            for dx_ in (-5, 0, 5):
                g.drawLine(QPointF(fx + dx_ - 2, hy - hr), QPointF(fx + dx_ - 1.4, hy - hr + 7))
        g.restore()
        self._ears(g, hx, hy, hr, pose, before_head=False)
        if pose.headband:
            self.headband(g, hx, hy, hr)
        ey = hy + 1
        self.cheeks(g, fx, hy + 8, 25, pose)
        self.eyes(g, fx, ey, 16, pose)
        self._muzzle(g, fx, hy, pose)
        g.restore()  # head
        g.restore()  # tilt

    # ------------------------------------------------------------------
    def _chain_path(self, pts) -> QPainterPath:
        path = QPainterPath(QPointF(*pts[0]))
        if len(pts) == 2:
            path.lineTo(QPointF(*pts[1]))
            return path
        for i in range(1, len(pts) - 1):
            mx, my = (pts[i][0] + pts[i + 1][0]) / 2, (pts[i][1] + pts[i + 1][1]) / 2
            if i == len(pts) - 2:
                mx, my = pts[i + 1]
            path.quadTo(QPointF(*pts[i]), QPointF(mx, my))
        return path

    def _tail_chain(self, g: QPainter, rp: RigPose, pose: Pose):
        pal = self.pal
        sp = self.spec.species
        pts = rp.tail
        if not pts:
            return
        g.save()
        if sp in ("cat", "dog"):
            path = self._chain_path(pts)
            g.setBrush(Qt.NoBrush)
            g.setPen(self.pen(pal.outline, 9.5))
            g.drawPath(path)
            g.setPen(self.pen(pal.main, 6.5))
            g.drawPath(path)
            if sp == "cat":
                g.setPen(self.pen(pal.extra, 6.5))
                g.drawPath(self._chain_path([((pts[-2][0] * 2 + pts[-1][0]) / 3, (pts[-2][1] * 2 + pts[-1][1]) / 3),
                                             pts[-1]]))
        elif sp == "fox":
            widths = list(rp.tail_widths) + [6.0] * len(pts)
            left, right = [], []
            for i, (x, y) in enumerate(pts):
                a = pts[max(0, i - 1)]
                b = pts[min(len(pts) - 1, i + 1)]
                tx, ty = b[0] - a[0], b[1] - a[1]
                ln = math.hypot(tx, ty) or 1.0
                nx, ny = -ty / ln, tx / ln
                w = (widths[i - 1] if i > 0 else 3.0) * 0.5 if i < len(pts) - 1 else 1.0
                left.append((x + nx * w, y + ny * w))
                right.append((x - nx * w, y - ny * w))
            ring = left + right[::-1]
            path = QPainterPath(QPointF(*ring[0]))
            for i in range(1, len(ring)):
                p0, p1 = ring[i], ring[(i + 1) % len(ring)]
                path.quadTo(QPointF(*p0), QPointF((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2))
            path.closeSubpath()
            g.setPen(self.pen())
            g.setBrush(_c(pal.main))
            g.drawPath(path)
            g.save()
            g.setClipPath(path)
            g.setPen(Qt.NoPen)
            g.setBrush(_c(pal.light))
            g.drawEllipse(QPointF(*pts[-1]), 10, 9)
            g.restore()
            g.setPen(self.pen())
            g.setBrush(Qt.NoBrush)
            g.drawPath(path)
        else:   # rabbit puff
            x, y = pts[0]
            g.setPen(self.pen())
            g.setBrush(_c(pal.light))
            g.drawEllipse(QPointF(x - 3, y - 3), 7.5, 7)
        g.restore()

    def _ears(self, g: QPainter, hx, hy, hr, pose: Pose, before_head: bool):
        pal = self.pal
        sp = self.spec.species
        perk = pose.ear
        if sp in ("cat", "fox"):
            if not before_head:
                return
            hw, hh = (8.5, 16) if sp == "cat" else (9.5, 21)
            hh *= 1 + 0.08 * perk
            for (ox, oy, rot) in ((-11, -14, -22), (9, -16, 18)):
                rot = rot * (1 - 0.35 * perk) + (-28 if rot < 0 else 28) * max(0.0, -perk) + pose.ear_flop * 0.7
                g.save()
                g.translate(hx + ox, hy + oy)
                g.rotate(rot)
                tri = QPainterPath(QPointF(-hw, 2))
                tri.lineTo(hw, 2)
                tri.lineTo(0, -hh)
                tri.closeSubpath()
                g.setPen(self.pen())
                g.setBrush(_c(pal.main))
                g.drawPath(tri)
                if sp == "fox":
                    g.save()
                    g.setClipPath(tri)
                    g.setPen(Qt.NoPen)
                    g.setBrush(_c(pal.extra))
                    g.drawRect(QRectF(-hw, -hh, hw * 2, hh * 0.45))
                    g.restore()
                inner = QPainterPath(QPointF(-hw * 0.5, 1))
                inner.lineTo(hw * 0.5, 1)
                inner.lineTo(0, -hh * 0.6)
                inner.closeSubpath()
                g.setPen(Qt.NoPen)
                g.setBrush(_c(pal.accent))
                g.drawPath(inner)
                g.restore()
        elif sp == "rabbit":
            if not before_head:
                return
            for (ox, oy, rot) in ((-8, -30, -12), (6, -32, 8)):
                r = rot - 55 * max(0.0, -perk) + 6 * perk + pose.ear_flop * 1.1
                g.save()
                g.translate(hx + ox, hy + oy + 14)
                g.rotate(r)
                g.translate(0, -14)
                self.ellipse(g, 0, 0, 6.8, 18, pal.main)
                self.ellipse(g, 0, 2, 3.2, 12.5, pal.accent, outline=False)
                g.restore()
        elif sp == "dog":
            if before_head:
                return
            for (ox, side) in ((-17, -1), (19, 1)):
                rot = side * (12 - 30 * max(0.0, perk)) + pose.ear_flop * 0.9
                g.save()
                g.translate(hx + ox, hy - 10)
                g.rotate(rot)
                ear = QPainterPath(QPointF(0, -4))
                ear.cubicTo(side * 10, -2, side * 9, 18, side * 3, 20)
                ear.cubicTo(side * -4, 21, side * -5, 6, 0, -4)
                g.setPen(self.pen())
                g.setBrush(_c(pal.extra))
                g.drawPath(ear)
                g.restore()

    def _muzzle(self, g: QPainter, fx, hy, pose: Pose):
        pal = self.pal
        sp = self.spec.species
        my = hy + 8
        if pose.tongue > 0.05:
            tl = 2.5 + 4.5 * _clamp(pose.tongue)
            g.save()
            g.setPen(self.pen("#B0465A", 1.0))
            g.setBrush(_c("#F58C9A"))
            g.drawRoundedRect(QRectF(fx - 2.4, my + 3.0, 4.8, tl), 2.4, 2.4)
            g.restore()
        nz = getattr(pose, "nose", 0.0) * 0.9
        my += nz * 0.35
        if sp == "cat":
            self.ellipse(g, fx, my + 1.5, 7.5, 4.8, pal.light, outline=False)
            nose = QPainterPath(QPointF(fx - 2.4, my - 1.6))
            nose.lineTo(fx + 2.4, my - 1.6)
            nose.lineTo(fx, my + 0.8)
            nose.closeSubpath()
            g.setPen(Qt.NoPen)
            g.setBrush(_c("#E9788A"))
            g.drawPath(nose)
            g.setPen(self.pen("#7A4318", 0.9))
            for side in (-1, 1):
                for dy in (-0.5, 2.2):
                    g.drawLine(QPointF(fx + side * 8, my + dy), QPointF(fx + side * 15, my + dy + (dy - 1) * 0.8))
            self.mouth(g, fx, my + 2.2, pose, style_override="w" if pose.mouth == "smile" else None)
        elif sp == "dog":
            self.ellipse(g, fx, my + 2, 9.5, 6.5, pal.light, outline=False)
            self.ellipse(g, fx, my - 1, 3.6, 2.6, "#2A1E2E", outline=False)
            self.mouth(g, fx, my + 3.4, pose)
            if pose.mouth in ("grin", "open"):
                self.ellipse(g, fx + 1, my + 8, 2.4, 3.2, "#F58C9A", outline=False)
        elif sp == "fox":
            self.ellipse(g, fx, my - 0.5, 2.8, 2.0, "#2A1E2E", outline=False)
            self.mouth(g, fx, my + 2.5, pose)
        else:  # rabbit
            nose = QPainterPath(QPointF(fx - 2.2, my - 1.4))
            nose.lineTo(fx + 2.2, my - 1.4)
            nose.lineTo(fx, my + 0.6)
            nose.closeSubpath()
            g.setPen(Qt.NoPen)
            g.setBrush(_c("#E9788A"))
            g.drawPath(nose)
            if pose.mouth in ("grin", "smile"):
                g.setPen(self.pen(pal.outline, 0.9))
                g.setBrush(_c("#FFFFFF"))
                g.drawRect(QRectF(fx - 2, my + 2.2, 4, 3.2))
                g.drawLine(QPointF(fx, my + 2.2), QPointF(fx, my + 5.4))
            else:
                self.mouth(g, fx, my + 2.6, pose)


# =========================================================================
class BipedRenderer(BaseRenderer):
    """Robot and tiny human 'study buddy'."""

    SHOULDER = (11.0, -45.0)
    ARM = 18.0

    def _lid_color(self) -> str:
        return self.pal.extra if self.spec.species == "robot" else self.pal.light

    def press_point(self) -> tuple[float, float]:
        return self.rig.press_point()

    def _tilted(self, g: QPainter, rp: RigPose):
        g.save()
        px, py = rp.pivot
        g.translate(px, py)
        g.rotate(rp.tilt)
        g.translate(-px, -py)

    def _leg(self, g, ls, pants, shoe):
        path = QPainterPath(QPointF(*ls.root))
        path.lineTo(QPointF(*ls.joint))
        path.lineTo(QPointF(*ls.end))
        g.setBrush(Qt.NoBrush)
        g.setPen(self.pen(self.pal.outline, ls.width + 2 * self.OUTLINE_W))
        g.drawPath(path)
        pen = QPen(_c(pants), ls.width)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        g.setPen(pen)
        g.drawPath(path)
        g.setPen(self.pen())
        g.setBrush(_c(shoe))
        fx, fy = ls.end
        g.drawEllipse(QPointF(fx + ls.width * 0.25, fy + 0.4), ls.width * 0.66, ls.width * 0.46)

    def _arm2(self, g, ls, col, hand, pose: Pose, near: bool):
        path = QPainterPath(QPointF(*ls.root))
        path.lineTo(QPointF(*ls.joint))
        path.lineTo(QPointF(*ls.end))
        g.setBrush(Qt.NoBrush)
        g.setPen(self.pen(self.pal.outline, ls.width + 2 * self.OUTLINE_W))
        g.drawPath(path)
        pen = QPen(_c(col), ls.width)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        g.setPen(pen)
        g.drawPath(path)
        hx, hy = ls.end
        g.setPen(self.pen())
        g.setBrush(_c(hand))
        g.drawEllipse(QPointF(hx, hy), 4.6, 4.6)
        if near and pose.thumb > 0.4:
            g.drawRoundedRect(QRectF(hx - 1.6, hy - 9.5, 3.4, 7.5), 1.7, 1.7)
        if near and pose.point > 0.4:
            dx, dy = ls.end[0] - ls.joint[0], ls.end[1] - ls.joint[1]
            d = math.hypot(dx, dy) or 1.0
            g.save()
            g.translate(hx, hy)
            g.rotate(math.degrees(math.atan2(dy, dx)))
            g.drawRoundedRect(QRectF(1.5, -1.6, 7.5, 3.2), 1.6, 1.6)
            g.restore()

    def draw(self, g: QPainter, pose: Pose) -> None:
        pal = self.pal
        robot = self.spec.species == "robot"
        rp = self.rig_pose(pose)
        lay = rp.layout
        sleeping = lay["sleeping"]
        body_top = lay["body_top"]
        sx = lay["shift"]
        hx, hy, hr = lay["hx"], lay["hy"], lay["hr"]
        self.head_pos = rotate(hx, hy, rp.tilt, rp.pivot)
        self.head_r = hr
        pants = pal.shade if robot else "#3E4A6E"
        shoe = pal.outline if robot else "#FFFFFF"
        arm_col = pal.shade if robot else pal.main
        hand = pal.accent if robot else pal.light

        self._leg(g, rp.legs["L"], pants, shoe)
        self._arm2(g, rp.arms["arm_L"], arm_col, hand, pose, False)

        self._tilted(g, rp)
        breath = 1 + 0.03 * getattr(pose, "breath", 0.0)
        if robot:
            g.setPen(self.pen())
            g.setBrush(_c(pal.main))
            g.drawRoundedRect(QRectF(-16 + sx, body_top, 32, 29 * breath), 7, 7)
            g.setBrush(_c(pal.light))
            g.drawRoundedRect(QRectF(-9 + sx, body_top + 6, 18, 13), 4, 4)
            g.setPen(Qt.NoPen)
            beat = 0.6 + 0.4 * abs(math.sin(pose.mark_phase * 3))
            g.setBrush(_c(pal.accent, int(255 * beat)))
            g.drawEllipse(QPointF(sx, body_top + 12.5), 3.2, 3.2)
        else:
            g.setPen(self.pen())
            g.setBrush(_c(pal.main))
            g.drawRoundedRect(QRectF(-15 + sx, body_top, 30, 30 * breath), 10, 10)
            g.setPen(self.pen(pal.shade, 1.4))
            g.drawLine(QPointF(-7 + sx, body_top + 20), QPointF(7 + sx, body_top + 20))
            g.setPen(self.pen("#FFFFFF", 1.4))
            g.drawLine(QPointF(-3 + sx, body_top + 3), QPointF(-3 + sx, body_top + 11))
            g.drawLine(QPointF(3 + sx, body_top + 3), QPointF(3 + sx, body_top + 11))
        g.restore()

        self._leg(g, rp.legs["R"], pants, shoe)

        self._tilted(g, rp)
        g.save()
        g.translate(hx, hy)
        g.rotate(pose.head_tilt + (10 if sleeping else 0))
        g.translate(-hx, -hy)
        if robot:
            self._robot_head(g, hx, hy, pose)
        else:
            self._human_head(g, hx, hy, hr, pose)
        g.restore()
        prop = max(pose.read, getattr(pose, "write", 0.0))
        if prop > 0.05:
            g.save()
            g.setOpacity(_clamp(prop * 1.4))
            g.translate(18 + sx, body_top + 15)
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
                g.setPen(self.pen("#C9BFAE", 0.8))
                for i in range(3):
                    g.drawLine(QPointF(-9, -3.5 + i * 3.2), QPointF(-2, -3.5 + i * 3.2))
                    g.drawLine(QPointF(2, -3.5 + i * 3.2), QPointF(9, -3.5 + i * 3.2))
            g.restore()
        g.restore()

        ls = rp.arms["arm_R"]
        self._arm2(g, ls, arm_col, hand, pose, True)
        if pose.write > 0.3 and not robot:      # pencil
            g.save()
            g.setPen(self.pen("#5A4318", 1.0))
            g.setBrush(_c("#FFD84D"))
            g.translate(*ls.end)
            g.rotate(-55)
            g.drawRect(QRectF(-1.2, -9, 2.4, 9))
            g.restore()
        self._tip = ls.end

    def _human_head(self, g, hx, hy, hr, pose: Pose):
        pal = self.pal
        # hair back (+ a tuft that follows through with the hair spring)
        g.setPen(self.pen())
        g.setBrush(_c(pal.extra))
        g.save()
        g.translate(hx - hr * 0.55, hy - hr * 0.8)
        g.rotate(-25 + _clamp(pose.hair, -30, 30))
        g.drawEllipse(QPointF(0, -4), 4.5, 7.5)
        g.restore()
        g.drawEllipse(QPointF(hx - 1 - pose.hair * 0.05, hy - 2), hr + 1.5, hr + 0.5)
        # face
        face = QPainterPath()
        face.addEllipse(QPointF(hx + 1, hy + 2), hr - 2, hr - 3)
        g.setBrush(_c(pal.light))
        g.drawPath(face)
        # fringe
        fr = QPainterPath(QPointF(hx - hr + 1, hy - 2))
        fr.cubicTo(hx - hr + 2, hy - hr - 4, hx + hr, hy - hr - 6, hx + hr - 1, hy - 4)
        fr.quadTo(hx + 12, hy - 10, hx + 6, hy - 6)
        fr.quadTo(hx + 1, hy - 11, hx - 4, hy - 6)
        fr.quadTo(hx - 10, hy - 11, hx - 14, hy - 4)
        fr.quadTo(hx - 18, hy - 6, hx - hr + 1, hy - 2)
        g.setBrush(_c(pal.extra))
        g.drawPath(fr)
        if pose.headband:
            self.headband(g, hx, hy - 1, hr)
        fx = hx + 4 * (1 - pose.front)
        self.cheeks(g, fx, hy + 9, 24, pose)
        self.eyes(g, fx, hy + 3, 15, pose)
        self.mouth(g, fx, hy + 11, pose)

    def _robot_head(self, g, hx, hy, pose: Pose):
        pal = self.pal
        # antenna (spring: wobbles with acceleration / bounces)
        g.save()
        g.translate(hx, hy - 19)
        g.rotate(_clamp(pose.antenna, -35, 35))
        g.setPen(self.pen(pal.outline, 2))
        g.drawLine(QPointF(0, 0), QPointF(2, -11))
        led = "#FF4D4D" if (pose.mark == "!" or pose.glitch > 0.2 or getattr(pose, "flush", 0.0) > 0.55) else pal.accent
        on = int(pose.mark_phase * 2) % 2 == 0
        g.setPen(self.pen())
        g.setBrush(_c(led if on else "#FFB27A"))
        g.drawEllipse(QPointF(2, -13), 4, 4)
        g.restore()
        # ear bolts
        g.setBrush(_c(pal.shade))
        g.drawEllipse(QPointF(hx - 25, hy + 1), 4, 6)
        g.drawEllipse(QPointF(hx + 27, hy + 1), 4, 6)
        # head shell
        g.setBrush(_c(pal.main))
        g.drawRoundedRect(QRectF(hx - 24, hy - 19, 50, 40), 13, 13)
        # face screen
        g.setBrush(_c(pal.extra))
        g.setPen(self.pen(pal.outline, 1.6))
        g.drawRoundedRect(QRectF(hx - 16, hy - 11, 36, 25), 8, 8)
        if pose.headband:
            g.setPen(QPen(_c("#E5484D"), 4.5))
            g.drawLine(QPointF(hx - 22, hy - 15), QPointF(hx + 24, hy - 15))
        fx = hx + 3 * (1 - pose.front)
        glow = pal.accent if pose.mark != "!" and getattr(pose, "flush", 0.0) <= 0.55 else "#FF6B6B"
        self.eyes(g, fx, hy, 15, pose, glow=glow)
        # mouth LED
        g.setPen(self.pen(glow, 1.5))
        g.setBrush(Qt.NoBrush)
        if pose.mouth in ("grin", "smile"):
            p = QPainterPath(QPointF(fx - 4, hy + 7))
            p.quadTo(fx, hy + 10.5, fx + 4, hy + 7)
            g.drawPath(p)
        elif pose.mouth in ("o", "open", "yawn"):
            r = 2 + 2.2 * pose.yawn
            g.drawEllipse(QPointF(fx, hy + 8), r, r)
        elif pose.mouth == "frown":
            p = QPainterPath(QPointF(fx - 4, hy + 9.5))
            p.quadTo(fx, hy + 6, fx + 4, hy + 9.5)
            g.drawPath(p)
        elif pose.mouth == "w":
            p = QPainterPath(QPointF(fx - 4, hy + 8))
            p.lineTo(fx - 2, hy + 6.5)
            p.lineTo(fx, hy + 8)
            p.lineTo(fx + 2, hy + 6.5)
            p.lineTo(fx + 4, hy + 8)
            g.drawPath(p)
        else:
            g.drawLine(QPointF(fx - 3, hy + 8), QPointF(fx + 3, hy + 8))
        if pose.blush > 0.7:
            g.setPen(Qt.NoPen)
            g.setBrush(_c("#FF8FB1", 140))
            g.drawEllipse(QPointF(fx - 12, hy + 6), 2.6, 1.6)
            g.drawEllipse(QPointF(fx + 12, hy + 6), 2.6, 1.6)
        if pose.scan > 0.05:          # laser scan beam from the visor
            g.save()
            g.setPen(Qt.NoPen)
            a = math.radians(pose.look_y * 18)
            beam = QPainterPath(QPointF(fx + 8, hy))
            for da in (-0.16, 0.16):
                beam.lineTo(QPointF(fx + 8 + 70 * math.cos(a + da), hy + 70 * math.sin(a + da)))
            beam.closeSubpath()
            g.setBrush(_c("#FF4D4D", int(70 * pose.scan)))
            g.drawPath(beam)
            g.setPen(QPen(_c("#FF4D4D", int(200 * pose.scan)), 1.2))
            g.drawLine(QPointF(fx + 8, hy), QPointF(fx + 8 + 70 * math.cos(a), hy + 70 * math.sin(a)))
            g.restore()
        if pose.glitch > 0.1:         # error glitch: displaced scanlines over the screen
            g.save()
            g.setPen(Qt.NoPen)
            t = pose.mark_phase
            for i in range(4):
                yy = hy - 10 + ((i * 37 + int(t * 30) * 13) % 22)
                off = math.sin(t * 50 + i * 2) * 6 * pose.glitch
                g.setBrush(_c(["#FF4D4D", "#4DF2FF", "#FFD84D", "#A16BFF"][i], int(150 * pose.glitch)))
                g.drawRect(QRectF(hx - 15 + off, yy, 34, 2.2))
            g.restore()


DEFAULT_MODE = {"mode": "2d", "quality": "normal"}     # app-wide choice (previews follow the setting)


def set_default_mode(mode: str, quality: str = "normal") -> None:
    DEFAULT_MODE["mode"] = mode
    DEFAULT_MODE["quality"] = quality


RENDER_MODES = {"illustrated": "Illustrated (character art)", "production": "3D models (GLB)", "3d": "3D Procedural", "2d": "2D Fallback"}
_ASSET_CACHE: dict = {}
_ASSET_LOCK = threading.Lock()
_ASSET_LOADING: dict = {}          # id -> threading.Event while one thread loads it


def load_asset(spec: CharacterSpec, rig=None):
    """Cached CharacterAsset for a character (GLB if installed, else the procedural description).
    Thread-safe: the dashboard previews load assets in the background (ui/preview_loader.py); a
    second caller for the same character waits for the first instead of loading it again."""
    key = spec.id
    while True:
        with _ASSET_LOCK:
            if key in _ASSET_CACHE:
                return _ASSET_CACHE[key]
            ev = _ASSET_LOADING.get(key)
            if ev is None:
                ev = _ASSET_LOADING[key] = threading.Event()
                break
        ev.wait()
    try:
        from .character_asset import CharacterAsset
        height = None
        if rig is not None:
            try:
                from .render_mesh import asset_rig_height
                height = asset_rig_height(rig)
            except Exception:  # noqa: BLE001
                height = None
        asset = CharacterAsset.load(spec.id, rig_height=height)
        with _ASSET_LOCK:
            _ASSET_CACHE[key] = asset
        return asset
    finally:
        with _ASSET_LOCK:
            _ASSET_LOADING.pop(key, None)
        ev.set()


def make_renderer(spec: CharacterSpec, mode: str = "2d", quality: str = "normal"):
    """Renderer for a character, with the fallback chain

        "production"  skinned GLB character (render_mesh.py)  -> procedural 3D -> 2D
        "3d"          procedural 3D models (render3d.py)       -> 2D
        "2d"          flat vector art (this module)

        "illustrated" the character-sheet art (illustrated.py)  -> production -> ...

    Sprite packs always use their frames. GPU -> CPU fallback happens inside the 3D renderers."""
    if mode == "illustrated":
        try:
            from .illustrated import IllustratedRenderer, pack_dir
            folder = pack_dir(spec.id)
            if folder is not None:
                return IllustratedRenderer(spec, folder)
        except Exception:  # noqa: BLE001 - a broken pack falls back to the 3D model
            import logging
            logging.getLogger("focusbuddy.companion").exception("Illustrated pack unavailable, using 3D")
        mode = "production"
    if mode == "production" and spec.skeleton != "sprite":
        try:
            from .render_mesh import MeshRenderer3D
            from .rig import Rig, load_skeleton
            kind = "biped" if spec.skeleton == "biped" else "quadruped"
            asset = load_asset(spec, Rig(load_skeleton(spec.id, spec.species, kind)))
            if asset.is_mesh:
                return MeshRenderer3D(spec, asset, quality)
        except Exception:  # noqa: BLE001
            import logging
            logging.getLogger("focusbuddy.companion").exception("Production renderer unavailable, using procedural 3D")
        mode = "3d"
    if mode == "3d" and spec.skeleton != "sprite":
        try:
            from .render3d import Renderer3D
            return Renderer3D(spec, quality)
        except Exception:  # noqa: BLE001 - missing numpy / unexpected error: keep the 2D look
            import logging
            logging.getLogger("focusbuddy.companion").exception("3D renderer unavailable, using 2D")
    if spec.skeleton == "sprite" and spec.sprite_dir:
        from .sprites import SpriteRenderer
        try:
            return SpriteRenderer(spec)
        except Exception:  # broken pack -> procedural fallback
            import logging
            logging.getLogger("focusbuddy.companion").exception("Sprite pack failed, using fallback")
            from .characters import BUILTIN
            return QuadrupedRenderer(BUILTIN["cat"])
    if spec.skeleton == "biped":
        return BipedRenderer(spec)
    return QuadrupedRenderer(spec)
