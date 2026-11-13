"""Animation debug overlay (developer setting).

* Info panel that follows the companion: FPS, state, emotion, action layers,
  velocity, acceleration, target, current animation + phase, ground contact,
  left / right foot IK, energy, attention target, monitor, plan.
* Transparent click-through layer drawing the planned path, the target point,
  window obstacles and the attention target across the whole virtual desktop.
* The bone skeleton, IK targets and foot contacts are drawn on the companion
  itself (setting "Draw skeleton on companion").
"""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRect, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QGuiApplication, QPainter, QPen
from PySide6.QtWidgets import QWidget

FLAGS = Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint | Qt.WindowDoesNotAcceptFocus \
    | Qt.WindowTransparentForInput


class _Panel(QWidget):
    def __init__(self):
        super().__init__(None, FLAGS)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.lines: list[str] = []
        self.setFixedSize(300, 300)

    def paintEvent(self, _):
        g = QPainter(self)
        g.setRenderHint(QPainter.Antialiasing)
        g.setPen(Qt.NoPen)
        g.setBrush(QColor(18, 22, 34, 215))
        g.drawRoundedRect(QRectF(self.rect()).adjusted(1, 1, -1, -1), 10, 10)
        f = QFont("Consolas")
        f.setStyleHint(QFont.Monospace)
        f.setPixelSize(11)
        g.setFont(f)
        for i, ln in enumerate(self.lines):
            g.setPen(QColor("#9BE7C4") if i == 0 else QColor("#E6E9F2"))
            g.drawText(QPointF(10, 18 + i * 15.5), ln)


class _PathLayer(QWidget):
    def __init__(self):
        super().__init__(None, FLAGS)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.path: list[tuple[float, float]] = []
        self.look = None
        self.feet = (0.0, 0.0)
        self.obstacles: list[tuple[float, float, float, float]] = []
        self.target = None
        self._fit()

    def _fit(self):
        rect = QRect()
        for s in QGuiApplication.screens():
            rect = rect.united(s.geometry())
        self.setGeometry(rect)

    def paintEvent(self, _):
        g = QPainter(self)
        g.setRenderHint(QPainter.Antialiasing)
        o = self.geometry().topLeft()
        g.setBrush(Qt.NoBrush)
        g.setPen(QPen(QColor(229, 72, 77, 140), 1.5, Qt.DashLine))
        for (x, y, w, h) in self.obstacles:
            g.drawRect(QRectF(x - o.x(), y - o.y(), w, h))
        if self.target is not None:
            tx, ty = self.target[0] - o.x(), self.target[1] - o.y()
            g.setPen(QPen(QColor(229, 72, 77, 230), 2))
            g.drawLine(QPointF(tx - 8, ty), QPointF(tx + 8, ty))
            g.drawLine(QPointF(tx, ty - 8), QPointF(tx, ty + 8))
            g.drawEllipse(QPointF(tx, ty), 5, 5)
        if len(self.path) > 1:
            g.setPen(QPen(QColor(91, 108, 255, 200), 2, Qt.DashLine))
            for a, b in zip(self.path, self.path[1:]):
                g.drawLine(QPointF(a[0] - o.x(), a[1] - o.y()), QPointF(b[0] - o.x(), b[1] - o.y()))
            g.setPen(Qt.NoPen)
            for i, p in enumerate(self.path):
                g.setBrush(QColor("#E5484D") if i == len(self.path) - 1 else QColor(91, 108, 255, 220))
                g.drawEllipse(QPointF(p[0] - o.x(), p[1] - o.y()), 4, 4)
        if self.look is not None and self.look.point is not None:
            fx, fy = self.feet
            g.setPen(QPen(QColor(34, 176, 125, 170), 1.5, Qt.DotLine))
            g.drawLine(QPointF(fx - o.x(), fy - 60 - o.y()),
                       QPointF(self.look.point[0] - o.x(), self.look.point[1] - o.y()))
            g.setBrush(QColor(34, 176, 125, 120))
            g.drawEllipse(QPointF(self.look.point[0] - o.x(), self.look.point[1] - o.y()), 7, 7)


class DebugOverlay:
    def __init__(self, companion):
        self.c = companion
        self.panel = _Panel()
        self.layer = _PathLayer()
        self.panel.show()
        self.layer.show()
        self._n = 0

    def info_lines(self) -> list[str]:
        c = self.c
        b = c.behavior
        m = b.mover
        info = c.env.screen_info(b.x, b.y) if hasattr(c.env, "screen_info") else {"index": 1, "dpr": 1.0}
        tgt = "-"
        if b.target_rect is not None:
            tgt = f"window {int(b.target_rect.w)}x{int(b.target_rect.h)}"
        elif m.target is not None:
            tgt = f"({m.target.x:.0f},{m.target.y:.0f})"
        look = b.look.kind.value if b.look is not None else "NONE"
        surf = "floor" if b.surface is None else f"{b.surface.kind}@{b.surface.y:.0f}"
        pose = c._pose
        rig = getattr(pose, "rig", None) if pose is not None else None
        an = c.animator
        layers = ", ".join(f"{a.name}{'' if a.track == 'full' else '/' + a.track[0]}" for a in b.actions.layers()) or "-"

        def foot(names):
            if rig is None:
                return "-"
            out = []
            for n in names:
                ls = rig.legs.get(n)
                if ls is not None:
                    out.append(f"{n}:{'PLANT' if ls.contact else ls.phase or 'SWING'}"
                               f"{'' if ls.reached else '!'} {ls.rel:+.0f}°")
            return "  ".join(out) or "-"
        if rig is not None and rig.kind == "biped":
            left, right = foot(["L"]), foot(["R"])
        else:
            left, right = foot(["FL", "BL"]), foot(["FR", "BR"])
        ground = "AIRBORNE" if m.airborne else ("contact" if rig is None else
                                               f"{sum(1 for l in rig.legs.values() if l.contact)}/{len(rig.legs)} feet")
        return [
            f"FPS: {c.fps:.0f}   size {c.px}px   particles {len(c.particles)}",
            f"State: {b.state.value}   ({b.mode}, prio {b.plan_priority.name})",
            f"Emotion: {b.emotion.current.value}",
            f"Action: {layers}",
            f"Velocity: {m.speed:.0f} px/s  {m.gait_name}  lvl {m.gait_level:.2f}",
            f"Acceleration: {m.accel_along:+.0f} px/s²   {m.mode}",
            f"Target: {tgt}",
            f"Animation: {an.phase_name}",
            f"Phase: {math.fmod(m.phase, 2 * math.pi):.2f} rad  cycle {(m.phase / (2 * math.pi)) % 1:.2f}",
            f"Ground: {ground}   on {surf}",
            f"Left IK: {left}",
            f"Right IK: {right}",
            f"Energy: {b.energy.level:.0f} ({b.energy.mood})   breath {an.breath_rate + 0.9 * an._exertion:.2f} Hz",
            f"Attention: {look}",
            f"Monitor: {info.get('index', 1)}  dpr {info.get('dpr', 1.0):.2f}",
            f"Plan: {', '.join(str(s[0]) for s in b._plan[:4]) or '-'}",
        ]

    def update_info(self) -> None:
        self._n += 1
        if self._n % 4:
            return   # ~15 Hz is plenty for text
        c = self.c
        self.panel.lines = self.info_lines()
        w = c.window
        x = w.x() + w.width() + 6
        y = w.y()
        geo = QGuiApplication.screenAt(w.geometry().center())
        if geo is not None and x + self.panel.width() > geo.geometry().right():
            x = w.x() - self.panel.width() - 6
        self.panel.move(int(x), int(y))
        self.panel.update()
        self.layer.path = list(c.behavior.path_points) if c.behavior.mover.moving else []
        self.layer.look = c.behavior.look
        self.layer.feet = (c.behavior.x, c.behavior.y)
        wins = c.behavior.windows()
        self.layer.obstacles = [(w["x"], w["y"], w["w"], w["h"]) for w in wins if w["w"] * w["h"] > 120_000]
        t = c.behavior.mover.target
        self.layer.target = (t.x, t.y) if t is not None else None
        self.layer.update()

    def close_all(self) -> None:
        self.panel.close()
        self.layer.close()
