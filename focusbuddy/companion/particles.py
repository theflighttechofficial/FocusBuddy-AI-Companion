"""Tiny, optional particle effects (sprint dust, landing puffs, celebration confetti,
robot sparks).

Particles live in global logical desktop coordinates so dust stays behind
while the companion runs on.  The system is capped (``MAX``) and updated
inside the companion's single frame loop - no timers, no threads.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

GRAVITY = 900.0


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: float
    max_life: float
    size: float
    color: str
    kind: str          # dust | puff | confetti | star | spark
    rot: float = 0.0
    spin: float = 0.0
    gravity: float = 0.0
    drag: float = 2.0


CONFETTI = ("#FFD84D", "#FF8FB1", "#7CE0C3", "#8FB2FF", "#FFB86B", "#A16BFF")


class ParticleSystem:
    MAX = 90

    def __init__(self, seed: int | None = None):
        self.items: list[Particle] = []
        self.enabled = True
        self.amount = 1.0            # Low 0.5 / Normal 1 / High 1.4
        self.rng = random.Random(seed)

    def _add(self, p: Particle) -> None:
        if len(self.items) >= self.MAX:
            self.items.pop(0)
        self.items.append(p)

    def _n(self, n: int) -> int:
        return max(0, int(round(n * self.amount)))

    # ------------------------------------------------------------- emitters
    def dust(self, x: float, y: float, direction: int, scale: float = 1.0, n: int = 2) -> None:
        if not self.enabled:
            return
        r = self.rng
        for _ in range(self._n(n)):
            self._add(Particle(x + r.uniform(-4, 4) * scale, y - r.uniform(0, 3) * scale,
                               -direction * r.uniform(20, 60) * scale, -r.uniform(10, 35) * scale,
                               0.0, r.uniform(0.35, 0.55), r.uniform(2.5, 4.5) * scale, "#C9BBA7", "dust", drag=3.0))

    def landing(self, x: float, y: float, impact: float, scale: float = 1.0) -> None:
        if not self.enabled:
            return
        r = self.rng
        n = self._n(4 + min(6, int(impact / 250)))
        for i in range(n):
            side = -1 if i % 2 else 1
            self._add(Particle(x + side * r.uniform(6, 14) * scale, y - 1, side * r.uniform(40, 90) * scale,
                               -r.uniform(10, 30) * scale, 0.0, r.uniform(0.3, 0.5), r.uniform(3, 5) * scale,
                               "#D5CBBE", "puff", drag=4.0))

    def confetti(self, x: float, y: float, scale: float = 1.0, n: int = 14) -> None:
        if not self.enabled:
            return
        r = self.rng
        for _ in range(self._n(n)):
            a = r.uniform(-math.pi * 0.9, -math.pi * 0.1)
            sp = r.uniform(120, 260) * scale
            kind = "star" if r.random() < 0.35 else "confetti"
            self._add(Particle(x, y, math.cos(a) * sp, math.sin(a) * sp, 0.0, r.uniform(0.9, 1.4),
                               r.uniform(3, 5) * scale, r.choice(CONFETTI), kind, r.uniform(0, 360),
                               r.uniform(-400, 400), GRAVITY * 0.55, 1.2))

    def sparks(self, x: float, y: float, scale: float = 1.0, n: int = 8) -> None:
        if not self.enabled:
            return
        r = self.rng
        for _ in range(self._n(n)):
            a = r.uniform(0, 2 * math.pi)
            sp = r.uniform(60, 160) * scale
            self._add(Particle(x, y, math.cos(a) * sp, math.sin(a) * sp - 40 * scale, 0.0, r.uniform(0.25, 0.45),
                               r.uniform(1.5, 2.5) * scale, r.choice(("#4DF2FF", "#FFD84D", "#FFFFFF")), "spark",
                               gravity=GRAVITY * 0.4, drag=1.5))

    # -------------------------------------------------------------- update
    def update(self, dt: float) -> None:
        if not self.items:
            return
        keep = []
        for p in self.items:
            p.life += dt
            if p.life >= p.max_life:
                continue
            f = math.exp(-p.drag * dt)
            p.vx *= f
            p.vy = p.vy * f + p.gravity * dt
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.rot += p.spin * dt
            keep.append(p)
        self.items = keep

    def clear(self) -> None:
        self.items.clear()

    def __len__(self) -> int:
        return len(self.items)

    # --------------------------------------------------------------- paint
    def paint(self, g, ox: float, oy: float) -> None:
        """Draw with QPainter ``g``; (ox, oy) = global position of the painter's origin."""
        if not self.items:
            return
        from PySide6.QtCore import QPointF, QRectF, Qt
        from PySide6.QtGui import QColor
        g.save()
        g.setPen(Qt.NoPen)
        for p in self.items:
            u = p.life / p.max_life
            a = max(0.0, 1.0 - u) if p.kind != "confetti" else max(0.0, min(1.0, (1 - u) * 2.5))
            c = QColor(p.color)
            c.setAlphaF(a * (0.55 if p.kind in ("dust", "puff") else 0.95))
            g.setBrush(c)
            x, y = p.x - ox, p.y - oy
            if p.kind in ("dust", "puff"):
                r = p.size * (1 + u * 1.2)
                g.drawEllipse(QPointF(x, y), r, r * 0.8)
            elif p.kind == "confetti":
                g.save()
                g.translate(x, y)
                g.rotate(p.rot)
                g.drawRect(QRectF(-p.size / 2, -p.size / 4, p.size, p.size / 2))
                g.restore()
            elif p.kind == "star":
                g.save()
                g.translate(x, y)
                g.rotate(p.rot)
                s = p.size
                from PySide6.QtGui import QPolygonF
                pts = [QPointF(math.cos(k * math.pi / 4 - math.pi / 2) * (s if k % 2 == 0 else s * 0.4),
                               math.sin(k * math.pi / 4 - math.pi / 2) * (s if k % 2 == 0 else s * 0.4))
                       for k in range(8)]
                g.drawPolygon(QPolygonF(pts))
                g.restore()
            else:   # spark
                g.drawEllipse(QPointF(x, y), p.size, p.size)
        g.restore()
