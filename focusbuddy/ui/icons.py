"""Icons rendered from the procedural characters (no image files needed)."""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPixmap

from ..companion.animation import Animator
from ..companion.characters import CharacterSpec
from ..companion.renderer import make_renderer
from ..companion.states import CompanionState as S


def character_pixmap(spec: CharacterSpec, size: int, state: S = S.IDLE, t: float = 0.4,
                     background: str | None = None) -> QPixmap:
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    g = QPainter(pm)
    g.setRenderHint(QPainter.Antialiasing)
    if background:
        g.setPen(Qt.NoPen)
        g.setBrush(QColor(background))
        g.drawEllipse(QRectF(0, 0, size, size))
    an = Animator(seed=3)
    an._next_blink = 1e12
    pose = an.pose(state, 10.0, 0.0)
    pose = an.pose(state, 10.0 + t, t)
    make_renderer(spec).paint(g, QRectF(0, 0, size, size), pose, 1, state.value)
    g.end()
    return pm


def head_pixmap(spec: CharacterSpec, size: int, background: str | None = "#ECEEFF", status: str | None = None) -> QPixmap:
    """Character zoomed on its head - used for tray / window icons. Characters with an illustrated
    pack use the neutral face from their character sheet."""
    try:
        from ..companion.illustrated import portrait_path
        pp = portrait_path(spec.id)
    except Exception:  # noqa: BLE001
        pp = None
    if pp is not None:
        face = QPixmap(str(pp))
        if not face.isNull():
            return _round_icon(face, size, status)
    big = character_pixmap(spec, size * 3)
    r = make_renderer(spec)
    # The renderer stores the head position in units after drawing; draw once to compute it.
    tmp = QPixmap(10, 10)
    tmp.fill(Qt.transparent)
    gg = QPainter(tmp)
    an = Animator(seed=3)
    r.paint(gg, QRectF(0, 0, 10, 10), an.pose(S.IDLE, 10.0, 0.0), 1)
    gg.end()
    hx, hy = r.head_pos
    scale = size * 3 / 140.0
    cx = size * 3 / 2 + hx * scale
    cy = size * 3 - 6 * scale + hy * scale
    half = (r.head_r + 15) * scale
    crop = big.copy(int(cx - half), int(cy - half - 4 * scale), int(half * 2), int(half * 2))
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    g = QPainter(pm)
    g.setRenderHint(QPainter.Antialiasing)
    g.setRenderHint(QPainter.SmoothPixmapTransform)
    if background:
        g.setPen(Qt.NoPen)
        g.setBrush(QColor(background))
        g.drawEllipse(QRectF(0, 0, size, size))
        clip = QPainterPath()
        clip.addEllipse(QRectF(0, 0, size, size))
        g.setClipPath(clip)
    g.drawPixmap(QRectF(0, 0, size, size), crop, QRectF(crop.rect()))
    g.setClipping(False)
    if status:
        d = size * 0.34
        g.setPen(Qt.NoPen)
        g.setBrush(QColor("white"))
        g.drawEllipse(QRectF(size - d - 1, size - d - 1, d + 1, d + 1))
        g.setBrush(QColor(status))
        g.drawEllipse(QRectF(size - d + 1, size - d + 1, d - 3, d - 3))
    g.end()
    return pm


def _round_icon(face: QPixmap, size: int, status: str | None) -> QPixmap:
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    g = QPainter(pm)
    g.setRenderHint(QPainter.Antialiasing)
    g.setRenderHint(QPainter.SmoothPixmapTransform)
    clip = QPainterPath()
    clip.addEllipse(QRectF(0, 0, size, size))
    g.setClipPath(clip)
    side = min(face.width(), face.height())
    src = QRectF((face.width() - side) / 2, (face.height() - side) / 2, side, side)
    g.drawPixmap(QRectF(0, 0, size, size), face, src)
    g.setClipping(False)
    if status:
        d = size * 0.34
        g.setPen(Qt.NoPen)
        g.setBrush(QColor("white"))
        g.drawEllipse(QRectF(size - d - 1, size - d - 1, d + 1, d + 1))
        g.setBrush(QColor(status))
        g.drawEllipse(QRectF(size - d + 1, size - d + 1, d - 3, d - 3))
    g.end()
    return pm


def app_icon(spec: CharacterSpec, status: str | None = None) -> QIcon:
    icon = QIcon()
    for s in (16, 24, 32, 48, 64, 128, 256):
        icon.addPixmap(head_pixmap(spec, s, status=status))
    return icon


# ------------------------------------------------------------------ Fluent system icons
# Segoe Fluent Icons (Windows 11) / Segoe MDL2 Assets (Windows 10) code points
FLUENT_GLYPHS = {
    "home": "", "focus": "", "stats": "", "history": "", "rules": "",
    "companion": "", "playground": "", "settings": "", "privacy": "",
    "help": "", "faq": "", "about": "",
    "pause": "", "play": "", "theme": "",
    "tasks": "\uE73A", "care": "\uEB51", "coach": "\uE945", "chat": "\uE8BD", "reports": "\uE8A5",
}
_ICON_FONT: list = []


def fluent_icon_font() -> str | None:
    """The Windows icon font if it is installed (None on other systems)."""
    if not _ICON_FONT:
        from PySide6.QtGui import QFontDatabase
        fams = set(QFontDatabase.families())
        _ICON_FONT.append(next((f for f in ("Segoe Fluent Icons", "Segoe MDL2 Assets") if f in fams), None))
    return _ICON_FONT[0]


def nav_icon(name: str, color: str, on_color: str, px: int = 18) -> QIcon | None:
    """A navigation icon: ``color`` normally, ``on_color`` while its item is selected (checked)."""
    a, b = fluent_icon(name, color, px), fluent_icon(name, on_color, px)
    if a is None or b is None:
        return None
    ic = QIcon()
    for mode in (QIcon.Normal, QIcon.Active):
        ic.addPixmap(a.pixmap(px, px), mode, QIcon.Off)
        ic.addPixmap(b.pixmap(px, px), mode, QIcon.On)
    return ic


def fluent_icon(name: str, color: str, px: int = 16) -> QIcon | None:
    """A crisp monochrome system icon (Windows), or None where the icon font is missing."""
    fam = fluent_icon_font()
    glyph = FLUENT_GLYPHS.get(name)
    if fam is None or glyph is None:
        return None
    from PySide6.QtGui import QFont, QGuiApplication
    dpr = max(1.0, QGuiApplication.primaryScreen().devicePixelRatio() if QGuiApplication.primaryScreen() else 1.0)
    pm = QPixmap(int(px * dpr), int(px * dpr))
    pm.setDevicePixelRatio(dpr)
    pm.fill(Qt.transparent)
    g = QPainter(pm)
    g.setRenderHint(QPainter.TextAntialiasing)
    f = QFont(fam)
    f.setPixelSize(px)
    g.setFont(f)
    g.setPen(QColor(color))
    g.drawText(QRectF(0, 0, px, px), Qt.AlignCenter, glyph)
    g.end()
    return QIcon(pm)


# ------------------------------------------------------------------ profile avatars
def avatar_pixmap(avatar: str, name: str, size: int, ring: bool = True) -> QPixmap:
    """Round avatar: a companion's portrait, or the profile's initials on a pastel gradient."""
    from PySide6.QtGui import QFont, QLinearGradient, QPen
    from .theme import C
    dpr = 2.0
    pm = QPixmap(int(size * dpr), int(size * dpr))
    pm.setDevicePixelRatio(dpr)
    pm.fill(Qt.transparent)
    g = QPainter(pm)
    g.setRenderHint(QPainter.Antialiasing)
    g.setRenderHint(QPainter.SmoothPixmapTransform)
    r = QRectF(1.5, 1.5, size - 3, size - 3)
    clip = QPainterPath()
    clip.addEllipse(r)
    face = None
    if avatar and avatar != "initials":
        try:
            from ..companion.illustrated import portrait_path
            pp = portrait_path(avatar)
            face = QPixmap(str(pp)) if pp is not None else None
        except Exception:  # noqa: BLE001
            face = None
    g.save()
    g.setClipPath(clip)
    if face is not None and not face.isNull():
        side = min(face.width(), face.height())
        g.drawPixmap(r, face, QRectF((face.width() - side) / 2, (face.height() - side) / 2, side, side))
    else:
        grad = QLinearGradient(r.topLeft(), r.bottomRight())
        grad.setColorAt(0, QColor(C.get("primary", "#7C6CF6")))
        grad.setColorAt(1, QColor(C.get("primary_grad", "#F49AC1")))
        g.fillRect(r, grad)
        parts = [p for p in (name or "?").split() if p]
        text = ("".join(p[0] for p in parts[:2]) or "?").upper()
        f = QFont()
        f.setPixelSize(int(size * 0.38))
        f.setBold(True)
        g.setFont(f)
        g.setPen(QColor("#FFFFFF"))
        g.drawText(r, Qt.AlignCenter, text)
    g.restore()
    if ring:
        g.setPen(QPen(QColor(255, 255, 255, 230), 2.4))
        g.setBrush(Qt.NoBrush)
        g.drawEllipse(r)
    g.end()
    return pm
