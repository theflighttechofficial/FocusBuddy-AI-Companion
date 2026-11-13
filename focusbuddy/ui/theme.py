"""Design tokens + Qt stylesheet: FocusBuddy's soft, playful premium look.

* airy pastel canvas (peach -> lavender -> sky in light mode, deep plum -> night blue in dark mode)
  with soft colour glows painted behind the UI (ui/motion.py ``AuroraBackground``)
* frosted "glass" cards with large radii, soft shadows and a bright hairline edge
* pill-shaped buttons with gradient fills, pastel accent presets (lavender, peach, mint, sky, rose)
  or the Windows accent colour; light + dark, "system" follows Windows live
* on Windows 11 the title bar is tinted to the canvas, follows light / dark and has rounded corners

``C`` is a module-level dict that is updated *in place* by :func:`apply_theme`, so widgets that read
``C[...]`` while painting pick up a theme change on their next repaint. ``MOTION`` holds the motion
level (full / reduced / off) for ui/motion.py.
"""

from __future__ import annotations

import sys

if sys.platform.startswith("win"):
    FONT_FAMILY = "'Segoe UI Variable Text', 'Segoe UI Variable', 'Segoe UI'"
    FONT_DISPLAY = "'Segoe UI Variable Display', 'Segoe UI Variable', 'Segoe UI'"
else:
    FONT_FAMILY = "Inter, 'Segoe UI', 'Noto Sans', sans-serif"
    FONT_DISPLAY = FONT_FAMILY

BRAND_ACCENT = "#7458FF"          # "Sunset Pop": electric violet ...
BRAND_GRAD = "#FF4F8B"            # ... into watermelon pink (the gradient's warm end)
ACCENTS = {                      # accent presets (Settings -> Appearance); each has its own gradient end
    "brand": "#7458FF",          # Sunset Pop (FocusBuddy): violet -> watermelon
    "peach": "#FF7A2F",          # Mango: orange -> sunflower
    "mint": "#0FBF94",           # Lagoon: jade -> sky
    "sky": "#2D7BFF",            # Ocean: cobalt -> cyan
    "rose": "#F23D7F",           # Berry: raspberry -> violet
}
ACCENT_GRADS = {"brand": BRAND_GRAD, "peach": "#FFC21A", "mint": "#33B6FF", "sky": "#18D3E8", "rose": "#9B5CFF"}

LIGHT = {
    "bg": "#FFF8F3",             # canvas base (behind the glows): warm cream
    "grad_a": "#FFF1E6",         # canvas gradient: apricot
    "grad_b": "#F6EEFF",         # lilac
    "grad_c": "#E8F7FF",         # ice blue
    "blob1": "#FFC79A",          # drifting glows: mango
    "blob2": "#C9B6FF",          # violet
    "blob3": "#9FEBD4",          # mint
    "blob4": "#FFB3D1",          # bubble-gum
    "sidebar": "#FFFFFF",
    "glass": "rgba(255, 255, 255, 200)",          # card fill (CSS)
    "glass_strong": "rgba(255, 255, 255, 236)",
    "glass_edge": "rgba(255, 255, 255, 245)",
    "glass_hex": "#D2FFFFFF",                     # the same for QPainter (#AARRGGBB)
    "surface": "#FFFFFF",
    "surface_alt": "#FBF7FF",
    "surface_hover": "#F4EEFF",
    "surface_press": "#EBE2FF",
    "border": "#EDE6F7",
    "border_strong": "#D7CBEE",
    "text": "#1F1838",
    "muted": "#625B80",
    "faint": "#9A93B4",
    "focus": "#0FA97F",
    "focus_soft": "#D9F7EC",
    "distract": "#FF7A45",
    "distract_soft": "#FFEADF",
    "danger": "#EE3D6D",
    "danger_soft": "#FFE3EB",
    "warning": "#F2A20C",
    "sun": "#FFB020",            # cheerful highlights (badges, streaks)
    "grid": "#EFE8F8",
    "toggle_off": "#D7CBEE",
    "shadow": "#4B2E9E",
    "tooltip_bg": "#1F1838",
    "tooltip_fg": "#FFFFFF",
    "on_accent": "#FFFFFF",
    "scroll": "#D7CBEE",
    "chart_focus": "#7458FF",
    "chart_distract": "#FF7A45",
}

DARK = {
    "bg": "#0E0B1A",             # midnight plum
    "grad_a": "#140F26",
    "grad_b": "#110F2A",
    "grad_c": "#0B1622",
    "blob1": "#5A2CC2",          # violet glow
    "blob2": "#9C2361",          # watermelon glow
    "blob3": "#0E6E66",          # teal glow
    "blob4": "#8A4A12",          # ember
    "sidebar": "#181331",
    "glass": "rgba(30, 24, 54, 205)",
    "glass_strong": "rgba(36, 29, 64, 236)",
    "glass_edge": "rgba(255, 255, 255, 28)",
    "glass_hex": "#CD1E1836",
    "surface": "#1E1836",
    "surface_alt": "#251E40",
    "surface_hover": "#2E2550",
    "surface_press": "#281F47",
    "border": "#342A58",
    "border_strong": "#463A72",
    "text": "#F6F2FF",
    "muted": "#B9B0D8",
    "faint": "#857CA8",
    "focus": "#3BE3B0",
    "focus_soft": "#123A31",
    "distract": "#FF9566",
    "distract_soft": "#3F2419",
    "danger": "#FF6E96",
    "danger_soft": "#451B2C",
    "warning": "#FFC547",
    "sun": "#FFC547",
    "grid": "#2C2449",
    "toggle_off": "#463A72",
    "shadow": "#000000",
    "tooltip_bg": "#2E2550",
    "tooltip_fg": "#F6F2FF",
    "on_accent": "#FFFFFF",
    "scroll": "#463A72",
    "chart_focus": "#9B86FF",
    "chart_distract": "#FF9566",
}

C: dict = {}
SERIES = ["#7C6CF6", "#21B58C", "#F08A5D", "#E2557A", "#3D9BF0", "#E9A93B"]
STATE = {"dark": False, "mode": "system", "accent": "brand"}
MOTION = {"level": "full"}       # full | reduced | off


def motion_level() -> str:
    return MOTION.get("level", "full")


def set_motion(level: str) -> None:
    MOTION["level"] = level if level in ("full", "reduced", "off") else "full"


# ------------------------------------------------------------------ colour helpers
def _rgb(hex_: str) -> tuple[int, int, int]:
    h = hex_.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _hex(r: float, g: float, b: float) -> str:
    return "#%02X%02X%02X" % tuple(max(0, min(255, int(round(v)))) for v in (r, g, b))


def mix(a: str, b: str, t: float) -> str:
    """Blend colour ``a`` toward ``b`` by ``t`` (0..1)."""
    ra, ga, ba = _rgb(a)
    rb, gb, bb = _rgb(b)
    return _hex(ra + (rb - ra) * t, ga + (gb - ga) * t, ba + (bb - ba) * t)


def luminance(c: str) -> float:
    r, g, b = (v / 255.0 for v in _rgb(c))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


# ------------------------------------------------------------------ system queries
def system_is_dark() -> bool:
    """The Windows (or desktop) app theme. Qt 6.5+ colorScheme first, then the Windows registry."""
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QGuiApplication
        if QGuiApplication.instance() is not None:
            scheme = QGuiApplication.styleHints().colorScheme()
            if scheme == Qt.ColorScheme.Dark:
                return True
            if scheme == Qt.ColorScheme.Light:
                return False
    except Exception:  # noqa: BLE001 - older Qt
        pass
    if sys.platform.startswith("win"):
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                                r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize") as k:
                return int(winreg.QueryValueEx(k, "AppsUseLightTheme")[0]) == 0
        except OSError:
            pass
    return False


def system_accent() -> str | None:
    """The Windows accent colour (None elsewhere / unavailable)."""
    if not sys.platform.startswith("win"):
        return None
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\DWM") as k:
            v = int(winreg.QueryValueEx(k, "AccentColor")[0]) & 0xFFFFFFFF      # 0xAABBGGRR
            return _hex(v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF)
    except OSError:
        pass
    try:
        from PySide6.QtGui import QGuiApplication, QPalette
        c = QGuiApplication.palette().color(QPalette.ColorRole.Accent)
        return c.name().upper()
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------------ palette
def palette(dark: bool, accent: str, grad: str | None = None) -> dict:
    p = dict(DARK if dark else LIGHT)
    if dark and luminance(accent) < 0.45:
        accent = mix(accent, "#FFFFFF", 0.28)          # a lighter accent reads better on dark glass
    grad = grad or mix(accent, "#FF5C9A", 0.45)
    if dark:
        grad = mix(grad, "#FFFFFF", 0.12)
    p["primary"] = accent
    p["primary_dark"] = mix(accent, "#000000", 0.14) if not dark else mix(accent, "#FFFFFF", 0.14)
    p["primary_light"] = mix(accent, "#FFFFFF", 0.35) if not dark else mix(accent, "#FFFFFF", 0.3)
    p["primary_soft"] = mix(p["surface"], accent, 0.12 if not dark else 0.28)
    p["primary_text"] = mix(accent, "#000000", 0.12) if not dark else mix(accent, "#FFFFFF", 0.3)
    p["on_accent"] = "#FFFFFF" if luminance(mix(accent, grad, 0.5)) < 0.7 else "#1B1530"
    # the gradient's second stop: every preset has its own vivid partner colour
    p["primary_grad"] = grad
    p["primary_grad_soft"] = mix(p["surface"], grad, 0.14 if not dark else 0.3)
    p["chart_focus"] = accent if not dark else mix(accent, "#FFFFFF", 0.12)
    return p


def tok(c: str) -> str:
    """A colour from a palette token name ("primary", "chart_focus", ...) or a literal "#RRGGBB"."""
    return C.get(c, c) if isinstance(c, str) else c


def resolve(mode: str = "system", accent: str = "brand") -> tuple[bool, str, str | None]:
    """(dark, accent colour, its gradient partner or None = derived)."""
    dark = system_is_dark() if mode == "system" else mode == "dark"
    acc, grad = BRAND_ACCENT, BRAND_GRAD
    if accent == "system":
        acc, grad = (system_accent() or BRAND_ACCENT), None
    elif accent in ACCENTS:
        acc, grad = ACCENTS[accent], ACCENT_GRADS.get(accent)
    elif isinstance(accent, str) and accent.startswith("#") and len(accent) == 7:
        acc, grad = accent.upper(), None
    return dark, acc, grad


def set_palette(mode: str = "system", accent: str = "brand") -> bool:
    dark, acc, grad = resolve(mode, accent)
    C.clear()
    C.update(palette(dark, acc, grad))
    STATE.update(dark=dark, mode=mode, accent=accent)
    SERIES[0] = C["primary"]
    return dark


# ------------------------------------------------------------------ stylesheet
def build_stylesheet() -> str:
    c = C
    dark = STATE.get("dark")
    hover_btn = "rgba(255, 255, 255, 235)" if not dark else "rgba(60, 52, 86, 235)"
    return f"""
* {{ font-family: {FONT_FAMILY}; color: {c['text']}; outline: none; }}
QMainWindow, QDialog {{ background: {c['bg']}; }}
#Root, QWidget#Page, QWidget#Transparent {{ background: transparent; }}
QScrollArea {{ background: transparent; border: none; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QLabel {{ background: transparent; }}
QLabel#H1 {{ font-family: {FONT_DISPLAY}; font-size: 30px; font-weight: 700; letter-spacing: -0.4px; }}
QLabel#H2 {{ font-family: {FONT_DISPLAY}; font-size: 19px; font-weight: 650; }}
QLabel#H3 {{ font-size: 15px; font-weight: 650; }}
QLabel#Eyebrow {{ font-size: 11px; font-weight: 700; letter-spacing: 1.2px; color: {c['primary_text']}; }}
QLabel#Muted {{ color: {c['muted']}; font-size: 13px; }}
QLabel#Faint {{ color: {c['faint']}; font-size: 12px; }}
QLabel#Error {{ color: {c['danger']}; font-size: 13px; font-weight: 600; }}
QLabel#FieldTitle {{ color: {c['muted']}; font-size: 12px; font-weight: 650; }}
QLabel#CardDesc {{ color: {c['muted']}; font-size: 12px; background: transparent; }}
QLabel#Big {{ font-family: {FONT_DISPLAY}; font-size: 42px; font-weight: 700; letter-spacing: -0.8px; }}
QLabel#Timer {{ font-family: {FONT_DISPLAY}; font-size: 58px; font-weight: 700; letter-spacing: -1px; }}
QLabel#Pill {{ background: {c['focus_soft']}; color: {c['focus']}; border-radius: 12px; padding: 5px 14px;
              font-size: 12px; font-weight: 650; }}
QLabel#PillPaused {{ background: {c['distract_soft']}; color: {c['distract']}; border-radius: 12px;
              padding: 5px 14px; font-size: 12px; font-weight: 650; }}
QFrame#Card {{ background: {c['glass']}; border: 1px solid {c['glass_edge']}; border-radius: 22px; }}
QFrame#SoftCard {{ background: {c['primary_soft']}; border: none; border-radius: 22px; }}
QFrame#Sidebar {{ background: {c['glass_strong']}; border: 1px solid {c['glass_edge']}; border-radius: 26px; }}
QFrame#TopBar {{ background: transparent; }}
QPushButton {{ background: {c['glass_strong']}; border: 1px solid {c['border']}; border-radius: 15px;
              padding: 8px 18px; font-size: 13px; font-weight: 650; }}
QPushButton:hover {{ background: {hover_btn}; border-color: {c['border_strong']}; }}
QPushButton:pressed {{ background: {c['surface_press']}; }}
QPushButton:disabled {{ color: {c['faint']}; }}
QPushButton#Primary {{ color: {c['on_accent']}; border: none;
              background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {c['primary']}, stop:1 {c['primary_grad']}); }}
QPushButton#Primary:hover {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {c['primary_dark']},
              stop:1 {c['primary']}); }}
QPushButton#Primary:pressed {{ background: {c['primary_dark']}; }}
QPushButton#Primary:disabled {{ background: {c['border_strong']}; color: {c['faint']}; }}
QPushButton#Danger {{ background: {c['danger_soft']}; color: {c['danger']}; border: none; }}
QPushButton#Danger:hover {{ background: {mix(c['danger_soft'], c['danger'], 0.18)}; }}
QPushButton#Ghost {{ background: transparent; border: none; color: {c['muted']}; }}
QPushButton#Ghost:hover {{ background: {c['surface_hover']}; color: {c['text']}; }}
QPushButton#Chip {{ border-radius: 15px; padding: 7px 16px; background: {c['glass']}; }}
QPushButton#Quick {{ border-radius: 18px; padding: 7px 16px; font-weight: 600; background: {c['glass_strong']};
              border: 1px solid {c['glass_edge']}; }}
QPushButton#Quick:hover {{ color: {c['primary']}; border: 1px solid {c['primary']}; background: {c['surface']}; }}
QPushButton#Quick:pressed {{ background: {c['surface_press']}; }}
QPushButton#Grip {{ border-radius: 8px; padding: 0px; font-size: 12px; color: {c['muted']};
              background: {c['surface']}; border: 1px solid {c['border']}; }}
QPushButton#Grip:hover {{ color: {c['primary']}; border: 1px solid {c['primary']}; }}
QPushButton#Chip:checked {{ color: {c['on_accent']}; border: none;
              background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {c['primary']}, stop:1 {c['primary_grad']}); }}
QPushButton#Nav {{ text-align: left; border: none; border-radius: 14px; padding: 9px 14px;
              font-size: 14px; font-weight: 600; color: {c['muted']}; background: transparent; }}
QPushButton#Nav:hover {{ color: {c['text']}; background: {c['surface_hover']}; }}
QPushButton#Nav:checked {{ color: {c['on_accent']}; background: transparent; }}
QPushButton#Nav:checked:hover {{ color: {c['on_accent']}; background: transparent; }}
QFrame#HeroCard QLabel {{ color: #FFFFFF; }}
QFrame#HeroCard QLabel#Eyebrow {{ color: rgba(255, 255, 255, 215); }}
QFrame#HeroCard QLabel#Muted {{ color: rgba(255, 255, 255, 228); font-size: 14px; }}
QFrame#HeroCard QLabel#Pill, QFrame#HeroCard QLabel#PillPaused {{ background: rgba(255, 255, 255, 60);
              color: #FFFFFF; }}
QFrame#HeroCard QPushButton {{ background: rgba(255, 255, 255, 46); color: #FFFFFF; border: 1px solid rgba(255, 255, 255, 90); }}
QFrame#HeroCard QPushButton:hover {{ background: rgba(255, 255, 255, 72); }}
QFrame#HeroCard QPushButton#Primary {{ background: #FFFFFF; color: {c['primary_text'] if not dark else mix(c['primary'], '#000000', 0.25)};
              border: none; }}
QFrame#HeroCard QPushButton#Primary:hover {{ background: rgba(255, 255, 255, 228); }}
QFrame#HeroCard QPushButton#Danger {{ background: rgba(255, 255, 255, 46); color: #FFFFFF; border: 1px solid rgba(255, 255, 255, 90); }}
QPushButton#Faq {{ text-align: left; background: transparent; border: none; border-radius: 12px;
              padding: 14px 16px; font-size: 15px; font-weight: 650; color: {c['text']}; }}
QPushButton#Faq:hover {{ background: {c['surface_hover']}; }}
QPushButton#Faq:checked {{ color: {c['primary_text']}; }}
QFrame#FaqItem {{ background: transparent; border: none; }}
QFrame#NavSep {{ background: {c['border']}; margin: 0px 12px; }}
QPushButton#Avatar {{ border-radius: 20px; padding: 0; border: 2px solid {c['glass_edge']}; background: transparent; }}
QPushButton#ChoiceCard {{ text-align: left; background: {c['glass']}; border: 1px solid {c['glass_edge']};
              border-radius: 20px; padding: 14px; font-weight: 500; }}
QPushButton#ChoiceCard:hover {{ background: {c['glass_strong']}; border-color: {c['primary_light']}; }}
QPushButton#ChoiceCard:checked {{ border: 2px solid {c['primary']}; background: {c['primary_soft']}; }}
QLineEdit, QSpinBox, QComboBox, QDoubleSpinBox {{ background: {c['glass_strong']}; border: 1px solid {c['border']};
              border-radius: 14px; padding: 7px 12px; font-size: 13px; min-height: 22px;
              selection-background-color: {c['primary']}; selection-color: {c['on_accent']}; }}
QLineEdit:hover, QSpinBox:hover, QComboBox:hover {{ border-color: {c['border_strong']}; }}
QLineEdit:focus, QSpinBox:focus, QComboBox:focus {{ border: 2px solid {c['primary']}; padding: 6px 11px; }}
QLineEdit#Search {{ border-radius: 18px; padding-left: 16px; background: {c['glass']}; }}
QLineEdit#Pin {{ font-size: 26px; letter-spacing: 10px; border-radius: 18px; padding: 10px 16px; }}
QComboBox::drop-down {{ border: none; width: 26px; }}
QComboBox QAbstractItemView {{ background: {c['surface']}; border: 1px solid {c['border']}; border-radius: 14px;
              padding: 6px; selection-background-color: {c['primary_soft']}; selection-color: {c['text']}; outline: none; }}
QSpinBox::up-button, QSpinBox::down-button {{ border: none; width: 18px; background: transparent; }}
QCheckBox {{ font-size: 13px; spacing: 8px; }}
QSlider::groove:horizontal {{ height: 6px; background: {c['border']}; border-radius: 3px; }}
QSlider::sub-page:horizontal {{ border-radius: 3px;
              background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {c['primary']}, stop:1 {c['primary_grad']}); }}
QSlider::handle:horizontal {{ background: {c['surface']}; border: 3px solid {c['primary']}; width: 14px; height: 14px;
              margin: -7px 0; border-radius: 10px; }}
QProgressBar {{ background: {c['border']}; border: none; border-radius: 5px; height: 10px; text-align: center; }}
QProgressBar::chunk {{ border-radius: 5px;
              background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {c['primary']}, stop:1 {c['primary_grad']}); }}
QTableView, QTableWidget {{ background: {c['glass_strong']}; border: 1px solid {c['glass_edge']}; border-radius: 18px;
              gridline-color: transparent; selection-background-color: {c['primary_soft']};
              selection-color: {c['text']}; font-size: 13px; alternate-background-color: {c['surface_alt']}; }}
QTableView::item {{ padding: 6px; }}
QListWidget {{ background: {c['glass_strong']}; border: 1px solid {c['glass_edge']}; border-radius: 18px; padding: 8px;
              font-size: 13px; outline: none; }}
QListWidget::item {{ padding: 8px 12px; border-radius: 12px; }}
QListWidget::item:selected {{ background: {c['primary_soft']}; color: {c['text']}; }}
QListWidget::item:hover {{ background: {c['surface_hover']}; }}
QHeaderView::section {{ background: transparent; border: none; border-bottom: 1px solid {c['border']};
              padding: 10px 8px; font-weight: 650; color: {c['muted']}; font-size: 12px; }}
QTableCornerButton::section {{ background: transparent; border: none; }}
QTabWidget::pane {{ border: none; }}
QTabBar::tab {{ background: transparent; padding: 8px 18px; margin-right: 6px; font-weight: 650; color: {c['muted']};
              border-radius: 15px; }}
QTabBar::tab:hover {{ color: {c['text']}; background: {c['surface_hover']}; }}
QTabBar::tab:selected {{ color: {c['on_accent']}; background: {c['primary']}; }}
QMenu {{ background: {c['surface']}; border: 1px solid {c['border']}; border-radius: 16px; padding: 6px; }}
QMenu::item {{ padding: 8px 28px 8px 14px; border-radius: 10px; font-size: 13px; color: {c['text']}; }}
QMenu::item:selected {{ background: {c['primary_soft']}; color: {c['text']}; }}
QMenu::item:disabled {{ color: {c['faint']}; }}
QMenu::separator {{ height: 1px; background: {c['border']}; margin: 5px 8px; }}
QToolTip {{ background: {c['tooltip_bg']}; color: {c['tooltip_fg']}; border: none; padding: 7px 10px;
              border-radius: 10px; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 4px 2px; }}
QScrollBar::handle:vertical {{ background: {c['scroll']}; border-radius: 4px; min-height: 36px; }}
QScrollBar::handle:vertical:hover {{ background: {c['faint']}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px 4px; }}
QScrollBar::handle:horizontal {{ background: {c['scroll']}; border-radius: 4px; min-width: 36px; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
QMessageBox, QInputDialog {{ background: {c['surface']}; }}
QMessageBox QLabel {{ color: {c['text']}; }}
"""


def stylesheet() -> str:
    """The stylesheet for the current theme (initialises the light/system palette on first use)."""
    if not C:
        set_palette(STATE["mode"], STATE["accent"])
    return build_stylesheet()


def apply_theme(qapp=None, mode: str = "system", accent: str = "brand") -> bool:
    """Set the palette, the Qt palette (native dialogs) and the app stylesheet. Returns True when dark."""
    dark = set_palette(mode, accent)
    global STYLESHEET
    STYLESHEET = build_stylesheet()
    try:
        from PySide6.QtGui import QColor, QPalette
        from PySide6.QtWidgets import QApplication
        qapp = qapp or QApplication.instance()
        if qapp is not None:
            pal = QPalette()
            for role, key in ((QPalette.Window, "bg"), (QPalette.WindowText, "text"), (QPalette.Base, "surface"),
                              (QPalette.AlternateBase, "surface_alt"), (QPalette.Text, "text"),
                              (QPalette.Button, "surface"), (QPalette.ButtonText, "text"),
                              (QPalette.ToolTipBase, "tooltip_bg"), (QPalette.ToolTipText, "tooltip_fg"),
                              (QPalette.Highlight, "primary"), (QPalette.HighlightedText, "on_accent"),
                              (QPalette.PlaceholderText, "faint"), (QPalette.Link, "primary_text")):
                pal.setColor(role, QColor(C[key]))
            qapp.setPalette(pal)
            qapp.setStyleSheet(STYLESHEET)
            for w in qapp.topLevelWidgets():
                if w.isVisible():
                    fluent_window(w)
    except Exception:  # noqa: BLE001 - theming must never break start-up
        import logging
        logging.getLogger("focusbuddy.ui").exception("theme apply failed")
    return dark


def themed(widget, css) -> None:
    """Give ``widget`` a style sheet built from the palette: ``css`` is a callable returning it. It is
    applied now and again by :func:`retheme_all` after every theme switch (an f-string with ``C[...]``
    set once would keep the old colours)."""
    widget._themed_css = css
    widget.setStyleSheet(css())


def retheme_all(qapp=None) -> None:
    """After :func:`apply_theme`: refresh everything that is not re-styled by the app style sheet -
    widgets made with :func:`themed`, widgets with a ``retheme()`` method (icons drawn in a colour),
    the Windows 11 title bars - and repaint every window. Nothing is closed or rebuilt."""
    from PySide6.QtWidgets import QApplication
    qapp = qapp or QApplication.instance()
    if qapp is None:
        return
    for w in qapp.allWidgets():
        css = getattr(w, "_themed_css", None)
        if css is not None:
            try:
                w.setStyleSheet(css())
            except Exception:  # noqa: BLE001
                pass
        fn = getattr(w, "retheme", None)
        if callable(fn):
            try:
                fn()
            except Exception:  # noqa: BLE001
                import logging
                logging.getLogger("focusbuddy.ui").exception("retheme failed for %s", type(w).__name__)
    for w in qapp.topLevelWidgets():
        if w.isVisible():
            fluent_window(w)
            w.update()
    for w in qapp.allWidgets():
        w.update()


def watch_system_theme(callback) -> bool:
    """Call ``callback()`` when the OS switches between light and dark (Qt 6.5+). Returns success."""
    try:
        from PySide6.QtGui import QGuiApplication
        QGuiApplication.styleHints().colorSchemeChanged.connect(lambda *_a: callback())
        return True
    except Exception:  # noqa: BLE001
        return False


# ------------------------------------------------------------------ Windows 11 frame
DWMWA_USE_IMMERSIVE_DARK_MODE = 20
DWMWA_WINDOW_CORNER_PREFERENCE = 33
DWMWA_SYSTEMBACKDROP_TYPE = 38
DWMWCP_ROUND = 2
DWMSBT_MAINWINDOW = 2          # Mica
DWMWA_CAPTION_COLOR = 35       # title bar tinted to the canvas (Windows 11)
DWMWA_TEXT_COLOR = 36


def fluent_window(widget) -> bool:
    """Windows 11 frame for a top-level window: title bar in the app theme (dark / light) and tinted to
    the canvas colour, rounded corners. No-op (False) elsewhere or on older Windows."""
    if not sys.platform.startswith("win") or widget is None:
        return False
    try:
        import ctypes
        from ctypes import wintypes
        hwnd = wintypes.HWND(int(widget.winId()))
        dwm = ctypes.windll.dwmapi
        ok = False
        def colorref(hex_: str) -> int:
            r, g, b = _rgb(hex_)
            return r | (g << 8) | (b << 16)
        for attr, value in ((DWMWA_USE_IMMERSIVE_DARK_MODE, 1 if STATE["dark"] else 0),
                            (DWMWA_WINDOW_CORNER_PREFERENCE, DWMWCP_ROUND),
                            (DWMWA_CAPTION_COLOR, colorref(C.get("grad_a", "#FFFFFF"))),
                            (DWMWA_TEXT_COLOR, colorref(C.get("text", "#000000")))):
            v = ctypes.c_int(value)
            ok = dwm.DwmSetWindowAttribute(hwnd, attr, ctypes.byref(v), ctypes.sizeof(v)) == 0 or ok
        return ok
    except Exception:  # noqa: BLE001 - older Windows: the classic frame stays
        return False


# initial palette (light until apply_theme() runs with the user's choice)
set_palette("light", "brand")
STYLESHEET = build_stylesheet()


def fmt_duration(seconds: float, short: bool = False) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if short:
        if h:
            return f"{h}h {m:02d}m"
        if m:
            return f"{m}m"
        return f"{s}s"
    if h:
        return f"{h}h {m:02d}m"
    return f"{m}m {s:02d}s" if m else f"{s}s"


def fmt_clock(seconds: float) -> str:
    seconds = int(max(0, round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
