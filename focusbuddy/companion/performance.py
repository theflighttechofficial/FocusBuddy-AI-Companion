"""Device performance profile and the live quality governor.

Two parts keep the companion fluid on any PC:

* :func:`probe` looks at the machine once: CPU cores, memory, the graphics adapter OpenGL reports,
  and whether it runs on battery / battery saver. It places the device in a class (``low``,
  ``mid``, ``high``) and from that picks the quality the companion *starts* at and the highest
  quality it may use (the ceiling).
* :class:`QualityGovernor` then watches the real frame times. If frames keep using too much of
  their budget it steps the quality down at once (big misses) or after a few seconds (small
  ones). When there is plenty of headroom for a long time it tries one step up again, and backs
  off for longer each time an upgrade fails, so it never flips back and forth.

The quality ladder is the renderer's own tiers (``low`` < ``medium`` < ``high`` < ``ultra``):
level of detail, supersampling, shadows, highlights. ``low`` also renders at 30 FPS on the GPU.

Performance modes (Settings -> Companion -> Performance):

* ``auto`` (default): start from the device class, adapt both ways, respect battery saver;
* ``quality``: the chosen quality, never lowered (except after a GPU failure);
* ``balanced``: at most ``medium``, still lowered when needed;
* ``battery``: ``low``, 30 FPS, no particles.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

TIERS = ("low", "medium", "high", "ultra")
MODES = ("auto", "quality", "balanced", "battery")


def tier_index(quality: str) -> int:
    q = str(quality).lower()
    if q in ("normal", "auto"):
        q = "medium"
    return TIERS.index(q) if q in TIERS else 1


# graphics adapters, by the OpenGL renderer string
_SOFTWARE = re.compile(r"llvmpipe|softpipe|swiftshader|software|gdi generic|basic render|mesa offscreen", re.I)
_DISCRETE = re.compile(r"nvidia|geforce|quadro|rtx|gtx|radeon\s*(rx|pro|r9|r7|hd\s*[6-9])|\barc\b|firepro", re.I)
_MID_IGPU = re.compile(r"uhd|iris|radeon\(tm\)\s*graphics|radeon graphics|vega|adreno|apple|mali", re.I)


@dataclass
class DeviceProfile:
    cores: int = 4
    memory_gb: float = 8.0
    gpu: str = ""
    gpu_class: str = "unknown"         # software | old_igpu | igpu | discrete | unknown | none
    on_battery: bool = False
    battery_saver: bool = False
    device_class: str = "mid"          # low | mid | high
    reasons: list = field(default_factory=list)

    @property
    def ceiling(self) -> str:
        """Highest quality this device should use."""
        return {"low": "medium", "mid": "high", "high": "ultra"}[self.device_class]

    @property
    def start(self) -> str:
        """Quality to start at (the governor moves from here)."""
        return {"low": "low", "mid": "medium", "high": "high"}[self.device_class]

    def summary(self) -> str:
        bits = [f"{self.device_class} ({self.gpu_class})", f"{self.cores} cores", f"{self.memory_gb:.0f} GB"]
        if self.battery_saver:
            bits.append("battery saver")
        elif self.on_battery:
            bits.append("on battery")
        return ", ".join(bits)


def classify_gpu(name: str) -> str:
    n = (name or "").strip()
    if not n:
        return "none"
    if _SOFTWARE.search(n):
        return "software"
    if _DISCRETE.search(n):
        return "discrete"
    if _MID_IGPU.search(n):
        return "igpu"
    low = n.lower()
    if "intel" in low:
        if re.search(r"gma|g41|g45|q45|q43", low):
            return "old_igpu"
        m = re.search(r"hd graphics\s*(\d+)?", low)
        if m:
            num = m.group(1)
            # HD 500-630 (2015+) are fine; HD 2000-5000 / P4000 / plain "HD Graphics" are older
            return "igpu" if num and len(num) == 3 else "old_igpu"
    return "unknown"


def memory_gb() -> float:
    if sys.platform.startswith("win"):
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return st.ullTotalPhys / 2 ** 30
        except Exception:  # noqa: BLE001
            pass
        return 8.0
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
    except (ValueError, OSError, AttributeError):
        return 8.0


def power_status() -> tuple[bool, bool]:
    """(on battery, battery saver on). (False, False) when unknown / not Windows."""
    if not sys.platform.startswith("win"):
        return False, False
    try:
        import ctypes

        class SYSTEM_POWER_STATUS(ctypes.Structure):
            _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                        ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                        ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]
        st = SYSTEM_POWER_STATUS()
        if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(st)):
            return False, False
        no_battery = st.BatteryFlag == 128
        on_battery = st.ACLineStatus == 0 and not no_battery
        return on_battery, st.SystemStatusFlag == 1
    except Exception:  # noqa: BLE001
        return False, False


def probe(gpu_name: Optional[str] = None) -> DeviceProfile:
    """Profile this machine. ``gpu_name`` is the OpenGL renderer string ('' = no GPU path)."""
    p = DeviceProfile()
    p.cores = max(1, os.cpu_count() or 4)
    p.memory_gb = memory_gb()
    if gpu_name is None:
        try:
            from PySide6.QtGui import QGuiApplication
            if QGuiApplication.instance() is None:       # no Qt application: no OpenGL to ask
                raise RuntimeError("no QGuiApplication")
            from .render3d_gl import GLSkinRaster
            gl = GLSkinRaster.create()
            gpu_name = getattr(gl, "renderer_name", "") if gl is not None else ""
        except Exception:  # noqa: BLE001
            gpu_name = ""
    p.gpu = gpu_name or ""
    p.gpu_class = classify_gpu(p.gpu)
    p.on_battery, p.battery_saver = power_status()
    score = {"discrete": 3, "igpu": 2, "unknown": 2, "old_igpu": 1, "software": 0, "none": 0}[p.gpu_class]
    if p.cores <= 2:
        score -= 1
        p.reasons.append("2 or fewer CPU cores")
    if p.memory_gb < 6:
        score -= 1
        p.reasons.append("less than 6 GB memory")
    if p.gpu_class in ("software", "none"):
        p.reasons.append("no hardware OpenGL")
    p.device_class = "high" if score >= 3 else "mid" if score >= 2 else "low"
    return p


def plan(mode: str, preferred: str, profile: DeviceProfile) -> dict:
    """What a performance mode means on this device: start / ceiling quality, whether the governor
    may lower it, the frame-rate cap and particles."""
    mode = mode if mode in MODES else "auto"
    pref = TIERS[tier_index(preferred)]
    if str(preferred).lower() in ("normal", "auto"):
        pref = "ultra"                       # "Automatic" quality: let the device decide
    if mode == "battery" or (mode == "auto" and profile.battery_saver):
        return {"start": "low", "ceiling": "low", "adaptive": False, "fps_cap": 30, "particles": False,
                "mode": "battery"}
    if mode == "quality":
        q = pref if str(preferred).lower() not in ("normal", "auto") else profile.ceiling
        return {"start": q, "ceiling": q, "adaptive": False, "fps_cap": 0, "particles": True, "mode": mode}
    if mode == "balanced":
        ceil = TIERS[min(tier_index(pref), 1)]
        return {"start": ceil, "ceiling": ceil, "adaptive": True, "fps_cap": 60, "particles": True, "mode": mode}
    ceil_i = min(tier_index(pref), tier_index(profile.ceiling))
    if profile.on_battery:
        ceil_i = min(ceil_i, 1)
    start_i = min(ceil_i, tier_index(profile.start))
    return {"start": TIERS[start_i], "ceiling": TIERS[ceil_i], "adaptive": True, "fps_cap": 0,
            "particles": profile.device_class != "low", "mode": "auto"}


class QualityGovernor:
    """Decides when to step the quality down or up from measured frame times (pure logic).

    ``feed(cost_ms, budget_ms, now)`` per rendered frame; returns ``"down"``, ``"up"`` or ``None``.
    Frames are judged in windows of ``WINDOW`` seconds by their 90th percentile cost:

    * severe: the median over ``1.6 x`` the budget -> down after one window;
    * over: the 90th percentile over ``0.9 x`` the budget -> down after two windows in a row;
    * headroom: the 90th percentile under ``0.45 x`` the budget for ``UP_AFTER`` seconds -> up one
      step (if below the ceiling and no downgrade happened recently). An upgrade that is followed
      by a downgrade within ``PROBATION`` seconds doubles the wait before the next try.
    """

    WINDOW = 2.0
    UP_AFTER = 20.0
    PROBATION = 12.0
    COOLDOWN = 4.0

    def __init__(self):
        self.reset()

    def reset(self, now: float = 0.0) -> None:
        self._t0: Optional[float] = None
        self._samples: list[float] = []
        self._over = 0
        self._calm_since: Optional[float] = None
        self._last_change = now
        self._last_up: Optional[float] = None
        self.up_wait = self.UP_AFTER

    def changed(self, direction: str, now: float) -> None:
        """Tell the governor a step was taken (resets the measurement)."""
        if direction == "down" and self._last_up is not None and now - self._last_up < self.PROBATION:
            self.up_wait = min(600.0, self.up_wait * 2)            # that upgrade did not hold
        if direction == "up":
            self._last_up = now
        self._last_change = now
        self._samples.clear()
        self._t0 = None
        self._over = 0
        self._calm_since = None

    def feed(self, cost_ms: float, budget_ms: float, now: float, can_up: bool = True,
             can_down: bool = True) -> Optional[str]:
        if cost_ms <= 0 or budget_ms <= 0:
            return None
        if self._t0 is None:
            self._t0 = now
        self._samples.append(cost_ms)
        if now - self._t0 < self.WINDOW or len(self._samples) < 8:
            return None
        s = sorted(self._samples)
        med = s[len(s) // 2]
        p90 = s[min(len(s) - 1, int(len(s) * 0.9))]
        self._samples.clear()
        self._t0 = now
        if now - self._last_change < self.COOLDOWN:
            return None
        if can_down and med > budget_ms * 1.6:
            return "down"
        if p90 > budget_ms * 0.9:
            self._over += 1
            self._calm_since = None
            if can_down and self._over >= 2:
                return "down"
            return None
        self._over = 0
        if p90 < budget_ms * 0.45:
            if self._calm_since is None:
                self._calm_since = now
            if can_up and now - self._calm_since >= self.up_wait and now - self._last_change >= self.up_wait:
                return "up"
        else:
            self._calm_since = None
        return None
