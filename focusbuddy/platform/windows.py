"""Windows 10/11 backend using only ctypes (no pywin32 needed).

Privacy: we read the foreground window's title, executable and rectangle
plus the *time* since last input.  No keystrokes, no screenshots.

Safety: windows are closed only with a polite WM_CLOSE (the app may ask
to save).  TerminateProcess is never used anywhere in FocusBuddy.
"""

from __future__ import annotations

import ctypes
import os
import sys
import time
from ctypes import wintypes as W
from pathlib import Path
from typing import Optional

from ..utils.logging_setup import get_logger
from .base import AutostartManager, WindowInfo, WindowManager, WinRect

log = get_logger("platform")

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
try:
    dwmapi = ctypes.WinDLL("dwmapi")
except OSError:  # pragma: no cover
    dwmapi = None


class RECT(ctypes.Structure):
    _fields_ = [("left", W.LONG), ("top", W.LONG), ("right", W.LONG), ("bottom", W.LONG)]


class MONITORINFOEXW(ctypes.Structure):
    _fields_ = [("cbSize", W.DWORD), ("rcMonitor", RECT), ("rcWork", RECT),
                ("dwFlags", W.DWORD), ("szDevice", W.WCHAR * 32)]


class LASTINPUTINFO(ctypes.Structure):
    _fields_ = [("cbSize", W.UINT), ("dwTime", W.DWORD)]


WNDENUMPROC = ctypes.WINFUNCTYPE(W.BOOL, W.HWND, W.LPARAM)

user32.GetForegroundWindow.restype = W.HWND
user32.GetForegroundWindow.argtypes = []
user32.GetWindowTextLengthW.argtypes = [W.HWND]
user32.GetWindowTextLengthW.restype = ctypes.c_int
user32.GetWindowTextW.argtypes = [W.HWND, W.LPWSTR, ctypes.c_int]
user32.GetWindowTextW.restype = ctypes.c_int
user32.GetClassNameW.argtypes = [W.HWND, W.LPWSTR, ctypes.c_int]
user32.GetClassNameW.restype = ctypes.c_int
user32.GetWindowThreadProcessId.argtypes = [W.HWND, ctypes.POINTER(W.DWORD)]
user32.GetWindowThreadProcessId.restype = W.DWORD
for _fn in ("IsWindow", "IsWindowVisible", "IsIconic", "SetForegroundWindow", "BringWindowToTop"):
    getattr(user32, _fn).argtypes = [W.HWND]
    getattr(user32, _fn).restype = W.BOOL
user32.GetWindowRect.argtypes = [W.HWND, ctypes.POINTER(RECT)]
user32.GetWindowRect.restype = W.BOOL
user32.ShowWindow.argtypes = [W.HWND, ctypes.c_int]
user32.ShowWindow.restype = W.BOOL
user32.PostMessageW.argtypes = [W.HWND, W.UINT, W.WPARAM, W.LPARAM]
user32.PostMessageW.restype = W.BOOL
user32.MonitorFromWindow.argtypes = [W.HWND, W.DWORD]
user32.MonitorFromWindow.restype = W.HMONITOR
user32.GetMonitorInfoW.argtypes = [W.HMONITOR, ctypes.POINTER(MONITORINFOEXW)]
user32.GetMonitorInfoW.restype = W.BOOL
user32.GetLastInputInfo.argtypes = [ctypes.POINTER(LASTINPUTINFO)]
user32.GetLastInputInfo.restype = W.BOOL
user32.EnumChildWindows.argtypes = [W.HWND, WNDENUMPROC, W.LPARAM]
user32.EnumChildWindows.restype = W.BOOL
user32.EnumWindows.argtypes = [WNDENUMPROC, W.LPARAM]
user32.EnumWindows.restype = W.BOOL
user32.GetWindowLongW.argtypes = [W.HWND, ctypes.c_int]
user32.GetWindowLongW.restype = W.LONG
user32.GetAncestor.argtypes = [W.HWND, W.UINT]
user32.GetAncestor.restype = W.HWND
kernel32.GetTickCount.restype = W.DWORD


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", W.WORD), ("wScan", W.WORD), ("dwFlags", W.DWORD), ("time", W.DWORD),
                ("dwExtraInfo", ctypes.c_size_t)]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", W.LONG), ("dy", W.LONG), ("mouseData", W.DWORD), ("dwFlags", W.DWORD),
                ("time", W.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", W.DWORD), ("u", _INPUTUNION)]


user32.SendInput.argtypes = [W.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
user32.SendInput.restype = W.UINT
INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
VK_CONTROL, VK_W = 0x11, 0x57
kernel32.OpenProcess.argtypes = [W.DWORD, W.BOOL, W.DWORD]
kernel32.OpenProcess.restype = W.HANDLE
kernel32.QueryFullProcessImageNameW.argtypes = [W.HANDLE, W.DWORD, W.LPWSTR, ctypes.POINTER(W.DWORD)]
kernel32.QueryFullProcessImageNameW.restype = W.BOOL
kernel32.CloseHandle.argtypes = [W.HANDLE]
kernel32.CloseHandle.restype = W.BOOL
if dwmapi is not None:
    dwmapi.DwmGetWindowAttribute.argtypes = [W.HWND, W.DWORD, ctypes.c_void_p, W.DWORD]
    dwmapi.DwmGetWindowAttribute.restype = ctypes.c_long

PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
WM_CLOSE = 0x0010
SW_MINIMIZE = 6
SW_RESTORE = 9
MONITOR_DEFAULTTONEAREST = 2
DWMWA_EXTENDED_FRAME_BOUNDS = 9
DWMWA_CLOAKED = 14
GA_ROOT = 2
GWL_EXSTYLE = -20
WS_EX_TOOLWINDOW = 0x00000080


class WindowsWindowManager(WindowManager):
    def __init__(self):
        self._path_cache: dict[int, tuple[str, str, float]] = {}
        self.own_pid = os.getpid()

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _title(hwnd) -> str:
        n = user32.GetWindowTextLengthW(hwnd)
        if n <= 0:
            return ""
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, buf, n + 1)
        return buf.value

    @staticmethod
    def _class(hwnd) -> str:
        buf = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, buf, 256)
        return buf.value

    @staticmethod
    def _pid(hwnd) -> int:
        pid = W.DWORD(0)
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        return int(pid.value)

    def _exe(self, pid: int) -> str:
        return self._exe_path(pid)[0]

    def _exe_path(self, pid: int) -> tuple[str, str]:
        """(exe name lower-case, full image path) - cached for two minutes per pid."""
        now = time.monotonic()
        cached = self._path_cache.get(pid)
        if cached and now - cached[2] < 120:
            return cached[0], cached[1]
        name, full = "", ""
        h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if h:
            try:
                size = W.DWORD(1024)
                buf = ctypes.create_unicode_buffer(1024)
                if kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
                    full = buf.value
                    name = Path(full).name.lower()
            finally:
                kernel32.CloseHandle(h)
        if len(self._path_cache) > 512:
            self._path_cache.clear()
        self._path_cache[pid] = (name, full, now)
        return name, full

    @staticmethod
    def _rect(hwnd) -> Optional[WinRect]:
        r = RECT()
        ok = False
        if dwmapi is not None:
            # Visible frame bounds (excludes the invisible resize borders on Win10/11)
            ok = dwmapi.DwmGetWindowAttribute(hwnd, DWMWA_EXTENDED_FRAME_BOUNDS, ctypes.byref(r),
                                              ctypes.sizeof(r)) == 0
        if not ok and not user32.GetWindowRect(hwnd, ctypes.byref(r)):
            return None
        return WinRect(r.left, r.top, r.right - r.left, r.bottom - r.top)

    @staticmethod
    def _cloaked(hwnd) -> bool:
        if dwmapi is None:
            return False
        val = ctypes.c_int(0)
        dwmapi.DwmGetWindowAttribute(hwnd, DWMWA_CLOAKED, ctypes.byref(val), ctypes.sizeof(val))
        return bool(val.value)

    @staticmethod
    def _monitor(hwnd) -> tuple[Optional[str], Optional[WinRect]]:
        hmon = user32.MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST)
        if not hmon:
            return None, None
        mi = MONITORINFOEXW()
        mi.cbSize = ctypes.sizeof(MONITORINFOEXW)
        if not user32.GetMonitorInfoW(hmon, ctypes.byref(mi)):
            return None, None
        m = mi.rcMonitor
        return mi.szDevice, WinRect(m.left, m.top, m.right - m.left, m.bottom - m.top)

    def _uwp_exe(self, hwnd, host_pid: int) -> str:
        """UWP apps live inside ApplicationFrameHost; find the real app process."""
        found: list[int] = []

        @WNDENUMPROC
        def _cb(child, _lp):
            pid = self._pid(child)
            if pid and pid != host_pid:
                found.append(pid)
                return False
            return True

        user32.EnumChildWindows(hwnd, _cb, 0)
        return self._exe(found[0]) if found else "applicationframehost.exe"

    def _info(self, hwnd) -> Optional[WindowInfo]:
        if isinstance(hwnd, W.HWND):
            hwnd = hwnd.value
        if not hwnd or not user32.IsWindow(hwnd):
            return None
        hwnd = user32.GetAncestor(hwnd, GA_ROOT) or hwnd
        pid = self._pid(hwnd)
        exe, path = self._exe_path(pid)
        if exe == "applicationframehost.exe":
            exe = self._uwp_exe(hwnd, pid)
            path = ""
        mon_name, mon_rect = self._monitor(hwnd)
        return WindowInfo(
            handle=int(hwnd), title=self._title(hwnd), exe=exe, pid=pid, rect=self._rect(hwnd),
            monitor_name=mon_name, monitor_rect=mon_rect, minimized=bool(user32.IsIconic(hwnd)),
            coords="physical", extra={"class": self._class(hwnd), "path": path})

    # ----------------------------------------------------------------- API
    def get_active_window(self) -> Optional[WindowInfo]:
        try:
            hwnd = user32.GetForegroundWindow()
            if not hwnd or self._cloaked(hwnd):
                return None
            return self._info(hwnd)
        except OSError as exc:
            log.debug("get_active_window failed err=%s", exc)
            return None

    def get_window(self, handle: int) -> Optional[WindowInfo]:
        try:
            return self._info(W.HWND(handle))
        except OSError:
            return None

    def is_window(self, handle: int) -> bool:
        return bool(handle) and bool(user32.IsWindow(W.HWND(handle)))

    def minimize_window(self, handle: int) -> bool:
        if not self.is_window(handle):
            return False
        user32.ShowWindow(W.HWND(handle), SW_MINIMIZE)
        return bool(user32.IsIconic(W.HWND(handle)))

    def close_window(self, handle: int) -> bool:
        if not self.is_window(handle):
            return False
        # Graceful request; the target app decides (and can prompt to save).
        return bool(user32.PostMessageW(W.HWND(handle), WM_CLOSE, 0, 0))

    def close_browser_tab(self, handle: int, expect_title: str) -> bool:
        """Ctrl+W to the verified foreground browser window (no mouse, no other window)."""
        if not self.is_window(handle):
            return False
        fg = user32.GetForegroundWindow()
        fg = int(user32.GetAncestor(fg, GA_ROOT) or fg or 0)
        if fg != int(handle) or self._title(W.HWND(handle)) != (expect_title or ""):
            return False
        seq = [(VK_CONTROL, 0), (VK_W, 0), (VK_W, KEYEVENTF_KEYUP), (VK_CONTROL, KEYEVENTF_KEYUP)]
        arr = (INPUT * len(seq))()
        for i, (vk, flags) in enumerate(seq):
            arr[i].type = INPUT_KEYBOARD
            arr[i].u.ki = KEYBDINPUT(vk, 0, flags, 0, 0)
        sent = user32.SendInput(len(seq), arr, ctypes.sizeof(INPUT))
        return sent == len(seq)

    def activate_window(self, handle: int) -> bool:
        if not self.is_window(handle):
            return False
        h = W.HWND(handle)
        if user32.IsIconic(h):
            user32.ShowWindow(h, SW_RESTORE)
        user32.BringWindowToTop(h)
        return bool(user32.SetForegroundWindow(h))

    def list_windows(self) -> list[WindowInfo]:
        """Visible, non-minimized, non-cloaked top-level app windows in Z-order (front first)."""
        handles: list[int] = []

        @WNDENUMPROC
        def _cb(hwnd, _lp):
            try:
                if user32.IsWindowVisible(hwnd) and not user32.IsIconic(hwnd) and \
                        user32.GetWindowTextLengthW(hwnd) > 0 and not self._cloaked(hwnd) and \
                        not (user32.GetWindowLongW(hwnd, GWL_EXSTYLE) & WS_EX_TOOLWINDOW):
                    handles.append(int(hwnd))
            except OSError:
                pass
            return True

        try:
            user32.EnumWindows(_cb, 0)
        except OSError:
            return []
        out = []
        for h in handles[:40]:
            info = self._info(h)
            if info is None or info.pid == self.own_pid or info.rect is None:
                continue
            if info.rect.w < 150 or info.rect.h < 80:
                continue
            out.append(info)
        return out

    def get_idle_seconds(self) -> float:
        lii = LASTINPUTINFO()
        lii.cbSize = ctypes.sizeof(LASTINPUTINFO)
        if not user32.GetLastInputInfo(ctypes.byref(lii)):
            return 0.0
        elapsed_ms = (kernel32.GetTickCount() - lii.dwTime) & 0xFFFFFFFF
        return elapsed_ms / 1000.0


class WindowsAutostart(AutostartManager):
    KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
    NAME = "FocusBuddy"

    @staticmethod
    def command() -> str:
        if getattr(sys, "frozen", False):
            return f'"{sys.executable}" --minimized'
        exe = Path(sys.executable)
        pyw = exe.with_name("pythonw.exe")
        main = Path(__file__).resolve().parent.parent.parent / "main.py"
        return f'"{pyw if pyw.exists() else exe}" "{main}" --minimized'

    def is_enabled(self) -> bool:
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, self.KEY) as k:
                winreg.QueryValueEx(k, self.NAME)
                return True
        except OSError:
            return False

    def set_enabled(self, on: bool) -> bool:
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, self.KEY, 0, winreg.KEY_SET_VALUE) as k:
                if on:
                    winreg.SetValueEx(k, self.NAME, 0, winreg.REG_SZ, self.command())
                else:
                    try:
                        winreg.DeleteValue(k, self.NAME)
                    except FileNotFoundError:
                        pass
            return True
        except OSError as exc:
            log.warning("Autostart change failed err=%s", exc)
            return False
