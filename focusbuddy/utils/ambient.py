"""Ambient focus sounds, generated on this PC (no recordings, no downloads).

Each soundscape is synthesised once with NumPy into a seamless loop (a few tens of seconds,
cross-faded end to start) and cached as a WAV in the data folder:

* ``rain``       soft rain: filtered noise, gusts, and scattered droplets
* ``cafe``       café murmur: band-limited "voices", the odd cup clink
* ``brown``      brown noise (deep, steady - good for masking speech)
* ``pink``       pink noise (balanced)
* ``forest``     wind in the leaves and a few distant birds
* ``fireplace``  a low rumble with crackles and pops

On Windows the loop plays through ``waveOut`` (winmm) with a hardware loop, alongside the short
effect sounds; elsewhere playback is a no-op (the files are still generated, for tests).
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path
from typing import Optional

from .logging_setup import get_logger

log = get_logger("ambient")
RATE = 22050
SOUNDS = {"rain": "Soft rain", "cafe": "Café", "brown": "Brown noise", "pink": "Pink noise",
          "forest": "Forest", "fireplace": "Fireplace"}


def _np():
    import numpy as np
    return np


def _fft_filter(x, lo: float, hi: float, tilt: float = 0.0):
    """Band-pass in the frequency domain; ``tilt`` dB/octave (-3 pink, -6 brown)."""
    np = _np()
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / RATE)
    g = ((f >= lo) & (f <= hi)).astype(float)
    if tilt:
        g *= (np.maximum(f, 20.0) / 1000.0) ** (tilt / 6.02)
    return np.fft.irfft(X * g, n=len(x))


def _norm(x, peak: float = 0.6):
    np = _np()
    m = float(np.max(np.abs(x))) or 1.0
    return x * (peak / m)


def _loop(x, fade: float = 1.5):
    """Seamless loop: cross-fade the tail into the head."""
    np = _np()
    n = int(fade * RATE)
    head, body, tail = x[:n], x[n:-n], x[-n:]
    w = np.linspace(0.0, 1.0, n)
    return np.concatenate([tail * (1 - w) + head * w, body])


def generate(name: str, seconds: float = 24.0, seed: int = 7):
    np = _np()
    rng = np.random.default_rng(seed)
    n = int((seconds + 3.0) * RATE)
    t = np.arange(n) / RATE
    white = rng.standard_normal(n)
    if name == "brown":
        x = _fft_filter(white, 20, 2500, -6.0)
    elif name == "pink":
        x = _fft_filter(white, 20, 9000, -3.0)
    elif name == "rain":
        hiss = _fft_filter(white, 400, 7000, -2.0)
        gust = 0.75 + 0.25 * np.sin(2 * np.pi * t / 9.0) * np.sin(2 * np.pi * t / 3.7 + 1.0)
        x = _norm(hiss) * gust
        drops = np.zeros(n)
        for _ in range(int(seconds * 18)):
            i = int(rng.integers(0, n - 2000))
            k = np.arange(1200)
            f0 = rng.uniform(1800, 4200)
            drops[i:i + 1200] += np.sin(2 * np.pi * f0 * k / RATE) * np.exp(-k / rng.uniform(60, 160)) * rng.uniform(0.1, 0.4)
        x = x + drops
    elif name == "cafe":
        voices = np.zeros(n)
        for _ in range(7):
            band = _fft_filter(rng.standard_normal(n), rng.uniform(250, 400), rng.uniform(1200, 2600))
            env = 0.5 + 0.5 * np.sin(2 * np.pi * t * rng.uniform(0.15, 0.6) + rng.uniform(0, 6)) ** 2
            syll = 0.6 + 0.4 * np.abs(np.sin(2 * np.pi * t * rng.uniform(2.5, 4.5) + rng.uniform(0, 6)))
            voices += _norm(band) * env * syll
        x = _norm(voices) + 0.25 * _norm(_fft_filter(white, 60, 400, -6.0))
        for _ in range(int(seconds / 3)):
            i = int(rng.integers(0, n - 4000))
            k = np.arange(3000)
            f0 = rng.uniform(2600, 4200)
            x[i:i + 3000] += (np.sin(2 * np.pi * f0 * k / RATE) + 0.5 * np.sin(2 * np.pi * f0 * 2.7 * k / RATE)) \
                * np.exp(-k / 500.0) * 0.25
    elif name == "forest":
        wind = _fft_filter(white, 150, 3000, -4.0)
        x = _norm(wind) * (0.55 + 0.45 * np.sin(2 * np.pi * t / 7.0) ** 2)
        for _ in range(int(seconds / 2.5)):
            i = int(rng.integers(0, n - 9000))
            k = np.arange(int(rng.uniform(1500, 4000)))
            f0 = rng.uniform(2500, 5000)
            chirp = np.sin(2 * np.pi * (f0 + 600 * np.sin(2 * np.pi * 9 * k / RATE)) * k / RATE)
            env = np.sin(np.pi * k / len(k)) ** 2 * rng.uniform(0.08, 0.2)
            x[i:i + len(k)] += chirp * env
    elif name == "fireplace":
        rumble = _fft_filter(white, 30, 300, -6.0)
        x = _norm(rumble) * 0.7
        for _ in range(int(seconds * 7)):
            i = int(rng.integers(0, n - 800))
            k = np.arange(int(rng.uniform(80, 600)))
            x[i:i + len(k)] += rng.standard_normal(len(k)) * np.exp(-k / rng.uniform(20, 120)) * rng.uniform(0.1, 0.6)
    else:
        raise KeyError(name)
    return _norm(_loop(x), 0.5)


def ensure_file(name: str, cache_dir: Path) -> Optional[Path]:
    path = cache_dir / "ambient" / f"{name}.wav"
    if path.exists():
        return path
    try:
        np = _np()
        x = generate(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes((np.clip(x, -1, 1) * 32000).astype("<i2").tobytes())
        return path
    except Exception as exc:  # noqa: BLE001 - ambient sound is optional
        log.warning("ambient sound %s not generated: %s", name, exc)
        return None


class AmbientPlayer:
    """One looping soundscape at a time (Windows waveOut). ``play(name)`` / ``stop()`` / ``volume``."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.current = ""
        self.volume = 0.5
        self._h = None
        self._hdr = None
        self._buf = None

    @property
    def available(self) -> bool:
        return sys.platform.startswith("win")

    def play(self, name: str) -> bool:
        if name == self.current and self._h is not None:
            return True
        self.stop()
        if name not in SOUNDS:
            return False
        path = ensure_file(name, self.cache_dir)
        if path is None:
            return False
        self.current = name
        if not self.available:
            return False
        try:
            self._open(path)
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("ambient playback failed: %s", exc)
            self._h = None
            return False

    def set_volume(self, v: float) -> None:
        self.volume = max(0.0, min(1.0, float(v)))
        if self._h is not None:
            import ctypes
            lv = int(self.volume * 0xFFFF)
            ctypes.windll.winmm.waveOutSetVolume(self._h, lv | (lv << 16))

    def stop(self) -> None:
        self.current = ""
        if self._h is None:
            return
        try:
            import ctypes
            mm = ctypes.windll.winmm
            mm.waveOutReset(self._h)
            mm.waveOutUnprepareHeader(self._h, ctypes.byref(self._hdr), ctypes.sizeof(self._hdr))
            mm.waveOutClose(self._h)
        except Exception:  # noqa: BLE001
            pass
        self._h = self._hdr = self._buf = None

    # winmm: one buffer, looped by the driver (WHDR_BEGINLOOP | WHDR_ENDLOOP)
    def _open(self, path: Path) -> None:
        import ctypes
        from ctypes import wintypes

        class WAVEFORMATEX(ctypes.Structure):
            _fields_ = [("wFormatTag", wintypes.WORD), ("nChannels", wintypes.WORD),
                        ("nSamplesPerSec", wintypes.DWORD), ("nAvgBytesPerSec", wintypes.DWORD),
                        ("nBlockAlign", wintypes.WORD), ("wBitsPerSample", wintypes.WORD), ("cbSize", wintypes.WORD)]

        class WAVEHDR(ctypes.Structure):
            pass
        WAVEHDR._fields_ = [("lpData", ctypes.c_void_p), ("dwBufferLength", wintypes.DWORD),
                            ("dwBytesRecorded", wintypes.DWORD), ("dwUser", ctypes.c_void_p),
                            ("dwFlags", wintypes.DWORD), ("dwLoops", wintypes.DWORD),
                            ("lpNext", ctypes.c_void_p), ("reserved", ctypes.c_void_p)]
        with wave.open(str(path), "rb") as w:
            data = w.readframes(w.getnframes())
        mm = ctypes.windll.winmm
        fmt = WAVEFORMATEX(1, 1, RATE, RATE * 2, 2, 16, 0)
        h = wintypes.HANDLE()
        if mm.waveOutOpen(ctypes.byref(h), 0xFFFFFFFF, ctypes.byref(fmt), 0, 0, 0) != 0:
            raise OSError("waveOutOpen failed")
        self._buf = ctypes.create_string_buffer(data, len(data))
        hdr = WAVEHDR()
        hdr.lpData = ctypes.cast(self._buf, ctypes.c_void_p)
        hdr.dwBufferLength = len(data)
        hdr.dwFlags = 0x04 | 0x08                      # WHDR_BEGINLOOP | WHDR_ENDLOOP
        hdr.dwLoops = 0xFFFFFFFF
        mm.waveOutPrepareHeader(h, ctypes.byref(hdr), ctypes.sizeof(hdr))
        hdr.dwFlags |= 0x04 | 0x08
        self._h, self._hdr = h, hdr
        self.set_volume(self.volume)
        mm.waveOutWrite(h, ctypes.byref(hdr), ctypes.sizeof(hdr))
