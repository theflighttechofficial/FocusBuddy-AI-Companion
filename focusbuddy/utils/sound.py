"""Tiny procedural sound effects (no audio files or extra deps needed).

Sounds are synthesised once into WAV files in the cache folder (one per
volume step) and played asynchronously with ``winsound`` on Windows.
On other platforms sound is a silent no-op.  Custom WAVs placed in
assets/sounds/<name>.wav override the synthesised ones.
"""

from __future__ import annotations

import math
import struct
import sys
import time
import wave
from pathlib import Path

from ..config import ASSETS_DIR
from .logging_setup import get_logger

log = get_logger("sound")
RATE = 22050


def _tone(freqs, dur, vol=0.5, decay=6.0, sweep_to=None):
    n = int(RATE * dur)
    try:
        import numpy as np                   # vectorised: the pure-Python loop cost ~0.4 s at start-up
        i = np.arange(n, dtype=np.float64)
        t = i / RATE
        env = np.minimum(1.0, t * 120) * np.exp(-decay * t)
        f0 = freqs[0]
        f = f0 + (sweep_to - f0) * (i / n) if sweep_to else np.full(n, float(f0))
        v = np.zeros(n)
        for k in range(len(freqs)):
            fk = f * (k + 1) if len(freqs) == 1 else np.full(n, float(freqs[k]))
            v += np.sin(2 * np.pi * fk * t) / (k + 1)
        return (v * env * vol).tolist()
    except ImportError:
        pass
    out = []
    for i in range(n):
        t = i / RATE
        env = min(1.0, t * 120) * math.exp(-decay * t)
        f0 = freqs[0]
        f = f0 + (sweep_to - f0) * (i / n) if sweep_to else f0
        v = sum(math.sin(2 * math.pi * (f * (k + 1) if len(freqs) == 1 else freqs[k]) * t) / (k + 1)
                for k in range(len(freqs)))
        out.append(v * env * vol)
    return out


def _seq(*parts):
    out = []
    for p in parts:
        out.extend(p)
    return out


SYNTH = {
    "alert": lambda: _seq(_tone([660], 0.11, 0.45, 9), _tone([880], 0.16, 0.45, 8)),
    "notify": lambda: _tone([880, 1320], 0.45, 0.4, 7),
    "celebrate": lambda: _seq(*[_tone([f], 0.09, 0.4, 10) for f in (523, 659, 784)], _tone([1047], 0.3, 0.42, 6)),
    "sleep": lambda: _tone([440], 0.5, 0.3, 4, sweep_to=300),
    "step": lambda: _tone([180], 0.035, 0.25, 60),
    "close": lambda: _tone([1200], 0.12, 0.45, 18, sweep_to=260),
    "tick": lambda: _tone([1500], 0.03, 0.3, 80),
    "pop": lambda: _tone([520], 0.08, 0.4, 25, sweep_to=900),
    "click": lambda: _seq(_tone([1800], 0.018, 0.45, 120), _tone([900], 0.05, 0.35, 60)),
    "jump": lambda: _tone([300], 0.16, 0.35, 12, sweep_to=780),
    "land": lambda: _tone([150], 0.09, 0.4, 35, sweep_to=90),
    "wake": lambda: _tone([330], 0.35, 0.3, 5, sweep_to=520),
    # emotion pack: cartoon "HEY!" honk, angry stomp thud, sigh, love chime, dizzy wobble, laugh
    "shout": lambda: _seq(_tone([392, 523], 0.07, 0.42, 6), _tone([466, 622], 0.22, 0.45, 5, sweep_to=700)),
    "stomp": lambda: _tone([95], 0.1, 0.5, 30, sweep_to=60),
    "sigh": lambda: _tone([520], 0.45, 0.18, 4, sweep_to=300),
    "love": lambda: _seq(_tone([784], 0.1, 0.3, 9), _tone([1047], 0.22, 0.3, 7)),
    "dizzy": lambda: _seq(*[_tone([f], 0.07, 0.25, 12) for f in (600, 500, 600, 500, 400)]),
    "laugh": lambda: _seq(*[_tone([f], 0.06, 0.28, 18) for f in (700, 620, 700, 620, 700)]),
    # 2.0 reactions: a soft "choo", a tiny "hic", a cosy hummed phrase
    "sneeze": lambda: _seq(_tone([300], 0.12, 0.12, 6, sweep_to=480), _tone([900], 0.09, 0.32, 30, sweep_to=260)),
    "hiccup": lambda: _tone([620], 0.06, 0.28, 40, sweep_to=980),
    "hum": lambda: _seq(*[_tone([f], 0.2, 0.12, 4) for f in (392, 440, 494, 440, 392)]),
}


def _write_wav(path: Path, samples, volume: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        try:
            import numpy as np
            frames = (np.clip(np.asarray(samples, dtype=np.float64) * volume, -1.0, 1.0) * 32000).astype("<i2").tobytes()
        except ImportError:
            frames = b"".join(struct.pack("<h", int(max(-1.0, min(1.0, s * volume)) * 32000)) for s in samples)
        w.writeframes(frames)


class SoundManager:
    def __init__(self, cache_dir: Path, enabled: bool = True, volume: float = 0.5):
        self.cache_dir = cache_dir / "sounds"
        self.enabled = enabled
        self.volume = volume
        self.muted = False
        self._last: dict[str, float] = {}
        self._winsound = None
        if sys.platform.startswith("win"):
            try:
                import winsound
                self._winsound = winsound
            except ImportError:  # pragma: no cover
                self._winsound = None

    def _file_for(self, name: str) -> Path | None:
        step = max(0, min(10, round(self.volume * 10)))
        if step == 0:
            return None
        custom = ASSETS_DIR / "sounds" / f"{name}.wav"
        if custom.exists() and step == 10:
            return custom
        path = self.cache_dir / f"{name}_{step}.wav"
        if not path.exists():
            gen = SYNTH.get(name)
            if gen is None:
                return None
            try:
                if custom.exists():
                    with wave.open(str(custom), "rb") as w:
                        raw = w.readframes(w.getnframes())
                        samples = [s[0] / 32768 for s in struct.iter_unpack("<h", raw)] if w.getsampwidth() == 2 else gen()
                else:
                    samples = gen()
                _write_wav(path, samples, step / 10)
            except OSError as exc:
                log.debug("sound cache write failed err=%s", exc)
                return None
        return path

    def play(self, name: str) -> None:
        if not self.enabled or self.muted or self._winsound is None:
            return
        now = time.perf_counter()
        if now - self._last.get(name, 0) < 0.15:
            return
        self._last[name] = now
        path = self._file_for(name)
        if path is None:
            return
        try:
            ws = self._winsound
            ws.PlaySound(str(path), ws.SND_FILENAME | ws.SND_ASYNC | ws.SND_NODEFAULT)
        except Exception as exc:  # noqa: BLE001 - sound must never crash anything
            log.debug("sound failed name=%s err=%s", name, exc)

    def prepare(self) -> None:
        """Pre-generate files for the current volume (called off the hot path)."""
        for name in SYNTH:
            self._file_for(name)
