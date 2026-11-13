"""Face life: the small things that make a face look inhabited.

``FaceLife`` is one additive layer on top of the emotion pack. It never chooses an expression
itself; it adds the motion a real face has *between* expressions:

* **talking** - when the companion says something, the jaw and mouth open per syllable
  (vowel groups of the actual words), pause at commas and full stops, the head nods on stressed
  words, the brows lift at the end of a question and pop on an exclamation;
* **micro-saccades** - the eyes never sit dead still: tiny jumps every half second or so;
* **eyebrow flash** - a quick brow lift when attention jumps to something new;
* **micro-smiles** - in a good mood a faint smile comes and goes on its own;
* **sighs** - when bored, sad, tired or lonely: a deep breath in, shoulders rise, then drop with
  a long out-breath and a slight slump;
* **thinking squint** - while thinking or reading the eyes narrow a little and drift upward;
* **head noise** - a slow, irregular drift of the head so the companion is never statue-still.

Everything is scaled by ``amount`` (the "Lifelike detail" setting: low 0.4, normal 0.75,
high 1.0) and is deterministic for a given seed, so tests and screenshots are repeatable.
"""

from __future__ import annotations

import math
import random
import re
import zlib
from typing import Optional

LEVELS = {"off": 0.0, "low": 0.4, "normal": 0.75, "high": 1.0}
_VOWELS = re.compile(r"[aeiouy]+", re.I)
_TOKEN = re.compile(r"[A-Za-z']+|[0-9]+|[,;:]|[.!?…]+")
GOOD_MOODS = {"HAPPY", "EXCITED", "PROUD", "LOVE", "GRATEFUL", "RELAXED", "PLAYFUL", "CONFIDENT", "SMUG",
              "RELIEVED", "CELEBRATING", "LAUGHING", "CALMING", "NEUTRAL", ""}
LOW_MOODS = {"BORED": 1.0, "SAD": 1.0, "LONELY": 1.0, "SLEEPY": 0.8, "WORRIED": 0.5, "GRUMPY": 0.6,
             "IMPATIENT": 0.5, "NERVOUS": 0.4}


def syllables(word: str) -> int:
    w = word.lower()
    n = len(_VOWELS.findall(w))
    if w.endswith("e") and n > 1 and not w.endswith(("le", "ee", "ye")):
        n -= 1
    return max(1, n)


def speech_track(text: str, rate: float = 5.2, rng: Optional[random.Random] = None) -> list[tuple]:
    """Turn text into timed mouth events: ``(t0, dur, open, kind)`` with kind in syl / stress / pause.

    ``open`` is the jaw opening for that syllable (0..1). Numbers are read as about two syllables
    per digit group; punctuation adds pauses. The track is what the mouth does, not audio."""
    rng = rng or random.Random(len(text))
    out: list[tuple] = []
    t = 0.05
    step = 1.0 / max(1.0, rate)
    words = 0
    for tok in _TOKEN.findall(text or ""):
        if tok in (",", ";", ":"):
            out.append((t, 0.2, 0.0, "pause"))
            t += 0.2
            continue
        if tok[0] in ".!?…":
            out.append((t, 0.34, 0.0, "pause"))
            t += 0.34
            continue
        n = min(6, syllables(tok) if tok[0].isalpha() else 1 + len(tok) // 2)
        words += 1
        stress = words == 1 or (len(tok) > 5 and rng.random() < 0.5)
        for i in range(n):
            d = step * rng.uniform(0.82, 1.18)
            o = rng.uniform(0.3, 0.62) * (1.2 if (stress and i == 0) else 1.0)
            out.append((t, d, min(1.0, o), "stress" if (stress and i == 0) else "syl"))
            t += d
        t += step * 0.18                                   # tiny gap between words
    return out


class FaceLife:
    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)
        self.amount = 1.0
        self.track: list[tuple] = []
        self.talk_t0 = -100.0
        self.talk_end = -100.0
        self.question = False
        self.exclaim = False
        self._sacc = (0.0, 0.0)
        self._sacc_next = 0.0
        self._flash_t0 = -100.0
        self._smile_t0 = -100.0
        self._smile_next = 6.0
        self._smile_amp = 0.0
        self._sigh_t0 = -100.0
        self._sigh_next = 18.0
        self._seed = self._rng.uniform(0, 100)
        self.sighs = 0          # counters (tests / debug)
        self.flashes = 0

    # -------------------------------------------------------------- input
    def talk(self, text: str, now: float, duration: float | None = None) -> float:
        """Start lip motion for ``text``. Returns how long the mouth moves (seconds)."""
        if not text or self.amount <= 0:
            return 0.0
        tr = speech_track(text, rng=random.Random(zlib.crc32(text.encode()) ^ int(self._seed)))
        if not tr:
            return 0.0
        length = tr[-1][0] + tr[-1][1]
        cap = max(1.2, min(9.0, (duration or 9.0) * 0.8))
        if length > cap:                                   # long bubble: speak faster, don't run on
            k = cap / length
            tr = [(a * k, d * k, o, kind) for a, d, o, kind in tr]
            length = cap
        self.track = tr
        self.talk_t0 = now
        self.talk_end = now + length
        s = text.strip()
        self.question = s.endswith("?")
        self.exclaim = "!" in s[-3:] if s else False
        return length

    def stop_talking(self) -> None:
        self.talk_end = -100.0
        self.track = []

    def flash(self, now: float) -> None:
        """Eyebrow flash: attention just jumped to something new."""
        if now - self._flash_t0 > 1.2:
            self._flash_t0 = now
            self.flashes += 1

    @property
    def talking(self) -> bool:
        return bool(self.track)

    # ------------------------------------------------------------- output
    def mouth_at(self, now: float) -> tuple[float, bool]:
        """(jaw opening 0..1, on a stressed syllable) at ``now``."""
        if not self.track:
            return 0.0, False
        t = now - self.talk_t0
        if now > self.talk_end + 0.1:
            self.track = []
            return 0.0, False
        for a, d, o, kind in self.track:
            if a <= t < a + d:
                if kind == "pause":
                    return 0.0, False
                u = (t - a) / d
                # quick open, softer close - like a real syllable
                env = math.sin(math.pi * min(1.0, u * 1.15)) if u < 0.87 else 0.0
                return o * max(0.0, env), kind == "stress" and u < 0.5
            if a > t:
                break
        return 0.0, False

    def apply(self, p, now: float, dt: float, emotion: str = "", asleep: bool = False,
              thinking: bool = False, moving: float = 0.0, mechanical: bool = False) -> None:
        a = self.amount
        if a <= 0.0:
            return
        if asleep:
            self.stop_talking()
            return
        # --- talking ---------------------------------------------------
        if self.track:
            o, stress = self.mouth_at(now)
            p.mouth_open = max(p.mouth_open, o * 0.75 * (0.6 + 0.4 * a))
            p.jaw = max(p.jaw, o * 0.9)
            p.mouth_width += (o - 0.3) * -0.12             # rounder when open
            t = now - self.talk_t0
            span = max(0.2, self.talk_end - self.talk_t0)
            if stress:
                p.head_dy += 1.1 * a                       # little nod on the stressed word
                p.brow_raise += 0.12 * a
            p.head_dx += math.sin(t * 2.3 + self._seed) * 0.6 * a
            p.head_roll += math.sin(t * 1.7 + self._seed) * 1.6 * a
            if self.question and t > span * 0.65:          # questions end with lifted brows, head cocked
                q = min(1.0, (t - span * 0.65) / max(0.15, span * 0.35))
                p.brow_raise += 0.45 * q * a
                p.head_roll += 5.0 * q * a
            if self.exclaim and t < 0.45:
                p.brow_raise += 0.4 * a * (1 - t / 0.45)
                p.lift += 1.2 * a * math.sin(math.pi * t / 0.45)
            p.smile += 0.08 * a
        # --- eyebrow flash ---------------------------------------------
        ft = now - self._flash_t0
        if ft < 0.42:
            p.brow_raise += 0.55 * a * math.sin(math.pi * ft / 0.42)
            p.eye_open *= 1.0 + 0.08 * math.sin(math.pi * ft / 0.42)
        # --- micro-saccades (eyes never sit dead still) -----------------
        if now >= self._sacc_next:
            r = 0.05 * a
            self._sacc = (self._rng.uniform(-r, r), self._rng.uniform(-r * 0.6, r * 0.6))
            self._sacc_next = now + self._rng.uniform(0.35, 1.3)
        p.look_x += self._sacc[0]
        p.look_y += self._sacc[1]
        # --- head noise: slow, irregular drift --------------------------
        still = 1.0 - min(1.0, moving)
        s = self._seed
        hx = math.sin(now * 0.37 + s) * 0.6 + math.sin(now * 0.91 + s * 1.7) * 0.4
        hy = math.sin(now * 0.29 + s * 0.6) * 0.6 + math.sin(now * 0.73 + s * 2.3) * 0.4
        mech = 0.35 if mechanical else 1.0
        p.head_dx += hx * 0.45 * a * still * mech
        p.head_dy += hy * 0.3 * a * still * mech
        p.head_roll += hx * 1.1 * a * still * mech
        # --- thinking squint -------------------------------------------
        if thinking:
            p.lid += 0.14 * a
            p.lid_low += 0.1 * a
            p.look_y -= 0.18 * a
            p.brow_raise += 0.1 * a * math.sin(now * 0.8 + s)
        # --- micro-smiles in a good mood --------------------------------
        if emotion in GOOD_MOODS and not self.track:
            if now >= self._smile_next:
                self._smile_t0 = now
                self._smile_amp = self._rng.uniform(0.15, 0.32)
                self._smile_next = now + self._rng.uniform(7.0, 19.0) / max(0.4, a)
            st = now - self._smile_t0
            if st < 1.6:
                w = math.sin(math.pi * st / 1.6) ** 2 * self._smile_amp * a
                p.smile += w
                p.lid_low += w * 0.5
        # --- sighs when low --------------------------------------------
        low = LOW_MOODS.get(emotion, 0.0)
        if low > 0 and not self.track and moving < 0.2:
            if now >= self._sigh_next:
                self._sigh_t0 = now
                self.sighs += 1
                self._sigh_next = now + self._rng.uniform(14.0, 30.0) / (low * max(0.4, a))
        sg = now - self._sigh_t0
        if sg < 2.6:
            if sg < 1.0:                                   # slow inhale: chest and shoulders rise
                u = math.sin(0.5 * math.pi * sg / 1.0)
                p.breath = max(p.breath, 0.9 * u)
                p.shoulders += 0.35 * u * a
                p.head_dy -= 1.0 * u * a
                p.puff = max(p.puff, 0.25 * u * a)
            else:                                          # long exhale: drop, slump, lids heavy
                u = (sg - 1.0) / 1.6
                d = math.sin(math.pi * min(1.0, u))
                p.breath = min(p.breath, 0.9 - 1.8 * min(1.0, u * 1.4))
                p.shoulders -= 0.3 * d * a
                p.slump += 0.25 * d * a
                p.lid += 0.18 * d * a
                p.mouth_open = max(p.mouth_open, 0.12 * d * a)
