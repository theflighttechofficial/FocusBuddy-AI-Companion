"""2.1 daily rhythm around the companion: care, naps, check-ins and focus extras.

``Routines`` is owned by the app and hooks into its timers and focus events:

* **Companion care** (``care.py``): needs drift slowly; focus earns treats and bond XP; the Care page
  and the companion's right-click menu feed / play / pet; level-ups and badges get a little party.
* **Naps**: "Take a nap" sends the companion off-screen to sleep (``Life.nap``). It also naps by
  itself while you're away from the PC (Settings → Routines) and comes back when you return, and
  it can keep a nap schedule (e.g. 13:30 for 20 minutes).
* **Morning check-in**: the first time you're at the PC each day, a hello and "what's the one thing
  today?" with a shortcut to your tasks or a session.
* **Wind-down**: at your wind-down time, a gentle "let's wrap up" with your day, then a nap.
* **Focus extras**: a goal per session ("Did you finish …?" at the end), ambient sounds that play
  during sessions, streak rewards, and Deep focus (a quiet companion, no chatter, strict closing).
"""

from __future__ import annotations

import datetime as dt
import time
from typing import Optional

from ..utils.logging_setup import get_logger
from .care import BADGES, CompanionCare

log = get_logger("routines")

PLAY_STEPS = {"dog": "fetch", "cat": ("POUNCE", "PLAY"), "fox": ("PLAY_BOW", "POUNCE"),
              "rabbit": ("BOUNCE", "HOP"), "robot": ("DANCE",), "buddy": ("DANCE", "CHEER"),
              "girl": ("DANCE", "CHEER"), "saree": ("SPIN", "CLAP")}
PET_STEPS = {"cat": ("HEAD_BUMP", "PURR"), "dog": ("TAIL_WAG", "LOVE"), "fox": ("TAIL_WAG", "LOVE"),
             "rabbit": ("NOSE_TWITCH", "LOVE"), "robot": ("LOVE",), "buddy": ("LOVE", "WAVE"),
             "girl": ("LOVE", "WAVE"), "saree": ("LOVE", "WAVE")}


def parse_times(text: str) -> list[tuple[int, int]]:
    """'13:30, 16:00' -> [(13, 30), (16, 0)] (invalid parts are skipped)."""
    out = []
    for part in (text or "").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            h, m = part.split(":") if ":" in part else (part, "0")
            h, m = int(h), int(m)
        except ValueError:
            continue
        if 0 <= h < 24 and 0 <= m < 60:
            out.append((h, m))
    return out


class Routines:
    def __init__(self, app, now=time.time, wall=dt.datetime.now):
        self.app = app
        self._now = now
        self._wall = wall
        self.care = CompanionCare(app.settings["care"] or None, now=now)
        self.care.listeners.append(self._on_care_event)
        from ..utils.ambient import AmbientPlayer
        self.ambient = AmbientPlayer(app.data_dir)
        self.ambient.set_volume(float(app.settings["ambient_volume"]))
        self.session_goal = ""
        self.away_nap = False
        self._last_hint = 0.0
        self._saved = now()
        self._deep_prev: Optional[dict] = None
        self._sched_done: set = set()
        self._wind_after: Optional[dt.datetime] = None
        prev = app.settings["deep_restore"]
        if isinstance(prev, dict) and prev:             # FocusBuddy closed mid-session: undo Deep focus
            self._deep_prev = prev
            self._deep_off()

    # ------------------------------------------------------------ helpers
    @property
    def life(self):
        return self.app.companion.life

    def _today(self) -> str:
        return self._wall().strftime("%Y-%m-%d")

    def save(self) -> None:
        self.app.settings.set("care", self.care.snapshot())
        self._saved = self._now()

    def _ask(self, text: str, buttons: list, title: str = "") -> None:
        self.app.companion.ask(text, buttons, title=title)

    def _play(self, names) -> None:
        """Queue a few actions on the companion as a short scene (skipped if it's busy elsewhere)."""
        from ..companion.actions import Priority
        from ..companion.life import LifeState
        L_ = self.life
        if self.app.engine.episode is not None:
            return
        steps = []
        for n in names:
            steps.append(("action", n, None, Priority.INTERACTION))
        try:
            L_._run("care", steps, LifeState.INTERACTING)
        except Exception:  # noqa: BLE001 - a reaction is decoration
            log.exception("care scene failed")

    # ------------------------------------------------------------ care
    def care_action(self, action: str) -> dict:
        """Feed / play / pet from the Care page or the companion's menu."""
        if self.life.napping:
            return {"ok": False, "reason": "napping"}
        res = self.care.do(action)
        if not res.get("ok"):
            if res.get("reason") == "no_treats":
                self.app.companion.say("No treats left - ten focused minutes earns one! 🍪", duration=4)
            return res
        flavor = self.life.flavor
        if action == "feed":
            self._play(("EAT",))
            self.app.companion.say("Nom nom… thank you! 🍪", duration=3.5)
        elif action == "play":
            if flavor == "dog":
                try:
                    from ..companion.life import LifeState
                    self.life._run("care", self.app.companion.behavior.fetch_steps(), LifeState.PLAYING)
                except Exception:  # noqa: BLE001
                    self._play(("PLAY",))
            else:
                self._play(PLAY_STEPS.get(flavor, ("PLAY",)))
            self.app.companion.say("Again! Again! 🎾", duration=3)
        elif action == "pet":
            self._play(PET_STEPS.get(flavor, ("LOVE",)))
            self.app.companion.say("Mmm, that's nice 💛", duration=3)
        self.save()
        return res

    def _on_care_event(self, event: str, data: dict) -> None:
        c = self.app.companion
        if event == "level_up":
            msg = f"Bond level {data['level']} - {data['title']}!"
            if data.get("unlock"):
                msg += f" New: {data['unlock'].lower()}."
            c.say(msg, duration=6, title="Level up")
            self._play(("CELEBRATE",))
            self.app.sound.play("celebrate")
        elif event == "badge":
            c.say(f"{data['icon']} New badge: {data['name']}!", duration=5)
            self._play(("HAPPY",))
        if self.app.dashboard is not None:
            try:
                self.app.dashboard.toast(f"{data.get('icon', '⭐')} {data.get('name') or data.get('title')}", "ok")
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------ minute tick
    def tick(self) -> None:
        a = self.app
        self.care.tick(True)
        now = self._now()
        if now - self._saved > 300:
            self.save()
        wall = self._wall()
        free = a.focus.phase.name == "IDLE" and a.engine.episode is None and not self.life.busy()
        present = a._idle_seconds() < 120
        # gentle hints about needs, at most every 2 hours
        line = self.care.need_line()
        if line and free and present and now - self._last_hint > 7200 and a._can_chat(1.0):
            self._last_hint = now
            self._ask(line, [("care_open", "Take care of it", True), ("care_later", "Later", False)])
            return
        today = self._today()
        s = a.settings
        # morning check-in: the first time you're here each day (5 am - noon)
        if (s["morning_checkin"] and s["last_checkin_day"] != today and 5 <= wall.hour < 12 and present
                and free and a._can_chat(1.0) and not self.life.napping):
            s.set("last_checkin_day", today)
            name = s["user_name"] or "friend"
            tasks = a.db.tasks(include_done=False)
            top = f" Your list starts with “{tasks[0]['title']}”." if tasks else ""
            self._ask(f"Good morning, {name}! ☀️ What's the one thing that matters most today?{top}",
                      [("checkin_focus", f"Start {s['focus_minutes']} min", True), ("checkin_tasks", "Plan my day", False),
                       ("checkin_later", "Later", False)], title="Morning check-in")
            return
        # wind-down
        wd = parse_times(s["wind_down_time"])
        if (s["wind_down"] and wd and s["last_wind_down_day"] != today and present and free
                and (wall.hour, wall.minute) >= wd[0] and a._can_chat(1.0)
                and (self._wind_after is None or wall >= self._wind_after)):
            s.set("last_wind_down_day", today)
            mins = int(round(a.focus_today_seconds() / 60))
            done = mins >= 30
            text = (f"It's getting late - you focused {mins // 60}h {mins % 60:02d}m today. "
                    f"{'Great work!' if done else 'Every bit counts.'} Time to wind down? 🌙")
            self._ask(text, [("wind_report", "Show my day", False), ("wind_nap", "Good night 💤", True),
                             ("wind_later", "5 more minutes", False)], title="Wind-down")
            return
        # nap schedule
        dur = int(s["nap_schedule_minutes"])
        for hm in parse_times(s["nap_schedule"]):
            key = f"{today} {hm[0]:02d}:{hm[1]:02d}"
            if key in self._sched_done:
                continue
            start = wall.replace(hour=hm[0], minute=hm[1], second=0, microsecond=0)
            if start <= wall < start + dt.timedelta(minutes=10):
                self._sched_done.add(key)
                if free and not self.life.napping:
                    self.life.nap(announce=a._can_chat(1.0), rest_seconds=dur * 60)

    # ------------------------------------------------------------ presence
    def presence(self, idle: float) -> None:
        """Every 1.5 s: nap while you're away, come back when you're back."""
        s = self.app.settings
        if not s["nap_when_away"]:
            return
        away_after = max(2, int(s["nap_away_minutes"])) * 60
        a = self.app
        if (not self.away_nap and idle > away_after and a.focus.phase.name == "IDLE" and a.engine.episode is None
                and not self.life.napping and not self.life.busy() and a.companion.visible_wanted):
            self.away_nap = True
            self.life.nap(announce=False)
        elif self.away_nap and idle < 3:
            self.away_nap = False
            if self.life.napping:
                self.life.wake_from_nap()

    # ------------------------------------------------------------ focus
    def on_focus_event(self, kind: str, data: dict) -> None:
        a = self.app
        if kind == "focus_started":
            if self.life.napping:
                self.away_nap = False
            self._ambient_update(True)
            if a.settings["deep_focus"]:
                self._deep_on()
        elif kind in ("focus_completed", "focus_cancelled"):
            self._ambient_update(False)
            self._deep_off()
            mins = float(data.get("focused_seconds", 0)) / 60.0
            res = self.care.on_focus(mins)
            if kind == "focus_completed":
                if res["treats"]:
                    a.companion.say(f"+{res['treats']} treat{'s' if res['treats'] > 1 else ''} earned 🍪",
                                    duration=4) if not self.session_goal else None
                streak = a.db.streak_days()
                hit = self.care.on_streak(streak)
                if hit:
                    a.companion.say(f"🔥 {hit}-day streak! Bonus treats for you… well, for me.", duration=6)
                try:
                    self.care.on_totals(a.db.total_focus_seconds() / 3600.0)
                except Exception:  # noqa: BLE001 - older databases
                    pass
                if self.session_goal:
                    goal = self.session_goal
                    from PySide6.QtCore import QTimer
                    QTimer.singleShot(2500, lambda: self._ask(f"Did you finish: “{goal}”?",
                                                              [("goal_yes", "Yes! ✅", True), ("goal_no", "Not yet", False)],
                                                              title="Session goal"))
            self.save()
        elif kind == "break_started":
            self._ambient_update(False)

    def set_goal(self, text: str) -> None:
        self.session_goal = (text or "").strip()[:120]

    def _ambient_update(self, focusing: bool) -> None:
        s = self.app.settings
        name = s["ambient_sound"]
        if name and name != "off" and (focusing or s["ambient_mode"] == "always"):
            self.ambient.play(name)
        else:
            self.ambient.stop()

    def apply_ambient(self) -> None:
        self.ambient.set_volume(float(self.app.settings["ambient_volume"]))
        self._ambient_update(self.app.focus.phase.name in ("FOCUS", "PAUSED"))

    def _deep_on(self) -> None:
        s = self.app.settings
        if self._deep_prev is not None:
            return
        self._deep_prev = {k: s[k] for k in ("quiet_mode", "focus_enforcement", "focus_behavior")}
        s.set("deep_restore", dict(self._deep_prev))
        s.set("quiet_mode", True)
        s.set("focus_behavior", "strict")
        if s["focus_enforcement"] not in ("auto_close",):
            s.set("focus_enforcement", "auto_close")

    def _deep_off(self) -> None:
        if self._deep_prev is None:
            return
        prev, self._deep_prev = self._deep_prev, None
        for k, v in prev.items():
            self.app.settings.set(k, v)
        self.app.settings.set("deep_restore", {})

    # ------------------------------------------------------------ bubble buttons
    def on_bubble(self, action: str) -> bool:
        a = self.app
        if action in ("care_open", "checkin_tasks", "wind_report"):
            a.companion.bubble.hide_now()
            a.show_dashboard({"care_open": "care", "checkin_tasks": "tasks", "wind_report": "reports"}[action])
            return True
        if action == "checkin_focus":
            a.companion.bubble.hide_now()
            a.start_focus(int(a.settings["focus_minutes"]))
            return True
        if action in ("care_later", "checkin_later", "goal_no"):
            a.companion.bubble.hide_now()
            if action == "goal_no":
                a.companion.say("That's okay - you moved it forward. Next session! 💪", duration=4)
            self.session_goal = ""
            return True
        if action == "goal_yes":
            a.companion.bubble.hide_now()
            self.care.on_goal_done()
            self.session_goal = ""
            a.companion.say("Yes!! Goal done - here's a treat for both of us 🍪", duration=4)
            self._play(("CELEBRATE",))
            self.save()
            return True
        if action == "wind_nap":
            a.companion.bubble.hide_now()
            self.life.nap(announce=True)
            return True
        if action == "wind_later":
            a.companion.bubble.hide_now()
            a.settings.set("last_wind_down_day", "")
            self._wind_after = self._wall() + dt.timedelta(minutes=5)
            return True
        return False

    def shutdown(self) -> None:
        self._deep_off()
        self.ambient.stop()
        self.save()

    def badges(self) -> list[tuple[str, str, bool]]:
        """[(icon, name, earned)] for every badge, earned first."""
        got = set(self.care.s.badges)
        rows = [(BADGES[k][0], BADGES[k][1], k in got) for k in BADGES]
        return sorted(rows, key=lambda r: not r[2])
