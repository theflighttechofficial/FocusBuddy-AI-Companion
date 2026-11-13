"""The small facade the chat assistant (and the new pages) use to read and steer the app."""

from __future__ import annotations

import datetime as dt
import time

from .reports import day_report, mins, week_report


class AppApi:
    def __init__(self, app):
        self.app = app

    # identity
    def user_name(self) -> str:
        return self.app.settings["user_name"] or "friend"

    def companion_name(self) -> str:
        from ..companion.characters import companion_name, get_character
        spec = get_character(self.app.settings["character"])
        return companion_name(spec, self.app.settings["companion_name"])

    def personality(self) -> str:
        from ..companion.characters import get_character
        return self.app.settings["personality"] or get_character(self.app.settings["character"]).personality

    def mood(self) -> str:
        return self.app.context.mood

    # focus
    def focus_running(self) -> bool:
        from ..core.focus_manager import Phase
        return self.app.focus.phase in (Phase.FOCUS, Phase.PAUSED)

    def default_minutes(self) -> int:
        return int(self.app.settings["focus_minutes"])

    def start_focus(self, minutes: int, task_id=None) -> None:
        self.app.start_focus(int(minutes), task_id=task_id)

    def end_focus(self) -> None:
        self.app.end_focus()

    def pause_focus(self) -> None:
        self.app.pause_focus()

    def resume_focus(self) -> None:
        self.app.resume_focus()

    def start_break(self, minutes: int) -> None:
        self.app.start_break(int(minutes))

    def set_monitoring(self, on: bool) -> None:
        self.app.set_monitoring(bool(on))

    # tasks
    def open_tasks(self) -> list:
        return self.app.db.tasks(include_done=False)

    def add_task(self, title: str, estimate: int = 25) -> int:
        tid = self.app.db.add_task(title, estimate_minutes=int(estimate))
        self.app.tasks_changed()
        return tid

    def complete_task(self, task_id: int) -> None:
        self.app.complete_task(task_id)

    # knowledge
    def day_report(self) -> dict:
        self.app.stats.flush()
        return day_report(self.app.db, coach=self.app.coach)

    def week_report(self) -> dict:
        self.app.stats.flush()
        return week_report(self.app.db)

    def insights(self) -> list:
        return self.app.coach.insights()

    def top_distractors(self) -> list:
        return self.app.coach.top_distractors(k=3)

    def streak(self) -> int:
        return self.app.db.streak_days()

    def memories_today(self) -> list:
        start = dt.datetime.combine(dt.date.today(), dt.time()).timestamp()
        return list(reversed(self.app.db.memories(start, limit=12)))

    def context_summary(self) -> str:
        try:
            r = self.day_report()
            s = r["summary"]
            parts = [f"focused {mins(s['focus_seconds'])} today, distracted {mins(s['distraction_seconds'])}",
                     f"{s['sessions']} focus sessions done", f"streak {self.streak()} days"]
            ts = self.open_tasks()
            if ts:
                parts.append("open tasks: " + ", ".join(t["title"] for t in ts[:4]))
            parts.append(f"time now {time.strftime('%H:%M')}")
            parts.append(f"your mood: {self.mood().lower()}")
            ins = [i.title for i in self.insights()[:2]]
            if ins:
                parts.append("coach notes: " + "; ".join(ins))
            return "; ".join(parts) + "."
        except Exception:  # noqa: BLE001 - context is a nicety
            return ""
