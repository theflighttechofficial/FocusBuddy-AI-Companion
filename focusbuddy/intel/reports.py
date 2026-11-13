"""Daily and weekly reports: what you did, what helped, what to try next."""

from __future__ import annotations

import datetime as dt
from typing import Optional


def mins(seconds: float) -> str:
    m = int(round((seconds or 0) / 60))
    if m < 60:
        return f"{m}m"
    return f"{m // 60}h {m % 60:02d}m"


def day_report(db, day: Optional[dt.date] = None, coach=None) -> dict:
    day = day or dt.date.today()
    key = day.strftime("%Y-%m-%d")
    s = db.day_summary(key)
    start = dt.datetime.combine(day, dt.time()).timestamp()
    end = start + 86400
    done = [t for t in db.tasks(include_done=True) if t.get("done_at") and start <= t["done_at"] < end]
    sessions = [x for x in db.sessions_since(start) if x["started_at"] < end]
    by_task: dict = {}
    for x in sessions:
        if x.get("task_id") and x["status"] == "completed":
            by_task[x["task_id"]] = by_task.get(x["task_id"], 0) + x["focused_seconds"]
    tasks = {t["id"]: t for t in db.tasks(include_done=True)}
    moments = [m for m in db.memories(start) if m["ts"] < end]
    tops = db.top_distractions(key, 3)
    tops = [(t, v) for t, v in tops]  # today and later only (top_distractions is >= day)
    total = s["focus_seconds"] + s["distraction_seconds"]
    score = int(round(100 * s["focus_seconds"] / total)) if total > 0 else 0
    lines = []
    if total <= 0:
        lines.append("No tracked activity yet today.")
    else:
        lines.append(f"Focused {mins(s['focus_seconds'])}, distracted {mins(s['distraction_seconds'])} - "
                     f"a focus score of {score}%.")
    if s["sessions"]:
        lines.append(f"{s['sessions']} focus session{'s' if s['sessions'] != 1 else ''} completed "
                     f"(longest {mins(s['longest_session_seconds'])}).")
    if done:
        lines.append("Done: " + ", ".join(f"“{t['title']}”" for t in done[:6]) + ("…" if len(done) > 6 else "") + ".")
    if by_task:
        parts = [f"{tasks[k]['title']} {mins(v)}" for k, v in sorted(by_task.items(), key=lambda kv: -kv[1])
                 if k in tasks][:4]
        if parts:
            lines.append("Time on tasks: " + ", ".join(parts) + ".")
    if tops:
        lines.append("Most distracting: " + ", ".join(f"{t} {mins(v)}" for t, v in tops) + ".")
    if s["prevented"]:
        lines.append(f"Your companion helped you out of {s['prevented']} distraction"
                     f"{'s' if s['prevented'] != 1 else ''}.")
    tip = None
    if coach is not None:
        try:
            plan = coach.plan_today(db.tasks(include_done=False))
            tip = plan[0] if plan else None
        except Exception:  # noqa: BLE001 - advice is optional
            tip = None
    return {"day": key, "summary": s, "score": score, "done": done, "moments": moments, "top": tops,
            "by_task": by_task, "lines": lines, "tip": tip}


def week_report(db, end: Optional[dt.date] = None) -> dict:
    end = end or dt.date.today()
    days = db.daily_series(7, end)
    focus = sum(d["focus_seconds"] for d in days)
    dist = sum(d["distraction_seconds"] for d in days)
    sessions = sum(d["sessions"] for d in days)
    best = max(days, key=lambda d: d["focus_seconds"]) if days else None
    start_ts = dt.datetime.combine(end - dt.timedelta(days=6), dt.time()).timestamp()
    done = [t for t in db.tasks(include_done=True) if t.get("done_at") and t["done_at"] >= start_ts]
    tops = db.top_distractions((end - dt.timedelta(days=6)).strftime("%Y-%m-%d"), 5)
    lines = [f"Focused {mins(focus)} this week, distracted {mins(dist)}.",
             f"{sessions} focus session{'s' if sessions != 1 else ''}, {len(done)} task{'s' if len(done) != 1 else ''} done."]
    if best and best["focus_seconds"] > 0:
        d = dt.datetime.strptime(best["day"], "%Y-%m-%d").strftime("%A")
        lines.append(f"Best day: {d} ({mins(best['focus_seconds'])}).")
    if tops:
        lines.append("Top distractions: " + ", ".join(f"{t} {mins(v)}" for t, v in tops[:3]) + ".")
    return {"days": days, "focus": focus, "distract": dist, "sessions": sessions, "done": done, "top": tops,
            "lines": lines, "streak": db.streak_days(end)}
