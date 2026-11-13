"""The focus coach: learns *your* patterns from FocusBuddy's own history and turns them into advice.

It looks at the last few weeks (hours of focused and distracted time, the distraction episodes,
the focus sessions) and finds:

* your **best focus hours** (where focused time is high and distraction low),
* your **risky hours** (when you usually drift) and **what** pulls you away then,
* how your **sessions** go (completion rate, which length you finish most often),
* the **trend** (this week against the one before),
* how fast you **come back** after a distraction,

and from that a **plan for today** and a live **risk** for the current hour, which the companion
uses to offer a focus session *before* the usual slip. All local, plain arithmetic - no model,
nothing sent anywhere.
"""

from __future__ import annotations

import datetime as dt
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

MIN_HOURS_OF_DATA = 3.0          # below this the coach says it's still learning


@dataclass
class Insight:
    kind: str                    # best_hours | risky_hours | top_distractor | trend | sessions | recovery | ...
    title: str
    text: str
    tone: str = "info"           # good | warn | info
    data: dict = field(default_factory=dict)


@dataclass
class Profile:
    days: int
    focus: list                  # [24] average focused seconds per hour of day (per day with data)
    distract: list               # [24]
    events_by_hour: list         # [24] distraction episodes per hour (total)
    hours_tracked: float
    weekday_focus: list          # [7] average focused seconds per weekday
    top_by_hour: dict            # hour -> most common distraction target
    day_count: int = 0


def _fmt_h(h: int) -> str:
    h = h % 24
    suffix = "am" if h < 12 else "pm"
    return f"{(h % 12) or 12}{suffix}"


def fmt_window(a: int, b: int) -> str:
    """Hours a..b (b exclusive) as "9am-11am"."""
    return f"{_fmt_h(a)}–{_fmt_h(b)}"


def _mins(seconds: float) -> str:
    m = int(round(seconds / 60))
    if m < 60:
        return f"{m} min"
    return f"{m // 60}h {m % 60:02d}m"


class FocusCoach:
    def __init__(self, db, now=None):
        self.db = db
        self._now = now or (lambda: dt.datetime.now())
        self._cache: Optional[tuple[float, Profile]] = None

    # ------------------------------------------------------------ data
    def profile(self, days: int = 28, max_age: float = 120.0) -> Profile:
        """Hour-of-day averages over the last ``days`` (cached for ``max_age`` seconds)."""
        import time
        if self._cache is not None and time.monotonic() - self._cache[0] < max_age and self._cache[1].days == days:
            return self._cache[1]
        now = self._now()
        since = (now.date() - dt.timedelta(days=days - 1))
        rows = self.db.hourly(since.strftime("%Y-%m-%d"))
        focus = [0.0] * 24
        distract = [0.0] * 24
        days_seen: set = set()
        wd_focus = [0.0] * 7
        wd_days: dict = defaultdict(set)
        tracked = 0.0
        for r in rows:
            h = int(r["hour"]) % 24
            s = float(r["seconds"])
            tracked += s
            days_seen.add(r["day"])
            if r["category"] == "productive":
                focus[h] += s
                try:
                    wd = dt.datetime.strptime(r["day"], "%Y-%m-%d").weekday()
                    wd_focus[wd] += s
                    wd_days[wd].add(r["day"])
                except ValueError:
                    pass
            elif r["category"] in ("distracting", "blocked"):
                distract[h] += s
        n = max(1, len(days_seen))
        focus = [v / n for v in focus]
        distract = [v / n for v in distract]
        wd_focus = [wd_focus[i] / max(1, len(wd_days[i])) for i in range(7)]
        events = self.db.events_since(dt.datetime.combine(since, dt.time()).timestamp())
        by_hour = [0] * 24
        targets: dict = defaultdict(Counter)
        for e in events:
            h = dt.datetime.fromtimestamp(e["timestamp"]).hour
            by_hour[h] += 1
            t = e.get("website") or e.get("application")
            if t:
                targets[h][t] += 1
        top = {h: c.most_common(1)[0][0] for h, c in targets.items() if c}
        # an install from before 2.0 has no hourly table yet: the episodes still show when you drift
        if tracked <= 0 and events:
            tracked = sum(float(e.get("duration") or 0) for e in events)
            for e in events:
                h = dt.datetime.fromtimestamp(e["timestamp"]).hour
                distract[h] += float(e.get("duration") or 0) / n
        p = Profile(days, focus, distract, by_hour, tracked / 3600.0, wd_focus, top, len(days_seen))
        self._cache = (time.monotonic(), p)
        return p

    # ------------------------------------------------------------ analysis
    def best_window(self, p: Profile, width: int = 2) -> Optional[tuple[int, int, float]]:
        """The ``width``-hour window with the most focused time (and little distraction)."""
        best = None
        for h in range(24):
            hs = [(h + k) % 24 for k in range(width)]
            f = sum(p.focus[x] for x in hs)
            d = sum(p.distract[x] for x in hs)
            if f < 300:                       # at least 5 min of focus there on an average day
                continue
            score = f - 0.8 * d
            if best is None or score > best[2]:
                best = (h, (h + width) % 24, score)
        return best

    def risky_hours(self, p: Profile, k: int = 2) -> list[int]:
        scored = []
        for h in range(24):
            d = p.distract[h] + 120 * p.events_by_hour[h] / max(1, p.day_count or 1)
            if d >= 180:                      # 3 min of drifting there on an average day
                scored.append((d, h))
        scored.sort(reverse=True)
        return [h for _d, h in scored[:k]]

    def risk_now(self, when: Optional[dt.datetime] = None) -> float:
        """0..1: how likely you are to drift in this hour, from your own history."""
        p = self.profile()
        if p.hours_tracked < MIN_HOURS_OF_DATA:
            return 0.0
        h = (when or self._now()).hour
        d = p.distract[h] + 90 * p.events_by_hour[h] / max(1, p.day_count or 1)
        f = p.focus[h]
        total = d + f
        if total < 120:
            return 0.0
        share = d / total
        scale = min(1.0, d / 900.0)           # 15+ min of drifting there = fully confident
        return max(0.0, min(1.0, share * (0.4 + 0.6 * scale)))

    def session_stats(self, days: int = 28) -> dict:
        since = (self._now() - dt.timedelta(days=days)).timestamp()
        ss = [s for s in self.db.sessions_since(since) if s["status"] in ("completed", "cancelled")]
        done = [s for s in ss if s["status"] == "completed"]
        by_len: dict = defaultdict(lambda: [0, 0])
        for s in ss:
            m = int(round(s["planned_seconds"] / 60 / 5) * 5)
            by_len[m][1] += 1
            if s["status"] == "completed":
                by_len[m][0] += 1
        best_len = None
        for m, (ok, n) in by_len.items():
            if n >= 2 and (best_len is None or (ok / n, m) > (by_len[best_len][0] / by_len[best_len][1], best_len)):
                best_len = m
        return {"total": len(ss), "completed": len(done), "rate": (len(done) / len(ss)) if ss else 0.0,
                "avg_distractions": (sum(s["distractions"] for s in done) / len(done)) if done else 0.0,
                "best_length": best_len, "focused_seconds": sum(s["focused_seconds"] for s in done)}

    def trend(self) -> Optional[dict]:
        today = self._now().date()
        rows = self.db.activity_days((today - dt.timedelta(days=13)).strftime("%Y-%m-%d"))
        this_w = last_w = 0.0
        this_d = last_d = 0.0
        for r in rows:
            day = dt.datetime.strptime(r["day"], "%Y-%m-%d").date()
            age = (today - day).days
            s = float(r["seconds"])
            if r["category"] == "productive":
                if age < 7:
                    this_w += s
                else:
                    last_w += s
            elif r["category"] in ("distracting", "blocked"):
                if age < 7:
                    this_d += s
                else:
                    last_d += s
        if last_w + last_d < 1800 or this_w + this_d < 1800:
            return None
        return {"focus_change": (this_w - last_w) / max(1.0, last_w), "distract_change": (this_d - last_d) / max(1.0, last_d),
                "this_focus": this_w, "last_focus": last_w, "this_distract": this_d, "last_distract": last_d}

    def recovery(self, days: int = 14) -> Optional[float]:
        """Median minutes a distraction lasted before you came back."""
        since = (self._now() - dt.timedelta(days=days)).timestamp()
        ds = sorted(float(e["duration"]) for e in self.db.events_since(since) if e.get("duration"))
        if len(ds) < 3:
            return None
        return ds[len(ds) // 2] / 60.0

    def top_distractors(self, days: int = 14, k: int = 3) -> list[tuple[str, float]]:
        since = (self._now().date() - dt.timedelta(days=days - 1)).strftime("%Y-%m-%d")
        return self.db.top_distractions(since, k)

    # ------------------------------------------------------------ output
    def insights(self) -> list[Insight]:
        p = self.profile()
        out: list[Insight] = []
        if p.hours_tracked < MIN_HOURS_OF_DATA:
            left = max(0.5, MIN_HOURS_OF_DATA - p.hours_tracked)
            out.append(Insight("learning", "Still learning your rhythm",
                               f"Use your computer with FocusBuddy running for about {left:.0f} more hour"
                               f"{'s' if left >= 1.5 else ''} and I'll start spotting your best focus hours and the "
                               "moments you usually drift.", "info"))
        bw = self.best_window(p)
        if bw:
            a, b, _ = bw
            out.append(Insight("best_hours", f"Your best focus: {fmt_window(a, b)}",
                               f"You get the most done between {fmt_window(a, b)}. Protect that time for your "
                               "hardest task.", "good", {"start": a, "end": b}))
        risky = self.risky_hours(p)
        if risky:
            h = risky[0]
            what = p.top_by_hour.get(h)
            out.append(Insight("risky_hours", f"Watch out around {_fmt_h(h)}",
                               f"That's when you drift most{f' - usually to {what}' if what else ''}. A short focus "
                               "session started just before it helps.", "warn", {"hours": risky, "target": what}))
        top = self.top_distractors()
        if top:
            name, secs = top[0]
            out.append(Insight("top_distractor", f"Biggest time sink: {name}",
                               f"{_mins(secs)} in the last two weeks. Consider marking it Blocked during focus "
                               "sessions.", "warn", {"target": name, "seconds": secs}))
        tr = self.trend()
        if tr:
            ch = tr["focus_change"]
            if ch >= 0.1:
                out.append(Insight("trend", f"Focus up {ch * 100:.0f}% this week",
                                   f"{_mins(tr['this_focus'])} focused vs {_mins(tr['last_focus'])} last week. "
                                   "Keep it going!", "good", tr))
            elif ch <= -0.1:
                out.append(Insight("trend", f"Focus down {-ch * 100:.0f}% this week",
                                   f"{_mins(tr['this_focus'])} focused vs {_mins(tr['last_focus'])} last week. A "
                                   "couple of short sessions a day turns it around.", "warn", tr))
        ss = self.session_stats()
        if ss["total"] >= 3:
            line = f"You finish {ss['rate'] * 100:.0f}% of your focus sessions"
            if ss["best_length"]:
                line += f"; {ss['best_length']}-minute ones work best for you"
            out.append(Insight("sessions", "Your sessions", line + ".", "good" if ss["rate"] >= 0.7 else "info", ss))
        rec = self.recovery()
        if rec is not None:
            out.append(Insight("recovery", f"You bounce back in ~{rec:.0f} min",
                               "That's the typical length of a detour before you return. Shorter every week is the "
                               "goal.", "good" if rec <= 5 else "info", {"minutes": rec}))
        return out

    def plan_today(self, tasks: Optional[list] = None) -> list[str]:
        """Concrete suggestions for the rest of today."""
        p = self.profile()
        now = self._now()
        out = []
        bw = self.best_window(p)
        if bw and (bw[0] > now.hour or (bw[0] <= now.hour < bw[1])):
            out.append(f"Deep work {fmt_window(bw[0], bw[1])} - your strongest hours.")
        for h in self.risky_hours(p):
            if h >= now.hour:
                out.append(f"Start a focus session at {_fmt_h(h)} - that's when you usually drift.")
                break
        ss = self.session_stats()
        length = ss["best_length"] or 25
        open_tasks = [t for t in (tasks or []) if not t.get("done_at")]
        if open_tasks:
            t = sorted(open_tasks, key=lambda t: (-int(t.get("priority", 1)), t.get("sort", 0)))[0]
            n = max(1, round((t.get("estimate_minutes") or length) / length))
            out.append(f"Next task: “{t['title']}” - about {n} × {length}-minute session{'s' if n > 1 else ''}.")
        else:
            out.append(f"Add today's tasks - I'll plan them into {length}-minute focus sessions.")
        if now.hour >= 22:
            out.append("It's late: wrap up, and let tomorrow-you take the rest.")
        return out

    def nudge(self, threshold: float = 0.55) -> Optional[str]:
        """A pre-emptive line for this hour when it's a risky one, else None."""
        r = self.risk_now()
        if r < threshold:
            return None
        p = self.profile()
        what = p.top_by_hour.get(self._now().hour)
        if what:
            return f"This is usually when {what} sneaks in. Want to start a focus session now?"
        return "This hour is usually a wobbly one for you. Want to start a focus session now?"
