"""Local SQLite storage.  Nothing here ever leaves the machine.

If the database file cannot be opened (permissions, corruption, locked
drive) we fall back to an in-memory database so the app keeps working and
the UI shows a warning (``Database.degraded``).
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable

from ..utils.logging_setup import get_logger

log = get_logger("database")

SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(
    key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS settings(
    key TEXT PRIMARY KEY, value TEXT NOT NULL);

-- One table holds every website/app rule; the views below expose the
-- classic lists (blocked_sites, allowed_sites, blocked_apps, allowed_apps).
CREATE TABLE IF NOT EXISTS rules(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL CHECK(kind IN ('website','app')),
    pattern TEXT NOT NULL,
    category TEXT NOT NULL CHECK(category IN ('productive','neutral','distracting','blocked')),
    auto_close INTEGER NOT NULL DEFAULT 0,
    UNIQUE(kind, pattern));

CREATE TABLE IF NOT EXISTS focus_sessions(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at REAL NOT NULL,
    ended_at REAL,
    planned_seconds INTEGER NOT NULL,
    focused_seconds INTEGER NOT NULL DEFAULT 0,
    distractions INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'running');   -- running|completed|cancelled

CREATE TABLE IF NOT EXISTS distraction_events(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    application TEXT,
    website TEXT,
    duration REAL NOT NULL DEFAULT 0,
    score INTEGER NOT NULL DEFAULT 0,
    level INTEGER NOT NULL DEFAULT 0,
    action TEXT NOT NULL DEFAULT 'none',
    focus_session INTEGER NOT NULL DEFAULT 0);
CREATE INDEX IF NOT EXISTS idx_events_ts ON distraction_events(timestamp);

-- Aggregated seconds per day/category/target (app or domain).
CREATE TABLE IF NOT EXISTS activity_time(
    day TEXT NOT NULL, category TEXT NOT NULL, target TEXT NOT NULL,
    seconds REAL NOT NULL DEFAULT 0,
    PRIMARY KEY(day, category, target));

CREATE TABLE IF NOT EXISTS statistics(
    day TEXT PRIMARY KEY,
    recovered_seconds REAL NOT NULL DEFAULT 0,
    blocked INTEGER NOT NULL DEFAULT 0,
    prevented INTEGER NOT NULL DEFAULT 0);

-- 2.0: seconds per hour of the day (the focus coach's patterns)
CREATE TABLE IF NOT EXISTS activity_hourly(
    day TEXT NOT NULL, hour INTEGER NOT NULL, category TEXT NOT NULL,
    seconds REAL NOT NULL DEFAULT 0,
    PRIMARY KEY(day, hour, category));

-- 2.0: tasks and goals (focus sessions can be tied to a task)
CREATE TABLE IF NOT EXISTS tasks(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    project TEXT NOT NULL DEFAULT '',
    notes TEXT NOT NULL DEFAULT '',
    estimate_minutes INTEGER NOT NULL DEFAULT 25,
    priority INTEGER NOT NULL DEFAULT 1,          -- 0 low, 1 normal, 2 high
    due_day TEXT,
    created_at REAL NOT NULL,
    done_at REAL,
    spent_seconds REAL NOT NULL DEFAULT 0,
    sessions INTEGER NOT NULL DEFAULT 0,
    sort INTEGER NOT NULL DEFAULT 0);

-- 2.0: what the companion remembers about the day (shown in chat / reports, never sent anywhere)
CREATE TABLE IF NOT EXISTS companion_memory(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL, kind TEXT NOT NULL, text TEXT NOT NULL, data TEXT NOT NULL DEFAULT '{}');
CREATE INDEX IF NOT EXISTS idx_memory_ts ON companion_memory(ts);

CREATE TABLE IF NOT EXISTS chat_log(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL, role TEXT NOT NULL, text TEXT NOT NULL);

CREATE VIEW IF NOT EXISTS blocked_sites AS
    SELECT pattern, auto_close FROM rules WHERE kind='website' AND category IN ('distracting','blocked');
CREATE VIEW IF NOT EXISTS allowed_sites AS
    SELECT pattern FROM rules WHERE kind='website' AND category='productive';
CREATE VIEW IF NOT EXISTS blocked_apps AS
    SELECT pattern, auto_close FROM rules WHERE kind='app' AND category IN ('distracting','blocked');
CREATE VIEW IF NOT EXISTS allowed_apps AS
    SELECT pattern FROM rules WHERE kind='app' AND category='productive';
"""

PREVENTED_ACTIONS = ("closed", "minimized", "redirected", "returned", "left")
BLOCKED_ACTIONS = ("closed", "minimized", "redirected")


def day_key(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


class Database:
    def __init__(self, path: Path | str | None):
        self.path = str(path) if path else ":memory:"
        self.degraded = False
        self._lock = threading.RLock()
        try:
            self.conn = self._open(self.path)
        except (sqlite3.Error, OSError) as exc:
            log.error("Database unavailable, using in-memory fallback err=%s", exc)
            self.degraded = True
            self.path = ":memory:"
            self.conn = self._open(":memory:")

    # ------------------------------------------------------------------ setup
    def _open(self, path: str) -> sqlite3.Connection:
        conn = sqlite3.connect(path, check_same_thread=False, timeout=5)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL" if path != ":memory:" else "PRAGMA journal_mode=MEMORY")
        conn.executescript(SCHEMA)
        # 2.0: focus sessions can belong to a task (an older database gains the column in place)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(focus_sessions)").fetchall()}
        if "task_id" not in cols:
            conn.execute("ALTER TABLE focus_sessions ADD COLUMN task_id INTEGER")
        conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('schema_version',?)", (str(SCHEMA_VERSION),))
        conn.commit()
        return conn

    def _exec(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            try:
                cur = self.conn.execute(sql, tuple(params))
                self.conn.commit()
                return cur
            except sqlite3.Error as exc:
                log.error("DB error err=%s", exc)
                raise

    def _query(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            try:
                return self.conn.execute(sql, tuple(params)).fetchall()
            except sqlite3.Error as exc:
                log.error("DB query error err=%s", exc)
                return []

    def close(self) -> None:
        with self._lock:
            try:
                self.conn.close()
            except sqlite3.Error:
                pass

    # --------------------------------------------------------------- settings
    def load_settings(self) -> dict:
        out = {}
        for row in self._query("SELECT key, value FROM settings"):
            try:
                out[row["key"]] = json.loads(row["value"])
            except ValueError:
                continue
        return out

    def save_setting(self, key: str, value: Any) -> None:
        try:
            self._exec("INSERT OR REPLACE INTO settings(key,value) VALUES(?,?)", (key, json.dumps(value)))
        except sqlite3.Error:
            pass

    # ------------------------------------------------------------------ rules
    def rules(self, kind: str | None = None) -> list[dict]:
        if kind:
            rows = self._query("SELECT * FROM rules WHERE kind=? ORDER BY pattern", (kind,))
        else:
            rows = self._query("SELECT * FROM rules ORDER BY kind, pattern")
        return [dict(r) | {"auto_close": bool(r["auto_close"])} for r in rows]

    def upsert_rule(self, kind: str, pattern: str, category: str, auto_close: bool = False) -> None:
        self._exec(
            "INSERT INTO rules(kind,pattern,category,auto_close) VALUES(?,?,?,?) "
            "ON CONFLICT(kind,pattern) DO UPDATE SET category=excluded.category, auto_close=excluded.auto_close",
            (kind, pattern.strip().lower(), category, int(bool(auto_close))))

    def delete_rule(self, kind: str, pattern: str) -> None:
        self._exec("DELETE FROM rules WHERE kind=? AND pattern=?", (kind, pattern))

    def replace_rules(self, rules: Iterable[dict]) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM rules")
            for r in rules:
                self.conn.execute(
                    "INSERT OR REPLACE INTO rules(kind,pattern,category,auto_close) VALUES(?,?,?,?)",
                    (r["kind"], r["pattern"].lower(), r["category"], int(bool(r.get("auto_close")))))
            self.conn.commit()

    def rule_count(self) -> int:
        rows = self._query("SELECT COUNT(*) AS n FROM rules")
        return rows[0]["n"] if rows else 0

    # ---------------------------------------------------------- focus sessions
    def start_session(self, started_at: float, planned_seconds: int) -> int:
        try:
            cur = self._exec("INSERT INTO focus_sessions(started_at, planned_seconds) VALUES(?,?)",
                             (started_at, planned_seconds))
            return int(cur.lastrowid)
        except sqlite3.Error:
            return -1

    def finish_session(self, session_id: int, ended_at: float, focused_seconds: int,
                       distractions: int, status: str) -> None:
        if session_id < 0:
            return
        try:
            self._exec("UPDATE focus_sessions SET ended_at=?, focused_seconds=?, distractions=?, status=? WHERE id=?",
                       (ended_at, int(focused_seconds), distractions, status, session_id))
        except sqlite3.Error:
            pass

    def close_dangling_sessions(self, now: float) -> None:
        """Sessions left 'running' by a crash/forced shutdown become cancelled."""
        try:
            self._exec("UPDATE focus_sessions SET status='cancelled', ended_at=COALESCE(ended_at, ?) "
                       "WHERE status='running'", (now,))
        except sqlite3.Error:
            pass

    # ------------------------------------------------------ distraction events
    def add_distraction_event(self, *, timestamp: float, application: str | None, website: str | None,
                              duration: float, score: int, level: int, action: str,
                              focus_session: bool) -> None:
        try:
            self._exec("INSERT INTO distraction_events(timestamp,application,website,duration,score,level,action,focus_session) "
                       "VALUES(?,?,?,?,?,?,?,?)",
                       (timestamp, application, website, round(duration, 1), int(score), int(level), action,
                        int(focus_session)))
        except sqlite3.Error:
            pass

    def history(self, since: float | None = None, search: str = "", limit: int = 500) -> list[dict]:
        sql = "SELECT * FROM distraction_events WHERE 1=1"
        params: list[Any] = []
        if since is not None:
            sql += " AND timestamp >= ?"
            params.append(since)
        if search:
            sql += " AND (COALESCE(application,'') LIKE ? OR COALESCE(website,'') LIKE ? OR action LIKE ?)"
            params += [f"%{search}%"] * 3
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        return [dict(r) for r in self._query(sql, params)]

    def clear_history(self) -> None:
        with self._lock:
            for t in ("distraction_events", "activity_time", "statistics", "focus_sessions", "activity_hourly",
                      "companion_memory", "chat_log"):
                self.conn.execute(f"DELETE FROM {t}")
            self.conn.commit()

    def export_history_csv(self, path: Path) -> int:
        rows = self.history(limit=1_000_000)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["time", "application", "website", "duration_s", "score", "level", "action", "focus_session"])
            for r in rows:
                w.writerow([dt.datetime.fromtimestamp(r["timestamp"]).isoformat(timespec="seconds"),
                            r["application"] or "", r["website"] or "", r["duration"], r["score"],
                            r["level"], r["action"], r["focus_session"]])
        return len(rows)

    # ---------------------------------------------------------- activity time
    def add_activity_time(self, items: dict[tuple[str, str, str], float]) -> None:
        """items: {(day, category, target): seconds}"""
        if not items:
            return
        with self._lock:
            try:
                for (day, cat, target), secs in items.items():
                    self.conn.execute(
                        "INSERT INTO activity_time(day,category,target,seconds) VALUES(?,?,?,?) "
                        "ON CONFLICT(day,category,target) DO UPDATE SET seconds = seconds + excluded.seconds",
                        (day, cat, target, secs))
                self.conn.commit()
            except sqlite3.Error as exc:
                log.error("activity flush failed err=%s", exc)

    def add_activity_hourly(self, items: dict[tuple[str, int, str], float]) -> None:
        """items: {(day, hour, category): seconds}"""
        if not items:
            return
        with self._lock:
            try:
                for (day, hour, cat), secs in items.items():
                    self.conn.execute(
                        "INSERT INTO activity_hourly(day,hour,category,seconds) VALUES(?,?,?,?) "
                        "ON CONFLICT(day,hour,category) DO UPDATE SET seconds = seconds + excluded.seconds",
                        (day, int(hour), cat, secs))
                self.conn.commit()
            except sqlite3.Error as exc:
                log.error("hourly flush failed err=%s", exc)

    def hourly(self, since_day: str) -> list[dict]:
        return [dict(r) for r in self._query(
            "SELECT day, hour, category, seconds FROM activity_hourly WHERE day>=? ORDER BY day, hour", (since_day,))]

    def events_since(self, since: float) -> list[dict]:
        return [dict(r) for r in self._query(
            "SELECT * FROM distraction_events WHERE timestamp>=? ORDER BY timestamp", (since,))]

    def sessions_since(self, since: float) -> list[dict]:
        return [dict(r) for r in self._query(
            "SELECT * FROM focus_sessions WHERE started_at>=? ORDER BY started_at", (since,))]

    def activity_days(self, since_day: str) -> list[dict]:
        return [dict(r) for r in self._query(
            "SELECT day, category, target, seconds FROM activity_time WHERE day>=?", (since_day,))]

    # ------------------------------------------------------------------ tasks
    def add_task(self, title: str, *, project: str = "", notes: str = "", estimate_minutes: int = 25,
                 priority: int = 1, due_day: str | None = None, now: float | None = None) -> int:
        import time as _t
        rows = self._query("SELECT COALESCE(MAX(sort),0)+1 AS n FROM tasks")
        cur = self._exec("INSERT INTO tasks(title,project,notes,estimate_minutes,priority,due_day,created_at,sort) "
                         "VALUES(?,?,?,?,?,?,?,?)",
                         (title.strip(), project.strip(), notes, int(estimate_minutes), int(priority), due_day,
                          now if now is not None else _t.time(), rows[0]["n"] if rows else 0))
        return int(cur.lastrowid)

    def tasks(self, include_done: bool = True, done_since: float | None = None) -> list[dict]:
        sql = "SELECT * FROM tasks"
        params: list[Any] = []
        if not include_done:
            sql += " WHERE done_at IS NULL"
        elif done_since is not None:
            sql += " WHERE done_at IS NULL OR done_at>=?"
            params.append(done_since)
        sql += " ORDER BY (done_at IS NOT NULL), priority DESC, sort, id"
        return [dict(r) for r in self._query(sql, params)]

    def task(self, task_id: int) -> dict | None:
        rows = self._query("SELECT * FROM tasks WHERE id=?", (task_id,))
        return dict(rows[0]) if rows else None

    def update_task(self, task_id: int, **fields) -> None:
        allowed = {"title", "project", "notes", "estimate_minutes", "priority", "due_day", "done_at", "sort"}
        sets = [(k, v) for k, v in fields.items() if k in allowed]
        if not sets:
            return
        self._exec(f"UPDATE tasks SET {', '.join(k + '=?' for k, _ in sets)} WHERE id=?",
                   [v for _, v in sets] + [task_id])

    def add_task_time(self, task_id: int, seconds: float, session: bool = True) -> None:
        self._exec("UPDATE tasks SET spent_seconds = spent_seconds + ?, sessions = sessions + ? WHERE id=?",
                   (max(0.0, float(seconds)), 1 if session else 0, task_id))

    def delete_task(self, task_id: int) -> None:
        self._exec("DELETE FROM tasks WHERE id=?", (task_id,))

    def set_session_task(self, session_id: int, task_id: int | None) -> None:
        if session_id >= 0:
            self._exec("UPDATE focus_sessions SET task_id=? WHERE id=?", (task_id, session_id))

    # ---------------------------------------------------------- companion memory
    def remember(self, kind: str, text: str, data: dict | None = None, ts: float | None = None) -> None:
        import time as _t
        try:
            self._exec("INSERT INTO companion_memory(ts,kind,text,data) VALUES(?,?,?,?)",
                       (ts if ts is not None else _t.time(), kind, text, json.dumps(data or {})))
        except sqlite3.Error:
            pass

    def memories(self, since: float, kinds: Iterable[str] | None = None, limit: int = 200) -> list[dict]:
        sql = "SELECT * FROM companion_memory WHERE ts>=?"
        params: list[Any] = [since]
        kinds = list(kinds or [])
        if kinds:
            sql += f" AND kind IN ({','.join('?' * len(kinds))})"
            params += kinds
        sql += " ORDER BY ts DESC LIMIT ?"
        params.append(limit)
        out = []
        for r in self._query(sql, params):
            d = dict(r)
            try:
                d["data"] = json.loads(d.get("data") or "{}")
            except ValueError:
                d["data"] = {}
            out.append(d)
        return out

    def log_chat(self, role: str, text: str, ts: float | None = None) -> None:
        import time as _t
        try:
            self._exec("INSERT INTO chat_log(ts,role,text) VALUES(?,?,?)", (ts if ts is not None else _t.time(), role,
                                                                            text[:4000]))
        except sqlite3.Error:
            pass

    def chat_history(self, limit: int = 60) -> list[dict]:
        rows = self._query("SELECT * FROM chat_log ORDER BY id DESC LIMIT ?", (limit,))
        return [dict(r) for r in reversed(rows)]

    def clear_chat(self) -> None:
        self._exec("DELETE FROM chat_log")

    def bump_stats(self, day: str, recovered_seconds: float = 0, blocked: int = 0, prevented: int = 0) -> None:
        try:
            self._exec("INSERT INTO statistics(day,recovered_seconds,blocked,prevented) VALUES(?,?,?,?) "
                       "ON CONFLICT(day) DO UPDATE SET recovered_seconds=recovered_seconds+excluded.recovered_seconds, "
                       "blocked=blocked+excluded.blocked, prevented=prevented+excluded.prevented",
                       (day, recovered_seconds, blocked, prevented))
        except sqlite3.Error:
            pass

    # ------------------------------------------------------------ statistics
    def day_summary(self, day: str) -> dict:
        def secs(cat_sql: str) -> float:
            rows = self._query(f"SELECT COALESCE(SUM(seconds),0) AS s FROM activity_time WHERE day=? AND category {cat_sql}", (day,))
            return float(rows[0]["s"]) if rows else 0.0

        start = dt.datetime.strptime(day, "%Y-%m-%d").timestamp()
        end = start + 86400
        ev = self._query("SELECT COUNT(*) AS n, "
                         "SUM(CASE WHEN action IN ('closed','minimized','redirected') THEN 1 ELSE 0 END) AS blocked "
                         "FROM distraction_events WHERE timestamp>=? AND timestamp<?", (start, end))
        st = self._query("SELECT * FROM statistics WHERE day=?", (day,))
        sess = self._query("SELECT COUNT(*) AS n, COALESCE(MAX(focused_seconds),0) AS longest, "
                           "COALESCE(SUM(focused_seconds),0) AS total FROM focus_sessions "
                           "WHERE started_at>=? AND started_at<? AND status='completed'", (start, end))
        return {
            "day": day,
            "focus_seconds": secs("= 'productive'"),
            "distraction_seconds": secs("IN ('distracting','blocked')"),
            "neutral_seconds": secs("= 'neutral'"),
            "distractions": int(ev[0]["n"] or 0) if ev else 0,
            "blocked": int(ev[0]["blocked"] or 0) if ev else 0,
            "prevented": int(st[0]["prevented"]) if st else 0,
            "recovered_seconds": float(st[0]["recovered_seconds"]) if st else 0.0,
            "sessions": int(sess[0]["n"]) if sess else 0,
            "session_seconds": float(sess[0]["total"]) if sess else 0.0,
            "longest_session_seconds": float(sess[0]["longest"]) if sess else 0.0,
        }

    def daily_series(self, days: int, today: dt.date | None = None) -> list[dict]:
        today = today or dt.date.today()
        out = []
        for i in range(days - 1, -1, -1):
            d = (today - dt.timedelta(days=i)).strftime("%Y-%m-%d")
            out.append(self.day_summary(d))
        return out

    def top_distractions(self, since_day: str, limit: int = 6) -> list[tuple[str, float]]:
        rows = self._query("SELECT target, SUM(seconds) AS s FROM activity_time WHERE day>=? AND "
                           "category IN ('distracting','blocked') GROUP BY target ORDER BY s DESC LIMIT ?",
                           (since_day, limit))
        return [(r["target"], float(r["s"])) for r in rows]

    def longest_session(self) -> float:
        rows = self._query("SELECT COALESCE(MAX(focused_seconds),0) AS m FROM focus_sessions WHERE status='completed'")
        return float(rows[0]["m"]) if rows else 0.0

    def total_focus_seconds(self) -> float:
        """All focused time ever (completed and cancelled sessions)."""
        rows = self._query("SELECT COALESCE(SUM(focused_seconds),0) AS s FROM focus_sessions WHERE status!='running'")
        return float(rows[0]["s"]) if rows else 0.0

    def streak_days(self, today: dt.date | None = None) -> int:
        """Consecutive days (ending today or yesterday) with >=1 completed session."""
        today = today or dt.date.today()
        rows = self._query("SELECT DISTINCT started_at FROM focus_sessions WHERE status='completed'")
        days = {dt.date.fromtimestamp(r["started_at"]) for r in rows}
        if not days:
            return 0
        cur = today if today in days else today - dt.timedelta(days=1)
        n = 0
        while cur in days:
            n += 1
            cur -= dt.timedelta(days=1)
        return n
