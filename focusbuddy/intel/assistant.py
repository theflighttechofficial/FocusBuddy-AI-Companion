"""Talk to your companion.

Two layers, both on this computer:

1. **Built-in understanding** (always on): the companion knows its own app. "start a 45 minute
   focus on the essay", "add task: read chapter 3", "what's on my list?", "done with the essay",
   "how did I do today?", "when do I focus best?", "what distracts me?", "pause monitoring",
   "I can't focus", "tell me a joke" ... are understood and *done*, not just answered.
2. **A local language model** (optional): if Ollama (ollama.com) is running on this PC, free
   conversation goes to it - with a short summary of your day as context, in the companion's
   voice. No Ollama, no problem: the built-in replies cover it. Nothing is sent over the
   internet either way (the model runs on localhost).
"""

from __future__ import annotations

import json
import random
import re
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Callable, Optional

from .reports import mins

OLLAMA_URL = "http://127.0.0.1:11434"
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


def is_local_url(url: str) -> bool:
    """True only for http(s) URLs whose host is this computer (127.0.0.1, localhost, ::1).
    "http://localhost.example.com" or "http://127.0.0.1@evil.example" are not."""
    from urllib.parse import urlsplit
    try:
        u = urlsplit(url or "")
        return u.scheme in ("http", "https") and (u.hostname or "") in LOCAL_HOSTS and not u.username
    except ValueError:
        return False


@dataclass
class Reply:
    text: str
    source: str = "local"                    # local | ollama
    intent: str = ""
    actions: list = field(default_factory=list)  # what was done, e.g. [("focus", 25)]
    emotion: str = "HAPPY"


# ------------------------------------------------------------------ local model (Ollama)
class LocalLLM:
    """Minimal Ollama client (localhost only). ``available()`` is cached for a minute."""

    def __init__(self, url: str = OLLAMA_URL, model: str = "", timeout: float = 25.0):
        self.url = (url or OLLAMA_URL).rstrip("/")
        self.model = model
        self.timeout = timeout
        self._avail: Optional[tuple[float, list]] = None
        self._lock = threading.Lock()

    def models(self) -> list:
        import time
        with self._lock:
            if self._avail is not None and time.monotonic() - self._avail[0] < 60:
                return self._avail[1]
        names: list = []
        if not is_local_url(self.url):
            return names                     # local only: never talk to another machine
        try:
            with urllib.request.urlopen(self.url + "/api/tags", timeout=0.8) as r:
                data = json.loads(r.read().decode("utf-8"))
            names = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
        except (OSError, ValueError, urllib.error.URLError):
            names = []
        with self._lock:
            self._avail = (time.monotonic(), names)
        return names

    def available(self) -> bool:
        return bool(self.models())

    def configure(self, url: str, model: str) -> None:
        """New address / model from Settings (forgets the cached model list)."""
        self.url = (url or OLLAMA_URL).strip().rstrip("/")
        self.model = (model or "").strip()
        with self._lock:
            self._avail = None

    def pick_model(self) -> str:
        ms = self.models()
        if self.model and self.model in ms:
            return self.model
        for pref in ("qwen", "llama3", "gemma", "phi", "mistral"):
            for m in ms:
                if m.lower().startswith(pref):
                    return m
        return ms[0] if ms else ""

    def chat(self, messages: list[dict]) -> Optional[str]:
        if not is_local_url(self.url):
            return None
        model = self.pick_model()
        if not model:
            return None
        body = json.dumps({"model": model, "messages": messages, "stream": False,
                           "options": {"temperature": 0.7, "num_predict": 160}}).encode("utf-8")
        req = urllib.request.Request(self.url + "/api/chat", data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
            text = (data.get("message") or {}).get("content", "").strip()
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()     # reasoning models
            return text or None
        except (OSError, ValueError, urllib.error.URLError):
            return None


# ------------------------------------------------------------------ the assistant
JOKES = ("Why did the developer go broke? Because they used up all their cache.",
         "I told my computer I needed a break. It gave me a KitKat error.",
         "Why don't programmers like nature? Too many bugs.",
         "My focus is like a browser - 37 tabs open and one of them is playing music.",
         "Parallel lines have so much in common. It's a shame they'll never meet.",
         "I'd tell you a UDP joke, but you might not get it.")
MOTIVATION = ("Tiny steps count. Just the next five minutes - I'll keep watch.",
              "You don't have to feel ready. Start ugly, fix it later.",
              "Future-you is going to be so grateful for this session.",
              "One task, one tab, one timer. You've got this.",
              "Done is better than perfect. Let's get it done.")
NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "ten": 10, "fifteen": 15, "twenty": 20,
             "thirty": 30, "forty": 40, "forty-five": 45, "fifty": 50, "sixty": 60, "ninety": 90, "half": 30,
             "an": 60, "a": 60}


class Assistant:
    """``api`` is the app facade (see ``AppApi`` in app.py): stats, tasks, focus, monitoring, coach."""

    def __init__(self, api, llm: Optional[LocalLLM] = None, rng: Optional[random.Random] = None):
        self.api = api
        self.llm = llm
        self.rng = rng or random.Random()
        self.history: list[dict] = []             # recent turns for the language model

    # ------------------------------------------------------------ entry
    def reply(self, text: str, allow_llm: bool = True) -> Reply:
        t = (text or "").strip()
        if not t:
            return Reply("I'm listening!", intent="empty")
        low = t.lower().strip(" .!?")
        for intent in (self._focus, self._tasks, self._stats, self._monitoring, self._feelings, self._smalltalk):
            r = intent(low, t)
            if r is not None:
                self._remember(t, r.text)
                return r
        if allow_llm and self.llm is not None and self.llm.available():
            out = self.llm.chat(self._llm_messages(t))
            if out:
                self._remember(t, out)
                return Reply(out, source="ollama", intent="chat")
        r = Reply(self.rng.choice((
            "Hmm, I'm not sure about that one. I can start a focus session, keep your task list, tell you how "
            "your day went or when you focus best - just ask!",
            "I didn't quite get that. Try “start a 25 minute focus”, “add task …”, “how did I do today?” or "
            "“what distracts me?”.")), intent="unknown", emotion="CONFUSED")
        self._remember(t, r.text)
        return r

    def _remember(self, user: str, bot: str) -> None:
        self.history += [{"role": "user", "content": user}, {"role": "assistant", "content": bot}]
        self.history = self.history[-12:]

    def _llm_messages(self, text: str) -> list[dict]:
        a = self.api
        ctx = a.context_summary() if hasattr(a, "context_summary") else ""
        system = (f"You are {a.companion_name()}, a small, warm, playful desktop companion ({a.personality()}) in "
                  f"the FocusBuddy app, talking with {a.user_name()}. You help them focus kindly, never bossy. "
                  "Answer in 1-3 short sentences, casual and friendly, no lists, no markdown. You can't open "
                  "websites or do things outside the app; for app actions suggest phrases like 'start a 25 minute "
                  "focus' or 'add task ...'.\n"
                  f"What you know about today: {ctx}")
        return [{"role": "system", "content": system}] + self.history[-8:] + [{"role": "user", "content": text}]

    # ------------------------------------------------------------ intents
    @staticmethod
    def _minutes(low: str) -> Optional[int]:
        """Minutes mentioned in a phrase: "45 min", "1.5 hours", "an hour", "half an hour", "twenty minutes"."""
        m = re.search(r"(\d+(?:\.\d+)?)\s*(h|hr|hrs|hour|hours|m|min|mins|minute|minutes)\b", low)
        if m:
            v = float(m.group(1))
            return int(round(v * 60 if m.group(2).startswith("h") else v))
        if re.search(r"\bhalf an? hour\b", low):
            return 30
        if re.search(r"\b(an|a|one) hour\b", low):
            return 60
        words = {k: v for k, v in NUM_WORDS.items() if k not in ("an", "a", "half")}
        m = re.search(r"\b(" + "|".join(sorted(words, key=len, reverse=True)) + r")[\s-]+(hours?|minutes?|mins?)\b", low)
        if m:
            v = words[m.group(1)]
            return v * 60 if m.group(2).startswith("hour") else v
        m = re.search(r"\b(\d{1,3})\b", low)
        return int(m.group(1)) if m and 5 <= int(m.group(1)) <= 180 else None

    def _match_task(self, name: str) -> Optional[dict]:
        name = name.strip(" .!?\"'“”").lower()
        if not name:
            return None
        best, score = None, 0.0
        for t in self.api.open_tasks():
            title = t["title"].lower()
            s = 1.0 if name == title else 0.9 if name in title or title in name else \
                SequenceMatcher(None, name, title).ratio()
            if s > score:
                best, score = t, s
        return best if score >= 0.55 else None

    def _focus(self, low: str, raw: str) -> Optional[Reply]:
        a = self.api
        if re.search(r"\b(stop|end|cancel|quit|finish)\b.*\b(focus|session|timer|pomodoro)\b", low) or \
                low in ("stop", "end session", "stop focus"):
            if not a.focus_running():
                return Reply("There's no focus session running right now.", intent="focus_stop")
            a.end_focus()
            return Reply("Okay, I've ended the session. Take a breath - you can start again whenever you like.",
                         intent="focus_stop", actions=[("focus_stop",)], emotion="CALM")
        if re.search(r"\bpause\b.*\b(focus|session|timer)\b|^pause$", low):
            if not a.focus_running():
                return Reply("Nothing to pause - no session is running.", intent="focus_pause")
            a.pause_focus()
            return Reply("Paused. Say “resume” when you're back.", intent="focus_pause", actions=[("focus_pause",)])
        if re.search(r"^(resume|continue|unpause)\b", low):
            a.resume_focus()
            return Reply("And we're back! Let's go.", intent="focus_resume", actions=[("focus_resume",)])
        if re.search(r"\b(take|start|need)\b.*\bbreak\b", low):
            m = self._minutes(low) or 5
            a.start_break(m)
            return Reply(f"Break time - {m} minutes. Stand up, stretch, drink some water!", intent="break",
                         actions=[("break", m)], emotion="RELIEVED")
        if re.search(r"\b(focus|pomodoro|session|deep work|study|let'?s (work|go|do this))\b", low) and \
                (re.search(r"\b(start|begin|let'?s|lets|go|want|need|do|time for|i'?ll|a)\b", low) or
                 re.match(r"^(focus|pomodoro|session)\b", low)) and \
                not re.search(r"\b(how|when|what|did|best)\b", low):
            m = self._minutes(low) or a.default_minutes()
            m = max(5, min(180, m))
            task = None
            tm = re.search(r"\b(?:on|for)\s+(?:the\s+|my\s+)?(?!\d)(.+)$", raw.strip(" .!?"), flags=re.I)
            if tm and not re.match(r"\d", tm.group(1)):
                cand = re.sub(r"\b(\d+\s*(min|mins|minutes|hour|hours|h|m))\b", "", tm.group(1), flags=re.I).strip()
                task = self._match_task(cand) if cand else None
            a.start_focus(m, task["id"] if task else None)
            what = f" on “{task['title']}”" if task else ""
            return Reply(f"Starting a {m}-minute focus{what}. I'll guard the door - you do the magic! 💪",
                         intent="focus_start", actions=[("focus", m, task["id"] if task else None)],
                         emotion="DETERMINED")
        return None

    def _tasks(self, low: str, raw: str) -> Optional[Reply]:
        a = self.api
        m = re.match(r"^(?:please\s+)?(?:add|new|create)\s+(?:a\s+)?(?:task|todo|to-do)(?:\s*[:\-]\s*|\s+to\s+|\s+)(.+)$",
                     raw.strip(), flags=re.I) or \
            re.match(r"^(?:todo|to-do|task)\s*[:\-]\s*(.+)$", raw.strip(), flags=re.I) or \
            re.match(r"^remind me to\s+(.+)$", raw.strip(), flags=re.I)
        if m:
            title = m.group(1).strip(" .!")
            est = self._minutes(title.lower())
            title = re.sub(r"\s*\(?\b(?:for\s+)?\d+\s*(?:min|mins|minutes|h|hour|hours)\b\)?", "", title, flags=re.I)
            title = title.strip(" .,-") or "New task"
            title = title[0].upper() + title[1:]
            tid = a.add_task(title, est or 25)
            return Reply(f"Added “{title}” to your list.{' (~' + str(est) + ' min)' if est else ''} "
                         "Say “start focus on " + title.lower() + "” when you're ready.",
                         intent="task_add", actions=[("task_add", tid)], emotion="HAPPY")
        m = re.match(r"^(?:i(?:'ve| have)?\s+)?(?:done|finished|completed|did)\s+(?:with\s+|the\s+|my\s+)*(.+)$", low) or \
            re.match(r"^(?:mark|tick)\s+(.+?)\s+(?:as\s+)?(?:done|complete|finished)$", low)
        if m and not re.search(r"\b(today|for the day|for now)\b$", low):
            t = self._match_task(m.group(1))
            if t is not None:
                a.complete_task(t["id"])
                return Reply(f"Ticked off “{t['title']}”! 🎉 That's what I call progress.", intent="task_done",
                             actions=[("task_done", t["id"])], emotion="EXCITED")
            if re.search(r"\btask\b", low):
                return Reply("I couldn't find that task on your list. Say “what's on my list?” to see it.",
                             intent="task_done")
        if re.search(r"\b(my tasks|task list|to-?do list|my list|what'?s on my (list|plate)|what (should|do) i (do|work on)|"
                     r"what'?s next|next task)\b", low):
            ts = a.open_tasks()
            if not ts:
                return Reply("Your list is empty! Add one with “add task …”.", intent="task_list")
            top = ts[:5]
            names = "; ".join(f"{i + 1}. {t['title']}" for i, t in enumerate(top))
            more = f" (+{len(ts) - 5} more)" if len(ts) > 5 else ""
            first = top[0]["title"]
            return Reply(f"On your list: {names}{more}. I'd start with “{first}”.", intent="task_list")
        return None

    def _stats(self, low: str, raw: str) -> Optional[Reply]:
        a = self.api
        if re.search(r"\b(this|the|my|last) week\b|\bweekly\b", low) and re.search(r"\b(how|report|summary|stats|did|focus)\b", low):
            w = a.week_report()
            return Reply(" ".join(w["lines"][:3]), intent="stats_week")
        if re.search(r"\bhow (did|am|was|have) i\b|\b(today'?s|daily) (report|summary|stats)\b|\bhow much (did i )?focus|"
                     r"\bfocus time\b|\bmy (progress|stats|day)\b|\bsummary\b", low):
            r = a.day_report()
            return Reply(" ".join(r["lines"][:3]) + (f" Tip: {r['tip']}" if r.get("tip") else ""), intent="stats_day",
                         emotion="PROUD" if r["score"] >= 60 else "HAPPY")
        if re.search(r"\b(best|most productive|peak)\b.*\b(time|hours?|focus)\b|\bwhen (do|should) i (focus|work|study)\b",
                     low):
            ins = [i for i in a.insights() if i.kind in ("best_hours", "risky_hours", "learning")]
            if not ins:
                return Reply("I don't have enough history yet to tell - give me a few days!", intent="coach")
            return Reply(" ".join(i.text for i in ins[:2]), intent="coach")
        if re.search(r"\b(what|who)\b.*\bdistract", low) or re.search(r"\btop distractions?\b|\btime sinks?\b", low):
            top = a.top_distractors()
            if not top:
                return Reply("No big distractions recorded lately - nice!", intent="distractors", emotion="PROUD")
            return Reply("Lately: " + ", ".join(f"{t} ({mins(v)})" for t, v in top) +
                         ". Want me to block one during focus? Mark it Blocked in Sites & Apps.", intent="distractors")
        if re.search(r"\bstreak\b", low):
            n = a.streak()
            return Reply(f"Your streak is {n} day{'s' if n != 1 else ''}! {'Keep it alive today 🔥' if n else 'One session today starts a new one.'}",
                         intent="streak")
        if re.search(r"\b(what did i do|remember|today so far|recap)\b", low):
            ms = a.memories_today()
            if not ms:
                return Reply("Nothing special to report yet today - the day's still young!", intent="recap")
            return Reply("Today: " + " ".join(m["text"] for m in ms[:4]), intent="recap")
        return None

    def _monitoring(self, low: str, raw: str) -> Optional[Reply]:
        a = self.api
        if re.search(r"\b(pause|stop|turn off|disable)\b.*\b(monitoring|watching|tracking)\b", low):
            a.set_monitoring(False)
            return Reply("Monitoring paused. I'll keep you company, but I won't watch the screen.",
                         intent="monitor_off", actions=[("monitoring", False)])
        if re.search(r"\b(resume|start|turn on|enable)\b.*\b(monitoring|watching|tracking)\b", low):
            a.set_monitoring(True)
            return Reply("Monitoring is back on. I've got your back!", intent="monitor_on",
                         actions=[("monitoring", True)])
        return None

    def _feelings(self, low: str, raw: str) -> Optional[Reply]:
        if re.search(r"\b(can'?t|cannot|unable to) (focus|concentrate)\b|\bprocrastinat|\bdistracted\b|\bunmotivated\b|"
                     r"\bmotivate me\b|\bno motivation\b", low):
            return Reply(self.rng.choice(MOTIVATION) + " Want me to start a 15-minute session? Just say “start 15 minute focus”.",
                         intent="motivate", emotion="DETERMINED")
        if re.search(r"\b(tired|exhausted|sleepy|burn(ed|t)? out)\b", low):
            return Reply("You've earned a rest. A 10-minute break, some water and a stretch work wonders - say “take a "
                         "break” and I'll time it.", intent="tired", emotion="WORRIED")
        if re.search(r"\b(stressed|anxious|overwhelmed|sad|upset|bad day)\b", low):
            return Reply("That sounds like a lot. Let's make it smaller: pick just one tiny task, and I'll sit right "
                         "here with you. You're doing better than you think. 💛", intent="support", emotion="LOVE")
        if re.search(r"\b(bored|boring)\b", low):
            return Reply("Boredom is just your brain asking for a challenge. Try a 20-minute sprint - race me!",
                         intent="bored", emotion="SMUG")
        return None

    def _smalltalk(self, low: str, raw: str) -> Optional[Reply]:
        a = self.api
        name = a.user_name()
        if re.fullmatch(r"(hi|hello|hey|yo|hiya|good (morning|afternoon|evening)|sup|hola)( there)?( \w+)?", low):
            return Reply(self.rng.choice((f"Hi {name}! 👋 What are we working on today?",
                                          f"Hey {name}! Ready to make today count?")), intent="greet",
                         emotion="EXCITED")
        if re.search(r"\bhow are you\b|\bhow'?s it going\b|\bhow do you feel\b", low):
            mood = a.mood().lower()
            return Reply({"proud": "Proud of you, honestly! You've been on a roll.",
                          "tired": "A bit sleepy - we've been at it a while. Stretch with me?",
                          "sleepy": "Yawn... it's late! But I'm here for you.",
                          "worried": "A little worried - lots of tab hopping lately. Shall we pick one thing?"}.get(
                mood, "I'm great - happiest when we're getting stuff done together!"), intent="how_are_you")
        if re.search(r"\b(joke|make me laugh|funny)\b", low):
            return Reply(self.rng.choice(JOKES) + " 😄", intent="joke", emotion="LAUGHING")
        if re.search(r"\b(thanks|thank you|thx|ty)\b", low):
            return Reply(self.rng.choice(("Anytime! 💛", "Happy to help!", "That's what buddies are for!")),
                         intent="thanks", emotion="LOVE")
        if re.search(r"\b(bye|good ?night|see you|cya|later)\b", low):
            return Reply(f"Bye {name}! Proud of you today. 🌙", intent="bye", emotion="HAPPY")
        if re.search(r"\b(help|what can you do|commands|how do (i|you) use)\b", low):
            return Reply("I can start, pause or end focus sessions (“start a 45 minute focus on my essay”), keep your "
                         "tasks (“add task …”, “what's on my list?”, “done with …”), tell you how your day or week "
                         "went, when you focus best and what distracts you, take a break with you, or just chat.",
                         intent="help")
        if re.search(r"\b(who are you|your name|what are you)\b", low):
            return Reply(f"I'm {a.companion_name()}, your focus buddy! I live on your desktop and cheer you on.",
                         intent="who")
        if re.search(r"\b(love you|you'?re (cute|the best|awesome))\b", low):
            return Reply("Aww! 💕 You're pretty great yourself.", intent="love", emotion="LOVE")
        return None
