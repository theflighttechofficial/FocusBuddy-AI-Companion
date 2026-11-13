"""Fallback: guess the site from a browser window title.

Titles are unreliable (sites change them, tabs can be renamed), so this is
used only when the FocusBuddy browser extension is not connected, and
results are tagged ``domain_source="title"`` (lower confidence).
"""

from __future__ import annotations

import re
from typing import Optional

BROWSER_SUFFIXES = re.compile(
    r"\s*[-—–|]\s*(google chrome|mozilla firefox|firefox|microsoft​?\s*edge|brave|opera|"
    r"vivaldi|chromium|arc)\s*$", re.IGNORECASE)
EDGE_EXTRA = re.compile(r"\s+and \d+ more pages?$", re.IGNORECASE)
PROFILE = re.compile(r"\s*[-—]\s*(personal|work|profile \d+|[^-—]{1,25} \(profile\))\s*$", re.IGNORECASE)
SPLIT = re.compile(r"\s+[-—–|•·/]\s+|\s*\|\s*")
DOMAIN_RE = re.compile(r"\b((?:[a-z0-9-]+\.)+(?:com|org|net|io|dev|ai|app|co|in|edu|gov|tv|me|so|gg|ly|uk|de))\b",
                       re.IGNORECASE)

# Common sites whose tab titles end with a brand name.
TITLE_BRANDS = {
    "youtube": "youtube.com", "instagram": "instagram.com", "facebook": "facebook.com",
    "reddit": "reddit.com", "netflix": "netflix.com", "tiktok": "tiktok.com", "x": "x.com",
    "twitter": "twitter.com", "twitch": "twitch.tv", "pinterest": "pinterest.com",
    "linkedin": "linkedin.com", "github": "github.com", "stack overflow": "stackoverflow.com",
    "google docs": "docs.google.com", "google sheets": "sheets.google.com",
    "google slides": "slides.google.com", "google drive": "drive.google.com", "gmail": "mail.google.com",
    "whatsapp": "web.whatsapp.com", "discord": "discord.com", "prime video": "primevideo.com",
    "chatgpt": "chatgpt.com", "claude": "claude.ai", "notion": "notion.so", "leetcode": "leetcode.com",
    "wikipedia": "wikipedia.org", "coursera": "coursera.org", "udemy": "udemy.com", "kaggle": "kaggle.com",
    "overleaf": "overleaf.com", "hotstar": "hotstar.com", "jiocinema": "jiocinema.com", "spotify": "open.spotify.com",
    "youtube music": "music.youtube.com", "disney+ hotstar": "hotstar.com", "jiohotstar": "hotstar.com",
    "zee5": "zee5.com", "sonyliv": "sonyliv.com", "hulu": "hulu.com", "disney+": "disneyplus.com",
    "crunchyroll": "crunchyroll.com", "snapchat": "snapchat.com", "threads": "threads.net", "tumblr": "tumblr.com",
    "9gag": "9gag.com", "imgur": "imgur.com", "quora": "quora.com", "amazon.in": "amazon.in",
    "amazon.com": "amazon.com", "flipkart": "flipkart.com", "myntra": "myntra.com", "meesho": "meesho.com",
    "cricbuzz": "cricbuzz.com", "espncricinfo": "espncricinfo.com", "telegram": "web.telegram.org",
    "telegram web": "web.telegram.org", "messenger": "messenger.com", "roblox": "roblox.com",
    "chess.com": "chess.com", "lichess.org": "lichess.org", "steam": "store.steampowered.com",
    "kick": "kick.com", "linkedin feed": "linkedin.com", "dailymotion": "dailymotion.com",
}

# Distracting brands distinctive enough to be recognised anywhere in a title segment
# ("@user • Instagram photos and videos", "(20+) Facebook", "Home / X" is too short and is not here).
EMBEDDED_BRANDS = {
    "youtube": "youtube.com", "instagram": "instagram.com", "facebook": "facebook.com", "netflix": "netflix.com",
    "tiktok": "tiktok.com", "twitch": "twitch.tv", "reddit": "reddit.com", "pinterest": "pinterest.com",
    "hotstar": "hotstar.com", "jiocinema": "jiocinema.com", "crunchyroll": "crunchyroll.com",
    "prime video": "primevideo.com", "snapchat": "snapchat.com", "9gag": "9gag.com", "whatsapp": "web.whatsapp.com",
    "disney+": "disneyplus.com", "sonyliv": "sonyliv.com", "zee5": "zee5.com", "cricbuzz": "cricbuzz.com",
}
# words that mean the page is *about* a brand, not on it ("YouTube API docs - Google Search")
_ABOUT = re.compile(r"\b(api|sdk|docs?|documentation|developer|tutorial|download(er)?|-dl|clone|search|"
                    r"wikipedia|github|stack overflow|npm|pypi)\b", re.IGNORECASE)


def clean_title(title: str) -> str:
    t = title.strip()
    for _ in range(3):
        t2 = BROWSER_SUFFIXES.sub("", t)
        t2 = PROFILE.sub("", t2) if t2 != t else t2
        t2 = EDGE_EXTRA.sub("", t2)
        if t2 == t:
            break
        t = t2
    return t


def infer_domain(title: str, extra_brands: Optional[dict[str, str]] = None) -> Optional[str]:
    if not title:
        return None
    t = clean_title(title)
    segments = [s.strip().lower() for s in SPLIT.split(t) if s.strip()]
    brands = dict(TITLE_BRANDS)
    if extra_brands:
        for k, v in extra_brands.items():
            brands.setdefault(k, v)
    # brand names are usually the last segment ("Video name - YouTube"), sometimes the first
    for seg in list(reversed(segments)) + segments[:1]:
        seg = re.sub(r"^\(\d+\+?\)\s*", "", seg)   # "(3) YouTube" / "(20+) Facebook" notification counters
        seg = seg.strip(" •·")
        if seg in brands:
            return brands[seg]
    # distinctive distracting brands inside a segment ("@user • Instagram photos and videos")
    if not _ABOUT.search(t):
        low = t.lower()
        for word, dom in EMBEDDED_BRANDS.items():
            if re.search(r"(?<![a-z0-9])" + re.escape(word) + r"(?![a-z0-9])", low):
                return dom
    m = DOMAIN_RE.search(t)
    if m:
        d = m.group(1).lower()
        return d[4:] if d.startswith("www.") else d
    # reddit titles: "r/python"
    if re.search(r"(^|\s)r/[a-z0-9_]+", t, re.IGNORECASE):
        return "reddit.com"
    return None
