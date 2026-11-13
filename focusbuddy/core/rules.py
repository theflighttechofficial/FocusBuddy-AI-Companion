"""User-configurable website / application rules.

Categories
    productive   - focus websites / allowed applications
    neutral      - everything not listed (lightly scored)
    distracting  - distraction websites / apps (scored, escalates over time)
    blocked      - explicitly blocked (high score immediately)

``auto_close`` marks rules the user allows FocusBuddy to close automatically
(only used when "Automatic blocking" is enabled).

Domain patterns match the domain and its subdomains; the most specific
(longest) pattern wins, so ``music.youtube.com`` can be productive while
``youtube.com`` is distracting.  A pattern may contain a path prefix
(``youtube.com/shorts``).  App patterns match the executable name and
support ``*`` wildcards.
"""

from __future__ import annotations

import fnmatch
import enum
from dataclasses import dataclass
from typing import Iterable, Optional
from urllib.parse import urlsplit


class Category(str, enum.Enum):
    PRODUCTIVE = "productive"
    NEUTRAL = "neutral"
    DISTRACTING = "distracting"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class Rule:
    kind: str          # website | app
    pattern: str
    category: Category
    auto_close: bool = False


def normalize_domain(value: str | None) -> Optional[str]:
    """'https://www.YouTube.com:443/watch?v=1' -> 'youtube.com'."""
    if not value:
        return None
    v = value.strip().lower()
    if "://" not in v:
        v = "http://" + v
    try:
        host = urlsplit(v).hostname or ""
    except ValueError:
        return None
    if host.startswith("www."):
        host = host[4:]
    return host or None


def url_path(url: str | None) -> str:
    if not url:
        return ""
    try:
        return urlsplit(url if "://" in url else "http://" + url).path or ""
    except ValueError:
        return ""


def normalize_app(value: str) -> str:
    v = value.strip().lower()
    if v and "*" not in v and "." not in v:
        v += ".exe"
    return v


_W, _A = "website", "app"
P, D, B = Category.PRODUCTIVE, Category.DISTRACTING, Category.BLOCKED

DEFAULT_RULES: list[Rule] = [
    # ---- focus websites ----
    *[Rule(_W, d, P) for d in (
        "github.com", "gitlab.com", "bitbucket.org", "stackoverflow.com", "stackexchange.com",
        "superuser.com", "serverfault.com", "docs.python.org", "pypi.org", "developer.mozilla.org",
        "docs.google.com", "sheets.google.com", "slides.google.com", "drive.google.com",
        "calendar.google.com", "scholar.google.com", "colab.research.google.com", "notion.so",
        "overleaf.com", "leetcode.com", "hackerrank.com", "codeforces.com", "kaggle.com",
        "coursera.org", "edx.org", "khanacademy.org", "udemy.com", "nptel.ac.in", "wikipedia.org",
        "arxiv.org", "learn.microsoft.com", "w3schools.com", "geeksforgeeks.org", "huggingface.co",
        "readthedocs.io", "readthedocs.org", "figma.com", "trello.com", "atlassian.net", "linear.app",
        "office.com", "onedrive.live.com", "chatgpt.com", "claude.ai", "localhost", "127.0.0.1",
        "music.youtube.com", "gemini.google.com", "perplexity.ai", "copilot.microsoft.com", "replit.com",
        "codepen.io", "vercel.com", "netlify.com", "render.com", "railway.app", "supabase.com", "firebase.google.com",
        "console.cloud.google.com", "portal.azure.com", "aws.amazon.com", "docker.com", "npmjs.com",
        "medium.com/@", "dev.to", "freecodecamp.org", "brilliant.org", "datacamp.com", "codecademy.com",
        "classroom.google.com", "meet.google.com", "teams.microsoft.com", "canva.com", "miro.com",
        "mail.google.com", "outlook.office.com", "outlook.live.com", "zotero.org", "researchgate.net",
        "ieeexplore.ieee.org", "sciencedirect.com", "springer.com", "acm.org", "jstor.org", "desmos.com",
        "wolframalpha.com", "mathworks.com", "tableau.com", "powerbi.microsoft.com", "deepl.com")],
    # ---- distraction websites ----
    Rule(_W, "youtube.com", D, True),
    Rule(_W, "instagram.com", D, True),
    Rule(_W, "tiktok.com", D, True),
    *[Rule(_W, d, D) for d in (
        "facebook.com", "twitter.com", "x.com", "reddit.com", "netflix.com", "primevideo.com",
        "hotstar.com", "jiocinema.com", "twitch.tv", "9gag.com", "pinterest.com", "tumblr.com",
        "snapchat.com", "discord.com", "buzzfeed.com", "crunchyroll.com", "store.steampowered.com",
        "epicgames.com", "roblox.com", "chess.com", "imgur.com", "threads.net", "web.whatsapp.com",
        "web.telegram.org", "hulu.com", "disneyplus.com", "amazon.in", "flipkart.com", "myntra.com",
        "cricbuzz.com", "espncricinfo.com",
        # video / streaming
        "zee5.com", "sonyliv.com", "mxplayer.in", "aha.video", "sunnxt.com", "voot.com", "dailymotion.com",
        "kick.com", "bilibili.com", "vimeo.com", "max.com", "peacocktv.com", "paramountplus.com", "tv.apple.com",
        "9anime.to", "aniwatch.to", "hianime.to",
        # social / chat / forums
        "linkedin.com/feed", "quora.com", "messenger.com", "telegram.org", "vk.com", "weibo.com", "bsky.app",
        "mastodon.social", "lemmy.world", "4chan.org", "ifunny.co", "knowyourmeme.com",
        # shopping
        "amazon.com", "ebay.com", "meesho.com", "ajio.com", "nykaa.com", "aliexpress.com", "temu.com",
        "shein.com", "etsy.com", "walmart.com", "bigbasket.com", "swiggy.com", "zomato.com",
        # games
        "lichess.org", "poki.com", "crazygames.com", "miniclip.com", "friv.com", "y8.com", "itch.io",
        "store.epicgames.com", "playvalorant.com", "leagueoflegends.com", "nytimes.com/games",
        # sports / news / gossip
        "espn.com", "espn.in", "skysports.com", "goal.com", "onefootball.com", "nba.com", "iplt20.com",
        "fantasy.iplt20.com", "dream11.com", "bollywoodhungama.com", "pinkvilla.com", "tmz.com",
        )],
    # ---- allowed applications ----
    *[Rule(_A, a, P) for a in (
        "code.exe", "cursor.exe", "pycharm64.exe", "idea64.exe", "webstorm64.exe", "clion64.exe",
        "rider64.exe", "datagrip64.exe", "goland64.exe", "studio64.exe", "devenv.exe",
        "sublime_text.exe", "notepad++.exe", "windowsterminal.exe", "wt.exe", "cmd.exe",
        "powershell.exe", "pwsh.exe", "winword.exe", "excel.exe", "powerpnt.exe", "onenote.exe",
        "outlook.exe", "obsidian.exe", "notion.exe", "figma.exe", "acrobat.exe", "acrord32.exe",
        "matlab.exe", "rstudio.exe", "postman.exe", "teams.exe", "ms-teams.exe", "zoom.exe",
        "jupyter-lab.exe", "anaconda-navigator.exe", "dbeaver.exe", "blender.exe")],
    # ---- distraction applications (never auto-closed by default) ----
    *[Rule(_A, a, D) for a in (
        "steam.exe", "steamwebhelper.exe", "epicgameslauncher.exe", "discord.exe", "whatsapp.exe",
        "telegram.exe", "riotclientservices.exe", "leagueclient.exe", "valorant.exe",
        "robloxplayerbeta.exe", "battle.net.exe", "eadesktop.exe", "minecraft.exe",
        "minecraftlauncher.exe", "netflix.exe",
        # launchers / stores
        "galaxyclient.exe", "ubisoftconnect.exe", "upc.exe", "origin.exe", "rockstarlauncher.exe",
        "xboxapp.exe", "gamingservices.exe", "playnite.desktopapp.exe", "itch.exe", "bluestacks.exe",
        "hd-player.exe", "ldplayer.exe", "nox.exe", "memu.exe",
        # popular games (more are caught by their install folder, see GAME_DIRS)
        "minecraft.windows.exe", "fortniteclient-win64-shipping.exe",
        "valorant-win64-shipping.exe", "league of legends.exe", "cs2.exe", "csgo.exe", "dota2.exe",
        "gta5.exe", "gta5_enhanced.exe", "rdr2.exe", "eldenring.exe", "genshinimpact.exe", "yuanshen.exe",
        "starrail.exe", "zenlesszonezero.exe", "pubg.exe", "tslgame.exe", "apex_legends.exe", "r5apex.exe",
        "overwatch.exe", "rocketleague.exe", "fifa23.exe", "fc24.exe", "fc25.exe", "cod.exe",
        "modernwarfare.exe", "warzone.exe", "freefire.exe", "amongus.exe", "terraria.exe", "stardew valley.exe",
        "hollow_knight.exe", "cyberpunk2077.exe", "witcher3.exe", "palworld-win64-shipping.exe",
        "marvelrivals-win64-shipping.exe", "deltaforceclient-win64-shipping.exe", "bgmi.exe",
        # chat / social / streaming apps
        "instagram.exe", "facebook.exe", "messenger.exe", "snapchat.exe", "tiktok.exe", "signal.exe",
        "primevideo.exe", "disneyplus.exe", "hotstar.exe", "crunchyroll.exe")],
]

# Install folders of games / launchers: any exe running from one of these is treated as a game
# (distracting) unless a rule says otherwise. Matched case-insensitively against the full path.
GAME_DIRS = ("\\steamapps\\common\\", "\\epic games\\", "\\riot games\\", "\\gog galaxy\\games\\",
             "\\ubisoft game launcher\\games\\", "\\ea games\\", "\\electronic arts\\",
             "\\rockstar games\\", "\\xboxgames\\", "\\windowsapps\\microsoft.minecraft",
             "\\battle.net\\", "\\blizzard entertainment\\", "\\hoyoplay\\", "\\genshin impact\\",
             "\\roblox\\versions\\")

# Bump when DEFAULT_RULES gains entries: existing installs receive the new defaults (their own
# rules and edits are never overwritten).
RULES_VERSION = 2


def is_game_path(path: str | None) -> bool:
    p = (path or "").lower().replace("/", "\\")
    return any(d in p for d in GAME_DIRS)


class RuleSet:
    def __init__(self, rules: Iterable[Rule]):
        self.sites: list[Rule] = []
        self.apps: list[Rule] = []
        for r in rules:
            if r.kind == "website":
                self.sites.append(r)
            else:
                self.apps.append(r)
        # most specific first
        self.sites.sort(key=lambda r: len(r.pattern), reverse=True)
        self.apps.sort(key=lambda r: ("*" in r.pattern, -len(r.pattern)))

    @classmethod
    def from_rows(cls, rows: Iterable[dict]) -> "RuleSet":
        out = []
        for row in rows:
            try:
                out.append(Rule(row["kind"], row["pattern"], Category(row["category"]), bool(row.get("auto_close"))))
            except (KeyError, ValueError):
                continue
        return cls(out)

    @classmethod
    def defaults(cls) -> "RuleSet":
        return cls(DEFAULT_RULES)

    # ------------------------------------------------------------- matching
    def match_domain(self, domain: str | None, path: str = "") -> Optional[Rule]:
        if not domain:
            return None
        domain = domain.lower()
        for r in self.sites:
            pat = r.pattern
            host, _, ppath = pat.partition("/")
            if domain == host or domain.endswith("." + host):
                if ppath and not path.lstrip("/").startswith(ppath):
                    continue
                return r
        return None

    def match_app(self, exe: str | None) -> Optional[Rule]:
        if not exe:
            return None
        exe = exe.lower()
        for r in self.apps:
            if r.pattern == exe or fnmatch.fnmatch(exe, r.pattern):
                return r
        return None

    def classify(self, exe: str, domain: str | None, path: str = "", is_browser: bool = False) -> tuple[Category, Optional[Rule]]:
        if is_browser and domain:
            r = self.match_domain(domain, path)
            if r:
                return r.category, r
            return Category.NEUTRAL, None
        r = self.match_app(exe)
        if r:
            return r.category, r
        return Category.NEUTRAL, None

    def known_site_names(self) -> dict[str, str]:
        """{'youtube': 'youtube.com', ...} used by the title fallback."""
        out: dict[str, str] = {}
        for r in self.sites:
            host = r.pattern.split("/")[0]
            parts = host.split(".")
            if len(parts) >= 2 and parts[0] not in ("docs", "sheets", "slides", "drive", "calendar", "music",
                                                    "web", "store", "learn", "scholar", "colab", "developer"):
                out.setdefault(parts[0], host)
        return out
