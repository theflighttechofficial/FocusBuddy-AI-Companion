"""Short, character-flavoured lines for the companion's life (wake-up, entrances, gags, the close
scene, study-partner moments).

Lines are deliberately short: the fun should come from what the companion does, not from walls of
text. ``{name}`` is the user's name and ``{buddy}`` the companion's; extra fields (``{n}``, ``{m}``)
are filled by the caller. ``pick`` avoids repeating a recently used line.
"""

from __future__ import annotations

import random
from typing import Optional

# flavour: cat dog fox rabbit robot buddy girl saree  (falls back to "any")
LINES: dict[str, dict[str, list[str]]] = {
    "WAKE": {
        "any": ["Mmh… morning already?", "*yaaawn* Oh, hi {name}!", "Five more minutes… okay, I'm up."],
        "cat": ["*stretch* …I was not asleep. I was resting my eyes.", "Mrrp. You're back."],
        "dog": ["You're here! You're HERE!", "Good morning, {name}! Is it walk time? It's work time. Okay!"],
        "rabbit": ["*ear twitch* …oh! Hi!", "*sniff sniff* Smells like a productive day."],
        "fox": ["*peeks* …is it safe? Oh, it's you.", "Morning. I definitely didn't knock anything over."],
        "robot": ["BOOT SEQUENCE COMPLETE. Hello, {name}.", "Systems online. Cheerfulness: 100 %."],
        "buddy": ["Morning, {name}! Let's do this.", "*stretch* Okay, I'm awake. Mostly."],
        "girl": ["Hi {name}! Ready for today?", "Morning! I brought snacks. Imaginary ones."],
        "saree": ["Good morning, {name}! 🌸", "Hello again! Shall we begin?"],
    },
    "ENTER": {
        "any": ["Hi!", "I'm back!", "Did you miss me?"],
        "cat": ["*saunters in*", "Mrrow."],
        "dog": ["I'M BACK!", "Hi hi hi!"],
        "rabbit": ["*hop hop*", "Boing!"],
        "fox": ["…you didn't see me come in.", "*sneaks in*"],
        "robot": ["Landing complete.", "Returning to base."],
        "saree": ["I'm here! 🌸"],
    },
    "LEAVE": {
        "any": ["Be right back!", "Going on patrol.", "Brb — exploring."],
        "cat": ["I have… cat business.", "Don't wait up."],
        "fox": ["Shh. Secret mission.", "If anyone asks, I was never here."],
        "robot": ["Initiating perimeter scan.", "Recharging elsewhere. Brb."],
        "dog": ["I'll be RIGHT back!"],
    },
    "CLOSE_ASK": {
        "any": ["Wait, we're done already? Should I go curl up for a nap? 💤"],
        "robot": ["Dashboard closing detected. Enter sleep mode? 💤"],
        "saree": ["Oh, done already? Should I take a little nap? 💤"],
    },
    "CLOSE_STAY": {
        "any": ["Yay! I'll stay right here 🐾", "Okay! I'll keep you company."],
        "robot": ["Staying online. 🤖"],
        "dog": ["YES! Staying! Best day!"],
    },
    "CLOSE_NAP": {
        "any": ["Okay… nap time. Wake me if you need me 💤", "Night night, {name}."],
        "cat": ["*curls up* Finally.", "Do not disturb. Unless treats."],
        "robot": ["Entering low-power mode… 💤"],
    },
    "GOODBYE": {
        "any": ["Bye {name}! See you soon 👋", "See you next time!"],
        "dog": ["Bye bye! Come back soon, okay?!"],
        "cat": ["Fine. Go. …come back soon."],
        "robot": ["Shutting down. Goodbye, {name}. 👋"],
        "fox": ["Until next time… 🦊"],
        "saree": ["Bye {name}! Take care 🌸"],
    },
    "NAP_ASK": {
        "any": ["I'm getting sleepy… nap time? 💤", "*yawn* Mind if I take a little nap?"],
        "robot": ["Battery low. Permission to charge? 🔋"],
        "cat": ["It's nap o'clock. Agreed?"],
    },
    "NAP_YES": {"any": ["Thank youuu… 💤"], "robot": ["Charging… 🔋"]},
    "NAP_NO": {"any": ["Okay! Coffee mode ☕", "Fine, I'll stay awake… for you."],
               "robot": ["Overclocking. ☕"]},
    "GIFT": {
        "any": ["I brought you something! …it's imaginary. 🎁", "A present! Don't ask where I found it."],
        "dog": ["I got the ball! THE BALL! 🎾", "Here! For you! It's a stick! A very good stick!"],
        "cat": ["I caught you a… thing. You're welcome.", "A gift. It was moving earlier."],
        "robot": ["Delivering one (1) virtual gift. 🎁"],
        "girl": ["I found a good book for you 📚", "Here — a bookmark! 🔖"],
        "saree": ["A little flower for you 🌼"],
    },
    "FAKE_LEAVE": {"any": ["…just kidding!", "Ha! Got you."], "fox": ["Fooled you. 😏"]},
    "FORGOT": {"any": ["Wait… why did I come here?", "I had a plan. It's gone now."],
               "robot": ["Error 404: purpose not found."]},
    "CAUGHT_STARING": {"any": ["…I wasn't looking.", "What? Nothing."], "cat": ["*looks away innocently*"]},
    "PROUD": {"any": ["Nailed it.", "Did you see that?!"], "fox": ["Totally meant to do that."]},
    "OFFENDED": {"any": ["Hmph. Fine.", "Ignored. Again. *sigh*"], "cat": ["I see how it is."],
                 "dog": ["…oh. Okay."], "robot": ["Attention request: denied. Sad."]},
    "OVER_IT": {"any": ["…okay, I'm over it.", "Fine, you're forgiven."], "cat": ["I forgive you. This once."]},
    "CHALLENGE": {
        "any": ["Bet you can't do a 25-minute session without peeking at your phone. 😏",
                "Focus challenge: one session, zero detours. You in?"],
        "fox": ["I bet you can't out-focus a fox. Prove me wrong."],
        "robot": ["Challenge issued: 25 minutes of focus. Accept?"],
    },
    "SMUG": {"any": ["Called it. 😏", "Knew you'd go there."], "fox": ["Predictable. 😏"],
             "cat": ["*smug blink*"]},
    "STREAK": {"any": ["{n} sessions today! Look at us go! 🔥", "That's {n} today. We're on a roll!"],
               "robot": ["Streak counter: {n}. Impressive. 🔥"]},
    "MILESTONE": {"any": ["{n} focus sessions together! 🎉", "Session number {n}! I'm so proud of us!"]},
    "STUDY_CHEER": {"any": ["You've got this.", "Nice focus. 👍", "Steady work. Keep going."],
                    "robot": ["Productivity nominal. 👍"], "dog": ["You're doing great!!"]},
    "STUCK": {"any": ["Stuck? Want a hint or a short break?"], "robot": ["Progress stalled. Hint or break?"]},
    "HINT": {"any": ["Try writing down the very next tiny step.", "Explain it out loud, like to a rubber duck. Or me.",
                     "Skip it for now and come back — your brain keeps working on it.",
                     "Re-read the question slowly. The answer often hides there.",
                     "Draw it. Boxes and arrows help."]},
    "SLEEPY_FOCUS": {"any": ["Long session… mind if I nap beside you? 💤"], "robot": ["Long task. Entering standby nearby?"]},
    "BREAK_OFFER": {"any": ["You've been at it for {m} minutes. Stretch break?", "{m} minutes straight! Water and a stretch?"],
                    "robot": ["{m} minutes of continuous work. Maintenance break recommended."]},
    "REMARK": {
        "any": ["Nice day for getting stuff done.", "I like it here.", "*hums*"],
        "cat": ["This window is warm. I'm keeping it.", "*tail flick*"],
        "dog": ["Everything is great!", "I love this desktop!"],
        "rabbit": ["*sniff* …new file smell.", "*nose twitch*"],
        "fox": ["I see everything.", "Interesting window you've got there."],
        "robot": ["Scanning… 3 windows, 0 threats.", "Beep. Boop. Just checking in."],
        "buddy": ["What are we working on?", "Coffee break later?"],
        "girl": ["Good playlist for studying?", "I love a tidy desktop."],
        "saree": ["Such a calm day 🌸", "Let me know if you need me."],
    },
    "RETURN_TO_WORK": {"any": ["Back on track ✨", "That's the spirit."]},
    "PET_REPLY": {"any": ["Hehe.", "♥"], "cat": ["*purrs*"], "dog": ["*happy wiggle*"], "robot": ["Affection logged. ♥"]},
    # 2.0 reactions
    "SNEEZE": {"any": ["Excuse me!", "*sniff* …bless me.", "Oops. Dusty screen."],
               "cat": ["*offended by own sneeze*"], "dog": ["Ha! That tickled."], "robot": ["Dust in the fan. Ahem."],
               "saree": ["Oh! Excuse me. 🌸"]},
    "HICCUP": {"any": ["*hic* …sorry. *hic*", "Hiccups! *hic* Don't look at me."], "robot": ["Glitch. *hic* Glitch."]},
    "SHIVER": {"any": ["Brrr. Is it cold in here?", "*shiver* Who opened a window?"], "robot": ["Temperature: chilly."]},
    "DAYDREAM": {"any": ["…sorry, I was miles away.", "Hm? Oh! I was daydreaming."],
                 "cat": ["I was thinking about fish. Important fish."], "dog": ["I was dreaming of a park. A big one."]},
    "HUM": {"any": ["♪ hm-hm-hmm ♪", "♪ la-la… ♪"], "robot": ["♪ beep-boop-beep ♪"]},
}


def flavor_of(char_id: str, species: str) -> str:
    """The line flavour for a character: its own id for the classic six, human companions by look."""
    if char_id in ("cat", "dog", "fox", "rabbit", "robot"):
        return char_id
    if char_id == "study_girl":
        return "girl"
    if char_id == "saree_girl":
        return "saree"
    if char_id == "buddy" or species == "buddy":
        return "buddy"
    return species if species in LINES["WAKE"] else "any"


class LinePicker:
    """Chooses a line, avoiding the most recently used ones (kept in the persisted memory)."""

    def __init__(self, rng: Optional[random.Random] = None, recent: Optional[list] = None, keep: int = 24):
        self.rng = rng or random.Random()
        self.recent: list[str] = list(recent or [])
        self.keep = keep

    def pick(self, key: str, flavor: str = "any", **fields) -> str:
        table = LINES.get(key)
        if not table:
            return ""
        pool = list(table.get(flavor, [])) + list(table.get("any", []))
        if not pool:
            return ""
        fresh = [s for s in pool if s not in self.recent] or pool
        own = [s for s in fresh if s in table.get(flavor, [])]
        line = self.rng.choice(own if own and self.rng.random() < 0.7 else fresh)
        self.recent = (self.recent + [line])[-self.keep:]
        try:
            return line.format(**{"name": "friend", "buddy": "me", "n": "", "m": "", **fields})
        except (KeyError, IndexError, ValueError):
            return line
