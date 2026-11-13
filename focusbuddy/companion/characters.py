"""Character registry: looks, default names and personalities.

Built-in characters are drawn procedurally (see renderer.py) or from their GLB assets.
Any other folder ``characters/<id>/`` (bundled ``assets/characters`` or the user's data
folder) that contains ``model.glb`` (or ``model.gltf``) and a ``metadata.json`` is
registered automatically - no code change: copy the folder, run
``python -m tools.validate_asset <folder>``, restart, and it appears in the selector.
Its behaviour, skeleton fallback and dialogue come from ``species`` / ``personality``
(one of cat, dog, fox, rabbit, robot, buddy).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Palette:
    main: str
    shade: str
    light: str
    outline: str
    accent: str = "#F7A1A8"     # inner ears, cheeks
    extra: str = "#3A2A25"      # tips, nose, hair...


@dataclass(frozen=True)
class CharacterSpec:
    id: str
    display_name: str
    default_name: str
    emoji: str
    skeleton: str               # quadruped | biped | sprite
    species: str
    personality: str            # dialogue file id
    traits: tuple[str, ...]
    tagline: str
    palette: Palette
    sprite_dir: str | None = None
    extra: dict = field(default_factory=dict, compare=False, hash=False)


BUILTIN: dict[str, CharacterSpec] = {
    "cat": CharacterSpec(
        "cat", "Cat", "Momo", "🐱", "quadruped", "cat", "cat",
        ("Lazy", "Playful", "Curious", "Slightly sarcastic"),
        "Naps a lot, judges your tabs.",
        Palette("#F7B46A", "#E08B42", "#FFE9CC", "#7A4318", "#F7A1A8", "#C9702E")),
    "dog": CharacterSpec(
        "dog", "Dog", "Biscuit", "🐶", "quadruped", "dog", "dog",
        ("Energetic", "Supportive", "Encouraging"),
        "Your biggest fan. Very excited about your to-do list.",
        Palette("#D9A066", "#B47A41", "#F6E3C6", "#5A3A1E", "#F5A3A3", "#7B4A22")),
    "fox": CharacterSpec(
        "fox", "Fox", "Ember", "🦊", "quadruped", "fox", "fox",
        ("Clever", "Witty", "Gentle"),
        "Sly, smart, and always one step ahead of your distractions.",
        Palette("#F2843A", "#D2651E", "#FFF5EA", "#7A3510", "#F7A1A8", "#3A2A25")),
    "rabbit": CharacterSpec(
        "rabbit", "Rabbit", "Mochi", "🐰", "quadruped", "rabbit", "rabbit",
        ("Shy", "Sweet", "Easily startled"),
        "Soft-spoken, but hops into action when needed.",
        Palette("#F4F1EE", "#D9D1CA", "#FFFFFF", "#857B73", "#F6B3C2", "#B9AFA6")),
    "robot": CharacterSpec(
        "robot", "Robot", "Bolt", "🤖", "biped", "robot", "robot",
        ("Logical", "Efficient", "Funny"),
        "Runs on coffee.exe and distraction statistics.",
        Palette("#D3DCE8", "#A9B6C8", "#F2F6FB", "#4E5D73", "#5CF2FF", "#1E2A3A")),
    "buddy": CharacterSpec(
        "buddy", "Study Buddy", "Sam", "🧑", "biped", "human", "buddy",
        ("Calm", "Coach-like", "Kind"),
        "A tiny study partner who keeps you company.",
        Palette("#7C93FF", "#5C72DD", "#F3C9A4", "#3A3550", "#F59C9C", "#3B2A20")),
    "study_girl": CharacterSpec(
        "study_girl", "Study Girl", "Mia", "👧", "biped", "human", "buddy",
        ("Cheerful", "Focused", "Encouraging"),
        "Headphones on, laptop open - studies right beside you.",
        Palette("#EEF1F8", "#C9D1E6", "#F6D2B8", "#2C2A3A", "#F7A1A8", "#2B2020")),
    "saree_girl": CharacterSpec(
        "saree_girl", "Saree Girl", "Meera", "🌸", "biped", "human", "buddy",
        ("Gentle", "Graceful", "Wise"),
        "Calm and kind, with jasmine in her hair and a book in her hands.",
        Palette("#F7EDE2", "#E7B9A0", "#F3CFB3", "#4A2E28", "#E98A7A", "#2B1B18")),
}


_registry: dict[str, CharacterSpec] = dict(BUILTIN)
_discovered = False
SPECIES_BASE = {"cat": "cat", "dog": "dog", "fox": "fox", "rabbit": "rabbit", "robot": "robot", "human": "buddy",
                "buddy": "buddy"}


def register(spec: CharacterSpec) -> None:
    _registry[spec.id] = spec


def spec_from_metadata(char_id: str, meta: dict, folder: str = "") -> CharacterSpec:
    """CharacterSpec for a GLB character folder (``metadata.json``)."""
    species = str(meta.get("species") or "cat").lower()
    fallback = "buddy" if meta.get("kind") == "biped" else "cat"
    base = BUILTIN[SPECIES_BASE.get(species, fallback)]
    kind = meta.get("kind") or base.skeleton
    pal = meta.get("palette") or {}
    palette = Palette(**{k: pal.get(k, getattr(base.palette, k)) for k in
                         ("main", "shade", "light", "outline", "accent", "extra")})
    personality = str(meta.get("personality") or base.personality)
    if personality not in BUILTIN:
        personality = base.personality
    return CharacterSpec(char_id, str(meta.get("display_name") or char_id.replace("_", " ").title()),
                         str(meta.get("default_name") or base.default_name), str(meta.get("emoji") or base.emoji),
                         "biped" if kind == "biped" else "quadruped", base.species, personality,
                         tuple(meta.get("traits") or base.traits), str(meta.get("tagline") or base.tagline), palette,
                         extra={"asset_dir": folder, "production": bool(meta.get("production")),
                                "license": meta.get("license", ""), "glb": True,
                                "reference": bool(meta.get("reference"))})


def discover(dirs=None) -> list[str]:
    """Register GLB character folders that aren't built-in characters. Returns the new ids."""
    import json
    from pathlib import Path
    if dirs is None:
        try:
            from .character_asset import search_dirs
            dirs = search_dirs()
        except Exception:  # noqa: BLE001
            dirs = []
    added = []
    for d in dirs:
        d = Path(d)
        if not d.is_dir():
            continue
        for folder in sorted(p for p in d.iterdir() if p.is_dir()):
            cid = folder.name
            if cid in BUILTIN or cid in _registry or not ((folder / "model.glb").is_file() or
                                                          (folder / "model.gltf").is_file()):
                continue
            meta = {}
            if (folder / "metadata.json").is_file():
                try:
                    meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
            if meta.get("selectable", True) is False:
                continue                                   # test / internal assets stay out of the list
            register(spec_from_metadata(cid, meta, str(folder)))
            added.append(cid)
    return added


def _ensure() -> None:
    global _discovered
    if not _discovered:
        _discovered = True
        try:
            discover()
        except Exception:  # noqa: BLE001 - a bad folder never breaks the character list
            import logging
            logging.getLogger("focusbuddy.companion").exception("character discovery failed")


def all_characters() -> list[CharacterSpec]:
    _ensure()
    return list(_registry.values())


def selectable_characters() -> list[CharacterSpec]:
    """Characters offered in the pickers (wizard, settings, tray menu). Reference / pipeline assets
    (``"reference": true`` in metadata.json) stay loadable and appear in the studio and showcase."""
    return [s for s in all_characters() if not (s.extra or {}).get("reference")]


def get_character(char_id: str) -> CharacterSpec:
    _ensure()
    return _registry.get(char_id) or BUILTIN["cat"]


def companion_name(spec: CharacterSpec, custom: str = "") -> str:
    return custom.strip() or spec.default_name


def asset_label(spec: CharacterSpec) -> str:
    """Honest label for the 3D model behind a character (shown in the UI)."""
    try:
        from .illustrated import has_pack
        from .renderer import DEFAULT_MODE
        if DEFAULT_MODE.get("mode") == "illustrated" and has_pack(spec.id):
            return "Illustrated from its character sheet"
    except Exception:  # noqa: BLE001
        pass
    try:
        from .character_asset import asset_dir
        import json
        d = asset_dir(spec.id)
        if d is None:
            return "Procedural 3D model"
        meta = json.loads((d / "metadata.json").read_text(encoding="utf-8")) if (d / "metadata.json").is_file() else {}
    except Exception:  # noqa: BLE001
        return "Procedural 3D model"
    if meta.get("source") == "triposr":
        return "3D model reconstructed from its character sheet"
    if meta.get("source") == "hunyuan3d":
        return "3D model generated from its character art (Hunyuan3D), auto-rigged"
    if meta.get("production"):
        return "Artist-made GLB character"
    return "Procedural Demo Character (placeholder GLB)"
