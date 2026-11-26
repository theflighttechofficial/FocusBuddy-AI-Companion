"""Loads character renderers for the dashboard's previews in the background.

Building a 3D character (decoding its GLB and textures, retargeting the motion capture, measuring
its clips) takes 1-2 s per character. The Companion page alone shows eight of them, and the
setup wizard more: built on the UI thread at once, opening the dashboard froze for over 20 s.

``request(spec, mode, quality, callback)`` queues the work on one background thread (one
character at a time, so the rest of the app keeps most of the CPU) and calls ``callback(renderer)``
on the UI thread when it is ready (``None`` -> the caller builds one itself). Previews show the
character's portrait meanwhile.

Only the asset loading and the renderer's own construction run off the UI thread; anything that
needs an OpenGL context is created up front on the UI thread (``GLSkinRaster.create()``).
"""

from __future__ import annotations

import queue
import time
import threading
from typing import Callable

from PySide6.QtCore import QObject, Signal

from ..utils.logging_setup import get_logger

log = get_logger("ui")


class PreviewLoader(QObject):
    done = Signal(int, object)          # ticket, renderer (or None)

    _instance: "PreviewLoader | None" = None

    @classmethod
    def get(cls) -> "PreviewLoader":
        if cls._instance is None:
            cls._instance = PreviewLoader()
        return cls._instance

    def __init__(self):
        super().__init__()
        self._q: "queue.Queue" = queue.Queue()
        self._callbacks: dict[int, Callable] = {}
        self._ticket = 0
        self._thread: threading.Thread | None = None
        self.done.connect(self._deliver)

    def request(self, spec, mode: str, quality: str, callback: Callable) -> int:
        """Queue a renderer build; returns a ticket (``cancel(ticket)`` drops the callback)."""
        self._ticket += 1
        t = self._ticket
        self._callbacks[t] = callback
        self._ensure_gl()
        self._q.put((t, spec, mode, quality))
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, name="FocusBuddyPreviews", daemon=True)
            self._thread.start()
        return t

    def cancel(self, ticket: int) -> None:
        self._callbacks.pop(ticket, None)

    @staticmethod
    def _ensure_gl() -> None:
        """The shared OpenGL context belongs to the UI thread: create it here, never in the worker."""
        try:
            from ..companion.render3d_gl import GLSkinRaster
            GLSkinRaster.create()
        except Exception:  # noqa: BLE001 - no GPU: the CPU path needs no context
            pass

    def _run(self) -> None:
        while True:
            try:
                t, spec, mode, quality = self._q.get(timeout=5.0)
            except queue.Empty:
                return                                   # idle: the thread ends, a new request restarts it
            if t not in self._callbacks:
                continue                                 # cancelled (the preview went away)
            r = None
            try:
                r = self._build(spec, mode, quality)
            except Exception:  # noqa: BLE001 - the UI thread builds it then (and shows any error there)
                log.debug("background preview build failed for %s", getattr(spec, "id", "?"), exc_info=True)
                r = None
            self.done.emit(t, r)
            time.sleep(0.03)                             # let the UI thread breathe between characters

    @staticmethod
    def _build(spec, mode: str, quality: str):
        """The production renderer only (its GPU context exists already); anything else -> None."""
        if mode not in ("production", "illustrated") or spec.skeleton == "sprite":
            return None
        if mode == "illustrated":
            from ..companion.illustrated import IllustratedRenderer, pack_dir
            folder = pack_dir(spec.id)
            return IllustratedRenderer(spec, folder) if folder is not None else None
        from ..companion.render3d_gl import GLSkinRaster
        if GLSkinRaster._shared is None and not GLSkinRaster._failed:
            return None                                  # no context yet: never create one off the UI thread
        from ..companion.render_mesh import MeshRenderer3D
        from ..companion.renderer import load_asset
        from ..companion.rig import Rig, load_skeleton
        kind = "biped" if spec.skeleton == "biped" else "quadruped"
        asset = load_asset(spec, Rig(load_skeleton(spec.id, spec.species, kind)))
        if not asset.is_mesh:
            return None
        return MeshRenderer3D(spec, asset, quality)

    def _deliver(self, ticket: int, renderer) -> None:
        cb = self._callbacks.pop(ticket, None)
        if cb is None:
            return
        try:
            cb(renderer)
        except RuntimeError:                             # the widget was deleted meanwhile
            pass
