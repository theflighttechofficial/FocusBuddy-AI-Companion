"""Ground abstraction for foot contact.

The desktop is a 2D world: the character stands on the taskbar, a window's top edge, a screen
edge or (later) a virtual platform. The rig and the clip IK only need the local ground height
under each foot (renderer units, +y is down, 0 = the feet line of the character's box), so new
surface types plug in here without touching the rig.
"""

from __future__ import annotations

from dataclasses import dataclass

KINDS = ("taskbar", "window_ledge", "screen_edge", "platform", "air")


@dataclass
class GroundSurface:
    kind: str = "taskbar"
    height: float = 0.0          # local ground height (internal units, + = lower)
    slope: float = 0.0           # dy per unit x (a tilted platform; 0 for windows / taskbar)
    left: float = -1e9           # local x extent of the surface (feet beyond it are unsupported)
    right: float = 1e9

    def height_at(self, x: float) -> float:
        return self.height + self.slope * x

    def supports(self, x: float) -> bool:
        return self.kind != "air" and self.left <= x <= self.right


def from_behavior(behavior, char_px: float, units: float = 140.0) -> GroundSurface:
    """The surface the behaviour says the character is on, in the renderer's local units."""
    surf = getattr(behavior, "surface", None)
    mover = getattr(behavior, "mover", None)
    if mover is not None and getattr(mover, "airborne", False):
        kind = "air"
    elif surf is None:
        kind = "taskbar"
    else:
        kind = "window_ledge"
    g = GroundSurface(kind)
    if surf is not None and char_px > 0:
        k = units / char_px
        x = float(getattr(behavior, "x", 0.0))
        g.left = (float(getattr(surf, "left", x - 1e6)) - x) * k
        g.right = (float(getattr(surf, "right", x + 1e6)) - x) * k
    return g
