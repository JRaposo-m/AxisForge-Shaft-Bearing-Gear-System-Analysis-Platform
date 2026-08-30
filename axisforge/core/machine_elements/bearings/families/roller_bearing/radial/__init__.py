# families/ball_bearing/radial/__init__.py  (ROLLUP, só de .subtypes)
"""
axisforge/core/machine_elements/bearings/families/ball_bearing/radial/__init__.py

Roll-up de subtypes/ -- functions/ fica de fora de proposito (ver o
docstring de functions/__init__.py).
"""
from __future__ import annotations
from typing import TYPE_CHECKING

__all__ = ["CylindricalRollerFamily"]

_LAZY = {
    "CylindricalRollerFamily": ".subtypes",
}

def __getattr__(name: str):
    if name in _LAZY:
        import importlib
        module = importlib.import_module(_LAZY[name], __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY.keys()))

if TYPE_CHECKING:  # pragma: no cover
    from .subtypes import CylindricalRollerFamily