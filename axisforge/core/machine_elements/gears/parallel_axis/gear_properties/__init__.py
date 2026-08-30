# families/roller_bearing/thrust/subtypes/__init__.py  (LEAF)
"""
axisforge/core/machine_elements/gears/parallel_axis/gear_properties/__init__.py

"""
from __future__ import annotations
from typing import TYPE_CHECKING

__all__ = ["SpurHelicalGear","InternalGear"]

_LAZY = {
    "SpurHelicalGear": ".spur_helical_gear",
    "InternalGear": ".internal_gear",
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
    from .spur_helical_gear import SpurHelicalGear
    from .internal_gear import InternalGear
