# families/roller_bearing/thrust/subtypes/__init__.py  (LEAF)
"""
axisforge/core/mechanical_system/parallel_axis/spur_helical/__init__.py

"""
from __future__ import annotations
from typing import TYPE_CHECKING

__all__ = ["SpurHelicalMeshLink", "SpurHelicalGearSystem",
           "GearElement", "ShaftSystem"]

_LAZY = {
    "SpurHelicalMeshLink": ".gear_system",
    "SpurHelicalGearSystem": ".gear_system",
    "GearElement": ".shaft_system",
    "ShaftSystem": ".shaft_system",
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
    from .gear_system import SpurHelicalMeshLink, SpurHelicalGearSystem
    from .shaft_system import GearElement, ShaftSystem
