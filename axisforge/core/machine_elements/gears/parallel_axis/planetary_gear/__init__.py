# families/roller_bearing/thrust/subtypes/__init__.py  (LEAF)
"""
axisforge/core/machine_elements/gears/parallel_axis/planetary_gear/__init__.py

Still need to improve the organization of the planetary gear train meshing module.  
The current structure needs improvement to better organize the planetary gear train meshing module. 
The current structure is not optimal and requires reorganization to enhance clarity and maintainability.

"""
from __future__ import annotations
from typing import TYPE_CHECKING

__all__ = ["PlanetaryGearTrainMeshing"]

_LAZY = {
    "PlanetaryGearTrainMeshing": ".planetary_gear_meshing"
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
    from .planetary_gear_meshing import PlanetaryGearTrainMeshing
