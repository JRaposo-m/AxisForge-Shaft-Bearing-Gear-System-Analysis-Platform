# families/__init__.py  (ROLLUP de ball_bearing + roller_bearing — a base para a seleção)
"""
axisforge/core/machine_elements/gears/parallel_axis/__init__.py

Lazy-loaded: nunca importa os .py finais diretamente, reencaminha
sempre para o __init__.py do subpacote (rollup em cascata).
"""
from __future__ import annotations
from typing import TYPE_CHECKING

__all__ = [
    "SpurHelicalGear", "InternalGear",
    "SpurHelicalGearMeshing","InternalGearMeshing",
    "PlanetaryGearTrainMeshing",
]

_LAZY = {
    "SpurHelicalGear": ".gear_properties", 
    "InternalGear": ".gear_properties",
    "SpurHelicalGearMeshing": ".gear_meshing",
    "InternalGearMeshing": ".gear_meshing",
    "PlanetaryGearTrainMeshing": ".planetary_gear",

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
    from .gear_properties import SpurHelicalGear, InternalGear
    from .gear_meshing import SpurHelicalGearMeshing, InternalGearMeshing
    from .planetary_gear import PlanetaryGearTrainMeshing