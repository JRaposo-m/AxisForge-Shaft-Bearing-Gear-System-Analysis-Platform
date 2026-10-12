# axisforge/core/machine_elements/gears/parallel_axis/__init__.py
"""
Single import point of the parallel-axis gear subsystem (eager re-export).

Exposes the gear definitions (``SpurHelicalGear``, ``InternalGear``), the pair
objects (``SpurHelicalGearMeshing``, ``InternalGearMeshing``) and the planetary
train (``PlanetaryGearTrainMeshing``). The folders ``gear_properties/``,
``gear_meshing/`` and ``planetary_gear/`` have no ``__init__.py`` of their own
(namespace packages); every import is made here.
"""
from __future__ import annotations

from .gear_properties.spur_helical_gear import SpurHelicalGear
from .gear_properties.internal_gear import InternalGear
from .gear_meshing.spurhelical_meshing import SpurHelicalGearMeshing
from .gear_meshing.internal_meshing import InternalGearMeshing
from .planetary_gear.planetary_gear_meshing import PlanetaryGearTrainMeshing

__all__ = [
    "SpurHelicalGear", "InternalGear",
    "SpurHelicalGearMeshing", "InternalGearMeshing",
    "PlanetaryGearTrainMeshing",
]