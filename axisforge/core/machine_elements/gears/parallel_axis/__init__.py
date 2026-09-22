# axisforge/core/machine_elements/gears/parallel_axis/__init__.py
"""
axisforge/core/machine_elements/gears/parallel_axis/__init__.py

Ponto único de import para o subsistema parallel_axis -- estilo
bearings/, eager, sem lazy loading. gear_properties/, gear_meshing/ e
planetary_gear/ deixam de ter __init__.py próprio (namespace packages);
toda a lógica de import vive só aqui.
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