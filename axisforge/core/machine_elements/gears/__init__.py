# axisforge/core/machine_elements/gears/__init__.py
"""
Single import point of the gear subsystem (eager re-export, as in bearings).

Exposes the family contract and the two assembled objects (``GearFamily``,
``register_gear_family``, ``Gear``, ``GearPair``), the cylindrical family for
external and internal involute gears (``CylindricalGearData``,
``CylindricalFamily``) and the planetary train (``PlanetaryGearTrainMeshing``,
still in ``parallel_axis/planetary_gear/``, a folder without ``__init__.py``).
"""
from __future__ import annotations

from .base import GearFamily, register_gear_family, Gear, GearPair
from .cylindrical import CylindricalGearData, CylindricalFamily
from .parallel_axis.planetary_gear.planetary_gear_meshing import PlanetaryGearTrainMeshing

__all__ = [
    "GearFamily", "register_gear_family", "Gear", "GearPair",
    "CylindricalGearData", "CylindricalFamily",
    "PlanetaryGearTrainMeshing",
]
