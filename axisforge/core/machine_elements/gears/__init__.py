# axisforge/core/machine_elements/gears/__init__.py
"""
Single import point of the gear subsystem (eager re-export, as in bearings).

Exposes the family contract and the two assembled objects (``GearFamily``,
``register_gear_family``, ``Gear``, ``GearPair``), the record of the forces of
a mesh (``MeshForces``), the cylindrical family for
external and internal involute gears (``CylindricalGearData``,
``CylindricalFamily``), the manufacturing inputs of a cylindrical gear
(``CylindricalGearManufacturingData``, ``GeneratingRackProfile``), the gear
as manufactured (``ManufacturedCylindricalGear``) and the
planetary train (``PlanetaryGearTrainMeshing``, still in
``parallel_axis/planetary_gear/``, a folder without ``__init__.py``).
"""
from __future__ import annotations

from .base import GearFamily, register_gear_family, Gear, GearPair, MeshForces
from .cylindrical import (
    CylindricalGearData, CylindricalFamily, CylindricalGearManufacturingData, GeneratingRackProfile,
    ManufacturedCylindricalGear,
)
from .parallel_axis.planetary_gear.planetary_gear_meshing import PlanetaryGearTrainMeshing

__all__ = [
    "GearFamily", "register_gear_family", "Gear", "GearPair", "MeshForces",
    "CylindricalGearData", "CylindricalFamily", "CylindricalGearManufacturingData", "GeneratingRackProfile",
    "ManufacturedCylindricalGear",
    "PlanetaryGearTrainMeshing",
]
