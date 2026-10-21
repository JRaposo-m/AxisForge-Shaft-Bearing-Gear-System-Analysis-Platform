# axisforge/core/machine_elements/gears/__init__.py
"""
Single import point of the gear subsystem (eager re-export, as in bearings).

Exposes:

- the family contract and the assembled objects: ``GearFamily``, ``register_gear_family``, ``Gear``,
  ``GearPair``, ``MeshForces``;
- the ideal cylindrical family: ``CylindricalGearData``, ``CylindricalFamily``;
- the manufactured cylindrical family (a subclass of the ideal one): ``CylindricalGearManufacturingData``,
  ``GeneratingRackProfile``, ``GeneratingPinionCutter`` (placeholder), ``InspectionLimits``,
  ``ManufacturedCylindricalFamily``;
- the planetary train ``PlanetaryGearTrainMeshing`` (still in ``parallel_axis/planetary_gear/``,
  a folder without ``__init__.py``).
"""
from __future__ import annotations

from .base import GearFamily, register_gear_family, Gear, GearPair, MeshForces
from .cylindrical import (
    CylindricalGearData, CylindricalFamily, CylindricalGearManufacturingData, GeneratingRackProfile,
    GeneratingPinionCutter, InspectionLimits, ManufacturedCylindricalFamily,
)
from .parallel_axis.planetary_gear.planetary_gear_meshing import PlanetaryGearTrainMeshing

__all__ = [
    "GearFamily", "register_gear_family", "Gear", "GearPair", "MeshForces",
    "CylindricalGearData", "CylindricalFamily", "CylindricalGearManufacturingData", "GeneratingRackProfile",
    "GeneratingPinionCutter", "InspectionLimits", "ManufacturedCylindricalFamily",
    "PlanetaryGearTrainMeshing",
]
