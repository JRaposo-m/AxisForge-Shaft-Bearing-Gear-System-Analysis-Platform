"""
Mechanical systems: assemblies of shafts, bearings and gears.

Re-exports the single-shaft container (``ShaftSystem``, ``GearElement``) of ``shaft_system`` and
the written power path of ``transmission``: ``PairStage``, ``Transmission``,
``ResolvedTransmission`` and ``mesh_forces_to_shaft_loads``. Only parallel-axis systems of
cylindrical gears exist.
"""
from __future__ import annotations

from .shaft_system import GearElement, ShaftSystem
from .transmission import PairStage, ResolvedTransmission, Transmission, mesh_forces_to_shaft_loads

__all__ = ["GearElement", "ShaftSystem",
           "mesh_forces_to_shaft_loads", "PairStage", "Transmission", "ResolvedTransmission"]
