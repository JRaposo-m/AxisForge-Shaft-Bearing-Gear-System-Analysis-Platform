"""
Mechanical systems: assemblies of shafts, bearings and gears.

Re-exports the single-shaft container (``ShaftSystem``, ``GearElement``) of ``shaft_system`` and
the multi-shaft assembler (``SpurHelicalGearSystem``, ``SpurHelicalMeshLink``) of ``gear_system``.
Only parallel-axis (spur and helical) systems exist.
"""
from __future__ import annotations

from .gear_system import SpurHelicalGearSystem, SpurHelicalMeshLink
from .shaft_system import GearElement, ShaftSystem

__all__ = ["SpurHelicalMeshLink", "SpurHelicalGearSystem", "GearElement", "ShaftSystem"]
