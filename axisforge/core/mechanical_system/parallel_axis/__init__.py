"""
Parallel-axis mechanical systems (spur and helical gear trains).

Re-exports the single-shaft container (``ShaftSystem``, ``GearElement``) and
the multi-shaft assembler (``SpurHelicalGearSystem``, ``SpurHelicalMeshLink``)
defined in ``spur_helical/``.
"""
from __future__ import annotations

from .spur_helical.gear_system import SpurHelicalMeshLink, SpurHelicalGearSystem
from .spur_helical.shaft_system import GearElement, ShaftSystem

__all__ = ["SpurHelicalMeshLink", "SpurHelicalGearSystem", "GearElement", "ShaftSystem"]