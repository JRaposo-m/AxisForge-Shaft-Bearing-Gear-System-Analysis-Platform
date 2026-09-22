# axisforge/core/mechanical_system/parallel_axis/__init__.py
"""
axisforge/core/mechanical_system/parallel_axis/__init__.py

Ponto único de import para o subsistema parallel_axis de
mechanical_system -- eager, sem lazy loading. spur_helical/ deixa de
ter __init__.py próprio (namespace package); toda a lógica de import
vive só aqui.
"""
from __future__ import annotations

from .spur_helical.gear_system import SpurHelicalMeshLink, SpurHelicalGearSystem
from .spur_helical.shaft_system import GearElement, ShaftSystem

__all__ = ["SpurHelicalMeshLink", "SpurHelicalGearSystem", "GearElement", "ShaftSystem"]