# solvers/gears/__init__.py
"""
Gear analysis solvers.

Public interface — import from here, never from sub-modules directly:
    from solvers.gears import GearSolver

Phase 2+:  GearStrengthSolver (ISO 6336-2/3), ProfileShiftSolver.
Phase 4+:  MicropittingSolver.
"""

from solvers.gears.geometry import GearSolver

__all__ = [
    "GearSolver",
]
