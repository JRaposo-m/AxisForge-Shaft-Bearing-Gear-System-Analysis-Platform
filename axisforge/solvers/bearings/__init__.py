# solvers/bearings/__init__.py
"""
Rolling bearing analysis solvers.

Public interface — import from here, never from sub-modules directly:
    from solvers.bearings import BearingLifeSolver

Phase 3+:  FrictionSolver (SKF friction model).
Phase 4+:  ArrangementSolver, MisalignmentSolver.
"""

from solvers.bearings.life import BearingLifeSolver

__all__ = [
    "BearingLifeSolver",
]
