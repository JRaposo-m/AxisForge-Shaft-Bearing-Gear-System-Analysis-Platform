# solvers/shaft/__init__.py
"""
Shaft analysis solvers.

Public interface — import from here, never from sub-modules directly:
    from solvers.shaft import StaticsSolver, StaticFailureSolver, StressSolver

Phase 4+:  DeflectionSolver, CriticalSpeedSolver (not yet implemented).
"""

from solvers.shaft.statics import StaticsSolver
from solvers.shaft.static_failure import StaticFailureSolver
from solvers.shaft.stress import StressSolver
from solvers.shaft.utils import (
    # Marin factor helpers — exposed for testing and transparency
    ka_surface_finish,
    kb_size,
    kc_load,
    kd_temperature,
    ke_reliability,
    ke_reliability_from_z,
    endurance_limit_corrected,
    # Peterson / notch sensitivity helpers
    kt_shoulder_bending,
    kt_shoulder_torsion,
    neuber_constant_sqrt_a,
    notch_sensitivity,
    kf_from_kt,
)

__all__ = [
    # Solver classes
    "StaticsSolver",
    "StaticFailureSolver",
    "StressSolver",
    # Marin factors
    "ka_surface_finish",
    "kb_size",
    "kc_load",
    "kd_temperature",
    "ke_reliability",
    "ke_reliability_from_z",
    "endurance_limit_corrected",
    # Notch / stress concentration
    "kt_shoulder_bending",
    "kt_shoulder_torsion",
    "neuber_constant_sqrt_a",
    "notch_sensitivity",
    "kf_from_kt",
]