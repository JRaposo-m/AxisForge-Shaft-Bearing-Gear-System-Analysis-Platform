"""
axisforge/solvers/machine_elements/bearings/ISO_16281/Roller_Bearing/__init__.py

Public surface for the line-contact (radial cylindrical roller bearing)
ISO/TS 16281 §5.2 lamina-model solver package. Mirrors the split kept by
the sibling Ball_Bearing package: solving lives in
roller_bearing.py, everything that consumes an already-solved
result lives in roller_bearing_postprocessing.py.

roller_profile() (eq.42-44) is NOT re-exported here as a free function —
it lives as a static method on ISO16281RollerSolver (call it as
ISO16281RollerSolver.roller_profile(...)), since the solver's own
_elements() is its only real consumer. Lamina midpoints (x_k) are not
computed by this package at all — they must already be on the bearing
(see RollerBearingGeometry.setup() / CylindricalRollerBearing).
"""
from .roller_bearing import (
    ISO16281RollerSolver,
    RollerLoadDistributionResult,
    RollerElementCapacity,
    debug_radial_capacity,
)
from .roller_bearing_postprocessing import (
    Q_j,
    phi_j_global,
    contact_distribution,
    lamina_distribution,
    bearing_stiffness,
    stress_riser_factor,
    LaminaDynamicEquivalentLoad,
)

__all__ = [
    "ISO16281RollerSolver",
    "RollerLoadDistributionResult",
    "RollerElementCapacity",
    "debug_radial_capacity",
    "Q_j",
    "phi_j_global",
    "contact_distribution",
    "lamina_distribution",
    "bearing_stiffness",
    "stress_riser_factor",
    "LaminaDynamicEquivalentLoad",
]