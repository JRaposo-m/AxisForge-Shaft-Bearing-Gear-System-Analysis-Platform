"""
axisforge/solvers/machine_elements/bearings/ISO_16281/Roller_Bearing/__init__.py

Public surface for the line-contact (radial cylindrical roller bearing)
ISO/TS 16281 Sec 5.2 lamina-model solver package. Mirrors the split kept by
the sibling Ball_Bearing package: solving lives in roller_bearing_solver.py,
the LOCAL library -- both the result shape a solve produces
(RollerLoadDistributionResult) and the registry that holds one per bearing
label (RollerLoadDistributionLibrary) -- lives together in
roller_bearing_results.py, and everything that consumes an already-solved
result lives in roller_bearing_postprocessing.py.

RollerElementCapacity is NOT re-exported here anymore -- capacity
(Q_ci/Q_ce, per-lamina q_ci/q_ce, Cr/Ca) is owned by core/.../families/
roller/{radial,thrust}/; call it via
bearing.family.per_element_dynamic_capacity(bearing, ...) and
bearing.family.per_lamina_dynamic_capacity(bearing, Q_ci, Q_ce).
debug_radial_capacity() (manual cross-check, not production) is the only
capacity-adjacent thing still exported from this package.

roller_profile() (eq.42-44) is likewise not re-exported here as a free
function -- it lives as _reference_roller_profile() on each concrete
subtype (CylindricalRollerFamily, ThrustCylindricalRollerFamily,
ThrustNeedleRollerFamily), cached onto bearing.P_xk at assembly.
"""
from .roller_bearing_solver import (
    ISO16281RollerSolver,
    debug_radial_capacity,
)
from .roller_bearing_results import (
    RollerLoadDistributionResult,
    RollerLoadDistributionLibrary,
)
from .roller_bearing_postprocessing import (
    Q_j,
    phi_j_global,
    contact_distribution,
    lamina_distribution,
    bearing_stiffness,
    RollerBearingStiffness,
    stress_riser_factor,
    LaminaDynamicEquivalentLoad,
)

__all__ = [
    "ISO16281RollerSolver",
    "RollerLoadDistributionResult",
    "RollerLoadDistributionLibrary",
    "debug_radial_capacity",
    "Q_j",
    "phi_j_global",
    "contact_distribution",
    "lamina_distribution",
    "bearing_stiffness",
    "RollerBearingStiffness",
    "stress_riser_factor",
    "LaminaDynamicEquivalentLoad",
]