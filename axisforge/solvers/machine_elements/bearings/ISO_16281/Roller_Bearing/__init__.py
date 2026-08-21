"""
axisforge/solvers/machine_elements/bearings/ISO_16281/Roller_Bearing/__init__.py

Public surface for the line-contact (radial cylindrical roller bearing)
ISO/TS 16281 Sec 5.2 lamina-model solver package. Mirrors the split kept by
the sibling Ball_Bearing package: solving lives in roller_bearing_solver.py,
the LOCAL library -- both the result shapes a solve produces
(RollerLoadDistributionResult, the per-row raw result; RollerBearingResult,
the unified per-bearing container) and the registry that holds one
RollerBearingResult per bearing label (RollerLoadDistributionLibrary) --
lives together in roller_bearing_results.py, and everything that consumes
an already-solved result lives in roller_bearing_postprocessing.py.

UPDATED, this turn -- RollerBearingResult now exported
----------------------------------------------------------
RollerBearingResult was missing from this file's exports -- added below,
alongside RollerLoadDistributionResult. It is the container type
ISO16281RollerSolver.solve() actually returns (wrapped via .single()) and
every roller_bearing_postprocessing.py function actually takes; a caller
importing only RollerLoadDistributionResult from this package had no way
to type-hint or construct the container itself.

RollerBearingResult.rows is length 1 for CylindricalRollerFamily (radial
NU/N-type, the only family in core/ today) -- a .multirow() classmethod
exists on the class (mirroring BallBearingResult) but nothing produces
that shape yet: there is no ISO16281MultiRowRollerSolver / multi-row
roller family exported here, unlike the ball side's
ISO16281MultiRowBallSolver, because a thrust roller family (the case that
would actually need it, mirroring MultiRowThrustBallFamily on the ball
side) does not exist in core/ yet. See roller_bearing_results.py's module
docstring for the full reasoning -- add ISO16281MultiRowRollerSolver's
export here once that solver is written.

RollerElementCapacity is NOT re-exported here -- capacity (Q_ci/Q_ce,
per-lamina q_ci/q_ce, Cr/Ca) is owned by core/.../families/
roller/{radial,thrust}/; call it via
bearing.family.per_element_dynamic_capacity(bearing, ...) and
bearing.family.per_lamina_dynamic_capacity(bearing, Q_ci, Q_ce).
debug_radial_capacity() (manual cross-check, not production) is the only
capacity-adjacent thing still exported from this package.

roller_profile() (eq.42-44) is likewise not re-exported here as a free
function -- it lives as _reference_roller_profile() on each concrete
subtype (CylindricalRollerFamily, and a future ThrustCylindricalRollerFamily/
ThrustNeedleRollerFamily), cached onto bearing.P_xk at assembly.
"""
from .roller_bearing_solver import (
    ISO16281RollerSolver,
    debug_radial_capacity,
)
from .roller_bearing_results import (
    RollerLoadDistributionResult,
    RollerBearingResult,
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
    "debug_radial_capacity",
    "RollerLoadDistributionResult",
    "RollerBearingResult",
    "RollerLoadDistributionLibrary",
    "Q_j",
    "phi_j_global",
    "contact_distribution",
    "lamina_distribution",
    "bearing_stiffness",
    "RollerBearingStiffness",
    "stress_riser_factor",
    "LaminaDynamicEquivalentLoad",
]