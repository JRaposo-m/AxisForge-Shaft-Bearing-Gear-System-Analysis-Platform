"""
axisforge/solvers/machine_elements/bearings/ISO_16281/Ball_Bearing/__init__.py

Public surface for the point-contact (ball bearing) ISO/TS 16281 solver
package. Mirrors the split kept by the sibling Roller_Bearing package:
solving lives in ball_bearing.py, the LOCAL library — both the result shape
a solve produces (BallLoadDistributionResult) and the registry that holds
one per bearing label (BallLoadDistributionLibrary) — lives together in
ball_bearing_results.py, and everything that consumes an already-solved
result lives in ball_bearing_postprocessing.py.
"""
from .ball_bearing import (
    ISO16281BallSolver,
    RollingElementCapacity,
    debug_radial_capacity,
)
from .ball_bearing_results import (
    BallLoadDistributionResult,
    BallLoadDistributionLibrary,
)
from .ball_bearing_postprocessing import (
    Q_j,
    phi_j_global,
    contact_distribution,
    bearing_stiffness,
    BallBearingStiffness,
    DynamicEquivalentRollingElementLoad,
)

__all__ = [
    "ISO16281BallSolver",
    "RollingElementCapacity",
    "debug_radial_capacity",
    "BallLoadDistributionResult",
    "BallLoadDistributionLibrary",
    "Q_j",
    "phi_j_global",
    "contact_distribution",
    "bearing_stiffness",
    "BallBearingStiffness",
    "DynamicEquivalentRollingElementLoad",
]