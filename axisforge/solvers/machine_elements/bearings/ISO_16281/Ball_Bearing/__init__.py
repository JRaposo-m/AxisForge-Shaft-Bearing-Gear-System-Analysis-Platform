"""
axisforge/solvers/machine_elements/bearings/ISO_16281/Ball_Bearing/__init__.py

Public surface for the point-contact (ball bearing) ISO/TS 16281 solver
package. Mirrors the split kept by the sibling Roller_Bearing package:
solving lives in ball_bearing_solver.py, the LOCAL library -- both the
result shape a solve produces (BallLoadDistributionResult) and the
registry that holds one per bearing label (BallLoadDistributionLibrary) --
lives together in ball_bearing_results.py, and everything that consumes an
already-solved result lives in ball_bearing_postprocessing.py.

Multi-row orchestration (ISO16281MultiRowBallSolver,
MultiRowBallLoadDistributionResult, in ball_bearing_multirow_solver.py) is
re-exported here too, alongside the single-row solver it composes -- but it
is NOT part of the RollingBearingSolver dispatch table (_SOLVER_MAP /
_POSTPROC_MAP in rolling_bearing_solver.py): its result shape
(MultiRowBallLoadDistributionResult, a list of per-row BallLoadDistribution
Result plus the converged split fractions) is not interchangeable with a
single BallLoadDistributionResult, so it is called directly by whoever
needs a multi-row thrust bearing's load distribution, not routed through
the orchestrator. See ball_bearing_multirow_solver.py's own module
docstring for the physical idealization (co-located rows) and method
(iterative load-split fixed point, converging every row to the same
rigid-ring delta_r/delta_a).

Multi-row POST-processing (multirow_dynamic_equivalent_load, in
ball_bearing_multirow_postprocessing.py) follows the exact same pattern:
it does not reimplement DynamicEquivalentRollingElementLoad's ISO/TS 16281
eq.(25)-(28) math, it calls that existing single-row-shaped function once
per row (i of them, any i >= 2 -- not hardcoded to 2). Per-row Q_ei/Q_ee
feed a raceway-level fatigue analysis; they are never merged across rows
or scaled by a row-count factor -- see that file's own module docstring
for why (short version: no "x2" anywhere, and a whole-bearing L10 estimate
uses the standard P = X*Fr + Y*Fa against the i**0.7-scaled multi-row
capacity instead, a completely separate calculation).

RollingElementCapacity is NOT re-exported here anymore -- capacity
(Q_ci/Q_ce, Cr/Ca) is owned by core/.../families/ball/{radial,thrust}/;
call it via bearing.family.per_element_dynamic_capacity(bearing, ...).
debug_radial_capacity() (manual cross-check, not production) is the only
capacity-adjacent thing still exported from this package.
"""
from .ball_bearing_solver import (
    ISO16281BallSolver,
    debug_radial_capacity,
)
from .ball_bearing_multirow_solver import (
    ISO16281MultiRowBallSolver,
    MultiRowBallLoadDistributionResult,
)
from .ball_bearing_multirow_postprocessing import (
    multirow_dynamic_equivalent_load,
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
    "debug_radial_capacity",
    "ISO16281MultiRowBallSolver",
    "MultiRowBallLoadDistributionResult",
    "multirow_dynamic_equivalent_load",
    "BallLoadDistributionResult",
    "BallLoadDistributionLibrary",
    "Q_j",
    "phi_j_global",
    "contact_distribution",
    "bearing_stiffness",
    "BallBearingStiffness",
    "DynamicEquivalentRollingElementLoad",
]