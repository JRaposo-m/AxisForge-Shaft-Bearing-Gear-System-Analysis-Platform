"""
axisforge/solvers/machine_elements/bearings/ISO_16281/Ball_Bearing/__init__.py

Public surface for the point-contact (ball bearing) ISO/TS 16281 solver
package. Mirrors the split kept by the sibling Roller_Bearing package:
solving lives in ball_bearing_solver.py, the LOCAL library -- both the
result shapes a solve produces (BallLoadDistributionResult, the per-row
raw result; BallBearingResult, the unified per-bearing container, single-
row or multi-row through the SAME type) and the registry that holds one
BallBearingResult per bearing label (BallLoadDistributionLibrary) -- lives
together in ball_bearing_results.py, and everything that consumes an
already-solved result lives in ball_bearing_postprocessing.py.

UPDATED, this turn -- reflects the single/multi-row unification
-------------------------------------------------------------------
Two things this file used to export no longer exist, and are removed here:

  MultiRowBallLoadDistributionResult (used to live in
  ball_bearing_multirow_solver.py) -- retired. ISO16281MultiRowBallSolver.
  solve_bearing() now returns a BallBearingResult (via its .multirow()
  classmethod) instead, the exact same type ISO16281BallSolver.solve()
  returns (via .single()). Import BallBearingResult below to type either
  one -- there is no longer a separate multi-row-only result class to
  import.

  ball_bearing_multirow_postprocessing.py and its one function,
  multirow_dynamic_equivalent_load() -- retired outright, the file is
  gone. Its job is now DynamicEquivalentRollingElementLoad.
  from_bearing_result() in ball_bearing_postprocessing.py, which handles
  single-row (1 row) and multi-row (i rows) through the exact same code
  path -- no separate multi-row entry point needed anymore.

Multi-row orchestration (ISO16281MultiRowBallSolver, in
ball_bearing_multirow_solver.py) is still re-exported here, alongside the
single-row solver it composes against -- but it is NOT part of the
RollingBearingSolver dispatch table (_SOLVER_MAP / _POSTPROC_MAP in
rolling_bearing_solver.py): that orchestrator has not yet been taught to
route a multi-row thrust bearing to this solver instead of the single-row
one (see rolling_bearing_solver.py's own known-breakage notes). It is
called directly today by whoever needs a multi-row thrust bearing's load
distribution, not routed through the orchestrator. See
ball_bearing_multirow_solver.py's own module docstring for the physical
idealization (co-located rows) and method (iterative load-split fixed
point, converging every row to the same rigid-ring delta_r/delta_a).

RollingElementCapacity is NOT re-exported here -- capacity (Q_ci/Q_ce,
Cr/Ca) is owned by core/.../families/ball/{radial,thrust}/; call it via
bearing.family.per_element_dynamic_capacity(bearing, ...).
debug_radial_capacity() (manual cross-check, not production) is the only
capacity-adjacent thing still exported from this package.
"""
from .ball_bearing_solver import (
    ISO16281BallSolver,
    debug_radial_capacity,
)
from .ball_bearing_multirow_solver import (
    ISO16281MultiRowBallSolver,
)
from .ball_bearing_results import (
    BallLoadDistributionResult,
    BallBearingResult,
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
    "BallLoadDistributionResult",
    "BallBearingResult",
    "BallLoadDistributionLibrary",
    "Q_j",
    "phi_j_global",
    "contact_distribution",
    "bearing_stiffness",
    "BallBearingStiffness",
    "DynamicEquivalentRollingElementLoad",
]