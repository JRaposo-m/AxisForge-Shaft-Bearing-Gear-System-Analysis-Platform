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

UPDATED, this turn -- shared-displacement multi-row solver re-exported
-------------------------------------------------------------------
ISO16281MultiRowBallSolverSharedDisplacement (in
ball_bearing_multirow_solver_shared_displacement.py) is now re-exported
here too, alongside ISO16281MultiRowBallSolver. Same status as that
solver already had before this turn: NOT part of RollingBearingSolver's
dispatch table (_SOLVER_MAP / _POSTPROC_MAP in rolling_bearing_solver.py)
-- if anything, LESS wired in than ISO16281MultiRowBallSolver, since it
was written explicitly as a side-by-side comparison against the fraction-
based solver, not as a candidate replacement in the dispatch path (see its
own module docstring's "Formulation" and "Encapsulation note" sections for
the full trade-off writeup, and
design_bearing_combination_comparison.py's "SOLVER COMPARISON" section for
the actual side-by-side numeric comparison). It returns the exact same
BallBearingResult type as ISO16281MultiRowBallSolver (via the same
.multirow() classmethod), so nothing downstream needs a new type import to
consume either solver's output. Call it directly, the same way
ISO16281MultiRowBallSolver is called directly today (see that solver's own
note above about not going through the orchestrator).

Two things this file used to export no longer exist, and are removed here
(unchanged from before this turn, restated for completeness):

  MultiRowBallLoadDistributionResult (used to live in
  ball_bearing_multirow_solver.py) -- retired. ISO16281MultiRowBallSolver.
  solve_bearing() now returns a BallBearingResult (via its .multirow()
  classmethod) instead, the exact same type ISO16281BallSolver.solve()
  returns (via .single()). Import BallBearingResult below to type any of
  the three call paths (single-row solve, ISO16281MultiRowBallSolver,
  ISO16281MultiRowBallSolverSharedDisplacement) -- there is no longer a
  separate multi-row-only result class to import, and the new solver never
  needed one either.

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
from .ball_bearing_multirow_solver_shared_displacement import (
    ISO16281MultiRowBallSolverSharedDisplacement,
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
    "ISO16281MultiRowBallSolverSharedDisplacement",
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