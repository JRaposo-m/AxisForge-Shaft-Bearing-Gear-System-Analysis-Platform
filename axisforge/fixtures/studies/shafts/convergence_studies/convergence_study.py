"""
axisforge/fixtures/studies/shafts/convergence_studies/convergence_study.py

Studies stage: runs a mesh convergence study (MeshConvergenceStudy) for
every ShaftSystem in a SpurHelicalGearSystem's system.shafts, publishing
every MeshRefinementResult into one ConvergenceResultsLibrary.

This is convergence_studies/'s "does the work" fixture -- same role
comparison_study.py plays for fem_studies/, and same role fem_simple.py's
solve_system() plays for a plain resolution: object assembly already
happened in Construction, this module performs the real work (here, the
mesh convergence study itself), not just wiring.

Guarded on construction exactly like fem_simple.solve_system() and
comparison_study.run_comparison() -- run_convergence() takes the
ConstructionCapabilities that built `system` as a second required
argument and raises ValueError immediately unless
construction.has_capability("systems.parallel_axis_linear"). Same
reasoning as those two: without a resolved system (gear-mesh loads +
shaft positions already present), the FEM baseline this study refines
around would be solved with zero gear-mesh loads, and every GCI number
that came out of it would be meaningless.

One fresh RigidBearingFEMSolver AND one fresh MeshConvergenceStudy per
shaft -- same "one solver instance per shaft" rule fem_simple.py
documents for itself (solve() overwrites; re-using one instance across
shafts would silently clobber the previous shaft's global solve, and
MeshConvergenceStudy holds a reference to whichever global_solver it
was built with, so reusing the MeshConvergenceStudy instance itself
across shafts would leave it pointed at the wrong shaft's baseline).

theory / distribute_gear_labels are applied UNIFORMLY to every shaft's
global solver, same convention fem_simple.solve_system() already uses
for its own theory / distribute_gear_labels kwargs -- defaults match
the only configuration that exists today (RigidBearingFEMSolver(
theory="timoshenko")). There is no constraint_bearing kwarg here
either, for the same reason fem_simple.py's own docstring gives: rigid
supports are this solver's identity, not a mode it selects between.

global_solver.solve(ss) is called WITHOUT extra_mandatory -- this is
deliberately the coarse/baseline global solve the convergence study
refines *around*; passing pre-refined nodes into it would defeat the
point of measuring convergence from a genuine baseline. Per-shaft
extra_mandatory (the refinement THIS study discovers, e.g. via
MeshRefinementResult.all_extra_nodes) is consumed downstream by
fem_simple.solve_system()'s own extra_mandatory dict, once a caller
decides to re-solve the final resolution mesh with it -- not built
here, this module only produces the MeshRefinementResult that would
feed that decision.

intervals_from_shaft_study() builds intervals per shaft (gear face
widths, bearing extents, distributed-load spans -- see
MeshConvergenceStudy.intervals_from_shaft_system()'s own docstring for
exactly which regions qualify) fresh for every shaft, since two shafts
in the same system practically always have different gear positions,
face widths and bearing spans. Its second return value (label
ordering) is not needed here -- MeshRefinementResult keys its
per_load dict by the same labels intervals_from_shaft_system() already
assigns, so nothing downstream needs a separate ordering list to line
results back up; run_convergence() only forwards the interval list
itself to MeshConvergenceStudy.run().

print_convergence()/write_convergence_report() (the outputs/ content-
block module mirroring comparison_report.py's shaft_comparison_block())
do not exist yet -- MeshRefinementResult.print_report() was removed on
its move to results/ (see convergence_results.py's own docstring) and
its replacement has not been built. This module deliberately does not
reintroduce ad-hoc printing to fill that gap: the loop below only
solves and stores, matching fem_simple.solve_system()'s own shape
(no printing there either -- callers print from the library they get
back, e.g. check_resolution_fem.py's own per-shaft summary lines).

Dependency (solvers + fixtures, read-only access -- no core modification):
  axisforge.solvers.machine_elements.shaft.fem_solvers.rigid_bearing
      RigidBearingFEMSolver -- builds each shaft's baseline global solve.
  axisforge.solvers.mesh.convergence_solver
      MeshConvergenceStudy -- does the actual refinement/GCI work.
  axisforge.fixtures.studies.shafts.convergence_studies.convergence_library
      ConvergenceResultsLibrary -- the result SHAPE this module publishes into.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from axisforge.solvers.machine_elements.shaft.fem_solvers.rigid_bearing import (
    RigidBearingFEMSolver,
)
from axisforge.solvers.mesh.convergence_solver import MeshConvergenceStudy
from axisforge.fixtures.studies.shafts.convergence_studies.convergence_library import (
    ConvergenceResultsLibrary,
)

if TYPE_CHECKING:  # pragma: no cover
    from axisforge.core.mechanical_system.parallel_axis.spur_helical.gear_system import (
        SpurHelicalGearSystem,
    )
    from axisforge.fixtures.construction.construction_capabilities import (
        ConstructionCapabilities,
    )

__all__ = ["run_convergence"]


def run_convergence(
    system: "SpurHelicalGearSystem",
    construction: "ConstructionCapabilities",
    library: ConvergenceResultsLibrary | None = None,
    *,
    theory: str = "timoshenko",
    distribute_gear_labels: set[str] | None = None,
    gci_threshold: float = 0.01,
    safety_factor: float = 1.25,
    max_levels: int = 8,
) -> ConvergenceResultsLibrary:
    """
    Run a mesh convergence study on every ShaftSystem in system.shafts
    (one fresh RigidBearingFEMSolver + one fresh MeshConvergenceStudy per
    shaft), storing each shaft's MeshRefinementResult in `library`.

    Parameters
    ----------
    system : SpurHelicalGearSystem
        Already built AND resolved (gear-mesh loads + shaft positions
        already present -- e.g. via build_linear_system()). Purely read
        here -- this function does not itself resolve or validate it.
    construction : ConstructionCapabilities
        The request that built `system`. Checked via
        construction.has_capability("systems.parallel_axis_linear") --
        raises ValueError immediately if False. See this module's own
        top docstring for why.
    library : ConvergenceResultsLibrary | None
        Reused if given -- re-running a shaft name already present
        overwrites it (ConvergenceResultsLibrary.store()'s own "fresh
        solve replaces stale" behaviour). A new, empty library is
        created if omitted.
    theory, distribute_gear_labels : RigidBearingFEMSolver constructor
        kwargs for every shaft's baseline GLOBAL solve, applied
        uniformly across all shafts in `system`. Defaults match the
        only configuration that exists today.
    gci_threshold, safety_factor, max_levels : MeshConvergenceStudy
        constructor kwargs, applied uniformly across all shafts.

    Returns
    -------
    ConvergenceResultsLibrary
        The same `library` passed in (or a new one), now holding one
        MeshRefinementResult per shaft in system.shafts, keyed by
        shaft name.
    """
    if not construction.has_capability("systems.parallel_axis_linear"):
        raise ValueError(
            "run_convergence: construction did not request "
            "'systems.parallel_axis_linear' -- the only Construction "
            "capability that resolves the system (gear-mesh loads + "
            "shaft positions) before returning it. Without it, `system` "
            "cannot be trusted to have been resolved, and the baseline "
            "global solve this study refines around would be built with "
            "zero gear-mesh loads."
        )

    library = library if library is not None else ConvergenceResultsLibrary()

    for ss in system.shafts:
        global_solver = RigidBearingFEMSolver(
            theory=theory,
            distribute_gear_labels=distribute_gear_labels,
        )
        global_solver.solve(ss)

        study = MeshConvergenceStudy(
            global_solver,
            gci_threshold=gci_threshold,
            safety_factor=safety_factor,
            max_levels=max_levels,
        )

        intervals, _labels_order = MeshConvergenceStudy.intervals_from_shaft_system(ss)
        result = study.run(ss, intervals)
        library.store(result)

    return library