"""
fixtures/solvers/fem_simple.py

Resolution stage: solves every ShaftSystem in a SpurHelicalGearSystem's
system.shafts via RigidBearingFEMSolver + ShaftResultsReader, publishing
every result into one RigidBearingFEMResultsLibrary.

Guarded on construction -- solve_system() takes the ConstructionCapabilities
that built `system` as a second required argument, and raises ValueError
immediately unless construction.has_capability("systems.parallel_axis_linear")
-- the only Construction capability that resolves the system (gear-mesh
loads + shaft positions) before returning it. This is a DECLARATIVE
check (was Construction asked to produce an already-resolved system),
not an inspection of `system`'s own runtime state -- ConstructionCapabilities.
has_capability() reads only that instance's own requested capability
strings, nothing about what actually happened when resolve() ran. No
core change needed for this: SpurHelicalGearSystem itself is untouched,
and this module still never reaches into its private _resolved record.

Fills in the "Planned" fixtures/solvers/fem_simple.py entry from
fixtures/README.md's own Status table -- step 5 of "Typical Analysis
Script Structure" ("Solve FEM"). Mirrors linear_chain_fixture.py's own
build_linear_system(): a fixture-level convenience that performs the
real work (here, the FEM solve itself), not just object assembly --
same reasoning already accepted there, and confirmed again for this
module in ResolutionCapabilities' own docstring, fixtures/capabilities/__init__.py
("Resolution" IS "solve").

Only ONE solver configuration exists today -- RigidBearingFEMSolver(theory=
"timoshenko", constraint_bearing="rigid"). Note: constraint_bearing is
accepted by RigidBearingFEMSolver.__init__ but not actually wired to any
alternative behaviour yet -- every bearing is always a rigid support
(v=0 always; u=0 additionally for "locating" bearings -- see
_boundary_dofs() in rigid_bearing.py, which never reads
constraint_bearing at all). "rigid" names the physics this solver
actually applies today, not a switch between two real options --
solve_system()'s own default kwargs match this exactly, and
resolution.py's "shaft_fem.timoshenko_rigid" capability string is built
on the same reasoning.

Dependency (solvers only, read-only access -- no core modification):
  axisforge.solvers.machine_elements.shaft.oneD_analysis.FEM_solvers.rigid_bearing
      RigidBearingFEMSolver
  axisforge.solvers.machine_elements.shaft.oneD_analysis.static.static_analysis
      ShaftResultsReader, RigidBearingFEMResultsLibrary
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from axisforge.solvers.machine_elements.shaft.fem_solvers.rigid_bearing import (
    RigidBearingFEMSolver,
)
from axisforge.solvers.machine_elements.shaft.static.results_reader import (
    ShaftResultsReader,
)
from axisforge.fixtures.studies.shafts.results_library import (
    RigidBearingFEMResultsLibrary,
)

if TYPE_CHECKING:  # pragma: no cover
    from axisforge.core.mechanical_system.parallel_axis.spur_helical.gear_system import (
        SpurHelicalGearSystem,
    )
    from axisforge.fixtures.capabilities import ConstructionCapabilities


def solve_system(
    system: "SpurHelicalGearSystem",
    construction: "ConstructionCapabilities",
    library: RigidBearingFEMResultsLibrary | None = None,
    *,
    theory: str = "timoshenko",
    constraint_bearing: str = "rigid",
    distribute_gear_labels: set[str] | None = None,
    extra_mandatory: dict[str, list[float]] | None = None,
) -> RigidBearingFEMResultsLibrary:
    """
    Solve every ShaftSystem in system.shafts (one fresh RigidBearingFEMSolver
    per shaft -- solve() overwrites, so re-using one instance across
    shafts would silently clobber the previous shaft's results; see
    this project's own "one solver instance per shaft" rule), reading
    each solved shaft into `library` via ShaftResultsReader.

    Parameters
    ----------
    system : SpurHelicalGearSystem
        Already built AND resolved (gear-mesh loads + shaft positions
        already present -- e.g. via build_linear_system(), which
        resolves before returning). Otherwise purely reads `system` --
        does not itself resolve or validate it.
    construction : ConstructionCapabilities
        The request that built `system`. Checked via
        construction.has_capability("systems.parallel_axis_linear") --
        raises ValueError immediately if False, rather than silently
        FEM-solving a system that (per this declarative check) was
        never actually resolved. See this module's own top docstring
        for why this check lives here, on the Construction request,
        rather than on `system` itself.
    library : RigidBearingFEMResultsLibrary | None
        Reused if given -- re-solving a shaft name already present
        overwrites it (RigidBearingFEMResultsLibrary.store()'s own
        behaviour: "fresh solve replaces stale"). A new, empty library
        is created if omitted.
    theory, constraint_bearing, distribute_gear_labels : RigidBearingFEMSolver
        constructor kwargs, applied UNIFORMLY to every shaft's solver
        (same values for all shafts in `system`). Defaults match the
        only configuration that actually exists today -- see this
        module's own top docstring for why "rigid" is accurate despite
        constraint_bearing not being wired to an alternative yet.
    extra_mandatory : dict[str, list[float]] | None
        Optional per-shaft extra mesh-refinement node positions, keyed
        by ShaftSystem.name, forwarded to that shaft's own
        solver.solve(shaft, extra_mandatory=...) call (e.g. gear
        contact-zone refinement via Grader(lo, hi,
        base_nodes).get_grade(grade) -- built by the caller, this
        function only forwards the resulting list). Shafts whose name
        is not a key get extra_mandatory=None (RigidBearingFEMSolver's own
        default -- no extra refinement).

    Returns
    -------
    RigidBearingFEMResultsLibrary
        The same `library` passed in (or a new one), now holding one
        ShaftResults per shaft in system.shafts, keyed by shaft name.
    """
    if not construction.has_capability("systems.parallel_axis_linear"):
        raise ValueError(
            "solve_system: construction did not request "
            "'systems.parallel_axis_linear' -- the only Construction "
            "capability that resolves the system (gear-mesh loads + "
            "shaft positions) before returning it. Without it, `system` "
            "cannot be trusted to have been resolved, and solving it "
            "would run the FEM with zero gear-mesh loads."
        )

    library = library if library is not None else RigidBearingFEMResultsLibrary()
    extra_mandatory = extra_mandatory or {}

    for ss in system.shafts:
        solver = RigidBearingFEMSolver(
            theory=theory,
            constraint_bearing=constraint_bearing,
            distribute_gear_labels=distribute_gear_labels,
        )
        solver.solve(ss, extra_mandatory=extra_mandatory.get(ss.name))
        ShaftResultsReader(solver, ss).read(library)

    return library