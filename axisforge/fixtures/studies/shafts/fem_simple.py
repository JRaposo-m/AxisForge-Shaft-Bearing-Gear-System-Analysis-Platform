"""
fixtures/studies/shafts/fem_simple.py

Studies stage: solves every ShaftSystem in a SpurHelicalGearSystem's
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

Step 5 of fixtures/README.md's own "Script structure" ("Solve FEM").
Mirrors linear_chain_fixture.py's own build_linear_system(): a
fixture-level convenience that performs the real work (here, the FEM
solve itself), not just object assembly -- same reasoning already
accepted there, and confirmed again for this module in
StudyCapabilities' own docstring, fixtures/studies/study_capabilities.py
("Studies" IS "solve").

Only ONE solver configuration exists today -- RigidBearingFEMSolver(
theory="timoshenko"). There is NO constraint_bearing switch, and this
module must not reintroduce one: rigid supports are this solver's
identity, not one of two options it selects between (v=0 always; u=0
additionally for "locating" bearings -- see _boundary_dofs() in
rigid_bearing.py). An earlier version of this module passed
constraint_bearing="rigid" through to the constructor; the parameter
was never wired to any alternative behaviour, _boundary_dofs() never
read it, and it has since been removed from both sides. A
compliant-bearing solve would be a sibling solver module, not a mode of
this one. study_capabilities.py's "shaft_fem.timoshenko_rigid"
capability string names the same physics -- "rigid" there describes
what this solver does, it does not select it.

ShaftResultsReader.read() returns a ShaftResults and stores nothing --
the reader deliberately does not know libraries exist (see its own
docstring in results_reader.py). Publishing into the library is THIS
module's job, one library.store() per shaft, which is why the loop
below is two statements rather than a single read(library) call.

Dependency (solvers + fixtures, read-only access -- no core modification):
  axisforge.solvers.machine_elements.shaft.fem_solvers.rigid_bearing
      RigidBearingFEMSolver
  axisforge.solvers.machine_elements.shaft.static.results_reader
      ShaftResultsReader
  axisforge.fixtures.studies.shafts.results_library
      RigidBearingFEMResultsLibrary
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
    from axisforge.fixtures.construction.construction_capabilities import (
        ConstructionCapabilities,
    )


def solve_system(
    system: "SpurHelicalGearSystem",
    construction: "ConstructionCapabilities",
    library: RigidBearingFEMResultsLibrary | None = None,
    *,
    theory: str = "timoshenko",
    distribute_gear_labels: set[str] | None = None,
    extra_mandatory: dict[str, list[float]] | None = None,
) -> RigidBearingFEMResultsLibrary:
    """
    Solve every ShaftSystem in system.shafts (one fresh RigidBearingFEMSolver
    per shaft -- solve() overwrites, so re-using one instance across
    shafts would silently clobber the previous shaft's results; see
    this project's own "one solver instance per shaft" rule), reading
    each solved shaft with ShaftResultsReader and storing the result
    in `library`.

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
    theory, distribute_gear_labels : RigidBearingFEMSolver constructor
        kwargs, applied UNIFORMLY to every shaft's solver (same values
        for all shafts in `system`). Defaults match the only
        configuration that exists today. There is no constraint_bearing
        kwarg -- see this module's own top docstring for why, and why
        it should not come back.
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
            distribute_gear_labels=distribute_gear_labels,
        )
        solver.solve(ss, extra_mandatory=extra_mandatory.get(ss.name))
        result = ShaftResultsReader(solver, ss).read()
        library.store(result)

    return library