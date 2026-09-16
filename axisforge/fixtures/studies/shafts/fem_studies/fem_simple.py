"""
fixtures/studies/shafts/fem_simple.py

Studies stage: solves every ShaftSystem in a SpurHelicalGearSystem's
system.shafts via RigidSupportFEMSolver + ShaftResultsReader, publishing
every result into one RigidSupportFEMResultsLibrary.

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

Only ONE solver configuration exists today -- rigid supports. There is
NO constraint_bearing switch, and this module must not reintroduce
one: rigid supports are RigidSupportFEMSolver's identity, not one of
two options it selects between (v=0 always; u=0 additionally for
"locating" bearings -- see boundary_dofs() in
fem_solvers/constraints/boundary_conditions.py). An earlier version of
this module passed constraint_bearing="rigid" through to the
constructor; the parameter was never wired to any alternative
behaviour, boundary_dofs() never read it, and it has since been
removed from both sides. A compliant-bearing solve would be a sibling
solver module, not a mode of this one. study_capabilities.py's
"shaft_fem.timoshenko_rigid" capability string names the same physics
-- "rigid" there describes what this solver does, it does not select
it.

beam_theory/shear_theory/integration_method are kept as THREE SEPARATE
parameters here, not bundled into a single BeamModelSettings argument,
even though RigidSupportFEMSolver itself takes settings as one object.
This is deliberate: study_capabilities.py pins `theory` via
functools.partial while leaving shear_theory (and now
integration_method) for the caller to supply at each call site -- e.g.
a validation case script sweeping shear_theory across
("cowper", "hutchinson") for the SAME pinned theory. partial() can only
fix one parameter at a time; it cannot fix "part of" a dataclass
passed as a single argument. The BeamModelSettings instance is
assembled once, right here, immediately before constructing the
solver -- this is the one and only place that conversion happens.
There is no default for any of the three, same as everywhere else
this triple appears (BeamModelSettings itself, RigidSupportFEMSolver) --
every caller of solve_system(), including every validation case
script, must now supply integration_method explicitly.

Torsion is NOT touched by this module at all anymore -- it used to be
solved here (once per shaft, via the old module-level solve_torsion())
and its three return values were threaded into
ShaftResultsReader.read(T_total, tau_total, torsion_contributions).
That function no longer exists: static_solvers/torsion.py now exposes
a TorsionSolver class (solve() returning T_total, tau_total, phi_total,
contributions), and ShaftResultsReader.read() builds its own
TorsionSolver internally and calls it as part of read() -- see that
module's own docstring. This module's job shrank accordingly: it just
runs the FEM solve and calls read() with no arguments, same as any
other post-processing it doesn't need to know about (bending/axial
recovery were already handled the same way, inside read(), before this
change).

ShaftResultsReader.read() returns a ShaftResults and stores nothing --
the reader deliberately does not know libraries exist (see its own
docstring in results_reader.py). Publishing into the library is THIS
module's job, one library.store() per shaft, which is why the loop
below is several statements rather than a single read(library) call.

Dependency (solvers + fixtures, read-only access -- no core modification):
  axisforge.solvers.machine_elements.shaft.fem_solvers.rigid_support
      RigidSupportFEMSolver
  axisforge.solvers.machine_elements.shaft.static_solvers.results_reader
      ShaftResultsReader
  axisforge.mesh.shaft.beam_model_settings
      BeamModelSettings
  axisforge.fixtures.studies.shafts.fem_studies.results_library
      RigidSupportFEMResultsLibrary
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from axisforge.mesh.shaft.beam_model_settings import BeamModelSettings
from axisforge.solvers.machine_elements.shaft.fem_solvers.rigid_support import (
    RigidSupportFEMSolver,
)
from axisforge.solvers.machine_elements.shaft.static_solvers.results_reader import (
    ShaftResultsReader,
)
from axisforge.fixtures.studies.shafts.fem_studies.results_library import (
    RigidSupportFEMResultsLibrary,
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
    library: RigidSupportFEMResultsLibrary | None = None,
    *,
    theory: str,
    shear_theory: str | None,
    integration_method: str | None,
    distribute_gear_labels: set[str] | None = None,
    extra_mandatory: dict[str, list[float]] | None = None,
    kGA_override: float | None = None,
) -> RigidSupportFEMResultsLibrary:
    """
    Solve every ShaftSystem in system.shafts (one fresh RigidSupportFEMSolver
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
    library : RigidSupportFEMResultsLibrary | None
        Reused if given -- re-solving a shaft name already present
        overwrites it (RigidSupportFEMResultsLibrary.store()'s own
        behaviour: "fresh solve replaces stale"). A new, empty library
        is created if omitted.
    theory, shear_theory, integration_method
        Assembled into one BeamModelSettings, applied UNIFORMLY to
        every shaft's solver (same values for all shafts in `system`).
        No defaults -- the caller must decide explicitly, same as
        BeamModelSettings itself requires. theory="euler_bernoulli"
        requires shear_theory=None, integration_method=None (see
        BeamModelSettings.__post_init__); theory="timoshenko" requires
        both non-None. Kept as three separate parameters rather than
        one BeamModelSettings argument specifically so
        study_capabilities.py can pin `theory` via functools.partial
        while a caller still supplies shear_theory/integration_method
        per call -- see this module's own top docstring.
    distribute_gear_labels : RigidSupportFEMSolver constructor kwarg,
        applied uniformly to every shaft's solver. None (default): no
        gear-mesh load is treated as distributed.
    extra_mandatory : dict[str, list[float]] | None
        Optional per-shaft extra mesh-refinement node positions, keyed
        by ShaftSystem.name, forwarded to that shaft's own
        solver.solve(shaft, extra_mandatory=...) call (e.g. gear
        contact-zone refinement via Grader(lo, hi,
        base_nodes).get_grade(grade) -- built by the caller, this
        function only forwards the resulting list). Shafts whose name
        is not a key get extra_mandatory=None (RigidSupportFEMSolver's own
        default -- no extra refinement).
    kGA_override : float | None
        Forwarded UNIFORMLY to every shaft's RigidSupportFEMSolver
        constructor -- see RigidSupportFEMSolver.__init__'s own
        docstring. Replaces the transverse shear stiffness K*G*A with
        this exact value for every Timoshenko element in every shaft,
        instead of computing it from shear_theory (e.g. to test a
        value read directly from Abaqus's own *Preprint, model=YES
        section-properties printout). Has no effect when
        theory="euler_bernoulli" -- forwarded regardless, but Elem
        itself ignores it for euler_bernoulli elements.

    Returns
    -------
    RigidSupportFEMResultsLibrary
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

    settings = BeamModelSettings(
        beam_theory=theory,
        shear_theory=shear_theory,
        integration_method=integration_method,
    )

    library = library if library is not None else RigidSupportFEMResultsLibrary()
    extra_mandatory = extra_mandatory or {}

    for ss in system.shafts:
        solver = RigidSupportFEMSolver(
            settings,
            distribute_gear_labels=distribute_gear_labels,
            kGA_override=kGA_override,
        )
        solver.solve(ss, extra_mandatory=extra_mandatory.get(ss.name))

        result = ShaftResultsReader(solver, ss).read()
        library.store(result)

    return library