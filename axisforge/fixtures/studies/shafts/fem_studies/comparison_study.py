"""
fixtures/studies/shafts/fem_studies/comparison_study.py

Fixture-level convenience that DOES the comparison, as opposed to
comparison_report.py (builds/writes text from two libraries it is
handed) and fem_simple.py (solves ONE theory at a time). This module
sits between them: it calls fem_simple.solve_system() twice -- same
`system`/`construction`, two different `theory` values -- then hands
both resulting libraries to comparison_report.py's own block/writer
functions rather than re-deriving any table logic here. Same reasoning
already applied to fem_simple.solve_system() itself and to
linear_chain_fixture.py's build_linear_system() (see fem_simple.py's
own top docstring): a fixture-level convenience that performs the real
work, not just object assembly.

Solving the SAME `system` object twice (never two separately-built
systems) is deliberate and load-bearing for the comparison to mean
anything -- see comparison_report.py's own top docstring for why two
ShaftResults from different systems would silently produce a
meaningless table. run_comparison() takes one already-built `system`
and calls solve_system() on it twice; it does not build `system`
itself, the same "reads, does not construct" boundary fem_simple.py's
own solve_system() keeps toward its `system` parameter.

print_comparison() prints to stdout using comparison_report.py's own
shaft_comparison_block() -- not a separate ad-hoc console format. The
first version of this comparison (check_resolution_fem_compare.py, an
exploratory script under studies/01_stepped_shaft_static/) built its
own inline print table; that duplicated comparison_report.py's table
logic in a second place, and the two were only checked to have the
same numbers, never guaranteed to render identically. Reusing the same
block function for both stdout and the .txt file (write_comparison_report(),
re-exported here for convenience) means the console preview and the
written report are always the same content, not two maintained copies.

Dependency (fixtures/ only, read-only access to solvers/core -- no
core modification):
  axisforge.fixtures.studies.shafts.fem_simple
      solve_system() -- called twice, once per theory. Not
      TYPE_CHECKING-only: this module actually calls it, unlike the
      report modules which only ever read pre-built results.
  axisforge.fixtures.studies.shafts.fem_studies.results_library
      RigidBearingFEMResultsLibrary -- the return type of each
      solve_system() call.
  axisforge.fixtures.studies.shafts.fem_studies.outputs.comparison_report
      shaft_comparison_block(), write_comparison_report() -- reused
      verbatim, not reimplemented.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from axisforge.fixtures.studies.shafts.fem_studies.fem_simple import solve_system
from axisforge.fixtures.studies.shafts.fem_studies.results_library import (
    RigidBearingFEMResultsLibrary,
)
from axisforge.fixtures.studies.shafts.fem_studies.outputs.comparison_report import (
    shaft_comparison_block,
    write_comparison_report,
)

RULE = "=" * 88

if TYPE_CHECKING:  # pragma: no cover
    from axisforge.core.mechanical_system.parallel_axis.spur_helical.gear_system import (
        SpurHelicalGearSystem,
    )
    from axisforge.fixtures.construction.construction_capabilities import (
        ConstructionCapabilities,
    )

__all__ = ["run_comparison", "print_comparison", "write_comparison_report"]


def run_comparison(
    system: "SpurHelicalGearSystem",
    construction: "ConstructionCapabilities",
    theory_a: str = "timoshenko",
    theory_b: str = "euler",
    distribute_gear_labels: set[str] | None = None,
    extra_mandatory: dict[str, list[float]] | None = None,
) -> tuple[RigidBearingFEMResultsLibrary, RigidBearingFEMResultsLibrary]:
    """
    Solve the SAME `system` twice via fem_simple.solve_system() -- once
    with theory=`theory_a`, once with theory=`theory_b` -- and return
    the two resulting libraries in that order. `distribute_gear_labels`
    and `extra_mandatory` are forwarded UNCHANGED to both calls, so the
    only thing that differs between the two solves is `theory` itself;
    anything else differing between the two libraries would mean the
    comparison is no longer isolating beam theory as the one variable
    (see this module's own top docstring on why that matters).

    Parameters
    ----------
    system : SpurHelicalGearSystem
        Already built AND resolved (see fem_simple.solve_system()'s own
        docstring). Solved twice, never rebuilt.
    construction : ConstructionCapabilities
        The request that built `system` -- forwarded to both
        solve_system() calls, which each re-check
        construction.has_capability("systems.parallel_axis_linear")
        independently (fem_simple.py's own guard, not duplicated here).
    theory_a, theory_b : str
        Passed straight through to StiffnessMatrixBuilder via
        RigidBearingFEMSolver -- must be keys of
        StiffnessMatrixBuilder._BEAM_THEORIES ("timoshenko", "euler",
        ...). Not validated here; an unknown value surfaces as
        ValueError from inside solve_system()'s own RigidBearingFEMSolver
        construction, same error either side would raise standalone.
    distribute_gear_labels, extra_mandatory : forwarded verbatim to
        both solve_system() calls -- see fem_simple.solve_system()'s
        own docstring for their meaning.

    Returns
    -------
    tuple[RigidBearingFEMResultsLibrary, RigidBearingFEMResultsLibrary]
        (library_a, library_b) -- one fresh library per call
        (solve_system()'s own `library=None` default), never merged.
    """
    library_a = solve_system(
        system, construction,
        theory=theory_a,
        distribute_gear_labels=distribute_gear_labels,
        extra_mandatory=extra_mandatory,
    )
    library_b = solve_system(
        system, construction,
        theory=theory_b,
        distribute_gear_labels=distribute_gear_labels,
        extra_mandatory=extra_mandatory,
    )
    return library_a, library_b


def print_comparison(
    library_a: RigidBearingFEMResultsLibrary,
    library_b: RigidBearingFEMResultsLibrary,
    system: "SpurHelicalGearSystem",
    label_a: str,
    label_b: str,
) -> None:
    """
    Print one "SHAFT: <name>" block per shaft in system.shafts (same
    order convention as every report writer in this package), each via
    comparison_report.shaft_comparison_block() -- the exact same
    content write_comparison_report() would put in the .txt file, so
    stdout and file are never two separately-maintained formats.
    """
    for ss in system.shafts:
        print(RULE)
        print(f"SHAFT: {ss.name}")
        print(RULE)

        result_a = library_a.get_or_none(ss.name)
        result_b = library_b.get_or_none(ss.name)
        if result_a is None or result_b is None:
            missing = []
            if result_a is None:
                missing.append(label_a)
            if result_b is None:
                missing.append(label_b)
            print(f"  (missing from: {', '.join(missing)})")
        else:
            print(shaft_comparison_block(ss.name, result_a, result_b, label_a, label_b))
        print()