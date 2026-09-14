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
    RigidSupportFEMResultsLibrary,
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
    system, construction,
    theory_a: str = "timoshenko",
    theory_b: str = "euler_bernoulli",
    *,
    shear_theory: str | None,
    integration_method: str | None,
    distribute_gear_labels=None,
    extra_mandatory=None,
):
    def _settings_for(theory):
        if theory == "timoshenko":
            return shear_theory, integration_method
        return None, None

    st_a, im_a = _settings_for(theory_a)
    st_b, im_b = _settings_for(theory_b)

    library_a = solve_system(system, construction, theory=theory_a,
                              shear_theory=st_a, integration_method=im_a,
                              distribute_gear_labels=distribute_gear_labels,
                              extra_mandatory=extra_mandatory)
    library_b = solve_system(system, construction, theory=theory_b,
                              shear_theory=st_b, integration_method=im_b,
                              distribute_gear_labels=distribute_gear_labels,
                              extra_mandatory=extra_mandatory)
    return library_a, library_b


def print_comparison(
    library_a: RigidSupportFEMResultsLibrary,
    library_b: RigidSupportFEMResultsLibrary,
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