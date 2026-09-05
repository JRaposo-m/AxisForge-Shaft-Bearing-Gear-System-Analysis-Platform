"""
fixtures/studies/text_report.py

THE principal writer for the whole Studies domain -- mirrors
fixtures/outputs/construction/text_report.py's own role exactly:
assembles every Studies domain's own content blocks --
  fixtures/studies/shafts/fem_studies/outputs/resolution_report.py -> shaft_result_block()
  fixtures/studies/shafts/fem_studies/outputs/comparison_report.py -> shaft_comparison_block()
  (bearing_iso16281's own block module, once that domain has one)
-- into ONE combined .txt file. Domain modules like resolution_report.py
build text blocks only -- this is now the only module in the whole
Studies package that actually opens a file and writes, the same split
Construction already has between its four outputs/*_report.py modules
and this one's Construction counterpart.

CONTENT IS PRESENCE-DRIVEN, same principle write_construction_report()
already follows -- it never reads a ConstructionCapabilities either, it
just asks the `system` object "do you have bearings/gears/links" and
prints "(none)" where the answer is no. Studies cannot do that same
structural check on `system` itself, because which studies exist is
NOT inherent to `system` (unlike a shaft's bearings, which are part of
its own assembly) -- it depends on which StudyCapabilities were
actually requested and resolved into results BEFORE this module is
ever called. So the presence check here lives one level up, on this
function's own optional parameters: pass `shaft_fem_library` if you ran
a single-theory study, pass `comparison` if you also ran
"shaft_fem.comparison", pass neither and get a report that just says so
per shaft/section -- never a KeyError or a silent empty file. This
module never touches StudyCapabilities or catalogue.py itself; the
caller resolves capabilities, produces result objects, and hands this
module only the objects it has -- same "reads, does not build" boundary
every report writer in this package keeps.

write_studies_report(system, path, title="", shaft_fem_library=None,
comparison=None) is the only thing this module does. ONE combined .txt
per Studies run, same explicit "one file per pipeline stage" preference
already applied to Construction.

comparison_report.py's write_comparison_report() still exists and still
works standalone (e.g. to get JUST a comparison .txt, without a full
Studies report around it) -- `comparison=` here does not replace it, it
reuses the same shaft_comparison_block() content inside this bigger
report when the caller wants both in one file.

Writes with encoding="utf-8" explicitly, same reason as every other
report writer in this package (Construction's own top text_report.py
included): result content can carry Unicode glyphs from core
`.summary()`-style methods elsewhere in the platform, and Python's
default text-file encoding on Windows is the system codepage, which
does not cover them.

Dependency (fixtures/ only, read-only access -- no core modification):
  axisforge.fixtures.studies.shafts.fem_studies.outputs.resolution_report
      shaft_result_block() -- the shaft_fem single-solve content block.
  axisforge.fixtures.studies.shafts.fem_studies.outputs.comparison_report
      shaft_comparison_block() -- the shaft_fem two-solve comparison
      content block, reused here for the optional `comparison` section.
  Neither import is TYPE_CHECKING-only: this module actually CALLS both,
  it does not just type-annotate against them.
  axisforge.fixtures.studies.shafts.fem_studies.results_library
      RigidBearingFEMResultsLibrary -- the registry `shaft_fem_library`
      and each half of `comparison` are expected to be. TYPE_CHECKING-only:
      this module only ever reads a library handed to it, it never
      builds one; see fem_simple.py's own solve_system() for that.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from axisforge.fixtures.studies.shafts.fem_studies.outputs.resolution_report import (
    shaft_result_block,
)
from axisforge.fixtures.studies.shafts.fem_studies.outputs.comparison_report import (
    shaft_comparison_block,
)

RULE = "=" * 72

if TYPE_CHECKING:  # pragma: no cover
    from axisforge.core.mechanical_system.parallel_axis.spur_helical.gear_system import (
        SpurHelicalGearSystem,
    )
    from axisforge.fixtures.studies.shafts.fem_studies.results_library import (
        RigidBearingFEMResultsLibrary,
    )


def write_studies_report(
    system: "SpurHelicalGearSystem",
    path: "str | Path",
    title: str = "",
    shaft_fem_library: "RigidBearingFEMResultsLibrary | None" = None,
    comparison: "tuple[RigidBearingFEMResultsLibrary, RigidBearingFEMResultsLibrary, str, str] | None" = None,
) -> str:
    """
    Write ONE .txt covering the whole Studies domain for `system`: a
    SHAFT_FEM section (one "SHAFT: <name>" per shaft in system.shafts,
    system's own order -- matching every other report writer's own
    convention) if `shaft_fem_library` is given, then a COMPARISON
    section (same per-shaft shape, via shaft_comparison_block()) if
    `comparison` is given. Either, both, or neither -- an empty call
    (both None) still writes a valid report, just one that says nothing
    was run, same as a Construction report for a shaft with no bearings
    says "(none)" rather than omitting the heading. Returns the written
    text, so a caller that wants to inspect/verify it doesn't have to
    re-open the file it just wrote.

    A future bearing_iso16281 section (once that domain has its own
    report module) joins as another optional parameter here, the same
    place MESHES joins Construction's own aggregator after its own
    per-shaft loop -- see fixtures/outputs/construction/text_report.py
    for that shape.

    Parameters
    ----------
    system : SpurHelicalGearSystem
        Supplies the shaft ORDER and NAMES both sections walk -- a
        library itself has no ordering guarantee beyond insertion, and
        a shaft system's own shaft order is the more meaningful one to
        read a report in (matches every Construction report writer's
        own convention).
    path : str | Path
        The .txt file to write (parent directory created if missing).
        A relative path resolves against the process's current working
        directory -- NOT this module's location, and not the calling
        script's location either. To have the report always land next
        to the script that built it, build an absolute path at the
        call site: Path(__file__).resolve().parent / "report.txt".
    title : str
        Header title. Defaults to system.label or "Studies report".
    shaft_fem_library : RigidBearingFEMResultsLibrary | None
        Already solved (e.g. via fem_simple.solve_system(), one theory
        at a time -- from "shaft_fem.timoshenko_rigid" or
        "shaft_fem.euler_bernoulli_rigid"). Omit if that study wasn't
        run; the SHAFT_FEM section is skipped entirely (not printed as
        empty) in that case.
    comparison : tuple[library_a, library_b, label_a, label_b] | None
        From "shaft_fem.comparison" (comparison_study.run_comparison()
        returns the two libraries; pair them here with the labels you
        want in the table, e.g. ("timoshenko", "euler")). Omit if that
        study wasn't run; the COMPARISON section is skipped entirely in
        that case. Both `library_a`/`library_b` must have been solved
        against this SAME `system` -- see comparison_report.py's own
        top docstring for why a mismatched pair silently produces a
        meaningless table; this function does not detect that for you.
    """
    header_title = title or system.label or "Studies report"
    sections = [RULE, header_title.center(72), RULE, ""]

    if shaft_fem_library is not None:
        sections.append(RULE)
        sections.append("SHAFT_FEM")
        sections.append(RULE)
        for ss in system.shafts:
            sections.append(RULE)
            sections.append(f"SHAFT: {ss.name}")
            sections.append(RULE)

            result = shaft_fem_library.get_or_none(ss.name)
            if result is None:
                sections.append("  (no result -- not solved, or solved "
                                 "into a different library)")
            else:
                sections.append(shaft_result_block(result))
            sections.append("")

    if comparison is not None:
        library_a, library_b, label_a, label_b = comparison
        sections.append(RULE)
        sections.append(f"SHAFT_FEM COMPARISON ({label_a} vs {label_b})")
        sections.append(RULE)
        for ss in system.shafts:
            sections.append(RULE)
            sections.append(f"SHAFT: {ss.name}")
            sections.append(RULE)

            result_a = library_a.get_or_none(ss.name)
            result_b = library_b.get_or_none(ss.name)
            if result_a is None or result_b is None:
                missing = []
                if result_a is None:
                    missing.append(label_a)
                if result_b is None:
                    missing.append(label_b)
                sections.append(f"  (missing from: {', '.join(missing)})")
            else:
                sections.append(shaft_comparison_block(ss.name, result_a, result_b, label_a, label_b))
            sections.append("")

    if shaft_fem_library is None and comparison is None:
        sections.append("  (no studies run -- nothing was passed to "
                         "write_studies_report())")
        sections.append("")

    # future: a bearing_iso16281 section joins here, one RULE-delimited
    # block gated on its own optional parameter, the same place MESHES
    # joins Construction's own aggregator.

    sections.append(RULE)
    text = "\n".join(sections) + "\n"
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text