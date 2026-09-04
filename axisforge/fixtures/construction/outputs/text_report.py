"""
fixtures/outputs/construction/text_report.py

THE principal writer for the whole Construction domain: assembles every
Construction domain's own content blocks --
  fixtures/outputs/construction/shafts/shaft_report.py     -> shaft_geometry_block()
  fixtures/outputs/construction/bearings/bearing_report.py -> bearing_block()
  fixtures/outputs/construction/gears/gear_report.py       -> gear_block()
  fixtures/outputs/construction/systems/system_report.py   -> topology_block(),
                                                                shaft_system_block(),
                                                                loads_block(),
                                                                mesh_block()
-- in Construction order (topology, then per shaft: geometry, bearings,
gears, shaft-system summary + loads, then meshes) into ONE combined .txt
file. Those four domain modules build text blocks only -- this is the
only module in the package that actually opens a file and writes.

This module lives INSIDE construction/, not at fixtures/outputs/ top
level, because it is a construction-domain writer specifically -- not a
generic "outputs" facility. fixtures/outputs/ is a cross-cutting
reporting utility shared by every fixtures/ pipeline stage (construction,
and later solvers/plots/convergence/integration, mirroring
fixtures/construction/'s own siblings); each of those stages gets its
own principal text_report.py living next to its own domain content, the
same way this one lives next to shafts/bearings/gears/systems -- so a
future solvers-stage equivalent would live at
fixtures/outputs/solvers/text_report.py and write its own single file,
never here.

write_construction_report(system, path, title="") is the only thing
this module does. ONE combined .txt, on explicit user preference --
several small per-domain .txt files were tried and rejected; one file
per pipeline stage, containing everything about that stage, is what
this module produces.

Writes with encoding="utf-8" explicitly: SpurHelicalGearSystem.summary()
and ShaftSystem.summary() (used by systems.topology_block/
shaft_system_block) use box-drawing / degree / middle-dot Unicode
glyphs, and Python's default text-file encoding on Windows is the
system codepage (commonly cp1252), which does not cover them and would
raise UnicodeEncodeError on write.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from axisforge.fixtures.construction.shafts.outputs.shaft_report import (
    shaft_geometry_block,
)
from axisforge.fixtures.construction.bearings.outputs.bearing_report import (
    bearing_block,
)
from axisforge.fixtures.construction.gears.parallel_axis.fixed.outputs.gear_report import (
    gear_block,
)
from axisforge.fixtures.construction.systems.parallel_axis.spur_helical.outputs.system_report import (
    topology_block,
    shaft_system_block,
    loads_block,
    mesh_block,
)

RULE = "=" * 72
SUB = "-" * 72

if TYPE_CHECKING:  # pragma: no cover
    from axisforge.core.mechanical_system.parallel_axis.spur_helical.gear_system import (
        SpurHelicalGearSystem,
    )


def write_construction_report(
    system: "SpurHelicalGearSystem",
    path: "str | Path",
    title: str = "",
) -> str:
    """
    Write ONE .txt covering the whole Construction domain for `system`:
    topology/resolved state, then per shaft (in system.shafts order)
    its geometry, bearings, gears, ShaftSystem summary and loads, then
    every mesh. Returns the written text, so a caller that wants to
    inspect/verify it doesn't have to re-open the file it just wrote.

    Parameters
    ----------
    system : SpurHelicalGearSystem
        Already built (e.g. via build_linear_system()). Purely reads
        `system` -- does not validate or resolve anything.
    path : str | Path
        The .txt file to write (parent directory created if missing).
        A relative path resolves against the process's current working
        directory -- NOT this module's location, and not the calling
        script's location either. To have the report always land next
        to the script that built it (the usual intent, regardless of
        where Python was launched from), build an absolute path at the
        call site: Path(__file__).resolve().parent / "report.txt".
    title : str
        Header title. Defaults to system.label or "Construction report".
    """
    header_title = title or system.label or "Construction report"
    sections = [RULE, header_title.center(72), RULE, ""]

    sections.append("TOPOLOGY / RESOLVED STATE")
    sections.append(SUB)
    sections.append(topology_block(system))
    sections.append("")

    for ss in system.shafts:
        sections.append(RULE)
        sections.append(f"SHAFT: {ss.name}")
        sections.append(RULE)

        sections.append(shaft_geometry_block(ss.shaft))
        sections.append("")

        if ss.bearings:
            sections.append(f"  bearings ({len(ss.bearings)}):")
            for b in ss.bearings:
                sections.append(bearing_block(b))
        else:
            sections.append("  bearings: (none)")
        sections.append("")

        if ss.gears:
            sections.append(f"  gears ({len(ss.gears)}):")
            for ge in ss.gears:
                sections.append(gear_block(ge))
        else:
            sections.append("  gears: (none)")
        sections.append("")

        sections.append(shaft_system_block(ss))
        sections.append("")
        sections.append(loads_block(ss))
        sections.append("")

    if system.links:
        sections.append(RULE)
        sections.append(f"MESHES ({len(system.links)})")
        sections.append(RULE)
        for link in system.links:
            sections.append(mesh_block(link))
            sections.append("")

    sections.append(RULE)
    text = "\n".join(sections) + "\n"
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text