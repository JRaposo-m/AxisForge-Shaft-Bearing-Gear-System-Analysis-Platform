"""
fixtures/outputs/construction/shafts/shaft_report.py

Report block for the `shafts` Construction domain -- mirrors
fixtures/construction/shafts/ one-to-one, so it is always obvious which
report you are calling: "the shafts report" lives at the same address
in outputs/ as the shafts fixture lives in construction/.

shaft_geometry_block(shaft) formats one Shaft's own section/shoulder
geometry -- the same information ShaftFixture.summary()
(fixtures/construction/shafts/shaft_fixture.py) prints, rebuilt here
from the bare core Shaft (sections + shoulders()) because a
systems.parallel_axis_linear ShaftSpec only carries the unwrapped
Shaft, not that fixture-level wrapper (see linear_chain_fixture.py's
own ShaftSpec.shaft docstring for why).

CONTENT ONLY -- this module builds a text block, it does not write any
file. fixtures/outputs/construction/text_report.py is the single place
that assembles every domain's blocks (this module's plus
bearings/gears/systems') into the one combined construction report .txt
and actually writes it. See that module's own docstring for the full
reasoning.

Dependency (core only, read-only access):
  axisforge.core.machine_elements.shaft.shaft
      Shaft, ShaftSection, Shoulder
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from axisforge.core.machine_elements.shaft.shaft import Shaft


def shaft_geometry_block(shaft: "Shaft") -> str:
    """
    ASCII block: section-by-section geometry (index, label, axial
    extent, diameter, length, material, surface finish) followed by a
    "shoulders:" block for every transition that carries one (fillet
    radius, diameter range, r/d, D/d).

    Purely reads shaft.sections and shaft.shoulders() -- does not
    validate or compute anything the Shaft itself doesn't already
    expose.
    """
    lines = [
        f"  geometry ({shaft.n_sections} sections, "
        f"total_length={shaft.total_length:.2f} mm):"
    ]
    z = 0.0
    for i, s in enumerate(shaft.sections):
        z_start, z_end = z, z + s.length
        z = z_end
        label_str = f" ({s.label})" if s.label else ""
        lines.append(
            f"    [{i}]{label_str} z=[{z_start:.1f}, {z_end:.1f}] mm  "
            f"d={s.diameter:.1f} mm  L={s.length:.1f} mm  "
            f"mat={s.material_id}  Ra={s.surface_finish_ra} um"
        )
    shoulders = shaft.shoulders()
    if shoulders:
        lines.append("  shoulders:")
        for z_pos, sh in shoulders:
            lines.append(
                f"    z={z_pos:.1f} mm  r={sh.fillet_radius:.2f} mm  "
                f"d {sh.diameter_small:.1f}->{sh.diameter_large:.1f}  "
                f"r/d={sh.r_over_d:.4f}  D/d={sh.D_over_d:.4f}"
            )
    return "\n".join(lines)