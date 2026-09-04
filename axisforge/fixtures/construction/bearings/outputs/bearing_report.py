"""
fixtures/outputs/construction/bearings/bearing_report.py

Report block for the `bearings` Construction domain -- mirrors
fixtures/construction/bearings/ one-to-one, same reasoning as
shaft_report.py: a predictable, domain-mirrored address for "the
bearings report", not a detail buried inside one big text_report.py.

bearing_block() is a THIN wrapper -- Bearing.summary() (core) already
covers everything this domain's report needs (family, bearing_type,
duty, position, arrangement, d/D/b, C/C0, enabled analyses). It exists
anyway, rather than every caller just doing `bearing.summary()`
directly, so that bearings has the same one-call-per-domain shape as
shafts/gears/systems -- calling bearing_block(b) from this module is
how you ask "give me the bearings report for b", the same way you'd
call shaft_geometry_block(shaft) or gear_block(ge). If Bearing.summary()
ever stops covering everything this domain's report should show, extend
it here rather than duplicating fields.

CONTENT ONLY -- this module builds a text block, it does not write any
file. fixtures/outputs/construction/text_report.py is the single place
that assembles every domain's blocks (this module's plus
shafts/gears/systems') into the one combined construction report .txt
and actually writes it. See that module's own docstring for the full
reasoning.

Dependency (core only, read-only access):
  axisforge.core.machine_elements.bearings.bearing
      Bearing
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from axisforge.core.machine_elements.bearings.bearing import Bearing


def bearing_block(bearing: "Bearing") -> str:
    """ASCII block for one Bearing -- delegates entirely to
    Bearing.summary() (core already formats this well)."""
    return bearing.summary()