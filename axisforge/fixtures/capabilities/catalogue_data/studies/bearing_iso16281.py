"""axisforge/fixtures/capabilities/catalogue_data/studies/bearing_iso16281.py

Leaf dict for CATALOGUE["studies"]["bearings"]["bearing_iso16281"]. See
catalogue.py's own module docstring for the leaf shape and `requires`
token rules.
"""

from __future__ import annotations

BEARING_ISO16281: dict = {
    "bearing_iso16281.single_row": {
        "description": (
            "ISO/TS 16281 single-row internal load "
            "distribution for point- and line-contact bearings "
            "(RollingBearingSolver, dispatching per bearing to "
            "ISO16281BallSolver/ISO16281RollerSolver) into a "
            "BearingResultsLibrary. Multi-row (THRUST) bearings "
            "are out of scope for this capability."
        ),
        "requires": {
            "construction": ("systems.parallel_axis_linear",),
            "studies": ("shaft_fem",),
        },
    },
}