"""axisforge/fixtures/capabilities/catalogue_data/construction/shafts.py

Leaf dict for CATALOGUE["construction"]["shafts"]. See catalogue.py's own
module docstring for the leaf shape and `requires` token rules.
"""

from __future__ import annotations

SHAFTS: dict = {
    "shafts.generic": {
        "description": (
            "N-section stepped shaft from an explicit SectionSpec "
            "list (make_shaft) -- full control over each section."
        ),
    },
    "shafts.stepped": {
        "description": (
            "N-section stepped shaft, one call (make_stepped_shaft) "
            "-- sections carry no shoulders; each of the N-1 "
            "internal Shoulder objects is derived from the two "
            "adjacent SectionSpec.diameter values plus one fillet "
            "radius, never re-typed by the caller."
        ),
    },
}