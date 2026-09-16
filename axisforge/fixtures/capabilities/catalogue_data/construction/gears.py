"""axisforge/fixtures/capabilities/catalogue_data/construction/gears.py

Leaf dict for CATALOGUE["construction"]["gears"]. See catalogue.py's own
module docstring for the leaf shape and `requires` token rules.
"""

from __future__ import annotations

GEARS: dict = {
    "gears.spur": {
        "description": (
            "External spur gear (make_spur_gear) -- SpurHelicalGear "
            "with beta_n_deg pinned to 0.0, not exposed as a "
            "parameter. Construction only, no meshing."
        ),
    },
    "gears.helical": {
        "description": (
            "External helical gear (make_helical_gear) -- "
            "SpurHelicalGear with beta_n_deg required and "
            "validated > 0 (0 is rejected, use gears.spur instead). "
            "Construction only, no meshing."
        ),
    },
    "gears.internal": {
        "description": (
            "Internal (ring) gear (make_internal_gear) -- "
            "InternalGear, a different core class from external "
            "spur/helical. z is required to be NEGATIVE here "
            "(e.g. z=-60 for 60 teeth) -- the core's da/df formulas "
            "use z/abs(z) as a sign switch, and only z<0 makes the "
            "addendum genuinely shrink the bore (da<d), as a real "
            "internal gear's tip circle must. Construction only, "
            "no meshing."
        ),
    },
    "gears.spur_helical_meshing": {
        "description": (
            "Fixed-axis pair of two external gears "
            "(make_spur_helical_meshing) -- SpurHelicalGearMeshing: "
            "working centre distance, working pressure angle, "
            "contact ratios (epsilon_alpha/beta/gamma), "
            "interference/undercut checks via .validate(). Takes "
            "two already-built SpurHelicalGear objects (gears.spur/"
            "gears.helical), does not build them. Does not call "
            ".forces() -- that needs a real driving torque, a "
            "MeshLoads-stage concern."
        ),
        # Kept AND, exactly as drafted -- but flagging, not
        # silently fixing: the pair can be spur+spur,
        # helical+helical or one of each, so gears.spur alone
        # (or gears.helical alone) already covers "can build an
        # external gear" for BOTH slots. Requiring both tokens
        # forces a script that only ever wants two spur gears to
        # also request gears.helical for nothing. Same shape as
        # the gears.internal_meshing fix below -- say if you
        # want this one changed to
        # {"construction": (("gears.spur", "gears.helical"),)}
        # (an OR-group of one) too.
        "requires": {"construction": ("gears.helical", "gears.spur")},
    },
    "gears.internal_meshing": {
        "description": (
            "Fixed-axis external-pinion + internal-ring pair "
            "(make_internal_meshing) -- InternalGearMeshing: same "
            "working-geometry/contact-ratio outputs as "
            "gears.spur_helical_meshing, plus interference checks "
            "delegated to InternalGear.validate_mesh(). Takes an "
            "already-built SpurHelicalGear (pinion, gears.spur/"
            "helical) and InternalGear (ring, gears.internal; z<0 "
            "is re-checked here as a guard). Does not call "
            ".forces()."
        ),
        # AND(gears.internal, OR(gears.spur, gears.helical)) --
        # the ring is always gears.internal, but the pinion is
        # EITHER an external spur OR helical gear, never both at
        # once, so the pinion slot is a nested OR-group rather
        # than two ANDed exact tokens.
        "requires": {
            "construction": ("gears.internal", ("gears.spur", "gears.helical")),
        },
    },
}