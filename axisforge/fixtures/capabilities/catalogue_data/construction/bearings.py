"""axisforge/fixtures/capabilities/catalogue_data/construction/bearings.py

Leaf dict for CATALOGUE["construction"]["bearings"]. See catalogue.py's
own module docstring for the leaf shape and `requires` token rules.
"""

from __future__ import annotations

BEARINGS: dict = {
    "bearings.deep_groove_ball": {
        "description": (
            "DGBB (make_deep_groove_ball_bearing) -- point contact, "
            "radial duty. Clearance-specified (s), not contact-"
            "angle-specified. i in {1, 2} rows."
        ),
    },
    "bearings.angular_contact": {
        "description": (
            "ACB (make_angular_contact_bearing) -- point contact, "
            "radial duty. Contact-angle-specified (alpha_0_deg), "
            "bounded to (0, 45] -- above 45deg is thrust duty. "
            "i in {1, 2} rows."
        ),
    },
    "bearings.self_aligning": {
        "description": (
            "Self-aligning ball bearing "
            "(make_self_aligning_ball_bearing) -- point contact, "
            "radial duty. Contact-angle-specified (alpha_0_deg), "
            "bounded to (0, 45]. Outer raceway radius re is derived "
            "from gamma(alpha_0), not a fixed ratio of Dw like the "
            "other two ball-radial families."
        ),
    },
    "bearings.thrust_ball_single_row": {
        "description": (
            "Single-row thrust ball bearing "
            "(make_thrust_ball_bearing) -- point contact, thrust "
            "duty. Contact-angle-specified (alpha_0_deg), bounded "
            "to (45, 90] -- 90 is pure thrust. Always exactly one "
            "row; see bearings.thrust_ball_multirow for i >= 2."
        ),
    },
    "bearings.thrust_ball_multirow": {
        "description": (
            "Multi-row (i >= 2) thrust ball bearing "
            "(make_thrust_ball_multirow_bearing) -- ONE catalog "
            "part / ONE Bearing built from a row_specs list, each "
            "row in SingleRowThrustBallFamily's own shape. Not i "
            "independent Bearings."
        ),
    },
    "bearings.cylindrical_roller": {
        "description": (
            "NU/N-type cylindrical roller bearing "
            "(make_cylindrical_roller_bearing) -- line contact, "
            "radial duty. arrangement must be 'floating' or "
            "'non-locating' (no flange to react axial load) -- the "
            "factory defaults to 'floating' rather than "
            "BearingCatalog's own 'locating' default."
        ),
    },
    "bearings.thrust_cylindrical_roller": {
        "description": (
            "Single-row cylindrical roller thrust bearing "
            "(make_thrust_cylindrical_roller_bearing) -- line "
            "contact, thrust duty. alpha_0_deg is a required kwarg "
            "(no class-level default)."
        ),
    },
    "bearings.thrust_cylindrical_roller_multirow": {
        "description": (
            "Multi-row (i >= 2) cylindrical roller thrust bearing "
            "(make_thrust_cylindrical_roller_multirow_bearing) -- "
            "ONE catalog part / ONE Bearing built from a row_specs "
            "list, same shape convention as "
            "bearings.thrust_ball_multirow."
        ),
    },
    "bearings.thrust_needle_roller": {
        "description": (
            "Needle roller thrust bearing "
            "(make_thrust_needle_roller_bearing) -- line contact, "
            "thrust duty. alpha_0 fixed at 90deg inside the family "
            "(flat-race); no alpha_0_deg parameter exists."
        ),
    },
}