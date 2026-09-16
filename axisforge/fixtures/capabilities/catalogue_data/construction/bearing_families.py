"""axisforge/fixtures/capabilities/catalogue_data/construction/bearing_families.py

Leaf dict for CATALOGUE["construction"]["bearing_families"]. See
catalogue.py's own module docstring for the leaf shape and `requires`
token rules.
"""

from __future__ import annotations

BEARING_FAMILIES: dict = {
    "bearing_families.deep_groove_ball": {
        "description": (
            "Bare DeepGrooveBallFamily class, for a script that "
            "wants to call Bearing.assemble() itself instead of "
            "going through bearings.deep_groove_ball's factory."
        ),
    },
    "bearing_families.angular_contact": {
        "description": (
            "Bare AngularContactFamily class -- see "
            "bearing_families.deep_groove_ball for why this domain "
            "exists."
        ),
    },
    "bearing_families.self_aligning": {
        "description": (
            "Bare SelfAligningBallFamily class -- see "
            "bearing_families.deep_groove_ball for why this domain "
            "exists."
        ),
    },
    "bearing_families.thrust_ball_single_row": {
        "description": (
            "Bare SingleRowThrustBallFamily class -- see "
            "bearing_families.deep_groove_ball for why this domain "
            "exists."
        ),
    },
    "bearing_families.thrust_ball_multirow": {
        "description": (
            "Bare MultiRowThrustBallFamily class -- see "
            "bearing_families.deep_groove_ball for why this domain "
            "exists."
        ),
    },
    "bearing_families.cylindrical_roller": {
        "description": (
            "Bare CylindricalRollerFamily class -- see "
            "bearing_families.deep_groove_ball for why this domain "
            "exists."
        ),
    },
    "bearing_families.thrust_cylindrical_roller": {
        "description": (
            "Bare ThrustCylindricalRollerFamily class -- see "
            "bearing_families.deep_groove_ball for why this domain "
            "exists."
        ),
    },
    "bearing_families.thrust_cylindrical_roller_multirow": {
        "description": (
            "Bare MultiRowThrustCylindricalRollerFamily class -- "
            "see bearing_families.deep_groove_ball for why this "
            "domain exists."
        ),
    },
    "bearing_families.thrust_needle_roller": {
        "description": (
            "Bare ThrustNeedleRollerFamily class -- see "
            "bearing_families.deep_groove_ball for why this domain "
            "exists."
        ),
    },
}