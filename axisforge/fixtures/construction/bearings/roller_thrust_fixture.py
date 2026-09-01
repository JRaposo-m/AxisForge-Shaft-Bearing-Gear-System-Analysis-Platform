"""
fixtures/construction/bearings/roller_thrust_fixture.py

Construction-stage factories for the three thrust-duty roller bearing
families (core/machine_elements/bearings/families/roller_bearing/thrust/):
ThrustCylindricalRollerFamily, MultiRowThrustCylindricalRollerFamily,
ThrustNeedleRollerFamily -- all line contact.

Import note: MultiRowThrustCylindricalRollerFamily is imported straight
from its own leaf module (.cylindrical_multirow), NOT via
families/roller_bearing/thrust/subtypes/__init__.py's lazy re-export --
that __init__'s _LAZY dict maps this name to ".cylindrical_single" (a
copy/paste bug; its own TYPE_CHECKING block two lines below says
".cylindrical_multirow", which is what's actually correct and what
cylindrical_multirow.py actually defines). Not touched here -- core is
out of scope for Construction fixtures -- but flagging so this import
doesn't look like an inconsistency with the ball-side fixture.

make_thrust_cylindrical_roller_bearing() : one row. alpha_0_deg is a
    REQUIRED kwarg here (unlike CylindricalRollerFamily on the radial
    side, this family has no default for it) -- verified against the
    real assemble_geometry() signature, not assumed from the family's
    fixed-90deg framing in its own module docstring.

make_thrust_cylindrical_roller_multirow_bearing() : same row_specs-list
    shape as ball_thrust_fixture.py's multirow factory -- row dicts use
    ThrustCylindricalRollerFamily's own kwargs (Dwe, Lwe, Dpw, Z, s,
    alpha_0_deg, n_s=30).

make_thrust_needle_roller_bearing() : alpha_0 is fixed internally by the
    family (ALPHA_0_DEG = 90.0, flat-race) -- there is no alpha_0_deg
    parameter on this family's assemble_geometry() at all, so none is
    exposed here either.
"""
from __future__ import annotations
from typing import Any

from axisforge.core.machine_elements.bearings.bearing import Bearing
from axisforge.core.machine_elements.bearings.catalog import BearingCatalog
from axisforge.core.machine_elements.bearings.families.roller_bearing.thrust.subtypes.cylindrical_single import (
    ThrustCylindricalRollerFamily,
)
from axisforge.core.machine_elements.bearings.families.roller_bearing.thrust.subtypes.cylindrical_multirow import (
    MultiRowThrustCylindricalRollerFamily,
)
from axisforge.core.machine_elements.bearings.families.roller_bearing.thrust.subtypes.needle_single import (
    ThrustNeedleRollerFamily,
)


def make_thrust_cylindrical_roller_bearing(
    *,
    d: float,
    D: float,
    Dwe: float,
    Lwe: float,
    Dpw: float,
    Z: int,
    s: float,
    alpha_0_deg: float,
    b: float = 0.0,
    C: float = 0.0,
    C0: float = 0.0,
    designation: str = "",
    label: str = "",
    position: float = 0.0,
    arrangement: str = "locating",
    n_s: int = 30,
) -> Bearing:
    """Single-row cylindrical roller thrust bearing -- line contact,
    thrust duty. alpha_0_deg is required (no class-level default in the
    family)."""
    catalog = BearingCatalog(
        d=d, D=D, b=b, C=C, C0=C0, designation=designation,
        label=label, position=position, arrangement=arrangement,
    )
    return Bearing.assemble(
        family=ThrustCylindricalRollerFamily(),
        catalog=catalog,
        geometry=dict(Dwe=Dwe, Lwe=Lwe, Dpw=Dpw, Z=Z, s=s, alpha_0_deg=alpha_0_deg, n_s=n_s),
        analyses={"line_contact": True},
    )


def make_thrust_cylindrical_roller_multirow_bearing(
    *,
    d: float,
    D: float,
    row_specs: list[dict[str, Any]],
    b: float = 0.0,
    C: float = 0.0,
    C0: float = 0.0,
    designation: str = "",
    label: str = "",
    position: float = 0.0,
    arrangement: str = "locating",
) -> Bearing:
    """Multi-row (i >= 2) cylindrical roller thrust bearing, ONE catalog
    part.

    row_specs : list of kwargs dicts, one per row, each exactly
        ThrustCylindricalRollerFamily.assemble_geometry()'s own shape
        (Dwe, Lwe, Dpw, Z, s, alpha_0_deg, n_s=30). Forwarded verbatim.
    """
    catalog = BearingCatalog(
        d=d, D=D, b=b, C=C, C0=C0, designation=designation,
        label=label, position=position, arrangement=arrangement,
    )
    return Bearing.assemble(
        family=MultiRowThrustCylindricalRollerFamily(),
        catalog=catalog,
        geometry=dict(row_specs=row_specs),
        analyses={"multirow_capacity": True},
    )


def make_thrust_needle_roller_bearing(
    *,
    d: float,
    D: float,
    Dwe: float,
    Lwe: float,
    Dpw: float,
    Z: int,
    s: float,
    b: float = 0.0,
    C: float = 0.0,
    C0: float = 0.0,
    designation: str = "",
    label: str = "",
    position: float = 0.0,
    arrangement: str = "locating",
    n_s: int = 30,
) -> Bearing:
    """Needle roller thrust bearing -- line contact, thrust duty.
    alpha_0 is fixed at 90deg inside the family -- no alpha_0_deg kwarg
    exists on assemble_geometry(), so none is exposed here."""
    catalog = BearingCatalog(
        d=d, D=D, b=b, C=C, C0=C0, designation=designation,
        label=label, position=position, arrangement=arrangement,
    )
    return Bearing.assemble(
        family=ThrustNeedleRollerFamily(),
        catalog=catalog,
        geometry=dict(Dwe=Dwe, Lwe=Lwe, Dpw=Dpw, Z=Z, s=s, n_s=n_s),
        analyses={"line_contact": True},
    )