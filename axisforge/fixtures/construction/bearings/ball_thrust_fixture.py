"""
fixtures/construction/bearings/ball_thrust_fixture.py

Construction-stage factories for the two thrust-duty ball bearing
families (core/machine_elements/bearings/families/ball_bearing/thrust/):
SingleRowThrustBallFamily and MultiRowThrustBallFamily.

make_thrust_ball_bearing() : one row, point contact, alpha_0_deg in
    (45, 90] -- 90 = pure thrust. Mirrors ball_radial_fixture.py's
    factories exactly except no `i` (SingleRowThrustBallFamily is always
    exactly one row -- multi-row lives in the OTHER factory below, not
    as an `i` kwarg here; see that family's own module docstring for why).

make_thrust_ball_multirow_bearing() : ONE Bearing (one BearingCatalog,
    per Bearing.assemble()'s own contract) built from >= 2 rows of balls.
    Takes `row_specs: list[dict]` directly and forwards it verbatim to
    MultiRowThrustBallFamily.assemble_geometry(row_specs=...) -- no
    n_rows-plus-shared-kwargs convenience wrapper. Caller builds the list
    themselves: repeat the same dict i times for identical rows (the
    common case), or pass i distinct dicts for heterogeneous rows -- both
    go through the same code path in the family, so no special-casing is
    needed here either. Each row dict uses exactly
    SingleRowThrustBallFamily.assemble_geometry()'s own kwargs shape:
    Dw, Dpw, Z, E, alpha_0_deg, nu=0.3 (no `i` inside a row dict --
    row count is len(row_specs), not a per-row field).
    MultiRowThrustBallFamily.assemble_geometry() itself raises ValueError
    if len(row_specs) < 2 -- not re-checked here.
"""
from __future__ import annotations
from typing import Any

from axisforge.core.machine_elements.bearings.bearing import Bearing
from axisforge.core.machine_elements.bearings.catalog import BearingCatalog
from axisforge.core.machine_elements.bearings.families.ball_bearing.thrust.subtypes.thrust_single import (
    SingleRowThrustBallFamily,
)
from axisforge.core.machine_elements.bearings.families.ball_bearing.thrust.subtypes.thrust_multirow import (
    MultiRowThrustBallFamily,
)


def make_thrust_ball_bearing(
    *,
    d: float,
    D: float,
    Dw: float,
    Dpw: float,
    Z: int,
    E: float,
    alpha_0_deg: float,
    b: float = 0.0,
    C: float = 0.0,
    C0: float = 0.0,
    designation: str = "",
    label: str = "",
    position: float = 0.0,
    arrangement: str = "locating",
    nu: float = 0.3,
) -> Bearing:
    """Single-row thrust ball bearing. alpha_0_deg must be in (45, 90] --
    enforced by the family itself, not re-checked here."""
    catalog = BearingCatalog(
        d=d, D=D, b=b, C=C, C0=C0, designation=designation,
        label=label, position=position, arrangement=arrangement,
    )
    return Bearing.assemble(
        family=SingleRowThrustBallFamily(),
        catalog=catalog,
        geometry=dict(Dw=Dw, Dpw=Dpw, Z=Z, E=E, alpha_0_deg=alpha_0_deg, nu=nu),
        analyses={"point_contact": True},
    )


def make_thrust_ball_multirow_bearing(
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
    """Multi-row (i >= 2) thrust ball bearing, ONE catalog part.

    row_specs : list of kwargs dicts, one per row, each exactly
        SingleRowThrustBallFamily.assemble_geometry()'s own shape
        (Dw, Dpw, Z, E, alpha_0_deg, nu=0.3). Forwarded verbatim --
        this factory does not repeat/expand it.
    """
    catalog = BearingCatalog(
        d=d, D=D, b=b, C=C, C0=C0, designation=designation,
        label=label, position=position, arrangement=arrangement,
    )
    return Bearing.assemble(
        family=MultiRowThrustBallFamily(),
        catalog=catalog,
        geometry=dict(row_specs=row_specs),
        analyses={"multirow_capacity": True},
    )