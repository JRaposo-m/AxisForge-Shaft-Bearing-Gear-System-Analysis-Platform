"""
core/machine_elements/Bearings/families/ball/thrust/subtypes/thrust_ball_multirow.py

MultiRowThrustBallFamily -- BearingFamily for a thrust ball bearing bought
as ONE catalog part but built from i >= 2 rows of balls (ISO 281:2007
Sec 6.4).

Why this isn't "call Bearing.assemble() i times": Bearing.__init__ takes a
single BearingCatalog (see bearing.py) -- a multi-row thrust bearing's
rows don't each have their own catalog entry, C/C0/d/D/b are bearing-level
values for the one part actually bought, not per-row. There is no such
thing as "the catalog for just one row" to hand to i separate
Bearing.assemble() calls. So this family does NOT produce i Bearing
objects -- it produces exactly ONE Bearing (one catalog, one family, per
Bearing.assemble()'s contract), whose `rows` attribute holds the list of
i per-row geometry dicts.

assemble_geometry() takes row_specs -- a list of kwargs dicts, one per
row, in the exact shape SingleRowThrustBallFamily.assemble_geometry()
already accepts (Dw, Dpw, Z, E, alpha_0_deg, nu). Built by the caller: the
same dict repeated i times for identical rows (the common case), or i
distinct dicts for heterogeneous rows -- both go through the same code
path here, no special-casing needed. Each row_specs[j] is forwarded
directly to SingleRowThrustBallFamily().assemble_geometry(catalog, **spec)
-- this reuses that family's point-contact math (contact_stiffness calls,
Table No. 4 lookups, alpha_0 bound checks) instead of duplicating it.

CAPABILITIES is deliberately NOT "point_contact" -- per-element Q_ci/Q_ce
across multiple rows (which row shares how much load, phi_j per row, a
combined stiffness matrix, etc.) is a genuinely different, harder problem
than aggregating Ca, and hasn't been asked for yet. This family only
supports "multirow_capacity" (dynamic_capacity() -> Ca via Formula (29)).
If/when per-row point-contact solving across rows is needed, that's a new
capability to design and add here -- not a reason to change what's
implemented now. Flagging this explicitly since it's a scope choice, not
a technical constraint -- rename/extend if you want it sooner.

Reference:
  ISO 281:2007 Sec 6.4, Formula (26)-(29) -- Ca, two or more rows of balls.
"""
from __future__ import annotations
from typing import Any

from axisforge.core.machine_elements.Bearings.family import BearingFamily
from axisforge.core.machine_elements.Bearings.bearing_types import BearingType
from axisforge.core.machine_elements.Bearings.catalog import BearingCatalog
from ..functions import capacity as bcap
from .thrust_ball import SingleRowThrustBallFamily


class MultiRowThrustBallFamily(BearingFamily):
    """
    Usage
    -----
    row = dict(Dw=9.0, Dpw=54.0, Z=14, E=206000, alpha_0_deg=90.0)
    bearing = Bearing.assemble(
        family=MultiRowThrustBallFamily(),
        catalog=BearingCatalog(d=40, D=68, b=30, C=28000, C0=44000,
                                designation="52208", position=50.0),
        geometry=dict(row_specs=[row, row]),   # i=2 identical rows here;
                                                 # pass distinct dicts for
                                                 # heterogeneous rows
        analyses={"multirow_capacity": True},
    )
    bearing.family.dynamic_capacity(bearing)   # Ca [N], Formula (29)
    bearing.rows[0]["Dw"]                      # per-row fields, e.g. row 0's Dw
    """

    BEARING_TYPE = BearingType.THRUST_BALL
    DUTY = "thrust"

    CAPABILITIES = frozenset({"multirow_capacity"})

    REQUIRED_FOR = {
        "multirow_capacity": frozenset({"rows", "i"}),
    }

    @property
    def name(self) -> str:
        return "thrust_ball_multirow"

    def assemble_geometry(self, catalog: BearingCatalog, row_specs: list[dict]) -> dict[str, Any]:
        """
        row_specs : list of kwargs, one dict per row, each forwarded as-is
                    to SingleRowThrustBallFamily.assemble_geometry(catalog, **spec)
                    -- same keys that family takes (Dw, Dpw, Z, E,
                    alpha_0_deg, nu).

        Returns bearing_type/rows/i only -- deliberately NOT ri/re/Dw/...
        at the top level, those only mean something per-row (see module
        docstring). Callers needing a specific row's fields go through
        bearing.rows[j][...].
        """
        if len(row_specs) < 2:
            raise ValueError(
                f"MultiRowThrustBallFamily needs >= 2 row_specs (i > 1); got "
                f"{len(row_specs)}. Use SingleRowThrustBallFamily for one row."
            )

        single = SingleRowThrustBallFamily()
        rows = [single.assemble_geometry(catalog, **spec) for spec in row_specs]

        return dict(
            bearing_type=self.BEARING_TYPE,
            rows=rows,
            i=len(rows),
        )

    @staticmethod
    def dynamic_capacity(bearing) -> float:
        """
        Ca [N] -- ISO 281:2007 Formula (29), Sec 6.4.

        Computes each row's single-row Ca via
        SingleRowThrustBallFamily.dynamic_capacity_from_fields() (Sec 6.3,
        dispatched per row on that row's own alpha_0 -- rows need not
        share the same contact angle), then combines all of them via
        BearingCapacity.dynamic_multirow().
        """
        Ca_rows = [
            SingleRowThrustBallFamily.dynamic_capacity_from_fields(
                Z=row["Z"], Dw=row["Dw"], alpha_0=row["alpha_0"],
                ri=row["ri"], re=row["re"], gamma=row["gamma"],
                reduction_factor=row["reduction_factor"],
            )
            for row in bearing.rows
        ]
        Z_rows = [row["Z"] for row in bearing.rows]
        return bcap.BearingCapacity.dynamic_multirow(Z_rows=Z_rows, Ca_rows=Ca_rows)