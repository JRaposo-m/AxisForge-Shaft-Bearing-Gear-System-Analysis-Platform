"""
core/machine_elements/Bearings/families/roller/thrust/subtypes/thrust_cylindrical_roller_multirow.py

MultiRowThrustCylindricalRollerFamily -- BearingFamily for a cylindrical
roller thrust bearing bought as ONE catalog part but built from i >= 2
rows of rollers (ISO 281:2007 Formula (46)).

Same reasoning and shape as
../../../ball/thrust/subtypes/thrust_ball_multirow.py's
MultiRowThrustBallFamily -- read that module's docstring for the full
"why this isn't Bearing.assemble() called i times" argument; it applies
here unchanged (Bearing.__init__ still takes one BearingCatalog, a
multi-row thrust bearing's rows still don't each have their own catalog
entry). This family produces exactly ONE Bearing, whose `rows` attribute
holds the list of i per-row geometry dicts.

assemble_geometry() takes row_specs -- a list of kwargs dicts, one per
row, in the exact shape ThrustCylindricalRollerFamily.assemble_geometry()
already accepts (Dwe, Lwe, Dpw, Z, s, n_s). No alpha_0_deg key -- that
family fixes ALPHA_0_DEG = 90.0 itself (flat-race, pure axial), so it
isn't a per-row input here either. Each row_specs[j] is forwarded
directly to ThrustCylindricalRollerFamily().assemble_geometry(catalog,
**spec) -- this reuses that family's line-contact math (P_xk profile,
gamma, cL/cs) instead of duplicating it, same reuse pattern as the ball
side reusing SingleRowThrustBallFamily.

CAPABILITIES is "multirow_capacity" only, same scope choice and same
caveat as the ball side: per-element Q_ci/Q_ce across multiple rows is a
genuinely different, harder problem than aggregating Ca, and hasn't been
asked for yet. If/when it is, that's a new capability to design and add
here -- not a reason to change what's implemented now.

DIVERGES from MultiRowThrustBallFamily.dynamic_capacity(bearing) in one
way, flagged rather than silently copied: that method takes ONLY
`bearing` and reads `row["reduction_factor"]` off each row dict -- which
implies SingleRowThrustBallFamily.assemble_geometry() stores
reduction_factor (and presumably lam/eta) as part of each row's geometry.
I have not seen that file's actual contents in this session, only this
multirow wrapper, so I'm not assuming its exact shape.
ThrustCylindricalRollerFamily.assemble_geometry() (this duty, this
family's own single-row sibling) does NOT take reduction_factor/nu/eta at
all -- ThrustCylindricalRollerFamily.dynamic_capacity() takes them as
plain call-time arguments instead, never stored geometry (see that file,
and functions/capacity.py's module docstring, for why -- none of the
three has a confirmed ISO 281:2007 Table 2 value transcribed into this
codebase yet). dynamic_capacity() below follows THAT convention -- the
one actually implemented on this duty's own single-row family -- rather
than guessing at the ball side's storage-based approach. Not reconciled
across families without you confirming which convention should win; this
is a design choice, not a technical necessity.

Reference:
  ISO 281:2007 Formula (46) -- Ca, two or more rows of rollers.
"""
from __future__ import annotations
from typing import Any, Sequence, Union

from axisforge.core.machine_elements.bearings.family import BearingFamily
from axisforge.core.machine_elements.bearings.bearing_types import BearingType
from axisforge.core.machine_elements.bearings.catalog import BearingCatalog
from ..functions import capacity as rcap
from .cylindrical_single import ThrustCylindricalRollerFamily


class MultiRowThrustCylindricalRollerFamily(BearingFamily):
    """
    Usage
    -----
    row = dict(Dwe=8.0, Lwe=8.0, Dpw=72.5, Z=18, s=0.01, n_s=40)
    bearing = Bearing.assemble(
        family=MultiRowThrustCylindricalRollerFamily(),
        catalog=BearingCatalog(d=60, D=85, b=17, C=110_000.0, C0=200_000.0,
                                designation="81212M", position=100.0),
        geometry=dict(row_specs=[row, row]),   # i=2 identical rows here;
                                                 # pass distinct dicts for
                                                 # heterogeneous rows
        analyses={"multirow_capacity": True},
    )
    bearing.family.dynamic_capacity(bearing, reduction_factor=1.0, nu=1.0, eta=1.0)  # Ca [N], Formula (46)
    bearing.rows[0]["Dwe"]                      # per-row fields, e.g. row 0's Dwe
    """

    BEARING_TYPE = BearingType.THRUST_CYLINDRICAL_ROLLER
    DUTY = "thrust"

    CAPABILITIES = frozenset({"multirow_capacity"})

    REQUIRED_FOR = {
        "multirow_capacity": frozenset({"rows", "i"}),
    }

    @property
    def name(self) -> str:
        return "thrust_cylindrical_roller_multirow"

    def assemble_geometry(self, catalog: BearingCatalog, row_specs: list[dict]) -> dict[str, Any]:
        """
        row_specs : list of kwargs, one dict per row, each forwarded as-is
                    to ThrustCylindricalRollerFamily.assemble_geometry(catalog, **spec)
                    -- same keys that family takes (Dwe, Lwe, Dpw, Z, s, n_s).

        Returns bearing_type/rows/i only -- deliberately NOT Dwe/Lwe/...
        at the top level, those only mean something per-row (see module
        docstring). Callers needing a specific row's fields go through
        bearing.rows[j][...].
        """
        if len(row_specs) < 2:
            raise ValueError(
                f"MultiRowThrustCylindricalRollerFamily needs >= 2 row_specs (i > 1); "
                f"got {len(row_specs)}. Use ThrustCylindricalRollerFamily for one row."
            )

        single = ThrustCylindricalRollerFamily()
        rows = [single.assemble_geometry(catalog, **spec) for spec in row_specs]

        return dict(
            bearing_type=self.BEARING_TYPE,
            rows=rows,
            i=len(rows),
        )

    @staticmethod
    def dynamic_capacity(bearing,
                          reduction_factor: Union[float, Sequence[float]],
                          nu: Union[float, Sequence[float]],
                          eta: Union[float, Sequence[float]]) -> float:
        """
        Ca [N] -- ISO 281:2007 Formula (46).

        Computes each row's single-row Ca via
        BearingCapacity.dynamic_90deg() (Sec 6.6.2 -- every row of this
        family is alpha_0 = 90deg, since ThrustCylindricalRollerFamily
        fixes that for the row it builds), then combines all of them via
        BearingCapacity.dynamic_multirow(). See module docstring's
        "DIVERGES" note for why reduction_factor/nu/eta are plain
        arguments here rather than pulled off bearing.rows.

        reduction_factor, nu, eta : each either a single value (broadcast
        to every row) or a sequence of length i (one value per row) --
        rows need not share the same reduction_factor/nu/eta any more than
        they need share Z/Dwe/Lwe.
        """
        n = len(bearing.rows)

        def _per_row(value, label):
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                if len(value) != n:
                    raise ValueError(f"{label} sequence length ({len(value)}) must match row count ({n}).")
                return list(value)
            return [value] * n

        reduction_factor_rows = _per_row(reduction_factor, "reduction_factor")
        nu_rows = _per_row(nu, "nu")
        eta_rows = _per_row(eta, "eta")

        Ca_rows = [
            rcap.BearingCapacity.dynamic_90deg(
                Z=row["Z"], Dwe=row["Dwe"], Lwe=row["Lwe"], gamma=row["gamma"],
                reduction_factor=rf, nu=nu_j, eta=eta_j,
            )
            for row, rf, nu_j, eta_j in zip(bearing.rows, reduction_factor_rows, nu_rows, eta_rows)
        ]
        Z_rows = [row["Z"] for row in bearing.rows]
        Lwe_rows = [row["Lwe"] for row in bearing.rows]
        return rcap.BearingCapacity.dynamic_multirow(Z_rows=Z_rows, Lwe_rows=Lwe_rows, Ca_rows=Ca_rows)