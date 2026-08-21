"""
core/machine_elements/Bearings/families/roller/radial/subtypes/cylindrical_roller.py

CylindricalRollerFamily -- BearingFamily implementation for NU/N-type
cylindrical roller bearings, ISO/TS 16281 line contact, radial duty.

Owns: the arrangement guard (can't be "locating") and P_xk -- the
reference roller profile, Sec 6.2 eq.(42)-(44). P_xk lives here, NOT in
../functions/, because it's subtype-specific: a tapered/spherical roller
family (Phase 2) will need a different profile formula.

UPDATED, this turn -- `i` as input, mirroring DeepGrooveBallFamily
---------------------------------------------------------------------
assemble_geometry() now takes `i: int = 1`, stored on the bearing as
`bearing.i`, exactly like DeepGrooveBallFamily's own `i` parameter. This is
the SAME kind of row count as DGBB's -- a bearing catalogued and rated as
one physical unit with more than one closely-set row of rolling elements,
solved by ISO16281RollerSolver as ONE raceway (one delta_r, one set of Z
elements), with row count only entering the ISO/TS 16281 eq.(47)-(48)
Q_ci/Q_ce capacity formula as a multiplier -- NOT the
MultiRowCylindricalRollerFamily / ISO16281MultiRowRollerSolver kind of
multi-row (cylindrical_roller_multirow.py, roller_bearing_multirow_solver.py),
where `rows` are genuinely independent raceways solved with a load-split
compatibility condition. Two different "row count" concepts existing at two
different structural levels -- see cylindrical_roller_multirow.py's module
docstring for the same distinction spelled out from the other side.

Unlike DeepGrooveBallFamily, there is no REDUCTION_FACTOR_BY_ROWS lookup
table added here: DGBB's lambda-by-i table (ISO 281:2007 Table 1) is used
only by DeepGrooveBallFamily.dynamic_capacity() (the Cr-rating Formula
13-15), which CylindricalRollerFamily does not implement either (see the
existing "dynamic_capacity()/static_capacity() ... NotImplementedError"
note below, unchanged by this update) -- so there is currently no method on
this family that would consume a roller-specific row-count-dependent lambda
even if one were sourced. ISO 281:2007 Table 2 ("Radial roller bearings")
may or may not define its own row-count-dependent reduction factor the same
way Table 1 does for balls -- NOT confirmed here, no such table is
transcribed into this file. `i` is validated only as `>= 1`, not restricted
to a specific set like DGBB's `{1, 2}`, because that restriction would be
inventing a constraint from a table this session does not have in front of
it. Tighten the validation once Table 2's actual row-count support is
confirmed.

per_element_dynamic_capacity() previously took a standalone `i: int = 1`
keyword, disconnected from the bearing's own geometry (nothing enforced
that the capacity call's `i` matched how many rows the bearing was actually
assembled with, because assemble_geometry() didn't store a row count at
all). It now reads `i=bearing.i` internally instead, exactly mirroring
DeepGrooveBallFamily.per_element_dynamic_capacity() (which has no `i`
keyword at all, for the same reason). This is a breaking change to that
method's signature -- any caller passing `i=` explicitly needs updating;
see roller_bearing_solver.py's debug_radial_capacity(), updated alongside
this file, for the one in-repo call site affected.
"""
from __future__ import annotations
import numpy as np

from axisforge.core.machine_elements.Bearings.family import BearingFamily
from axisforge.core.machine_elements.Bearings.bearing_types import BearingType
from axisforge.core.machine_elements.Bearings.catalog import BearingCatalog
from ..functions import contact_stiffness as rc
from ..functions import capacity as rcap

_LOG_ARG_EPS = 1e-12  # floor for the log() argument in the profile function


class CylindricalRollerFamily(BearingFamily):
    """
    Usage
    -----
    bearing = Bearing.assemble(
        family=CylindricalRollerFamily(),
        catalog=BearingCatalog(d=20, D=47, b=14, C=28_500.0, C0=22_000.0,
                                designation="NU204", position=100.0,
                                arrangement="floating"),
        geometry=dict(Dwe=6.5, Lwe=6.0, Dpw=33.5, Z=12, s=0.015, n_s=40, i=1),
        analyses={"line_contact": True},
    )
    """

    BEARING_TYPE = BearingType.CYLINDRICAL_ROLLER
    DUTY = "radial"

    # ISO 281:2007 Table 2, Table No. 7 -- "Radial roller bearings". Owned
    # here, not in ../functions/capacity.py -- same reasoning as the ball
    # side's RI_OVER_DW/REDUCTION_FACTOR (subtype input, not generic math),
    # so a future tapered/spherical/needle roller subtype can declare its
    # own value without touching this one. Not (yet) row-count-dependent --
    # see module docstring.
    LAMBDA_V_RADIAL = 0.83

    CAPABILITIES = frozenset({"line_contact"})

    REQUIRED_FOR = {
        "line_contact": frozenset({
            "Dwe", "Lwe", "Dpw", "Z", "s", "n_s", "alpha_0",
            "x_k", "phi_j", "gamma", "cL", "cs", "P_xk", "i",
        }),
    }

    @property
    def name(self) -> str:
        return "cylindrical_roller"

    def assemble_geometry(self,
                           catalog: BearingCatalog,
                           Dwe: float,
                           Lwe: float,
                           Dpw: float,
                           Z: int,
                           s: float,
                           n_s: int = 30,
                           alpha_0_deg: float = 0.0,
                           i: int = 1) -> dict:
        """
        Dwe, Lwe, Dpw, Z, s, n_s, alpha_0_deg : as ever
        i : number of rows -- 1 (single, default) or more. Enters ONLY the
            ISO/TS 16281 eq.(47)-(48) Q_ci/Q_ce capacity formula as a
            multiplier, via per_element_dynamic_capacity() reading
            bearing.i -- see module docstring for why there is no
            REDUCTION_FACTOR_BY_ROWS table here unlike DeepGrooveBallFamily,
            and for the distinction from MultiRowCylindricalRollerFamily's
            `rows`-based multi-row.
        """
        if catalog.arrangement not in ("floating", "non-locating"):
            raise ValueError(
                f"CylindricalRollerFamily(label={catalog.label or catalog.designation!r}): "
                f"arrangement={catalog.arrangement!r} is not valid -- an NU/N-type "
                f"bearing has no flange to react axial load, so it cannot be "
                f"'locating'. Use 'floating' or 'non-locating' on the BearingCatalog."
            )
        if n_s < 30:
            raise ValueError(f"CylindricalRollerFamily: n_s must be >= 30, got {n_s}")
        if s < 0:
            raise ValueError(f"CylindricalRollerFamily: s must be >= 0, got {s}")
        if i < 1:
            raise ValueError(f"CylindricalRollerFamily: i (number of rows) must be >= 1, got {i}")

        alpha_0 = np.radians(alpha_0_deg)
        x_k = rc.lamina_positions(Lwe, n_s)
        phi_j = np.linspace(0, 2 * np.pi, Z, endpoint=False)
        g = rc.gamma(Dwe, Dpw, alpha_0)
        cL, cs_val = rc.line_contact_spring_constant(Lwe, n_s)
        P_xk = self._reference_roller_profile(x_k, Dwe, Lwe)

        return dict(
            bearing_type=self.BEARING_TYPE,
            Dwe=Dwe, Lwe=Lwe, Dpw=Dpw, Z=Z, s=s, n_s=n_s, alpha_0=alpha_0,
            x_k=x_k, phi_j=phi_j, gamma=g, cL=cL, cs=cs_val, P_xk=P_xk,
            i=i,
        )

    @staticmethod
    def _reference_roller_profile(x_k: np.ndarray, Dwe: float, Lwe: float) -> np.ndarray:
        """P(x_k) [mm] -- ISO/TS 16281 Sec 6.2 eq.(42)-(44)."""
        P = np.zeros_like(x_k)
        if Lwe <= 2.5 * Dwe:
            arg = 1.0 - (2.0 * x_k / Lwe) ** 2
            arg = np.maximum(arg, _LOG_ARG_EPS)
            P = 0.000350 * Dwe * np.log(1.0 / arg)
        else:
            half_flat = (Lwe - 2.5 * Dwe) / 2.0
            edge = np.abs(x_k) > half_flat
            if np.any(edge):
                xe = x_k[edge]
                arg = 1.0 - ((2.0 * np.abs(xe) - (Lwe - 2.5 * Dwe)) / (2.5 * Dwe)) ** 2
                arg = np.maximum(arg, _LOG_ARG_EPS)
                P[edge] = 0.000500 * Dwe * np.log(1.0 / arg)
        return P

    # ------------------------------------------------------------------
    # Capacity -- given an assembled Bearing (this family), pulls the
    # right attributes and calls the shared functions/capacity.py math.
    # dynamic_capacity()/static_capacity() upstream (BearingCapacity) are
    # still NotImplementedError -- f_c/f_0 text not provided yet.
    # ------------------------------------------------------------------

    @staticmethod
    def per_element_dynamic_capacity(bearing, Cr: float | None = None,
                                      lambda_v: float | None = None) -> tuple[float, float]:
        """
        (Q_ci, Q_ce) [N] -- whole-roller dynamic capacity, ISO/TS 16281
        Sec 5.3.1.2 eq.(47)-(48). Cr defaults to bearing.C (catalog value)
        if not overridden.

        CHANGED, this turn: `i` is no longer a keyword here -- reads
        bearing.i (set at assemble_geometry() time) instead, mirroring
        DeepGrooveBallFamily.per_element_dynamic_capacity(). See module
        docstring for why.
        """
        return rcap.RollingElementCapacity.radial(
            Z=bearing.Z, alpha_0=bearing.alpha_0, gamma=bearing.gamma,
            Cr=Cr if Cr is not None else bearing.C,
            lambda_v=lambda_v if lambda_v is not None else CylindricalRollerFamily.LAMBDA_V_RADIAL,
            i=bearing.i,
        )

    @staticmethod
    def per_lamina_dynamic_capacity(bearing, Q_ci: float, Q_ce: float) -> tuple[float, float]:
        """(q_ci, q_ce) [N] -- per-lamina dynamic load rating, ISO/TS 16281
        Sec 5.3.2 eq.(56)-(57). Takes the whole-roller Q_ci/Q_ce from
        per_element_dynamic_capacity() -- not resolved here."""
        return rcap.RollingElementCapacity.per_lamina(Q_ci, Q_ce, bearing.n_s)