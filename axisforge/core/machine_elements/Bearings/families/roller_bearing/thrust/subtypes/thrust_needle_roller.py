"""
core/machine_elements/Bearings/families/roller/thrust/subtypes/thrust_needle_roller.py

ThrustNeedleRollerFamily -- BearingFamily implementation for needle roller
thrust bearings, ISO/TS 16281 line contact, thrust duty.

Same as ThrustCylindricalRollerFamily in every respect -- reference roller
profile P_xk confirmed identical (same roller, same profile math, ISO/TS
16281 Sec 6.2 eq.(42)-(44)), alpha_0 fixed at 90deg, LAMBDA_V_THRUST from
the same ISO 281:2007 Table 2, Table No. 10 row ("Thrust roller bearings"
-- not split by roller shape). Kept as its own file/class rather than
folded into ThrustCylindricalRollerFamily because a needle roller's own
geometry constraints (Lwe/Dwe ratio, catalog fields) may diverge later --
same reasoning as keeping DGBB and ACB separate despite sharing math.
"""
from __future__ import annotations
import numpy as np

from axisforge.core.machine_elements.Bearings.family import BearingFamily
from axisforge.core.machine_elements.Bearings.bearing_types import BearingType
from axisforge.core.machine_elements.Bearings.catalog import BearingCatalog
from ..functions import contact_stiffness as rc
from ..functions import capacity as rcap

_LOG_ARG_EPS = 1e-12  # floor for the log() argument in the profile function


class ThrustNeedleRollerFamily(BearingFamily):
    """
    Usage
    -----
    bearing = Bearing.assemble(
        family=ThrustNeedleRollerFamily(),
        catalog=BearingCatalog(d=20, D=35, b=2.5, C=25_000.0, C0=50_000.0,
                                designation="AXK2035", position=100.0),
        geometry=dict(Dwe=2.5, Lwe=13.8, Dpw=27.5, Z=20, s=0.01, n_s=40),
        analyses={"line_contact": True},
    )
    """

    BEARING_TYPE = BearingType.THRUST_NEEDLE_ROLLER
    DUTY = "thrust"

    CAPABILITIES = frozenset({"line_contact"})

    REQUIRED_FOR = {
        "line_contact": frozenset({
            "Dwe", "Lwe", "Dpw", "Z", "s", "n_s", "alpha_0",
            "x_k", "phi_j", "gamma", "cL", "cs", "P_xk",
        }),
    }

    ALPHA_0_DEG = 90.0   # fixed -- flat-race, pure axial. See module docstring.

    # ISO 281:2007 Table 2, Table No. 10 -- "Thrust roller bearings".
    LAMBDA_V_THRUST = 0.73

    @property
    def name(self) -> str:
        return "thrust_needle_roller"

    def assemble_geometry(self,
                           catalog: BearingCatalog,
                           Dwe: float,
                           Lwe: float,
                           Dpw: float,
                           Z: int,
                           s: float,
                           n_s: int = 30) -> dict:
        if n_s < 30:
            raise ValueError(f"ThrustNeedleRollerFamily: n_s must be >= 30, got {n_s}")
        if s < 0:
            raise ValueError(f"ThrustNeedleRollerFamily: s must be >= 0, got {s}")

        alpha_0 = np.radians(self.ALPHA_0_DEG)
        x_k = rc.lamina_positions(Lwe, n_s)
        phi_j = np.linspace(0, 2 * np.pi, Z, endpoint=False)
        g = rc.gamma(Dwe, Dpw, alpha_0)
        cL, cs_val = rc.line_contact_spring_constant(Lwe, n_s)
        P_xk = self._reference_roller_profile(x_k, Dwe, Lwe)

        return dict(
            bearing_type=self.BEARING_TYPE,
            Dwe=Dwe, Lwe=Lwe, Dpw=Dpw, Z=Z, s=s, n_s=n_s, alpha_0=alpha_0,
            x_k=x_k, phi_j=phi_j, gamma=g, cL=cL, cs=cs_val, P_xk=P_xk,
        )

    @staticmethod
    def _reference_roller_profile(x_k: np.ndarray, Dwe: float, Lwe: float) -> np.ndarray:
        """P(x_k) [mm] -- ISO/TS 16281 Sec 6.2 eq.(42)-(44). Identical to
        CylindricalRollerFamily's (radial) -- same roller, same profile."""
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
    # Capacity -- alpha_0 is fixed at 90deg for this family, so both
    # dynamic_capacity() and per_element_dynamic_capacity() always
    # dispatch to their respective *_90deg() function. dynamic_capacity()
    # (Ca, overall bearing) is now implemented -- see below.
    # static_capacity() (BearingCapacity.static(), Ca0) is still
    # NotImplementedError -- formula text not provided yet.
    # ------------------------------------------------------------------
 
    @staticmethod
    def dynamic_capacity(bearing, reduction_factor: float, nu: float, eta: float) -> float:
        """
        Ca [N] -- overall bearing dynamic axial load rating, ISO 281:2007
        Sec 6.6.2 Formula (41)/(42) (alpha_0 fixed at 90deg for this
        family), via functions/capacity.py's BearingCapacity.dynamic_90deg().
 
        reduction_factor, nu, eta : Formula (42) inputs -- ISO 281:2007
        Table 2's row/subtype-dependent factor and the formula's other two
        factors. None has a confirmed table value transcribed into this
        codebase yet, so all three are required here rather than class
        constants -- unlike LAMBDA_V_THRUST above, which IS a confirmed
        table value but for a different quantity (ISO/TS 16281's lambda_v,
        consumed by per_element_dynamic_capacity() below, NOT this method).
        Do not pass LAMBDA_V_THRUST as reduction_factor, nu, or eta -- see
        functions/capacity.py's module docstring for why they are
        unrelated despite all stemming from a table numbered 2.
 
        Single row only -- see module docstring's "UPDATED, this turn"
        note if this family is ever assembled with more than one row.
        """
        return rcap.BearingCapacity.dynamic_90deg(
            Z=bearing.Z, Dwe=bearing.Dwe, Lwe=bearing.Lwe, gamma=bearing.gamma,
            reduction_factor=reduction_factor, nu=nu, eta=eta,
        )

    @classmethod
    def per_element_dynamic_capacity(cls, bearing, Ca: float | None = None,
                                      lambda_v: float | None = None) -> tuple[float, float]:
        """(Q_ci, Q_ce) [N] -- ISO/TS 16281, thrust_90deg (alpha_0 fixed at
        90deg for this family). Ca defaults to bearing.C (catalog value)
        if not overridden."""
        return rcap.RollingElementCapacity.thrust_90deg(
            Z=bearing.Z, Ca=Ca if Ca is not None else bearing.C,
            lambda_v=lambda_v if lambda_v is not None else cls.LAMBDA_V_THRUST,
        )

    @staticmethod
    def per_lamina_dynamic_capacity(bearing, Q_ci: float, Q_ce: float) -> tuple[float, float]:
        """(q_ci, q_ce) [N] -- per-lamina dynamic load rating, ISO/TS 16281
        Sec 5.3.2 eq.(56)-(57)."""
        return rcap.RollingElementCapacity.per_lamina(Q_ci, Q_ce, bearing.n_s)