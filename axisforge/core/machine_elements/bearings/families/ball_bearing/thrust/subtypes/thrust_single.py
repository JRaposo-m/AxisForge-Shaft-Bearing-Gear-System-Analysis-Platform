"""
core/machine_elements/Bearings/families/ball/thrust/subtypes/thrust_ball.py

SingleRowThrustBallFamily -- BearingFamily implementation for a SINGLE ROW
thrust ball bearing, ISO/TS 16281 point contact, thrust duty.

Same point-contact math as DeepGrooveBallFamily/AngularContactFamily (all
three built on ../functions/contact_stiffness.py) -- idiom matches
AngularContactFamily: specified by nominal contact angle (alpha_0_deg),
never by clearance (s). Unlike ACB, alpha_0_deg here is bounded to
(45, 90] -- the ISO 281 radial/thrust classification boundary, mirroring
the bound already enforced on the radial side (AngularContactFamily caps
at 45 from below; this family starts there and goes to 90).

Owns its own ISO 281:2007 Table 1 constants -- Table No. 4 "Thrust ball
bearings": ri = re = 0.535*Dw, lambda = 0.90. Confirmed directly against
the ISO 281:2007 Table 1 image.

eta (the OTHER Table 1 column, needed for Formula (18)-(20)/(23)-(25)'s
Ca) is confirmed: eta = 1 - sin(alpha)/3.

No `i` (row count) anywhere in this file anymore -- previously ThrustBallFamily
carried an `i` kwarg that dynamic_capacity() used to fake multi-row support
by repeating this SAME geometry i times (identical rows, forced). That was
wrong for two reasons: (1) it silently assumed identical rows with no way
to express otherwise, and (2) it doesn't fit how a real multi-row thrust
bearing is bought -- ONE catalog part (one C/C0/d/D/b), not i independent
catalog rows, and Bearing.__init__ only ever takes one BearingCatalog (see
bearing.py). This family now only ever describes ONE row -- clean single-
row point contact, nothing else. Multi-row (i >= 2) lives in
MultiRowThrustBallFamily (thrust_ball_multirow.py, same directory), which
wraps this class: it calls SingleRowThrustBallFamily().assemble_geometry()
once per row (not Bearing.assemble() -- no per-row catalog exists) and
combines the resulting per-row Ca's via BearingCapacity.dynamic_multirow()
(Sec 6.4, Formula (29)). See that file for the actual multi-row wiring.

dynamic_capacity_from_fields() is the reusable core of dynamic_capacity()
below, split out specifically so MultiRowThrustBallFamily can compute each
row's Ca from a plain row dict (no Bearing object, no catalog) instead of
duplicating the alpha_0==90deg dispatch logic.

Capacity dispatch (per_element_dynamic_capacity / dynamic_capacity):
alpha_0 == 90deg exactly routes to the *_90deg() variant; anything else
routes to the *_nonzero_alpha() variant -- per your instruction, this
family always takes alpha_0 as the input and the 90deg-vs-not split
happens downstream at the capacity call, not at assembly.

Reference:
  ISO 281:2007 Table 1 (Table No. 4) -- raceway groove radius, reduction
  factor and eta, thrust ball bearings.
  ISO 281:2007 Sec 6.3, Formula (18)-(20)/(23)-(25) -- Ca, single row.
"""
from __future__ import annotations
import numpy as np

from axisforge.core.machine_elements.bearings.family import BearingFamily
from axisforge.core.machine_elements.bearings.bearing_types import BearingType
from axisforge.core.machine_elements.bearings.catalog import BearingCatalog
from ..functions import contact_stiffness as bc
from ..functions import capacity as bcap


class SingleRowThrustBallFamily(BearingFamily):
    """
    Usage
    -----
    bearing = Bearing.assemble(
        family=SingleRowThrustBallFamily(),
        catalog=BearingCatalog(d=40, D=68, b=15, C=28000, C0=44000,
                                designation="51208", position=50.0),
        geometry=dict(Dw=9.0, Dpw=54.0, Z=14, E=206000, alpha_0_deg=90.0),
        analyses={"point_contact": True},
    )
    """

    BEARING_TYPE = BearingType.THRUST_BALL
    DUTY = "thrust"

    CAPABILITIES = frozenset({"point_contact"})

    REQUIRED_FOR = {
        "point_contact": frozenset({
            "ri", "re", "Dw", "Dpw", "Z", "E", "nu",
            "A", "alpha_0", "Ri", "phi_j", "gamma", "cp",
        }),
    }

    # ISO 281:2007 Table 1, Table No. 4 -- "Thrust ball bearings".
    RI_OVER_DW = 0.535
    RE_OVER_DW = 0.535
    REDUCTION_FACTOR = 0.90   # lambda

    # eta = 1 - sin(alpha)/3 -- ISO 281:2007 Table 1, Table No. 4, eta column.
    @staticmethod
    def eta(alpha_0: float) -> float:
        """Reduction factor eta for Formula (18)-(20)/(23)-(25). alpha_0 in radians."""
        return 1.0 - np.sin(alpha_0) / 3.0

    @property
    def name(self) -> str:
        return "thrust_ball_single_row"

    @classmethod
    def reference_raceway_radii(cls, Dw: float) -> tuple[float, float]:
        """(ri, re) from ball diameter alone -- ISO 281:2007 Table 1, Table No. 4."""
        return cls.RI_OVER_DW * Dw, cls.RE_OVER_DW * Dw

    def assemble_geometry(self,
                           catalog: BearingCatalog,
                           Dw: float,
                           Dpw: float,
                           Z: int,
                           E: float,
                           alpha_0_deg: float,
                           nu: float = 0.3) -> dict:
        """
        alpha_0_deg : nominal contact angle [deg], in (45, 90] -- thrust
                      duty per ISO 281 classification (AngularContactFamily
                      covers (0, 45]). 90 = pure thrust; anything else is
                      handled the same way at assembly (contact_stiffness.gamma()
                      already special-cases 90deg) -- the split only matters
                      later, at capacity time (see per_element_dynamic_capacity()
                      / dynamic_capacity() below).

        No `i` here -- this family is always exactly one row. See module
        docstring for where multi-row lives.
        """
        if not (45.0 < alpha_0_deg <= 90.0):
            raise ValueError(
                f"SingleRowThrustBallFamily: alpha_0_deg must be in (45, 90], got "
                f"{alpha_0_deg} -- 45deg and below is radial duty (see "
                f"AngularContactFamily)."
            )

        ri, re = self.reference_raceway_radii(Dw)

        if ri <= Dw / 2.0:
            raise ValueError(f"SingleRowThrustBallFamily: ri must be > Dw/2, got ri={ri}, Dw={Dw}")
        if re <= Dw / 2.0:
            raise ValueError(f"SingleRowThrustBallFamily: re must be > Dw/2, got re={re}, Dw={Dw}")

        A = ri + re - Dw
        alpha_0, s = bc.contact_angle_and_clearance(A, alpha_0_deg=alpha_0_deg)
        g = bc.gamma(Dw, Dpw, alpha_0)
        Ri = bc.raceway_contact_radius(Dpw, ri, Dw, alpha_0)
        phi_j = np.linspace(0, 2 * np.pi, Z, endpoint=False)
        cp = bc.hertz_spring_constant(Dw, ri, re, E, nu, alpha_0, Dpw)

        return dict(
            bearing_type=self.BEARING_TYPE,
            ri=ri, re=re, Dw=Dw, Dpw=Dpw, Z=Z, s=s, E=E, nu=nu,
            A=A, alpha_0=alpha_0, Ri=Ri, phi_j=phi_j, gamma=g, cp=cp,
            raceway_radii_from_reference=True,
            reduction_factor=self.REDUCTION_FACTOR,
        )

    # ------------------------------------------------------------------
    # Capacity
    # ------------------------------------------------------------------

    @staticmethod
    def per_element_dynamic_capacity(bearing, Ca: float | None = None) -> tuple[float, float]:
        """(Q_ci, Q_ce) [N] -- ISO/TS 16281 Sec 4.3.1.3 (alpha_0 != 90deg) or
        Sec 4.3.1.4 (alpha_0 = 90deg), dispatched on bearing.alpha_0.
        """
        Ca_val = Ca if Ca is not None else bearing.C
        if np.isclose(bearing.alpha_0, np.pi / 2, atol=1e-9):
            return bcap.RollingElementCapacity.thrust_90deg(
                Z=bearing.Z, ri=bearing.ri, re=bearing.re, Dw=bearing.Dw, Ca=Ca_val,
            )
        return bcap.RollingElementCapacity.thrust_nonzero_alpha(
            Z=bearing.Z, alpha_0=bearing.alpha_0, ri=bearing.ri, re=bearing.re,
            Dw=bearing.Dw, gamma=bearing.gamma, Ca=Ca_val,
        )

    @classmethod
    def dynamic_capacity_from_fields(cls, Z: int, Dw: float, alpha_0: float,
                                      ri: float, re: float, gamma: float,
                                      reduction_factor: float | None = None) -> float:
        """
        Ca [N] -- ISO 281:2007 Formula (18)/(19)/(20) (alpha_0 != 90deg) or
        Formula (23)/(24)/(25) (alpha_0 = 90deg), Sec 6.3, single row.
        """
        eta_val = cls.eta(alpha_0)
        if np.isclose(alpha_0, np.pi / 2, atol=1e-9):
            return bcap.BearingCapacity.dynamic_90deg(
                Z=Z, Dw=Dw, ri=ri, re=re, gamma=gamma, 
                lam=reduction_factor if reduction_factor is not None else cls.REDUCTION_FACTOR,
                eta=eta_val,
            )
        return bcap.BearingCapacity.dynamic_nonzero_alpha(
            Z=Z, Dw=Dw, alpha_0=alpha_0, ri=ri, re=re, gamma=gamma,
            lam=reduction_factor if reduction_factor is not None else cls.REDUCTION_FACTOR,
            eta=eta_val,
        )

    @classmethod
    def dynamic_capacity(cls, bearing) -> float:
        """Ca [N] -- thin wrapper around dynamic_capacity_from_fields(),
        pulling fields off an assembled single-row Bearing.
        """
        return cls.dynamic_capacity_from_fields(
            Z=bearing.Z, Dw=bearing.Dw, alpha_0=bearing.alpha_0,
            ri=bearing.ri, re=bearing.re, gamma=bearing.gamma,
            reduction_factor=bearing.reduction_factor,
        )