"""
core/machine_elements/Bearings/families/ball/thrust/subtypes/thrust_ball.py

ThrustBallFamily -- BearingFamily implementation for thrust ball bearings,
ISO/TS 16281 point contact, thrust duty.

Same point-contact math as DeepGrooveBallFamily/AngularContactFamily (all
three built on ../functions/contact_stiffness.py) -- idiom matches
AngularContactFamily: specified by nominal contact angle (alpha_0_deg),
never by clearance (s). Unlike ACB, alpha_0_deg here is bounded to
(45, 90] -- the ISO 281 radial/thrust classification boundary, mirroring
the bound already enforced on the radial side (AngularContactFamily caps
at 45 from below; this family starts there and goes to 90).

Owns its own ISO 281:2007 Table 1 constants -- Table No. 4 "Thrust ball
bearings": ri = re = 0.535*Dw, lambda = 0.90, BOTH independent of row
count -- unlike DGBB (separate table rows for single/double) or even ACB
(capped at i in {1, 2} since its table row is literally titled "single
and double row"), Table No. 4 has no row-count column at all, so i here
is NOT restricted to {1, 2} -- a thrust ball bearing can stack n rows.
lambda is a single scalar (REDUCTION_FACTOR), not a per-i dict. Confirmed
directly against the ISO 281:2007 Table 1 image.

eta (the OTHER Table 1 column, needed for Formula (20)/(21)'s Ca) is now
also confirmed: eta = 1 - sin(alpha)/3. NOT wired into BearingCapacity.dynamic()
yet -- Formula (20)/(21)'s full text still isn't provided, and i (row
count) matters there per your own note ("vou tratar disso depois") --
left stubbed deliberately, only the coefficient itself is recorded here
so it isn't lost.

Capacity dispatch (per_element_dynamic_capacity): alpha_0 == 90deg exactly
routes to RollingElementCapacity.thrust_90deg(); anything else routes to
thrust_nonzero_alpha() -- per your instruction, this family always takes
alpha_0 as the input and the 90deg-vs-not split happens downstream at the
capacity call, not at assembly.

Reference:
  ISO 281:2007 Table 1 (Table No. 4) -- raceway groove radius, reduction
  factor and eta, thrust ball bearings.
"""
from __future__ import annotations
import numpy as np

from axisforge.core.machine_elements.Bearings.family import BearingFamily
from axisforge.core.machine_elements.Bearings.bearing_types import BearingType
from axisforge.core.machine_elements.Bearings.catalog import BearingCatalog
from ..functions import contact_stiffness as bc
from ..functions import capacity as bcap


class ThrustBallFamily(BearingFamily):
    """
    Usage
    -----
    bearing = Bearing.assemble(
        family=ThrustBallFamily(),
        catalog=BearingCatalog(d=40, D=68, b=15, C=28000, C0=44000,
                                designation="51208", position=50.0),
        geometry=dict(Dw=9.0, Dpw=54.0, Z=14, E=206000, alpha_0_deg=90.0, i=1),
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

    # ISO 281:2007 Table 1, Table No. 4 -- "Thrust ball bearings". No
    # row-count column -- lambda is a single scalar, i is unrestricted.
    RI_OVER_DW = 0.535
    RE_OVER_DW = 0.535
    REDUCTION_FACTOR = 0.90   # lambda -- independent of i

    # eta = 1 - sin(alpha)/3 -- ISO 281:2007 Table 1, Table No. 4, eta column.
    # Recorded for when Formula (20)/(21) gets implemented; not consumed yet.
    @staticmethod
    def eta(alpha_0: float) -> float:
        """Reduction factor eta for Formula (20)/(21) -- NOT wired into
        BearingCapacity.dynamic() yet, that formula's full text is still
        pending. alpha_0 in radians."""
        return 1.0 - np.sin(alpha_0) / 3.0

    @property
    def name(self) -> str:
        return "thrust_ball"

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
                           nu: float = 0.3,
                           i: int = 1) -> dict:
        """
        alpha_0_deg : nominal contact angle [deg], in (45, 90] -- thrust
                      duty per ISO 281 classification (AngularContactFamily
                      covers (0, 45]). 90 = pure thrust; anything else is
                      handled the same way at assembly (contact_stiffness.gamma()
                      already special-cases 90deg) -- the split only matters
                      later, at capacity time (see
                      ThrustBallFamily.per_element_dynamic_capacity()).
        i           : number of rows, >= 1 (default 1). Table No. 4 has no
                      row-count column -- lambda doesn't depend on i for
                      this family, and i is not capped at 2 (a thrust ball
                      bearing can stack n rows). i still matters for
                      Formula (20)/(21) later.
        """
        if not (45.0 < alpha_0_deg <= 90.0):
            raise ValueError(
                f"ThrustBallFamily: alpha_0_deg must be in (45, 90], got "
                f"{alpha_0_deg} -- 45deg and below is radial duty (see "
                f"AngularContactFamily)."
            )
        if not isinstance(i, int) or i < 1:
            raise ValueError(f"ThrustBallFamily: i (number of rows) must be a positive integer, got {i}")

        ri, re = self.reference_raceway_radii(Dw)

        if ri <= Dw / 2.0:
            raise ValueError(f"ThrustBallFamily: ri must be > Dw/2, got ri={ri}, Dw={Dw}")
        if re <= Dw / 2.0:
            raise ValueError(f"ThrustBallFamily: re must be > Dw/2, got re={re}, Dw={Dw}")

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
            i=i, reduction_factor=self.REDUCTION_FACTOR,
        )

    # ------------------------------------------------------------------
    # Capacity -- dispatches on alpha_0 == 90deg (thrust_90deg) vs not
    # (thrust_nonzero_alpha), per your instruction. dynamic_capacity()
    # (Ca, Formula 20/21) deliberately not added yet -- pending that
    # formula's text and i's exact role in it.
    # ------------------------------------------------------------------

    @staticmethod
    def per_element_dynamic_capacity(bearing, Ca: float | None = None) -> tuple[float, float]:
        """(Q_ci, Q_ce) [N] -- ISO/TS 16281 Sec 4.3.1.3 (alpha_0 != 90deg) or
        Sec 4.3.1.4 (alpha_0 = 90deg), dispatched on bearing.alpha_0.

        Ca defaults to bearing.C (catalog dynamic rating -- thrust duty
        rates on the axial capacity, published under the same generic C
        field) if not overridden."""
        Ca_val = Ca if Ca is not None else bearing.C
        if np.isclose(bearing.alpha_0, np.pi / 2, atol=1e-9):
            return bcap.RollingElementCapacity.thrust_90deg(
                Z=bearing.Z, ri=bearing.ri, re=bearing.re, Dw=bearing.Dw, Ca=Ca_val,
            )
        return bcap.RollingElementCapacity.thrust_nonzero_alpha(
            Z=bearing.Z, alpha_0=bearing.alpha_0, ri=bearing.ri, re=bearing.re,
            Dw=bearing.Dw, gamma=bearing.gamma, Ca=Ca_val,
        )