"""
core/machine_elements/Bearings/families/ball/radial/subtypes/self_aligning.py

SelfAligningBallFamily -- BearingFamily implementation for self-aligning
ball bearings, ISO/TS 16281 point contact, radial duty.

Idiom matches AngularContactFamily: alpha_0_deg direct input, bounded to
(0, 45] (radial duty, ISO 281 classification boundary) -- not DGBB's s
idiom, and for a structural reason, not just consistency: re here is
DEFINED via gamma (see below), and gamma only depends on Dw/Dpw/alpha_0,
never on ri/re -- so alpha_0 must be known up front. With the s idiom,
alpha_0 is derived FROM A (which needs re first) -- circular. With
alpha_0_deg given directly, there's no cycle: alpha_0 (input) -> gamma ->
re. One-directional, no iteration needed.

Owns its own ISO 281:2007 Table 1 constants -- "Single and double row
self-aligning ball bearings" row:
    ri = 0.53*Dw
    re = 0.5*(1/gamma + 1)*Dw      -- NOT a fixed ratio of Dw, unlike every
                                       other ball subtype so far
    lambda = 1                     -- both i=1 and i=2 (row title says
                                       "single and double row", same
                                       lambda -- REDUCTION_FACTOR_BY_ROWS
                                       pattern as AngularContactFamily, NOT
                                       ThrustBallFamily's unrestricted i
                                       (that row had no row-count wording
                                       at all; this one explicitly says
                                       "single and double row", so i is
                                       capped at {1, 2} here, same
                                       reasoning as ACB).
    eta = "--"                     -- not applicable, radial duty.
Confirmed directly against the ISO 281:2007 Table 1 image.

Reference:
  ISO 281:2007 Table 1 -- raceway groove radius and reduction factor,
  single and double row self-aligning ball bearings.
"""
from __future__ import annotations
import numpy as np

from axisforge.core.machine_elements.Bearings.family import BearingFamily
from axisforge.core.machine_elements.Bearings.bearing_types import BearingType
from axisforge.core.machine_elements.Bearings.catalog import BearingCatalog
from ..functions import contact_stiffness as bc
from ..functions import capacity as bcap


class SelfAligningBallFamily(BearingFamily):
    """
    Usage
    -----
    bearing = Bearing.assemble(
        family=SelfAligningBallFamily(),
        catalog=BearingCatalog(d=30, D=62, b=20, C=22000, C0=9500,
                                designation="1206", position=50.0),
        geometry=dict(Dw=8.0, Dpw=46.0, Z=11, E=206000, alpha_0_deg=8.0, i=1),
        analyses={"point_contact": True},
    )
    """

    BEARING_TYPE = BearingType.SELF_ALIGNING_BALL
    DUTY = "radial"

    CAPABILITIES = frozenset({"point_contact"})

    REQUIRED_FOR = {
        "point_contact": frozenset({
            "ri", "re", "Dw", "Dpw", "Z", "E", "nu",
            "A", "alpha_0", "Ri", "phi_j", "gamma", "cp",
        }),
    }

    # ISO 281:2007 Table 1 -- "Single and double row self-aligning ball
    # bearings". ri is a fixed ratio of Dw; re is NOT -- see re_from_gamma().
    RI_OVER_DW = 0.53
    REDUCTION_FACTOR_BY_ROWS = {1: 1.0, 2: 1.0}   # lambda, keyed by i (row count)

    @property
    def name(self) -> str:
        return "self_aligning"

    @staticmethod
    def re_from_gamma(Dw: float, gamma: float) -> float:
        """re = 0.5*(1/gamma + 1)*Dw -- ISO 281:2007 Table 1, self-aligning
        row. Unlike every other ball subtype, re is not a fixed ratio of Dw."""
        if gamma <= 0.0:
            raise ValueError(f"SelfAligningBallFamily: gamma must be > 0, got {gamma}.")
        return 0.5 * (1.0 / gamma + 1.0) * Dw

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
        alpha_0_deg : nominal contact angle [deg], in (0, 45] -- radial duty,
                      ISO 281 classification boundary (same bound as
                      AngularContactFamily). Required up front -- re is
                      derived from gamma(alpha_0), see module docstring.
        i           : number of rows -- 1 (single, default) or 2 (double).
                      lambda is 1 either way (see REDUCTION_FACTOR_BY_ROWS).
        """
        if not (0.0 < alpha_0_deg <= 45.0):
            raise ValueError(
                f"SelfAligningBallFamily: alpha_0_deg must be in (0, 45], got "
                f"{alpha_0_deg} -- above 45deg is thrust duty."
            )
        if i not in self.REDUCTION_FACTOR_BY_ROWS:
            raise ValueError(
                f"SelfAligningBallFamily: i (number of rows) must be one of "
                f"{sorted(self.REDUCTION_FACTOR_BY_ROWS)}, got {i}"
            )

        alpha_0 = np.radians(alpha_0_deg)
        g = bc.gamma(Dw, Dpw, alpha_0)

        ri = self.RI_OVER_DW * Dw
        re = self.re_from_gamma(Dw, g)

        if ri <= Dw / 2.0:
            raise ValueError(f"SelfAligningBallFamily: ri must be > Dw/2, got ri={ri}, Dw={Dw}")
        if re <= Dw / 2.0:
            raise ValueError(f"SelfAligningBallFamily: re must be > Dw/2, got re={re}, Dw={Dw}")

        A = ri + re - Dw
        alpha_0, s = bc.contact_angle_and_clearance(A, alpha_0_deg=alpha_0_deg)
        Ri = bc.raceway_contact_radius(Dpw, ri, Dw, alpha_0)
        phi_j = np.linspace(0, 2 * np.pi, Z, endpoint=False)
        # NOT the generic hertz_spring_constant() -- this subtype's outer
        # race has F_e(rho)=0 identically (see
        # SelfAligningContactStiffness docstring), which the general
        # elliptical chi-solve can't handle. Stubbed until that closed-form
        # term is filled in.
        cp = bc.SelfAligningContactStiffness.hertz_spring_constant(Dw, ri, re, E, nu, alpha_0, Dpw)

        return dict(
            bearing_type=self.BEARING_TYPE,
            ri=ri, re=re, Dw=Dw, Dpw=Dpw, Z=Z, s=s, E=E, nu=nu,
            A=A, alpha_0=alpha_0, Ri=Ri, phi_j=phi_j, gamma=g, cp=cp,
            raceway_radii_from_reference=True,
            i=i, reduction_factor=self.REDUCTION_FACTOR_BY_ROWS[i],
        )

    # ------------------------------------------------------------------
    # Capacity -- same eq.(19)-(20) formula as DeepGrooveBallFamily/
    # AngularContactFamily (radial ball bearing, generic).
    # ------------------------------------------------------------------

    @staticmethod
    def per_element_dynamic_capacity(bearing, Cr: float | None = None) -> tuple[float, float]:
        """(Q_ci, Q_ce) [N] -- ISO/TS 16281 Sec 4.3.1.2 eq.(19)-(20).

        Cr defaults to bearing.C (catalog value) if not overridden."""
        return bcap.RollingElementCapacity.radial(
            Z=bearing.Z, alpha_0=bearing.alpha_0, ri=bearing.ri, re=bearing.re,
            Dw=bearing.Dw, gamma=bearing.gamma, Cr=Cr if Cr is not None else bearing.C,
            i=bearing.i,
        )