"""
core/machine_elements/Bearings/families/ball/radial/subtypes/angular_contact.py

AngularContactFamily -- BearingFamily implementation for ACB, ISO/TS
16281 point contact, radial duty.

Same point-contact math as DeepGrooveBallFamily (both built on
../functions/contact_stiffness.py) -- differs only in idiom: specified
by nominal contact angle (alpha_0_deg), not clearance (s).

Owns its own ISO 281:2007 Table 1 constants. Table data is subtype
input, not generic math (see deep_groove_ball.py's module docstring for
the reasoning).

NEW vs. the pre-rewrite file: alpha_0_deg is now bounded to (0, 45] here.
The original only checked alpha_0 > 0. The upper bound is new, added
because this class now lives under families/ball/radial/ specifically --
an ACB spec'd above 45deg is thrust duty (ISO 281 classification), which
belongs in families/ball/thrust/ once that exists. Remove the check if
you don't want that boundary enforced yet.

Reference:
  ISO 281:2007 Table 1 -- raceway groove radius and reduction factor
"""
from __future__ import annotations
import numpy as np

from axisforge.core.machine_elements.Bearings.family import BearingFamily
from axisforge.core.machine_elements.Bearings.bearing_types import BearingType
from axisforge.core.machine_elements.Bearings.catalog import BearingCatalog
from ..functions import contact_stiffness as bc
from ..functions import capacity as bcap


class AngularContactFamily(BearingFamily):
    """
    Usage
    -----
    bearing = Bearing.assemble(
        family=AngularContactFamily(),
        catalog=BearingCatalog(d=40, D=80, b=18, C=35000, C0=26000,
                                designation="7208B", position=50.0),
        geometry=dict(Dw=11.5, Dpw=60.0, Z=13, E=206000, alpha_0_deg=40.0, i=1),
        analyses={"point_contact": True},
    )
    """

    BEARING_TYPE = BearingType.ANGULAR_CONTACT
    DUTY = "radial"

    CAPABILITIES = frozenset({"point_contact"})

    REQUIRED_FOR = {
        "point_contact": frozenset({
            "ri", "re", "Dw", "Dpw", "Z", "E", "nu",
            "A", "alpha_0", "Ri", "phi_j", "gamma", "cp",
        }),
    }

    # ISO 281:2007 Table 1 -- "single and double row angular contact
    # groove ball bearings" is ONE table row: unlike DGBB's radial
    # contact groove (which drops to lambda=0.90 for double row), ACB
    # keeps lambda=0.95 regardless of row count. Subtype input, not
    # generic math.
    RI_OVER_DW = 0.52
    RE_OVER_DW = 0.52
    REDUCTION_FACTOR_BY_ROWS = {1: 0.95, 2: 0.95}   # lambda, keyed by i (row count)

    @property
    def name(self) -> str:
        return "angular_contact"

    @classmethod
    def reference_raceway_radii(cls, Dw: float) -> tuple[float, float]:
        """(ri, re) from ball diameter alone -- ISO 281:2007 Table 1."""
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
        alpha_0_deg : nominal (free) contact angle [deg], in (0, 45] for
                      this radial-duty family -- the ONLY contact input
                      accepted (no s -- DGBB's idiom).
        i           : number of rows -- 1 (single, default) or 2
                      (double). Doesn't change lambda for this family
                      (see REDUCTION_FACTOR_BY_ROWS comment), but still
                      validated/carried through -- it will matter for
                      the dynamic capacity formula later (Formulae 15/20).
        """
        if not (0.0 < alpha_0_deg <= 45.0):
            raise ValueError(
                f"AngularContactFamily: alpha_0_deg must be in (0, 45], got "
                f"{alpha_0_deg} -- above 45deg is thrust duty."
            )
        if i not in self.REDUCTION_FACTOR_BY_ROWS:
            raise ValueError(
                f"AngularContactFamily: i (number of rows) must be one of "
                f"{sorted(self.REDUCTION_FACTOR_BY_ROWS)}, got {i}"
            )

        ri, re = self.reference_raceway_radii(Dw)

        if ri <= Dw / 2.0:
            raise ValueError(f"AngularContactFamily: ri must be > Dw/2, got ri={ri}, Dw={Dw}")
        if re <= Dw / 2.0:
            raise ValueError(f"AngularContactFamily: re must be > Dw/2, got re={re}, Dw={Dw}")

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
            i=i, reduction_factor=self.REDUCTION_FACTOR_BY_ROWS[i],
        )

    # ------------------------------------------------------------------
    # Capacity -- same eq.(19)-(20) formula as DeepGrooveBallFamily
    # (radial ball bearing, generic), called with THIS bearing's own
    # attributes. See that class for why it isn't shared/inherited.
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

    @staticmethod
    def dynamic_capacity(bearing) -> float:
        """Cr [N] -- ISO 281:2007 Formula (13)/(14)/(15).
        NOT VALIDATED -- see BearingCapacity._fc() docstring."""
        return bcap.BearingCapacity.dynamic(
            Z=bearing.Z, Dw=bearing.Dw, alpha_0=bearing.alpha_0,
            ri=bearing.ri, re=bearing.re, gamma=bearing.gamma,
            reduction_factor=bearing.reduction_factor, i=bearing.i,
        )