"""
core/machine_elements/bearings/families/ball_bearing/radial/subtypes/deep_groove.py

DeepGrooveBallFamily -- BearingFamily implementation for DGBB, ISO/TS
16281 point contact, radial duty.

Owns: the ISO 281:2007 Table 1 inputs for this bearing type (RI_OVER_DW,
RE_OVER_DW, REDUCTION_FACTOR_BY_ROWS) and the idiomatic input choice (s
only, not alpha_0_deg -- that's angular_contact.py's idiom).

All contact math (parameterised, no baked-in numbers) is in
../functions/contact_stiffness.py.

Reference:
  ISO 281:2007 Table 1 -- raceway groove radius and reduction factor
"""
from __future__ import annotations
import numpy as np

from axisforge.core.machine_elements.bearings.family import BearingFamily
from axisforge.core.machine_elements.bearings.bearing_types import BearingType
from axisforge.core.machine_elements.bearings.catalog import BearingCatalog
from ..functions import contact_stiffness as bc
from ..functions import capacity as bcap


class DeepGrooveBallFamily(BearingFamily):
    """
    Usage
    -----
    bearing = Bearing.assemble(
        family=DeepGrooveBallFamily(),
        catalog=BearingCatalog(d=40, D=80, b=18, C=29600, C0=17800,
                                designation="6208", position=50.0, label="brg1a"),
        geometry=dict(Dw=12.0, Dpw=60.0, Z=9, E=206000, s=0.015, i=1),
        analyses={"point_contact": True},
    )
    """
    
    BEARING_TYPE = BearingType.DEEP_GROOVE_BALL
    DUTY = "radial"

    CAPABILITIES = frozenset({"point_contact"})

    REQUIRED_FOR = {
        "point_contact": frozenset({
            "ri", "re", "Dw", "Dpw", "Z", "E", "nu",
            "A", "alpha_0", "Ri", "phi_j", "gamma", "cp",
        }),
    }

    # ISO 281:2007 Table 1 -- radial contact groove ball bearings.
    RI_OVER_DW = 0.52
    RE_OVER_DW = 0.52
    REDUCTION_FACTOR_BY_ROWS = {1: 0.95, 2: 0.90}   # lambda, keyed by i (row count)

    @property
    def name(self) -> str:
        return "deep_groove_ball"

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
                           s: float,
                           nu: float = 0.3,
                           i: int = 1) -> dict:
        """
        Dw, Dpw, Z, E : as ever
        s             : diametral operating clearance [mm] -- the ONLY
                        clearance input this family accepts
        nu            : Poisson's ratio
        i             : number of rows -- 1 (single, default) or 2
                        (double). Those are the only values accepted for this subtype.
        """
        if i not in self.REDUCTION_FACTOR_BY_ROWS:
            raise ValueError(
                f"DeepGrooveBallFamily: i (number of rows) must be one of "
                f"{sorted(self.REDUCTION_FACTOR_BY_ROWS)}, got {i}"
            )

        ri, re = self.reference_raceway_radii(Dw)

        if ri <= Dw / 2.0:
            raise ValueError(f"DeepGrooveBallFamily: ri must be > Dw/2, got ri={ri}, Dw={Dw}")
        if re <= Dw / 2.0:
            raise ValueError(f"DeepGrooveBallFamily: re must be > Dw/2, got re={re}, Dw={Dw}")
        if s < 0:
            raise ValueError(f"DeepGrooveBallFamily: s must be >= 0, got {s}")

        A = ri + re - Dw
        alpha_0, s = bc.contact_angle_and_clearance(A, s=s)
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
    # Capacity
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