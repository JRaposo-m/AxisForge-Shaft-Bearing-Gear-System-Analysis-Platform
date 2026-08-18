"""
axisforge/solvers/machine_elements/bearings/ISO_281/Ball_Bearing/radial_ball_bearing.py

ISO 281:2007 -- basic dynamic radial load rating Cr for radial ball
bearings, Formula (13)/(14), fc from Formula (15). Geometry (ri, re,
lambda) comes from Table 1 (Tables_ball_bearings.py). THRUST is out of
scope (Formula (20)/(21), different eta-based fc) -- compute() raises
NotImplementedError for it.

NOT VALIDATED: fc as coded here comes out ~1000x above literature
reference values (NASA/TP-2016-218937: fc ~ 60-77 for Dw*cos(alpha)/Dpw
~ 0.20-0.30, same fc used in this Formula (13)). Recheck Formula (15)'s
bracket placement against ISO 281:2007 clause 6.2 before trusting Cr.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

from axisforge.solvers.machine_elements.bearings.ISO_281.Ball_Bearing.Tables_ball_bearings import (
    BallBearingFamily,
    get_raceway_groove_factors,
)

_A1_0089_N = 98_066.5    # "0,089*A1", Formula (15) note -- Cr in N, Dw in mm
_DW_THRESHOLD_MM = 25.4  # Formula (13)/(14) switch point


@dataclass(frozen=True)
class BasicDynamicRadialLoadRating:
    """Cr [N] for a radial ball bearing -- ISO 281:2007 Formula (13)/(14)/(15)."""

    label  : str
    Cr     : float
    fc     : float
    family : BallBearingFamily
    i      : int
    Z      : int
    Dw     : float
    Dpw    : float
    alpha  : float
    gamma  : float

    @staticmethod
    def _fc(ri: float, re: float, Dw: float, gamma: float, lam: float) -> float:
        """Formula (15) -- geometry factor fc. See module docstring: unvalidated."""
        warnings.warn(
            "BasicDynamicRadialLoadRating: Formula (15) not validated against "
            "ISO 281:2007 primary text -- see module docstring.",
            stacklevel=2,
        )
        if not (0.0 < gamma < 1.0):
            raise ValueError(f"gamma = Dw*cos(alpha)/Dpw must be in (0, 1); got {gamma}.")
        if 2.0 * ri <= Dw:
            raise ValueError(f"2*ri ({2*ri:.4f}) <= Dw ({Dw:.4f}) -- invalid groove geometry.")

        groove_conformity = (2.0 * ri / (2.0 * ri - Dw)) ** 0.41
        gamma_term = gamma ** 0.3 * (1.0 - gamma) ** 1.39 / (1.0 + gamma) ** (1.0 / 3.0)

        if math.isinf(re):
            # re -> infinity (magneto): (ri/re)*(2re-Dw)/(2ri-Dw) is a 0*inf
            # indeterminate form, NOT 0 -- ri/re -> 0 but (2re-Dw) -> inf at
            # the same time. Taking the limit properly:
            #   (ri/re)*(2re-Dw)/(2ri-Dw) = [2ri/(2ri-Dw)] - [ri*Dw/(re*(2ri-Dw))]
            # and the second term -> 0 as re -> inf, so the limit is
            # 2ri/(2ri-Dw) -- the same ratio already used in groove_conformity.
            radii_ratio = 2.0 * ri / (2.0 * ri - Dw)
        else:
            if 2.0 * re <= Dw:
                raise ValueError(f"2*re ({2*re:.4f}) <= Dw ({Dw:.4f}) -- invalid groove geometry.")
            radii_ratio = (ri / re) * ((2.0 * re - Dw) / (2.0 * ri - Dw))

        bracket = 1.04 * radii_ratio ** 0.41 * ((1.0 - gamma) / (1.0 + gamma)) ** 1.72
        correction = (1.0 + bracket ** (10.0 / 3.0)) ** (-3.0 / 10.0)

        return _A1_0089_N * 0.41 * lam * groove_conformity * gamma_term * correction

    @staticmethod
    def _basic_dynamic_rating(fc: float, i: int, alpha_rad: float, Z: int, Dw: float) -> float:
        """Formula (13) (Dw <= 25,4 mm) / Formula (14) (Dw > 25,4 mm)."""
        cos_term = (i * math.cos(alpha_rad)) ** 0.7
        if Dw <= _DW_THRESHOLD_MM:
            return fc * cos_term * Z ** (2.0 / 3.0) * Dw ** 1.8
        return 3.647 * fc * cos_term * Z ** (2.0 / 3.0) * Dw ** 1.4

    @classmethod
    def compute(cls,
                Dw: float,
                Dpw: float,
                Z: int,
                i: int = 1, # must be a string as single or double, its easier and prevents missuses
                alpha_deg: float = 0.0,
                family: BallBearingFamily = BallBearingFamily.RADIAL_CONTACT_GROOVE,
                label: str = "") -> "BasicDynamicRadialLoadRating":
        """
        Dw, Dpw [mm], Z balls/row, i rows carrying load in the same
        direction, alpha_deg nominal contact angle, family selects the
        Table 1 row (default RADIAL_CONTACT_GROOVE). Raises
        NotImplementedError for family=THRUST.
        """
        if family is BallBearingFamily.THRUST:
            raise NotImplementedError(
                "Thrust ball bearings use ISO 281:2007 Formula (20)/(21) "
                "(eta reduction factor), not implemented here."
            )
        if Z <= 0 or Dw <= 0.0 or Dpw <= 0.0 or i <= 0:
            raise ValueError(
                f"Z, Dw, Dpw and i must all be positive; got Z={Z}, "
                f"Dw={Dw}, Dpw={Dpw}, i={i}."
            )

        row = get_raceway_groove_factors(family)
        alpha = math.radians(alpha_deg)
        gamma = Dw * math.cos(alpha) / Dpw

        ri = row.ri(Dw)
        re = row.re(Dw, gamma)
        fc = cls._fc(ri, re, Dw, gamma, row.lam)
        Cr = cls._basic_dynamic_rating(fc, i, alpha, Z, Dw)

        return cls(
            label=label, Cr=Cr, fc=fc, family=family,
            i=i, Z=Z, Dw=Dw, Dpw=Dpw, alpha=alpha, gamma=gamma,
        )

    @classmethod
    def from_bearing(cls,
                      bearing,
                      family: BallBearingFamily = BallBearingFamily.RADIAL_CONTACT_GROOVE,
                      i: int = 1,
                      label: str = "") -> "BasicDynamicRadialLoadRating":
        """
        Same as compute(), but pulls Dw/Dpw/Z/alpha_deg off an already
        set-up bearing (any ISO/TS16281 subtype with setup_internal_
        geometry() already called -- DeepGrooveBallBearing,
        AngularContactBallBearing, ...) instead of repeating them by hand,
        and stamps `label` from the bearing's own identity. This is the
        only thing tying a Cr result back to a specific bearing instance
        -- compute() itself stays decoupled/pure.

        `family` and `i` are rating-specific (which Table 1 row, how many
        rows carry load) and independent of the bearing's own ri/re
        (ISO/TS16281 stiffness values) -- they don't come from the
        bearing object, same reasoning as Tables_ball_bearings.py.
        """
        if not bearing.has_internal_geometry():
            raise RuntimeError(
                f"Bearing '{bearing.label or bearing.designation}': call "
                "setup_internal_geometry() before computing Cr."
            )
        return cls.compute(
            Dw=bearing.Dw, Dpw=bearing.Dpw, Z=bearing.Z,
            i=i, alpha_deg=math.degrees(bearing.alpha_0),
            family=family, label=label or bearing.label or bearing.designation,
        )