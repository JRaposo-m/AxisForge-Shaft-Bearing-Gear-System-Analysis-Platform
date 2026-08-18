"""
axisforge/solvers/machine_elements/bearings/ISO_281/Ball_Bearing/Tables_ball_bearings.py

ISO 281:2007, Table 1 -- raceway groove radius and reduction factor for
ball bearings. Data only, no fc/Cr math -- see radial_ball_bearing.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable


class BallBearingFamily(Enum):
    RADIAL_CONTACT_GROOVE = auto()  # single row radial contact groove
    ANGULAR_CONTACT       = auto()  # single/double row angular contact groove
    DOUBLE_ROW_RADIAL     = auto()  # double row radial contact groove
    SELF_ALIGNING         = auto()  # single/double row self-aligning
    MAGNETO               = auto()  # single row radial contact separable (magneto)
    THRUST                = auto()  # thrust ball bearings (Table 4, not Table 2)


@dataclass(frozen=True)
class RacewayGrooveFactors:
    """One row of Table 1: description, ri(Dw), re(Dw, gamma), lambda, eta."""
    description  : str
    iso281_table : int
    ri           : Callable[[float], float]
    re           : Callable[[float, float | None], float]
    lam          : float
    eta          : Callable[[float], float] | None = None


def _re_self_aligning(Dw: float, gamma: float | None) -> float:
    if gamma is None:
        raise ValueError("SELF_ALIGNING needs gamma to evaluate re.")
    return 0.5 * (1.0 / gamma + 1.0) * Dw


def _re_magneto(Dw: float, gamma: float | None = None) -> float:
    return math.inf  # separable race, no closed outer groove


def _eta_thrust(alpha_rad: float) -> float:
    return 1.0 - math.sin(alpha_rad) / 3.0


TABLE_1: dict[BallBearingFamily, RacewayGrooveFactors] = {
    BallBearingFamily.RADIAL_CONTACT_GROOVE: RacewayGrooveFactors(
        description="Single row radial contact groove ball bearings",
        iso281_table=2,
        ri=lambda Dw: 0.52 * Dw,
        re=lambda Dw, gamma=None: 0.52 * Dw,
        lam=0.95,
    ),
    BallBearingFamily.ANGULAR_CONTACT: RacewayGrooveFactors(
        description="Single and double row angular contact groove ball bearings",
        iso281_table=2,
        ri=lambda Dw: 0.52 * Dw,
        re=lambda Dw, gamma=None: 0.52 * Dw,
        lam=0.95,
    ),
    BallBearingFamily.DOUBLE_ROW_RADIAL: RacewayGrooveFactors(
        description="Double row radial contact groove ball bearings",
        iso281_table=2,
        ri=lambda Dw: 0.52 * Dw,
        re=lambda Dw, gamma=None: 0.52 * Dw,
        lam=0.90,
    ),
    BallBearingFamily.SELF_ALIGNING: RacewayGrooveFactors(
        description="Single and double row self-aligning ball bearings",
        iso281_table=2,
        ri=lambda Dw: 0.53 * Dw,
        re=_re_self_aligning,
        lam=1.0,
    ),
    BallBearingFamily.MAGNETO: RacewayGrooveFactors(
        description="Single row radial contact separable ball bearings (magneto bearings)",
        iso281_table=2,
        ri=lambda Dw: 0.52 * Dw,
        re=_re_magneto,
        lam=0.95,
    ),
    BallBearingFamily.THRUST: RacewayGrooveFactors(
        description="Thrust ball bearings",
        iso281_table=4,
        ri=lambda Dw: 0.535 * Dw,
        re=lambda Dw, gamma=None: 0.535 * Dw,
        lam=0.90,
        eta=_eta_thrust,
    ),
}


def get_raceway_groove_factors(family: BallBearingFamily) -> RacewayGrooveFactors:
    try:
        return TABLE_1[family]
    except KeyError:
        raise ValueError(f"No ISO 281:2007 Table 1 row for {family!r}") from None