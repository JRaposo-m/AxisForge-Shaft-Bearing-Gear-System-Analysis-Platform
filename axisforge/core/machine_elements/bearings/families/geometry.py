"""
axisforge/core/machine_elements/bearings/families/_geometry.py
(renamed from curvatures.py -- it's radii-of-curvature contact geometry now,
not a flat curvature value object)

Two-solid Hertz contact geometry, Harris / Hamrock-Dowson notation: solid a
is the rolling element, solid b is the raceway. x = rolling direction,
y = transverse direction. ContactGeometry is the contract; concrete
subclasses supply r_ax/r_ay/r_bx/r_by for a given rolling-element/raceway
pair (ball-inner, ball-outer, later roller-inner, roller-outer, ...) -- the
curvature-sum/curvature-difference combination is generic and lives here
once, not reimplemented per subclass.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np



# ---------------------------------------------------------------------
# ---- shared low-level geometry helpers (same across radial/thrust) --
# ---------------------------------------------------------------------

def contact_angle_and_clearance(A: float, *, s: float | None = None,
                                 alpha_0_deg: float | None = None) -> tuple[float, float]:
    """Exactly one of s / alpha_0_deg required. Returns (alpha_0 [rad], s [mm]).
    Shared by every point-contact radial family -- DeepGrooveBallFamily
    supplies s, AngularContactFamily supplies alpha_0_deg, each family
    just picks which idiom makes sense for it."""
    if (s is None) == (alpha_0_deg is None):
        raise ValueError("Exactly one of 's' or 'alpha_0_deg' must be provided, not both or neither.")
    if s is not None:
        return np.arccos(1.0 - s / (2.0 * A)), s
    alpha_0 = np.radians(alpha_0_deg)
    return alpha_0, 2.0 * A * (1.0 - np.cos(alpha_0))


# ---------------------------------------------------------------------
# ---- Geometry definition -------------------------------------
# ---------------------------------------------------------------------

class Geometry(ABC):
    """Contract: radii of curvature for one rolling-element/raceway
    contact pair. r_bx=inf / r_by=inf are valid (flat in that direction,
    e.g. a thrust ball bearing's r_bx, or a straight cylindrical roller's
    r_ay) -- ordinary float division handles that natively, no special
    casing needed at this level."""

    @property
    @abstractmethod
    def gamma(self) -> float: ...

    @property
    @abstractmethod
    def free_end_play(self) -> float: ...

    @property
    @abstractmethod
    def f_i(self) -> float: ...

    @property
    @abstractmethod
    def f_o(self) -> float: ...

    @property
    @abstractmethod
    def r_ax(self) -> float: ...

    @property
    @abstractmethod
    def r_ay(self) -> float: ...

    @property
    @abstractmethod
    def r_bx_inner(self) -> float: ...

    @property
    @abstractmethod
    def r_by_inner(self) -> float: ...

    @property
    @abstractmethod
    def r_bx_outer(self) -> float: ...

    @property
    @abstractmethod
    def r_by_outer(self) -> float: ...

# ---------------------------------------------------------------------
# ---- Ball Bearing Geometry definition -------------------------------
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class BallBearingGeometry(Geometry):
    Dw: float
    Dpw: float
    A: float
    s: float
    ri: float
    re: float
    alpha_0: float


    @property
    def gamma(self) -> float:
        if np.isclose(self.alpha_0, np.pi / 2, atol=1e-9):
            return self.Dw / self.Dpw
        return self.Dw * np.cos(self.alpha_0) / self.Dpw

    @property
    def free_end_play(self) -> float:
        return (4 * self.A * self.s - self.s**2)**0.5
    @property
    def f_i(self) -> float:
        return self.ri / self.Dw
    @property
    def f_o(self) -> float:
        return self.re / self.Dw 
    
    @property
    def r_ax(self) -> float:
        return self.Dw/2.0

    @property
    def r_ay(self) -> float:
        return self.Dw/2.0

    @property
    def r_bx_inner(self) -> float:
        return (self.Dpw - self.Dw * np.cos(self.alpha_0)) / (2 * np.cos(self.alpha_0))

    @property
    def r_by_inner(self) -> float:
        return -1 * self.ri

    @property
    def r_bx_outer(self) -> float:
        return (self.Dpw + self.Dw * np.cos(self.alpha_0)) / (2 * np.cos(self.alpha_0))

    @property
    def r_by_outer(self) -> float:
        return -1 * self.re


# ---------------------------------------------------------------------
# ---- Roller Bearing Geometry definition -----------------------------
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class CylindricalRollerBearingGeometry(Geometry):

    Dwe: float
    Dpw: float

    @property
    def gamma(self) -> float:
        return self.Dwe / self.Dpw

    @property
    def lamina_positions(self) -> np.ndarray:
        """x_k -- lamina midpoints, strictly inside (-Lwe/2, Lwe/2). Sec 5.2.2."""
        lamina_length = self.Lwe / self.n_s
        return lamina_length * (np.arange(self.n_s) + 0.5) - self.Lwe / 2.0
    
    @property
    def free_end_play(self) -> float:
        raise NotImplementedError(
            "CylindricalRollerGeometry.free_end_play not implemented yet -- "
            "cylindrical roller bearings don't have a groove-conformity clearance "
            "the same way ball bearings do; radial clearance is given directly "
            "by tolerance class (C0-C5), not derived here."
        )

    @property
    def f_i(self) -> float:
        raise NotImplementedError(
            "CylindricalRollerGeometry.f_i not implemented yet"
        )


    @property
    @abstractmethod
    def f_o(self) -> float: 
        raise NotImplementedError(
            "CylindricalRollerGeometry.f_o not implemented yet"
        )

    @property
    def r_ax(self):
        return self.Dwe/2

    @property
    def r_ay(self):
        return float("inf")

    @property
    def r_bx_inner(self) -> float:
        return (self.Dpw - self.Dwe) / 2

    @property
    def r_by_inner(self) -> float:
        return float("inf")

    @property
    def r_bx_outer(self) -> float:
        return (self.Dpw + self.Dwe) / 2

    @property
    def r_by_outer(self) -> float:
        return float("inf")