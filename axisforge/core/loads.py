"""
core/loads.py
Point load primitives applied to the shaft.

All loads are positioned by absolute axial coordinate [mm].
Sign convention: magnitude is always positive (direction is encoded in LoadPlane
and by the solver sign convention documented in solvers/statics.py).

Units: N for forces, N·mm for moments and torques.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto


class LoadPlane(Enum):
    """
    Plane in which a transverse load acts.

    XZ — horizontal plane (tangential gear forces are assigned here by default)
    XY — vertical plane (radial gear forces, gravity)
    AXIAL — along shaft axis (helical gear thrust, axial preload)
    """
    XZ = auto()
    XY = auto()
    AXIAL = auto()


@dataclass
class RadialLoad:
    """
    Point radial (transverse) force applied at a specific axial position.

    Parameters
    ----------
    position : float
        Axial coordinate [mm]. Must be ≥ 0.
    magnitude : float
        Force magnitude [N]. Pass absolute value; sign convention handled by solver.
    plane : LoadPlane
        XZ or XY — the plane in which the force acts.
    label : str
        Optional identifier.
    """
    position: float
    magnitude: float
    plane: LoadPlane
    label: str = ""

    def __post_init__(self) -> None:
        if self.position < 0:
            raise ValueError(
                f"RadialLoad position cannot be negative, got {self.position}"
            )
        if self.plane == LoadPlane.AXIAL:
            raise ValueError(
                "RadialLoad cannot have plane=AXIAL. Use AxialLoad instead."
            )


@dataclass
class AxialLoad:
    """
    Point axial force (along shaft axis).

    Positive direction: +x (right, away from datum).

    Parameters
    ----------
    position : float
        Axial coordinate [mm].
    magnitude : float
        Force [N]. Positive = tensile (+x direction), negative = compressive.
    label : str
        Optional identifier.
    """
    position: float
    magnitude: float
    label: str = ""

    def __post_init__(self) -> None:
        if self.position < 0:
            raise ValueError(
                f"AxialLoad position cannot be negative, got {self.position}"
            )


@dataclass
class TorqueLoad:
    """
    External torque applied at a specific axial position.

    Convention: positive = counter-clockwise viewed from +x (right-hand rule).
    Typically applied at gear/coupling positions.

    Parameters
    ----------
    position : float
        Axial coordinate [mm].
    magnitude : float
        Torque [N·mm]. Sign encodes direction.
    label : str
        Optional identifier.
    """
    position: float
    magnitude: float
    label: str = ""

    def __post_init__(self) -> None:
        if self.position < 0:
            raise ValueError(
                f"TorqueLoad position cannot be negative, got {self.position}"
            )


@dataclass
class ExternalMoment:
    """
    External bending moment applied at a specific axial position.

    Used for overhanging loads, coupling misalignment moments, etc.

    Parameters
    ----------
    position : float
        Axial coordinate [mm].
    magnitude : float
        Moment [N·mm]. Sign encodes direction per right-hand rule.
    plane : LoadPlane
        XZ or XY — the plane in which the moment acts.
    label : str
        Optional identifier.
    """
    position: float
    magnitude: float
    plane: LoadPlane
    label: str = ""

    def __post_init__(self) -> None:
        if self.position < 0:
            raise ValueError(
                f"ExternalMoment position cannot be negative, got {self.position}"
            )
        if self.plane == LoadPlane.AXIAL:
            raise ValueError(
                "ExternalMoment cannot have plane=AXIAL."
            )


# Union type alias — use in type hints for any load container
Load = RadialLoad | AxialLoad | TorqueLoad | ExternalMoment


# ===========================================================================
# LoadingProfile — fatigue cycle decomposition
# ===========================================================================

@dataclass(frozen=True)
class LoadingProfile:
    """
    Cycle decomposition parameters for combined bending + torsion fatigue.

    Defines how the peak loads from StaticsResult are split into alternating
    and mean components, independently for bending and torsion.

    Stress ratio (Shigley §6-12, Eq. 6-37):
        R = σ_min / σ_max        [-1 ≤ R ≤ 1]

    Decomposition from peak value (treated as σ_max):
        Ma = M_peak · (1 - R_bend) / 2
        Mm = M_peak · (1 + R_bend) / 2
        Ta = T_peak · (1 - R_tors) / 2
        Tm = T_peak · (1 + R_tors) / 2

    Special cases:
        R = -1  → fully reversed  → Ma = M_peak, Mm = 0
        R =  0  → pulsating       → Ma = Mm = M_peak / 2
        R = +1  → static          → Ma = 0,      Mm = M_peak

    Amplitude ratio (Eq. 6-38):
        A = σ_a / σ_m = (1 - R) / (1 + R)    [R ≠ -1]

    Factory methods:
        LoadingProfile.rotating_shaft()  → R_bend=-1, R_tors=+1  (default)
        LoadingProfile.pulsating()       → R_bend= 0, R_tors= 0
        LoadingProfile.static_load()     → R_bend=+1, R_tors=+1
        LoadingProfile.custom(R_b, R_t)  → arbitrary

    References:
        Shigley 10th ed. §6-12, Fig. 6-23, Eq. 6-36 to 6-38.
        Shigley §7-1 — rotating shaft canonical assumption.
    """

    R_bend: float = -1.0   # fully reversed bending (rotating shaft default)
    R_tors: float = +1.0   # steady torsion (rotating shaft default)

    def __post_init__(self) -> None:
        for name, val in (("R_bend", self.R_bend), ("R_tors", self.R_tors)):
            if not (-1.0 <= val <= 1.0):
                raise ValueError(
                    f"{name}={val} outside valid range [-1, 1]. "
                    f"R = σ_min / σ_max requires |R| ≤ 1."
                )

    # ------------------------------------------------------------------
    # Derived: amplitude ratios
    # ------------------------------------------------------------------

    @property
    def A_bend(self) -> float:
        """A = σ_a / σ_m for bending. Returns inf when R_bend = -1 (σ_m = 0)."""
        denom = 1.0 + self.R_bend
        if abs(denom) < 1e-12:
            return float("inf")
        return (1.0 - self.R_bend) / denom

    @property
    def A_tors(self) -> float:
        """A = τ_a / τ_m for torsion. Returns inf when R_tors = -1 (τ_m = 0)."""
        denom = 1.0 + self.R_tors
        if abs(denom) < 1e-12:
            return float("inf")
        return (1.0 - self.R_tors) / denom

    # ------------------------------------------------------------------
    # Decomposition
    # ------------------------------------------------------------------

    def decompose_bending(self, M_peak: float) -> tuple[float, float]:
        """
        Split peak bending moment into (Ma, Mm) [N·mm].

        M_peak is treated as σ_max (the maximum of the cycle).
        """
        Ma = M_peak * (1.0 - self.R_bend) / 2.0
        Mm = M_peak * (1.0 + self.R_bend) / 2.0
        return Ma, Mm

    def decompose_torsion(self, T_peak: float) -> tuple[float, float]:
        """
        Split peak torque into (Ta, Tm) [N·mm].

        T_peak is treated as τ_max (the maximum of the cycle).
        """
        Ta = T_peak * (1.0 - self.R_tors) / 2.0
        Tm = T_peak * (1.0 + self.R_tors) / 2.0
        return Ta, Tm

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def rotating_shaft(cls) -> "LoadingProfile":
        """
        Standard rotating shaft: fully reversed bending, steady torsion.
        R_bend = -1, R_tors = +1.
        """
        return cls(R_bend=-1.0, R_tors=1.0)

    @classmethod
    def pulsating(cls) -> "LoadingProfile":
        """Pulsating cycle (0 to max) for both bending and torsion."""
        return cls(R_bend=0.0, R_tors=0.0)

    @classmethod
    def static_load(cls) -> "LoadingProfile":
        """No alternating component. For static verification only."""
        return cls(R_bend=1.0, R_tors=1.0)

    @classmethod
    def custom(cls, R_bend: float, R_tors: float) -> "LoadingProfile":
        """Arbitrary stress ratios for bending and torsion."""
        return cls(R_bend=R_bend, R_tors=R_tors)