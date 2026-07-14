"""
mechanical_system/loads.py

Point load primitives applied to a shaft.

Transverse loads (RadialLoad, ExternalMoment) are defined by a magnitude and
an angular position theta around the shaft cross-section, then decomposed
into the two principal bending planes (XY, XZ) for the StaticsSolver.

Sign convention:
  - theta_deg measured from +Y toward +Z, right-hand rule about +X.
  - magnitude always >= 0; direction fully encoded by theta.
  - AxialLoad, TorqueLoad act along/about +X — no angular decomposition needed.

Units: N for forces, N*mm for moments/torques, mm for position.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
import numpy as np


class LoadPlane(Enum):
    """Principal bending plane — used only as a decomposition result key,
    never as user input."""
    XY = auto()
    XZ = auto()


class Load:
    """Base class — not instantiated directly."""

    def __init__(self, position: float, label: str = "", source: str = "user"):
        self.position = position
        self.label = label
        self.source = source   # "user" | "gear_mesh" | "bearing_reaction"

    def validate(self) -> list[str]:
        errors: list[str] = []
        tag = self.label or self.__class__.__name__
        if self.position < 0:
            errors.append(f"{tag}: position must be >= 0, got {self.position}")
        return errors

    def validate_or_raise(self) -> None:
        errors = self.validate()
        if errors:
            raise ValueError("\n".join(errors))


class RadialLoad(Load):
    """
    Transverse point force at arbitrary angular position theta.

    Parameters
    ----------
    position  : axial coordinate [mm]
    magnitude : force magnitude [N], >= 0
    theta_deg : angular position [deg], measured from +Y toward +Z
    source    : "user" | "gear_mesh" | "bearing_reaction"
    """

    def __init__(self, position: float, magnitude: float,
                 theta_deg: float = 0.0, label: str = "",
                 source: str = "user"):
        super().__init__(position, label, source)
        self.magnitude = magnitude
        self.theta_deg = theta_deg % 360.0
        self.theta = np.radians(self.theta_deg)

    @property
    def Fy(self) -> float:
        return self.magnitude * np.cos(self.theta)

    @property
    def Fz(self) -> float:
        return self.magnitude * np.sin(self.theta)

    def component(self, plane: LoadPlane) -> float:
        """Signed magnitude projected onto a principal plane."""
        return self.Fy if plane == LoadPlane.XY else self.Fz

    def validate(self) -> list[str]:
        errors = super().validate()
        tag = self.label or self.__class__.__name__
        if self.magnitude < 0:
            errors.append(f"{tag}: magnitude must be >= 0, got {self.magnitude}")
        return errors

    def __repr__(self) -> str:
        return (f"RadialLoad(position={self.position:.2f} mm, "
                f"F={self.magnitude:.2f} N, theta={self.theta_deg:.1f}°, "
                f"Fy={self.Fy:.2f} N, Fz={self.Fz:.2f} N, "
                f"source={self.source!r}, label={self.label!r})")


class AxialLoad(Load):
    """Force along shaft axis. Positive = tensile (+X)."""

    def __init__(self, position: float, magnitude: float, label: str = "",
                 source: str = "user"):
        super().__init__(position, label, source)
        self.magnitude = magnitude

    def __repr__(self) -> str:
        return (f"AxialLoad(position={self.position:.2f} mm, "
                f"Fa={self.magnitude:.2f} N, source={self.source!r}, "
                f"label={self.label!r})")


class TorqueLoad(Load):
    """Torque about shaft axis. Positive = CCW viewed from +X. [N*m]"""

    def __init__(self, position: float, magnitude: float, label: str = "",
                 source: str = "user"):
        super().__init__(position, label, source)
        self.magnitude = magnitude

    def __repr__(self) -> str:
        return (f"TorqueLoad(position={self.position:.2f} mm, "
                f"T={self.magnitude:.2f} N·m, source={self.source!r}, "
                f"label={self.label!r})")


class ExternalMoment(Load):
    """
    Applied bending moment at arbitrary angular orientation theta.
    Same decomposition convention as RadialLoad — theta defines the
    moment vector direction in the YZ cross-section.
    """

    def __init__(self, position: float, magnitude: float,
                 theta_deg: float = 0.0, label: str = "",
                 source: str = "user"):
        super().__init__(position, label, source)
        self.magnitude = magnitude
        self.theta_deg = theta_deg % 360.0
        self.theta = np.radians(self.theta_deg)

    @property
    def My(self) -> float:
        return self.magnitude * np.cos(self.theta)

    @property
    def Mz(self) -> float:
        return self.magnitude * np.sin(self.theta)

    def component(self, plane: LoadPlane) -> float:
        return self.My if plane == LoadPlane.XY else self.Mz

    def __repr__(self) -> str:
        return (f"ExternalMoment(position={self.position:.2f} mm, "
                f"M={self.magnitude:.2f} N·m, theta={self.theta_deg:.1f}°, "
                f"source={self.source!r}, label={self.label!r})")


Load_T = RadialLoad | AxialLoad | TorqueLoad | ExternalMoment


# ===========================================================================
# LoadingProfile — fatigue cycle decomposition
# ===========================================================================

@dataclass(frozen=True)
class LoadingProfile:
    label: str = ""
    R: float = 1.0

    def __post_init__(self) -> None:
        if not -1.0 <= self.R <= 1.0:
            raise ValueError(f"R must be in [-1.0, 1.0], got {self.R}")

    @property
    def sigma_mean_factor(self) -> float:
        return (1 + self.R) / 2

    @property
    def sigma_amplitude_factor(self) -> float:
        return abs(1 - self.R) / 2

    @property
    def is_static(self) -> bool:
        return self.R == 1.0

    @property
    def is_fully_reversed(self) -> bool:
        return self.R == -1.0