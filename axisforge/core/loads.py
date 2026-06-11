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

    # aqui posso trabalhar força a força
        # tenho os campos de tensao definidos para cada força no codigo ou seja posso fazer definição individual do tipo de carga e depois aplicar a cada uma a sua contribuição de tensão e depois somar tudo no final para obter o resultado total da tensão em cada ponto do eixo

        # ou seja isto é algo modular para poder usar depois e chamar à vontade 

        # depois no FatiguePostProcessing pego nos valores depois de serem tratados aqui e aplico as concentrações de carga e tenho o valor total do campo de tensões no veio
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