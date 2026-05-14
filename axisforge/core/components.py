"""
core/components.py
Shaft-mounted mechanical components: bearings and gear elements.

Design rules:
  - position is always an absolute axial coordinate [mm], never a section index.
  - Bearing does not resolve axial load distribution — that is the solver's job.
  - GearElement stores force components as computed/imported; consistency checked in __post_init__.

Units: N, mm, degrees.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional
import math


class BearingType(Enum):
    """Rolling bearing type — determines life exponent and X/Y factors."""
    DEEP_GROOVE_BALL = auto()      # p = 3
    ANGULAR_CONTACT_BALL = auto()  # p = 3
    CYLINDRICAL_ROLLER = auto()    # p = 10/3
    TAPER_ROLLER = auto()          # p = 10/3
    SPHERICAL_ROLLER = auto()      # p = 10/3


@dataclass
class Bearing:
    """
    Rolling bearing positioned on the shaft.

    In Phase 1, C and C0 are supplied manually. Phase 3 adds catalogue lookup.

    Parameters
    ----------
    position : float
        Axial coordinate [mm]. Must be ≥ 0.
    bearing_type : BearingType
        Determines life exponent and default X/Y load factors.
    designation : str
        SKF or other catalogue designation (e.g. '6210'). Optional in Phase 1.
    C : float
        Basic dynamic load rating [N]. Required for L10 calculation.
    C0 : float
        Basic static load rating [N]. Required for S0 calculation.
    arrangement : str
        'fixed' — carries axial load; 'floating' — free to slide axially.
    contact_angle : float
        Nominal contact angle α [°]. 0° for DGBB in pure radial service.
    X : float
        Radial load factor for equivalent load P = X*Fr + Y*Fa.
        Phase 1 default: 1.0 (radial dominant simplification).
    Y : float
        Axial load factor. Phase 1 default: 0.0.
    label : str
        Human-readable identifier, e.g. 'A', 'B', 'input_side'.
    """
    position: float
    bearing_type: BearingType = BearingType.DEEP_GROOVE_BALL
    designation: str = ""
    C: float = 0.0
    C0: float = 0.0
    arrangement: str = "fixed"
    contact_angle: float = 0.0
    X: float = 1.0
    Y: float = 0.0
    label: str = ""

    def __post_init__(self) -> None:
        if self.position < 0:
            raise ValueError(
                f"Bearing position cannot be negative, got {self.position}"
            )
        if self.C < 0:
            raise ValueError(f"C must be ≥ 0, got {self.C}")
        if self.C0 < 0:
            raise ValueError(f"C0 must be ≥ 0, got {self.C0}")
        if self.arrangement not in ("fixed", "floating"):
            raise ValueError(
                f"arrangement must be 'fixed' or 'floating', "
                f"got '{self.arrangement}'"
            )
        if self.contact_angle < 0 or self.contact_angle >= 90:
            raise ValueError(
                f"contact_angle must be in [0, 90), got {self.contact_angle}"
            )
        if not 0.0 < self.X <= 1.0:
            raise ValueError(
                f"X (radial factor) must be in (0, 1], got {self.X}"
            )
        if self.Y < 0:
            raise ValueError(f"Y (axial factor) must be ≥ 0, got {self.Y}")

    @property
    def has_catalogue_data(self) -> bool:
        """True if both C and C0 are specified (non-zero)."""
        return self.C > 0.0 and self.C0 > 0.0

    @property
    def life_exponent(self) -> float:
        """
        p = 3 for ball bearings, 10/3 for roller bearings (ISO 281 §6).
        """
        _ball_types = {
            BearingType.DEEP_GROOVE_BALL,
            BearingType.ANGULAR_CONTACT_BALL,
        }
        return 3.0 if self.bearing_type in _ball_types else 10.0 / 3.0

    def __repr__(self) -> str:
        tag = self.designation or self.bearing_type.name
        return (
            f"Bearing(label={self.label!r}, pos={self.position} mm, "
            f"type={tag}, C={self.C:.0f} N, arrangement={self.arrangement!r})"
        )


@dataclass
class GearElement:
    """
    Gear (or sprocket/pulley) transmitting forces and torque into the shaft.

    Force components are in shaft coordinates:
      tangential_force (Wt) → typically in XZ plane
      radial_force     (Wr) → typically in YZ plane
      axial_force      (Wa) → along shaft axis (0 for spur gears)

    Consistency check: if pitch_diameter and tangential_force are both given,
    the stored torque is cross-checked against Wt × d/2. A >2% deviation raises
    ValueError (indicates data entry error or unit mismatch).

    Parameters
    ----------
    position : float
        Axial coordinate of gear centre [mm].
    tangential_force : float
        Wt [N]. Tangential component of resultant gear force.
    radial_force : float
        Wr [N]. Radial component (in tooth plane).
    axial_force : float
        Wa [N]. Thrust component (zero for spur gears).
    pitch_diameter : float
        d [mm]. Used to derive torque if not explicitly given.
    torque : float
        T [N·mm]. Transmitted torque. If 0 and pitch_diameter > 0, computed
        automatically as Wt × d/2.
    pressure_angle : float
        Normal pressure angle α_n [°]. Default 20°.
    helix_angle : float
        β [°]. 0° for spur gears.
    gearpie_source : str | None
        Path to source GEARpie file, for traceability.
    label : str
        Human-readable identifier.
    """
    position: float
    tangential_force: float
    radial_force: float
    axial_force: float = 0.0
    pitch_diameter: float = 0.0
    torque: float = 0.0
    pressure_angle: float = 20.0
    helix_angle: float = 0.0
    gearpie_source: Optional[str] = None
    label: str = ""

    def __post_init__(self) -> None:
        if self.position < 0:
            raise ValueError(
                f"GearElement position cannot be negative, got {self.position}"
            )
        if self.tangential_force < 0:
            raise ValueError(
                f"tangential_force must be ≥ 0 (magnitude), got {self.tangential_force}"
            )
        if self.radial_force < 0:
            raise ValueError(
                f"radial_force must be ≥ 0 (magnitude), got {self.radial_force}"
            )
        if self.axial_force < 0:
            raise ValueError(
                f"axial_force must be ≥ 0 (magnitude), got {self.axial_force}"
            )
        if self.pitch_diameter < 0:
            raise ValueError(f"pitch_diameter cannot be negative, got {self.pitch_diameter}")
        if not 0.0 <= self.pressure_angle < 90.0:
            raise ValueError(
                f"pressure_angle must be in [0, 90), got {self.pressure_angle}"
            )
        if not 0.0 <= self.helix_angle < 90.0:
            raise ValueError(
                f"helix_angle must be in [0, 90), got {self.helix_angle}"
            )

        # Auto-compute torque from geometry if not supplied
        if self.torque == 0.0 and self.pitch_diameter > 0.0:
            object.__setattr__(
                self, "torque", self.tangential_force * self.pitch_diameter / 2.0
            )
        elif self.torque > 0.0 and self.pitch_diameter > 0.0 and self.tangential_force > 0.0:
            expected = self.tangential_force * self.pitch_diameter / 2.0
            deviation = abs(self.torque - expected) / max(expected, 1.0)
            if deviation > 0.02:
                raise ValueError(
                    f"GearElement torque inconsistency: "
                    f"stored T = {self.torque:.1f} N·mm, "
                    f"Wt×d/2 = {expected:.1f} N·mm "
                    f"(deviation {deviation*100:.1f}% > 2%)"
                )

    @property
    def is_helical(self) -> bool:
        return self.helix_angle > 0.1

    @property
    def resultant_transverse_force(self) -> float:
        """√(Wt² + Wr²) — transverse resultant in the gear plane [N]."""
        return math.hypot(self.tangential_force, self.radial_force)

    def __repr__(self) -> str:
        return (
            f"GearElement(label={self.label!r}, pos={self.position} mm, "
            f"Wt={self.tangential_force:.0f} N, Wr={self.radial_force:.0f} N, "
            f"T={self.torque:.0f} N·mm)"
        )
