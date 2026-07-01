"""
AxisForge — Shaft Geometry Model

Geometric primitives for shaft modelling, in (r, theta, z) convention.
z is the absolute axial coordinate [mm]. theta [deg] is used only for
localized features (Keyway) — the shaft body itself is a solid of revolution.

Units: mm (lengths), mm^4 (area moments), degrees (angles).

References:
  Shigley §7-1 (shaft geometry conventions)
  Peterson §1 (stress raisers at geometric discontinuities)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from config import TOL_GEOMETRY_mm
from database.shaft.keyway.Parallel.parallel_keyway import lookup_parallel_keyway
from database.shaft.keyway.Woodruff_key.iso3912 import lookup_woodruff_keyway


class KeywayType(Enum):
    PARALLEL = auto()
    WOODRUFF = auto()
    SPLINE = auto()


@dataclass
class Keyway:
    """Localized stress raiser at absolute position (z_position, theta)."""
    label: str
    z_position: float
    width: float
    depth: float
    length: float
    theta: float = 0.0
    keyway_type: KeywayType = KeywayType.PARALLEL
    depth_min: Optional[float] = None
    depth_max: Optional[float] = None
    L_min: Optional[float] = None
    L_max: Optional[float] = None
    Kf_bending: Optional[float] = None
    Kf_torsion: Optional[float] = None

    @classmethod
    def from_standard(cls, label, shaft_diameter, z_position, theta=0.0,
                    keyway_type=KeywayType.PARALLEL, length=None, series=1):
        if keyway_type == KeywayType.PARALLEL:
            if length is None:
                raise ValueError("length is required for PARALLEL keyways")
            dims = lookup_parallel_keyway(shaft_diameter)
            return cls(
                label=label, z_position=z_position,
                width=dims["b"], depth=dims["t_shaft"], length=length,
                theta=theta, keyway_type=keyway_type,
                depth_min=dims["t_shaft_min"], depth_max=dims["t_shaft_max"],
                L_min=dims["L_min"], L_max=dims["L_max"],
            )
        if keyway_type == KeywayType.WOODRUFF:
            dims = lookup_woodruff_keyway(shaft_diameter, series=series)
            return cls(
                label=label, z_position=z_position,
                width=dims["b"], depth=dims["t_shaft"], length=dims["D"],
                theta=theta, keyway_type=keyway_type,
                depth_min=dims["t_shaft_min"], depth_max=dims["t_shaft_max"],
            )
        raise NotImplementedError(f"from_standard not implemented for {keyway_type}")

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("Keyway label cannot be empty")
        if self.z_position < 0:
            raise ValueError(f"z_position cannot be negative, got {self.z_position}")
        if self.width <= 0:
            raise ValueError(f"width must be > 0, got {self.width}")
        if self.depth <= 0:
            raise ValueError(f"depth must be > 0, got {self.depth}")
        if self.length <= 0:
            raise ValueError(f"length must be > 0, got {self.length}")
        if not 0.0 <= self.theta < 360.0:
            raise ValueError(f"theta must be in [0, 360), got {self.theta}")

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.width <= 0:
            errors.append(f"width must be > 0 (got {self.width})")
        if self.depth <= 0:
            errors.append(f"depth must be > 0 (got {self.depth})")
        if self.length <= 0:
            errors.append(f"length must be > 0 (got {self.length})")
        if self.L_min is not None and self.length < self.L_min:
            errors.append(f"length ({self.length}) below standard L_min ({self.L_min})")
        if self.L_max is not None and self.length > self.L_max:
            errors.append(f"length ({self.length}) above standard L_max ({self.L_max})")
        return errors


@dataclass
class Shoulder:
    """
    Fillet transition between two adjacent sections.

    r_over_d and D_over_d feed Peterson interpolation for Kt.
    Constraints: fillet_radius > 0, diameter_large > diameter_small,
    fillet_radius <= (D - d) / 2.
    """
    fillet_radius: float
    diameter_large: float
    diameter_small: float

    def __post_init__(self) -> None:
        if self.fillet_radius <= 0:
            raise ValueError(f"fillet_radius must be > 0, got {self.fillet_radius}")
        if self.diameter_large <= self.diameter_small:
            raise ValueError(
                f"diameter_large ({self.diameter_large}) must be > "
                f"diameter_small ({self.diameter_small})"
            )
        step_height = (self.diameter_large - self.diameter_small) / 2.0
        if self.fillet_radius > step_height + TOL_GEOMETRY_mm:
            raise ValueError(
                f"fillet_radius ({self.fillet_radius} mm) cannot exceed "
                f"step height ({step_height:.4f} mm)"
            )

    @property
    def r_over_d(self) -> float:
        return self.fillet_radius / self.diameter_small

    @property
    def D_over_d(self) -> float:
        return self.diameter_large / self.diameter_small

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.fillet_radius <= 0:
            errors.append(f"fillet_radius must be > 0, got {self.fillet_radius}")
        if self.diameter_large <= self.diameter_small:
            errors.append(
                f"diameter_large ({self.diameter_large}) must exceed "
                f"diameter_small ({self.diameter_small})"
            )
        return errors


@dataclass
class ShaftSection:
    """Uniform cylindrical segment. Shoulders are end transitions, not part of length."""
    length: float
    diameter: float
    inner_diameter: float = 0.0
    material_id: str = "S355"
    surface_finish_ra: float = 0.8
    shoulder_left: Optional[Shoulder] = None
    shoulder_right: Optional[Shoulder] = None
    keyways: list[Keyway] = field(default_factory=list)
    label: str = ""

    def __post_init__(self) -> None:
        if self.length <= 0:
            raise ValueError(f"ShaftSection length must be > 0, got {self.length}")
        if self.diameter <= 0:
            raise ValueError(f"ShaftSection diameter must be > 0, got {self.diameter}")
        if self.inner_diameter < 0:
            raise ValueError(f"inner_diameter cannot be negative, got {self.inner_diameter}")
        if self.inner_diameter >= self.diameter:
            raise ValueError(
                f"inner_diameter ({self.inner_diameter}) must be < "
                f"outer diameter ({self.diameter})"
            )
        if self.surface_finish_ra <= 0:
            raise ValueError(f"surface_finish_ra must be > 0, got {self.surface_finish_ra}")
        for kw in self.keyways:
            if kw.depth >= self.diameter / 2.0:
                raise ValueError(
                    f"Keyway '{kw.label}' depth ({kw.depth}) must be < "
                    f"section radius ({self.diameter / 2.0})"
                )

    @property
    def is_hollow(self) -> bool:
        return self.inner_diameter > 0.0

    @property
    def radius(self) -> float:
        return self.diameter / 2.0

    @property
    def area(self) -> float:
        return (math.pi / 4.0) * (self.diameter**2 - self.inner_diameter**2)

    @property
    def second_moment_of_area(self) -> float:
        """I = pi/64 * (d^4 - di^4) [mm^4]"""
        return (math.pi / 64.0) * (self.diameter**4 - self.inner_diameter**4)

    @property
    def polar_moment(self) -> float:
        """J = pi/32 * (d^4 - di^4) = 2I [mm^4]"""
        return (math.pi / 32.0) * (self.diameter**4 - self.inner_diameter**4)

    @property
    def section_modulus(self) -> float:
        """W = I / (d/2) [mm^3]. sigma_b = M / W"""
        return self.second_moment_of_area / (self.diameter / 2.0)

    @property
    def polar_section_modulus(self) -> float:
        """Wt = J / (d/2) [mm^3]. tau = T / Wt"""
        return self.polar_moment / (self.diameter / 2.0)

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.length <= 0:
            errors.append(f"length must be > 0 (got {self.length})")
        if self.diameter <= 0:
            errors.append(f"diameter must be > 0 (got {self.diameter})")
        if self.inner_diameter < 0:
            errors.append(f"inner_diameter cannot be negative (got {self.inner_diameter})")
        if self.inner_diameter >= self.diameter:
            errors.append(
                f"inner_diameter ({self.inner_diameter}) must be < diameter ({self.diameter})"
            )
        if self.shoulder_left is not None:
            errors.extend(f"shoulder_left: {e}" for e in self.shoulder_left.validate())
        if self.shoulder_right is not None:
            errors.extend(f"shoulder_right: {e}" for e in self.shoulder_right.validate())
        for kw in self.keyways:
            errors.extend(f"keyway '{kw.label}': {e}" for e in kw.validate())
            if kw.depth >= self.diameter / 2.0:
                errors.append(
                    f"keyway '{kw.label}': depth ({kw.depth}) must be < "
                    f"section radius ({self.diameter / 2.0})"
                )
        return errors


@dataclass
class Shaft:
    """
    Ordered sequence of ShaftSection. z=0 is the left face of sections[0].
    section_at(z): interior boundaries belong to the right section.
    """
    label: str
    sections: list[ShaftSection] = field(default_factory=list)

    def add_section(self, section: ShaftSection) -> None:
        if not isinstance(section, ShaftSection):
            raise TypeError(f"Expected ShaftSection, got {type(section)}")
        self.sections.append(section)

    @property
    def total_length(self) -> float:
        return sum(s.length for s in self.sections)

    @property
    def n_sections(self) -> int:
        return len(self.sections)

    def axial_start(self, index: int) -> float:
        if index < 0 or index >= len(self.sections):
            raise IndexError(
                f"Section index {index} out of range "
                f"(shaft has {len(self.sections)} sections)"
            )
        return sum(self.sections[i].length for i in range(index))

    def axial_end(self, index: int) -> float:
        if index < 0 or index >= len(self.sections):
            raise IndexError(
                f"Section index {index} out of range "
                f"(shaft has {len(self.sections)} sections)"
            )
        return sum(self.sections[i].length for i in range(index + 1))

    def section_at(self, z: float) -> tuple[ShaftSection, int]:
        L = self.total_length
        if z < -TOL_GEOMETRY_mm or z > L + TOL_GEOMETRY_mm:
            raise ValueError(f"z={z:.4f} mm is outside shaft extent [0, {L:.4f}] mm")

        z = max(0.0, min(z, L))
        cumulative = 0.0
        for i, section in enumerate(self.sections):
            next_cumulative = cumulative + section.length
            if z < next_cumulative  or i == len(self.sections) - 1:
                return section, i
            cumulative = next_cumulative

        return self.sections[-1], len(self.sections) - 1

    def diameter_at(self, z: float) -> float:
        section, _ = self.section_at(z)
        return section.diameter

    def I_at(self, z: float) -> float:
        section, _ = self.section_at(z)
        return section.second_moment_of_area

    def J_at(self, z: float) -> float:
        section, _ = self.section_at(z)
        return section.polar_moment

    def W_at(self, z: float) -> float:
        section, _ = self.section_at(z)
        return section.section_modulus

    def Wt_at(self, z: float) -> float:
        section, _ = self.section_at(z)
        return section.polar_section_modulus

    def shoulders(self) -> list[tuple[float, Shoulder]]:
        """List of (z_position, Shoulder), reporting only shoulder_right of each section."""
        result: list[tuple[float, Shoulder]] = []
        cumulative = 0.0
        for section in self.sections:
            cumulative += section.length
            if section.shoulder_right is not None:
                result.append((cumulative, section.shoulder_right))
        return result

    def validate(self) -> list[str]:
        errors: list[str] = []

        if not self.sections:
            errors.append("Shaft has no sections")
            return errors

        for i, s in enumerate(self.sections):
            errors.extend(f"Section {i} ({s.label!r}): {e}" for e in s.validate())

        for i in range(len(self.sections) - 1):
            left = self.sections[i]
            right = self.sections[i + 1]

            if left.shoulder_right is not None:
                d_small = left.shoulder_right.diameter_small
                d_large = left.shoulder_right.diameter_large
                if abs(d_small - right.diameter) > TOL_GEOMETRY_mm:
                    errors.append(
                        f"Shoulder mismatch at boundary {i}/{i+1}: "
                        f"section[{i}].shoulder_right.diameter_small={d_small:.4f} mm "
                        f"≠ section[{i+1}].diameter={right.diameter:.4f} mm"
                    )
                if abs(d_large - left.diameter) > TOL_GEOMETRY_mm:
                    errors.append(
                        f"Shoulder mismatch at boundary {i}/{i+1}: "
                        f"section[{i}].shoulder_right.diameter_large={d_large:.4f} mm "
                        f"≠ section[{i}].diameter={left.diameter:.4f} mm"
                    )

            if right.shoulder_left is not None:
                d_small = right.shoulder_left.diameter_small
                d_large = right.shoulder_left.diameter_large
                if abs(d_small - left.diameter) > TOL_GEOMETRY_mm:
                    errors.append(
                        f"Shoulder mismatch at boundary {i}/{i+1}: "
                        f"section[{i+1}].shoulder_left.diameter_small={d_small:.4f} mm "
                        f"≠ section[{i}].diameter={left.diameter:.4f} mm"
                    )
                if abs(d_large - right.diameter) > TOL_GEOMETRY_mm:
                    errors.append(
                        f"Shoulder mismatch at boundary {i}/{i+1}: "
                        f"section[{i+1}].shoulder_left.diameter_large={d_large:.4f} mm "
                        f"≠ section[{i+1}].diameter={right.diameter:.4f} mm"
                    )

        return errors

    def validate_or_raise(self) -> None:
        errors = self.validate()
        if errors:
            raise ValueError("Shaft validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    def __len__(self) -> int:
        return len(self.sections)

    def __repr__(self) -> str:
        return (
            f"Shaft(label={self.label!r}, "
            f"sections={len(self.sections)}, "
            f"total_length={self.total_length:.1f} mm)"
        )