"""
Geometry of a stepped shaft: sections, shoulders and keyways, with the section properties used by the beam model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from axisforge.config import TOL_GEOMETRY_mm
from axisforge.database.shaft.keyway.Parallel.parallel_keyway import lookup_parallel_keyway
from axisforge.database.shaft.keyway.Woodruff_key.iso3912 import lookup_woodruff_keyway


class KeywayType(Enum):
    """Keyway family: parallel key, Woodruff key or spline (spline lookup is not implemented)."""
    PARALLEL = auto()
    WOODRUFF = auto()
    SPLINE = auto()


@dataclass
class Keyway:
    """Localized stress raiser (keyway) at an axial position and a circumferential angle.

    The shaft body is a solid of revolution; only the keyway carries an angle.

    Parameters
    ----------
    label : str
        Identifier; must not be empty.
    z_position : float
        Axial position of the keyway [mm]; must be >= 0. The reference is
        open (review item): this module does not use the value, and the only
        consumer (``ShaftPostProcessor.process``) adds the start of the owning
        section to it, i.e. reads it as measured from the left face of that
        section, not from the left face of ``sections[0]``. Not checked
        against the section or shaft length.
    width : float
        Keyway width ``b`` [mm]; must be > 0.
    depth : float
        Keyway depth in the shaft ``t_shaft`` [mm]; must be > 0 and, inside a
        ShaftSection, < section radius.
    length : float
        Keyway length [mm] (for a Woodruff key, the key diameter ``D``);
        must be > 0.
    theta : float, optional
        Circumferential angle of the keyway [deg], in [0, 360). Its
        reference direction is not established in this module.
    keyway_type : KeywayType, optional
        Keyway family.
    depth_min, depth_max : float or None, optional
        Standard tolerance limits of ``depth`` [mm].
    L_min, L_max : float or None, optional
        Standard limits of ``length`` [mm]; checked in :meth:`validate`.
    Kf_bending, Kf_torsion : float or None, optional
        Fatigue notch factors in bending and torsion [-]; not computed here.
    """
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
        """Build a keyway whose dimensions come from the standard-dimension tables.

        Parameters
        ----------
        label : str
            Identifier.
        shaft_diameter : float
            Shaft diameter at the keyway [mm], used to look up the table row.
        z_position : float
            Axial position of the keyway [mm]; see the class docstring for its reference.
        theta : float, optional
            Circumferential angle [deg].
        keyway_type : KeywayType, optional
            ``PARALLEL`` or ``WOODRUFF``.
        length : float or None, optional
            Keyway length [mm]; required for ``PARALLEL`` (a Woodruff key
            takes its length from the table diameter ``D``).
        series : int, optional
            Woodruff series passed to the table lookup [-].

        Returns
        -------
        Keyway
            Keyway with ``b``, ``t_shaft`` and its limits taken from the
            tables of ``axisforge.database.shaft.keyway``.

        Raises
        ------
        ValueError
            If ``length`` is None for a ``PARALLEL`` keyway.
        NotImplementedError
            For any other ``keyway_type``.

        Notes
        -----
        The standard behind each table is named in the database modules, not
        here (source to find).
        """
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
        """Raise ValueError if label is empty, z_position < 0, width/depth/length <= 0 or theta outside [0, 360) deg."""
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
        """Geometry errors of this keyway (empty list if none).

        Returns
        -------
        list of str
            Messages for width, depth or length <= 0, and for ``length``
            outside ``[L_min, L_max]`` when those limits are set [mm].
        """
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
    """Fillet shoulder between two adjacent sections of different diameter.

    ``r_over_d`` and ``D_over_d`` are the ratios r/d and D/d used to read a
    theoretical stress-concentration factor Kt from a chart (named
    "Peterson" in the code; source to find). The shoulder belongs to the
    Shaft, attached with :meth:`Shaft.set_transition` and keyed by the
    boundary index; a ShaftSection does not hold it.

    Parameters
    ----------
    fillet_radius : float
        Fillet radius ``r`` [mm]; must be > 0 and <= (D - d) / 2 (plus
        ``TOL_GEOMETRY_mm``).
    diameter_large : float
        Larger diameter ``D`` [mm]; must exceed ``diameter_small``.
    diameter_small : float
        Smaller diameter ``d`` [mm].
    """
    fillet_radius: float
    diameter_large: float
    diameter_small: float

    def __post_init__(self) -> None:
        """Raise ValueError if r <= 0, D <= d or r exceeds the step height (D - d) / 2."""
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
        """Ratio r/d of fillet radius to smaller diameter [-]."""
        return self.fillet_radius / self.diameter_small

    @property
    def D_over_d(self) -> float:
        """Ratio D/d of larger to smaller diameter [-]."""
        return self.diameter_large / self.diameter_small

    def validate(self) -> list[str]:
        """Geometry errors of this shoulder (empty list if none).

        Returns
        -------
        list of str
            Messages for ``fillet_radius`` <= 0 or ``diameter_large`` <=
            ``diameter_small``.
        """
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
    """Uniform cylindrical (solid or hollow) shaft segment.

    The section properties below feed the beam elements (``area`` and
    ``second_moment_of_area``) and the stress evaluation (``section_modulus``,
    ``polar_section_modulus``). Shoulders are held by the Shaft
    (``Shaft.transitions``), not by the section.

    Parameters
    ----------
    length : float
        Axial length [mm]; must be > 0.
    diameter : float
        Outer diameter ``d`` [mm]; must be > 0.
    inner_diameter : float, optional
        Bore diameter ``di`` [mm]; 0 for a solid section; must be
        < ``diameter``.
    material_id : str, optional
        Key of the material in ``axisforge.core.materials``.
    surface_finish_ra : float, optional
        Surface roughness Ra; must be > 0. No unit is enforced here; the
        construction records label it [um] (``outputs/construction/shaft.py``).
    keyways : list of Keyway, optional
        Keyways on this section; each ``depth`` must be < ``radius``.
    label : str, optional
        Identifier.

    Notes
    -----
    Section properties are those of a thick-walled circular ring:
    ``A = pi/4 (d^2 - di^2)``, ``I = pi/64 (d^4 - di^4)``,
    ``J = pi/32 (d^4 - di^4)``, ``W = I / (d/2)``, ``Wt = J / (d/2)``.
    Keyways do not reduce them.

    References
    ----------
    .. [1] Onate2013_vol2, Fig. 1.3 (solid circle A = pi R^2, I = pi R^4 / 4).
    .. [2] Onate2013_vol2, Fig. 4.11 (J of the solid circle and of the thick
       cylinder, 1/2 pi (ro^4 - ri^4)).

    Thick-hollow A and I, and the moduli W and Wt: (source to find).
    """
    length: float
    diameter: float
    inner_diameter: float = 0.0
    material_id: str = "S355"
    surface_finish_ra: float = 0.8
    keyways: list[Keyway] = field(default_factory=list)
    label: str = ""

    def __post_init__(self) -> None:
        """Raise ValueError for length or diameter <= 0, inner_diameter < 0 or >= diameter, Ra <= 0, or a keyway deeper than the radius."""
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
        """True if the section has a bore (``inner_diameter`` > 0)."""
        return self.inner_diameter > 0.0

    @property
    def radius(self) -> float:
        """Outer radius d/2 [mm]."""
        return self.diameter / 2.0

    @property
    def area(self) -> float:
        """Cross-section area A = pi/4 (d^2 - di^2) [mm^2]."""
        return (math.pi / 4.0) * (self.diameter**2 - self.inner_diameter**2)

    @property
    def second_moment_of_area(self) -> float:
        """Second moment of area I = pi/64 (d^4 - di^4) about a diameter [mm^4]."""
        return (math.pi / 64.0) * (self.diameter**4 - self.inner_diameter**4)

    @property
    def polar_moment(self) -> float:
        """Polar second moment of area J = pi/32 (d^4 - di^4) = 2 I [mm^4]."""
        return (math.pi / 32.0) * (self.diameter**4 - self.inner_diameter**4)

    @property
    def section_modulus(self) -> float:
        """Bending section modulus W = I / (d/2) [mm^3], so that sigma_b = M / W."""
        return self.second_moment_of_area / (self.diameter / 2.0)

    @property
    def polar_section_modulus(self) -> float:
        """Torsional section modulus Wt = J / (d/2) [mm^3], so that tau = T / Wt."""
        return self.polar_moment / (self.diameter / 2.0)

    def validate(self) -> list[str]:
        """Geometry errors of this section and its keyways (empty list if none).

        Returns
        -------
        list of str
            Messages for length or diameter <= 0 [mm], bore out of range,
            keyway errors and keyways deeper than the radius.
        """
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
    """Stepped shaft: ordered ShaftSections plus the shoulders between them.

    The axial coordinate (named ``z`` in the arguments of this module; the
    same axis that the mesh and the README call ``x``, increasing left to
    right) is measured from the left face of ``sections[0]`` [mm]. Radial and
    angular conventions: see the README section Conventions.

    Parameters
    ----------
    label : str
        Identifier.
    sections : list of ShaftSection, optional
        Sections from left to right.
    transitions : dict of int to Shoulder, optional
        Key ``i`` is the boundary between ``sections[i]`` and
        ``sections[i + 1]``. Set through :meth:`set_transition`, not by
        writing to a section.

    Notes
    -----
    In :meth:`section_at`, an interior boundary belongs to the right section.
    """
    label: str
    sections: list[ShaftSection] = field(default_factory=list)
    transitions: dict[int, Shoulder] = field(default_factory=dict)

    def add_section(self, section: ShaftSection) -> None:
        """Append a section at the right end of the shaft.

        Raises
        ------
        TypeError
            If ``section`` is not a ShaftSection.
        """
        if not isinstance(section, ShaftSection):
            raise TypeError(f"Expected ShaftSection, got {type(section)}")
        self.sections.append(section)

    def set_transition(self, index: int, shoulder: Shoulder) -> None:
        """Attach a Shoulder to the boundary between sections[index] and sections[index + 1].

        Parameters
        ----------
        index : int
            Boundary index, checked against the current number of sections
            (add the sections first).
        shoulder : Shoulder
            Fillet shoulder at that boundary.

        Raises
        ------
        TypeError
            If ``shoulder`` is not a Shoulder.
        IndexError
            If ``index`` is not a boundary of the current shaft.
        """
        if not isinstance(shoulder, Shoulder):
            raise TypeError(f"Expected Shoulder, got {type(shoulder)}")
        n_boundaries = max(len(self.sections) - 1, 0)
        if index < 0 or index >= n_boundaries:
            raise IndexError(
                f"transition index {index} out of range "
                f"(shaft has {len(self.sections)} sections, "
                f"{n_boundaries} boundaries)"
            )
        self.transitions[index] = shoulder

    @property
    def total_length(self) -> float:
        """Sum of the section lengths [mm]."""
        return sum(s.length for s in self.sections)

    @property
    def n_sections(self) -> int:
        """Number of sections [-]."""
        return len(self.sections)

    def axial_start(self, index: int) -> float:
        """Axial coordinate of the left face of ``sections[index]`` [mm].

        Raises
        ------
        IndexError
            If ``index`` is out of range.
        """
        if index < 0 or index >= len(self.sections):
            raise IndexError(
                f"Section index {index} out of range "
                f"(shaft has {len(self.sections)} sections)"
            )
        return sum(self.sections[i].length for i in range(index))

    def axial_end(self, index: int) -> float:
        """Axial coordinate of the right face of ``sections[index]`` [mm].

        Raises
        ------
        IndexError
            If ``index`` is out of range.
        """
        if index < 0 or index >= len(self.sections):
            raise IndexError(
                f"Section index {index} out of range "
                f"(shaft has {len(self.sections)} sections)"
            )
        return sum(self.sections[i].length for i in range(index + 1))

    def section_at(self, z: float) -> tuple[ShaftSection, int]:
        """Section containing the axial coordinate ``z``.

        Parameters
        ----------
        z : float
            Axial coordinate [mm]; values within ``TOL_GEOMETRY_mm`` outside
            [0, total_length] are clamped to the ends.

        Returns
        -------
        section : ShaftSection
        index : int
            Index of ``section`` in ``sections`` [-]. A ``z`` on an interior
            boundary returns the section to its right.

        Raises
        ------
        ValueError
            If ``z`` lies outside the shaft by more than ``TOL_GEOMETRY_mm``.
        """
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
        """Outer diameter of the section at ``z`` [mm]."""
        section, _ = self.section_at(z)
        return section.diameter

    def I_at(self, z: float) -> float:
        """Second moment of area I of the section at ``z`` [mm^4]."""
        section, _ = self.section_at(z)
        return section.second_moment_of_area

    def J_at(self, z: float) -> float:
        """Polar second moment of area J of the section at ``z`` [mm^4]."""
        section, _ = self.section_at(z)
        return section.polar_moment

    def W_at(self, z: float) -> float:
        """Bending section modulus W of the section at ``z`` [mm^3]."""
        section, _ = self.section_at(z)
        return section.section_modulus

    def Wt_at(self, z: float) -> float:
        """Torsional section modulus Wt of the section at ``z`` [mm^3]."""
        section, _ = self.section_at(z)
        return section.polar_section_modulus

    def shoulders(self) -> list[tuple[float, Shoulder]]:
        """Shoulders with their axial position, in ascending boundary-index order.

        Returns
        -------
        list of (float, Shoulder)
            ``(z, shoulder)`` with ``z = axial_end(i) = axial_start(i + 1)``
            [mm], one per entry of ``transitions``.
        """
        return [
            (self.axial_end(i), self.transitions[i])
            for i in sorted(self.transitions)
        ]

    def validate(self) -> list[str]:
        """Geometry errors of the shaft (empty list if none).

        Returns
        -------
        list of str
            Errors of every section and shoulder, transitions on a boundary
            that does not exist, and shoulder diameters that differ from the
            two adjoining section diameters [mm].
        """
        errors: list[str] = []

        if not self.sections:
            errors.append("Shaft has no sections")
            return errors

        for i, s in enumerate(self.sections):
            errors.extend(f"Section {i} ({s.label!r}): {e}" for e in s.validate())

        n_boundaries = len(self.sections) - 1
        for i, shoulder in self.transitions.items():
            if i < 0 or i >= n_boundaries:
                errors.append(
                    f"transitions[{i}]: no such boundary "
                    f"(shaft has {n_boundaries} boundaries)"
                )
                continue

            errors.extend(f"transitions[{i}]: {e}" for e in shoulder.validate())

            left, right = self.sections[i], self.sections[i + 1]
            shoulder_diams = {shoulder.diameter_large, shoulder.diameter_small}
            section_diams = {left.diameter, right.diameter}
            if shoulder_diams != section_diams:
                errors.append(
                    f"Shoulder mismatch at boundary {i}/{i+1}: "
                    f"transitions[{i}] diameters "
                    f"({shoulder.diameter_small:.4f}, {shoulder.diameter_large:.4f}) mm "
                    f"do not match section diameters "
                    f"({left.diameter:.4f}, {right.diameter:.4f}) mm"
                )

        return errors

    def validate_or_raise(self) -> None:
        """Raise ValueError listing every error returned by :meth:`validate`."""
        errors = self.validate()
        if errors:
            raise ValueError("Shaft validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    def __len__(self) -> int:
        """Number of sections."""
        return len(self.sections)

    def __repr__(self) -> str:
        """Label, number of sections and total length [mm]."""
        return (
            f"Shaft(label={self.label!r}, "
            f"sections={len(self.sections)}, "
            f"total_length={self.total_length:.1f} mm)"
        )