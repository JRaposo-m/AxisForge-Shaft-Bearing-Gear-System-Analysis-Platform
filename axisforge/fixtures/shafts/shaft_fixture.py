"""
fixtures/shafts/shaft_fixture.py

Generic shaft fixture for N-section stepped shafts.

Classes
-------
  SectionSpec    -- declarative descriptor for one shaft section.
                    Carries length, diameter, material, surface finish,
                    label, and optional Shoulder objects at each face.

  ShaftFixture   -- immutable wrapper around a Shaft, storing the ordered
                    list of SectionSpec objects for derived copies.

Factories
---------
  make_shaft(sections, name)          -> ShaftFixture
      Generic factory: builds a Shaft from an arbitrary list of SectionSpec.

  make_stepped_3section(...)          -> ShaftFixture
      Convenience factory for the canonical seat_A | body | seat_B geometry.
      Derives SectionSpec list internally and calls make_shaft.

Named reference instance
------------------------
  SHAFT_50_30_200  -- seat_A(d=30, L=30) | body(d=50, L=140) | seat_B(d=30, L=30)
                      total_length=200 mm, fillet_r=2.0 mm, AISI_1045

Warning: do not mutate named instances. Use with_sections() or
with_params() to produce modified copies.

Dependencies (core only):
  axisforge.core.machine_elements.Shaft.shaft  Shaft, ShaftSection, Shoulder

References:
  Shigley SS7-1  -- shaft geometry conventions
  Peterson SS1   -- stress raisers at geometric discontinuities
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from axisforge.core.machine_elements.Shaft.shaft import Shaft, ShaftSection, Shoulder


# ===========================================================================
# SectionSpec -- declarative section descriptor
# ===========================================================================

@dataclass(frozen=True)
class SectionSpec:
    """
    Declarative descriptor for one shaft section.

    Used as input to make_shaft(). Each SectionSpec produces exactly one
    ShaftSection in the assembled Shaft. Shoulder objects (from the core)
    are attached directly -- no intermediate spec layer.

    Attributes
    ----------
    length         : float          -- axial length [mm]
    diameter       : float          -- outer diameter [mm]
    material_id    : str            -- material identifier (core/materials.py)
    surface_ra     : float          -- arithmetic mean roughness Ra [um]
    label          : str            -- section label (used in reporting)
    shoulder_left  : Shoulder|None  -- transition at the left face of this section
    shoulder_right : Shoulder|None  -- transition at the right face of this section

    Notes
    -----
    Shoulders describe the geometric transition (step) between adjacent
    sections. Shoulder(diameter_large, diameter_small, fillet_radius) is
    imported from the core. For a symmetric stepped shaft the same Shoulder
    object may be reused on both faces of the body section.

    Inner diameter (hollow sections) is not exposed here. Construct
    ShaftSection directly and wrap via make_shaft() if a hollow section
    is required.
    """

    length:         float
    diameter:       float
    material_id:    str            = "AISI_1045"
    surface_ra:     float          = 0.8
    label:          str            = ""
    shoulder_left:  Optional[Shoulder] = field(default=None, compare=False)
    shoulder_right: Optional[Shoulder] = field(default=None, compare=False)


# ===========================================================================
# ShaftFixture -- immutable N-section shaft wrapper
# ===========================================================================

@dataclass(frozen=True)
class ShaftFixture:
    """
    Immutable fixture for a stepped N-section shaft.

    Wraps a Shaft core object and stores the ordered list of SectionSpec
    objects that produced it, so that with_sections() can reconstruct a
    modified copy without parsing Shaft internals.

    Attributes
    ----------
    shaft    : Shaft            -- core shaft object
    sections : list[SectionSpec]-- ordered section descriptors (left to right)
    name     : str              -- shaft name (matches Shaft.name)

    Derived
    -------
    total_length : float  -- sum of all section lengths [mm]
    n_sections   : int    -- number of sections
    """

    shaft:    Shaft
    sections: tuple[SectionSpec, ...]   # tuple for hashability (frozen dataclass)
    name:     str

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def total_length(self) -> float:
        """Total shaft length [mm]."""
        return sum(s.length for s in self.sections)

    @property
    def n_sections(self) -> int:
        """Number of sections."""
        return len(self.sections)

    # ------------------------------------------------------------------
    # Derived copies
    # ------------------------------------------------------------------

    def with_sections(self, sections: list[SectionSpec]) -> ShaftFixture:
        """
        Return a new ShaftFixture built from a different list of SectionSpec.

        The name is inherited. Useful for parametric geometry sweeps where
        only one or two sections change.
        """
        return make_shaft(sections=sections, name=self.name)

    def with_name(self, name: str) -> ShaftFixture:
        """Return a new ShaftFixture with a different shaft name."""
        return make_shaft(sections=list(self.sections), name=name)

    # ------------------------------------------------------------------
    # Convenience: stepped-3-section modifier
    # ------------------------------------------------------------------

    def with_params(
        self,
        total_length: Optional[float] = None,
        d_seat:       Optional[float] = None,
        d_body:       Optional[float] = None,
        l_seat_a:     Optional[float] = None,
        l_seat_b:     Optional[float] = None,
        fillet_r:     Optional[float] = None,
        material_id:  Optional[str]   = None,
        surface_ra:   Optional[float] = None,
        name:         Optional[str]   = None,
    ) -> ShaftFixture:
        """
        Convenience modifier for fixtures produced by make_stepped_3section().

        Reads the current 3-section parameters from self.sections, applies
        any overrides, and returns a new ShaftFixture via make_stepped_3section.

        Raises
        ------
        ValueError
            If self does not have exactly 3 sections (i.e. was not produced
            by make_stepped_3section). For N-section fixtures use with_sections()
            directly.
        """
        if self.n_sections != 3:
            raise ValueError(
                f"with_params() requires exactly 3 sections "
                f"(seat_A | body | seat_B), but this fixture has "
                f"{self.n_sections} sections. Use with_sections() instead."
            )

        s_a, body, s_b = self.sections

        # Recover fillet_r from the body shoulder if present.
        current_fillet_r: float = 0.0
        if body.shoulder_left is not None:
            current_fillet_r = body.shoulder_left.fillet_radius
        elif body.shoulder_right is not None:
            current_fillet_r = body.shoulder_right.fillet_radius

        return make_stepped_3section(
            total_length = total_length if total_length is not None
                           else (s_a.length + body.length + s_b.length),
            d_seat       = d_seat       if d_seat       is not None else s_a.diameter,
            d_body       = d_body       if d_body       is not None else body.diameter,
            l_seat_a     = l_seat_a     if l_seat_a     is not None else s_a.length,
            l_seat_b     = l_seat_b     if l_seat_b     is not None else s_b.length,
            fillet_r     = fillet_r     if fillet_r     is not None else current_fillet_r,
            material_id  = material_id  if material_id  is not None else s_a.material_id,
            surface_ra   = surface_ra   if surface_ra   is not None else s_a.surface_ra,
            name         = name         if name         is not None else self.name,
        )

    # ------------------------------------------------------------------
    # Validation (delegates to core)
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """Return list of geometry error strings from the core Shaft."""
        return self.shaft.validate()

    def validate_or_raise(self) -> None:
        """Raise ValueError if the core Shaft has geometry errors."""
        self.shaft.validate_or_raise()

    # ------------------------------------------------------------------
    # Summary / repr
    # ------------------------------------------------------------------

    def summary(self) -> str:
        tag = self.name or "ShaftFixture"
        lines = [f"-- {tag} (N={self.n_sections}) --",
                 f"  total_length : {self.total_length:.2f} mm"]
        for i, s in enumerate(self.sections):
            shoulder_info = ""
            if s.shoulder_left is not None or s.shoulder_right is not None:
                parts = []
                if s.shoulder_left is not None:
                    parts.append(
                        f"left r={s.shoulder_left.fillet_radius:.2f} mm"
                    )
                if s.shoulder_right is not None:
                    parts.append(
                        f"right r={s.shoulder_right.fillet_radius:.2f} mm"
                    )
                shoulder_info = "  shoulders: " + ", ".join(parts)
            label_str = f" ({s.label})" if s.label else ""
            lines.append(
                f"  [{i}]{label_str} d={s.diameter:.1f} mm  "
                f"L={s.length:.1f} mm  mat={s.material_id}  "
                f"Ra={s.surface_ra} um"
                + (f"\n      {shoulder_info}" if shoulder_info else "")
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ShaftFixture(name={self.name!r}, "
            f"n_sections={self.n_sections}, "
            f"total_length={self.total_length:.1f} mm)"
        )


# ===========================================================================
# Generic factory
# ===========================================================================

def make_shaft(
    sections: list[SectionSpec],
    name:     str = "shaft",
) -> ShaftFixture:
    """
    Generic factory for ShaftFixture with N sections.

    Parameters
    ----------
    sections : list[SectionSpec]
        Ordered section descriptors, left to right. Must contain at least
        one entry. Each SectionSpec produces one ShaftSection.
    name     : str
        Shaft name passed to Shaft.name (default: 'shaft').

    Returns
    -------
    ShaftFixture

    Raises
    ------
    ValueError
        If sections is empty, or if any SectionSpec has non-positive
        length or diameter.

    Notes
    -----
    Shoulder consistency (diameter_large / diameter_small matching adjacent
    section diameters) is validated by Shaft.validate(). The factory does
    not duplicate that check. Call fixture.validate_or_raise() after
    construction if strict upfront checking is required.
    """
    if not sections:
        raise ValueError("sections must contain at least one SectionSpec.")

    shaft = Shaft(label=name)

    for spec in sections:
        if spec.length <= 0.0:
            raise ValueError(
                f"SectionSpec '{spec.label or '?'}': "
                f"length must be > 0, got {spec.length} mm."
            )
        if spec.diameter <= 0.0:
            raise ValueError(
                f"SectionSpec '{spec.label or '?'}': "
                f"diameter must be > 0, got {spec.diameter} mm."
            )

        shaft.add_section(ShaftSection(
            length=spec.length,
            diameter=spec.diameter,
            material_id=spec.material_id,
            surface_finish_ra=spec.surface_ra,
            shoulder_left=spec.shoulder_left,
            shoulder_right=spec.shoulder_right,
            label=spec.label,
        ))

    return ShaftFixture(
        shaft=shaft,
        sections=tuple(sections),
        name=name,
    )


# ===========================================================================
# Convenience factory -- canonical stepped 3-section shaft
# ===========================================================================

def make_stepped_3section(
    total_length: float,
    d_seat:       float,
    d_body:       float,
    l_seat_a:     float,
    l_seat_b:     float,
    fillet_r:     float,
    material_id:  str   = "AISI_1045",
    surface_ra:   float = 0.8,
    name:         str   = "shaft",
) -> ShaftFixture:
    """
    Convenience factory for the canonical seat_A | body | seat_B geometry.

    Derives the SectionSpec list from scalar parameters and delegates to
    make_shaft(). The body section carries Shoulder objects at both faces.

    Parameters
    ----------
    total_length : float  -- total shaft length [mm]
    d_seat       : float  -- bearing seat diameter (both ends) [mm]
    d_body       : float  -- body diameter under gear [mm]; must be > d_seat
    l_seat_a     : float  -- left seat section length [mm]
    l_seat_b     : float  -- right seat section length [mm]
    fillet_r     : float  -- shoulder fillet radius [mm]
    material_id  : str    -- material identifier        (default: 'AISI_1045')
    surface_ra   : float  -- arithmetic mean roughness  (default: 0.8 um)
    name         : str    -- shaft name                 (default: 'shaft')

    Returns
    -------
    ShaftFixture with 3 sections.

    Raises
    ------
    ValueError
        If d_body <= d_seat, if body length is non-positive, or if
        fillet_r exceeds the step height (d_body - d_seat) / 2.

    Notes
    -----
    l_body = total_length - l_seat_a - l_seat_b  (derived, never supplied).
    The same Shoulder object is applied to shoulder_left and shoulder_right
    of the body section. If the two faces require different fillet radii,
    build the SectionSpec list manually and call make_shaft() directly.
    """
    if d_body <= d_seat:
        raise ValueError(
            f"d_body ({d_body} mm) must be greater than d_seat ({d_seat} mm)."
        )

    l_body = total_length - l_seat_a - l_seat_b
    if l_body <= 0.0:
        raise ValueError(
            f"Body section length is non-positive: "
            f"total_length={total_length} mm, "
            f"l_seat_a={l_seat_a} mm, l_seat_b={l_seat_b} mm -> "
            f"l_body={l_body:.4f} mm. "
            "Reduce seat lengths or increase total_length."
        )

    step_height = (d_body - d_seat) / 2.0
    if fillet_r > step_height:
        raise ValueError(
            f"fillet_r ({fillet_r} mm) exceeds the step height "
            f"({step_height:.4f} mm = (d_body - d_seat) / 2). "
            "Reduce fillet_r or increase the diameter step."
        )

    shoulder = Shoulder(
        fillet_radius=fillet_r,
        diameter_large=d_body,
        diameter_small=d_seat,
    )

    sections = [
        SectionSpec(
            length=l_seat_a,
            diameter=d_seat,
            material_id=material_id,
            surface_ra=surface_ra,
            label=f"{name}_seat_A",
        ),
        SectionSpec(
            length=l_body,
            diameter=d_body,
            material_id=material_id,
            surface_ra=surface_ra,
            shoulder_left=shoulder,
            shoulder_right=shoulder,
            label=f"{name}_body",
        ),
        SectionSpec(
            length=l_seat_b,
            diameter=d_seat,
            material_id=material_id,
            surface_ra=surface_ra,
            label=f"{name}_seat_B",
        ),
    ]

    return make_shaft(sections=sections, name=name)


# ===========================================================================
# Named reference instance
# ===========================================================================
# Do not mutate. Use with_params() or with_sections() to produce copies.

SHAFT_50_30_200 = make_stepped_3section(
    total_length=200.0,
    d_seat=30.0,
    d_body=50.0,
    l_seat_a=30.0,
    l_seat_b=30.0,
    fillet_r=2.0,
    material_id="AISI_1045",
    name="SHAFT_50_30_200",
)
"""
Reference stepped shaft: d_body=50 mm, d_seat=30 mm, total_length=200 mm.
Seat sections: 30 mm each. Body: 140 mm. Fillet radius: 2 mm. AISI 1045.
r/d = 2/30 = 0.067. D/d = 50/30 = 1.667.
"""