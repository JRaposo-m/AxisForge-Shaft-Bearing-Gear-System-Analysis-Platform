"""
fixtures/construction/shafts/shaft_fixture.py

Generic shaft fixture. Always N sections -- there is no hardcoded
2-section or 3-section case here; N is just len(sections) at the call
site.

Classes
-------
  SectionSpec    -- declarative descriptor for one shaft section.
                    Carries length, diameter, material, surface finish
                    and label. Does NOT carry shoulder information --
                    shoulders belong to the Shaft now (see below).

  ShaftFixture   -- immutable wrapper around a Shaft, storing the ordered
                    list of SectionSpec objects for derived copies.

Factories
---------
  make_shaft(sections, transitions, name)   -> ShaftFixture
      Fully manual: transitions is an optional dict[int, Shoulder],
      keyed by boundary index (i = boundary between sections[i] and
      sections[i+1]), attached to the assembled Shaft via
      Shaft.set_transition(). Omit for a shaft with no shoulders.

  make_stepped_shaft(sections, fillet_radii, name)  -> ShaftFixture
      One call for the common case: a Shoulder is derived for every
      one of the N-1 internal transitions from the two adjacent
      section diameters plus fillet_radii[i], and attached to the
      Shaft as a transition. The caller never repeats a diameter
      that's already on a SectionSpec, and never decides which "side"
      owns a shoulder -- that whole notion is gone (see the core
      shaft.py module docstring: Shoulder ownership moved off
      ShaftSection onto Shaft.transitions).

Named reference instance
------------------------
  SHAFT_50_30_200  -- seat_A(d=30, L=30) | body(d=50, L=140) | seat_B(d=30, L=30)
                      total_length=200 mm, fillet_r=2.0 mm, AISI_1045
                      (built via make_stepped_shaft, 2 fillet radii)

Warning: do not mutate named instances. Use with_sections() or
with_name() to produce modified copies.

Dependencies (core only):
  axisforge.core.machine_elements.shaft.shaft  Shaft, ShaftSection, Shoulder

References:
  Shigley SS7-1  -- shaft geometry conventions
  Peterson SS1   -- stress raisers at geometric discontinuities

CHANGED vs. the previous version of this file
----------------------------------------------
SectionSpec used to carry shoulder_left / shoulder_right, and
make_stepped_shaft() had to decide which single side of a transition
legitimately "owned" the derived Shoulder -- an artifact of the old
core convention where Shaft.validate() read shoulder_left and
shoulder_right asymmetrically. That convention is gone: Shoulder now
belongs to the Shaft itself (Shaft.transitions: dict[int, Shoulder]),
so a transition either has a Shoulder or it doesn't -- no ownership
side to get backwards. This file is the fixture-side half of that
redesign; the core half is in shaft.py.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

from axisforge.core.machine_elements.shaft.shaft import Shaft, ShaftSection, Shoulder


# ===========================================================================
# SectionSpec -- declarative section descriptor
# ===========================================================================

@dataclass(frozen=True)
class SectionSpec:
    """
    Declarative descriptor for one shaft section.

    Used as input to make_shaft() / make_stepped_shaft(). Each
    SectionSpec produces exactly one ShaftSection in the assembled
    Shaft.

    Attributes
    ----------
    length         : float          -- axial length [mm]
    diameter       : float          -- outer diameter [mm]
    material_id    : str            -- material identifier (core/materials.py)
    surface_ra     : float          -- arithmetic mean roughness Ra [um]
    label          : str            -- section label (used in reporting)

    Notes
    -----
    No shoulder fields here -- a Shoulder describes a TRANSITION
    (between two sections), not a single section, and now lives on the
    Shaft (Shaft.transitions), attached via make_shaft()'s
    `transitions` parameter or derived automatically by
    make_stepped_shaft().

    Inner diameter (hollow sections) and keyways are not exposed here
    either. Construct ShaftSection directly and wrap via make_shaft()
    if either is required.
    """

    length:      float
    diameter:    float
    material_id: str   = "AISI_1045"
    surface_ra:  float = 0.8
    label:       str   = ""


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
    shaft    : Shaft            -- core shaft object (transitions live
                                    on shaft.transitions)
    sections : list[SectionSpec]-- ordered section descriptors (left to right)
    name     : str              -- shaft name (matches Shaft.label)

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

    def with_sections(
        self,
        sections: list[SectionSpec],
        transitions: Optional[dict[int, Shoulder]] = None,
    ) -> ShaftFixture:
        """
        Return a new ShaftFixture built from a different list of SectionSpec.

        The name is inherited. Any N -- there is nothing special about
        the current n_sections.

        transitions is NOT carried over from self automatically: a
        different section count or different diameters can invalidate
        old boundary indices or diameter pairs, so pass the new
        transitions dict explicitly (or omit it for a shaft with no
        shoulders).
        """
        return make_shaft(sections=sections, transitions=transitions, name=self.name)

    def with_name(self, name: str) -> ShaftFixture:
        """Return a new ShaftFixture with a different shaft name.
        Sections and transitions are carried over unchanged -- only the
        label changes."""
        return make_shaft(
            sections=list(self.sections),
            transitions=dict(self.shaft.transitions),
            name=name,
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
        """
        Quick geometric overview -- axial position and diameter per
        section, followed by a separate block listing every transition
        that carries a Shoulder (fillet radius, diameter range, r/d,
        D/d). Deliberately stops there: engineering properties (area,
        I, J, W, Wt) and validate() status are NOT included here --
        that belongs to the check/report scripts that consume a
        ShaftFixture, not to the fixture's own summary.

        Reads shoulders from self.shaft.transitions -- SectionSpec
        carries no shoulder fields any more, so there is nothing to
        read off individual sections.
        """
        n_shoulders = len(self.shaft.transitions)
        tag = self.name or "ShaftFixture"
        lines = [
            f"-- {tag} (N={self.n_sections} sections, "
            f"{n_shoulders} shoulders) --",
            f"  total_length : {self.total_length:.2f} mm",
        ]

        z = 0.0
        for i, s in enumerate(self.sections):
            z_start, z_end = z, z + s.length
            z = z_end
            label_str = f" ({s.label})" if s.label else ""
            lines.append(
                f"  [{i}]{label_str} z=[{z_start:.1f}, {z_end:.1f}] mm  "
                f"d={s.diameter:.1f} mm  L={s.length:.1f} mm  "
                f"mat={s.material_id}  Ra={s.surface_ra} um"
            )

        if self.shaft.transitions:
            lines.append("  shoulders:")
            for i in sorted(self.shaft.transitions):
                sh = self.shaft.transitions[i]
                z_pos = self.shaft.axial_end(i)
                left_label = self.sections[i].label or f"[{i}]"
                right_label = self.sections[i + 1].label or f"[{i + 1}]"
                lines.append(
                    f"    boundary {i}/{i + 1} ({left_label} -> {right_label})  "
                    f"z={z_pos:.1f} mm  r={sh.fillet_radius:.2f} mm  "
                    f"d {sh.diameter_small:.1f}->{sh.diameter_large:.1f}  "
                    f"r/d={sh.r_over_d:.4f}  D/d={sh.D_over_d:.4f}"
                )

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ShaftFixture(name={self.name!r}, "
            f"n_sections={self.n_sections}, "
            f"total_length={self.total_length:.1f} mm)"
        )


# ===========================================================================
# Manual factory -- caller supplies transitions (if any) directly
# ===========================================================================

def make_shaft(
    sections:    list[SectionSpec],
    transitions: Optional[dict[int, Shoulder]] = None,
    name:        str = "shaft",
) -> ShaftFixture:
    """
    Manual factory for ShaftFixture with N sections.

    Parameters
    ----------
    sections : list[SectionSpec]
        Ordered section descriptors, left to right. Must contain at least
        one entry.
    transitions : dict[int, Shoulder] | None
        Optional Shoulder objects to attach, keyed by boundary index
        (i = boundary between sections[i] and sections[i+1]). Attached
        to the assembled Shaft via Shaft.set_transition() after every
        section has been added. Omit (or pass {}) for a shaft with no
        shoulders.
    name     : str
        Shaft name passed to Shaft.label (default: 'shaft').

    Returns
    -------
    ShaftFixture

    Raises
    ------
    ValueError
        If sections is empty, or if any SectionSpec has non-positive
        length or diameter.
    IndexError
        If a transitions key is not a valid boundary index for the
        given sections (propagated from Shaft.set_transition()).

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
            label=spec.label,
        ))

    for index, shoulder in (transitions or {}).items():
        shaft.set_transition(index, shoulder)

    return ShaftFixture(
        shaft=shaft,
        sections=tuple(sections),
        name=name,
    )


# ===========================================================================
# Stepped factory -- one call, shoulders derived from adjacent diameters
# ===========================================================================

def make_stepped_shaft(
    sections:     list[SectionSpec],
    fillet_radii: list[float],
    name:         str = "shaft",
) -> ShaftFixture:
    """
    One-call factory for an N-section stepped shaft where every internal
    transition is described by ONLY a fillet radius -- diameter_large
    and diameter_small are derived from the two adjacent SectionSpec.diameter
    values, never re-typed by the caller.

    Parameters
    ----------
    sections     : list[SectionSpec]
        N sections, left to right.
    fillet_radii : list[float]
        Exactly N-1 entries [mm], one per internal transition:
        fillet_radii[i] is the fillet between sections[i] and
        sections[i+1]. Ignored (no Shoulder built) for a transition
        where the two adjacent diameters are equal -- no step, nothing
        to fillet, and the core Shoulder requires diameter_large >
        diameter_small.
    name         : str
        Shaft name passed to Shaft.label (default: 'shaft').

    Returns
    -------
    ShaftFixture with N sections. Each differing-diameter transition
    contributes exactly one Shoulder, attached to the Shaft via
    Shaft.transitions -- there is no "which side owns it" decision any
    more (see the module docstring).

    Raises
    ------
    ValueError
        If fewer than 2 sections are given, or if len(fillet_radii) !=
        len(sections) - 1.
    """
    if len(sections) < 2:
        raise ValueError(
            "make_stepped_shaft needs at least 2 sections to have a "
            "transition -- use make_shaft() directly for a single section."
        )
    if len(fillet_radii) != len(sections) - 1:
        raise ValueError(
            f"fillet_radii must have exactly {len(sections) - 1} entries "
            f"for {len(sections)} sections (one per internal transition), "
            f"got {len(fillet_radii)}."
        )

    transitions: dict[int, Shoulder] = {}
    for i in range(len(sections) - 1):
        d_left = sections[i].diameter
        d_right = sections[i + 1].diameter
        if d_left == d_right:
            continue  # no step -- nothing to fillet, no Shoulder possible
        transitions[i] = Shoulder(
            fillet_radius=fillet_radii[i],
            diameter_large=max(d_left, d_right),
            diameter_small=min(d_left, d_right),
        )

    return make_shaft(sections=sections, transitions=transitions, name=name)


# ===========================================================================
# Named reference instance
# ===========================================================================
# Do not mutate. Use with_sections() or with_name() to produce copies.

SHAFT_50_30_200 = make_stepped_shaft(
    sections=[
        SectionSpec(length=30.0, diameter=30.0, label="SHAFT_50_30_200_seat_A"),
        SectionSpec(length=140.0, diameter=50.0, label="SHAFT_50_30_200_body"),
        SectionSpec(length=30.0, diameter=30.0, label="SHAFT_50_30_200_seat_B"),
    ],
    fillet_radii=[2.0, 2.0],
    name="SHAFT_50_30_200",
)
"""
Reference stepped shaft: d_body=50 mm, d_seat=30 mm, total_length=200 mm.
Seat sections: 30 mm each. Body: 140 mm. Fillet radius: 2 mm. AISI 1045.
r/d = 2/30 = 0.067. D/d = 50/30 = 1.667.
"""