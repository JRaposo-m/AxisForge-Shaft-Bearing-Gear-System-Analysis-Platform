"""
fixtures/bearings/dgbb_generic.py

Bearing fixture for Deep Groove Ball Bearings (DGBB).

Two classes:
  BearingFixture   -- immutable wrapper around a Bearing (or subclass),
                      storing catalog and mounting parameters for derived
                      copies. Base class; subclasses extend for specific
                      bearing types.

  DGBBFixture      -- extends BearingFixture for DeepGrooveBallBearing.
                      Stores internal geometry parameters and exposes
                      make_ready() which calls setup_internal_geometry()
                      and compute_hertz_point_contact() in one step,
                      producing a bearing ready for IterativeBearingFEMSolver.

Factory
-------
  make_dgbb(...)   -> DGBBFixture

The factory covers the common DGBB case. Bearings requiring non-standard
X/Y factors or contact angle should be built from DeepGrooveBallBearing
directly and wrapped in DGBBFixture manually.

Named reference instances (DGBBFixture, position=0.0, arrangement='locating'):
  DGBB_6208   -- SKF 6208 : d=40, D=80,  b=18, C=29600 N,  C0=17800 N
  DGBB_6210   -- SKF 6210 : d=50, D=90,  b=20, C=35100 N,  C0=23200 N
  DGBB_6304   -- SKF 6304 : d=20, D=52,  b=15, C=15900 N,  C0=7800  N

Internal geometry is NOT set on the named instances — call make_ready()
on a positioned copy to produce a solver-ready bearing.

Warning: do not mutate the named instances. Use with_position() /
with_arrangement() to produce assembly-ready copies.

Dependencies (core only):
  axisforge.core.machine_elements.Bearings.bearing           Bearing
  axisforge.core.machine_elements.Bearings.bearing_types     BearingType
  axisforge.core.machine_elements.Bearings.subtypes.deep_groove_ball
      DeepGrooveBallBearing

References:
  ISO/TS 16281:2008 -- internal load distribution, rolling bearing life
  ISO 281:2007      -- dynamic load rating, basic life
  SKF General Catalogue 2018 -- catalog data for reference instances
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from axisforge.core.machine_elements.Bearings.bearing import Bearing
from axisforge.core.machine_elements.Bearings.bearing_types import BearingType
from axisforge.core.machine_elements.Bearings.subtypes.deep_groove_ball import (
    DeepGrooveBallBearing,
)


# ===========================================================================
# BearingFixture -- base, catalog + mounting parameters
# ===========================================================================

@dataclass(frozen=True)
class BearingFixture:
    """
    Immutable base fixture for a positioned rolling bearing.

    Wraps a Bearing (or subclass) and stores catalog and mounting parameters
    so that with_position() and with_arrangement() can produce modified
    copies without parsing the core bearing object.

    This class is not instantiated directly in normal use -- use DGBBFixture
    or a future subclass (CRBFixture, ACBFixture).

    Attributes
    ----------
    bearing      : Bearing   -- core bearing object
    d            : float     -- bore diameter [mm]
    D            : float     -- outer diameter [mm]
    b            : float     -- width [mm]
    C            : float     -- dynamic load rating [N]
    C0           : float     -- static load rating [N]
    designation  : str       -- catalog designation (e.g. '6208')
    position     : float     -- axial position on shaft [mm]
    arrangement  : str       -- 'locating' | 'floating' | 'non-locating'
    label        : str       -- identifier for reporting
    """

    bearing:     Bearing
    d:           float
    D:           float
    b:           float
    C:           float
    C0:          float
    designation: str
    position:    float
    arrangement: str
    label:       str

    # ------------------------------------------------------------------
    # Derived copies
    # ------------------------------------------------------------------

    def with_position(self, position: float) -> BearingFixture:
        """Return a new fixture identical to this one but at a new axial position."""
        raise NotImplementedError(
            "with_position must be implemented by a subclass."
        )

    def with_arrangement(self, arrangement: str) -> BearingFixture:
        """Return a new fixture identical to this one but with a different arrangement."""
        raise NotImplementedError(
            "with_arrangement must be implemented by a subclass."
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """Return list of validation error strings from the core bearing."""
        return self.bearing.validate()

    def validate_or_raise(self) -> None:
        """Raise ValueError if the core bearing has validation errors."""
        self.bearing.validate_or_raise()

    # ------------------------------------------------------------------
    # Summary / repr
    # ------------------------------------------------------------------

    def summary(self) -> str:
        tag = self.label or self.designation or "BearingFixture"
        lines = [
            f"-- {tag} --",
            f"  designation : {self.designation}",
            f"  d / D / b   : {self.d:.1f} / {self.D:.1f} / {self.b:.1f} mm",
            f"  C / C0      : {self.C:.0f} / {self.C0:.0f} N",
            f"  position    : {self.position:.2f} mm",
            f"  arrangement : {self.arrangement}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"designation={self.designation!r}, "
            f"position={self.position:.2f} mm, "
            f"arrangement={self.arrangement!r}, "
            f"label={self.label!r})"
        )


# ===========================================================================
# DGBBFixture -- DGBB with internal geometry support
# ===========================================================================

@dataclass(frozen=True)
class DGBBFixture(BearingFixture):
    """
    Immutable fixture for a Deep Groove Ball Bearing.

    Extends BearingFixture with internal geometry parameters required by
    IterativeBearingFEMSolver (ISO/TS 16281). Internal geometry is stored
    here but is NOT applied to the bearing object until make_ready() is
    called -- this preserves immutability and makes the setup step explicit.

    Internal geometry parameters
    ----------------------------
    ri          : float        -- inner groove radius [mm]
    re          : float        -- outer groove radius [mm]
    Dw          : float        -- ball diameter [mm]
    Dpw         : float        -- pitch circle diameter [mm]
    Z           : int          -- number of balls
    E           : float        -- Young's modulus [MPa]
    nu          : float        -- Poisson's ratio
    s           : float | None -- diametral operating clearance [mm]
                                  Exactly one of s or alpha_0_deg must be set.
    alpha_0_deg : float | None -- free contact angle [deg]

    Usage
    -----
    fixture = make_dgbb(
        d=40, D=80, b=18, C=29600, C0=17800, designation="6208",
        ri=6.6, re=6.85, Dw=12.0, Dpw=60.0, Z=9, E=206000,
        s=0.015, position=30.0, arrangement="locating", label="A",
    )
    bearing = fixture.make_ready()
    # bearing.has_internal_geometry() is True
    # bearing.cp is set -- ready for IterativeBearingFEMSolver

    Notes
    -----
    s and alpha_0_deg are mutually exclusive. make_ready() will raise
    ValueError (via setup_internal_geometry) if both or neither are set.

    The bearing attribute on the base BearingFixture stores the
    DeepGrooveBallBearing BEFORE setup -- it is available for inspection
    but NOT ready for the ISO 16281 solver. Always use make_ready() to
    obtain the solver-ready bearing.
    """

    ri:          float
    re:          float
    Dw:          float
    Dpw:         float
    Z:           int
    E:           float
    nu:          float
    s:           Optional[float]
    alpha_0_deg: Optional[float]

    # ------------------------------------------------------------------
    # Derived copies
    # ------------------------------------------------------------------

    def with_position(self, position: float) -> DGBBFixture:
        """Return a new DGBBFixture at a different axial position."""
        return make_dgbb(
            d=self.d, D=self.D, b=self.b,
            C=self.C, C0=self.C0,
            designation=self.designation,
            ri=self.ri, re=self.re, Dw=self.Dw, Dpw=self.Dpw,
            Z=self.Z, E=self.E, nu=self.nu,
            s=self.s, alpha_0_deg=self.alpha_0_deg,
            position=position,
            arrangement=self.arrangement,
            label=self.label,
        )

    def with_arrangement(self, arrangement: str) -> DGBBFixture:
        """Return a new DGBBFixture with a different mounting arrangement."""
        return make_dgbb(
            d=self.d, D=self.D, b=self.b,
            C=self.C, C0=self.C0,
            designation=self.designation,
            ri=self.ri, re=self.re, Dw=self.Dw, Dpw=self.Dpw,
            Z=self.Z, E=self.E, nu=self.nu,
            s=self.s, alpha_0_deg=self.alpha_0_deg,
            position=self.position,
            arrangement=arrangement,
            label=self.label,
        )

    def with_label(self, label: str) -> DGBBFixture:
        """Return a new DGBBFixture with a different label."""
        return make_dgbb(
            d=self.d, D=self.D, b=self.b,
            C=self.C, C0=self.C0,
            designation=self.designation,
            ri=self.ri, re=self.re, Dw=self.Dw, Dpw=self.Dpw,
            Z=self.Z, E=self.E, nu=self.nu,
            s=self.s, alpha_0_deg=self.alpha_0_deg,
            position=self.position,
            arrangement=self.arrangement,
            label=label,
        )

    # ------------------------------------------------------------------
    # Solver-ready bearing
    # ------------------------------------------------------------------

    def make_ready(self) -> DeepGrooveBallBearing:
        """
        Construct and return a solver-ready DeepGrooveBallBearing.

        Builds a fresh DeepGrooveBallBearing from catalog parameters,
        calls setup_internal_geometry(), and calls
        compute_hertz_point_contact(). The returned object has
        has_internal_geometry() == True and cp set.

        The fixture itself is not modified (immutable). Each call returns
        a new independent DeepGrooveBallBearing instance.

        Returns
        -------
        DeepGrooveBallBearing  -- ready for IterativeBearingFEMSolver.

        Raises
        ------
        ValueError
            If internal geometry parameters are inconsistent (propagated
            from setup_internal_geometry).
        RuntimeError
            If internal geometry is not set before computing Hertz constant
            (should not occur if using this method).
        """
        if self.ri is None:
            raise ValueError(
                f"DGBBFixture '{self.label or self.designation}': internal geometry "
                "parameters are not set. Provide ri, re, Dw, Dpw, Z, E and either "
                "s or alpha_0_deg when constructing the fixture."
            )

        bearing = DeepGrooveBallBearing(
            d=self.d,
            D=self.D,
            b=self.b,
            C=self.C,
            C0=self.C0,
            designation=self.designation,
            position=self.position,
            arrangement=self.arrangement,
            label=self.label,
        )

        # Build kwargs for clearance specification -- exactly one of s / alpha_0_deg.
        clearance_kwargs: dict = {}
        if self.s is not None:
            clearance_kwargs["s"] = self.s
        elif self.alpha_0_deg is not None:
            clearance_kwargs["alpha_0_deg"] = self.alpha_0_deg
        else:
            raise ValueError(
                f"DGBBFixture '{self.label or self.designation}': "
                "exactly one of 's' or 'alpha_0_deg' must be provided."
            )

        bearing.setup_internal_geometry(
            ri=self.ri,
            re=self.re,
            Dw=self.Dw,
            Dpw=self.Dpw,
            Z=self.Z,
            E=self.E,
            nu=self.nu,
            **clearance_kwargs,
        )
        bearing.compute_hertz_point_contact()

        return bearing

    # ------------------------------------------------------------------
    # Summary / repr
    # ------------------------------------------------------------------

    def summary(self) -> str:
        tag = self.label or self.designation or "DGBBFixture"
        clearance_str = (
            f"s={self.s:.4f} mm" if self.s is not None
            else f"alpha_0={self.alpha_0_deg:.4f} deg"
        )
        lines = [
            f"-- {tag} (DGBB) --",
            f"  designation  : {self.designation}",
            f"  d / D / b    : {self.d:.1f} / {self.D:.1f} / {self.b:.1f} mm",
            f"  C / C0       : {self.C:.0f} / {self.C0:.0f} N",
            f"  position     : {self.position:.2f} mm",
            f"  arrangement  : {self.arrangement}",
            f"  Dw / Dpw     : {self.Dw:.3f} / {self.Dpw:.3f} mm",
            f"  Z            : {self.Z}",
            f"  ri / re      : {self.ri:.4f} / {self.re:.4f} mm",
            f"  E / nu       : {self.E:.0f} MPa / {self.nu}",
            f"  clearance    : {clearance_str}",
            f"  geometry set : no  (call make_ready() to obtain solver-ready bearing)",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"DGBBFixture(designation={self.designation!r}, "
            f"position={self.position:.2f} mm, "
            f"arrangement={self.arrangement!r}, "
            f"Z={self.Z}, Dw={self.Dw:.3f} mm, label={self.label!r})"
        )


# ===========================================================================
# Factory function
# ===========================================================================

def make_dgbb(
    d:           float,
    D:           float,
    b:           float,
    C:           float,
    C0:          float,
    designation: str,
    ri:          float,
    re:          float,
    Dw:          float,
    Dpw:         float,
    Z:           int,
    E:           float = 206_000.0,
    nu:          float = 0.3,
    s:           Optional[float] = None,
    alpha_0_deg: Optional[float] = None,
    position:    float = 0.0,
    arrangement: str   = "locating",
    label:       str   = "",
) -> DGBBFixture:
    """
    Factory for DGBBFixture.

    Parameters
    ----------
    d, D, b      : bore, outer, width [mm]
    C, C0        : dynamic / static load rating [N]
    designation  : catalog designation (e.g. '6208')
    ri, re       : inner / outer groove radius [mm]
    Dw           : ball diameter [mm]
    Dpw          : pitch circle diameter [mm]
    Z            : number of balls
    E            : Young's modulus [MPa]         (default: 206 000 MPa, steel)
    nu           : Poisson's ratio               (default: 0.3)
    s            : diametral operating clearance [mm]
                   Exactly one of s or alpha_0_deg must be provided.
    alpha_0_deg  : free contact angle [deg]
    position     : axial position on shaft [mm]  (default: 0.0)
    arrangement  : 'locating' | 'floating' | 'non-locating'
                                                 (default: 'locating')
    label        : identifier for reporting      (default: '')

    Returns
    -------
    DGBBFixture

    Raises
    ------
    ValueError
        If both or neither of s / alpha_0_deg are provided.

    Notes
    -----
    make_ready() must be called on the fixture (or a positioned copy) to
    obtain a DeepGrooveBallBearing with internal geometry set, ready for
    IterativeBearingFEMSolver.

    Groove radii convention (ISO/TS 16281):
      ri >= Dw/2  (conformity f_i = ri/Dw, typically 0.52-0.54)
      re >= Dw/2  (conformity f_e = re/Dw, typically 0.52-0.54)
    """
    if (s is None) == (alpha_0_deg is None):
        raise ValueError(
            "Exactly one of 's' or 'alpha_0_deg' must be provided, not both or neither."
        )

    # Build a base Bearing for catalog-level access (no internal geometry).
    bearing = DeepGrooveBallBearing(
        d=d, D=D, b=b, C=C, C0=C0,
        designation=designation,
        position=position,
        arrangement=arrangement,
        label=label,
    )

    return DGBBFixture(
        bearing=bearing,
        d=d, D=D, b=b,
        C=C, C0=C0,
        designation=designation,
        position=position,
        arrangement=arrangement,
        label=label,
        ri=ri, re=re, Dw=Dw, Dpw=Dpw,
        Z=Z, E=E, nu=nu,
        s=s,
        alpha_0_deg=alpha_0_deg,
    )


# ===========================================================================
# Named reference instances
# ===========================================================================
# Catalog data: SKF General Catalogue 2018.
# Internal geometry: approximate representative values for ISO/TS 16281 use.
# These are starting points -- replace with actual catalog / measured values
# for any design validation.
#
# position=0.0, arrangement='locating' -- assembly-specific; use
# with_position() / with_arrangement() to produce assembly-ready copies.
# Internal geometry is stored but NOT applied -- call make_ready() on a
# positioned copy to obtain a solver-ready DeepGrooveBallBearing.
#
# Clearance: s (diametral) in mm -- CN (Normal) group approximation.

DGBB_6208 = make_dgbb(
    d=40.0, D=80.0, b=18.0, C=29_600.0, C0=17_800.0,
    designation="6208",
    ri=6.60, re=6.85, Dw=12.303, Dpw=60.0, Z=9,
    E=206_000.0, nu=0.3, s=0.015,
    position=0.0, arrangement="locating", label="",
)
"""
SKF 6208 DGBB reference fixture.
d=40 mm, D=80 mm, b=18 mm. C=29 600 N, C0=17 800 N.
Z=9 balls, Dw=12.303 mm, Dpw=60.0 mm.
ri=6.60 mm (f_i~0.537), re=6.85 mm (f_e~0.557). s=0.015 mm (CN group).
"""

DGBB_6210 = make_dgbb(
    d=50.0, D=90.0, b=20.0, C=35_100.0, C0=23_200.0,
    designation="6210",
    ri=7.40, re=7.65, Dw=14.288, Dpw=70.0, Z=10,
    E=206_000.0, nu=0.3, s=0.015,
    position=0.0, arrangement="locating", label="",
)
"""
SKF 6210 DGBB reference fixture.
d=50 mm, D=90 mm, b=20 mm. C=35 100 N, C0=23 200 N.
Z=10 balls, Dw=14.288 mm, Dpw=70.0 mm.
ri=7.40 mm (f_i~0.518), re=7.65 mm (f_e~0.535). s=0.015 mm (CN group).
"""

DGBB_6304 = make_dgbb(
    d=20.0, D=52.0, b=15.0, C=15_900.0, C0=7_800.0,
    designation="6304",
    ri=4.15, re=4.35, Dw=7.938, Dpw=36.0, Z=8,
    E=206_000.0, nu=0.3, s=0.010,
    position=0.0, arrangement="locating", label="",
)
"""
SKF 6304 DGBB reference fixture.
d=20 mm, D=52 mm, b=15 mm. C=15 900 N, C0=7 800 N.
Z=8 balls, Dw=7.938 mm, Dpw=36.0 mm.
ri=4.15 mm (f_i~0.523), re=4.35 mm (f_e~0.548). s=0.010 mm (CN group).
"""
