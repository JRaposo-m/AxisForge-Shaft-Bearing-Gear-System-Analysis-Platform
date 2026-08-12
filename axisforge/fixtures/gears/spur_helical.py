"""
fixtures/gears/spur_helical.py

Gear fixtures for parallel-axis spur and helical gears.

Two independent classes:
  GearFixture     -- single gear, pre-pairing. Wraps SpurHelicalGear with
                     provisional x (may be 0 or prescribed). Immutable after
                     construction; use with_x() / with_position() to derive
                     updated copies.

  GearPairFixture -- meshed pair with full working geometry. Constructed from
                     two GearFixture objects via from_fixtures(). Holds the
                     SpurHelicalGearMeshing instance that carries the real
                     x1, x2, al, alphatw, epsilon values.

Neither class inherits from the other. GearPairFixture is the natural
extension point for future analysis fixtures (ISO 6336, lubrication).

Named reference instances (GearFixture, x=0, AISI_1045, Ra=0.8 um):
  SPUR_20T_MN2  -- z=20, mn=2, b=20 mm
  SPUR_40T_MN2  -- z=40, mn=2, b=20 mm
  SPUR_60T_MN2  -- z=60, mn=2, b=20 mm

Warning: do not mutate the named instances. Use with_x() / with_position()
to produce positioned or shifted copies for assembly.

Dependencies (core only -- no other fixtures):
  axisforge.core.machine_elements.Gears.Parallel_Axis_gears.spur_helical_gear
      SpurHelicalGear
  axisforge.core.mechanical_system.Parallel_Axis_systems.gear_meshing
      .spur_helical_gear_meshing.SpurHelicalGearMeshing

References:
  ISO 53:2013    -- standard basic rack tooth profile
  ISO 21771:2007 -- cylindrical involute gear geometry
  ISO 6336-1:2019 -- load on gear teeth (future: capacity methods)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from axisforge.core.machine_elements.Gears.Parallel_Axis_gears.spur_helical_gear import (
    SpurHelicalGear,
)
from axisforge.core.mechanical_system.Parallel_Axis_systems.gear_meshing.spur_helical_gear_meshing import (
    SpurHelicalGearMeshing,
)


# ===========================================================================
# GearFixture -- single gear, pre-pairing
# ===========================================================================

@dataclass(frozen=True)
class GearFixture:
    """
    Immutable fixture for a single spur or helical gear.

    Wraps SpurHelicalGear and stores the construction parameters so that
    derived copies (with_x, with_position) can be produced without
    modifying the original.

    The gear attribute is the SpurHelicalGear core object. All geometric
    quantities (d, db, da, etc.) are accessible via self.gear.

    Note
    ----
    x here is provisional -- it may be 0 or a designer-prescribed value.
    The real working x after profile-shift optimisation lives in
    GearPairFixture.meshing.x1 / .x2. Do not treat self.x as the final
    value until the gear has been paired.
    """

    gear: SpurHelicalGear

    # Construction parameters stored explicitly so with_x / with_position
    # can reconstruct without parsing the SpurHelicalGear internals.
    z:           int
    mn:          float
    b:           float
    position:    float
    label:       str
    alpha_n_deg: float
    beta_n_deg:  float
    x:           float
    material_id: str
    Ra:          float
    haP:         float
    cP:          float
    rfP:         float

    # ------------------------------------------------------------------
    # Derived copies
    # ------------------------------------------------------------------

    def with_x(self, x: float) -> GearFixture:
        """Return a new GearFixture identical to this one but with x updated."""
        return make_spur_helical(
            z=self.z,
            mn=self.mn,
            b=self.b,
            position=self.position,
            label=self.label,
            alpha_n_deg=self.alpha_n_deg,
            beta_n_deg=self.beta_n_deg,
            x=x,
            material_id=self.material_id,
            Ra=self.Ra,
            haP=self.haP,
            cP=self.cP,
            rfP=self.rfP,
        )

    def with_position(self, position: float) -> GearFixture:
        """Return a new GearFixture identical to this one but with position updated."""
        return make_spur_helical(
            z=self.z,
            mn=self.mn,
            b=self.b,
            position=position,
            label=self.label,
            alpha_n_deg=self.alpha_n_deg,
            beta_n_deg=self.beta_n_deg,
            x=self.x,
            material_id=self.material_id,
            Ra=self.Ra,
            haP=self.haP,
            cP=self.cP,
            rfP=self.rfP,
        )

    # ------------------------------------------------------------------
    # Validation (delegates to core)
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """Return list of geometry error strings from the core gear."""
        return self.gear.validate()

    def validate_or_raise(self) -> None:
        """Raise ValueError if the core gear has geometry errors."""
        self.gear.validate_or_raise()

    # ------------------------------------------------------------------
    # Summary / repr
    # ------------------------------------------------------------------

    def summary(self) -> str:
        tag = self.label or "GearFixture"
        lines = [
            f"-- {tag} --",
            f"  z           : {self.z}",
            f"  mn          : {self.mn} mm",
            f"  b           : {self.b} mm",
            f"  alpha_n     : {self.alpha_n_deg} deg",
            f"  beta_n      : {self.beta_n_deg} deg",
            f"  x           : {self.x}  (provisional)",
            f"  position    : {self.position} mm",
            f"  material_id : {self.material_id}",
            f"  Ra          : {self.Ra} um",
            f"  d (ref)     : {self.gear.d:.4f} mm",
            f"  da          : {self.gear.da:.4f} mm",
            f"  db          : {self.gear.db:.4f} mm",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"GearFixture(label={self.label!r}, z={self.z}, mn={self.mn}, "
            f"b={self.b}, beta={self.beta_n_deg} deg, x={self.x}, "
            f"pos={self.position} mm)"
        )


# ===========================================================================
# GearPairFixture -- meshed pair with working geometry
# ===========================================================================

@dataclass
class GearPairFixture:
    """
    Fixture for a meshed spur or helical gear pair.

    Encapsulates the SpurHelicalGearMeshing instance that holds the real
    working geometry: x1, x2 (after Henriot or al-imposed correction),
    al, alphatw, epsilon_alpha, epsilon_beta, epsilon_gamma.

    GearPairFixture is the extension point for future analysis fixtures:
      - ISO 6336-2 contact stress capacity
      - ISO 6336-3 bending stress capacity
      - ISO/TR 14179 thermal rating
      - Lubrication assessment (lambda ratio, film thickness)

    These are declared as stubs below and will be implemented in the
    corresponding output fixtures when the relevant solvers exist.

    Construction
    ------------
    Use GearPairFixture.from_fixtures(driver, driven, ...) -- do not
    instantiate directly. from_fixtures builds the SpurHelicalGearMeshing
    and resolves the working geometry.

    Attributes
    ----------
    driver   : GearFixture   -- driver gear fixture (original, x provisional)
    driven   : GearFixture   -- driven gear fixture (original, x provisional)
    meshing  : SpurHelicalGearMeshing
               Working geometry. x1, x2, al, alphatw, epsilon values are
               here, not in driver / driven.
    label    : str
    """

    driver:  GearFixture
    driven:  GearFixture
    meshing: SpurHelicalGearMeshing
    label:   str = ""

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_fixtures(
        cls,
        driver: GearFixture,
        driven: GearFixture,
        label: str = "",
        al: Optional[float] = None,
        equalise_gs: bool = False,
        addendum_reduction: bool = False,
        gear1_is_driver: bool = True,
    ) -> GearPairFixture:
        """
        Build a GearPairFixture from two GearFixture objects.

        Parameters
        ----------
        driver, driven      : GearFixture
        label               : str
        al                  : float | None
            Working centre distance [mm].
            None  -> derived from x1+x2 via involute function.
            float -> imposed; alphatw back-calculated; x consistency checked.
        equalise_gs         : bool
            If True, Henriot method computes x1, x2 before deriving al.
            Ignored when al is explicitly provided.
        addendum_reduction  : bool
            If True, applies MAAG addendum reduction coefficient k.
        gear1_is_driver     : bool
            Passed to SpurHelicalGearMeshing as driver="gear1" (True) or
            driver="gear2" (False).

        Returns
        -------
        GearPairFixture with meshing fully resolved.
        """
        meshing_driver_arg = "gear1" if gear1_is_driver else "gear2"

        meshing = SpurHelicalGearMeshing(
            gear1=driver.gear,
            gear2=driven.gear,
            label=label,
            al=al,
            equalise_gs=equalise_gs,
            addendum_reduction=addendum_reduction,
            driver=meshing_driver_arg,
        )

        return cls(
            driver=driver,
            driven=driven,
            meshing=meshing,
            label=label,
        )

    # ------------------------------------------------------------------
    # Convenience accessors for working geometry
    # ------------------------------------------------------------------

    @property
    def al(self) -> float:
        """Working centre distance [mm]."""
        return self.meshing.al

    @property
    def u(self) -> float:
        """Gear ratio z_driven / z_driver (always >= 1)."""
        return self.meshing.u

    @property
    def epsilon_alpha(self) -> float:
        """Transverse contact ratio [-]."""
        return self.meshing.epslon_alpha

    @property
    def epsilon_beta(self) -> float:
        """Overlap contact ratio [-]."""
        return self.meshing.epslon_beta

    @property
    def epsilon_gamma(self) -> float:
        """Total contact ratio [-]."""
        return self.meshing.epslon_gamma

    @property
    def x1(self) -> float:
        """Profile shift coefficient -- driver (working value)."""
        return self.meshing.x1

    @property
    def x2(self) -> float:
        """Profile shift coefficient -- driven (working value)."""
        return self.meshing.x2

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """Return list of geometry error strings from the meshing object."""
        return self.meshing.validate()

    def validate_or_raise(self) -> None:
        """Raise ValueError if the meshing has geometry errors."""
        self.meshing.validate_or_raise()

    # ------------------------------------------------------------------
    # Summary / repr
    # ------------------------------------------------------------------

    def summary(self) -> str:
        tag = self.label or "GearPairFixture"
        lines = [
            f"-- {tag} --",
            f"  driver : z={self.driver.z}, mn={self.driver.mn}, "
            f"b={self.driver.b} mm, beta={self.driver.beta_n_deg} deg",
            f"  driven : z={self.driven.z}, mn={self.driven.mn}, "
            f"b={self.driven.b} mm, beta={self.driven.beta_n_deg} deg",
            f"  u            : {self.u:.4f}",
            f"  a (ref)      : {self.meshing.a:.4f} mm",
            f"  al (working) : {self.al:.4f} mm",
            f"  x1 (working) : {self.x1:.4f}",
            f"  x2 (working) : {self.x2:.4f}",
            f"  epsilon_a    : {self.epsilon_alpha:.4f}",
            f"  epsilon_b    : {self.epsilon_beta:.4f}",
            f"  epsilon_g    : {self.epsilon_gamma:.4f}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"GearPairFixture(label={self.label!r}, "
            f"z1={self.driver.z}, z2={self.driven.z}, "
            f"mn={self.driver.mn}, al={self.al:.4f} mm, "
            f"eg={self.epsilon_gamma:.4f})"
        )

    # ------------------------------------------------------------------
    # Future analysis stubs -- ISO 6336, lubrication
    # ------------------------------------------------------------------

    def iso6336_contact_stress(self, *args, **kwargs):
        """
        ISO 6336-2 contact stress capacity.

        Not yet implemented. Will accept operating conditions (T, n, KA, KV,
        KHbeta, KHalpha) and return a structured result with sigma_H,
        sigma_HP, and safety factor SH.
        """
        raise NotImplementedError(
            "iso6336_contact_stress is not yet implemented. "
            "Planned for Phase 2 -- ISO 6336-2 contact stress solver."
        )

    def iso6336_bending_stress(self, *args, **kwargs):
        """
        ISO 6336-3 tooth root bending stress capacity.

        Not yet implemented. Will accept operating conditions and return
        sigma_F, sigma_FP, and safety factor SF.
        """
        raise NotImplementedError(
            "iso6336_bending_stress is not yet implemented. "
            "Planned for Phase 2 -- ISO 6336-3 bending stress solver."
        )

    def lubrication_assessment(self, *args, **kwargs):
        """
        Lubrication film thickness and lambda ratio assessment.

        Not yet implemented. Planned for Phase 3.
        """
        raise NotImplementedError(
            "lubrication_assessment is not yet implemented. "
            "Planned for Phase 3 -- lubrication solver."
        )


# ===========================================================================
# Factory function
# ===========================================================================

def make_spur_helical(
    z:           int,
    mn:          float,
    b:           float,
    position:    float,
    label:       str,
    alpha_n_deg: float = 20.0,
    beta_n_deg:  float = 0.0,
    x:           float = 0.0,
    material_id: str   = "AISI_1045",
    Ra:          float = 0.8,
    haP:         float = 1.0,
    cP:          float = 0.25,
    rfP:         float = 0.38,
) -> GearFixture:
    """
    Factory for GearFixture. Constructs the SpurHelicalGear core object and
    wraps it in a GearFixture with all parameters stored for derived copies.

    Parameters
    ----------
    z           : int    -- number of teeth
    mn          : float  -- normal module [mm]
    b           : float  -- face width [mm]
    position    : float  -- axial position on shaft [mm]
    label       : str    -- identifier
    alpha_n_deg : float  -- normal pressure angle [deg]  (default: 20.0)
    beta_n_deg  : float  -- helix angle [deg]            (default: 0.0 -> spur)
    x           : float  -- profile shift coefficient    (default: 0.0)
    material_id : str    -- shaft/gear material ID       (default: 'AISI_1045')
    Ra          : float  -- arithmetic mean roughness [um] (default: 0.8)
    haP         : float  -- addendum coefficient         (default: 1.0, ISO 53)
    cP          : float  -- tip clearance coefficient    (default: 0.25, ISO 53)
    rfP         : float  -- fillet radius coefficient    (default: 0.38, ISO 53)

    Returns
    -------
    GearFixture

    Notes
    -----
    Ca, Cf, Rq, Rz are not exposed here. They remain at SpurHelicalGear
    defaults. Callers requiring non-standard values should instantiate
    SpurHelicalGear directly and wrap it in GearFixture manually.

    The x supplied here is provisional. The real working profile shift is
    resolved by GearPairFixture.from_fixtures() via SpurHelicalGearMeshing.
    """
    gear = SpurHelicalGear(
        mn=mn,
        z=z,
        x=x,
        b=b,
        alpha_n_deg=alpha_n_deg,
        beta_n_deg=beta_n_deg,
        haP=haP,
        cP=cP,
        rfP=rfP,
        Ra=Ra,
        label=label,
        material_id=material_id,
        position=position,
    )

    return GearFixture(
        gear=gear,
        z=z,
        mn=mn,
        b=b,
        position=position,
        label=label,
        alpha_n_deg=alpha_n_deg,
        beta_n_deg=beta_n_deg,
        x=x,
        material_id=material_id,
        Ra=Ra,
        haP=haP,
        cP=cP,
        rfP=rfP,
    )


# ===========================================================================
# Named reference instances
# ===========================================================================
# These are GearFixture objects with typical defaults for quick scripting
# and unit testing. position=0.0 because position is assembly-specific.
# Do not mutate these objects. Use with_position() or with_x() to produce
# assembly-ready copies, or call make_spur_helical() for full control.

SPUR_20T_MN2 = make_spur_helical(
    z=20, mn=2, b=20.0, position=0.0, label="SPUR_20T_MN2",
)
"""Reference spur gear: z=20, mn=2, b=20 mm. Typical pinion, first stage."""

SPUR_40T_MN2 = make_spur_helical(
    z=40, mn=2, b=20.0, position=0.0, label="SPUR_40T_MN2",
)
"""Reference spur gear: z=40, mn=2, b=20 mm. Wheel or intermediate pinion."""

SPUR_60T_MN2 = make_spur_helical(
    z=60, mn=2, b=20.0, position=0.0, label="SPUR_60T_MN2",
)
"""Reference spur gear: z=60, mn=2, b=20 mm. Typical wheel, second stage."""
