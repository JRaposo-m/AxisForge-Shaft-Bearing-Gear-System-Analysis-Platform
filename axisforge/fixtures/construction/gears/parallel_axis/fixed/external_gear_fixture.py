"""
fixtures/construction/gears/external_gear_fixture.py

Factories for external parallel-axis gears -- spur and helical. Both come
from the SAME core class (SpurHelicalGear -- beta_n_deg=0 is a spur gear,
beta_n_deg>0 is helical), but the two factories here PIN which one you get
so the capability string ("gears.spur" vs "gears.helical") is a real
guarantee, not just a default that a caller could silently override:

  make_spur_gear(...)    -- beta_n_deg is NOT a parameter at all. Always 0.0.
  make_helical_gear(...) -- beta_n_deg IS a parameter, but REQUIRED (no
                            default) and validated > 0 -- a "helical" gear
                            built with beta_n_deg=0 is really a spur gear,
                            so that combination is rejected here rather
                            than silently accepted.

Construction only: this returns a standalone SpurHelicalGear object, fully
defined on its own (mn, z, x, b, angles, roughness, label/material/
position). Nothing about meshing (center distance, mesh ratio, contact
ratio, load sharing between two gears) lives here -- that is a later stage,
out of scope for Construction.

Internal (ring) gears are a DIFFERENT core class (InternalGear) -- see
internal_gear_fixture.py in this same package.

Dependency (core only):
  axisforge.core.machine_elements.gears.parallel_axis.gear_properties.spur_helical_gear
      SpurHelicalGear
  Path confirmed against design_bearing_combination_comparison.py's own
  working import of this exact module earlier in this conversation.

References:
  ISO 53:2013    -- standard basic rack tooth profile
  ISO 21771:2007 -- cylindrical involute gear geometry
"""

from __future__ import annotations

from axisforge.core.machine_elements.gears.parallel_axis.gear_properties.spur_helical_gear import (
    SpurHelicalGear,
)


# ===========================================================================
# Spur -- beta_n_deg pinned to 0.0, not exposed
# ===========================================================================

def make_spur_gear(
    mn: float,
    z: int,
    x: float = 0.0,
    b: float = 0.0,
    alpha_n_deg: float = 20.0,
    Ca: float = 0.0,
    Cf: float = 0.0,
    haP: float = 1.0,
    cP: float = 0.25,
    rfP: float = 0.38,
    Ra: float = 0.8,
    Rq: float = 1.0,
    Rz: float = 4.0,
    label: str = "",
    material_id: str = "",
    position: float = 0.0,
) -> SpurHelicalGear:
    """
    Spur gear -- SpurHelicalGear with beta_n_deg pinned to 0.0.

    beta_n_deg is deliberately NOT a parameter here: a caller who wants a
    helix angle should be using make_helical_gear() instead, so "spur" in
    the capability name (gears.spur) stays a real guarantee rather than
    just today's default.

    Parameters mirror SpurHelicalGear.__init__ exactly (see that class's
    own docstring for units/meaning of each) minus beta_n_deg.

    Does NOT call validate() -- the caller decides when to check
    (gear.validate() / gear.validate_or_raise() are both on the returned
    object already).
    """
    return SpurHelicalGear(
        mn=mn, z=z, Ca=Ca, Cf=Cf, x=x, b=b,
        alpha_n_deg=alpha_n_deg, beta_n_deg=0.0,
        haP=haP, cP=cP, rfP=rfP, Ra=Ra, Rq=Rq, Rz=Rz,
        label=label, material_id=material_id, position=position,
    )


# ===========================================================================
# Helical -- beta_n_deg required, validated > 0
# ===========================================================================

def make_helical_gear(
    mn: float,
    z: int,
    beta_n_deg: float,
    x: float = 0.0,
    b: float = 0.0,
    alpha_n_deg: float = 20.0,
    Ca: float = 0.0,
    Cf: float = 0.0,
    haP: float = 1.0,
    cP: float = 0.25,
    rfP: float = 0.38,
    Ra: float = 0.8,
    Rq: float = 1.0,
    Rz: float = 4.0,
    label: str = "",
    material_id: str = "",
    position: float = 0.0,
) -> SpurHelicalGear:
    """
    Helical gear -- SpurHelicalGear with beta_n_deg REQUIRED (no default,
    third positional parameter) and validated > 0.

    A helical gear built with beta_n_deg=0.0 is really a spur gear -- use
    make_spur_gear() for that instead. Rejecting beta_n_deg<=0 here keeps
    "helical" in the capability name (gears.helical) a real guarantee, the
    same way make_spur_gear() pins its own side of the distinction.

    Parameters mirror SpurHelicalGear.__init__ exactly (see that class's
    own docstring for units/meaning of each), with beta_n_deg promoted to
    a required positional argument instead of a defaulted keyword.

    Raises
    ------
    ValueError
        If beta_n_deg <= 0.

    Does NOT call validate() -- the caller decides when to check
    (gear.validate() / gear.validate_or_raise() are both on the returned
    object already; SpurHelicalGear.validate() also flags undercutting,
    which is more common at nonzero helix angles combined with low z).
    """
    if beta_n_deg <= 0.0:
        raise ValueError(
            f"make_helical_gear needs beta_n_deg > 0 (got {beta_n_deg}) -- "
            f"use make_spur_gear() for a 0 helix angle."
        )
    return SpurHelicalGear(
        mn=mn, z=z, Ca=Ca, Cf=Cf, x=x, b=b,
        alpha_n_deg=alpha_n_deg, beta_n_deg=beta_n_deg,
        haP=haP, cP=cP, rfP=rfP, Ra=Ra, Rq=Rq, Rz=Rz,
        label=label, material_id=material_id, position=position,
    )