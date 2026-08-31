"""
fixtures/construction/gears/internal_gear_fixture.py

Factory for internal (ring) gears -- a DIFFERENT core class from external
spur/helical (InternalGear, not SpurHelicalGear); see external_gear_
fixture.py in this same package for spur/helical.

Construction only: this returns a standalone InternalGear object, fully
defined on its own. Nothing about meshing (an internal gear only ever
meshes with an external pinion, and that pairing is what
InternalGear.validate_mesh()/involute_interference()/
trochoid_interference()/trimming_interference() check) lives here -- those
are real methods on the core class already and stay available on the
object this factory returns, but calling them is a meshing-stage decision,
out of scope for Construction.

Dependency (core only):
  axisforge.core.machine_elements.gears.parallel_axis.gear_properties.internal_gear
      InternalGear
  Path confirmed against design_bearing_combination_comparison.py's own
  working import of the sibling spur_helical_gear module (same package,
  same casing) plus the user's direct correction on this one.

References:
  ISO 21771:2007 -- involute gear geometry
  KHK Gear Technical Reference, S4.2 -- Internal Gear dimensions
  Gear Solutions: "Internal ring gears - design and considerations"
"""

from __future__ import annotations

from axisforge.core.machine_elements.gears.parallel_axis.gear_properties.internal_gear import (
    InternalGear,
)


def make_internal_gear(
    mn: float,
    z: int,
    x: float = 0.0,
    b: float = 0.0,
    alpha_n_deg: float = 20.0,
    beta_n_deg: float = 0.0,
    haP: float = 1.0,
    cP: float = 0.25,
    rfP: float = 0.38,
    Ra: float = 0.8,
    Rq: float = 1.0,
    Rz: float = 4.0,
    label: str = "",
    material_id: str = "",
    position: float = 0.0,
) -> InternalGear:
    """
    Internal (ring) gear -- thin factory around the core InternalGear.

    z is required to be NEGATIVE here (z >= 0 raises) -- this is the
    convention actually used for internal gears in AxisForge.
    
    Parameters mirror InternalGear.__init__ exactly (see that class's own
    docstring for units/meaning of each) -- pass z as the negative tooth
    count (e.g. z=-60 for 60 teeth).

    Raises
    ------
    ValueError
        If z >= 0.

    Does NOT call validate() -- the caller decides when to check
    (gear.validate() / gear.validate_or_raise() are both on the returned
    object already). Mesh-pair checks (validate_mesh() and friends) need a
    mating external gear's z/x and are a meshing-stage concern -- also not
    called here.
    """
    if z >= 0:
        raise ValueError(
            f"make_internal_gear expects z < 0 (got {z}) -- internal gears "
            f"in this codebase use a negative tooth count, e.g. z=-60 for "
            f"60 teeth. See this function's own docstring for why z>0 "
            f"actually produces invalid geometry here, not just a "
            f"non-standard one."
        )
    return InternalGear(
        mn=mn, z=z, x=x, b=b,
        alpha_n_deg=alpha_n_deg, beta_n_deg=beta_n_deg,
        haP=haP, cP=cP, rfP=rfP, Ra=Ra, Rq=Rq, Rz=Rz,
        label=label, material_id=material_id, position=position,
    )