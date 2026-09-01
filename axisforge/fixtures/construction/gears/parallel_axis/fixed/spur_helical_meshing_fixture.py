"""
fixtures/construction/gears/parallel_axis/fixed/spur_helical_meshing_fixture.py

Factory for a fixed-axis spur/helical gear PAIR -- pairs two already-built
external SpurHelicalGear objects (from make_spur_gear()/make_helical_gear()
in external_gear_fixture.py, this same package) into a
SpurHelicalGearMeshing.

Construction only: this builds the working geometry (working centre
distance, working pressure angle, contact ratios epsilon_alpha/beta/gamma,
addendum-reduction k) and exposes SpurHelicalGearMeshing.validate() (undercut/
interference/contact-ratio checks) and .summary(). It does NOT call
.forces() -- computing Ft/Fr/Fa from an actual driving torque is a MeshLoads-
stage decision (needs a real torque value from the load case), out of scope
here, same boundary as gear_fixture.py not calling gear.validate() for you
and bearing fixtures not dispatching analyses beyond assemble().

Does not build gear1/gear2 itself -- takes them as arguments. Compose with
external_gear_fixture.py's make_spur_gear()/make_helical_gear() at the call
site; this factory's only job is the pairing + a type guard (SpurHelicalGearMeshing's
own __init__ does isinstance-free duck-typing via attribute access, so
passing e.g. an InternalGear here would fail deep inside gear_geometry()
with a confusing AttributeError/wrong-sign result instead of a clear
message -- caught here instead).

Dependency (core only):
  axisforge.core.machine_elements.gears.parallel_axis.gear_properties.spur_helical_gear
      SpurHelicalGear
  axisforge.core.machine_elements.gears.parallel_axis.gear_meshing.spurhelical_meshing
      SpurHelicalGearMeshing

References:
  ISO 21771:2007  -- cylindrical involute gear geometry
  KHK Technical Reference S4.1-4.2 -- gear pair calculations & interference
  ISO 6336-1:2019 -- load on gear teeth (force definitions; not used here)
  MAAG Gear Book  -- working geometry, contact ratio, Ohlendorf loss factor
"""

from __future__ import annotations

from axisforge.core.machine_elements.gears.parallel_axis.gear_properties.spur_helical_gear import (
    SpurHelicalGear,
)
from axisforge.core.machine_elements.gears.parallel_axis.gear_meshing.spurhelical_meshing import (
    SpurHelicalGearMeshing,
)


def make_spur_helical_meshing(
    gear1: SpurHelicalGear,
    gear2: SpurHelicalGear,
    *,
    label: str = "",
    al: float | None = None,
    equalise_gs: bool = False,
    addendum_reduction: bool = False,
    driver: str = "gear1",
) -> SpurHelicalGearMeshing:
    """
    Pair two external gears (both spur, both helical, or one of each --
    SpurHelicalGearMeshing._check_compatibility() enforces matching mn,
    alpha_n_deg and beta_n_deg, so a spur/helical mismatch is rejected
    there, not here).

    gear1, gear2 : SpurHelicalGear
        Already built, e.g. via make_spur_gear()/make_helical_gear() in
        external_gear_fixture.py. NOT built here.
    label        : str
    al           : float | None
        Working centre distance [mm]. None (default) derives it from
        gear1.x + gear2.x. Passing a value imposes it directly and
        back-calculates the working pressure angle; validate() then flags
        an inconsistency if the imposed al doesn't match the declared
        profile shifts.
    equalise_gs  : bool
        If True (and al is None), compute x1/x2 via the Henriot method to
        equalise specific sliding speeds before deriving al -- ignored
        when al is given directly.
    addendum_reduction : bool
        MAAG/KHK addendum-reduction coefficient k -- off by default.
    driver       : "gear1" | "gear2" -- which gear delivers input torque.
        Independent of gear1/gear2 ordering; only matters for forces()
        later (MeshLoads stage), kept here only because it's an __init__
        arg on the core class.

    Does NOT call validate() -- the caller decides when to check
    (SpurHelicalGearMeshing.validate()/.validate_or_raise() are already on
    the returned object).
    """
    if not isinstance(gear1, SpurHelicalGear):
        raise TypeError(
            f"make_spur_helical_meshing: gear1 must be a SpurHelicalGear, "
            f"got {type(gear1).__name__}. For an external+internal pair use "
            f"make_internal_meshing() (internal_meshing_fixture.py) instead."
        )
    if not isinstance(gear2, SpurHelicalGear):
        raise TypeError(
            f"make_spur_helical_meshing: gear2 must be a SpurHelicalGear, "
            f"got {type(gear2).__name__}. For an external+internal pair use "
            f"make_internal_meshing() (internal_meshing_fixture.py) instead."
        )
    return SpurHelicalGearMeshing(
        gear1, gear2, label=label, al=al, equalise_gs=equalise_gs,
        addendum_reduction=addendum_reduction, driver=driver,
    )