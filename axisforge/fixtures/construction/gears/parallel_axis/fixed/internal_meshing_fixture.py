"""
fixtures/construction/gears/parallel_axis/fixed/internal_meshing_fixture.py

Factory for a fixed-axis external-pinion + internal-ring gear PAIR -- pairs
an already-built external SpurHelicalGear (gear1, pinion) with an
already-built InternalGear (gear2, ring) into an InternalGearMeshing.

z-sign note (confirmed with you, not assumed): InternalGearMeshing's own
module docstring claims "z > 0, conventional KHK definition" and that it
"does NOT negate x2 or z2 internally" -- that docstring is stale.
InternalGear itself requires z < 0 (make_internal_gear() enforces it), and
InternalGearMeshing.gear_geometry() already treats gear2.z as signed
throughout (the z2/abs(z2) switch in da2, epsilon_a2, galpha, T1T2, T2A,
T2C). The one place that looks unguarded, self.u = z2/z1 (negative when
z2<0), is DELIBERATE, not a bug -- confirmed by you: u negative is how
this class encodes that internal meshing does NOT reverse rotation
direction (ring and pinion turn the same way), unlike external meshing
which always does. So: gear2 here is expected with z<0, exactly what
make_internal_gear() already returns -- nothing special done in this
factory to compensate, and none should be.

Construction only: builds working geometry and exposes .validate() (which
delegates interference checks to InternalGear.validate_mesh() internally)
and .summary(). Does NOT call .forces() -- MeshLoads stage, out of scope
here (see spur_helical_meshing_fixture.py's own docstring for the same
boundary on the external-pair side).

Does not build gear1/gear2 itself -- compose with external_gear_fixture.py's
make_spur_gear()/make_helical_gear() (gear1) and internal_gear_fixture.py's
make_internal_gear() (gear2) at the call site.

Dependency (core only):
  axisforge.core.machine_elements.gears.parallel_axis.gear_properties.spur_helical_gear
      SpurHelicalGear
  axisforge.core.machine_elements.gears.parallel_axis.gear_properties.internal_gear
      InternalGear
  axisforge.core.machine_elements.gears.parallel_axis.gear_meshing.internal_meshing
      InternalGearMeshing

References:
  ISO 21771:2007  -- cylindrical involute gear geometry
  KHK Gear Technical Reference S4.2 -- internal gear pair calculations
  MAAG Gear Book  -- working geometry, contact ratio
"""

from __future__ import annotations

from axisforge.core.machine_elements.gears.parallel_axis.gear_properties.spur_helical_gear import (
    SpurHelicalGear,
)
from axisforge.core.machine_elements.gears.parallel_axis.gear_properties.internal_gear import (
    InternalGear,
)
from axisforge.core.machine_elements.gears.parallel_axis.gear_meshing.internal_meshing import (
    InternalGearMeshing,
)


def make_internal_meshing(
    gear1: SpurHelicalGear,
    gear2: InternalGear,
    *,
    label: str = "",
    al: float | None = None,
    equalise_gs: bool = False,
    addendum_reduction: bool = False,
    driver: str = "gear1",
) -> InternalGearMeshing:
    """
    Pair an external pinion (gear1) with an internal ring gear (gear2).

    gear1 : SpurHelicalGear -- external, already built (e.g. via
            make_spur_gear()/make_helical_gear()). NOT built here.
    gear2 : InternalGear -- ring gear, already built (e.g. via
            make_internal_gear(), which already enforces z<0). NOT built
            here; z>=0 is rejected below as a guard, not re-derived.
    label, al, equalise_gs, addendum_reduction, driver :
            same meaning as InternalGearMeshing.__init__ -- see that
            class's own docstring. al=None (default) derives the working
            centre distance from gear1.x + gear2.x.

    Raises
    ------
    TypeError
        If gear1 is not a SpurHelicalGear, or gear2 is not an InternalGear.
    ValueError
        If gear2.z >= 0 -- InternalGearMeshing's geometry (and
        InternalGear's own da/df formulas) require gear2 to be a properly
        built ring gear (z<0). InternalGearMeshing itself does not check
        this defensively -- see this module's own docstring's z-sign note.

    Does NOT call validate() -- the caller decides when to check.
    """
    if not isinstance(gear1, SpurHelicalGear):
        raise TypeError(
            f"make_internal_meshing: gear1 must be a SpurHelicalGear (the "
            f"external pinion), got {type(gear1).__name__}."
        )
    if not isinstance(gear2, InternalGear):
        raise TypeError(
            f"make_internal_meshing: gear2 must be an InternalGear (the "
            f"ring gear), got {type(gear2).__name__}. For two external "
            f"gears use make_spur_helical_meshing() "
            f"(spur_helical_meshing_fixture.py) instead."
        )
    if gear2.z >= 0:
        raise ValueError(
            f"make_internal_meshing: gear2.z must be < 0 (got {gear2.z}) -- "
            f"build gear2 via make_internal_gear(), which already enforces "
            f"this. See this module's own docstring for why."
        )
    return InternalGearMeshing(
        gear1, gear2, label=label, al=al, equalise_gs=equalise_gs,
        addendum_reduction=addendum_reduction, driver=driver,
    )