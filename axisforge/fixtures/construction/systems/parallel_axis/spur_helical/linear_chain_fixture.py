"""
fixtures/construction/systems/parallel_axis/spur_helical/linear_chain_fixture.py

Assembles a LINEAR (no fan-out, no convergent merge) chain of N gear
stages into one SpurHelicalGearSystem -- N+1 shafts, N meshes, exactly
the "standard case" documented on SpurHelicalGearSystem's own topology
rules (single-source DAG, one incoming link per shaft).

DELIBERATE EXCEPTION to the Construction/MeshLoads boundary used by
every other fixture in this package: build_linear_system() ALSO calls
SpurHelicalGearSystem.resolve() and returns an already-resolved system.
Every other domain in this Construction pass (shafts, bearings, gears,
gear meshing) stops at building objects and never dispatches an
analysis that needs a real load case (no .forces(), no bearing
analysis beyond assemble()) -- that discipline was deliberately
overturned HERE, on explicit user decision: a system's whole purpose is
to be resolved and positioned, so shipping it unresolved was judged an
incomplete deliverable rather than a clean stage boundary. P (power,
[W]) is therefore now a REQUIRED keyword-only argument, and rpm is
sourced from shaft_specs[0].speed_rpm (the source shaft) rather than a
separate parameter, to avoid the source rpm being declared in two
places that could disagree.

Consequences:
  - ShaftSpec now carries an optional `loads` field (already-built
    Load objects, source="user" by default in core) -- attached via
    ShaftSystem.add_load() before resolve() runs, so a caller no
    longer has to build the ShaftSystem/GearSystem outside this
    fixture just to declare an external load (an overhung sprocket,
    a coupling torque, ...). Gear-mesh loads (source="gear_mesh") are
    still exclusively resolve()'s own business -- never set here.
  - build_linear_system() calls system.validate_or_raise() BEFORE
    resolve() -- full validation (topology + per-shaft: bearing count,
    overlaps, shoulder coincidence, ...), not just resolve()'s own
    partial topology-only check, so a bad system fails fast with every
    error at once rather than piecemeal.
  - The returned SpurHelicalGearSystem is ALREADY resolved: every
    ShaftSystem carries its gear-mesh loads (RadialLoad/AxialLoad/
    TorqueLoad, or DistributedRadialLoad when a stage's
    distribute_loads=True) and its global shaft_position -- there is
    no separate "call resolve() yourself later" step for a linear
    chain built through this fixture.

Two dataclasses, one factory -- adapted from the StageSpec/ShaftSpec
pattern already documented in the project's own fixtures README, to
the current capabilities architecture. Deliberately takes ALREADY-BUILT
domain objects (Shaft, Bearing, SpurHelicalGear, Load) rather than raw
geometry kwargs -- this fixture composes shaft/bearings/gears/meshing/
loads, it does not construct any of them itself. Matches the
_PREREQUISITES comment already in fixtures/capabilities/__init__.py:
"build_linear_system() receives already-built Shaft/GearElement/
Bearing objects, it does not construct them itself."

Scope of this first cut (confirmed): linear chains only -- every
StageSpec.torque_split stays None (SpurHelicalGearSystem's own
fan-out/torque_split mechanics are simultaneous-driver machinery, not
exercised by a linear chain where each driver has exactly one outgoing
link) -- and spur/helical stages only (SpurHelicalGearMeshing, via
make_spur_helical_meshing from spur_helical_meshing_fixture.py).
Internal-gear stages and fan-out are both deferred, same as the rest of
this Construction pass.

Dependency (core only):
  axisforge.core.machine_elements.shaft.shaft
      Shaft
  axisforge.core.machine_elements.bearings.bearing
      Bearing
  axisforge.core.machine_elements.gears.parallel_axis.gear_properties.spur_helical_gear
      SpurHelicalGear
  axisforge.core.loads
      Load
  axisforge.core.mechanical_system.parallel_axis.spur_helical.shaft_system
      GearElement, ShaftSystem
  axisforge.core.mechanical_system.parallel_axis.spur_helical.gear_system
      SpurHelicalMeshLink, SpurHelicalGearSystem
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from axisforge.config import DEFAULT_DESIGN_LIFE_HOURS
from axisforge.core.machine_elements.gears.parallel_axis.gear_properties.spur_helical_gear import (
    SpurHelicalGear,
)
from axisforge.core.mechanical_system.parallel_axis.spur_helical.shaft_system import (
    GearElement, ShaftSystem,
)
from axisforge.core.mechanical_system.parallel_axis.spur_helical.gear_system import (
    SpurHelicalMeshLink, SpurHelicalGearSystem,
)
from axisforge.fixtures.construction.gears.parallel_axis.fixed.spur_helical_meshing_fixture import (
    make_spur_helical_meshing,
)

if TYPE_CHECKING:  # pragma: no cover
    from axisforge.core.machine_elements.shaft.shaft import Shaft
    from axisforge.core.machine_elements.bearings.bearing import Bearing
    from axisforge.core.loads import Load


# ===========================================================================
# ShaftSpec -- one per shaft (N+1 for an N-stage chain)
# ===========================================================================

@dataclass(frozen=True)
class ShaftSpec:
    """
    Describes ONE shaft in the chain. All heavy objects (shaft, bearings)
    are already built -- this only groups them plus the per-shaft
    ShaftSystem kwargs.

    shaft             : Shaft -- already built (e.g. via
                        make_stepped_shaft(...).shaft; ShaftSystem takes
                        the core Shaft, not the ShaftFixture wrapper).
    bearings          : already-built Bearing objects, positions already
                        set on their own BearingCatalog. Added via
                        ShaftSystem.add_bearing() in the order given.
    loads             : already-built Load objects (RadialLoad, AxialLoad,
                        TorqueLoad, ExternalMoment, DistributedRadialLoad --
                        axisforge.core.loads), typically source="user"
                        (core default). Added via ShaftSystem.add_load()
                        in the order given, BEFORE resolve() runs, so
                        they are already present when resolve()'s own
                        set_gear_loads() adds the gear-mesh loads
                        alongside them (set_gear_loads() only ever
                        replaces source=="gear_mesh" entries -- user
                        loads placed here are never touched). Do NOT
                        put a source="gear_mesh" Load here yourself --
                        that is exclusively resolve()'s business and
                        would be silently dropped on the next resolve.
    speed_rpm         : for shaft_specs[0] (the source shaft) this is
                        DUAL-USE -- stored on ShaftSystem.speed_rpm AND
                        consumed as build_linear_system()'s `rpm` input
                        to SpurHelicalGearSystem.resolve() (see that
                        function's own docstring for why the source rpm
                        is not also a separate parameter). Must be > 0
                        on shaft_specs[0] -- resolve() raises otherwise.
                        For every other shaft, still informational only
                        (per-shaft rpm downstream of the source is a
                        StaticsSolver kinematics concern, per
                        SpurHelicalGearSystem.resolve()'s own comment).
    design_life_hours : passed straight to ShaftSystem.
    shaft_origin_x    : global X offset of this shaft -- build_linear_system()
                        does NOT set this; SpurHelicalGearSystem's own
                        _axial_alignment_errors() reads it later, but
                        populating it consistently across a chain is the
                        caller's responsibility (same note as on
                        SpurHelicalGearSystem itself). NOTE this is
                        purely axial (X) placement, independent of and
                        prior to resolve()'s own (y, z) shaft_position,
                        which resolve() DOES set.
    name, label       : ShaftSystem.name / .label.
    """
    shaft: "Shaft"
    bearings: tuple["Bearing", ...] = ()
    loads: tuple["Load", ...] = ()
    speed_rpm: float = 0.0
    design_life_hours: float = DEFAULT_DESIGN_LIFE_HOURS
    shaft_origin_x: float = 0.0
    name: str = ""
    label: str = ""


# ===========================================================================
# StageSpec -- one per mesh (N for an N-stage chain)
# ===========================================================================

@dataclass(frozen=True)
class StageSpec:
    """
    Describes ONE mesh, connecting shaft_specs[i] (driver side) to
    shaft_specs[i+1] (driven side) in build_linear_system()'s own
    ordering -- StageSpec itself does not reference shaft index, that's
    positional in the stage_specs list.

    gear_driver, gear_driven : SpurHelicalGear, already built (e.g. via
        make_spur_gear()/make_helical_gear()), with .position already
        set on their respective shaft. Compatibility (matching mn,
        alpha_n_deg, beta_n_deg) is enforced by SpurHelicalGearMeshing
        itself, not re-checked here.
    phi_deg           : line-of-centres angle (driver -> driven), GLOBAL
                        frame [deg] -- forwarded to SpurHelicalMeshLink,
                        NOT derived.
    al, equalise_gs, addendum_reduction, meshing_driver :
                        forwarded to make_spur_helical_meshing() (the
                        last renamed from SpurHelicalGearMeshing's own
                        `driver` kwarg, to avoid confusion with
                        GearElement's driver ROLE).
    torque_split      : left None in this linear-only fixture -- see
                        this module's own docstring. Exposed (not
                        hardcoded away) only because it's a real
                        SpurHelicalMeshLink constructor kwarg; passing a
                        non-None value here is not validated against
                        fan-out rules by this fixture.
    distribute_loads  : forwarded to SpurHelicalMeshLink -- only matters
                        once resolve() runs, which build_linear_system()
                        now does itself (see this module's own
                        docstring); kept as a StageSpec field because
                        it's a constructor kwarg on the link, decided
                        per-stage before resolve() is reached.
    label             : SpurHelicalMeshLink.label.
    """
    gear_driver: SpurHelicalGear
    gear_driven: SpurHelicalGear
    phi_deg: float
    al: float | None = None
    equalise_gs: bool = False
    addendum_reduction: bool = False
    meshing_driver: str = "gear1"
    torque_split: float | None = None
    distribute_loads: bool = False
    label: str = ""


# ===========================================================================
# build_linear_system
# ===========================================================================

def build_linear_system(
    shaft_specs: list[ShaftSpec],
    stage_specs: list[StageSpec],
    *,
    P: float,
    rotation_dir_source: int = 1,
    source_position: tuple[float, float] = (0.0, 0.0),
    label: str = "",
) -> SpurHelicalGearSystem:
    """
    Assemble a linear N-stage chain: len(shaft_specs) == len(stage_specs) + 1
    -- and RESOLVE it (see this module's own docstring for why resolve()
    is now part of this fixture, unlike every other Construction fixture
    in this package).

    stage_specs[i] connects shaft_specs[i] (driver side) to
    shaft_specs[i+1] (driven side). Each intermediate shaft (0 < i <
    len(shaft_specs)-1) accumulates TWO GearElements across two
    different stages -- once as the driven side of stage_specs[i-1],
    once as the driver side of stage_specs[i] -- with no special-casing
    needed: the loop below just adds whatever gear each stage hands to
    whichever shaft, in order.

    P : REQUIRED, power [W] delivered at the source shaft (shaft_specs[0]).
        Keyword-only -- there is no sensible positional slot for it next
        to rotation_dir_source that wouldn't read backwards at a call
        site (`build_linear_system(specs, stages, 5000.0, 1)` -- which is
        which?).
    rotation_dir_source : +1 or -1, set on the driver GearElement of
        stage_specs[0] -- this is the kinematic SOURCE gear of the
        resulting SpurHelicalGearSystem (see GearElement's own
        docstring: rotation_dir is set explicitly ONLY for the source
        gear, None everywhere else). Reused directly as resolve()'s own
        rotation_dir_source argument -- one value, not declared twice.
    source_position : forwarded to resolve() as its own source_position
        -- the (y, z) global offset of the source shaft. Defaults to
        (0.0, 0.0), same as resolve()'s own default.

    rpm for resolve() is NOT a parameter here -- it is read from
    shaft_specs[0].speed_rpm (see ShaftSpec's own docstring for why:
    one field, not the source rpm declared in two places that could
    disagree). Must be > 0 -- resolve() raises ValueError otherwise.

    Returns
    -------
    SpurHelicalGearSystem, ALREADY RESOLVED: every ShaftSystem carries
    its gear-mesh loads and its global shaft_position. No further
    .resolve() call is needed (though calling it again is safe and
    idempotent -- set_gear_loads() only ever replaces the gear_mesh
    entries, never user loads from ShaftSpec.loads).

    Raises
    ------
    ValueError
        - If len(shaft_specs) != len(stage_specs) + 1 -- this fixture only
          builds a plain linear chain (no fan-out, no branching); that
          shape is enforced here up front with a clear message, rather
          than surfacing later and more confusingly from
          SpurHelicalGearSystem's own DAG validation.
        - Propagated from system.validate_or_raise() -- full topology +
          per-shaft validation (bearing count, overlaps, shoulder
          coincidence, axial alignment, ...), checked BEFORE resolve()
          so every error surfaces at once.
        - Propagated from SpurHelicalGearSystem.resolve() itself --
          shaft_specs[0].speed_rpm <= 0, or rotation_dir_source not in
          (1, -1).
    """
    if len(shaft_specs) != len(stage_specs) + 1:
        raise ValueError(
            f"build_linear_system: a linear chain of N stages needs "
            f"exactly N+1 shafts -- got {len(shaft_specs)} shaft_specs "
            f"and {len(stage_specs)} stage_specs "
            f"(expected {len(stage_specs) + 1} shafts)."
        )

    shaft_systems: list[ShaftSystem] = []
    for i, spec in enumerate(shaft_specs):
        ss = ShaftSystem(
            shaft=spec.shaft,
            name=spec.name or f"shaft{i + 1}",
            speed_rpm=spec.speed_rpm,
            design_life_hours=spec.design_life_hours,
            shaft_origin_x=spec.shaft_origin_x,
            label=spec.label,
        )
        for bearing in spec.bearings:
            ss.add_bearing(bearing)
        for load in spec.loads:
            ss.add_load(load)
        shaft_systems.append(ss)

    links: list[SpurHelicalMeshLink] = []
    for i, stage in enumerate(stage_specs):
        shaft_a = shaft_systems[i]
        shaft_b = shaft_systems[i + 1]

        ge_driver = GearElement(
            stage.gear_driver, role="driver",
            rotation_dir=rotation_dir_source if i == 0 else None,
            label=stage.gear_driver.label or f"{shaft_a.name}:driver",
        )
        ge_driven = GearElement(
            stage.gear_driven, role="driven",
            label=stage.gear_driven.label or f"{shaft_b.name}:driven",
        )

        shaft_a.add_gear(ge_driver)
        shaft_b.add_gear(ge_driven)

        meshing = make_spur_helical_meshing(
            stage.gear_driver, stage.gear_driven,
            label=stage.label, al=stage.al, equalise_gs=stage.equalise_gs,
            addendum_reduction=stage.addendum_reduction,
            driver=stage.meshing_driver,
        )

        link = SpurHelicalMeshLink(
            shaft_a, ge_driver, shaft_b, ge_driven, meshing,
            phi_deg=stage.phi_deg, torque_split=stage.torque_split,
            distribute_loads=stage.distribute_loads, label=stage.label,
        )
        links.append(link)

    system = SpurHelicalGearSystem(shafts=shaft_systems, links=links, label=label)

    # Full validation (topology + every ShaftSystem's own validate()) --
    # BEFORE resolve(), so a bad system fails with every error at once
    # rather than piecemeal from resolve()'s own partial topology-only
    # check (see this function's own docstring, Raises section).
    system.validate_or_raise()

    rpm = shaft_specs[0].speed_rpm
    system.resolve(P, rpm, rotation_dir_source, source_position)

    return system