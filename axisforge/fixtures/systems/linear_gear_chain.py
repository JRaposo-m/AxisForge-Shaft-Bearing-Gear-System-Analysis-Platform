"""
fixtures/systems/linear_gear_chain.py

System fixture for a linear n-stage parallel-axis gear chain.

Responsibility: assembly only.
  - Receives already-built ShaftFixture, GearPairFixture, and Bearing
    objects from the caller.
  - Creates ShaftSystem containers, places gears and bearings, computes
    shaft_origin_x, wires SpurHelicalMeshLink objects, assembles
    SpurHelicalGearSystem, and calls resolve().
  - Does NOT build shafts, gears, or bearings internally.

This separation means the caller retains full control over geometry and
can use any fixture or core object without being constrained by the
system fixture's assumptions.

Interface
---------
  StageSpec(dataclass)
    pos_driver : float   -- axial position of the driver gear on its shaft [mm]
    pos_driven : float   -- axial position of the driven gear on its shaft [mm]
    phi_deg    : float   -- line-of-centres angle [deg], +Y toward +Z
    label      : str     -- stage label (used in link and gear element labels)

  build_systems(
    shaft_fixtures     : list[ShaftFixture],
        # len == len(pairs) + 1; ordered source -> last driven
    pairs              : list[GearPairFixture],
        # one per stage; pair[i] connects shaft[i] to shaft[i+1]
    bearings_per_shaft : list[list[Bearing]],
        # one list per shaft; bearings already positioned and solver-ready
    stage_specs        : list[StageSpec],
        # one per stage; carries phi_deg and gear axial positions
    P_W                : float,     # input power [W]
    rpm_in             : float,     # input shaft speed [rpm]
    rotation_dir       : int = 1,
    label              : str = "",
    speed_rpm_per_shaft: list[float] | None = None,
        # nominal speed per shaft [rpm]; None -> computed from gear ratios
  ) -> GearSystemResult

  GearSystemResult(dataclass)
    gearbox        : SpurHelicalGearSystem
    shaft_systems  : list[ShaftSystem]
    pairs          : list[GearPairFixture]
    gear_fixtures  : list[tuple[GearFixture, GearFixture]]
    gear_elements  : list[tuple[GearElement, GearElement]]
    links          : list[SpurHelicalMeshLink]
    label          : str

Notes
-----
  - shaft_origin_x[0] = 0.0
    shaft_origin_x[i+1] = shaft_origin_x[i] + pos_driver[i] - pos_driven[i]
  - External loads (motor overhang, rope tension, etc.) are applied by
    the caller on result.shaft_systems after build_systems() returns.
  - Fan-out / power split: out of scope. Requires a dedicated pre-solver
    that sets torque_split on SpurHelicalMeshLink before resolve().

Dependencies:
  fixtures/gears/spur_helical.py    GearPairFixture, GearFixture
  fixtures/shafts/shaft_fixture.py  ShaftFixture
  axisforge core: ShaftSystem, GearElement, SpurHelicalMeshLink,
                  SpurHelicalGearSystem

CONSOLE OUTPUT: ASCII only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from axisforge.core.machine_elements.Bearings.bearing import Bearing
from axisforge.core.mechanical_system.Parallel_Axis_systems.systems.spur_helicoidal_system.shaft_system import (
    GearElement, ShaftSystem,
)
from axisforge.core.mechanical_system.Parallel_Axis_systems.systems.spur_helicoidal_system.SpurHelical_gear_system import (
    SpurHelicalMeshLink, SpurHelicalGearSystem,
)

from axisforge.fixtures.gears.spur_helical import GearPairFixture, GearFixture
from axisforge.fixtures.shafts.shaft_fixture import ShaftFixture


# ===========================================================================
# StageSpec -- minimal assembly descriptor
# ===========================================================================

@dataclass
class StageSpec:
    """
    Minimal descriptor for one gear mesh stage within the system.

    Carries only the information that the system fixture needs for
    assembly and that is not already present in the GearPairFixture:
    the axial positions of the gears on their respective shafts and the
    line-of-centres angle.

    All gear geometry (z, mn, b, x, alpha, beta, meshing) lives in the
    GearPairFixture. All shaft geometry lives in the ShaftFixture.

    Attributes
    ----------
    pos_driver : float
        Axial position of the driver gear on shaft[i] [mm].
    pos_driven : float
        Axial position of the driven gear on shaft[i+1] [mm].
    phi_deg    : float
        Line-of-centres angle in the shaft cross-section [deg],
        measured from +Y toward +Z.
    label      : str
        Stage label. Used in GearElement labels and SpurHelicalMeshLink
        label. Empty string is valid.
    """
    pos_driver: float
    pos_driven: float
    phi_deg:    float
    label:      str = ""


# ===========================================================================
# GearSystemResult
# ===========================================================================

@dataclass
class GearSystemResult:
    """
    Output of build_systems(). Exposes every intermediate object so
    downstream analyses can access any level of detail without
    reconstructing objects from scratch.

    Attributes
    ----------
    gearbox        : SpurHelicalGearSystem
        Fully resolved. Access _resolved for per-shaft T_out,
        rotation_dir, and shaft_position.

    shaft_systems  : list[ShaftSystem]
        Ordered source -> last driven. len == len(pairs) + 1.
        Add external loads here after build_systems() returns.

    pairs          : list[GearPairFixture]
        One per stage. Access pair.meshing for SpurHelicalGearMeshing
        (al, alphatw, epsilon, forces()), pair.x1/x2 for profile shifts,
        pair.driver/pair.driven for GearFixture objects.

    gear_fixtures  : list[tuple[GearFixture, GearFixture]]
        (driver_fixture, driven_fixture) per stage. Profile shift x1, x2
        are the values resolved by SpurHelicalGearMeshing, consistent
        with al, alphatw, and epsilon. Suitable for ISO 6336 or
        lubrication analysis.

    gear_elements  : list[tuple[GearElement, GearElement]]
        (ge_driver, ge_driven) per stage. The same GearElement objects
        used inside SpurHelicalMeshLink. Safe to pass to downstream
        solvers consuming GearElement.

    links          : list[SpurHelicalMeshLink]
        One per stage. Exposes meshing, phi_deg, torque_split,
        distribute_loads.

    label          : str
    """

    gearbox:       SpurHelicalGearSystem
    shaft_systems: list[ShaftSystem]
    pairs:         list[GearPairFixture]
    gear_fixtures: list[tuple[GearFixture, GearFixture]]
    gear_elements: list[tuple[GearElement, GearElement]]
    links:         list[SpurHelicalMeshLink]
    label:         str = ""

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Delegate to the resolved gearbox summary."""
        return self.gearbox.summary()

    def driver_fixture(self, stage_index: int) -> GearFixture:
        """Driver GearFixture for stage i."""
        return self.gear_fixtures[stage_index][0]

    def driven_fixture(self, stage_index: int) -> GearFixture:
        """Driven GearFixture for stage i."""
        return self.gear_fixtures[stage_index][1]

    def driver_element(self, stage_index: int) -> GearElement:
        """Driver GearElement for stage i."""
        return self.gear_elements[stage_index][0]

    def driven_element(self, stage_index: int) -> GearElement:
        """Driven GearElement for stage i."""
        return self.gear_elements[stage_index][1]

    def print_gear_summary(self) -> None:
        """
        Print a structured per-stage gear summary to console. ASCII only.
        """
        print("-" * 74)
        print("  GEAR PAIR SUMMARY")
        print("-" * 74)
        for i, (pair, (drv, drn)) in enumerate(
            zip(self.pairs, self.gear_fixtures)
        ):
            tag = pair.label or f"stage_{i}"
            print(f"  [{i}] {tag}")
            print(f"      driver : z={drv.z}, mn={drv.mn}, b={drv.b} mm, "
                  f"beta={drv.beta_n_deg} deg, x={drv.x:.4f}")
            print(f"      driven : z={drn.z}, mn={drn.mn}, b={drn.b} mm, "
                  f"beta={drn.beta_n_deg} deg, x={drn.x:.4f}")
            print(f"      u={pair.u:.4f}  a={pair.meshing.a:.4f} mm  "
                  f"al={pair.al:.4f} mm  "
                  f"ea={pair.epsilon_alpha:.4f}  "
                  f"eb={pair.epsilon_beta:.4f}  "
                  f"eg={pair.epsilon_gamma:.4f}")
        print("-" * 74)


# ===========================================================================
# build_systems
# ===========================================================================

def build_systems(
    shaft_fixtures:      list[ShaftFixture],
    pairs:               list[GearPairFixture],
    bearings_per_shaft:  list[list[Bearing]],
    stage_specs:         list[StageSpec],
    P_W:                 float,
    rpm_in:              float,
    rotation_dir:        int = 1,
    label:               str = "",
    speed_rpm_per_shaft: Optional[list[float]] = None,
) -> GearSystemResult:
    """
    Assemble and resolve a linear n-stage parallel-axis gear chain.

    All shaft, gear, and bearing objects must be fully constructed before
    calling this function. build_systems() performs assembly only.

    Parameters
    ----------
    shaft_fixtures : list[ShaftFixture]
        Ordered shaft fixtures, source -> last driven.
        len must equal len(pairs) + 1.
    pairs : list[GearPairFixture]
        Ordered gear pair fixtures. pairs[i] connects shaft_fixtures[i]
        (driver) to shaft_fixtures[i+1] (driven).
    bearings_per_shaft : list[list[Bearing]]
        One list of Bearing objects per shaft, in the same order as
        shaft_fixtures. Bearings must already be positioned (position
        attribute set) and solver-ready (make_ready() called if DGBB).
    stage_specs : list[StageSpec]
        One StageSpec per stage. Carries pos_driver, pos_driven, phi_deg,
        and label. len must equal len(pairs).
    P_W : float
        Input power at the source shaft [W].
    rpm_in : float
        Input shaft speed [rpm].
    rotation_dir : int
        Rotation direction of the source shaft: +1 or -1.
    label : str
        Label passed to SpurHelicalGearSystem.
    speed_rpm_per_shaft : list[float] | None
        Nominal speed [rpm] per shaft, in the same order as shaft_fixtures.
        If None, speed_rpm is set to 0.0 on all ShaftSystem objects and
        the caller is responsible for setting it after build.

    Returns
    -------
    GearSystemResult

    Raises
    ------
    ValueError
        If len(shaft_fixtures) != len(pairs) + 1.
        If len(bearings_per_shaft) != len(shaft_fixtures).
        If len(stage_specs) != len(pairs).
        If topology validation fails after assembly.

    Notes
    -----
    shaft_origin_x computation:
        shaft_origin_x[0] = 0.0
        shaft_origin_x[i+1] = shaft_origin_x[i]
                               + stage_specs[i].pos_driver
                               - stage_specs[i].pos_driven
    """
    n_stages = len(pairs)
    n_shafts = n_stages + 1

    if len(shaft_fixtures) != n_shafts:
        raise ValueError(
            f"build_systems: len(shaft_fixtures)={len(shaft_fixtures)} "
            f"must equal len(pairs)+1={n_shafts}."
        )
    if len(bearings_per_shaft) != n_shafts:
        raise ValueError(
            f"build_systems: len(bearings_per_shaft)={len(bearings_per_shaft)} "
            f"must equal len(shaft_fixtures)={n_shafts}."
        )
    if len(stage_specs) != n_stages:
        raise ValueError(
            f"build_systems: len(stage_specs)={len(stage_specs)} "
            f"must equal len(pairs)={n_stages}."
        )

    # ------------------------------------------------------------------
    # 1. shaft_origin_x
    # ------------------------------------------------------------------
    shaft_origin_xs: list[float] = [0.0] * n_shafts
    for i, spec in enumerate(stage_specs):
        shaft_origin_xs[i + 1] = (
            shaft_origin_xs[i] + spec.pos_driver - spec.pos_driven
        )

    # ------------------------------------------------------------------
    # 2. ShaftSystems
    # ------------------------------------------------------------------
    shaft_systems: list[ShaftSystem] = []

    for i, shaft_fix in enumerate(shaft_fixtures):
        shaft_fix.validate_or_raise()

        speed = (
            speed_rpm_per_shaft[i]
            if speed_rpm_per_shaft is not None
            else 0.0
        )

        sys = ShaftSystem(
            shaft_fix.shaft,
            name=shaft_fix.name,
            speed_rpm=speed,
        )
        sys.shaft_origin_x = shaft_origin_xs[i]

        for bearing in bearings_per_shaft[i]:
            sys.add_bearing(bearing)

        shaft_systems.append(sys)

    # ------------------------------------------------------------------
    # 3. GearElements and mesh links
    # ------------------------------------------------------------------
    gear_fixtures_out: list[tuple[GearFixture, GearFixture]] = []
    gear_elements:     list[tuple[GearElement, GearElement]] = []
    links:             list[SpurHelicalMeshLink]             = []

    for i, (pair, spec) in enumerate(zip(pairs, stage_specs)):
        driver_label = (
            f"{spec.label}_driver" if spec.label else f"stage{i}_driver"
        )
        driven_label = (
            f"{spec.label}_driven" if spec.label else f"stage{i}_driven"
        )

        # GearFixtures with resolved x1, x2 from the pair
        driver_fix = pair.driver.with_x(pair.x1).with_position(spec.pos_driver)
        driven_fix = pair.driven.with_x(pair.x2).with_position(spec.pos_driven)
        gear_fixtures_out.append((driver_fix, driven_fix))

        # GearElements using the positioned gear cores
        ge_driver = GearElement(
            driver_fix.gear,
            role="driver",
            rotation_dir=(rotation_dir if i == 0 else None),
            label=driver_label,
        )
        ge_driven = GearElement(
            driven_fix.gear,
            role="driven",
            label=driven_label,
        )
        gear_elements.append((ge_driver, ge_driven))

        shaft_systems[i].add_gear(ge_driver)
        shaft_systems[i + 1].add_gear(ge_driven)

        link = SpurHelicalMeshLink(
            shaft_a=shaft_systems[i],
            gear_a=ge_driver,
            shaft_b=shaft_systems[i + 1],
            gear_b=ge_driven,
            meshing=pair.meshing,
            phi_deg=spec.phi_deg,
            label=spec.label or f"stage_{i}",
        )
        links.append(link)

    # ------------------------------------------------------------------
    # 4. Assemble, validate topology, resolve
    # ------------------------------------------------------------------
    gearbox = SpurHelicalGearSystem(
        shafts=shaft_systems,
        links=links,
        label=label,
    )

    topo_errors = gearbox._topology_errors()
    if topo_errors:
        raise ValueError(
            "build_systems: topology errors after assembly:\n"
            + "\n".join(f"  - {e}" for e in topo_errors)
        )

    gearbox.resolve(
        P=P_W,
        rpm=rpm_in,
        rotation_dir_source=rotation_dir,
        source_position=(0.0, 0.0),
    )

    return GearSystemResult(
        gearbox=gearbox,
        shaft_systems=shaft_systems,
        pairs=pairs,
        gear_fixtures=gear_fixtures_out,
        gear_elements=gear_elements,
        links=links,
        label=label,
    )