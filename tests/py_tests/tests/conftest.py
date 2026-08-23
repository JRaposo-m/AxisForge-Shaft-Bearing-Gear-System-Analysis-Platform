"""
tests/conftest.py

Global pytest fixtures for AxisForge unit tests.

Scope of this file: core-level objects only (shaft geometry, materials,
loads, bearings, gears, single-shaft container). Solver, mesh and system-DAG
fixtures are deliberately NOT here -- they belong to their own conftest once
those suites exist.

Naming: fixtures describe what they ARE, not what they test.

Units: mm (length), N (force), N*mm (moment), N*m (torque), MPa (stress),
degrees (angles at the interface).

ASCII only.
"""

from __future__ import annotations

import pytest

from axisforge.core.machine_elements.Shaft.shaft import (
    Shaft,
    ShaftSection,
    Shoulder,
    Keyway,
    KeywayType,
)
from axisforge.core.materials import (
    Material,
    GearMaterial,
    S355,
    CrMo42,
    AISI_1045,
    AISI_4340,
)
from axisforge.core.machine_elements.Bearings.catalog import BearingCatalog
from axisforge.core.machine_elements.Bearings.bearing import Bearing
from axisforge.core.machine_elements.Bearings.families.ball_bearing.radial.subtypes.deep_groove_ball import (
    DeepGrooveBallFamily,
)
from axisforge.core.machine_elements.Gears.Parallel_Axis_gears.spur_helical_gear import (
    SpurHelicalGear,
)
from axisforge.core.mechanical_system.Parallel_Axis_systems.systems.spur_helicoidal_system.shaft_system import (
    ShaftSystem,
    GearElement,
)


# ===========================================================================
# Materials
# ===========================================================================

@pytest.fixture
def material_s355() -> Material:
    """S355 structural steel -- Sut=590, Sy=355, no explicit Se_base."""
    return S355


@pytest.fixture
def material_crmo42() -> Material:
    """42CrMo4 -- note material_id is '42CrMo4', not 'CrMo42'."""
    return CrMo42


@pytest.fixture
def material_aisi_1045() -> Material:
    """AISI 1045 normalised -- Shigley Table A-20."""
    return AISI_1045


@pytest.fixture
def material_aisi_4340() -> Material:
    """AISI 4340 OQT 600 -- explicit Se_base=700 MPa."""
    return AISI_4340


@pytest.fixture
def material_custom() -> Material:
    """Parametric material, no Se_base -- endurance_limit falls to 0.5*Sut."""
    return Material(
        material_id="CUSTOM",
        Sut=800.0,
        Sy=600.0,
        E=210_000.0,
        density=7850.0,
        description="Custom test material",
    )


@pytest.fixture
def material_high_sut() -> Material:
    """Sut > 1400 MPa with no Se_base -- exercises the 700 MPa cap."""
    return Material(
        material_id="HIGH_SUT",
        Sut=1600.0,
        Sy=1500.0,
        E=210_000.0,
        density=7850.0,
    )


@pytest.fixture
def gear_material_steel() -> GearMaterial:
    """Case-hardened gear steel, ISO 6336-5 ME grade."""
    return GearMaterial(
        material_id="TEST_GEAR_STEEL",
        E=206_000.0,
        poisson_ratio=0.3,
        density=7830.0,
        cp=465.0,
        k_thermal=46.0,
        sigma_Hlim=1500.0,
        sigma_Flim=430.0,
        material_class="ME",
    )


# ===========================================================================
# Shaft geometry primitives
# ===========================================================================

@pytest.fixture
def shoulder_50_40() -> Shoulder:
    """r=2, D=50, d=40. Step height 5 mm, so r=2 is admissible."""
    return Shoulder(fillet_radius=2.0, diameter_large=50.0, diameter_small=40.0)


@pytest.fixture
def solid_section_40mm() -> ShaftSection:
    """Solid: d=40, L=100, S355 (default material_id)."""
    return ShaftSection(length=100.0, diameter=40.0, label="s40")


@pytest.fixture
def solid_section_50mm() -> ShaftSection:
    """Solid: d=50, L=200."""
    return ShaftSection(length=200.0, diameter=50.0, label="s50")


@pytest.fixture
def hollow_section() -> ShaftSection:
    """Hollow: d=60, di=30, L=150."""
    return ShaftSection(length=150.0, diameter=60.0, inner_diameter=30.0, label="hollow")


@pytest.fixture
def keyway_parallel() -> Keyway:
    """Explicit parallel keyway, dimensions given by hand (no table lookup)."""
    return Keyway(
        label="kw1",
        z_position=120.0,
        width=12.0,
        depth=5.0,
        length=40.0,
        keyway_type=KeywayType.PARALLEL,
    )


# ===========================================================================
# Shafts
# ===========================================================================

@pytest.fixture
def uniform_shaft_300mm() -> Shaft:
    """Single section, d=50, L=300. The simplest valid shaft."""
    shaft = Shaft(label="uniform")
    shaft.add_section(ShaftSection(length=300.0, diameter=50.0, label="body"))
    return shaft


@pytest.fixture
def stepped_shaft_400mm() -> Shaft:
    """
    Canonical stepped shaft, consistent shoulders, total length 400 mm.

        [0, 100)   d=40   seat_A
        [100, 300) d=50   body      shoulder_left and shoulder_right
        [300, 400] d=40   seat_B

    Both shoulders declare diameter_large=50 (the body) and
    diameter_small=40 (the seats), which is what Shaft.validate() checks
    against the neighbouring sections.
    """
    step = Shoulder(fillet_radius=2.0, diameter_large=50.0, diameter_small=40.0)

    shaft = Shaft(label="stepped")
    shaft.add_section(ShaftSection(length=100.0, diameter=40.0, label="seat_A"))
    shaft.add_section(
        ShaftSection(
            length=200.0,
            diameter=50.0,
            label="body",
            shoulder_left=step,
            shoulder_right=step,
        )
    )
    shaft.add_section(ShaftSection(length=100.0, diameter=40.0, label="seat_B"))
    return shaft


@pytest.fixture
def shaft_with_right_shoulder_mismatch() -> Shaft:
    """
    section[0].shoulder_right.diameter_small = 30 but section[1].diameter = 50.

    The Shoulder itself is geometrically legal (step height 10 >= r 2), so
    construction succeeds and the inconsistency only surfaces in
    Shaft.validate().
    """
    shaft = Shaft(label="right_mismatch")
    shaft.add_section(
        ShaftSection(
            length=100.0,
            diameter=40.0,
            label="s0",
            shoulder_right=Shoulder(
                fillet_radius=2.0, diameter_large=50.0, diameter_small=30.0
            ),
        )
    )
    shaft.add_section(ShaftSection(length=200.0, diameter=50.0, label="s1"))
    return shaft


@pytest.fixture
def shaft_with_left_shoulder_mismatch() -> Shaft:
    """section[1].shoulder_left.diameter_small = 30 but section[0].diameter = 40."""
    shaft = Shaft(label="left_mismatch")
    shaft.add_section(ShaftSection(length=100.0, diameter=40.0, label="s0"))
    shaft.add_section(
        ShaftSection(
            length=200.0,
            diameter=50.0,
            label="s1",
            shoulder_left=Shoulder(
                fillet_radius=2.0, diameter_large=50.0, diameter_small=30.0
            ),
        )
    )
    return shaft


# ===========================================================================
# Gears
# ===========================================================================

@pytest.fixture
def spur_pinion_20t() -> SpurHelicalGear:
    """Spur pinion z=20, mn=2, b=20, alpha_n=20 deg, x=0. Not undercut."""
    return SpurHelicalGear(mn=2.0, z=20, b=20.0, label="pinion", position=0.0)


@pytest.fixture
def spur_wheel_40t() -> SpurHelicalGear:
    """Spur wheel z=40, mn=2, b=20."""
    return SpurHelicalGear(mn=2.0, z=40, b=20.0, label="wheel", position=0.0)


@pytest.fixture
def helical_pinion_20t() -> SpurHelicalGear:
    """Helical pinion z=20, mn=2, beta=15 deg, b=25."""
    return SpurHelicalGear(
        mn=2.0, z=20, b=25.0, beta_n_deg=15.0, label="helical_pinion", position=0.0
    )


# ===========================================================================
# Bearings
# ===========================================================================

@pytest.fixture
def catalog_6204() -> BearingCatalog:
    """SKF 6204 catalogue data, locating, at x=50 mm."""
    return BearingCatalog(
        d=20.0, D=47.0, b=14.0, C=12_700.0, C0=6_550.0,
        designation="6204", position=50.0, label="brg_A",
    )


@pytest.fixture
def dgbb_family() -> DeepGrooveBallFamily:
    return DeepGrooveBallFamily()


@pytest.fixture
def dgbb_6204_geometry() -> dict:
    """Internal geometry for a 6204, the set DeepGrooveBallFamily accepts."""
    return dict(Dw=7.94, Dpw=33.5, Z=8, E=206_000.0, nu=0.3, s=0.010)


@pytest.fixture
def bearing_6204(catalog_6204, dgbb_family, dgbb_6204_geometry) -> Bearing:
    """Fully assembled, point_contact enabled, sealed (immutable)."""
    return Bearing.assemble(
        family=dgbb_family,
        catalog=catalog_6204,
        geometry=dgbb_6204_geometry,
        analyses={"point_contact": True},
    )


# ===========================================================================
# Lightweight stand-ins for ShaftSystem tests
# ===========================================================================
# ShaftSystem only reads .position / .b / .label / .designation / .validate()
# off a bearing. Using a stub keeps these tests about ShaftSystem's own
# bookkeeping instead of dragging a full Bearing.assemble() into every case.

class StubBearing:
    """Minimal bearing-shaped object accepted by ShaftSystem."""

    def __init__(self, position: float, b: float = 0.0, label: str = "brg",
                 designation: str = "", arrangement: str = "locating",
                 errors: list[str] | None = None):
        self.position = position
        self.b = b
        self.label = label
        self.designation = designation
        self.arrangement = arrangement
        self._errors = list(errors or [])

    def validate(self) -> list[str]:
        return list(self._errors)

    def __repr__(self) -> str:
        return f"StubBearing(label={self.label!r}, position={self.position})"


class StubGear:
    """Minimal gear-shaped object: ShaftSystem reads .position and .b."""

    def __init__(self, position: float, b: float = 0.0, label: str = "gear",
                 errors: list[str] | None = None):
        self.position = position
        self.b = b
        self.label = label
        self._errors = list(errors or [])

    def validate(self) -> list[str]:
        return list(self._errors)


@pytest.fixture
def stub_bearing_factory():
    """Factory fixture -- call it to build StubBearing objects inside a test."""
    return StubBearing


@pytest.fixture
def stub_gear_factory():
    """Factory fixture -- call it to build StubGear objects inside a test."""
    return StubGear


@pytest.fixture
def shaft_system_two_bearings(uniform_shaft_300mm):
    """
    Determinate single shaft: L=300, bearings (b=14) at x=40 and x=260.
    Bearing extents [33, 47] and [253, 267] do not overlap.
    """
    sys = ShaftSystem(uniform_shaft_300mm, name="S1", speed_rpm=1500.0)
    sys.add_bearing(StubBearing(position=40.0, b=14.0, label="A"))
    sys.add_bearing(StubBearing(position=260.0, b=14.0, label="B"))
    return sys


@pytest.fixture
def gear_element_driver() -> GearElement:
    """
    Driver GearElement wrapping a real SpurHelicalGear, with an explicit
    rotation_dir -- the shape a kinematic source gear has.
    """
    gear = SpurHelicalGear(mn=2.0, z=20, b=20.0, label="pinion", position=150.0)
    return GearElement(gear, role="driver", rotation_dir=1, label="ge_driver")
