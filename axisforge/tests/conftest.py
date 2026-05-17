"""
tests/conftest.py
Global pytest fixtures shared across all test modules.

Naming: fixtures describe what they ARE, not what they test.
All positions in mm, forces in N, moments in N·mm.
"""

import pytest
from core.shaft import Shaft, ShaftSection, Shoulder
from core.components import Bearing, BearingType, GearElement
from core.loads import RadialLoad, AxialLoad, TorqueLoad, ExternalMoment, LoadPlane
from core.materials import Material, S355, CrMo42, AISI_1045, AISI_4340
from core.system import MechanicalSystem


# ---------------------------------------------------------------------------
# Material fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def material_s355():
    """S355 structural steel — primary reference material for Phase 1."""
    return S355


@pytest.fixture
def material_crmo42():
    """42CrMo4 alloy steel — transmission shaft reference."""
    return CrMo42


@pytest.fixture
def material_aisi_1045():
    """AISI 1045 normalised — Shigley Table A-20 reference."""
    return AISI_1045


@pytest.fixture
def material_aisi_4340():
    """AISI 4340 OQT 600 — high-strength; Se_base capped at 700 MPa."""
    return AISI_4340


@pytest.fixture
def material_custom():
    """Custom material for parametric testing."""
    return Material(
        material_id="CUSTOM",
        Sut=800.0,
        Sy=600.0,
        E=210.0,
        density=7850.0,
        description="Custom test material",
    )


@pytest.fixture
def material_high_sut():
    """Material with Sut > 1400 MPa — triggers Se cap at 700 MPa."""
    return Material(
        material_id="HIGH_SUT",
        Sut=1600.0,
        Sy=1500.0,
        E=210.0,
        density=7850.0,
    )


# ---------------------------------------------------------------------------
# Section fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def solid_section_40mm():
    """Standard solid section: d=40mm, L=100mm, S355."""
    return ShaftSection(length=100.0, diameter=40.0, material_id="S355")


@pytest.fixture
def solid_section_50mm():
    """Standard solid section: d=50mm, L=200mm, S355."""
    return ShaftSection(length=200.0, diameter=50.0, material_id="S355")


@pytest.fixture
def hollow_section():
    """Hollow section: d=60mm, d_i=30mm, L=150mm."""
    return ShaftSection(length=150.0, diameter=60.0, inner_diameter=30.0)


@pytest.fixture
def shoulder_2mm():
    """Standard shoulder: r=2mm, D=50mm, d=40mm."""
    return Shoulder(fillet_radius=2.0, diameter_large=50.0, diameter_small=40.0)


@pytest.fixture
def section_with_invalid_shoulder():
    """
    ShaftSection where shoulder_right has mismatched diameter_small.
    Used to trigger Shaft.validate() shoulder mismatch error path.
    The section itself is valid; the mismatch only appears at system level.

    shoulder_right.diameter_small=30mm but next section has d=50mm → mismatch.
    """
    return ShaftSection(
        length=100.0,
        diameter=40.0,
        label="mismatch_section",
        shoulder_right=Shoulder(
            fillet_radius=2.0,
            diameter_large=50.0,
            diameter_small=30.0,   # wrong: should match next section diameter
        ),
    )


# ---------------------------------------------------------------------------
# Shaft fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def three_section_shaft():
    """
    Simple 3-section shaft (total length 400mm):
      §1: L=100, d=40
      §2: L=200, d=50  (shoulder_left r=2mm, shoulder_right r=2mm)
      §3: L=100, d=40
    """
    shaft = Shaft(name="three_section")
    shaft.add_section(ShaftSection(
        length=100.0, diameter=40.0, label="§1",
    ))
    shaft.add_section(ShaftSection(
        length=200.0, diameter=50.0, label="§2",
        shoulder_left=Shoulder(fillet_radius=2.0, diameter_large=50.0, diameter_small=40.0),
        shoulder_right=Shoulder(fillet_radius=2.0, diameter_large=50.0, diameter_small=40.0),
    ))
    shaft.add_section(ShaftSection(
        length=100.0, diameter=40.0, label="§3",
    ))
    return shaft


@pytest.fixture
def uniform_shaft_300mm():
    """Single-section uniform shaft: d=50mm, L=300mm."""
    shaft = Shaft(name="uniform")
    shaft.add_section(ShaftSection(length=300.0, diameter=50.0))
    return shaft


@pytest.fixture
def shaft_with_shoulder_mismatch():
    """
    Shaft where shoulder_right.diameter_small does not match next section diameter.
    Triggers the shoulder mismatch branch in Shaft.validate().
    """
    shaft = Shaft(name="mismatch_shaft")
    shaft.add_section(ShaftSection(
        length=100.0, diameter=40.0, label="§1",
        shoulder_right=Shoulder(
            fillet_radius=2.0,
            diameter_large=50.0,
            diameter_small=30.0,   # mismatch: next section d=50, not 30
        ),
    ))
    shaft.add_section(ShaftSection(length=200.0, diameter=50.0, label="§2"))
    return shaft


@pytest.fixture
def shaft_with_left_shoulder_mismatch():
    """
    Shaft where section[1].shoulder_left.diameter_small does not match section[0].diameter.
    Triggers the shoulder_left mismatch branch in Shaft.validate().
    """
    shaft = Shaft(name="left_mismatch_shaft")
    shaft.add_section(ShaftSection(length=100.0, diameter=40.0, label="§1"))
    shaft.add_section(ShaftSection(
        length=200.0, diameter=50.0, label="§2",
        shoulder_left=Shoulder(
            fillet_radius=2.0,
            diameter_large=50.0,
            diameter_small=30.0,   # mismatch: section[0].diameter=40, not 30
        ),
    ))
    return shaft


# ---------------------------------------------------------------------------
# Bearing fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def skf_6210_fixed():
    """SKF 6210 bearing — fixed arrangement at x=50mm. C=35000N, C0=22000N."""
    return Bearing(
        position=50.0, designation="6210",
        C=35_000.0, C0=22_000.0,
        arrangement="fixed", label="A",
    )


@pytest.fixture
def skf_6210_floating():
    """SKF 6210 bearing — floating arrangement at x=350mm."""
    return Bearing(
        position=350.0, designation="6210",
        C=35_000.0, C0=22_000.0,
        arrangement="floating", label="B",
    )


# ---------------------------------------------------------------------------
# MechanicalSystem — primary integration fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_system(three_section_shaft, skf_6210_fixed, skf_6210_floating):
    """
    Reference system used throughout Phase 1 tests.

    Shaft: 400mm, 3 sections (40/50/40mm diameter)
    Bearings: SKF 6210 at x=50mm (fixed) and x=350mm (floating)
    Gear: Wt=3500N, Wr=1274N, T=175000N·mm at x=200mm
    Speed: 1450 rpm | Design life: 20000 h

    Hand-calculated reactions (span = 300mm, gear at midspan):
      R_A_xz = R_B_xz = 1750 N
      R_A_xy = R_B_xy = 637 N
    """
    system = MechanicalSystem(
        shaft=three_section_shaft,
        name="Demo_Shaft_001",
        speed_rpm=1450.0,
        design_life_hours=20_000.0,
    )
    system.add_bearing(skf_6210_fixed)
    system.add_bearing(skf_6210_floating)
    system.add_gear(GearElement(
        position=200.0,
        tangential_force=3500.0,
        radial_force=1274.0,
        axial_force=0.0,
        pitch_diameter=100.0,
        torque=175_000.0,
        label="G1",
    ))
    return system


@pytest.fixture
def simple_system_central_load(three_section_shaft):
    """
    System with a single central radial load (no gear, no torque).
    Used for pure static equilibrium tests.

    Load: F=5000N in XY plane at x=200mm (midspan of 50–350 span).
    Expected reactions: R_A_xy = R_B_xy = 2500N (symmetric).
    """
    system = MechanicalSystem(
        shaft=three_section_shaft,
        name="Central_Load_Test",
        speed_rpm=1450.0,
    )
    system.add_bearing(Bearing(
        position=50.0, C=35_000.0, C0=22_000.0,
        arrangement="fixed", label="A",
    ))
    system.add_bearing(Bearing(
        position=350.0, C=35_000.0, C0=22_000.0,
        arrangement="floating", label="B",
    ))
    system.add_load(RadialLoad(
        position=200.0, magnitude=5000.0,
        plane=LoadPlane.XY, label="F_central",
    ))
    return system


@pytest.fixture
def system_with_external_moment(three_section_shaft):
    """
    System with an ExternalMoment load.
    Used to exercise the external_moments filter in MechanicalSystem.
    """
    system = MechanicalSystem(
        shaft=three_section_shaft,
        name="Moment_Test",
        speed_rpm=1450.0,
    )
    system.add_bearing(Bearing(
        position=50.0, C=35_000.0, C0=22_000.0,
        arrangement="fixed", label="A",
    ))
    system.add_bearing(Bearing(
        position=350.0, C=35_000.0, C0=22_000.0,
        arrangement="floating", label="B",
    ))
    system.add_load(ExternalMoment(
        position=200.0, magnitude=50_000.0,
        plane=LoadPlane.XY, label="M_ext",
    ))
    return system
    

@pytest.fixture
def shigley_ex3_6():
    """
    Shigley Ex. 3-6 — simply-supported beam, off-centre point load.
    xA=0, xB=600mm, F=5000N (XY) at x=200mm.
    R_A=3333.3N, R_B=1666.7N, M_max=666667 N·mm at x=200mm.
    """
    shaft = Shaft(name="shigley_ex3_6")
    shaft.add_section(ShaftSection(length=600.0, diameter=50.0))
    system = MechanicalSystem(shaft=shaft, speed_rpm=0.0, name="Shigley_Ex3_6")
    system.add_bearing(Bearing(position=0.0, C=1.0, C0=1.0, arrangement="fixed", label="A"))
    system.add_bearing(Bearing(position=600.0, C=1.0, C0=1.0, arrangement="floating", label="B"))
    system.add_load(RadialLoad(position=200.0, magnitude=5000.0, plane=LoadPlane.XY))
    return system    