"""
tests/conftest.py
Global pytest fixtures shared across all test modules.

Naming: fixtures describe what they ARE, not what they test.
All positions in mm, forces in N, moments in N·mm.
"""

import pytest
import math
from __future__ import annotations
from axisforge.core.machine_elements.Gears.Parallel_Axis_gears.spur_gear import SpurGear
from axisforge.core.machine_elements.Gears.Parallel_Axis_gears.helical_gear import HelicalGear
from axisforge.core.machine_elements.Gears.Parallel_Axis_gears.internal_gear import InternalGear
from axisforge.core.mechanical_system.gear_meshing.internal_gear_meshing import InternalGearMeshing
from axisforge.core.machine_elements.Shaft.shaft import Shaft, ShaftSection, Shoulder
# from axisforge.core.components import Bearing, BearingType, GearElement
# from axisforge.core.loads import RadialLoad, AxialLoad, TorqueLoad, ExternalMoment, LoadPlane
# from axisforge.core.materials import Material, S355, CrMo42, AISI_1045, AISI_4340
# from axisforge.core.system import MechanicalSystem


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

"""
Lightweight fakes standing in for the real geometry/bearing/meshing classes.
They implement only the attribute surface that ShaftSystem / GearSystem touch,
so the systems logic can be tested without the full AxisForge stack.
"""


class FakeShaft:
    def __init__(self, total_length: float = 200.0, errors=None):
        self._L = total_length
        self._errors = errors or []

    @property
    def total_length(self) -> float:
        return self._L

    def validate(self) -> list[str]:
        return list(self._errors)


class FakeBearing:
    def __init__(self, position: float, arrangement: str = "floating",
                 Ka=None, errors=None, name="brg"):
        self.position = position
        self.arrangement = arrangement
        self.Ka = Ka
        self._errors = errors or []
        self.name = name

    def validate(self) -> list[str]:
        return list(self._errors)


class FakeGear:
    """Minimal gear geometry: only `.position` (+ optional validate)."""
    def __init__(self, position: float, errors=None):
        self.position = position
        self._errors = errors or []

    def validate(self) -> list[str]:
        return list(self._errors)


class FakeMeshing:
    """
    Configurable meshing whose forces() returns caller-controlled numbers,
    used to exercise topology / propagation / load-injection independently of
    real gear math.
    """
    def __init__(self, al: float = 100.0, u: float = 2.0,
                 Ft=1000.0, Fr=364.0, Fa=0.0, internal=False):
        self.al = al
        self.u = u
        self._Ft = Ft
        self._Fr = Fr
        self._Fa = Fa
        self._internal = internal
        self.calls: list[tuple] = []

    def forces(self, T_in: float, phi_deg: float = 0.0,
               rotation_dir_in: int = 1) -> dict:
        self.calls.append((T_in, phi_deg, rotation_dir_in))
        if self._internal:
            rotation_dir_out = rotation_dir_in           # internal: same sense
            T_out = T_in * abs(self.u)
        else:
            rotation_dir_out = -rotation_dir_in          # external: inverts
            T_out = T_in * self.u
        out = {
            "Ft": self._Ft, "Fr": self._Fr, "Fn": 0.0, "T_out": T_out,
            "theta_Fr_driver": phi_deg % 360.0,
            "theta_Ft_driver": (phi_deg + 90.0 * rotation_dir_in) % 360.0,
            "theta_Fr_driven": (phi_deg + 180.0) % 360.0,
            "theta_Ft_driven": (phi_deg + 180.0 + 90.0 * rotation_dir_out) % 360.0,
            "rotation_dir_out": rotation_dir_out,
        }
        if self._Fa:
            out["Fa"] = self._Fa
        return out


class SpurLikeMeshing:
    """
    Faithful re-implementation of SpurGearMeshing.forces() (documented spur
    formulae) so the reducer regression can be checked against a hand calc.

    rl1 in mm (working pitch radius of the driver), alphaw in rad, u = z2/z1.
    """
    def __init__(self, rl1_mm: float, alphaw_rad: float, u: float, al: float):
        self.rl1_mm = rl1_mm
        self.alphaw = alphaw_rad
        self.u = u
        self.al = al

    def forces(self, T1: float, phi_deg: float = 0.0,
               rotation_dir: int = 1, rotation_dir_in: int | None = None) -> dict:
        rot = rotation_dir_in if rotation_dir_in is not None else rotation_dir
        Ft = T1 / (self.rl1_mm / 1000.0)
        Fr = Ft * math.tan(self.alphaw)
        Fn = Ft / math.cos(self.alphaw)
        T_out = T1 * self.u
        theta_Fr_driver = phi_deg % 360.0
        theta_Ft_driver = (phi_deg + 90.0 * rot) % 360.0
        rotation_dir_out = -self.u / abs(self.u) * rot
        theta_Fr_driven = (phi_deg + 180.0) % 360.0
        theta_Ft_driven = (phi_deg + 180.0 + 90.0 * rotation_dir_out) % 360.0
        return {
            "Ft": Ft, "Fr": Fr, "Fn": Fn, "T_out": T_out,
            "theta_Fr_driver": theta_Fr_driver, "theta_Ft_driver": theta_Ft_driver,
            "theta_Fr_driven": theta_Fr_driven, "theta_Ft_driven": theta_Ft_driven,
            "rotation_dir_out": int(rotation_dir_out),
        }