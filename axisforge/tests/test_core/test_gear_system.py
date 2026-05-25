"""
tests/test_gear_system.py
Unit and validation tests for GearStage and GearSystem.

Coverage:
  - GearStage.ratio
  - GearStage.centre_distance_geometric vs geometry.al
  - GearStage.x_gear_*_global
  - GearStage.validate() — centre distance, X-contact, speed consistency
  - GearSystem.input_shaft / output_shaft
  - GearSystem._has_cycle
  - GearSystem.propagate_speeds
  - GearSystem.validate — full suite
  - End-to-end: single-stage reducer
  - End-to-end: two-stage reducer (synthetic)

Reference geometry (spur, x=0):
  mn=3, z1=20, z2=40, alpha_n=20°, beta=0°
  d1=60mm, d2=120mm, a=al=90mm

Reference geometry stage 2:
  mn=2.5, z1=25, z2=50, alpha_n=20°, beta=0°
  d1=62.5mm, d2=125mm, a=al=93.75mm

All fixtures are self-contained — no dependency on GearSolver.
"""
from __future__ import annotations

import math
import pytest

from core.shaft import Shaft, ShaftSection
from core.components import Bearing, GearElement
from core.system import MechanicalSystem
from core.gear_system import GearStage, GearSystem
from models.gear_result import GearGeometryResult, GearForceResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_shaft(name: str, length: float = 400.0, diameter: float = 40.0,
                speed_rpm: float = 0.0,
                shaft_position: tuple[float, float] = (0.0, 0.0),
                shaft_origin_x: float = 0.0) -> MechanicalSystem:
    """Minimal shaft with 2 bearings."""
    shaft = Shaft(name=name)
    shaft.add_section(ShaftSection(length=length, diameter=diameter))
    sys = MechanicalSystem(
        shaft=shaft,
        name=name,
        speed_rpm=speed_rpm,
        shaft_position=shaft_position,
        shaft_origin_x=shaft_origin_x,
    )
    sys.add_bearing(Bearing(position=20.0, C=35_000.0, C0=22_000.0,
                            arrangement="fixed", label="A"))
    sys.add_bearing(Bearing(position=380.0, C=35_000.0, C0=22_000.0,
                            arrangement="floating", label="B"))
    return sys


def _make_geometry_stage1() -> GearGeometryResult:
    """
    Spur pair: mn=3, z1=20, z2=40, α=20°, β=0°.
    al = a = 90 mm.  u = 2.0.
    """
    mn = 3.0
    z1, z2 = 20, 40
    alpha_n = 20.0
    beta = 0.0
    mt = mn                          # spur: mt=mn
    alpha_t = alpha_n
    beta_b = 0.0
    u = z2 / z1
    d1 = mn * z1
    d2 = mn * z2
    db1 = d1 * math.cos(math.radians(alpha_n))
    db2 = d2 * math.cos(math.radians(alpha_n))
    a = al = (d1 + d2) / 2.0
    dl1 = 2.0 * al / (u + 1.0)
    dl2 = 2.0 * al - dl1
    alpha_tw = alpha_t              # standard centre distance → αtw = αt
    p_bt = math.pi * db1 / z1
    # Contact ratio (approximate: εα ≈ 1.6 for this pair — not critical here)
    eps_alpha = 1.60
    return GearGeometryResult(
        mn=mn, z1=z1, z2=z2,
        alpha_n_deg=alpha_n, beta_deg=beta,
        x1=0.0, x2=0.0,
        mt=mt, alpha_t_deg=alpha_t, beta_b_deg=beta_b,
        u=u, a=a, al=al, alpha_tw_deg=alpha_tw,
        d1=d1, d2=d2, db1=db1, db2=db2,
        da1=d1 + 2.0 * mn, da2=d2 + 2.0 * mn,
        df1=d1 - 2.5 * mn, df2=d2 - 2.5 * mn,
        dl1=dl1, dl2=dl2,
        p_bt=p_bt,
        eps_alpha=eps_alpha, eps_beta=0.0, eps_gamma=eps_alpha,
        b=0.0,
    )


def _make_geometry_stage2() -> GearGeometryResult:
    """
    Spur pair: mn=2.5, z1=25, z2=50, α=20°, β=0°.
    al = a = 93.75 mm.  u = 2.0.
    """
    mn = 2.5
    z1, z2 = 25, 50
    alpha_n = 20.0
    beta = 0.0
    mt = mn
    alpha_t = alpha_n
    beta_b = 0.0
    u = z2 / z1
    d1 = mn * z1
    d2 = mn * z2
    db1 = d1 * math.cos(math.radians(alpha_n))
    db2 = d2 * math.cos(math.radians(alpha_n))
    a = al = (d1 + d2) / 2.0
    dl1 = 2.0 * al / (u + 1.0)
    dl2 = 2.0 * al - dl1
    alpha_tw = alpha_t
    p_bt = math.pi * db1 / z1
    eps_alpha = 1.65
    return GearGeometryResult(
        mn=mn, z1=z1, z2=z2,
        alpha_n_deg=alpha_n, beta_deg=beta,
        x1=0.0, x2=0.0,
        mt=mt, alpha_t_deg=alpha_t, beta_b_deg=beta_b,
        u=u, a=a, al=al, alpha_tw_deg=alpha_tw,
        d1=d1, d2=d2, db1=db1, db2=db2,
        da1=d1 + 2.0 * mn, da2=d2 + 2.0 * mn,
        df1=d1 - 2.5 * mn, df2=d2 - 2.5 * mn,
        dl1=dl1, dl2=dl2,
        p_bt=p_bt,
        eps_alpha=eps_alpha, eps_beta=0.0, eps_gamma=eps_alpha,
        b=0.0,
    )


def _make_forces(T1_Nm: float, geo: GearGeometryResult) -> GearForceResult:
    """Minimal GearForceResult consistent with geometry."""
    T1_Nmm = T1_Nm * 1000.0
    T2_Nmm = T1_Nmm * geo.u
    Ft = T1_Nmm / geo.rl1
    alpha_tw_rad = math.radians(geo.alpha_tw_deg)
    Fbt = T1_Nmm / geo.rb1
    Fr = Fbt * math.sin(alpha_tw_rad)
    Fa = 0.0
    Fn = Ft
    Fbn = Fbt
    return GearForceResult(
        Ft=Ft, Fr=Fr, Fa=Fa, Fn=Fn,
        Fbt=Fbt, Fbn=Fbn,
        T1_Nmm=T1_Nmm, T2_Nmm=T2_Nmm,
        gear_ratio=geo.u,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def geo1():
    return _make_geometry_stage1()


@pytest.fixture
def geo2():
    return _make_geometry_stage2()


@pytest.fixture
def forces1(geo1):
    return _make_forces(100.0, geo1)   # T1 = 100 N·m


@pytest.fixture
def forces2(geo2):
    return _make_forces(200.0, geo2)   # T1 = 200 N·m (driven by stage 1 output)


@pytest.fixture
def single_stage_system(geo1, forces1):
    """
    Single-stage reducer: shaft_1 (driver) → shaft_2 (driven).
    al = 90 mm → shaft_2.shaft_position = (0, 90).
    Gear contact at x_global = 200 mm.
      shaft_1: origin_x=0,   x_gear_local=200 → x_global=200
      shaft_2: origin_x=50,  x_gear_local=150 → x_global=200  ✓
    Speed: 1500 rpm input.
    """
    shaft_1 = _make_shaft("Shaft_1", speed_rpm=1500.0,
                           shaft_position=(0.0, 0.0), shaft_origin_x=0.0)
    shaft_2 = _make_shaft("Shaft_2",
                           shaft_position=(0.0, 90.0), shaft_origin_x=50.0)

    stage = GearStage(
        label="Stage_1",
        shaft_driver=shaft_1,
        shaft_driven=shaft_2,
        geometry=geo1,
        forces=forces1,
        x_gear_driver_local=200.0,
        x_gear_driven_local=150.0,
    )

    system = GearSystem(name="Single_Stage")
    system.add_stage(stage)
    return system


@pytest.fixture
def two_stage_system(geo1, geo2, forces1, forces2):
    """
    Two-stage reducer: shaft_1 → shaft_2 → shaft_3.
    Stage 1: al=90mm  → shaft_2 at (0, 90).
    Stage 2: al=93.75mm → shaft_3 at (0, 90+93.75=183.75).

    X-contact alignment:
      Stage 1 gear: shaft_1 origin=0,   x_local=200 → x_global=200
                    shaft_2 origin=50,  x_local=150 → x_global=200 ✓
      Stage 2 gear: shaft_2 origin=50,  x_local=300 → x_global=350
                    shaft_3 origin=100, x_local=250 → x_global=350 ✓
    Speed: 1500 rpm input → shaft_2: 750 rpm → shaft_3: 375 rpm.
    """
    shaft_1 = _make_shaft("Shaft_1", speed_rpm=1500.0,
                           shaft_position=(0.0, 0.0), shaft_origin_x=0.0)
    shaft_2 = _make_shaft("Shaft_2",
                           shaft_position=(0.0, 90.0), shaft_origin_x=50.0)
    shaft_3 = _make_shaft("Shaft_3",
                           shaft_position=(0.0, 183.75), shaft_origin_x=100.0)

    stage_1 = GearStage(
        label="Stage_1", shaft_driver=shaft_1, shaft_driven=shaft_2,
        geometry=geo1, forces=forces1,
        x_gear_driver_local=200.0, x_gear_driven_local=150.0,
    )
    stage_2 = GearStage(
        label="Stage_2", shaft_driver=shaft_2, shaft_driven=shaft_3,
        geometry=geo2, forces=forces2,
        x_gear_driver_local=300.0, x_gear_driven_local=250.0,
    )

    system = GearSystem(name="Two_Stage")
    system.add_stage(stage_1)
    system.add_stage(stage_2)
    return system


# ---------------------------------------------------------------------------
# GearStage — unit tests
# ---------------------------------------------------------------------------

class TestGearStageRatio:
    def test_ratio_stage1(self, geo1, forces1):
        shaft_1 = _make_shaft("S1", speed_rpm=1500.0,
                               shaft_position=(0.0, 0.0))
        shaft_2 = _make_shaft("S2", shaft_position=(0.0, 90.0))
        stage = GearStage("St1", shaft_1, shaft_2, geo1, forces1)
        assert stage.ratio == pytest.approx(2.0, rel=1e-9)

    def test_ratio_stage2(self, geo2, forces2):
        shaft_1 = _make_shaft("S1", speed_rpm=1000.0,
                               shaft_position=(0.0, 0.0))
        shaft_2 = _make_shaft("S2", shaft_position=(0.0, 93.75))
        stage = GearStage("St2", shaft_1, shaft_2, geo2, forces2)
        assert stage.ratio == pytest.approx(2.0, rel=1e-9)


class TestGearStageCentreDistance:
    def test_correct_centre_distance(self, geo1, forces1):
        # shaft_2 at y=90 → geometric distance = 90 = al
        shaft_1 = _make_shaft("S1", speed_rpm=1500.0,
                               shaft_position=(0.0, 0.0))
        shaft_2 = _make_shaft("S2", shaft_position=(0.0, 90.0))
        stage = GearStage("St", shaft_1, shaft_2, geo1, forces1)
        assert stage.centre_distance_geometric == pytest.approx(90.0, abs=1e-9)

    def test_centre_distance_diagonal(self, geo1, forces1):
        # dy=54, dz=72 → √(54²+72²) = 90
        shaft_1 = _make_shaft("S1", shaft_position=(0.0, 0.0))
        shaft_2 = _make_shaft("S2", shaft_position=(54.0, 72.0))
        stage = GearStage("St", shaft_1, shaft_2, geo1, forces1)
        assert stage.centre_distance_geometric == pytest.approx(90.0, abs=1e-6)


class TestGearStageXGlobal:
    def test_x_global_driver(self, geo1, forces1):
        shaft_1 = _make_shaft("S1", shaft_origin_x=0.0,
                               shaft_position=(0.0, 0.0))
        shaft_2 = _make_shaft("S2", shaft_origin_x=50.0,
                               shaft_position=(0.0, 90.0))
        stage = GearStage("St", shaft_1, shaft_2, geo1, forces1,
                          x_gear_driver_local=200.0, x_gear_driven_local=150.0)
        assert stage.x_gear_driver_global == pytest.approx(200.0)
        assert stage.x_gear_driven_global == pytest.approx(200.0)

    def test_x_global_mismatch_detected(self, geo1, forces1):
        shaft_1 = _make_shaft("S1", shaft_origin_x=0.0,
                               shaft_position=(0.0, 0.0))
        shaft_2 = _make_shaft("S2", shaft_origin_x=50.0,
                               shaft_position=(0.0, 90.0))
        # x_gear_driven_local=100 → x_global=150 ≠ 200
        stage = GearStage("St", shaft_1, shaft_2, geo1, forces1,
                          x_gear_driver_local=200.0, x_gear_driven_local=100.0)
        errors = stage.validate()
        assert any("X-contact" in e for e in errors)


class TestGearStageValidate:
    def test_valid_stage_no_errors(self, geo1, forces1):
        shaft_1 = _make_shaft("S1", speed_rpm=1500.0,
                               shaft_position=(0.0, 0.0), shaft_origin_x=0.0)
        shaft_2 = _make_shaft("S2", speed_rpm=750.0,
                               shaft_position=(0.0, 90.0), shaft_origin_x=50.0)
        stage = GearStage("St", shaft_1, shaft_2, geo1, forces1,
                          x_gear_driver_local=200.0, x_gear_driven_local=150.0)
        assert stage.validate() == []

    def test_centre_distance_error(self, geo1, forces1):
        # shaft_2 at y=100 → geometric distance = 100 ≠ 90 (al)
        shaft_1 = _make_shaft("S1", shaft_position=(0.0, 0.0))
        shaft_2 = _make_shaft("S2", shaft_position=(0.0, 100.0))
        stage = GearStage("St", shaft_1, shaft_2, geo1, forces1)
        errors = stage.validate()
        assert any("Centre distance" in e for e in errors)

    def test_speed_inconsistency_error(self, geo1, forces1):
        # i=2 → expected n2=750rpm, but set 800
        shaft_1 = _make_shaft("S1", speed_rpm=1500.0,
                               shaft_position=(0.0, 0.0))
        shaft_2 = _make_shaft("S2", speed_rpm=800.0,
                               shaft_position=(0.0, 90.0))
        stage = GearStage("St", shaft_1, shaft_2, geo1, forces1,
                          x_gear_driver_local=200.0, x_gear_driven_local=150.0)
        errors = stage.validate()
        assert any("Speed" in e for e in errors)

    def test_same_shaft_error(self, geo1, forces1):
        shaft = _make_shaft("S1", shaft_position=(0.0, 0.0))
        stage = GearStage("St", shaft, shaft, geo1, forces1)
        errors = stage.validate()
        assert any("same object" in e for e in errors)

    def test_zero_speed_skips_speed_check(self, geo1, forces1):
        # Both speeds zero → speed check is skipped, only geometry matters
        shaft_1 = _make_shaft("S1", speed_rpm=0.0,
                               shaft_position=(0.0, 0.0), shaft_origin_x=0.0)
        shaft_2 = _make_shaft("S2", speed_rpm=0.0,
                               shaft_position=(0.0, 90.0), shaft_origin_x=50.0)
        stage = GearStage("St", shaft_1, shaft_2, geo1, forces1,
                          x_gear_driver_local=200.0, x_gear_driven_local=150.0)
        errors = stage.validate()
        # No speed errors expected when speeds are 0
        assert not any("Speed" in e for e in errors)


# ---------------------------------------------------------------------------
# GearSystem — topology
# ---------------------------------------------------------------------------

class TestGearSystemTopology:
    def test_input_output_shaft_single_stage(self, single_stage_system):
        gs = single_stage_system
        inp = gs.input_shaft()
        out = gs.output_shaft()
        assert inp is not None
        assert out is not None
        assert inp.name == "Shaft_1"
        assert out.name == "Shaft_2"

    def test_input_output_shaft_two_stage(self, two_stage_system):
        gs = two_stage_system
        assert gs.input_shaft().name == "Shaft_1"
        assert gs.output_shaft().name == "Shaft_3"

    def test_no_cycle_valid(self, single_stage_system):
        assert not single_stage_system._has_cycle()

    def test_cycle_detected(self, geo1, forces1):
        # Deliberately create a cycle: A→B→C→A
        shaft_a = _make_shaft("A", shaft_position=(0.0, 0.0))
        shaft_b = _make_shaft("B", shaft_position=(0.0, 90.0))
        shaft_c = _make_shaft("C", shaft_position=(0.0, 0.0))

        gs = GearSystem(name="Cycle_Test")
        gs.shafts = [shaft_a, shaft_b, shaft_c]
        gs.stages = [
            GearStage("AB", shaft_a, shaft_b, geo1, forces1),
            GearStage("BC", shaft_b, shaft_c, geo1, forces1),
            GearStage("CA", shaft_c, shaft_a, geo1, forces1),
        ]
        assert gs._has_cycle()

    def test_add_stage_auto_registers_shafts(self, geo1, forces1):
        shaft_1 = _make_shaft("S1", shaft_position=(0.0, 0.0))
        shaft_2 = _make_shaft("S2", shaft_position=(0.0, 90.0))
        gs = GearSystem(name="AutoReg")
        # Do NOT call add_shaft manually
        stage = GearStage("St", shaft_1, shaft_2, geo1, forces1)
        gs.add_stage(stage)
        assert shaft_1 in gs.shafts
        assert shaft_2 in gs.shafts

    def test_add_shaft_deduplication(self, geo1, forces1):
        shaft_1 = _make_shaft("S1", shaft_position=(0.0, 0.0))
        shaft_2 = _make_shaft("S2", shaft_position=(0.0, 90.0))
        gs = GearSystem(name="Dedup")
        gs.add_shaft(shaft_1)
        gs.add_shaft(shaft_1)   # duplicate — same object
        gs.add_shaft(shaft_2)
        assert sum(1 for sh in gs.shafts if sh is shaft_1) == 1


# ---------------------------------------------------------------------------
# GearSystem — speed propagation
# ---------------------------------------------------------------------------

class TestPropagateSpeedsOneSage:
    def test_propagate_single_stage(self, single_stage_system):
        gs = single_stage_system
        gs.propagate_speeds()
        shaft_2 = gs.output_shaft()
        assert shaft_2.speed_rpm == pytest.approx(750.0, rel=1e-6)

    def test_propagate_two_stage(self, two_stage_system):
        gs = two_stage_system
        gs.propagate_speeds()
        shafts = {sh.name: sh for sh in gs.shafts}
        assert shafts["Shaft_2"].speed_rpm == pytest.approx(750.0, rel=1e-6)
        assert shafts["Shaft_3"].speed_rpm == pytest.approx(375.0, rel=1e-6)

    def test_propagate_zero_speed_raises(self, geo1, forces1):
        shaft_1 = _make_shaft("S1", speed_rpm=0.0,
                               shaft_position=(0.0, 0.0))
        shaft_2 = _make_shaft("S2", shaft_position=(0.0, 90.0))
        gs = GearSystem(name="ZeroSpeed")
        gs.add_stage(GearStage("St", shaft_1, shaft_2, geo1, forces1))
        with pytest.raises(ValueError, match="speed_rpm must be > 0"):
            gs.propagate_speeds()

    def test_propagate_idempotent(self, two_stage_system):
        gs = two_stage_system
        gs.propagate_speeds()
        gs.propagate_speeds()   # second call must give same result
        shaft_3 = gs.output_shaft()
        assert shaft_3.speed_rpm == pytest.approx(375.0, rel=1e-6)


# ---------------------------------------------------------------------------
# GearSystem — validate
# ---------------------------------------------------------------------------

class TestGearSystemValidate:
    def test_valid_single_stage(self, single_stage_system):
        single_stage_system.propagate_speeds()
        errors = single_stage_system.validate()
        assert errors == [], f"Unexpected errors: {errors}"

    def test_valid_two_stage(self, two_stage_system):
        two_stage_system.propagate_speeds()
        errors = two_stage_system.validate()
        assert errors == [], f"Unexpected errors: {errors}"

    def test_empty_system_errors(self):
        gs = GearSystem(name="Empty")
        errors = gs.validate()
        assert any("≥ 2 shafts" in e for e in errors)
        assert any("no gear stages" in e for e in errors)

    def test_missing_bearing_error(self, geo1, forces1):
        # shaft with only 1 bearing
        shaft = Shaft(name="S_bad")
        shaft.add_section(ShaftSection(length=400.0, diameter=40.0))
        sys_bad = MechanicalSystem(shaft=shaft, name="S_bad",
                                   shaft_position=(0.0, 90.0))
        sys_bad.add_bearing(Bearing(position=20.0, C=35_000.0, C0=22_000.0,
                                    arrangement="fixed", label="A"))
        # only 1 bearing — should fail Phase 1 check
        shaft_1 = _make_shaft("S1", speed_rpm=1000.0,
                               shaft_position=(0.0, 0.0))
        gs = GearSystem(name="BadBearing")
        gs.add_stage(GearStage("St", shaft_1, sys_bad, geo1, forces1,
                               x_gear_driver_local=200.0,
                               x_gear_driven_local=150.0))
        errors = gs.validate()
        assert any("bearing" in e.lower() for e in errors)

    def test_cycle_in_validate(self, geo1, forces1):
        shaft_a = _make_shaft("A", shaft_position=(0.0, 0.0))
        shaft_b = _make_shaft("B", shaft_position=(0.0, 90.0))
        gs = GearSystem(name="CycleVal")
        gs.shafts = [shaft_a, shaft_b]
        gs.stages = [
            GearStage("AB", shaft_a, shaft_b, geo1, forces1),
            GearStage("BA", shaft_b, shaft_a, geo1, forces1),
        ]
        errors = gs.validate()
        assert any("cycle" in e.lower() for e in errors)

    def test_validate_or_raise(self, single_stage_system):
        # Before propagating speeds both shaft speeds are valid:
        # shaft_1=1500, shaft_2=0 — speed check skipped (n2=0)
        # Geometry must be valid — no raise expected
        single_stage_system.validate_or_raise()

    def test_validate_or_raise_bad_geometry(self, geo1, forces1):
        # shaft_2 at wrong distance → centre distance error
        shaft_1 = _make_shaft("S1", speed_rpm=1500.0,
                               shaft_position=(0.0, 0.0))
        shaft_2 = _make_shaft("S2", shaft_position=(0.0, 200.0))  # wrong
        gs = GearSystem(name="BadGeo")
        gs.add_stage(GearStage("St", shaft_1, shaft_2, geo1, forces1,
                               x_gear_driver_local=200.0,
                               x_gear_driven_local=150.0))
        with pytest.raises(ValueError):
            gs.validate_or_raise()


# ---------------------------------------------------------------------------
# GearSystem — summary / repr
# ---------------------------------------------------------------------------

class TestGearSystemSummary:
    def test_summary_contains_name(self, single_stage_system):
        s = single_stage_system.summary()
        assert "Single_Stage" in s

    def test_summary_total_ratio_single(self, single_stage_system):
        # i_total = 2.0 for single stage (z2/z1 = 40/20)
        s = single_stage_system.summary()
        assert "2.0000" in s

    def test_summary_total_ratio_two_stage(self, two_stage_system):
        # i_total = 2.0 × 2.0 = 4.0
        s = two_stage_system.summary()
        assert "4.0000" in s

    def test_repr(self, single_stage_system):
        r = repr(single_stage_system)
        assert "GearSystem" in r
        assert "Single_Stage" in r