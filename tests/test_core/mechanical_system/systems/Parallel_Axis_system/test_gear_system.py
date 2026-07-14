"""Tests for GearSystem / GearMeshLink."""

from __future__ import annotations
import math
import pytest

from axisforge.core.loads import RadialLoad, AxialLoad, TorqueLoad
from axisforge.core.mechanical_system.systems.Parallel_Axis_system.shaft_system import GearElement, ShaftSystem
from axisforge.core.mechanical_system.systems.Parallel_Axis_system.gear_system import GearMeshLink, GearSystem
from tests.conftest import FakeShaft, FakeBearing, FakeMeshing, FakeGear, SpurLikeMeshing


def bearings(s):
    s.add_bearing(FakeBearing(10.0, arrangement="fixed", Ka=1e5))
    s.add_bearing(FakeBearing(190.0, arrangement="floating"))
    return s


def shaft(name, L=200.0):
    return bearings(ShaftSystem(FakeShaft(L), name=name, speed_rpm=1000.0))


# --------------------------------------------------------------------------
# topology enforcement
# --------------------------------------------------------------------------

def test_single_source_enforced_two_sources():
    a, b, c = shaft("A"), shaft("B"), shaft("C")
    ga = GearElement(FakeGear(100), "driver")
    gb = GearElement(FakeGear(100), "driven")
    # b<-a, and c is a second isolated source
    link = GearMeshLink(a, ga, b, gb, FakeMeshing(), phi_deg=0.0)
    gs = GearSystem([a, b, c], [link])
    errs = gs.validate()
    assert any("source shafts" in e for e in errs)


def test_zero_source_is_flagged_via_cycle():
    a, b = shaft("A"), shaft("B")
    g1, g2, g3, g4 = (GearElement(FakeGear(100), "driver"),
                      GearElement(FakeGear(100), "driven"),
                      GearElement(FakeGear(120), "driver"),
                      GearElement(FakeGear(120), "driven"))
    l1 = GearMeshLink(a, g1, b, g2, FakeMeshing(), 0.0)
    l2 = GearMeshLink(b, g3, a, g4, FakeMeshing(), 0.0)   # b -> a closes a cycle
    gs = GearSystem([a, b], [l1, l2])
    errs = gs.validate()
    assert any("cycle" in e for e in errs)
    assert any("no source" in e for e in errs)


def test_merge_rejected():
    a, b, c = shaft("A"), shaft("B"), shaft("C")
    l1 = GearMeshLink(a, GearElement(FakeGear(100), "driver"),
                      c, GearElement(FakeGear(100), "driven"), FakeMeshing(), 0.0)
    l2 = GearMeshLink(b, GearElement(FakeGear(120), "driver"),
                      c, GearElement(FakeGear(120), "driven"), FakeMeshing(), 0.0)
    gs = GearSystem([a, b, c], [l1, l2])
    errs = gs.validate()
    assert any("incoming links" in e for e in errs)


def test_mode_A_torque_split_must_sum_to_one():
    a, b, c = shaft("A"), shaft("B"), shaft("C")
    driver = GearElement(FakeGear(100), "driver")           # shared driver
    l1 = GearMeshLink(a, driver, b, GearElement(FakeGear(100), "driven"),
                      FakeMeshing(), 0.0, torque_split=0.5)
    l2 = GearMeshLink(a, driver, c, GearElement(FakeGear(100), "driven"),
                      FakeMeshing(), 90.0, torque_split=0.3)   # sums to 0.8
    gs = GearSystem([a, b, c], [l1, l2])
    errs = gs.validate()
    assert any("sums" in e for e in errs)


def test_mode_A_missing_split_on_shared_driver():
    a, b, c = shaft("A"), shaft("B"), shaft("C")
    driver = GearElement(FakeGear(100), "driver")
    l1 = GearMeshLink(a, driver, b, GearElement(FakeGear(100), "driven"),
                      FakeMeshing(), 0.0, torque_split=0.5)
    l2 = GearMeshLink(a, driver, c, GearElement(FakeGear(100), "driven"),
                      FakeMeshing(), 90.0, torque_split=None)   # inconsistent
    gs = GearSystem([a, b, c], [l1, l2])
    errs = gs.validate()
    assert any("must set torque_split" in e for e in errs)


# --------------------------------------------------------------------------
# resolve: propagation arithmetic
# --------------------------------------------------------------------------

def test_two_shaft_position_and_torque_propagation():
    a, b = shaft("A"), shaft("B")
    mesh = FakeMeshing(al=120.0, u=2.0)
    link = GearMeshLink(a, GearElement(FakeGear(100), "driver"),
                        b, GearElement(FakeGear(100), "driven"),
                        mesh, phi_deg=30.0)
    gs = GearSystem([a, b], [link])

    P, rpm = 5000.0, 1500.0
    gs.resolve(P, rpm, rotation_dir_source=1, source_position=(0.0, 0.0))

    omega = rpm * 2 * math.pi / 60.0
    T_source = P / omega
    # position: b = a + al*(cos, sin)
    assert b.shaft_position[0] == pytest.approx(120.0 * math.cos(math.radians(30.0)))
    assert b.shaft_position[1] == pytest.approx(120.0 * math.sin(math.radians(30.0)))
    # rotation inverts on external mesh
    assert gs._resolved[id(b)]["rotation_dir"] == -1
    # T_out on b = T_source * u
    assert gs._resolved[id(b)]["T_out_Nm"] == pytest.approx(T_source * 2.0)


def test_three_shaft_linear_chain_positions():
    a, b, c = shaft("A"), shaft("B"), shaft("C")
    m1 = FakeMeshing(al=100.0, u=2.0)
    m2 = FakeMeshing(al=80.0, u=1.5)
    l1 = GearMeshLink(a, GearElement(FakeGear(100), "driver"),
                      b, GearElement(FakeGear(100), "driven"), m1, phi_deg=0.0)
    l2 = GearMeshLink(b, GearElement(FakeGear(120), "driver"),
                      c, GearElement(FakeGear(120), "driven"), m2, phi_deg=90.0)
    gs = GearSystem([a, b, c], [l1, l2])
    gs.resolve(3000.0, 1000.0, rotation_dir_source=1, source_position=(5.0, 5.0))

    # b = a + (100, 0); c = b + (0, 80)
    assert b.shaft_position == pytest.approx((105.0, 5.0))
    assert c.shaft_position[0] == pytest.approx(105.0)
    assert c.shaft_position[1] == pytest.approx(85.0)
    # rotation alternates 1 -> -1 -> +1
    assert gs._resolved[id(b)]["rotation_dir"] == -1
    assert gs._resolved[id(c)]["rotation_dir"] == 1


def test_mode_B_uses_full_torque():
    a, b = shaft("A"), shaft("B")
    mesh = FakeMeshing(al=100.0, u=3.0)
    link = GearMeshLink(a, GearElement(FakeGear(100), "driver"),
                        b, GearElement(FakeGear(100), "driven"), mesh, 0.0)
    gs = GearSystem([a, b], [link])
    gs.resolve(4000.0, 1200.0, rotation_dir_source=1)
    omega = 1200.0 * 2 * math.pi / 60.0
    T_source = 4000.0 / omega
    # mode B: forces() called with the full source torque
    assert mesh.calls[0][0] == pytest.approx(T_source)


def test_mode_A_fanout_torque_distribution():
    a, b, c = shaft("A"), shaft("B"), shaft("C")
    driver = GearElement(FakeGear(100), "driver")
    m1 = FakeMeshing(al=100.0, u=2.0)
    m2 = FakeMeshing(al=100.0, u=2.0)
    l1 = GearMeshLink(a, driver, b, GearElement(FakeGear(100), "driven"),
                      m1, 0.0, torque_split=0.6)
    l2 = GearMeshLink(a, driver, c, GearElement(FakeGear(100), "driven"),
                      m2, 180.0, torque_split=0.4)
    gs = GearSystem([a, b, c], [l1, l2])
    gs.resolve(6000.0, 1000.0, rotation_dir_source=1)

    omega = 1000.0 * 2 * math.pi / 60.0
    T_source = 6000.0 / omega
    assert m1.calls[0][0] == pytest.approx(T_source * 0.6)
    assert m2.calls[0][0] == pytest.approx(T_source * 0.4)
    # each branch's output torque scales with its share
    assert gs._resolved[id(b)]["T_out_Nm"] == pytest.approx(T_source * 0.6 * 2.0)
    assert gs._resolved[id(c)]["T_out_Nm"] == pytest.approx(T_source * 0.4 * 2.0)


# --------------------------------------------------------------------------
# load injection onto shafts
# --------------------------------------------------------------------------

def test_loads_injected_on_both_shafts():
    a, b = shaft("A"), shaft("B")
    mesh = FakeMeshing(al=100.0, u=2.0, Ft=1000.0, Fr=364.0)
    link = GearMeshLink(a, GearElement(FakeGear(100), "driver", label="p"),
                        b, GearElement(FakeGear(120), "driven", label="w"),
                        mesh, phi_deg=0.0)
    gs = GearSystem([a, b], [link])
    gs.resolve(3000.0, 1000.0, rotation_dir_source=1)

    # driver shaft: 2 RadialLoad (Ft, Fr) + 1 TorqueLoad
    a_radial = a.radial_loads
    assert len(a_radial) == 2
    assert all(l.source == "gear_mesh" for l in a_radial)
    assert len(a.torque_loads) == 1
    assert len(a.axial_loads) == 0                     # spur-like: no Fa
    # driven shaft likewise
    assert len(b.radial_loads) == 2 and len(b.torque_loads) == 1


def test_helical_axial_load_injected_with_sign():
    a, b = shaft("A"), shaft("B")
    mesh = FakeMeshing(al=100.0, u=2.0, Ft=1000.0, Fr=364.0, Fa=250.0)
    link = GearMeshLink(a, GearElement(FakeGear(100), "driver"),
                        b, GearElement(FakeGear(120), "driven"),
                        mesh, phi_deg=0.0)
    gs = GearSystem([a, b], [link])
    gs.resolve(3000.0, 1000.0, rotation_dir_source=1)
    # placeholder sign: +Fa on driver, -Fa on driven
    assert a.axial_loads[0].magnitude == pytest.approx(+250.0)
    assert b.axial_loads[0].magnitude == pytest.approx(-250.0)


def test_resolve_is_idempotent():
    a, b = shaft("A"), shaft("B")
    a.add_load(RadialLoad(50.0, 999.0, 0.0, label="user"))    # user load survives
    mesh = FakeMeshing(al=100.0, u=2.0)
    link = GearMeshLink(a, GearElement(FakeGear(100), "driver"),
                        b, GearElement(FakeGear(120), "driven"), mesh, 0.0)
    gs = GearSystem([a, b], [link])
    gs.resolve(3000.0, 1000.0, rotation_dir_source=1)
    n_after_first = len(a.loads)
    gs.resolve(3000.0, 1000.0, rotation_dir_source=1)         # resolve again
    assert len(a.loads) == n_after_first                       # no accumulation
    assert sum(1 for l in a.loads if l.source == "user") == 1  # user kept


def test_resolve_rejects_bad_rpm():
    a, b = shaft("A"), shaft("B")
    link = GearMeshLink(a, GearElement(FakeGear(100), "driver"),
                        b, GearElement(FakeGear(120), "driven"), FakeMeshing(), 0.0)
    gs = GearSystem([a, b], [link])
    with pytest.raises(ValueError):
        gs.resolve(3000.0, 0.0, rotation_dir_source=1)


# --------------------------------------------------------------------------
# regression: 2-stage spur reducer vs hand calc (Shigley-style)
# --------------------------------------------------------------------------

def test_two_stage_spur_reducer_regression():
    """
    Stage 1: pinion z1=17, gear z2=51 (u=3), mn=2.5 mm, alpha=20°, x=0.
             standard centre dist a = mn/2 (z1+z2) = 85 mm; rl1 = mn*z1/2 = 21.25 mm.
    Stage 2: pinion z3=20, gear z4=40 (u=2), mn=3 mm; rl3 = 30 mm; a = 90 mm.
    Input: P = 10 kW at 1500 rpm, rotation_dir_source = +1.
    """
    a20 = math.radians(20.0)

    sA, sB, sC = shaft("A", 250.0), shaft("B", 250.0), shaft("C", 250.0)
    m1 = SpurLikeMeshing(rl1_mm=21.25, alphaw_rad=a20, u=3.0, al=85.0)
    m2 = SpurLikeMeshing(rl1_mm=30.0, alphaw_rad=a20, u=2.0, al=90.0)

    l1 = GearMeshLink(sA, GearElement(FakeGear(120), "driver"),
                      sB, GearElement(FakeGear(120), "driven"), m1, phi_deg=0.0)
    l2 = GearMeshLink(sB, GearElement(FakeGear(140), "driver"),
                      sC, GearElement(FakeGear(140), "driven"), m2, phi_deg=0.0)
    gs = GearSystem([sA, sB, sC], [l1, l2])

    P, rpm = 10_000.0, 1500.0
    gs.resolve(P, rpm, rotation_dir_source=1)

    omega = rpm * 2 * math.pi / 60.0
    T1 = P / omega                       # ~63.66 N·m
    # torque chain
    assert gs._resolved[id(sB)]["T_out_Nm"] == pytest.approx(T1 * 3.0)
    assert gs._resolved[id(sC)]["T_out_Nm"] == pytest.approx(T1 * 3.0 * 2.0)
    # rotation alternation: +1 -> -1 -> +1
    assert gs._resolved[id(sB)]["rotation_dir"] == -1
    assert gs._resolved[id(sC)]["rotation_dir"] == 1

    # stage-1 tangential/radial magnitudes on the driver, hand-checked
    Ft1 = T1 / (21.25 / 1000.0)
    Fr1 = Ft1 * math.tan(a20)
    a_radials = {round(l.theta_deg): l.magnitude for l in sA.radial_loads}
    # Fr at theta=0 (phi), Ft at theta=90 (phi + 90*rot, rot=+1)
    assert a_radials[0] == pytest.approx(Fr1, rel=1e-9)
    assert a_radials[90] == pytest.approx(Ft1, rel=1e-9)

    # torque stored in N·mm on the driver mesh point (= +T1 * 1000)
    assert sA.torque_loads[0].magnitude == pytest.approx(T1 * 1000.0)
