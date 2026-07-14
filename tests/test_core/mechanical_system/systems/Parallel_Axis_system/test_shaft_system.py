"""Tests for ShaftSystem / GearElement."""

from __future__ import annotations
import pytest

from mechanical_system.loads import RadialLoad, AxialLoad, TorqueLoad
from mechanical_system.systems.shaft_system import GearElement, ShaftSystem
from tests.conftest import FakeShaft, FakeBearing, FakeGear


def make_shaft(L=200.0):
    return ShaftSystem(FakeShaft(L), name="S", speed_rpm=1000.0)


# --------------------------------------------------------------------------
# bounds validation on add_*
# --------------------------------------------------------------------------

def test_add_bearing_out_of_bounds_raises():
    s = make_shaft(200.0)
    with pytest.raises(ValueError):
        s.add_bearing(FakeBearing(250.0))          # beyond L
    with pytest.raises(ValueError):
        s.add_bearing(FakeBearing(-1.0))           # negative


def test_add_gear_and_load_in_bounds_ok():
    s = make_shaft(200.0)
    s.add_gear(GearElement(FakeGear(100.0), role="driver"))
    s.add_load(RadialLoad(50.0, 500.0, 0.0))
    assert len(s.gears) == 1 and len(s.loads) == 1


def test_add_load_out_of_bounds_raises():
    s = make_shaft(100.0)
    with pytest.raises(ValueError):
        s.add_load(RadialLoad(150.0, 10.0))


# --------------------------------------------------------------------------
# set_gear_loads: replace semantics + idempotency
# --------------------------------------------------------------------------

def test_set_gear_loads_preserves_user_replaces_mesh():
    s = make_shaft(200.0)
    s.add_load(RadialLoad(10.0, 100.0, 0.0, label="user"))   # source="user"

    mesh1 = [RadialLoad(100.0, 200.0, 0.0, source="gear_mesh")]
    s.set_gear_loads(mesh1)
    assert len(s.loads) == 2
    assert sum(1 for l in s.loads if l.source == "user") == 1
    assert sum(1 for l in s.loads if l.source == "gear_mesh") == 1

    # re-resolve with different mesh loads -> old gear_mesh gone, user kept
    mesh2 = [RadialLoad(120.0, 300.0, 0.0, source="gear_mesh"),
             RadialLoad(120.0, 50.0, 90.0, source="gear_mesh")]
    s.set_gear_loads(mesh2)
    assert sum(1 for l in s.loads if l.source == "gear_mesh") == 2
    assert sum(1 for l in s.loads if l.source == "user") == 1


def test_set_gear_loads_idempotent_empty_clears_mesh():
    s = make_shaft()
    s.set_gear_loads([RadialLoad(10.0, 1.0, source="gear_mesh")])
    s.set_gear_loads([])                       # clears
    assert len(s.loads) == 0


# --------------------------------------------------------------------------
# type-filtered accessors are sorted and correctly typed
# --------------------------------------------------------------------------

def test_type_filtered_accessors():
    s = make_shaft()
    s.add_load(TorqueLoad(30.0, 5.0))
    s.add_load(RadialLoad(10.0, 5.0, 0.0))
    s.add_load(AxialLoad(20.0, 5.0))
    assert [l.position for l in s.loads] == [10.0, 20.0, 30.0]
    assert len(s.radial_loads) == 1
    assert len(s.axial_loads) == 1
    assert len(s.torque_loads) == 1


# --------------------------------------------------------------------------
# validate: >=2 bearings, coincidence guard, axial reaction path
# --------------------------------------------------------------------------

def test_validate_requires_two_bearings():
    s = make_shaft()
    s.add_bearing(FakeBearing(20.0, arrangement="floating"))
    errs = s.validate()
    assert any(">= 2 bearings" in e for e in errs)


def test_validate_flags_coincident_bearings():
    s = make_shaft()
    s.add_bearing(FakeBearing(20.0))
    s.add_bearing(FakeBearing(20.0 + 1e-4))    # < 1e-3 apart
    errs = s.validate()
    assert any("within" in e for e in errs)


def test_axial_gear_mesh_requires_fixed_bearing_with_Ka():
    s = make_shaft()
    s.add_bearing(FakeBearing(10.0, arrangement="floating"))
    s.add_bearing(FakeBearing(190.0, arrangement="floating"))
    s.set_gear_loads([AxialLoad(100.0, 250.0, source="gear_mesh")])
    errs = s.validate()
    assert any("axial reaction path" in e for e in errs)


def test_axial_gear_mesh_satisfied_by_fixed_bearing():
    s = make_shaft()
    s.add_bearing(FakeBearing(10.0, arrangement="fixed", Ka=1.0e5))
    s.add_bearing(FakeBearing(190.0, arrangement="floating"))
    s.set_gear_loads([AxialLoad(100.0, 250.0, source="gear_mesh")])
    errs = s.validate()
    assert not any("axial reaction path" in e for e in errs)


def test_user_axial_load_does_not_trigger_axial_check():
    # only gear_mesh axial loads require the fixed-bearing path
    s = make_shaft()
    s.add_bearing(FakeBearing(10.0, arrangement="floating"))
    s.add_bearing(FakeBearing(190.0, arrangement="floating"))
    s.add_load(AxialLoad(100.0, 250.0))        # source="user"
    errs = s.validate()
    assert not any("axial reaction path" in e for e in errs)


def test_gearelement_position_delegates():
    g = GearElement(FakeGear(77.0), role="driven")
    assert g.position == 77.0


def test_gearelement_bad_role_and_rotdir():
    g = GearElement(FakeGear(10.0), role="nonsense", rotation_dir=3)
    errs = g.validate()
    assert any("role" in e for e in errs)
    assert any("rotation_dir" in e for e in errs)
