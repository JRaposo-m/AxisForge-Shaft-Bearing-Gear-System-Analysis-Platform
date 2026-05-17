"""
tests/test_core/test_system.py
Unit tests for MechanicalSystem — add/validate/integrate.
"""

import pytest
from core.shaft import Shaft, ShaftSection
from core.components import Bearing, GearElement
from core.loads import RadialLoad, AxialLoad, TorqueLoad, ExternalMoment, LoadPlane
from core.system import MechanicalSystem


def _minimal_shaft() -> Shaft:
    shaft = Shaft()
    shaft.add_section(ShaftSection(length=400.0, diameter=50.0))
    return shaft


def _valid_system() -> MechanicalSystem:
    system = MechanicalSystem(shaft=_minimal_shaft(), speed_rpm=1450.0)
    system.add_bearing(Bearing(position=50.0, C=35_000.0, C0=22_000.0,
                               arrangement="fixed", label="A"))
    system.add_bearing(Bearing(position=350.0, C=35_000.0, C0=22_000.0,
                               arrangement="floating", label="B"))
    return system


class TestMechanicalSystemCreation:

    def test_default_name(self):
        system = MechanicalSystem(shaft=_minimal_shaft())
        assert system.name == "System_1"

    def test_custom_name(self):
        system = MechanicalSystem(shaft=_minimal_shaft(), name="Test")
        assert system.name == "Test"

    def test_default_speed_zero(self):
        system = MechanicalSystem(shaft=_minimal_shaft())
        assert system.speed_rpm == 0.0

    def test_design_life_default(self):
        system = MechanicalSystem(shaft=_minimal_shaft())
        assert system.design_life_hours == 20_000.0


class TestAddComponents:

    def test_add_bearing(self):
        system = MechanicalSystem(shaft=_minimal_shaft())
        system.add_bearing(Bearing(position=50.0, C=35_000.0, C0=22_000.0, label="A"))
        assert len(system.bearings) == 1

    def test_add_bearing_outside_shaft_raises(self):
        system = MechanicalSystem(shaft=_minimal_shaft())
        with pytest.raises(ValueError, match="outside shaft"):
            system.add_bearing(Bearing(position=500.0, C=35_000.0, C0=22_000.0))

    def test_add_gear(self):
        system = _valid_system()
        system.add_gear(GearElement(position=200.0, tangential_force=3500.0, radial_force=1274.0))
        assert len(system.gears) == 1

    def test_add_gear_outside_shaft_raises(self):
        system = MechanicalSystem(shaft=_minimal_shaft())
        with pytest.raises(ValueError, match="outside shaft"):
            system.add_gear(GearElement(position=500.0, tangential_force=1000.0, radial_force=400.0))

    def test_add_load_radial(self):
        system = _valid_system()
        system.add_load(RadialLoad(position=200.0, magnitude=5000.0, plane=LoadPlane.XY))
        assert len(system.loads) == 1

    def test_add_multiple_loads(self):
        system = _valid_system()
        system.add_load(RadialLoad(position=100.0, magnitude=1000.0, plane=LoadPlane.XZ))
        system.add_load(TorqueLoad(position=200.0, magnitude=50_000.0))
        assert len(system.loads) == 2


class TestSortedAccessors:

    def test_bearings_sorted_by_position(self):
        system = MechanicalSystem(shaft=_minimal_shaft())
        system.add_bearing(Bearing(position=350.0, C=35_000.0, C0=22_000.0, label="B"))
        system.add_bearing(Bearing(position=50.0, C=35_000.0, C0=22_000.0, label="A"))
        positions = [b.position for b in system.bearings]
        assert positions == sorted(positions)

    def test_gears_sorted_by_position(self):
        system = _valid_system()
        system.add_gear(GearElement(position=300.0, tangential_force=1000.0, radial_force=400.0, label="G2"))
        system.add_gear(GearElement(position=100.0, tangential_force=2000.0, radial_force=700.0, label="G1"))
        positions = [g.position for g in system.gears]
        assert positions == sorted(positions)

    def test_support_positions_sorted(self):
        system = _valid_system()
        assert system.support_positions == [50.0, 350.0]


class TestLoadFilters:

    def test_radial_loads_xz(self):
        system = _valid_system()
        system.add_load(RadialLoad(position=100.0, magnitude=1000.0, plane=LoadPlane.XZ))
        system.add_load(RadialLoad(position=200.0, magnitude=2000.0, plane=LoadPlane.XY))
        assert len(system.radial_loads_xz) == 1
        assert system.radial_loads_xz[0].plane == LoadPlane.XZ

    def test_radial_loads_xy(self):
        system = _valid_system()
        system.add_load(RadialLoad(position=200.0, magnitude=3000.0, plane=LoadPlane.XY))
        assert len(system.radial_loads_xy) == 1

    def test_axial_loads(self):
        system = _valid_system()
        system.add_load(AxialLoad(position=50.0, magnitude=500.0))
        assert len(system.axial_loads) == 1

    def test_torque_loads(self):
        system = _valid_system()
        system.add_load(TorqueLoad(position=200.0, magnitude=100_000.0))
        assert len(system.torque_loads) == 1

    def test_external_moments_filter(self):
        """external_moments property must return only ExternalMoment loads."""
        system = _valid_system()
        system.add_load(ExternalMoment(
            position=200.0, magnitude=50_000.0, plane=LoadPlane.XY, label="M1"
        ))
        system.add_load(RadialLoad(
            position=100.0, magnitude=1000.0, plane=LoadPlane.XY
        ))
        moments = system.external_moments
        assert len(moments) == 1
        assert moments[0].label == "M1"

    def test_external_moments_empty_when_none_added(self):
        system = _valid_system()
        assert system.external_moments == []


class TestValidation:

    def test_valid_system_no_errors(self):
        system = _valid_system()
        assert system.validate() == []

    def test_validate_or_raise_passes(self):
        system = _valid_system()
        system.validate_or_raise()

    def test_single_bearing_fails(self):
        system = MechanicalSystem(shaft=_minimal_shaft())
        system.add_bearing(Bearing(position=50.0, C=35_000.0, C0=22_000.0))
        errors = system.validate()
        assert any("≥ 2" in e for e in errors)

    def test_negative_speed_fails(self):
        system = _valid_system()
        system.speed_rpm = -100.0
        errors = system.validate()
        assert any("speed_rpm" in e for e in errors)

    def test_collocated_bearings_flagged(self):
        system = MechanicalSystem(shaft=_minimal_shaft())
        system.add_bearing(Bearing(position=100.0, C=35_000.0, C0=22_000.0, label="A"))
        system.add_bearing(Bearing(position=100.0, C=35_000.0, C0=22_000.0, label="B"))
        errors = system.validate()
        assert any("same location" in e for e in errors)

    def test_validate_or_raise_raises_for_single_bearing(self):
        system = MechanicalSystem(shaft=_minimal_shaft())
        system.add_bearing(Bearing(position=50.0, C=35_000.0, C0=22_000.0))
        with pytest.raises(ValueError, match="validation failed"):
            system.validate_or_raise()

    def test_empty_shaft_propagates_to_system_validate(self):
        system = MechanicalSystem(shaft=Shaft())
        errors = system.validate()
        assert any("no sections" in e for e in errors)

    def test_validate_bearing_outside_shaft_detected(self):
        """
        Bearing added via _bearings directly (bypassing add_bearing guard)
        at a position beyond shaft length must be caught by validate().
        """
        system = _valid_system()
        # Bypass add_bearing to plant an out-of-range bearing
        out_of_range = Bearing(position=10.0, C=35_000.0, C0=22_000.0, label="OUT")
        object.__setattr__(out_of_range, 'position', 999.0)  # force invalid position
        system._bearings.append(out_of_range)
        errors = system.validate()
        assert any("outside shaft" in e for e in errors)

    def test_validate_gear_outside_shaft_detected(self):
        """
        GearElement planted beyond shaft length must be caught by validate().
        """
        system = _valid_system()
        out_of_range = GearElement(
            position=200.0, tangential_force=1000.0, radial_force=400.0, label="GOUT"
        )
        object.__setattr__(out_of_range, 'position', 999.0)
        system._gears.append(out_of_range)
        errors = system.validate()
        assert any("outside shaft" in e for e in errors)


class TestSummary:

    def test_summary_contains_name(self):
        system = _valid_system()
        assert "System_1" in system.summary()

    def test_repr_contains_speed(self):
        system = _valid_system()
        assert "1450" in repr(system)