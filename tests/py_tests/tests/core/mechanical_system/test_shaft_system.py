"""
tests/core/mechanical_system/test_shaft_system.py

Unit tests for
axisforge.core.mechanical_system.Parallel_Axis_systems.systems
    .spur_helicoidal_system.shaft_system

Covers GearElement and ShaftSystem: placement guards, sorted and
type-filtered accessors, axial extents, gear-mesh load replacement, and
validate().

Bearings and gears here are the lightweight stand-ins from conftest.py
(StubBearing / StubGear). ShaftSystem only ever reads .position, .b,
.label, .designation and .validate() off them, so a stub keeps these tests
about ShaftSystem's own bookkeeping. The real Bearing.assemble() path is
covered in test_bearings.py.

ASCII only.
"""

from __future__ import annotations

import pytest

from axisforge.core.machine_elements.Shaft.shaft import Shaft, ShaftSection
from axisforge.core.loads import (
    RadialLoad,
    AxialLoad,
    TorqueLoad,
    ExternalMoment,
    DistributedRadialLoad,
)
from axisforge.core.mechanical_system.Parallel_Axis_systems.systems.spur_helicoidal_system.shaft_system import (
    ShaftSystem,
    GearElement,
)


# ===========================================================================
# GearElement
# ===========================================================================

class TestGearElement:

    def test_position_delegates_to_the_gear(self, stub_gear_factory):
        ge = GearElement(stub_gear_factory(position=120.0), role="driver")
        assert ge.position == pytest.approx(120.0)

    def test_position_tracks_the_gear_it_wraps(self, stub_gear_factory):
        """No geometry is duplicated -- the wrapper is a view, not a copy."""
        gear = stub_gear_factory(position=120.0)
        ge = GearElement(gear, role="driven")
        gear.position = 200.0
        assert ge.position == pytest.approx(200.0)

    @pytest.mark.parametrize("role", ["driver", "driven"])
    def test_valid_roles(self, stub_gear_factory, role):
        ge = GearElement(stub_gear_factory(position=10.0), role=role)
        assert ge.validate() == []

    @pytest.mark.parametrize("role", ["source", "DRIVER", "", None])
    def test_invalid_role_reported(self, stub_gear_factory, role):
        ge = GearElement(stub_gear_factory(position=10.0), role=role, label="ge")
        assert any("role must be one of" in e for e in ge.validate())

    @pytest.mark.parametrize("rotation_dir", [1, -1, None])
    def test_valid_rotation_directions(self, stub_gear_factory, rotation_dir):
        ge = GearElement(stub_gear_factory(position=10.0), role="driver",
                         rotation_dir=rotation_dir)
        assert ge.validate() == []

    @pytest.mark.parametrize("rotation_dir", [0, 2, -2, "cw"])
    def test_invalid_rotation_direction_reported(self, stub_gear_factory, rotation_dir):
        ge = GearElement(stub_gear_factory(position=10.0), role="driver",
                         rotation_dir=rotation_dir)
        assert any("rotation_dir must be" in e for e in ge.validate())

    def test_rotation_dir_defaults_to_none(self, stub_gear_factory):
        """Only the kinematic source gear declares a sense; the rest inherit."""
        ge = GearElement(stub_gear_factory(position=10.0), role="driven")
        assert ge.rotation_dir is None

    def test_gear_errors_are_delegated_and_prefixed(self, stub_gear_factory):
        gear = stub_gear_factory(position=10.0, errors=["mn must be > 0"])
        ge = GearElement(gear, role="driver", label="ge1")
        errors = ge.validate()
        assert any(e.startswith("ge1.gear:") for e in errors)

    def test_a_gear_without_validate_is_tolerated(self):
        """Delegation is duck-typed: no validate(), no error."""
        class Bare:
            position = 10.0

        assert GearElement(Bare(), role="driver").validate() == []

    def test_validate_or_raise(self, stub_gear_factory):
        ge = GearElement(stub_gear_factory(position=10.0), role="bogus")
        with pytest.raises(ValueError, match="role must be one of"):
            ge.validate_or_raise()

    def test_repr_is_ascii(self, stub_gear_factory):
        ge = GearElement(stub_gear_factory(position=10.0), role="driver", label="ge")
        repr(ge).encode("ascii")


# ===========================================================================
# ShaftSystem -- construction and placement
# ===========================================================================

class TestShaftSystemConstruction:

    def test_defaults(self, uniform_shaft_300mm):
        sys = ShaftSystem(uniform_shaft_300mm)
        assert sys.name == "System_1"
        assert sys.speed_rpm == 0.0
        assert sys.shaft_position == (0.0, 0.0)
        assert sys.shaft_origin_x == 0.0
        assert sys.bearings == []
        assert sys.gears == []
        assert sys.loads == []

    def test_design_life_default_is_20000_hours(self, uniform_shaft_300mm):
        assert ShaftSystem(uniform_shaft_300mm).design_life_hours == pytest.approx(20_000.0)

    def test_adders_are_chainable(self, uniform_shaft_300mm, stub_bearing_factory):
        sys = ShaftSystem(uniform_shaft_300mm)
        returned = (
            sys.add_bearing(stub_bearing_factory(position=40.0, label="A"))
               .add_bearing(stub_bearing_factory(position=260.0, label="B"))
               .add_load(RadialLoad(position=150.0, magnitude=1000.0))
        )
        assert returned is sys
        assert len(sys.bearings) == 2
        assert len(sys.loads) == 1


class TestShaftSystemPlacementGuards:

    @pytest.mark.parametrize("position", [-1.0, 300.001, 1e6])
    def test_bearing_outside_the_shaft_rejected(self, uniform_shaft_300mm,
                                                stub_bearing_factory, position):
        sys = ShaftSystem(uniform_shaft_300mm, name="S1")
        with pytest.raises(ValueError, match="outside"):
            sys.add_bearing(stub_bearing_factory(position=position))

    @pytest.mark.parametrize("position", [0.0, 150.0, 300.0])
    def test_bearing_at_the_shaft_ends_accepted(self, uniform_shaft_300mm,
                                                stub_bearing_factory, position):
        """The admissible interval is closed: [0, L]."""
        sys = ShaftSystem(uniform_shaft_300mm)
        sys.add_bearing(stub_bearing_factory(position=position))
        assert len(sys.bearings) == 1

    def test_gear_outside_the_shaft_rejected(self, uniform_shaft_300mm,
                                             stub_gear_factory):
        sys = ShaftSystem(uniform_shaft_300mm)
        ge = GearElement(stub_gear_factory(position=400.0), role="driver")
        with pytest.raises(ValueError, match="outside"):
            sys.add_gear(ge)

    def test_point_load_outside_the_shaft_rejected(self, uniform_shaft_300mm):
        sys = ShaftSystem(uniform_shaft_300mm)
        with pytest.raises(ValueError, match="outside"):
            sys.add_load(RadialLoad(position=350.0, magnitude=100.0))

    def test_distributed_load_is_checked_on_its_span_not_its_midpoint(
        self, uniform_shaft_300mm
    ):
        """
        Span [280, 320] has a midpoint of 300 -- inside the shaft. It must
        still be rejected, because x_hi runs off the end.
        """
        sys = ShaftSystem(uniform_shaft_300mm)
        q = DistributedRadialLoad(x_lo=280.0, x_hi=320.0, magnitude=500.0, label="q")
        with pytest.raises(ValueError, match="outside"):
            sys.add_load(q)

    def test_distributed_load_inside_the_shaft_accepted(self, uniform_shaft_300mm):
        sys = ShaftSystem(uniform_shaft_300mm)
        sys.add_load(DistributedRadialLoad(x_lo=140.0, x_hi=160.0, magnitude=500.0))
        assert len(sys.distributed_radial_loads) == 1

    def test_the_error_names_the_system(self, uniform_shaft_300mm,
                                        stub_bearing_factory):
        sys = ShaftSystem(uniform_shaft_300mm, name="input_shaft")
        with pytest.raises(ValueError, match="input_shaft"):
            sys.add_bearing(stub_bearing_factory(position=-1.0))


# ===========================================================================
# ShaftSystem -- accessors
# ===========================================================================

class TestShaftSystemAccessors:

    @pytest.fixture
    def populated(self, uniform_shaft_300mm, stub_bearing_factory, stub_gear_factory):
        """Elements added out of axial order on purpose."""
        sys = ShaftSystem(uniform_shaft_300mm, name="S1")
        sys.add_bearing(stub_bearing_factory(position=260.0, b=14.0, label="B"))
        sys.add_bearing(stub_bearing_factory(position=40.0, b=14.0, label="A"))
        sys.add_gear(GearElement(stub_gear_factory(position=200.0, b=20.0), role="driven",
                                 label="g2"))
        sys.add_gear(GearElement(stub_gear_factory(position=100.0, b=20.0), role="driver",
                                 label="g1"))
        sys.add_load(TorqueLoad(position=280.0, magnitude=50.0, label="T"))
        sys.add_load(RadialLoad(position=150.0, magnitude=1000.0, label="F"))
        sys.add_load(AxialLoad(position=20.0, magnitude=200.0, label="Fa"))
        sys.add_load(ExternalMoment(position=250.0, magnitude=300.0, label="M"))
        sys.add_load(DistributedRadialLoad(x_lo=90.0, x_hi=110.0,
                                           magnitude=400.0, label="q"))
        return sys

    def test_bearings_are_sorted_by_position(self, populated):
        assert [b.label for b in populated.bearings] == ["A", "B"]

    def test_gears_are_sorted_by_position(self, populated):
        assert [g.label for g in populated.gears] == ["g1", "g2"]

    def test_loads_are_sorted_by_position(self, populated):
        positions = [ld.position for ld in populated.loads]
        assert positions == sorted(positions)

    def test_support_positions(self, populated):
        assert populated.support_positions == pytest.approx([40.0, 260.0])

    def test_the_accessors_return_copies_not_the_internal_lists(self, populated):
        """Mutating a returned list must not corrupt the system."""
        populated.bearings.clear()
        populated.gears.clear()
        populated.loads.clear()
        assert len(populated.bearings) == 2
        assert len(populated.gears) == 2
        assert len(populated.loads) == 5

    @pytest.mark.parametrize("accessor,label", [
        ("radial_loads", "F"),
        ("axial_loads", "Fa"),
        ("torque_loads", "T"),
        ("external_moments", "M"),
        ("distributed_radial_loads", "q"),
    ])
    def test_type_filtered_accessors(self, populated, accessor, label):
        found = getattr(populated, accessor)
        assert [ld.label for ld in found] == [label]

    def test_distributed_loads_are_not_reported_as_radial_point_loads(self, populated):
        """DistributedRadialLoad is a sibling of RadialLoad, not a subclass."""
        assert "q" not in [ld.label for ld in populated.radial_loads]

    def test_every_load_is_reachable_through_exactly_one_typed_accessor(self, populated):
        typed = (
            populated.radial_loads + populated.axial_loads +
            populated.torque_loads + populated.external_moments +
            populated.distributed_radial_loads
        )
        assert len(typed) == len(populated.loads)

    def test_snapshot_contents(self, populated):
        snap = populated.snapshot()
        assert snap["name"] == "S1"
        assert snap["total_length"] == pytest.approx(300.0)
        assert snap["shaft"] is populated.shaft
        assert len(snap["bearings"]) == 2
        assert snap["support_positions"] == pytest.approx([40.0, 260.0])

    def test_snapshot_containers_are_detached(self, populated, stub_bearing_factory):
        snap = populated.snapshot()
        populated.add_bearing(stub_bearing_factory(position=150.0, label="C"))
        assert len(snap["bearings"]) == 2

    def test_to_dict_is_an_alias_of_snapshot(self, populated):
        assert populated.to_dict.__func__ is populated.snapshot.__func__


# ===========================================================================
# ShaftSystem -- axial extents
# ===========================================================================

class TestShaftSystemExtents:

    def test_gear_extent_is_centred_on_the_position(self, uniform_shaft_300mm,
                                                    stub_gear_factory):
        sys = ShaftSystem(uniform_shaft_300mm)
        ge = GearElement(stub_gear_factory(position=100.0, b=20.0), role="driver")
        assert sys.gear_extent(ge) == pytest.approx((90.0, 110.0))

    def test_bearing_extent_is_centred_on_the_position(self, uniform_shaft_300mm,
                                                       stub_bearing_factory):
        sys = ShaftSystem(uniform_shaft_300mm)
        b = stub_bearing_factory(position=40.0, b=14.0)
        assert sys.bearing_extent(b) == pytest.approx((33.0, 47.0))

    def test_zero_width_collapses_to_a_point(self, uniform_shaft_300mm,
                                             stub_gear_factory,
                                             stub_bearing_factory):
        sys = ShaftSystem(uniform_shaft_300mm)
        ge = GearElement(stub_gear_factory(position=100.0, b=0.0), role="driver")
        b = stub_bearing_factory(position=40.0, b=0.0)
        assert sys.gear_extent(ge) == pytest.approx((100.0, 100.0))
        assert sys.bearing_extent(b) == pytest.approx((40.0, 40.0))

    def test_missing_width_attribute_is_treated_as_a_point(self, uniform_shaft_300mm):
        """Width is read with getattr(..., 0.0) -- an undimensioned gear is a point."""
        class BareGear:
            position = 100.0

        sys = ShaftSystem(uniform_shaft_300mm)
        ge = GearElement(BareGear(), role="driver")
        assert sys.gear_extent(ge) == pytest.approx((100.0, 100.0))


# ===========================================================================
# ShaftSystem -- gear-mesh load injection
# ===========================================================================

class TestSetGearLoads:

    @pytest.fixture
    def with_user_load(self, uniform_shaft_300mm):
        sys = ShaftSystem(uniform_shaft_300mm, name="S1")
        sys.add_load(RadialLoad(position=150.0, magnitude=1000.0, label="user_F"))
        return sys

    def test_gear_loads_are_added(self, with_user_load):
        with_user_load.set_gear_loads([
            RadialLoad(position=100.0, magnitude=500.0, label="Ft",
                       source="gear_mesh"),
        ])
        assert len(with_user_load.loads) == 2

    def test_user_loads_survive(self, with_user_load):
        with_user_load.set_gear_loads([
            RadialLoad(position=100.0, magnitude=500.0, source="gear_mesh"),
        ])
        assert "user_F" in [ld.label for ld in with_user_load.loads]

    def test_repeated_calls_do_not_accumulate(self, with_user_load):
        """Idempotent: re-resolving the gear system must not duplicate loads."""
        for _ in range(3):
            with_user_load.set_gear_loads([
                RadialLoad(position=100.0, magnitude=500.0, label="Ft",
                           source="gear_mesh"),
                RadialLoad(position=100.0, magnitude=200.0, label="Fr",
                           source="gear_mesh"),
            ])
        assert len(with_user_load.loads) == 3          # 1 user + 2 gear_mesh

    def test_a_second_call_replaces_the_previous_values(self, with_user_load):
        with_user_load.set_gear_loads([
            RadialLoad(position=100.0, magnitude=500.0, label="Ft",
                       source="gear_mesh"),
        ])
        with_user_load.set_gear_loads([
            RadialLoad(position=100.0, magnitude=900.0, label="Ft",
                       source="gear_mesh"),
        ])
        gear_loads = [ld for ld in with_user_load.loads if ld.source == "gear_mesh"]
        assert len(gear_loads) == 1
        assert gear_loads[0].magnitude == pytest.approx(900.0)

    def test_an_empty_list_clears_the_gear_loads(self, with_user_load):
        with_user_load.set_gear_loads([
            RadialLoad(position=100.0, magnitude=500.0, source="gear_mesh"),
        ])
        with_user_load.set_gear_loads([])
        assert [ld.label for ld in with_user_load.loads] == ["user_F"]

    def test_set_gear_loads_bypasses_the_bounds_check(self, with_user_load):
        """
        Documented behaviour, worth pinning: set_gear_loads() writes to
        _loads directly, so it does NOT run _check_axial_bounds(). The gear
        system is trusted to place its own mesh loads.
        """
        with_user_load.set_gear_loads([
            RadialLoad(position=9999.0, magnitude=1.0, source="gear_mesh"),
        ])
        assert len(with_user_load.loads) == 2

    def test_other_non_user_sources_are_preserved(self, with_user_load):
        """Only source == 'gear_mesh' is swept; bearing reactions stay."""
        with_user_load.add_load(
            RadialLoad(position=40.0, magnitude=300.0, label="R_A",
                       source="bearing_reaction")
        )
        with_user_load.set_gear_loads([
            RadialLoad(position=100.0, magnitude=500.0, source="gear_mesh"),
        ])
        labels = [ld.label for ld in with_user_load.loads]
        assert "R_A" in labels and "user_F" in labels


# ===========================================================================
# ShaftSystem -- validate
# ===========================================================================

class TestShaftSystemValidate:

    def test_a_determinate_shaft_needs_no_report_on_support_count(
        self, shaft_system_two_bearings
    ):
        errors = shaft_system_two_bearings.validate()
        assert not any("needs >= 2 bearings" in e for e in errors)

    @pytest.mark.parametrize("n_bearings", [0, 1])
    def test_under_supported_shaft_reported(self, uniform_shaft_300mm,
                                            stub_bearing_factory, n_bearings):
        sys = ShaftSystem(uniform_shaft_300mm, name="S1")
        for i in range(n_bearings):
            sys.add_bearing(stub_bearing_factory(position=40.0 + 100.0 * i,
                                                 b=14.0, label=f"B{i}"))
        assert any("needs >= 2 bearings" in e for e in sys.validate())

    def test_shaft_errors_are_delegated_and_prefixed(self, stub_bearing_factory):
        shaft = Shaft(label="bad")
        shaft.add_section(ShaftSection(length=300.0, diameter=50.0, label="body"))
        object.__setattr__(shaft.sections[0], "diameter", -1.0)

        sys = ShaftSystem(shaft, name="S1")
        assert any(e.startswith("S1.shaft:") for e in sys.validate())

    def test_bearing_errors_are_delegated_and_prefixed(self, uniform_shaft_300mm,
                                                       stub_bearing_factory):
        sys = ShaftSystem(uniform_shaft_300mm, name="S1")
        sys.add_bearing(stub_bearing_factory(position=40.0, b=14.0, label="A",
                                             errors=["C must be >= 0"]))
        sys.add_bearing(stub_bearing_factory(position=260.0, b=14.0, label="B"))
        assert any(e.startswith("S1.bearing:") for e in sys.validate())

    def test_gear_errors_are_delegated(self, shaft_system_two_bearings,
                                       stub_gear_factory):
        gear = stub_gear_factory(position=150.0, b=20.0, errors=["mn must be > 0"])
        shaft_system_two_bearings.add_gear(
            GearElement(gear, role="driver", label="g1")
        )
        assert any("mn must be > 0" in e for e in shaft_system_two_bearings.validate())

    def test_load_errors_are_delegated(self, shaft_system_two_bearings):
        bad = RadialLoad(position=150.0, magnitude=-100.0, label="F")
        shaft_system_two_bearings.add_load(bad)
        errors = shaft_system_two_bearings.validate()
        assert any("magnitude must be >= 0" in e for e in errors)

    def test_overlapping_elements_reported(self, uniform_shaft_300mm,
                                           stub_bearing_factory, stub_gear_factory):
        """Bearing [33, 47] and gear [40, 60] share axial space."""
        sys = ShaftSystem(uniform_shaft_300mm, name="S1")
        sys.add_bearing(stub_bearing_factory(position=40.0, b=14.0, label="A"))
        sys.add_bearing(stub_bearing_factory(position=260.0, b=14.0, label="B"))
        sys.add_gear(GearElement(stub_gear_factory(position=50.0, b=20.0),
                                 role="driver", label="g1"))
        errors = sys.validate()
        assert any("g1" in e and "A" in e for e in errors)

    def test_coincident_point_elements_reported(self, uniform_shaft_300mm,
                                                stub_bearing_factory,
                                                stub_gear_factory):
        """Two zero-width elements at the same x fall back to the separation guard."""
        sys = ShaftSystem(uniform_shaft_300mm, name="S1")
        sys.add_bearing(stub_bearing_factory(position=100.0, b=0.0, label="A"))
        sys.add_bearing(stub_bearing_factory(position=260.0, b=0.0, label="B"))
        sys.add_gear(GearElement(stub_gear_factory(position=100.0, b=0.0),
                                 role="driver", label="g1"))
        assert any("coincide" in e for e in sys.validate())

    def test_validate_or_raise_on_a_broken_system(self, uniform_shaft_300mm):
        sys = ShaftSystem(uniform_shaft_300mm, name="S1")
        with pytest.raises(ValueError):
            sys.validate_or_raise()

    def test_summary_and_repr_do_not_raise(self, shaft_system_two_bearings):
        assert "S1" in shaft_system_two_bearings.summary()
        assert "S1" in repr(shaft_system_two_bearings)
