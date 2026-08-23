"""
tests/core/machine_elements/test_shaft.py

Unit tests for axisforge.core.machine_elements.Shaft.shaft

Covers: Shoulder, ShaftSection, Keyway, Shaft.

Reference values are closed-form (I = pi/64*(d^4 - di^4), etc.) or come
from DIN 6885 / ISO 3912 as shipped in axisforge/database/shaft/keyway/.

ASCII only.
"""

from __future__ import annotations

import math

import pytest

from axisforge.config import TOL_GEOMETRY_mm
from axisforge.core.machine_elements.Shaft.shaft import (
    Shaft,
    ShaftSection,
    Shoulder,
    Keyway,
    KeywayType,
)


# ===========================================================================
# Shoulder
# ===========================================================================

class TestShoulder:

    def test_ratios(self, shoulder_50_40):
        assert shoulder_50_40.r_over_d == pytest.approx(2.0 / 40.0)
        assert shoulder_50_40.D_over_d == pytest.approx(50.0 / 40.0)

    def test_clean_shoulder_validates(self, shoulder_50_40):
        assert shoulder_50_40.validate() == []

    def test_zero_fillet_rejected(self):
        with pytest.raises(ValueError, match="fillet_radius"):
            Shoulder(fillet_radius=0.0, diameter_large=50.0, diameter_small=40.0)

    def test_negative_fillet_rejected(self):
        with pytest.raises(ValueError, match="fillet_radius"):
            Shoulder(fillet_radius=-1.0, diameter_large=50.0, diameter_small=40.0)

    def test_inverted_diameters_rejected(self):
        with pytest.raises(ValueError, match="diameter_large"):
            Shoulder(fillet_radius=1.0, diameter_large=40.0, diameter_small=50.0)

    def test_equal_diameters_rejected(self):
        """A zero step is not a shoulder."""
        with pytest.raises(ValueError, match="diameter_large"):
            Shoulder(fillet_radius=1.0, diameter_large=40.0, diameter_small=40.0)

    def test_fillet_exceeding_step_height_rejected(self):
        """Step height = (50 - 40)/2 = 5 mm; r = 6 mm cannot fit."""
        with pytest.raises(ValueError, match="step height"):
            Shoulder(fillet_radius=6.0, diameter_large=50.0, diameter_small=40.0)

    def test_fillet_exactly_at_step_height_accepted(self):
        """r == step height is the limiting admissible case, not an error."""
        s = Shoulder(fillet_radius=5.0, diameter_large=50.0, diameter_small=40.0)
        assert s.fillet_radius == pytest.approx(5.0)


# ===========================================================================
# ShaftSection
# ===========================================================================

class TestShaftSectionProperties:

    def test_solid_section_defaults(self, solid_section_40mm):
        s = solid_section_40mm
        assert s.inner_diameter == 0.0
        assert s.material_id == "S355"
        assert s.is_hollow is False
        assert s.radius == pytest.approx(20.0)

    def test_solid_area(self, solid_section_40mm):
        expected = math.pi / 4.0 * 40.0 ** 2
        assert solid_section_40mm.area == pytest.approx(expected)

    def test_solid_second_moment(self, solid_section_40mm):
        expected = math.pi / 64.0 * 40.0 ** 4
        assert solid_section_40mm.second_moment_of_area == pytest.approx(expected)

    def test_polar_moment_is_twice_second_moment(self, solid_section_40mm):
        s = solid_section_40mm
        assert s.polar_moment == pytest.approx(2.0 * s.second_moment_of_area)

    def test_section_modulus(self, solid_section_40mm):
        s = solid_section_40mm
        assert s.section_modulus == pytest.approx(s.second_moment_of_area / 20.0)

    def test_polar_section_modulus_is_twice_section_modulus(self, solid_section_40mm):
        s = solid_section_40mm
        assert s.polar_section_modulus == pytest.approx(2.0 * s.section_modulus)

    def test_hollow_flag_and_area(self, hollow_section):
        s = hollow_section
        assert s.is_hollow is True
        expected = math.pi / 4.0 * (60.0 ** 2 - 30.0 ** 2)
        assert s.area == pytest.approx(expected)

    def test_hollow_second_moment(self, hollow_section):
        expected = math.pi / 64.0 * (60.0 ** 4 - 30.0 ** 4)
        assert hollow_section.second_moment_of_area == pytest.approx(expected)


class TestShaftSectionConstruction:

    @pytest.mark.parametrize("length", [0.0, -1.0])
    def test_non_positive_length_rejected(self, length):
        with pytest.raises(ValueError, match="length"):
            ShaftSection(length=length, diameter=40.0)

    @pytest.mark.parametrize("diameter", [0.0, -5.0])
    def test_non_positive_diameter_rejected(self, diameter):
        with pytest.raises(ValueError, match="diameter"):
            ShaftSection(length=100.0, diameter=diameter)

    def test_negative_inner_diameter_rejected(self):
        with pytest.raises(ValueError, match="inner_diameter"):
            ShaftSection(length=100.0, diameter=40.0, inner_diameter=-1.0)

    def test_inner_diameter_not_below_outer_rejected(self):
        with pytest.raises(ValueError, match="inner_diameter"):
            ShaftSection(length=100.0, diameter=40.0, inner_diameter=40.0)

    def test_non_positive_surface_finish_rejected(self):
        with pytest.raises(ValueError, match="surface_finish_ra"):
            ShaftSection(length=100.0, diameter=40.0, surface_finish_ra=0.0)

    def test_keyway_deeper_than_radius_rejected(self):
        deep = Keyway(label="kw", z_position=10.0, width=5.0, depth=25.0, length=30.0)
        with pytest.raises(ValueError, match="depth"):
            ShaftSection(length=100.0, diameter=40.0, keyways=[deep])


class TestShaftSectionValidate:

    def test_clean_section_has_no_errors(self, solid_section_40mm):
        assert solid_section_40mm.validate() == []

    def test_validate_delegates_to_shoulders(self, shoulder_50_40):
        """A shoulder corrupted after construction is reported, prefixed."""
        section = ShaftSection(length=100.0, diameter=50.0,
                               shoulder_right=shoulder_50_40)
        object.__setattr__(section.shoulder_right, "fillet_radius", -1.0)

        errors = section.validate()
        assert any(e.startswith("shoulder_right:") for e in errors)

    def test_validate_delegates_to_keyways(self, keyway_parallel):
        section = ShaftSection(length=200.0, diameter=50.0, keyways=[keyway_parallel])
        assert section.validate() == []


# ===========================================================================
# Keyway
# ===========================================================================

class TestKeywayConstruction:

    def test_empty_label_rejected(self):
        with pytest.raises(ValueError, match="label"):
            Keyway(label="", z_position=10.0, width=5.0, depth=3.0, length=20.0)

    def test_negative_z_position_rejected(self):
        with pytest.raises(ValueError, match="z_position"):
            Keyway(label="kw", z_position=-1.0, width=5.0, depth=3.0, length=20.0)

    @pytest.mark.parametrize("field,value", [
        ("width", 0.0),
        ("depth", 0.0),
        ("length", 0.0),
    ])
    def test_non_positive_dimensions_rejected(self, field, value):
        kwargs = dict(label="kw", z_position=10.0, width=5.0, depth=3.0, length=20.0)
        kwargs[field] = value
        with pytest.raises(ValueError, match=field):
            Keyway(**kwargs)

    @pytest.mark.parametrize("theta", [-1.0, 360.0, 400.0])
    def test_theta_outside_half_open_circle_rejected(self, theta):
        """theta must be in [0, 360) -- it is not normalised, it is rejected."""
        with pytest.raises(ValueError, match="theta"):
            Keyway(label="kw", z_position=10.0, width=5.0, depth=3.0,
                   length=20.0, theta=theta)

    def test_defaults(self, keyway_parallel):
        assert keyway_parallel.keyway_type is KeywayType.PARALLEL
        assert keyway_parallel.theta == 0.0
        assert keyway_parallel.Kf_bending is None


class TestKeywayFromStandard:
    """Validation against the DIN 6885 / ISO 3912 tables shipped in database/."""

    def test_parallel_requires_length(self):
        with pytest.raises(ValueError, match="length is required"):
            Keyway.from_standard(label="kw", shaft_diameter=40.0, z_position=100.0)

    def test_parallel_din6885_row_for_d40(self):
        """
        DIN 6885 row 38 < d <= 44: b = 12 mm, t_shaft = 5.0 mm,
        L_min = 28 mm, L_max = 140 mm.
        """
        kw = Keyway.from_standard(
            label="kw", shaft_diameter=40.0, z_position=100.0, length=40.0
        )
        assert kw.width == pytest.approx(12.0)
        assert kw.depth == pytest.approx(5.0)
        assert kw.depth_min == pytest.approx(4.8)
        assert kw.depth_max == pytest.approx(5.0)
        assert kw.length == pytest.approx(40.0)

    def test_parallel_row_boundary_is_upper_inclusive(self):
        """Row selection is d_min < d <= d_max, so d=38 belongs to the row below."""
        at_38 = Keyway.from_standard(label="a", shaft_diameter=38.0,
                                     z_position=0.0, length=40.0)
        at_39 = Keyway.from_standard(label="b", shaft_diameter=39.0,
                                     z_position=0.0, length=40.0)
        assert at_38.width == pytest.approx(10.0)   # 30 < 38 <= 38
        assert at_39.width == pytest.approx(12.0)   # 38 < 39 <= 44

    def test_shaft_diameter_outside_table_rejected(self):
        with pytest.raises(ValueError, match="outside DIN 6885 table range"):
            Keyway.from_standard(label="kw", shaft_diameter=500.0,
                                 z_position=0.0, length=40.0)

    def test_length_below_standard_minimum_flagged_by_validate(self):
        """Construction succeeds; validate() reports the out-of-range length."""
        kw = Keyway.from_standard(
            label="kw", shaft_diameter=40.0, z_position=100.0, length=5.0
        )
        errors = kw.validate()
        assert any("L_min" in e for e in errors)

    def test_woodruff_length_comes_from_the_table(self):
        """
        WOODRUFF ignores the caller's `length` -- it takes D (the disc
        diameter) from ISO 3912. The diameter under test is derived from the
        shipped table so this stays valid if the CSV is extended.
        """
        from axisforge.database.shaft.keyway.Woodruff_key.iso3912 import _TABLES

        row = _TABLES[1][0]
        d = (row["d_min"] + row["d_max"]) / 2.0

        kw = Keyway.from_standard(
            label="kw", shaft_diameter=d, z_position=50.0,
            keyway_type=KeywayType.WOODRUFF,
        )
        assert kw.keyway_type is KeywayType.WOODRUFF
        assert kw.width == pytest.approx(row["b"])
        assert kw.depth == pytest.approx(row["t_shaft"])
        assert kw.length == pytest.approx(row["D"])

    def test_woodruff_rejects_unknown_series(self):
        with pytest.raises(ValueError, match="series must be 1 or 2"):
            Keyway.from_standard(
                label="kw", shaft_diameter=20.0, z_position=0.0,
                keyway_type=KeywayType.WOODRUFF, series=3,
            )

    def test_spline_not_implemented(self):
        with pytest.raises(NotImplementedError):
            Keyway.from_standard(label="kw", shaft_diameter=40.0, z_position=0.0,
                                 keyway_type=KeywayType.SPLINE, length=40.0)


# ===========================================================================
# Shaft -- aggregate geometry
# ===========================================================================

class TestShaftAssembly:

    def test_empty_shaft(self):
        shaft = Shaft(label="empty")
        assert shaft.n_sections == 0
        assert shaft.total_length == 0.0
        assert len(shaft) == 0

    def test_add_section_rejects_wrong_type(self):
        shaft = Shaft(label="s")
        with pytest.raises(TypeError, match="ShaftSection"):
            shaft.add_section("not a section")

    def test_total_length_and_count(self, stepped_shaft_400mm):
        assert stepped_shaft_400mm.total_length == pytest.approx(400.0)
        assert stepped_shaft_400mm.n_sections == 3
        assert len(stepped_shaft_400mm) == 3

    def test_repr_is_ascii(self, stepped_shaft_400mm):
        text = repr(stepped_shaft_400mm)
        assert "stepped" in text
        text.encode("ascii")   # raises UnicodeEncodeError if not ASCII


class TestShaftAxialAddressing:

    @pytest.mark.parametrize("index,start,end", [
        (0, 0.0, 100.0),
        (1, 100.0, 300.0),
        (2, 300.0, 400.0),
    ])
    def test_axial_faces(self, stepped_shaft_400mm, index, start, end):
        assert stepped_shaft_400mm.axial_start(index) == pytest.approx(start)
        assert stepped_shaft_400mm.axial_end(index) == pytest.approx(end)

    @pytest.mark.parametrize("index", [-1, 3, 99])
    def test_out_of_range_index_raises(self, stepped_shaft_400mm, index):
        with pytest.raises(IndexError):
            stepped_shaft_400mm.axial_start(index)
        with pytest.raises(IndexError):
            stepped_shaft_400mm.axial_end(index)

    @pytest.mark.parametrize("z,expected_index", [
        (0.0, 0),
        (50.0, 0),
        (99.999, 0),
        (100.0, 1),     # interior boundary belongs to the RIGHT section
        (200.0, 1),
        (300.0, 2),
        (400.0, 2),     # right end clamps into the last section
    ])
    def test_section_at(self, stepped_shaft_400mm, z, expected_index):
        _, index = stepped_shaft_400mm.section_at(z)
        assert index == expected_index

    def test_section_at_returns_the_section_object(self, stepped_shaft_400mm):
        section, index = stepped_shaft_400mm.section_at(150.0)
        assert section is stepped_shaft_400mm.sections[index]
        assert section.label == "body"

    @pytest.mark.parametrize("z", [-1.0, 400.001, 1e6])
    def test_section_at_outside_extent_raises(self, stepped_shaft_400mm, z):
        with pytest.raises(ValueError, match="outside shaft extent"):
            stepped_shaft_400mm.section_at(z)

    def test_section_at_tolerates_geometric_epsilon(self, stepped_shaft_400mm):
        """Queries within TOL_GEOMETRY_mm of the ends are clamped, not rejected."""
        _, i_lo = stepped_shaft_400mm.section_at(-TOL_GEOMETRY_mm / 2.0)
        _, i_hi = stepped_shaft_400mm.section_at(400.0 + TOL_GEOMETRY_mm / 2.0)
        assert i_lo == 0
        assert i_hi == 2


class TestShaftSectionPropertyLookups:

    @pytest.mark.parametrize("z,diameter", [
        (50.0, 40.0),
        (200.0, 50.0),
        (350.0, 40.0),
    ])
    def test_diameter_at(self, stepped_shaft_400mm, z, diameter):
        assert stepped_shaft_400mm.diameter_at(z) == pytest.approx(diameter)

    def test_property_lookups_agree_with_the_owning_section(self, stepped_shaft_400mm):
        z = 200.0
        section, _ = stepped_shaft_400mm.section_at(z)
        assert stepped_shaft_400mm.I_at(z) == pytest.approx(section.second_moment_of_area)
        assert stepped_shaft_400mm.J_at(z) == pytest.approx(section.polar_moment)
        assert stepped_shaft_400mm.W_at(z) == pytest.approx(section.section_modulus)
        assert stepped_shaft_400mm.Wt_at(z) == pytest.approx(section.polar_section_modulus)

    def test_property_step_across_a_boundary(self, stepped_shaft_400mm):
        """I jumps at x=100 because the boundary belongs to the right section."""
        assert stepped_shaft_400mm.I_at(99.0) < stepped_shaft_400mm.I_at(100.0)


class TestShaftShoulders:

    def test_shoulders_reports_right_shoulders_at_their_absolute_position(
        self, stepped_shaft_400mm
    ):
        """
        shoulders() reports section.shoulder_right only, keyed by the
        section's right face -- here, the body's right face at x=300.
        """
        found = stepped_shaft_400mm.shoulders()
        assert len(found) == 1
        x, shoulder = found[0]
        assert x == pytest.approx(300.0)
        assert shoulder.diameter_large == pytest.approx(50.0)

    def test_shoulders_empty_when_none_declared(self, uniform_shaft_300mm):
        assert uniform_shaft_300mm.shoulders() == []


class TestShaftValidate:

    def test_clean_shaft_validates(self, stepped_shaft_400mm):
        assert stepped_shaft_400mm.validate() == []
        stepped_shaft_400mm.validate_or_raise()   # must not raise

    def test_empty_shaft_reports_and_short_circuits(self):
        errors = Shaft(label="empty").validate()
        assert errors == ["Shaft has no sections"]

    def test_section_errors_are_prefixed_with_index_and_label(self, uniform_shaft_300mm):
        bad = uniform_shaft_300mm.sections[0]
        object.__setattr__(bad, "inner_diameter", 999.0)

        errors = uniform_shaft_300mm.validate()
        assert any(e.startswith("Section 0 ('body')") for e in errors)

    def test_right_shoulder_mismatch_detected(self, shaft_with_right_shoulder_mismatch):
        errors = shaft_with_right_shoulder_mismatch.validate()
        assert any("shoulder_right.diameter_small" in e for e in errors)
        assert any("Shoulder mismatch at boundary 0/1" in e for e in errors)

    def test_left_shoulder_mismatch_detected(self, shaft_with_left_shoulder_mismatch):
        errors = shaft_with_left_shoulder_mismatch.validate()
        assert any("shoulder_left.diameter_small" in e for e in errors)

    def test_validate_or_raise_reports_every_error(self, shaft_with_right_shoulder_mismatch):
        with pytest.raises(ValueError, match="Shaft validation failed"):
            shaft_with_right_shoulder_mismatch.validate_or_raise()

    def test_single_section_shaft_has_no_boundary_checks(self, uniform_shaft_300mm):
        assert uniform_shaft_300mm.validate() == []
