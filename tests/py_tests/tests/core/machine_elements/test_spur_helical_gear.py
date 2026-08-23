"""
tests/core/machine_elements/test_spur_helical_gear.py

Unit tests for
axisforge.core.machine_elements.Gears.Parallel_Axis_gears.spur_helical_gear

Covers the reference geometry computed in __init__, the undercut check, and
validate().

References:
  ISO 53:2013     -- standard basic rack tooth profile (haP=1, cP=0.25, rfP=0.38)
  ISO 21771:2007  -- cylindrical involute gear geometry

Two of these tests are marked xfail: they encode the ISO 21771 tip diameter
and the standard whole depth, which the current implementation does not
reproduce (see tests/README.md, finding B). The companion regression tests
lock in what the code does today, so the pair together make the discrepancy
explicit instead of silently blessing it.

ASCII only.
"""

from __future__ import annotations

import math

import pytest

from axisforge.core.machine_elements.Gears.Parallel_Axis_gears.spur_helical_gear import (
    SpurHelicalGear,
)


# ===========================================================================
# Reference geometry -- spur (beta = 0)
# ===========================================================================

class TestSpurReferenceGeometry:
    """z=20, mn=2, alpha_n=20 deg, beta=0, x=0."""

    def test_transverse_module_equals_normal_module(self, spur_pinion_20t):
        assert spur_pinion_20t.mt == pytest.approx(2.0)

    def test_transverse_pressure_angle_equals_normal(self, spur_pinion_20t):
        assert spur_pinion_20t.alphat == pytest.approx(math.radians(20.0))

    def test_base_helix_angle_is_zero(self, spur_pinion_20t):
        assert spur_pinion_20t.betab == pytest.approx(0.0)

    def test_pitch_diameter(self, spur_pinion_20t):
        """d = mt * z = 2 * 20 = 40 mm."""
        assert spur_pinion_20t.d == pytest.approx(40.0)
        assert spur_pinion_20t.r == pytest.approx(20.0)

    def test_base_diameter(self, spur_pinion_20t):
        """db = d * cos(alphat)."""
        assert spur_pinion_20t.db == pytest.approx(40.0 * math.cos(math.radians(20.0)))
        assert spur_pinion_20t.rb == pytest.approx(spur_pinion_20t.db / 2.0)

    def test_transverse_pitch(self, spur_pinion_20t):
        assert spur_pinion_20t.pt == pytest.approx(math.pi * 2.0)

    def test_transverse_base_pitch(self, spur_pinion_20t):
        expected = math.pi * 2.0 * math.cos(math.radians(20.0))
        assert spur_pinion_20t.pbt == pytest.approx(expected)

    def test_base_pitch(self, spur_pinion_20t):
        """pb = pi * mn * cos(alpha_n); for a spur gear pb == pbt."""
        expected = math.pi * 2.0 * math.cos(math.radians(20.0))
        assert spur_pinion_20t.pb == pytest.approx(expected)

    def test_root_diameter(self, spur_pinion_20t):
        """df = d + 2*mn*(x - hfP), hfP = haP + cP = 1.25 -> 40 - 5 = 35 mm."""
        assert spur_pinion_20t.df == pytest.approx(35.0)
        assert spur_pinion_20t.rf == pytest.approx(17.5)

    def test_fillet_radius_from_the_basic_rack(self, spur_pinion_20t):
        """rhoF = rfP * mn = 0.38 * 2."""
        assert spur_pinion_20t.rhoF == pytest.approx(0.76)

    def test_equivalent_teeth_equals_z_for_a_spur_gear(self, spur_pinion_20t):
        assert spur_pinion_20t.zn == pytest.approx(20.0)

    def test_equivalent_diameter(self, spur_pinion_20t):
        """de = mn * zn."""
        assert spur_pinion_20t.de == pytest.approx(2.0 * spur_pinion_20t.zn)

    def test_tip_diameter_as_implemented(self, spur_pinion_20t):
        """
        REGRESSION LOCK, not an endorsement.
        Implemented as da = d + 2*mn*(haP + cP + x) = 40 + 2*2*1.25 = 45 mm.
        See the xfail below and tests/README.md, finding B.
        """
        assert spur_pinion_20t.da == pytest.approx(45.0)
        assert spur_pinion_20t.ra == pytest.approx(22.5)

    @pytest.mark.xfail(
        strict=True,
        reason="da includes the tip clearance cP; ISO 21771 gives "
               "da = d + 2*mn*(haP + x). See tests/README.md, finding B.",
    )
    def test_tip_diameter_iso21771(self, spur_pinion_20t):
        """da = d + 2*mn*(haP + x) = 40 + 4 = 44 mm for a standard x=0 gear."""
        assert spur_pinion_20t.da == pytest.approx(44.0)

    @pytest.mark.xfail(
        strict=True,
        reason="whole depth comes out at 2.5*mn instead of the ISO 53 2.25*mn, "
               "because cP is added at the tip as well as at the root. "
               "See tests/README.md, finding B.",
    )
    def test_whole_depth_is_2_25_mn(self, spur_pinion_20t):
        g = spur_pinion_20t
        assert (g.da - g.df) / 2.0 == pytest.approx(2.25 * g.mn)


# ===========================================================================
# Reference geometry -- helical
# ===========================================================================

class TestHelicalReferenceGeometry:
    """z=20, mn=2, alpha_n=20 deg, beta=15 deg."""

    def test_transverse_module_is_inflated_by_the_helix(self, helical_pinion_20t):
        assert helical_pinion_20t.mt == pytest.approx(2.0 / math.cos(math.radians(15.0)))

    def test_transverse_pressure_angle(self, helical_pinion_20t):
        expected = math.atan(math.tan(math.radians(20.0)) / math.cos(math.radians(15.0)))
        assert helical_pinion_20t.alphat == pytest.approx(expected)

    def test_transverse_pressure_angle_exceeds_the_normal_one(self, helical_pinion_20t):
        assert helical_pinion_20t.alphat > math.radians(20.0)

    def test_base_helix_angle(self, helical_pinion_20t):
        expected = math.asin(math.sin(math.radians(15.0)) * math.cos(math.radians(20.0)))
        assert helical_pinion_20t.betab == pytest.approx(expected)

    def test_base_helix_angle_is_below_the_reference_helix(self, helical_pinion_20t):
        assert 0.0 < helical_pinion_20t.betab < math.radians(15.0)

    def test_pitch_diameter_exceeds_the_spur_equivalent(self, helical_pinion_20t,
                                                        spur_pinion_20t):
        """Same mn and z: a helical gear is larger by 1/cos(beta)."""
        ratio = helical_pinion_20t.d / spur_pinion_20t.d
        assert ratio == pytest.approx(1.0 / math.cos(math.radians(15.0)))

    def test_virtual_tooth_count(self, helical_pinion_20t):
        """zn = z / cos^3(beta)."""
        expected = 20.0 / math.cos(math.radians(15.0)) ** 3
        assert helical_pinion_20t.zn == pytest.approx(expected)
        assert helical_pinion_20t.zn > 20.0

    def test_spur_limit_recovers_the_spur_gear(self):
        """beta -> 0 must reproduce the spur geometry exactly."""
        spur = SpurHelicalGear(mn=2.0, z=20, b=20.0, beta_n_deg=0.0)
        near = SpurHelicalGear(mn=2.0, z=20, b=20.0, beta_n_deg=1e-9)
        assert near.d == pytest.approx(spur.d)
        assert near.db == pytest.approx(spur.db)
        assert near.zn == pytest.approx(spur.zn)


# ===========================================================================
# Profile shift
# ===========================================================================

class TestProfileShift:

    def test_positive_shift_raises_both_tip_and_root(self):
        plain = SpurHelicalGear(mn=2.0, z=20, b=20.0, x=0.0)
        shifted = SpurHelicalGear(mn=2.0, z=20, b=20.0, x=0.5)
        assert shifted.da == pytest.approx(plain.da + 2 * 2.0 * 0.5)
        assert shifted.df == pytest.approx(plain.df + 2 * 2.0 * 0.5)

    def test_shift_leaves_the_pitch_and_base_circles_untouched(self):
        """Profile shift moves the rack, not the reference or base cylinder."""
        plain = SpurHelicalGear(mn=2.0, z=20, b=20.0, x=0.0)
        shifted = SpurHelicalGear(mn=2.0, z=20, b=20.0, x=0.5)
        assert shifted.d == pytest.approx(plain.d)
        assert shifted.db == pytest.approx(plain.db)

    def test_shift_preserves_the_whole_depth(self):
        plain = SpurHelicalGear(mn=2.0, z=20, b=20.0, x=0.0)
        shifted = SpurHelicalGear(mn=2.0, z=20, b=20.0, x=0.5)
        assert (shifted.da - shifted.df) == pytest.approx(plain.da - plain.df)


# ===========================================================================
# Undercutting
# ===========================================================================

class TestUndercutting:
    """
    z_min = 2*cos(beta)/sin^2(alphat) * (haP - x).
    For alpha_n = 20 deg, beta = 0, x = 0 this is the classic 17.1 teeth.
    """

    @pytest.mark.parametrize("z,undercut", [
        (12, True),
        (17, True),
        (18, False),
        (20, False),
        (40, False),
    ])
    def test_classic_seventeen_tooth_threshold(self, z, undercut):
        gear = SpurHelicalGear(mn=2.0, z=z, b=20.0)
        assert gear.undercutting() is undercut

    def test_positive_shift_cures_undercutting(self):
        assert SpurHelicalGear(mn=2.0, z=14, b=20.0, x=0.0).undercutting() is True
        assert SpurHelicalGear(mn=2.0, z=14, b=20.0, x=0.5).undercutting() is False

    def test_helix_angle_lowers_the_threshold(self):
        """A helical gear tolerates fewer teeth before undercutting."""
        assert SpurHelicalGear(mn=2.0, z=16, b=20.0, beta_n_deg=0.0).undercutting() is True
        assert SpurHelicalGear(mn=2.0, z=16, b=20.0, beta_n_deg=20.0).undercutting() is False

    def test_larger_pressure_angle_lowers_the_threshold(self):
        assert SpurHelicalGear(mn=2.0, z=15, b=20.0, alpha_n_deg=20.0).undercutting() is True
        assert SpurHelicalGear(mn=2.0, z=15, b=20.0, alpha_n_deg=25.0).undercutting() is False

    def test_threshold_is_independent_of_the_module(self):
        """Undercut is a tooth-count phenomenon; mn scales out of it."""
        for mn in (0.5, 2.0, 8.0):
            assert SpurHelicalGear(mn=mn, z=17, b=20.0).undercutting() is True
            assert SpurHelicalGear(mn=mn, z=18, b=20.0).undercutting() is False


# ===========================================================================
# Validation
# ===========================================================================

class TestSpurHelicalGearValidate:

    def test_clean_gear_validates(self, spur_pinion_20t):
        assert spur_pinion_20t.validate() == []
        spur_pinion_20t.validate_or_raise()

    def test_clean_helical_gear_validates(self, helical_pinion_20t):
        assert helical_pinion_20t.validate() == []

    @pytest.mark.parametrize("mn", [0.0, -1.0])
    def test_non_positive_module_reported(self, mn):
        errors = SpurHelicalGear(mn=mn, z=20, b=20.0).validate()
        assert any("mn must be > 0" in e for e in errors)

    def test_zero_teeth_reported(self):
        errors = SpurHelicalGear(mn=2.0, z=0, b=20.0).validate()
        assert any("z must be >= 1" in e for e in errors)

    def test_negative_face_width_reported(self):
        errors = SpurHelicalGear(mn=2.0, z=20, b=-1.0).validate()
        assert any("b must be >= 0" in e for e in errors)

    def test_zero_face_width_is_allowed(self):
        """b = 0 means 'not yet dimensioned', not an error."""
        errors = SpurHelicalGear(mn=2.0, z=20, b=0.0).validate()
        assert not any("b must be" in e for e in errors)

    # alpha_n = 0 makes sin(alphat) zero, so undercutting() divides by zero
    # inside the same validate() pass -- numpy returns inf and warns.
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @pytest.mark.parametrize("alpha_n_deg", [0.0, 45.0, 60.0, -5.0])
    def test_pressure_angle_out_of_range_reported(self, alpha_n_deg):
        errors = SpurHelicalGear(mn=2.0, z=20, b=20.0,
                                 alpha_n_deg=alpha_n_deg).validate()
        assert any("alpha_n_deg out of range" in e for e in errors)

    @pytest.mark.parametrize("beta_n_deg", [45.0, 60.0, -1.0])
    def test_helix_angle_out_of_range_reported(self, beta_n_deg):
        errors = SpurHelicalGear(mn=2.0, z=20, b=20.0,
                                 beta_n_deg=beta_n_deg).validate()
        assert any("beta_n_deg out of range" in e for e in errors)

    def test_zero_helix_angle_is_in_range(self):
        errors = SpurHelicalGear(mn=2.0, z=20, b=20.0, beta_n_deg=0.0).validate()
        assert not any("beta_n_deg" in e for e in errors)

    @pytest.mark.parametrize("field", ["Ra", "Rq", "Rz"])
    def test_negative_roughness_reported(self, field):
        gear = SpurHelicalGear(mn=2.0, z=20, b=20.0, **{field: -1.0})
        assert any(f"{field} must be >= 0" in e for e in gear.validate())

    def test_undercut_is_a_validation_error(self):
        errors = SpurHelicalGear(mn=2.0, z=12, b=20.0, label="tiny").validate()
        assert any("undercut" in e for e in errors)
        assert all(e.startswith("tiny:") for e in errors)

    def test_errors_are_tagged_with_the_class_name_when_unlabelled(self):
        errors = SpurHelicalGear(mn=2.0, z=12, b=20.0).validate()
        assert all(e.startswith("SpurHelicalGear:") for e in errors)

    def test_validate_or_raise_reports_every_error_at_once(self):
        gear = SpurHelicalGear(mn=-1.0, z=0, b=-1.0, label="broken")
        with pytest.raises(ValueError) as excinfo:
            gear.validate_or_raise()
        text = str(excinfo.value)
        assert "mn must be" in text
        assert "z must be" in text
        assert "b must be" in text


# ===========================================================================
# Metadata
# ===========================================================================

class TestSpurHelicalGearMetadata:

    def test_position_is_carried_but_not_used_by_the_geometry(self):
        at_zero = SpurHelicalGear(mn=2.0, z=20, b=20.0, position=0.0)
        at_150 = SpurHelicalGear(mn=2.0, z=20, b=20.0, position=150.0)
        assert at_150.position == pytest.approx(150.0)
        assert at_150.d == pytest.approx(at_zero.d)

    def test_default_basic_rack_is_iso53(self):
        g = SpurHelicalGear(mn=2.0, z=20, b=20.0)
        assert g.haP == pytest.approx(1.0)
        assert g.cP == pytest.approx(0.25)
        assert g.rfP == pytest.approx(0.38)
        assert g.hfP == pytest.approx(1.25)
        assert g.alpha_n_deg == pytest.approx(20.0)
