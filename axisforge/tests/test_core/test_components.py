"""
tests/test_core/test_components.py
Unit tests for Bearing, BearingType, GearElement.
"""

import math
import pytest
from core.components import Bearing, BearingType, GearElement


class TestBearing:

    def test_valid_fixed_bearing(self):
        b = Bearing(
            position=50.0, C=35_000.0, C0=22_000.0,
            arrangement="fixed", label="A"
        )
        assert b.position == 50.0
        assert b.C == 35_000.0
        assert b.arrangement == "fixed"

    def test_valid_floating_bearing(self):
        b = Bearing(position=350.0, C=35_000.0, C0=22_000.0, arrangement="floating")
        assert b.arrangement == "floating"

    def test_has_catalogue_data_true(self):
        b = Bearing(position=50.0, C=35_000.0, C0=22_000.0)
        assert b.has_catalogue_data is True

    def test_has_catalogue_data_false_c_zero(self):
        b = Bearing(position=50.0, C=0.0, C0=22_000.0)
        assert b.has_catalogue_data is False

    def test_has_catalogue_data_false_c0_zero(self):
        b = Bearing(position=50.0, C=35_000.0, C0=0.0)
        assert b.has_catalogue_data is False

    def test_raises_negative_position(self):
        with pytest.raises(ValueError, match="negative"):
            Bearing(position=-1.0, C=35_000.0, C0=22_000.0)

    def test_raises_negative_C(self):
        with pytest.raises(ValueError, match="C must"):
            Bearing(position=50.0, C=-1.0, C0=22_000.0)

    def test_raises_negative_C0(self):
        with pytest.raises(ValueError, match="C0 must"):
            Bearing(position=50.0, C=35_000.0, C0=-1.0)

    def test_raises_invalid_arrangement(self):
        with pytest.raises(ValueError, match="arrangement"):
            Bearing(position=50.0, C=35_000.0, C0=22_000.0, arrangement="locked")

    def test_raises_invalid_contact_angle_negative(self):
        with pytest.raises(ValueError, match="contact_angle"):
            Bearing(position=50.0, C=35_000.0, C0=22_000.0, contact_angle=-1.0)

    def test_raises_contact_angle_90(self):
        with pytest.raises(ValueError, match="contact_angle"):
            Bearing(position=50.0, C=35_000.0, C0=22_000.0, contact_angle=90.0)

    def test_life_exponent_ball(self):
        b = Bearing(position=50.0, C=35_000.0, C0=22_000.0,
                    bearing_type=BearingType.DEEP_GROOVE_BALL)
        assert b.life_exponent == 3.0

    def test_life_exponent_angular_contact(self):
        b = Bearing(position=50.0, C=35_000.0, C0=22_000.0,
                    bearing_type=BearingType.ANGULAR_CONTACT_BALL)
        assert b.life_exponent == 3.0

    def test_life_exponent_cylindrical_roller(self):
        b = Bearing(position=50.0, C=35_000.0, C0=22_000.0,
                    bearing_type=BearingType.CYLINDRICAL_ROLLER)
        assert pytest.approx(b.life_exponent, rel=1e-9) == 10.0 / 3.0

    def test_life_exponent_taper_roller(self):
        b = Bearing(position=50.0, C=35_000.0, C0=22_000.0,
                    bearing_type=BearingType.TAPER_ROLLER)
        assert pytest.approx(b.life_exponent, rel=1e-9) == 10.0 / 3.0

    def test_life_exponent_spherical_roller(self):
        b = Bearing(position=50.0, C=35_000.0, C0=22_000.0,
                    bearing_type=BearingType.SPHERICAL_ROLLER)
        assert pytest.approx(b.life_exponent, rel=1e-9) == 10.0 / 3.0

    def test_default_type_is_dgbb(self):
        b = Bearing(position=50.0)
        assert b.bearing_type == BearingType.DEEP_GROOVE_BALL

    def test_raises_X_zero(self):
        with pytest.raises(ValueError, match="X"):
            Bearing(position=50.0, C=35_000.0, C0=22_000.0, X=0.0)

    def test_raises_X_greater_than_1(self):
        with pytest.raises(ValueError, match="X"):
            Bearing(position=50.0, C=35_000.0, C0=22_000.0, X=1.1)

    def test_raises_Y_negative(self):
        with pytest.raises(ValueError, match="Y"):
            Bearing(position=50.0, C=35_000.0, C0=22_000.0, Y=-0.1)

    def test_repr_contains_label(self):
        b = Bearing(position=50.0, C=35_000.0, C0=22_000.0, label="A")
        assert "A" in repr(b)


class TestGearElement:

    def test_valid_spur_gear(self):
        g = GearElement(
            position=200.0, tangential_force=3500.0, radial_force=1274.0,
            pitch_diameter=100.0, torque=175_000.0
        )
        assert g.tangential_force == 3500.0
        assert g.torque == 175_000.0

    def test_torque_computed_if_zero(self):
        g = GearElement(
            position=200.0, tangential_force=3500.0, radial_force=1274.0,
            pitch_diameter=100.0
        )
        assert pytest.approx(g.torque, rel=1e-9) == 3500.0 * 50.0

    def test_torque_consistency_check_within_2pct_passes(self):
        T = 175_000.0 * 1.009
        g = GearElement(
            position=200.0, tangential_force=3500.0, radial_force=1274.0,
            pitch_diameter=100.0, torque=T
        )
        assert g.torque == T

    def test_torque_consistency_raises_beyond_2pct(self):
        T = 175_000.0 * 1.03
        with pytest.raises(ValueError, match="inconsistency"):
            GearElement(
                position=200.0, tangential_force=3500.0, radial_force=1274.0,
                pitch_diameter=100.0, torque=T
            )

    def test_raises_negative_position(self):
        with pytest.raises(ValueError, match="negative"):
            GearElement(position=-1.0, tangential_force=1000.0, radial_force=500.0)

    def test_raises_negative_tangential_force(self):
        with pytest.raises(ValueError, match="tangential"):
            GearElement(position=100.0, tangential_force=-100.0, radial_force=50.0)

    def test_raises_negative_radial_force(self):
        with pytest.raises(ValueError, match="radial"):
            GearElement(position=100.0, tangential_force=100.0, radial_force=-50.0)

    def test_raises_negative_axial_force(self):
        with pytest.raises(ValueError, match="axial"):
            GearElement(position=100.0, tangential_force=100.0,
                        radial_force=50.0, axial_force=-10.0)

    def test_raises_negative_pitch_diameter(self):
        """pitch_diameter cannot be negative — line 187 in components.py."""
        with pytest.raises(ValueError, match="pitch_diameter"):
            GearElement(position=100.0, tangential_force=1000.0,
                        radial_force=400.0, pitch_diameter=-10.0)

    def test_raises_invalid_pressure_angle(self):
        with pytest.raises(ValueError, match="pressure_angle"):
            GearElement(position=100.0, tangential_force=1000.0,
                        radial_force=400.0, pressure_angle=95.0)

    def test_raises_invalid_helix_angle(self):
        with pytest.raises(ValueError, match="helix_angle"):
            GearElement(position=100.0, tangential_force=1000.0,
                        radial_force=400.0, helix_angle=-5.0)

    def test_is_helical_false_for_spur(self):
        g = GearElement(position=100.0, tangential_force=1000.0, radial_force=400.0)
        assert g.is_helical is False

    def test_is_helical_true_for_helical(self):
        g = GearElement(
            position=100.0, tangential_force=1000.0,
            radial_force=400.0, helix_angle=20.0
        )
        assert g.is_helical is True

    def test_resultant_transverse_force(self):
        g = GearElement(position=200.0, tangential_force=3.0, radial_force=4.0)
        assert pytest.approx(g.resultant_transverse_force, rel=1e-9) == 5.0

    def test_no_pitch_diameter_torque_stays_zero(self):
        g = GearElement(position=100.0, tangential_force=1000.0, radial_force=400.0)
        assert g.torque == 0.0

    def test_repr_contains_label(self):
        g = GearElement(position=200.0, tangential_force=3500.0,
                        radial_force=1274.0, label="G1")
        assert "G1" in repr(g)