"""
tests/test_core/test_loads.py
Unit tests for RadialLoad, AxialLoad, TorqueLoad, ExternalMoment, LoadPlane,
and LoadingProfile.
"""

import math
import pytest
from core.loads import (
    RadialLoad, AxialLoad, TorqueLoad, ExternalMoment, LoadPlane,
    LoadingProfile,
)


class TestLoadPlane:
    def test_enum_values_exist(self):
        assert LoadPlane.XZ
        assert LoadPlane.XY
        assert LoadPlane.AXIAL

    def test_xz_xy_are_different(self):
        assert LoadPlane.XZ != LoadPlane.XY


class TestRadialLoad:

    def test_valid_creation_xz(self):
        l = RadialLoad(position=100.0, magnitude=5000.0, plane=LoadPlane.XZ)
        assert l.position == 100.0
        assert l.magnitude == 5000.0
        assert l.plane == LoadPlane.XZ

    def test_valid_creation_xy(self):
        l = RadialLoad(position=200.0, magnitude=3000.0, plane=LoadPlane.XY, label="F1")
        assert l.label == "F1"

    def test_raises_negative_position(self):
        with pytest.raises(ValueError, match="negative"):
            RadialLoad(position=-1.0, magnitude=1000.0, plane=LoadPlane.XY)

    def test_raises_axial_plane(self):
        with pytest.raises(ValueError, match="AXIAL"):
            RadialLoad(position=100.0, magnitude=1000.0, plane=LoadPlane.AXIAL)

    def test_zero_position_valid(self):
        l = RadialLoad(position=0.0, magnitude=1000.0, plane=LoadPlane.XZ)
        assert l.position == 0.0

    def test_zero_magnitude_valid(self):
        """Zero load is valid — solver will sum it without effect."""
        l = RadialLoad(position=100.0, magnitude=0.0, plane=LoadPlane.XY)
        assert l.magnitude == 0.0

    def test_negative_magnitude_valid(self):
        """Negative magnitude is valid — direction is sign-encoded."""
        l = RadialLoad(position=100.0, magnitude=-500.0, plane=LoadPlane.XZ)
        assert l.magnitude == -500.0


class TestAxialLoad:

    def test_valid_creation(self):
        l = AxialLoad(position=100.0, magnitude=2000.0)
        assert l.position == 100.0
        assert l.magnitude == 2000.0

    def test_raises_negative_position(self):
        with pytest.raises(ValueError, match="negative"):
            AxialLoad(position=-10.0, magnitude=1000.0)

    def test_positive_negative_magnitude_valid(self):
        l = AxialLoad(position=50.0, magnitude=-1000.0)
        assert l.magnitude == -1000.0

    def test_default_label(self):
        l = AxialLoad(position=0.0, magnitude=0.0)
        assert l.label == ""


class TestTorqueLoad:

    def test_valid_creation(self):
        l = TorqueLoad(position=200.0, magnitude=150_000.0)
        assert l.magnitude == 150_000.0

    def test_raises_negative_position(self):
        with pytest.raises(ValueError, match="negative"):
            TorqueLoad(position=-5.0, magnitude=100.0)

    def test_negative_torque_valid(self):
        """Negative torque = opposite rotation direction."""
        l = TorqueLoad(position=100.0, magnitude=-50_000.0)
        assert l.magnitude == -50_000.0

    def test_zero_torque_valid(self):
        l = TorqueLoad(position=0.0, magnitude=0.0)
        assert l.magnitude == 0.0


class TestExternalMoment:

    def test_valid_creation_xz(self):
        m = ExternalMoment(position=50.0, magnitude=10_000.0, plane=LoadPlane.XZ)
        assert m.plane == LoadPlane.XZ

    def test_valid_creation_xy(self):
        m = ExternalMoment(position=100.0, magnitude=5_000.0, plane=LoadPlane.XY)
        assert m.magnitude == 5_000.0

    def test_raises_negative_position(self):
        with pytest.raises(ValueError, match="negative"):
            ExternalMoment(position=-1.0, magnitude=1000.0, plane=LoadPlane.XY)

    def test_raises_axial_plane(self):
        with pytest.raises(ValueError, match="AXIAL"):
            ExternalMoment(position=100.0, magnitude=1000.0, plane=LoadPlane.AXIAL)


# ===========================================================================
# LoadingProfile
# ===========================================================================

class TestLoadingProfileValidation:

    def test_valid_default(self):
        lp = LoadingProfile()
        assert lp.R_bend == -1.0
        assert lp.R_tors == +1.0

    def test_valid_custom(self):
        lp = LoadingProfile(R_bend=0.0, R_tors=0.5)
        assert lp.R_bend == 0.0
        assert lp.R_tors == 0.5

    def test_boundary_minus_one_valid(self):
        lp = LoadingProfile(R_bend=-1.0, R_tors=-1.0)
        assert lp.R_bend == -1.0

    def test_boundary_plus_one_valid(self):
        lp = LoadingProfile(R_bend=1.0, R_tors=1.0)
        assert lp.R_tors == 1.0

    def test_raises_R_bend_below_minus_one(self):
        with pytest.raises(ValueError, match="R_bend"):
            LoadingProfile(R_bend=-1.1, R_tors=0.0)

    def test_raises_R_bend_above_one(self):
        with pytest.raises(ValueError, match="R_bend"):
            LoadingProfile(R_bend=1.1, R_tors=0.0)

    def test_raises_R_tors_out_of_range(self):
        with pytest.raises(ValueError, match="R_tors"):
            LoadingProfile(R_bend=0.0, R_tors=2.0)

    def test_frozen_immutable(self):
        """LoadingProfile is frozen — attributes cannot be changed."""
        lp = LoadingProfile()
        with pytest.raises(Exception):
            lp.R_bend = 0.0


class TestLoadingProfileAmplitudeRatio:

    def test_A_bend_fully_reversed(self):
        """R=-1 → A = inf (σ_m = 0)."""
        lp = LoadingProfile(R_bend=-1.0, R_tors=0.0)
        assert math.isinf(lp.A_bend)

    def test_A_tors_fully_reversed(self):
        lp = LoadingProfile(R_bend=0.0, R_tors=-1.0)
        assert math.isinf(lp.A_tors)

    def test_A_bend_pulsating(self):
        """R=0 → A = 1.0 (σ_a = σ_m)."""
        lp = LoadingProfile(R_bend=0.0, R_tors=0.0)
        assert lp.A_bend == pytest.approx(1.0)

    def test_A_tors_static(self):
        """R=+1 → A = 0 (τ_a = 0)."""
        lp = LoadingProfile(R_bend=0.0, R_tors=1.0)
        assert lp.A_tors == pytest.approx(0.0)

    def test_A_formula_general(self):
        """A = (1 - R) / (1 + R). Shigley Eq. 6-38."""
        R = 0.3
        lp = LoadingProfile(R_bend=R, R_tors=R)
        expected = (1.0 - R) / (1.0 + R)
        assert lp.A_bend == pytest.approx(expected)
        assert lp.A_tors == pytest.approx(expected)


class TestLoadingProfileDecomposeBending:
    """
    decompose_bending(M_peak) → (Ma, Mm).
    Ma = M_peak * (1 - R) / 2
    Mm = M_peak * (1 + R) / 2
    Shigley §6-12, Eq. 6-36.
    """

    def test_fully_reversed_R_minus_one(self):
        """R=-1: Ma=M_peak, Mm=0."""
        lp = LoadingProfile(R_bend=-1.0, R_tors=0.0)
        Ma, Mm = lp.decompose_bending(100_000.0)
        assert Ma == pytest.approx(100_000.0)
        assert Mm == pytest.approx(0.0)

    def test_static_R_plus_one(self):
        """R=+1: Ma=0, Mm=M_peak."""
        lp = LoadingProfile(R_bend=1.0, R_tors=0.0)
        Ma, Mm = lp.decompose_bending(100_000.0)
        assert Ma == pytest.approx(0.0)
        assert Mm == pytest.approx(100_000.0)

    def test_pulsating_R_zero(self):
        """R=0: Ma=Mm=M_peak/2."""
        lp = LoadingProfile(R_bend=0.0, R_tors=0.0)
        Ma, Mm = lp.decompose_bending(80_000.0)
        assert Ma == pytest.approx(40_000.0)
        assert Mm == pytest.approx(40_000.0)

    def test_Ma_plus_Mm_equals_sigma_max(self):
        """Ma + Mm = M_peak (σ_max). Always."""
        lp = LoadingProfile(R_bend=0.3, R_tors=0.0)
        M_peak = 120_000.0
        Ma, Mm = lp.decompose_bending(M_peak)
        assert Ma + Mm == pytest.approx(M_peak)

    def test_zero_peak_gives_zero_components(self):
        lp = LoadingProfile.rotating_shaft()
        Ma, Mm = lp.decompose_bending(0.0)
        assert Ma == pytest.approx(0.0)
        assert Mm == pytest.approx(0.0)


class TestLoadingProfileDecomposeTorsion:

    def test_steady_torsion_R_plus_one(self):
        """R=+1: Ta=0, Tm=T_peak (standard rotating shaft)."""
        lp = LoadingProfile(R_bend=-1.0, R_tors=1.0)
        Ta, Tm = lp.decompose_torsion(50_000.0)
        assert Ta == pytest.approx(0.0)
        assert Tm == pytest.approx(50_000.0)

    def test_fully_reversed_torsion_R_minus_one(self):
        """R=-1: Ta=T_peak, Tm=0."""
        lp = LoadingProfile(R_bend=0.0, R_tors=-1.0)
        Ta, Tm = lp.decompose_torsion(50_000.0)
        assert Ta == pytest.approx(50_000.0)
        assert Tm == pytest.approx(0.0)

    def test_pulsating_torsion_R_zero(self):
        lp = LoadingProfile(R_bend=0.0, R_tors=0.0)
        Ta, Tm = lp.decompose_torsion(60_000.0)
        assert Ta == pytest.approx(30_000.0)
        assert Tm == pytest.approx(30_000.0)

    def test_Ta_plus_Tm_equals_T_peak(self):
        lp = LoadingProfile(R_bend=0.0, R_tors=0.4)
        T_peak = 75_000.0
        Ta, Tm = lp.decompose_torsion(T_peak)
        assert Ta + Tm == pytest.approx(T_peak)


class TestLoadingProfileFactories:

    def test_rotating_shaft_defaults(self):
        """rotating_shaft(): R_bend=-1, R_tors=+1."""
        lp = LoadingProfile.rotating_shaft()
        assert lp.R_bend == pytest.approx(-1.0)
        assert lp.R_tors == pytest.approx(+1.0)

    def test_rotating_shaft_bending_is_fully_reversed(self):
        lp = LoadingProfile.rotating_shaft()
        Ma, Mm = lp.decompose_bending(100_000.0)
        assert Ma == pytest.approx(100_000.0)
        assert Mm == pytest.approx(0.0)

    def test_rotating_shaft_torsion_is_steady(self):
        lp = LoadingProfile.rotating_shaft()
        Ta, Tm = lp.decompose_torsion(50_000.0)
        assert Ta == pytest.approx(0.0)
        assert Tm == pytest.approx(50_000.0)

    def test_pulsating_factory(self):
        lp = LoadingProfile.pulsating()
        assert lp.R_bend == pytest.approx(0.0)
        assert lp.R_tors == pytest.approx(0.0)

    def test_static_load_factory(self):
        lp = LoadingProfile.static_load()
        Ma, Mm = lp.decompose_bending(100_000.0)
        assert Ma == pytest.approx(0.0)
        assert Mm == pytest.approx(100_000.0)

    def test_custom_factory(self):
        lp = LoadingProfile.custom(R_bend=0.2, R_tors=-0.5)
        assert lp.R_bend == pytest.approx(0.2)
        assert lp.R_tors == pytest.approx(-0.5)

    def test_custom_factory_invalid_raises(self):
        with pytest.raises(ValueError):
            LoadingProfile.custom(R_bend=2.0, R_tors=0.0)

    def test_factories_return_frozen_instances(self):
        """All factory methods return frozen LoadingProfile instances."""
        for lp in [
            LoadingProfile.rotating_shaft(),
            LoadingProfile.pulsating(),
            LoadingProfile.static_load(),
            LoadingProfile.custom(0.0, 0.0),
        ]:
            with pytest.raises(Exception):
                lp.R_bend = 0.5


class TestLoadingProfilePhysicalConsistency:

    def test_rotating_shaft_is_standard_assumption(self):
        """
        Shigley §7-1 canonical assumption for rotating shafts:
        Ma = M_res, Mm = 0, Ta = 0, Tm = T.
        Verify full cycle for typical shaft values.
        """
        lp = LoadingProfile.rotating_shaft()
        M_res = 175_000.0
        T = 87_500.0

        Ma, Mm = lp.decompose_bending(M_res)
        Ta, Tm = lp.decompose_torsion(T)

        assert Ma == pytest.approx(M_res)
        assert Mm == pytest.approx(0.0)
        assert Ta == pytest.approx(0.0)
        assert Tm == pytest.approx(T)

    def test_R_and_A_are_consistent(self):
        """
        R = (1 - A) / (1 + A)  ←→  A = (1 - R) / (1 + R).
        Shigley Eq. 6-37, 6-38.
        """
        for R in [-0.5, 0.0, 0.3, 0.7]:
            lp = LoadingProfile(R_bend=R, R_tors=R)
            A = lp.A_bend
            R_back = (1.0 - A) / (1.0 + A)
            assert R_back == pytest.approx(R, rel=1e-9)