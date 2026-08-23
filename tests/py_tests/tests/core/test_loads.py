"""
tests/core/test_loads.py

Unit tests for axisforge.core.loads

Covers Load, RadialLoad, AxialLoad, TorqueLoad, ExternalMoment,
DistributedRadialLoad and LoadingProfile.

Sign convention under test:
  theta_deg measured from +Y toward +Z, right-hand rule about +X;
  Fy = F*cos(theta), Fz = F*sin(theta);
  LoadPlane.XY picks the Y component, LoadPlane.XZ the Z component.

Reference values are closed-form. Distributed-load quadrature results are
checked against the analytic resultant of the same distribution, so a
regression in the numerical path is caught rather than blessed.

ASCII only.
"""

from __future__ import annotations

import math

import pytest

from axisforge.core.loads import (
    LoadPlane,
    RadialLoad,
    AxialLoad,
    TorqueLoad,
    ExternalMoment,
    DistributedRadialLoad,
    LoadingProfile,
)

ABS_TOL = 1e-9


# ===========================================================================
# Load -- shared base behaviour
# ===========================================================================

class TestLoadBase:

    def test_default_source_is_user(self):
        assert RadialLoad(position=10.0, magnitude=100.0).source == "user"

    def test_source_is_carried_through(self):
        ld = RadialLoad(position=10.0, magnitude=100.0, source="gear_mesh")
        assert ld.source == "gear_mesh"

    def test_negative_position_is_reported_not_raised(self):
        """Construction never raises; validate() is the gate."""
        ld = RadialLoad(position=-5.0, magnitude=100.0, label="L1")
        errors = ld.validate()
        assert any("position must be >= 0" in e for e in errors)

    def test_validate_tags_with_the_label_when_present(self):
        errors = RadialLoad(position=-5.0, magnitude=100.0, label="L1").validate()
        assert errors[0].startswith("L1:")

    def test_validate_falls_back_to_the_class_name(self):
        errors = RadialLoad(position=-5.0, magnitude=100.0).validate()
        assert errors[0].startswith("RadialLoad:")

    def test_validate_or_raise(self):
        with pytest.raises(ValueError, match="position must be >= 0"):
            RadialLoad(position=-5.0, magnitude=100.0).validate_or_raise()

    def test_clean_load_validate_or_raise_is_silent(self):
        RadialLoad(position=5.0, magnitude=100.0).validate_or_raise()


# ===========================================================================
# RadialLoad
# ===========================================================================

class TestRadialLoadDecomposition:

    @pytest.mark.parametrize("theta_deg,fy,fz", [
        (0.0,   1000.0,     0.0),
        (90.0,     0.0,  1000.0),
        (180.0, -1000.0,    0.0),
        (270.0,    0.0, -1000.0),
    ])
    def test_cardinal_directions(self, theta_deg, fy, fz):
        ld = RadialLoad(position=100.0, magnitude=1000.0, theta_deg=theta_deg)
        assert ld.Fy == pytest.approx(fy, abs=1e-6)
        assert ld.Fz == pytest.approx(fz, abs=1e-6)

    def test_forty_five_degrees(self):
        ld = RadialLoad(position=0.0, magnitude=1000.0, theta_deg=45.0)
        expected = 1000.0 / math.sqrt(2.0)
        assert ld.Fy == pytest.approx(expected)
        assert ld.Fz == pytest.approx(expected)

    def test_components_recover_the_magnitude(self):
        ld = RadialLoad(position=0.0, magnitude=1234.5, theta_deg=37.0)
        assert math.hypot(ld.Fy, ld.Fz) == pytest.approx(1234.5)

    @pytest.mark.parametrize("given,stored", [
        (370.0, 10.0),
        (-90.0, 270.0),
        (720.0, 0.0),
    ])
    def test_theta_is_wrapped_into_zero_360(self, given, stored):
        ld = RadialLoad(position=0.0, magnitude=1.0, theta_deg=given)
        assert ld.theta_deg == pytest.approx(stored)

    def test_theta_radians_matches_theta_deg(self):
        ld = RadialLoad(position=0.0, magnitude=1.0, theta_deg=30.0)
        assert ld.theta == pytest.approx(math.radians(30.0))

    def test_component_selects_the_plane(self):
        ld = RadialLoad(position=0.0, magnitude=1000.0, theta_deg=30.0)
        assert ld.component(LoadPlane.XY) == pytest.approx(ld.Fy)
        assert ld.component(LoadPlane.XZ) == pytest.approx(ld.Fz)

    def test_negative_magnitude_is_reported(self):
        errors = RadialLoad(position=0.0, magnitude=-1.0).validate()
        assert any("magnitude must be >= 0" in e for e in errors)

    @pytest.mark.xfail(
        strict=True,
        reason="repr embeds the degree sign and N.m middle dot -- violates the "
               "project's ASCII-only console rule. See tests/README.md, finding F.",
    )
    @pytest.mark.parametrize("load", [
        RadialLoad(position=0.0, magnitude=1.0),
        TorqueLoad(position=0.0, magnitude=1.0),
        ExternalMoment(position=0.0, magnitude=1.0),
    ])
    def test_repr_is_ascii(self, load):
        """
        Project rule: console output is ASCII only (Windows PowerShell,
        cp1252). xfail(strict) so this flips to a failure the moment the
        reprs are cleaned up and the marker should be removed.
        """
        repr(load).encode("ascii")


# ===========================================================================
# AxialLoad / TorqueLoad
# ===========================================================================

class TestAxialAndTorqueLoads:

    def test_axial_load_keeps_its_sign(self):
        assert AxialLoad(position=10.0, magnitude=-500.0).magnitude == -500.0

    def test_axial_load_has_no_angular_decomposition(self):
        ld = AxialLoad(position=10.0, magnitude=500.0)
        assert not hasattr(ld, "theta_deg")

    def test_torque_load_keeps_its_sign(self):
        assert TorqueLoad(position=10.0, magnitude=-42.0).magnitude == -42.0

    @pytest.mark.parametrize("cls", [AxialLoad, TorqueLoad])
    def test_position_validation_is_inherited(self, cls):
        errors = cls(position=-1.0, magnitude=1.0).validate()
        assert any("position must be >= 0" in e for e in errors)


# ===========================================================================
# ExternalMoment
# ===========================================================================

class TestExternalMoment:

    @pytest.mark.parametrize("theta_deg,my,mz", [
        (0.0,   500.0,   0.0),
        (90.0,    0.0, 500.0),
        (180.0, -500.0,  0.0),
    ])
    def test_decomposition(self, theta_deg, my, mz):
        m = ExternalMoment(position=50.0, magnitude=500.0, theta_deg=theta_deg)
        assert m.My == pytest.approx(my, abs=1e-6)
        assert m.Mz == pytest.approx(mz, abs=1e-6)

    def test_component_selects_the_plane(self):
        m = ExternalMoment(position=50.0, magnitude=500.0, theta_deg=30.0)
        assert m.component(LoadPlane.XY) == pytest.approx(m.My)
        assert m.component(LoadPlane.XZ) == pytest.approx(m.Mz)

    def test_shares_the_radial_load_angle_convention(self):
        """Same theta must give the same projection factors as RadialLoad."""
        theta = 37.0
        f = RadialLoad(position=0.0, magnitude=1.0, theta_deg=theta)
        m = ExternalMoment(position=0.0, magnitude=1.0, theta_deg=theta)
        assert m.My == pytest.approx(f.Fy)
        assert m.Mz == pytest.approx(f.Fz)


# ===========================================================================
# DistributedRadialLoad -- uniform, constant theta
# ===========================================================================

class TestDistributedUniform:

    @pytest.fixture
    def uniform(self) -> DistributedRadialLoad:
        """F = 1000 N spread over [50, 70] mm, theta = 0 (pure +Y)."""
        return DistributedRadialLoad(x_lo=50.0, x_hi=70.0,
                                     magnitude=1000.0, label="q1")

    def test_span_and_nominal_position(self, uniform):
        assert uniform.b == pytest.approx(20.0)
        assert uniform.position == pytest.approx(60.0)

    def test_is_uniform(self, uniform):
        assert uniform.is_uniform is True

    def test_intensity_inside_the_span(self, uniform):
        assert uniform.q(60.0) == pytest.approx(50.0)   # 1000 N / 20 mm

    @pytest.mark.parametrize("x", [49.0, 71.0, 0.0, 1000.0])
    def test_intensity_is_zero_outside_the_span(self, uniform, x):
        assert uniform.q(x) == 0.0

    def test_intensity_at_the_span_ends_is_inclusive(self, uniform):
        assert uniform.q(50.0) == pytest.approx(50.0)
        assert uniform.q(70.0) == pytest.approx(50.0)

    def test_resultant_equals_the_declared_magnitude(self, uniform):
        assert uniform.magnitude == pytest.approx(1000.0)
        assert uniform.resultant_Fy == pytest.approx(1000.0)
        assert uniform.resultant_Fz == pytest.approx(0.0, abs=ABS_TOL)

    def test_resultant_component_matches_the_named_properties(self, uniform):
        assert uniform.resultant_component(LoadPlane.XY) == pytest.approx(uniform.resultant_Fy)
        assert uniform.resultant_component(LoadPlane.XZ) == pytest.approx(uniform.resultant_Fz)

    def test_component_intensity_matches_qy_qz(self, uniform):
        assert uniform.component_intensity(60.0, LoadPlane.XY) == pytest.approx(uniform.qy(60.0))
        assert uniform.component_intensity(60.0, LoadPlane.XZ) == pytest.approx(uniform.qz(60.0))

    def test_centroid_is_the_midspan(self, uniform):
        assert uniform.centroid(LoadPlane.XY) == pytest.approx(60.0)

    def test_theta_at_is_constant(self, uniform):
        assert uniform.theta_at(52.0) == pytest.approx(0.0)
        assert uniform.theta_at(68.0) == pytest.approx(0.0)

    def test_signed_magnitude_is_preserved(self):
        """Gear-mesh components are signed; the sign is not stripped."""
        q = DistributedRadialLoad(x_lo=0.0, x_hi=10.0, magnitude=-400.0)
        assert q.magnitude == pytest.approx(-400.0)
        assert q.resultant_Fy == pytest.approx(-400.0)

    def test_inverted_span_rejected(self):
        with pytest.raises(ValueError, match="must be >"):
            DistributedRadialLoad(x_lo=70.0, x_hi=50.0, magnitude=1000.0)

    def test_zero_span_rejected(self):
        with pytest.raises(ValueError, match="must be >"):
            DistributedRadialLoad(x_lo=50.0, x_hi=50.0, magnitude=1000.0)


class TestDistributedBendingMoment:
    """
    M(x) = integral over [x_lo, min(x, x_hi)] of q(xi)*(x - xi) dxi.

    Downstream of the whole span this collapses to F*(x - x_centroid), which
    is the independent check used here.
    """

    @pytest.fixture
    def uniform(self) -> DistributedRadialLoad:
        return DistributedRadialLoad(x_lo=50.0, x_hi=70.0, magnitude=1000.0)

    @pytest.mark.parametrize("x", [0.0, 25.0, 50.0])
    def test_no_contribution_upstream_of_the_span(self, uniform, x):
        assert uniform.bending_moment_contribution(x, LoadPlane.XY) == pytest.approx(0.0)

    @pytest.mark.parametrize("x", [70.0, 100.0, 300.0])
    def test_downstream_collapses_to_the_point_load_result(self, uniform, x):
        expected = 1000.0 * (x - 60.0)
        got = uniform.bending_moment_contribution(x, LoadPlane.XY)
        assert got == pytest.approx(expected)

    def test_partial_span(self, uniform):
        """
        At x = 60 only [50, 60] is upstream: 500 N acting at 55 mm,
        lever arm 5 mm -> 2500 N*mm.
        """
        got = uniform.bending_moment_contribution(60.0, LoadPlane.XY)
        assert got == pytest.approx(2500.0)

    def test_projection_onto_the_orthogonal_plane_is_zero(self, uniform):
        got = uniform.bending_moment_contribution(100.0, LoadPlane.XZ)
        assert got == pytest.approx(0.0, abs=1e-6)

    def test_rotated_load_splits_between_the_planes(self):
        q = DistributedRadialLoad(x_lo=0.0, x_hi=20.0, magnitude=1000.0,
                                  theta_deg=30.0)
        m_xy = q.bending_moment_contribution(100.0, LoadPlane.XY)
        m_xz = q.bending_moment_contribution(100.0, LoadPlane.XZ)
        total = 1000.0 * (100.0 - 10.0)
        assert m_xy == pytest.approx(total * math.cos(math.radians(30.0)))
        assert m_xz == pytest.approx(total * math.sin(math.radians(30.0)))


class TestDistributedAsPointLoad:

    def test_equivalent_point_load(self):
        q = DistributedRadialLoad(x_lo=50.0, x_hi=70.0, magnitude=1000.0,
                                  theta_deg=30.0, label="q1", source="gear_mesh")
        p = q.as_point_load()

        assert isinstance(p, RadialLoad)
        assert p.position == pytest.approx(60.0)
        assert p.magnitude == pytest.approx(1000.0)
        assert p.theta_deg == pytest.approx(30.0)
        assert p.label == "q1"
        assert p.source == "gear_mesh"

    def test_equivalent_point_load_reproduces_the_components(self):
        q = DistributedRadialLoad(x_lo=0.0, x_hi=20.0, magnitude=800.0,
                                  theta_deg=200.0)
        p = q.as_point_load()
        assert p.Fy == pytest.approx(q.resultant_Fy)
        assert p.Fz == pytest.approx(q.resultant_Fz)


# ===========================================================================
# DistributedRadialLoad -- callable intensity
# ===========================================================================

class TestDistributedVariableIntensity:

    def test_constant_callable_matches_the_scalar_form(self):
        """q(x) = 10 N/mm over 20 mm is the same load as F = 200 N."""
        q = DistributedRadialLoad(x_lo=0.0, x_hi=20.0, magnitude=lambda x: 10.0)
        assert q.magnitude == pytest.approx(200.0)
        assert q.is_uniform is False           # declared callable, not uniform
        assert q.centroid(LoadPlane.XY) == pytest.approx(10.0)

    def test_linear_ramp_resultant_and_centroid(self):
        """
        q(x) = x N/mm over [0, 10]:
          F = 50 N, centroid = (integral x*q dx)/F = (1000/3)/50 = 6.667 mm.
        """
        q = DistributedRadialLoad(x_lo=0.0, x_hi=10.0, magnitude=lambda x: x)
        assert q.magnitude == pytest.approx(50.0)
        assert q.centroid(LoadPlane.XY) == pytest.approx(20.0 / 3.0)

    def test_linear_ramp_bending_moment_downstream(self):
        """Downstream, M = F*(x - centroid) regardless of the profile shape."""
        q = DistributedRadialLoad(x_lo=0.0, x_hi=10.0, magnitude=lambda x: x)
        got = q.bending_moment_contribution(100.0, LoadPlane.XY)
        assert got == pytest.approx(50.0 * (100.0 - 20.0 / 3.0))

    def test_intensity_lookup_follows_the_callable(self):
        q = DistributedRadialLoad(x_lo=0.0, x_hi=10.0, magnitude=lambda x: x)
        assert q.q(4.0) == pytest.approx(4.0)
        assert q.q(-1.0) == 0.0


# ===========================================================================
# DistributedRadialLoad -- callable direction
# ===========================================================================

class TestDistributedVariableDirection:

    @pytest.fixture
    def swept(self) -> DistributedRadialLoad:
        """
        Constant intensity 10 N/mm over [0, 90], theta(x) = x degrees.

          Fy = integral 10*cos(x deg) dx = 10*(180/pi)*sin(90 deg)
          Fz = integral 10*sin(x deg) dx = 10*(180/pi)*(1 - cos(90 deg))
        """
        return DistributedRadialLoad(
            x_lo=0.0, x_hi=90.0,
            magnitude=lambda x: 10.0,
            theta_deg=lambda x: x,
        )

    def test_components_against_the_closed_form(self, swept):
        scale = 10.0 * 180.0 / math.pi
        assert swept.resultant_Fy == pytest.approx(scale * 1.0)
        assert swept.resultant_Fz == pytest.approx(scale * 1.0)

    def test_magnitude_is_the_vector_norm(self, swept):
        assert swept.magnitude == pytest.approx(
            math.hypot(swept.resultant_Fy, swept.resultant_Fz)
        )

    def test_magnitude_is_positive_for_variable_theta(self, swept):
        assert swept.magnitude > 0.0

    def test_theta_at_follows_the_callable(self, swept):
        assert swept.theta_at(30.0) == pytest.approx(30.0)
        assert swept.theta_at(45.0) == pytest.approx(45.0)

    def test_is_not_uniform(self, swept):
        assert swept.is_uniform is False

    def test_as_point_load_is_refused(self, swept):
        with pytest.raises(TypeError, match="no single representative"):
            swept.as_point_load()

    def test_centroids_may_differ_between_planes(self, swept):
        """
        With theta varying, the two plane resultants are applied at
        different points -- that is the whole reason centroid takes a plane.
        """
        c_xy = swept.centroid(LoadPlane.XY)
        c_xz = swept.centroid(LoadPlane.XZ)
        assert c_xy != pytest.approx(c_xz)


# ===========================================================================
# LoadingProfile
# ===========================================================================

class TestLoadingProfile:

    @pytest.mark.parametrize("R", [-1.5, 1.5, 2.0])
    def test_stress_ratio_outside_range_rejected(self, R):
        with pytest.raises(ValueError, match="R must be in"):
            LoadingProfile(R=R)

    def test_default_is_static(self):
        p = LoadingProfile()
        assert p.R == 1.0
        assert p.is_static is True
        assert p.is_fully_reversed is False

    def test_static_factors(self):
        p = LoadingProfile(R=1.0)
        assert p.sigma_mean_factor == pytest.approx(1.0)
        assert p.sigma_amplitude_factor == pytest.approx(0.0)

    def test_fully_reversed_factors(self):
        p = LoadingProfile(R=-1.0)
        assert p.is_fully_reversed is True
        assert p.sigma_mean_factor == pytest.approx(0.0)
        assert p.sigma_amplitude_factor == pytest.approx(1.0)

    def test_zero_to_tension_factors(self):
        """R = 0 (repeated loading): mean = amplitude = sigma_max/2."""
        p = LoadingProfile(R=0.0)
        assert p.sigma_mean_factor == pytest.approx(0.5)
        assert p.sigma_amplitude_factor == pytest.approx(0.5)

    @pytest.mark.parametrize("R", [-1.0, -0.5, 0.0, 0.5, 1.0])
    def test_factors_reconstruct_the_extremes(self, R):
        """mean + amplitude = 1 (sigma_max) for every admissible R."""
        p = LoadingProfile(R=R)
        assert p.sigma_mean_factor + p.sigma_amplitude_factor == pytest.approx(1.0)

    def test_is_frozen(self):
        from dataclasses import FrozenInstanceError
        p = LoadingProfile(R=0.0)
        with pytest.raises(FrozenInstanceError):
            p.R = 0.5
