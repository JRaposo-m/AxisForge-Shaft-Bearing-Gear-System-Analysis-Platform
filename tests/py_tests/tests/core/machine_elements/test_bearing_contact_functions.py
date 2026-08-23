"""
tests/core/machine_elements/test_bearing_contact_functions.py

Unit tests for the pure contact-mechanics functions under
core/machine_elements/Bearings/families/*/*/functions/contact_stiffness.py

Four modules are covered, and they come in two documented pairs:

    ball_bearing/radial/functions/contact_stiffness.py
    ball_bearing/thrust/functions/contact_stiffness.py     "byte-identical"

    roller_bearing/radial/functions/contact_stiffness.py
    roller_bearing/thrust/functions/contact_stiffness.py   "byte-identical"

Each pair is duplicated deliberately (see the modules' own docstrings) so
the radial and thrust folders stay self-contained. The duplication is a
maintenance hazard: an edit to one copy that misses the other is silent.
TestDuplicatedModulesAgree below turns that into a test failure.

These are pure functions -- no bearing, no catalogue, no solver. Reference
values are closed-form or ISO/TS 16281 equation identities.

References:
  ISO/TS 16281:2008 Sec 5, eq.(2)-(11)   -- point contact (ball)
  ISO/TS 16281:2008 Sec 5.2, eq.(34)-(37) -- line contact (roller)
  ISO 281:2007 Table 1                    -- raceway radii (subtype data,
                                             deliberately NOT in these modules)

ASCII only.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from axisforge.core.machine_elements.Bearings.families.ball_bearing.radial.functions import (
    contact_stiffness as ball_radial,
)
from axisforge.core.machine_elements.Bearings.families.ball_bearing.thrust.functions import (
    contact_stiffness as ball_thrust,
)
from axisforge.core.machine_elements.Bearings.families.roller_bearing.radial.functions import (
    contact_stiffness as roller_radial,
)
from axisforge.core.machine_elements.Bearings.families.roller_bearing.thrust.functions import (
    contact_stiffness as roller_thrust,
)


# 6204 reference geometry, the same one used across this suite.
DW = 7.94        # ball diameter [mm]
DPW = 33.5       # pitch circle diameter [mm]
RI = RE = 0.52 * DW      # ISO 281 Table 1, radial contact groove
A_CURV = RI + RE - DW    # total curvature [mm]
E_STEEL = 206_000.0      # [MPa]
NU_STEEL = 0.3


# ===========================================================================
# contact_angle_and_clearance -- the s <-> alpha_0 pair
# ===========================================================================

class TestContactAngleAndClearance:

    def test_exactly_one_input_required(self):
        with pytest.raises(ValueError, match="Exactly one"):
            ball_radial.contact_angle_and_clearance(A_CURV)

    def test_both_inputs_rejected(self):
        with pytest.raises(ValueError, match="Exactly one"):
            ball_radial.contact_angle_and_clearance(A_CURV, s=0.01, alpha_0_deg=10.0)

    def test_zero_clearance_gives_zero_contact_angle(self):
        """s = 0 -> arccos(1) = 0. The contact sits in the radial plane."""
        alpha_0, s = ball_radial.contact_angle_and_clearance(A_CURV, s=0.0)
        assert alpha_0 == pytest.approx(0.0, abs=1e-12)
        assert s == 0.0

    def test_clearance_from_the_closed_form(self):
        """alpha_0 = arccos(1 - s/(2A))."""
        s = 0.010
        alpha_0, s_out = ball_radial.contact_angle_and_clearance(A_CURV, s=s)
        assert alpha_0 == pytest.approx(math.acos(1.0 - s / (2.0 * A_CURV)))
        assert s_out == pytest.approx(s)

    def test_angle_input_returns_the_matching_clearance(self):
        """s = 2A(1 - cos(alpha_0)) -- the inverse of the branch above."""
        alpha_0, s = ball_radial.contact_angle_and_clearance(A_CURV, alpha_0_deg=15.0)
        assert alpha_0 == pytest.approx(math.radians(15.0))
        assert s == pytest.approx(2.0 * A_CURV * (1.0 - math.cos(math.radians(15.0))))

    @pytest.mark.parametrize("s", [0.0, 0.005, 0.010, 0.050, 0.100])
    def test_round_trip_s_to_angle_to_s(self, s):
        """The two branches are exact inverses -- this is the strong check."""
        alpha_0, _ = ball_radial.contact_angle_and_clearance(A_CURV, s=s)
        _, s_back = ball_radial.contact_angle_and_clearance(
            A_CURV, alpha_0_deg=math.degrees(alpha_0))
        assert s_back == pytest.approx(s, abs=1e-12)

    @pytest.mark.parametrize("alpha_deg", [0.0, 5.0, 15.0, 40.0, 90.0])
    def test_round_trip_angle_to_s_to_angle(self, alpha_deg):
        _, s = ball_radial.contact_angle_and_clearance(A_CURV, alpha_0_deg=alpha_deg)
        alpha_back, _ = ball_radial.contact_angle_and_clearance(A_CURV, s=s)
        assert math.degrees(alpha_back) == pytest.approx(alpha_deg, abs=1e-9)

    def test_larger_clearance_opens_the_contact_angle(self):
        tight, _ = ball_radial.contact_angle_and_clearance(A_CURV, s=0.005)
        loose, _ = ball_radial.contact_angle_and_clearance(A_CURV, s=0.050)
        assert loose > tight


# ===========================================================================
# gamma
# ===========================================================================

class TestGammaBall:

    def test_closed_form(self):
        g = ball_radial.gamma(DW, DPW, 0.0)
        assert g == pytest.approx(DW / DPW)

    def test_contact_angle_reduces_gamma(self):
        assert ball_radial.gamma(DW, DPW, math.radians(30.0)) < ball_radial.gamma(DW, DPW, 0.0)

    def test_ninety_degrees_drops_the_cosine_instead_of_multiplying_by_zero(self):
        """
        cos(pi/2) is 6.1e-17, not 0.0 -- the special case exists so a pure
        thrust bearing gets gamma = Dw/Dpw rather than a near-zero number.
        """
        g = ball_radial.gamma(DW, DPW, math.pi / 2.0)
        assert g == pytest.approx(DW / DPW)
        assert g > 0.1          # would be ~1.4e-17 without the special case

    def test_ball_gamma_does_not_range_check(self):
        """
        Unlike the roller version, the ball gamma has no (0,1) guard.
        Documented here because the asymmetry is easy to trip over.
        """
        assert ball_radial.gamma(50.0, 33.5, 0.0) > 1.0


class TestGammaRoller:

    def test_closed_form(self):
        assert roller_radial.gamma(7.0, 33.5, 0.0) == pytest.approx(7.0 / 33.5)

    def test_ninety_degrees_special_case(self):
        assert roller_radial.gamma(7.0, 33.5, math.pi / 2.0) == pytest.approx(7.0 / 33.5)

    @pytest.mark.parametrize("Dwe,Dpw", [(50.0, 33.5), (33.5, 33.5)])
    def test_out_of_range_gamma_rejected(self, Dwe, Dpw):
        """The roller version guards (0, 1) once, so call sites need not."""
        with pytest.raises(ValueError, match="outside"):
            roller_radial.gamma(Dwe, Dpw, 0.0)


# ===========================================================================
# raceway_contact_radius
# ===========================================================================

class TestRacewayContactRadius:

    def test_closed_form(self):
        Ri = ball_radial.raceway_contact_radius(DPW, RI, DW, 0.0)
        assert Ri == pytest.approx(DPW / 2.0 + (RI - DW / 2.0))

    def test_sits_outside_the_pitch_radius_for_a_conforming_groove(self):
        """ri > Dw/2 always (the groove is wider than the ball), so Ri > Dpw/2."""
        assert ball_radial.raceway_contact_radius(DPW, RI, DW, 0.0) > DPW / 2.0

    def test_collapses_to_the_pitch_radius_at_ninety_degrees(self):
        Ri = ball_radial.raceway_contact_radius(DPW, RI, DW, math.pi / 2.0)
        assert Ri == pytest.approx(DPW / 2.0, abs=1e-9)


# ===========================================================================
# Curvature sums and differences -- eq.(5)-(8)
# ===========================================================================

class TestCurvature:

    @pytest.fixture
    def g(self) -> float:
        return ball_radial.gamma(DW, DPW, 0.0)

    def test_curvature_sums_are_positive(self, g):
        assert ball_radial.curvature_sum_inner(DW, RI, g) > 0.0
        assert ball_radial.curvature_sum_outer(DW, RE, g) > 0.0

    def test_inner_curvature_sum_exceeds_the_outer_one(self, g):
        """
        The inner raceway is convex against a convex ball, the outer is
        concave against it -- the inner contact is the tighter one.
        """
        assert (ball_radial.curvature_sum_inner(DW, RI, g)
                > ball_radial.curvature_sum_outer(DW, RE, g))

    def test_curvature_sum_inner_closed_form(self, g):
        expected = (2.0 / DW) * (2.0 + g / (1.0 - g) - DW / (2.0 * RI))
        assert ball_radial.curvature_sum_inner(DW, RI, g) == pytest.approx(expected)

    def test_curvature_sum_outer_closed_form(self, g):
        expected = (2.0 / DW) * (2.0 - g / (1.0 + g) - DW / (2.0 * RE))
        assert ball_radial.curvature_sum_outer(DW, RE, g) == pytest.approx(expected)

    def test_curvature_differences_are_in_the_open_unit_interval(self, g):
        """F(rho) is a normalised ratio -- outside (0, 1) the chi solve fails."""
        assert 0.0 < ball_radial.curvature_diff_inner(DW, RI, g) < 1.0
        assert 0.0 < ball_radial.curvature_diff_outer(DW, RE, g) < 1.0

    def test_self_aligning_geometry_cancels_the_outer_curvature_difference(self, g):
        """
        ISO 281 Table 1 gives a self-aligning ball bearing
        re = 0.5*(1/gamma + 1)*Dw. SelfAligningContactStiffness's docstring
        claims that makes F_e(rho) EXACTLY zero, algebraically, for any
        valid geometry -- which is why the generic brentq solve cannot
        bracket its root. This test verifies that claim rather than taking
        it on trust; if it ever stops holding, the stub's justification
        goes with it.
        """
        re_self_aligning = 0.5 * (1.0 / g + 1.0) * DW
        F_e = ball_radial.curvature_diff_outer(DW, re_self_aligning, g)
        assert F_e == pytest.approx(0.0, abs=1e-12)


# ===========================================================================
# hertz_spring_constant -- eq.(9)-(11)
# ===========================================================================

class TestHertzSpringConstant:

    @pytest.fixture
    def cp(self) -> float:
        return ball_radial.hertz_spring_constant(
            DW, RI, RE, E_STEEL, NU_STEEL, 0.0, DPW)

    def test_positive(self, cp):
        assert cp > 0.0

    def test_order_of_magnitude(self, cp):
        """
        c_p for a ~8 mm steel ball lands in the 1e4-1e6 N/mm^1.5 band. A
        loose bracket on purpose -- this catches a unit or constant slip by
        orders of magnitude without pretending to be a validated value.
        """
        assert 1.0e4 < cp < 1.0e6

    def test_scales_linearly_with_the_reduced_modulus(self):
        """
        c_p = 1.48 * E_star * (bi + be)^-1.5, and bi/be do not depend on E,
        so doubling E must exactly double c_p.
        """
        single = ball_radial.hertz_spring_constant(
            DW, RI, RE, E_STEEL, NU_STEEL, 0.0, DPW)
        double = ball_radial.hertz_spring_constant(
            DW, RI, RE, 2.0 * E_STEEL, NU_STEEL, 0.0, DPW)
        assert double == pytest.approx(2.0 * single)

    def test_stiffer_with_a_larger_ball(self):
        small = ball_radial.hertz_spring_constant(
            5.0, 0.52 * 5.0, 0.52 * 5.0, E_STEEL, NU_STEEL, 0.0, DPW)
        large = ball_radial.hertz_spring_constant(
            10.0, 0.52 * 10.0, 0.52 * 10.0, E_STEEL, NU_STEEL, 0.0, DPW)
        assert large > small

    def test_load_deflection_exponent_is_three_halves(self):
        assert ball_radial.LOAD_DEFLECTION_EXPONENT == pytest.approx(1.5)

    def test_chi_is_above_one_for_elliptical_contact(self):
        g = ball_radial.gamma(DW, DPW, 0.0)
        assert ball_radial._chi(ball_radial.curvature_diff_inner(DW, RI, g)) > 1.0
        assert ball_radial._chi(ball_radial.curvature_diff_outer(DW, RE, g)) > 1.0


class TestSelfAligningStub:
    """
    The self-aligning outer-race term is deliberately unimplemented -- see
    the class docstring in contact_stiffness.py. Pinning it here so the
    NotImplementedError is a decision on record, not an accident, and so
    the day it gets filled in these tests fail and demand attention.
    """

    def test_outer_race_term_not_implemented(self):
        with pytest.raises(NotImplementedError, match="circular-contact"):
            ball_radial.SelfAligningContactStiffness.outer_race_spring_term(
                DW, RE, E_STEEL, NU_STEEL, 0.0, DPW)

    def test_spring_constant_propagates_the_not_implemented(self):
        with pytest.raises(NotImplementedError):
            ball_radial.SelfAligningContactStiffness.hertz_spring_constant(
                DW, RI, RE, E_STEEL, NU_STEEL, 0.0, DPW)


# ===========================================================================
# Roller: lamina positions and line-contact spring constant
# ===========================================================================

class TestLaminaPositions:

    LWE = 10.0
    N_S = 30

    @pytest.fixture
    def x_k(self) -> np.ndarray:
        return roller_radial.lamina_positions(self.LWE, self.N_S)

    def test_count(self, x_k):
        assert len(x_k) == self.N_S

    def test_all_strictly_inside_the_roller_length(self, x_k):
        """Sec 5.2.2 -- midpoints, so never on the end faces."""
        assert np.all(x_k > -self.LWE / 2.0)
        assert np.all(x_k < self.LWE / 2.0)

    def test_symmetric_about_the_roller_centre(self, x_k):
        assert np.sum(x_k) == pytest.approx(0.0, abs=1e-12)
        assert np.allclose(x_k, -x_k[::-1])

    def test_uniform_spacing_equal_to_the_lamina_length(self, x_k):
        spacing = np.diff(x_k)
        assert np.allclose(spacing, self.LWE / self.N_S)

    def test_first_and_last_are_half_a_lamina_from_the_ends(self, x_k):
        half = self.LWE / self.N_S / 2.0
        assert x_k[0] == pytest.approx(-self.LWE / 2.0 + half)
        assert x_k[-1] == pytest.approx(self.LWE / 2.0 - half)

    @pytest.mark.parametrize("n_s", [1, 2, 5, 30, 100])
    def test_symmetry_holds_for_any_lamina_count(self, n_s):
        x_k = roller_radial.lamina_positions(self.LWE, n_s)
        assert np.sum(x_k) == pytest.approx(0.0, abs=1e-12)
        assert len(x_k) == n_s

    def test_single_lamina_sits_at_the_centre(self):
        assert roller_radial.lamina_positions(self.LWE, 1)[0] == pytest.approx(0.0)


class TestLineContactSpringConstant:

    def test_closed_form(self):
        """cL = 35948 * Lwe^(8/9), eq.(35)."""
        cL, _ = roller_radial.line_contact_spring_constant(10.0, 30)
        assert cL == pytest.approx(35948.0 * 10.0 ** (8.0 / 9.0))

    def test_per_lamina_constant_is_the_total_divided_by_the_count(self):
        """cs = cL / n_s, eq.(37)."""
        cL, cs = roller_radial.line_contact_spring_constant(10.0, 30)
        assert cs == pytest.approx(cL / 30)

    def test_lamina_constant_shrinks_as_the_roller_is_sliced_finer(self):
        _, cs_coarse = roller_radial.line_contact_spring_constant(10.0, 10)
        _, cs_fine = roller_radial.line_contact_spring_constant(10.0, 100)
        assert cs_fine < cs_coarse

    def test_total_stiffness_is_independent_of_the_lamina_count(self):
        """Slicing is a discretisation choice; cL must not move with it."""
        cL_a, _ = roller_radial.line_contact_spring_constant(10.0, 10)
        cL_b, _ = roller_radial.line_contact_spring_constant(10.0, 100)
        assert cL_a == pytest.approx(cL_b)

    def test_longer_roller_is_stiffer(self):
        cL_short, _ = roller_radial.line_contact_spring_constant(5.0, 30)
        cL_long, _ = roller_radial.line_contact_spring_constant(20.0, 30)
        assert cL_long > cL_short

    def test_load_deflection_exponent_is_ten_ninths(self):
        assert roller_radial.LOAD_DEFLECTION_EXPONENT == pytest.approx(10.0 / 9.0)

    def test_line_contact_is_stiffer_than_point_contact(self):
        """
        The exponents differ (10/9 vs 3/2): a roller carries load with much
        less deflection than a ball. Pinned as a sanity relation between
        the two contact laws.
        """
        assert roller_radial.LOAD_DEFLECTION_EXPONENT < ball_radial.LOAD_DEFLECTION_EXPONENT


# ===========================================================================
# The duplication guard
# ===========================================================================

class TestDuplicatedModulesAgree:
    """
    radial/ and thrust/ each hold their own copy of contact_stiffness.py,
    documented as byte-identical and duplicated on purpose. Nothing in the
    codebase enforces that. These tests do -- numerically, so a divergence
    in behaviour fails even if the files differ only in formatting.
    """

    @pytest.mark.parametrize("fn_name,args", [
        ("gamma", (DW, DPW, 0.3)),
        ("raceway_contact_radius", (DPW, RI, DW, 0.3)),
        ("curvature_sum_inner", (DW, RI, 0.23)),
        ("curvature_sum_outer", (DW, RE, 0.23)),
        ("curvature_diff_inner", (DW, RI, 0.23)),
        ("curvature_diff_outer", (DW, RE, 0.23)),
    ])
    def test_ball_radial_and_thrust_agree(self, fn_name, args):
        assert (getattr(ball_radial, fn_name)(*args)
                == pytest.approx(getattr(ball_thrust, fn_name)(*args)))

    def test_ball_hertz_spring_constants_agree(self):
        args = (DW, RI, RE, E_STEEL, NU_STEEL, 0.3, DPW)
        assert (ball_radial.hertz_spring_constant(*args)
                == pytest.approx(ball_thrust.hertz_spring_constant(*args)))

    def test_ball_contact_angle_helpers_agree(self):
        a_rad = ball_radial.contact_angle_and_clearance(A_CURV, s=0.01)
        a_thr = ball_thrust.contact_angle_and_clearance(A_CURV, s=0.01)
        assert a_rad[0] == pytest.approx(a_thr[0])
        assert a_rad[1] == pytest.approx(a_thr[1])

    def test_ball_exponents_agree(self):
        assert (ball_radial.LOAD_DEFLECTION_EXPONENT
                == ball_thrust.LOAD_DEFLECTION_EXPONENT)

    @pytest.mark.parametrize("Lwe,n_s", [(10.0, 30), (5.0, 8), (25.0, 100)])
    def test_roller_radial_and_thrust_agree_on_laminae(self, Lwe, n_s):
        assert np.allclose(roller_radial.lamina_positions(Lwe, n_s),
                           roller_thrust.lamina_positions(Lwe, n_s))

    @pytest.mark.parametrize("Lwe,n_s", [(10.0, 30), (5.0, 8), (25.0, 100)])
    def test_roller_radial_and_thrust_agree_on_spring_constants(self, Lwe, n_s):
        assert (roller_radial.line_contact_spring_constant(Lwe, n_s)
                == pytest.approx(roller_thrust.line_contact_spring_constant(Lwe, n_s)))

    def test_roller_gammas_agree(self):
        assert (roller_radial.gamma(7.0, 33.5, 0.2)
                == pytest.approx(roller_thrust.gamma(7.0, 33.5, 0.2)))

    def test_roller_exponents_agree(self):
        assert (roller_radial.LOAD_DEFLECTION_EXPONENT
                == roller_thrust.LOAD_DEFLECTION_EXPONENT)
