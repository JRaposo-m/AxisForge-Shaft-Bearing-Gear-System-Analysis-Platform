"""
tests/core/machine_elements/test_bearing_capacity.py

Unit tests for the load-rating functions under
core/machine_elements/Bearings/families/*/*/functions/capacity.py,
plus the per-family dispatch that calls them.

Two distinct standards live in these modules and the tests keep them apart:

  BearingCapacity          -- OVERALL rating Cr / Ca from geometry.
                              ISO 281:2007 Sec 6. Flagged NOT VALIDATED in
                              the source; the one anchor available is the
                              6204 fc value, pinned below.
  RollingElementCapacity   -- PER-ELEMENT Q_ci / Q_ce from an already-known
                              Cr / Ca. ISO/TS 16281:2008 Sec 4.3.1.

The 6204 anchor
---------------
solvers/README.md records that `_A1_0089_N` was corrected from 98_066.5 to
98.0665 (three decimal places too high, driving fc ~1000x above literature)
and that the corrected value gives fc ~= 59.2 for a 6204, in range against
NASA/TP-2016-218937. That number is reproduced independently by hand below
and pinned as a regression lock: it is the only external reference this
file has, and it is exactly the value that would move if the constant
regressed.

ASCII only.
"""

from __future__ import annotations

import math

import pytest

from axisforge.core.machine_elements.Bearings.families.ball_bearing.radial.functions import (
    capacity as ball_radial_cap,
)
from axisforge.core.machine_elements.Bearings.families.ball_bearing.thrust.functions import (
    capacity as ball_thrust_cap,
)
from axisforge.core.machine_elements.Bearings.families.ball_bearing.radial.subtypes.deep_groove_ball import (
    DeepGrooveBallFamily,
)


# --- 6204 reference geometry -----------------------------------------------
Z_6204 = 8
DW_6204 = 7.94
DPW_6204 = 33.5
RI_6204 = RE_6204 = 0.52 * DW_6204
C_6204 = 12_700.0                  # catalogue dynamic rating [N]
LAMBDA_SINGLE_ROW = 0.95           # ISO 281 Table 1, radial contact groove
S_6204 = 0.010                     # diametral operating clearance [mm]

_A = RI_6204 + RE_6204 - DW_6204
ALPHA_0_6204 = math.acos(1.0 - S_6204 / (2.0 * _A))
GAMMA_6204 = DW_6204 * math.cos(ALPHA_0_6204) / DPW_6204


# ===========================================================================
# The corrected constant
# ===========================================================================

class TestTranscriptionConstants:
    """
    Regression locks on two magic numbers. Neither is derived -- both are
    read off the standard -- so a test is the only thing that stops a
    re-transcription from silently changing them back.
    """

    def test_a1_constant_is_the_corrected_value(self):
        """
        98.0665, not 98_066.5. The wrong reading put fc (and therefore Cr)
        three orders of magnitude too high. See solvers/README.md.
        """
        assert ball_radial_cap.BearingCapacity._A1_0089_N == pytest.approx(98.0665)

    def test_thrust_side_carries_the_same_corrected_constant(self):
        assert (ball_thrust_cap.BearingCapacity._A1_0089_N
                == pytest.approx(ball_radial_cap.BearingCapacity._A1_0089_N))

    def test_ball_diameter_threshold(self):
        """Formula (13)/(14) switch at 25.4 mm (= 1 inch)."""
        assert ball_radial_cap.BearingCapacity._DW_THRESHOLD_MM == pytest.approx(25.4)


# ===========================================================================
# BearingCapacity -- fc and Cr
# ===========================================================================

class TestFcFactor:

    @pytest.fixture
    def fc(self) -> float:
        return ball_radial_cap.BearingCapacity._fc(
            ri=RI_6204, re=RE_6204, Dw=DW_6204,
            gamma=GAMMA_6204, reduction_factor=LAMBDA_SINGLE_ROW)

    def test_6204_reference_value(self, fc):
        """
        THE anchor for this module. fc ~= 59.2 for a 6204, as recorded in
        solvers/README.md and reproduced by hand from Formula (15). If this
        moves, either the constant regressed or the formula changed.
        """
        assert fc == pytest.approx(59.2, rel=0.01)

    def test_fc_scales_linearly_with_the_reduction_factor(self, fc):
        """lambda enters as a plain multiplier -- eq.(15)."""
        halved = ball_radial_cap.BearingCapacity._fc(
            ri=RI_6204, re=RE_6204, Dw=DW_6204,
            gamma=GAMMA_6204, reduction_factor=LAMBDA_SINGLE_ROW / 2.0)
        assert halved == pytest.approx(fc / 2.0)

    @pytest.mark.parametrize("gamma", [0.0, 1.0, 1.5, -0.1])
    def test_gamma_outside_the_open_unit_interval_rejected(self, gamma):
        with pytest.raises(ValueError, match="gamma"):
            ball_radial_cap.BearingCapacity._fc(
                ri=RI_6204, re=RE_6204, Dw=DW_6204,
                gamma=gamma, reduction_factor=LAMBDA_SINGLE_ROW)

    def test_inner_groove_narrower_than_the_ball_rejected(self):
        with pytest.raises(ValueError, match="2\\*ri"):
            ball_radial_cap.BearingCapacity._fc(
                ri=DW_6204 / 2.0, re=RE_6204, Dw=DW_6204,
                gamma=GAMMA_6204, reduction_factor=LAMBDA_SINGLE_ROW)

    def test_outer_groove_narrower_than_the_ball_rejected(self):
        with pytest.raises(ValueError, match="2\\*re"):
            ball_radial_cap.BearingCapacity._fc(
                ri=RI_6204, re=DW_6204 / 2.0, Dw=DW_6204,
                gamma=GAMMA_6204, reduction_factor=LAMBDA_SINGLE_ROW)

    def test_infinite_outer_radius_takes_the_flat_race_branch(self):
        """re = inf (a flat outer race) has its own radii_ratio path."""
        fc = ball_radial_cap.BearingCapacity._fc(
            ri=RI_6204, re=math.inf, Dw=DW_6204,
            gamma=GAMMA_6204, reduction_factor=LAMBDA_SINGLE_ROW)
        assert fc > 0.0
        assert math.isfinite(fc)


class TestDynamicRating:

    @pytest.fixture
    def Cr(self) -> float:
        return ball_radial_cap.BearingCapacity.dynamic(
            Z=Z_6204, Dw=DW_6204, alpha_0=ALPHA_0_6204,
            ri=RI_6204, re=RE_6204, gamma=GAMMA_6204,
            reduction_factor=LAMBDA_SINGLE_ROW)

    def test_positive_and_finite(self, Cr):
        assert Cr > 0.0
        assert math.isfinite(Cr)

    def test_within_a_factor_of_two_of_the_catalogue_value(self, Cr):
        """
        DELIBERATELY LOOSE. The source flags this formula NOT VALIDATED, and
        the computed value lands near 9.8 kN against SKF's catalogue 12.7 kN
        for a 6204 -- about 77%. That gap is worth understanding before
        anyone trusts Cr, but it is not what this test is for: the bracket
        here only catches a units or constant slip by orders of magnitude.
        Tighten it to a real tolerance once the formula is validated.
        """
        assert 0.5 * C_6204 < Cr < 2.0 * C_6204

    def test_scales_linearly_with_fc(self, Cr):
        """Cr = fc * (...) -- halving lambda halves fc and so halves Cr."""
        half = ball_radial_cap.BearingCapacity.dynamic(
            Z=Z_6204, Dw=DW_6204, alpha_0=ALPHA_0_6204,
            ri=RI_6204, re=RE_6204, gamma=GAMMA_6204,
            reduction_factor=LAMBDA_SINGLE_ROW / 2.0)
        assert half == pytest.approx(Cr / 2.0)

    def test_scales_as_z_to_the_two_thirds(self, Cr):
        """Formula (13): Z^(2/3)."""
        doubled = ball_radial_cap.BearingCapacity.dynamic(
            Z=2 * Z_6204, Dw=DW_6204, alpha_0=ALPHA_0_6204,
            ri=RI_6204, re=RE_6204, gamma=GAMMA_6204,
            reduction_factor=LAMBDA_SINGLE_ROW)
        assert doubled == pytest.approx(Cr * 2.0 ** (2.0 / 3.0))

    def test_row_count_enters_as_i_to_the_zero_point_seven(self, Cr):
        """Formula (13): (i*cos(alpha_0))^0.7."""
        two_rows = ball_radial_cap.BearingCapacity.dynamic(
            Z=Z_6204, Dw=DW_6204, alpha_0=ALPHA_0_6204,
            ri=RI_6204, re=RE_6204, gamma=GAMMA_6204,
            reduction_factor=LAMBDA_SINGLE_ROW, i=2)
        assert two_rows == pytest.approx(Cr * 2.0 ** 0.7)

    def test_large_balls_switch_to_formula_14(self):
        """
        Above 25.4 mm the exponent on Dw drops from 1.8 to 1.4 and a 3.647
        prefactor appears. Checked by straddling the threshold and
        confirming the two branches do not join smoothly -- if they ever
        did, one of them would be wrong.
        """
        common = dict(Z=Z_6204, alpha_0=0.0, gamma=0.2,
                      reduction_factor=LAMBDA_SINGLE_ROW)
        just_below = ball_radial_cap.BearingCapacity.dynamic(
            Dw=25.0, ri=0.52 * 25.0, re=0.52 * 25.0, **common)
        just_above = ball_radial_cap.BearingCapacity.dynamic(
            Dw=26.0, ri=0.52 * 26.0, re=0.52 * 26.0, **common)
        assert just_below > 0.0 and just_above > 0.0
        assert just_below != pytest.approx(just_above, rel=0.05)

    @pytest.mark.parametrize("kwargs", [
        dict(Z=0), dict(Z=-1), dict(Dw=0.0), dict(Dw=-1.0), dict(i=0), dict(i=-1),
    ])
    def test_non_positive_inputs_rejected(self, kwargs):
        base = dict(Z=Z_6204, Dw=DW_6204, alpha_0=ALPHA_0_6204,
                    ri=RI_6204, re=RE_6204, gamma=GAMMA_6204,
                    reduction_factor=LAMBDA_SINGLE_ROW)
        base.update(kwargs)
        with pytest.raises(ValueError, match="must all be positive"):
            ball_radial_cap.BearingCapacity.dynamic(**base)

    def test_static_rating_is_an_explicit_gap(self):
        """
        C0 is a documented hole (f_0 table not transcribed). Pinned so it
        stays a deliberate refusal rather than drifting into a wrong number.
        """
        with pytest.raises(NotImplementedError, match="f_0"):
            ball_radial_cap.BearingCapacity.static(
                Z=Z_6204, Dw=DW_6204, alpha_0=ALPHA_0_6204,
                reduction_factor=LAMBDA_SINGLE_ROW)


# ===========================================================================
# RollingElementCapacity -- per-element Q_ci / Q_ce
# ===========================================================================

class TestPerElementCapacityRadial:

    @pytest.fixture
    def QQ(self) -> tuple[float, float]:
        return ball_radial_cap.RollingElementCapacity.radial(
            Z=Z_6204, alpha_0=ALPHA_0_6204, ri=RI_6204, re=RE_6204,
            Dw=DW_6204, gamma=GAMMA_6204, Cr=C_6204)

    def test_both_positive(self, QQ):
        Q_ci, Q_ce = QQ
        assert Q_ci > 0.0 and Q_ce > 0.0

    def test_outer_raceway_capacity_exceeds_the_inner_one(self, QQ):
        """
        The outer raceway is concave against the ball, so it conforms
        better and carries more per element than the convex inner one.
        Standard for a radial ball bearing.
        """
        Q_ci, Q_ce = QQ
        assert Q_ce > Q_ci

    def test_both_scale_linearly_with_cr(self, QQ):
        """
        Cr enters as a plain multiplier in eq.(19)-(20). Exact, and
        independent of every geometry term -- the strongest invariant here.
        """
        Q_ci, Q_ce = QQ
        Q_ci2, Q_ce2 = ball_radial_cap.RollingElementCapacity.radial(
            Z=Z_6204, alpha_0=ALPHA_0_6204, ri=RI_6204, re=RE_6204,
            Dw=DW_6204, gamma=GAMMA_6204, Cr=2.0 * C_6204)
        assert Q_ci2 == pytest.approx(2.0 * Q_ci)
        assert Q_ce2 == pytest.approx(2.0 * Q_ce)

    def test_both_scale_inversely_with_the_element_count(self, QQ):
        """Z enters as 1/Z -- twice the balls, half the load each."""
        Q_ci, Q_ce = QQ
        Q_ci2, Q_ce2 = ball_radial_cap.RollingElementCapacity.radial(
            Z=2 * Z_6204, alpha_0=ALPHA_0_6204, ri=RI_6204, re=RE_6204,
            Dw=DW_6204, gamma=GAMMA_6204, Cr=C_6204)
        assert Q_ci2 == pytest.approx(Q_ci / 2.0)
        assert Q_ce2 == pytest.approx(Q_ce / 2.0)

    def test_geometry_bracket_is_positive(self):
        bracket = ball_radial_cap.RollingElementCapacity._geometry_bracket(
            GAMMA_6204, RI_6204, RE_6204, DW_6204)
        assert bracket > 0.0

    def test_equal_groove_radii_collapse_the_radii_ratio(self):
        """
        ri == re (the DGBB case, both 0.52*Dw) makes the radii ratio exactly
        1, so the bracket reduces to 1.044*((1-g)/(1+g))^1.72 alone.
        """
        bracket = ball_radial_cap.RollingElementCapacity._geometry_bracket(
            GAMMA_6204, RI_6204, RE_6204, DW_6204)
        expected = 1.044 * ((1.0 - GAMMA_6204) / (1.0 + GAMMA_6204)) ** 1.72
        assert bracket == pytest.approx(expected)

    @pytest.mark.parametrize("field,message", [
        ("ri", "2\\*ri"),
        ("re", "2\\*re"),
    ])
    def test_groove_narrower_than_the_ball_rejected(self, field, message):
        kwargs = dict(Z=Z_6204, alpha_0=ALPHA_0_6204, ri=RI_6204, re=RE_6204,
                      Dw=DW_6204, gamma=GAMMA_6204, Cr=C_6204)
        kwargs[field] = DW_6204 / 2.0
        with pytest.raises(ValueError, match=message):
            ball_radial_cap.RollingElementCapacity.radial(**kwargs)


class TestPerElementCapacityThrust:

    def test_ninety_degree_case_gives_equal_inner_and_outer_capacity(self):
        """
        At alpha_0 = 90 deg with ri == re the radii ratio is exactly 1, so
        eq.(23)-(24) both reduce to (Ca/Z) * 2^0.3 -- inner and outer must
        come out identical. An exact algebraic identity, not a tolerance.
        """
        Ca = 10_000.0
        Q_ci, Q_ce = ball_thrust_cap.RollingElementCapacity.thrust_90deg(
            Z=Z_6204, ri=RI_6204, re=RE_6204, Dw=DW_6204, Ca=Ca)
        assert Q_ci == pytest.approx(Q_ce)
        assert Q_ci == pytest.approx((Ca / Z_6204) * 2.0 ** 0.3)

    def test_ninety_degree_case_scales_linearly_with_ca(self):
        a = ball_thrust_cap.RollingElementCapacity.thrust_90deg(
            Z=Z_6204, ri=RI_6204, re=RE_6204, Dw=DW_6204, Ca=10_000.0)
        b = ball_thrust_cap.RollingElementCapacity.thrust_90deg(
            Z=Z_6204, ri=RI_6204, re=RE_6204, Dw=DW_6204, Ca=20_000.0)
        assert b[0] == pytest.approx(2.0 * a[0])
        assert b[1] == pytest.approx(2.0 * a[1])

    @pytest.mark.parametrize("field", ["ri", "re"])
    def test_groove_narrower_than_the_ball_rejected(self, field):
        kwargs = dict(Z=Z_6204, ri=RI_6204, re=RE_6204, Dw=DW_6204, Ca=10_000.0)
        kwargs[field] = DW_6204 / 2.0
        with pytest.raises(ValueError):
            ball_thrust_cap.RollingElementCapacity.thrust_90deg(**kwargs)


# ===========================================================================
# Family-level dispatch
# ===========================================================================

class TestDeepGrooveBallFamilyCapacityDispatch:
    """
    The families are the public entry point -- solvers call
    bearing.family.per_element_dynamic_capacity(bearing), never the
    functions directly. These tests check the wiring, not the math.
    """

    def test_per_element_capacity_matches_the_underlying_function(self, bearing_6204):
        from_family = DeepGrooveBallFamily.per_element_dynamic_capacity(bearing_6204)
        direct = ball_radial_cap.RollingElementCapacity.radial(
            Z=bearing_6204.Z, alpha_0=bearing_6204.alpha_0,
            ri=bearing_6204.ri, re=bearing_6204.re, Dw=bearing_6204.Dw,
            gamma=bearing_6204.gamma, Cr=bearing_6204.C, i=bearing_6204.i)
        assert from_family[0] == pytest.approx(direct[0])
        assert from_family[1] == pytest.approx(direct[1])

    def test_cr_defaults_to_the_catalogue_rating(self, bearing_6204):
        default = DeepGrooveBallFamily.per_element_dynamic_capacity(bearing_6204)
        explicit = DeepGrooveBallFamily.per_element_dynamic_capacity(
            bearing_6204, Cr=bearing_6204.C)
        assert default == pytest.approx(explicit)

    def test_cr_override_is_honoured(self, bearing_6204):
        base = DeepGrooveBallFamily.per_element_dynamic_capacity(bearing_6204)
        doubled = DeepGrooveBallFamily.per_element_dynamic_capacity(
            bearing_6204, Cr=2.0 * bearing_6204.C)
        assert doubled[0] == pytest.approx(2.0 * base[0])

    def test_row_count_is_read_off_the_bearing_not_passed_in(self, bearing_6204):
        """
        Documented contract: `i` comes from bearing.i, never from the
        caller. Assembling with i=1 must therefore give the i=1 answer.
        """
        assert bearing_6204.i == 1
        from_family = DeepGrooveBallFamily.per_element_dynamic_capacity(bearing_6204)
        i1 = ball_radial_cap.RollingElementCapacity.radial(
            Z=bearing_6204.Z, alpha_0=bearing_6204.alpha_0,
            ri=bearing_6204.ri, re=bearing_6204.re, Dw=bearing_6204.Dw,
            gamma=bearing_6204.gamma, Cr=bearing_6204.C, i=1)
        assert from_family == pytest.approx(i1)

    def test_dynamic_capacity_runs_off_the_assembled_bearing(self, bearing_6204):
        Cr = DeepGrooveBallFamily.dynamic_capacity(bearing_6204)
        assert Cr > 0.0
        assert math.isfinite(Cr)


class TestAngularContactFamilyCapacityDispatch:
    """
    REGRESSION -- this path used to be dead.

    AngularContactFamily.per_element_dynamic_capacity() called
    `bcap.per_element_dynamic_capacity_radial(...)`, a module-level function
    that does not exist in capacity.py; the module only ever defined
    `RollingElementCapacity.radial()`, the classmethod DeepGrooveBallFamily
    calls. So every angular contact bearing raised AttributeError the moment
    anything asked for per-element capacity -- including
    postprocess_and_record(), which has an ANGULAR_CONTACT entry in its
    dispatch table. Nothing had ever exercised it.

    Fixed by pointing the family at RollingElementCapacity.radial(), same as
    the DGBB side. The test below is what stops it going back: it does not
    care HOW capacity is reached, only that it is reached and agrees with
    the other radial ball subtype.
    """

    def test_angular_contact_capacity_agrees_with_the_dgbb_formula(self):
        """
        Both radial ball subtypes are documented as using the same
        eq.(19)-(20). Same geometry in, same numbers out -- and getting
        there at all is half the point, see the class docstring.
        """
        from axisforge.core.machine_elements.Bearings.catalog import BearingCatalog
        from axisforge.core.machine_elements.Bearings.bearing import Bearing
        from axisforge.core.machine_elements.Bearings.families.ball_bearing.radial.subtypes.angular_contact import (
            AngularContactFamily,
        )

        catalog = BearingCatalog(d=20.0, D=47.0, b=14.0, C=C_6204, C0=6_550.0,
                                 designation="7204", position=50.0, label="acbb")
        bearing = Bearing.assemble(
            family=AngularContactFamily(),
            catalog=catalog,
            geometry=dict(Dw=DW_6204, Dpw=DPW_6204, Z=Z_6204,
                          E=206_000.0, nu=0.3, alpha_0_deg=15.0),
            analyses={"point_contact": True},
        )

        from_family = AngularContactFamily.per_element_dynamic_capacity(bearing)
        direct = ball_radial_cap.RollingElementCapacity.radial(
            Z=bearing.Z, alpha_0=bearing.alpha_0, ri=bearing.ri, re=bearing.re,
            Dw=bearing.Dw, gamma=bearing.gamma, Cr=bearing.C, i=bearing.i)
        assert from_family[0] == pytest.approx(direct[0])
        assert from_family[1] == pytest.approx(direct[1])