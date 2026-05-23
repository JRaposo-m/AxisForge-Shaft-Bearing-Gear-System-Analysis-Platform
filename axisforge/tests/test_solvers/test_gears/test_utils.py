"""
tests/test_solvers/test_gears/test_utils.py
Unit tests for solvers/gears/utils.py

Coverage targets:
  - involute()              — mathematical identity
  - solve_alpha_tw()        — numerical solver correctness
  - contact_ratio_alpha()   — εα geometry
  - contact_ratio_beta()    — εβ geometry
  - undercut_z_min()        — Shigley Tab. 13-11
  - validate_geometry_inputs() — all guard branches

Reference: ISO 21771:2007; Shigley 10th ed. §13-7, §13-10.
"""
from __future__ import annotations

import math
import pytest

from solvers.gears.utils import (
    involute,
    solve_alpha_tw,
    contact_ratio_alpha,
    contact_ratio_beta,
    undercut_z_min,
    validate_geometry_inputs,
    HAP,
    HFP,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_hap(self):
        assert HAP == 1.0

    def test_hfp(self):
        assert HFP == 1.25


# ---------------------------------------------------------------------------
# involute()
# ---------------------------------------------------------------------------

class TestInvolute:
    def test_zero(self):
        """inv(0) = tan(0) − 0 = 0."""
        assert involute(0.0) == pytest.approx(0.0, abs=1e-12)

    def test_20_deg(self):
        """inv(20°) — known value used in gear standards."""
        alpha = math.radians(20.0)
        expected = math.tan(alpha) - alpha
        assert involute(alpha) == pytest.approx(expected, rel=1e-12)

    def test_positive_and_increasing(self):
        """Involute is strictly positive and increasing for α > 0."""
        angles = [math.radians(a) for a in (5, 10, 15, 20, 25, 30)]
        values = [involute(a) for a in angles]
        assert all(v > 0 for v in values)
        assert all(values[i] < values[i + 1] for i in range(len(values) - 1))


# ---------------------------------------------------------------------------
# solve_alpha_tw()
# ---------------------------------------------------------------------------

class TestSolveAlphaTw:
    """
    Validate against GEARpie reference cases.

    C14: m=4.5, z1=16, z2=24, α=20°, β=0°, x1=0.1817, x2=0.1715, al=91.5mm
    H501: m=3.5, z1=20, z2=30, α=20°, β=15°, x1=0.1809, x2=0.0891, al=91.5mm
    """

    @staticmethod
    def _base_params_c14():
        mn = 4.5
        z1, z2 = 16, 24
        alpha_n = math.radians(20.0)
        beta = math.radians(0.0)
        mt = mn / math.cos(beta)
        alpha_t = math.atan(math.tan(alpha_n) / math.cos(beta))
        d1, d2 = mt * z1, mt * z2
        db1 = d1 * math.cos(alpha_t)
        db2 = d2 * math.cos(alpha_t)
        return alpha_n, alpha_t, db1, db2

    def test_c14_al_matches_reference(self):
        """Computed al must match GEARpie reference al=91.5mm within 0.05mm."""
        alpha_n, alpha_t, db1, db2 = self._base_params_c14()
        al, _ = solve_alpha_tw(alpha_n, alpha_t, 0.1817, 0.1715, 16, 24, db1, db2)
        assert al == pytest.approx(91.5, abs=0.05)

    def test_c14_alpha_tw_reasonable(self):
        """αtw must be > αt for positive total profile shift."""
        alpha_n, alpha_t, db1, db2 = self._base_params_c14()
        _, alpha_tw = solve_alpha_tw(alpha_n, alpha_t, 0.1817, 0.1715, 16, 24, db1, db2)
        assert alpha_tw > alpha_t

    def test_zero_shift_gives_standard_centre(self):
        """
        x1=x2=0 → αtw = αt → al = standard centre distance.
        For m=3, z=20/40, α=20°: a = (60+120)/2 = 90mm.
        """
        mn = 3.0
        z1, z2 = 20, 40
        alpha_n = math.radians(20.0)
        beta = 0.0
        mt = mn
        alpha_t = alpha_n
        d1, d2 = mt * z1, mt * z2
        a_std = (d1 + d2) / 2.0
        db1 = d1 * math.cos(alpha_t)
        db2 = d2 * math.cos(alpha_t)
        al, alpha_tw = solve_alpha_tw(alpha_n, alpha_t, 0.0, 0.0, z1, z2, db1, db2)
        assert al == pytest.approx(a_std, rel=1e-6)
        assert alpha_tw == pytest.approx(alpha_t, rel=1e-6)

    def test_invalid_shift_raises(self):
        """
        solve_alpha_tw raises ValueError when inv_target > inv(89°) ≈ 55.74,
        i.e. no solution exists in the bracket [αt/2, 89°].

        inv_target = inv(αt) + 2·tan(αn)·(x1+x2)/(z1+z2)
        With αn=20°, x_sum=200, z_sum=2:
          inv_target ≈ 72.8 > 55.74  → brentq bracket fails → ValueError.

        Note: x-range guards live in validate_geometry_inputs, not here.
        This test bypasses those guards to exercise the numerical solver path.
        """
        alpha_n = math.radians(20.0)
        alpha_t = math.radians(20.0)
        db1, db2 = 30.0, 30.0
        with pytest.raises(ValueError, match="involute"):
            solve_alpha_tw(alpha_n, alpha_t, 100.0, 100.0, 1, 1, db1, db2)


# ---------------------------------------------------------------------------
# contact_ratio_alpha()
# ---------------------------------------------------------------------------

class TestContactRatioAlpha:
    """
    C14 reference: εα = 1.46 (GEARpie report).
    Tolerance ±0.02 (last digit rounding in GEARpie output).
    """

    def _c14_eps_alpha(self):
        mn = 4.5
        z1, z2 = 16, 24
        alpha_n = math.radians(20.0)
        beta = 0.0
        mt = mn
        alpha_t = alpha_n
        d1, d2 = mt * z1, mt * z2
        db1 = d1 * math.cos(alpha_t)
        db2 = d2 * math.cos(alpha_t)
        al = 91.5
        cos_atw = (d1 + d2) / 2.0 * math.cos(alpha_t) / al
        alpha_tw = math.acos(min(cos_atw, 1.0))
        x1, x2 = 0.1817, 0.1715
        da1 = d1 + 2.0 * mn * (HAP + x1)
        da2 = d2 + 2.0 * mn * (HAP + x2)
        p_bt = math.pi * mt * math.cos(alpha_t)
        return contact_ratio_alpha(z1, z2, da1, da2, db1, db2, alpha_tw, p_bt)

    def test_c14_eps_alpha(self):
        assert self._c14_eps_alpha() == pytest.approx(1.46, abs=0.02)

    def test_eps_alpha_greater_than_one(self):
        """εα > 1 is required for continuous meshing."""
        assert self._c14_eps_alpha() > 1.0

    def test_symmetric_spur_no_shift(self):
        """
        Symmetric pair (z1=z2, x=0) — εα is well-defined and > 1.
        m=2, z=20/20, α=20°, al=a.
        """
        mn = 2.0
        z1 = z2 = 20
        alpha_t = math.radians(20.0)
        mt = mn
        d1 = d2 = mt * z1
        db1 = db2 = d1 * math.cos(alpha_t)
        al = d1   # standard centre = d (since u=1)
        alpha_tw = alpha_t
        da1 = da2 = d1 + 2.0 * mn * HAP
        p_bt = math.pi * mt * math.cos(alpha_t)
        eps = contact_ratio_alpha(z1, z2, da1, da2, db1, db2, alpha_tw, p_bt)
        assert eps > 1.0


# ---------------------------------------------------------------------------
# contact_ratio_beta()
# ---------------------------------------------------------------------------

class TestContactRatioAlphaBeta:
    def test_spur_zero(self):
        """β=0 → εβ=0 regardless of b."""
        assert contact_ratio_beta(b=30.0, beta_b=0.0, p_bt=14.0) == pytest.approx(0.0)

    def test_no_face_width(self):
        """b=0 → εβ=0."""
        beta_b = math.radians(14.0)
        assert contact_ratio_beta(b=0.0, beta_b=beta_b, p_bt=10.0) == pytest.approx(0.0)

    def test_helical_positive(self):
        """β>0, b>0 → εβ>0."""
        beta_b = math.radians(14.0)
        eps = contact_ratio_beta(b=30.0, beta_b=beta_b, p_bt=10.0)
        assert eps > 0.0

    def test_formula(self):
        """Direct formula check: εβ = b·tan(βb)/p_bt."""
        beta_b = math.radians(10.0)
        b, p_bt = 25.0, 12.0
        expected = b * math.tan(beta_b) / p_bt
        assert contact_ratio_beta(b, beta_b, p_bt) == pytest.approx(expected, rel=1e-10)


# ---------------------------------------------------------------------------
# undercut_z_min()
# ---------------------------------------------------------------------------

class TestUndercutZMin:
    def test_standard_alpha20_spur(self):
        """α=20°, β=0° → z_min=17 (Shigley §13-10, Tab. 13-11)."""
        assert undercut_z_min(20.0, 0.0) == 17

    def test_larger_alpha_lower_zmin(self):
        """Larger pressure angle → smaller z_min."""
        z25 = undercut_z_min(25.0, 0.0)
        z20 = undercut_z_min(20.0, 0.0)
        assert z25 < z20

    def test_helical_lower_zmin(self):
        """Helical gear (β>0) has lower z_min than equivalent spur."""
        z_helical = undercut_z_min(20.0, 15.0)
        z_spur    = undercut_z_min(20.0, 0.0)
        assert z_helical < z_spur

    def test_near_zero_alpha_returns_large(self):
        """
        Very small α → sin(αt) ≈ 0 → z_min = floor(2/sin²) is very large.
        The guard sin_at < 1e-9 returns 999 only for essentially zero angles.
        For α=0.001° (sin≈1.7e-5), z_min > 10_000 — verify the threshold.
        """
        z = undercut_z_min(0.001, 0.0)
        assert z > 10_000


# ---------------------------------------------------------------------------
# validate_geometry_inputs()
# ---------------------------------------------------------------------------

class TestValidateGeometryInputs:
    """All guard branches — each raises ValueError with a meaningful message."""

    _valid = dict(
        mn=3.0, z1=20, z2=40,
        alpha_n_deg=20.0, beta_deg=0.0,
        al=None, x1=0.0, x2=0.0,
    )

    def _call(self, **overrides):
        kwargs = {**self._valid, **overrides}
        validate_geometry_inputs(**kwargs)

    def test_valid_passes(self):
        self._call()   # must not raise

    def test_mn_zero(self):
        with pytest.raises(ValueError, match="mn"):
            self._call(mn=0.0)

    def test_mn_negative(self):
        with pytest.raises(ValueError, match="mn"):
            self._call(mn=-1.0)

    def test_z1_zero(self):
        with pytest.raises(ValueError, match="z1"):
            self._call(z1=0)

    def test_z2_zero(self):
        with pytest.raises(ValueError, match="z2"):
            self._call(z2=0)

    def test_alpha_zero(self):
        with pytest.raises(ValueError, match="alpha_n_deg"):
            self._call(alpha_n_deg=0.0)

    def test_alpha_90(self):
        with pytest.raises(ValueError, match="alpha_n_deg"):
            self._call(alpha_n_deg=90.0)

    def test_beta_negative(self):
        with pytest.raises(ValueError, match="beta_deg"):
            self._call(beta_deg=-1.0)

    def test_beta_90(self):
        with pytest.raises(ValueError, match="beta_deg"):
            self._call(beta_deg=90.0)

    def test_al_zero(self):
        with pytest.raises(ValueError, match="al"):
            self._call(al=0.0)

    def test_al_negative(self):
        with pytest.raises(ValueError, match="al"):
            self._call(al=-5.0)

    def test_x1_out_of_range_high(self):
        with pytest.raises(ValueError, match="x1"):
            self._call(x1=2.5)

    def test_x1_out_of_range_low(self):
        with pytest.raises(ValueError, match="x1"):
            self._call(x1=-2.5)

    def test_x2_out_of_range_high(self):
        with pytest.raises(ValueError, match="x2"):
            self._call(x2=3.0)

    def test_al_supplied_valid(self):
        self._call(al=90.0)   # must not raise