"""
tests/solvers/shaft/test_shaft_utils.py
Unit tests for solvers/shaft/utils.py

Coverage:
  - ka_surface_finish      : Marin Tab. 6-2
  - kb_size                : Shigley Eq. 6-20 — all diameter regimes
  - kc_load                : Shigley Tab. 6-3
  - kd_temperature         : Phase 1 stub
  - ke_reliability         : Shigley Tab. 6-6
  - ke_reliability_from_z  : Shigley §6-6 formula
  - endurance_limit_corrected : full Marin chain
  - kt_shoulder_bending    : Peterson Fig. A-15-9
  - kt_shoulder_torsion    : Peterson Fig. A-15-10
  - neuber_constant_sqrt_a : Shigley Eq. 6-35a/b
  - notch_sensitivity      : Shigley Eq. 6-34
  - kf_from_kt             : Shigley Eq. 6-32

All expected values are independently hand-calculated or taken directly
from Shigley 10th ed. tables. Tolerances: ±0.5% for formula results,
exact for table lookups.
"""
from __future__ import annotations

import math
import pytest

from solvers.shaft.utils import (
    _KA_COEFFICIENTS,
    _KE_TABLE,
    endurance_limit_corrected,
    ka_surface_finish,
    kb_size,
    kc_load,
    kd_temperature,
    ke_reliability,
    ke_reliability_from_z,
    kf_from_kt,
    kt_shoulder_bending,
    kt_shoulder_torsion,
    neuber_constant_sqrt_a,
    notch_sensitivity,
)

TOL = 0.005   # 0.5% relative tolerance for formula-derived values


# ===========================================================================
# ka — surface finish factor
# ===========================================================================

class TestKaSurfaceFinish:

    def test_machined_590mpa(self):
        # S355: Sut=590 MPa, machined → a=4.51, b=-0.265
        # ka = 4.51 * 590^(-0.265) = 4.51 / 590^0.265
        expected = 4.51 * (590.0 ** -0.265)
        assert ka_surface_finish(590.0, "machined") == pytest.approx(expected, rel=TOL)

    def test_ground_finish_clips_to_one(self):
        # Ground on soft material (low Sut) can produce ka > 1 — must clip.
        ka = ka_surface_finish(200.0, "ground")
        assert ka == pytest.approx(1.0, abs=1e-9)

    def test_ground_high_sut(self):
        # 1000 MPa ground: ka = 1.58 * 1000^(-0.085) < 1
        expected = min(1.0, 1.58 * (1000.0 ** -0.085))
        assert ka_surface_finish(1000.0, "ground") == pytest.approx(expected, rel=TOL)

    def test_as_forged_590mpa(self):
        expected = min(1.0, 272.0 * (590.0 ** -0.995))
        assert ka_surface_finish(590.0, "as_forged") == pytest.approx(expected, rel=TOL)

    def test_hot_rolled_and_cold_drawn_available(self):
        # Both should return a value in (0, 1]
        for finish in ("hot_rolled", "cold_drawn"):
            ka = ka_surface_finish(600.0, finish)
            assert 0 < ka <= 1.0

    def test_cold_drawn_equals_machined(self):
        # Same coefficients per Shigley Tab. 6-2
        assert ka_surface_finish(500.0, "cold_drawn") == pytest.approx(
            ka_surface_finish(500.0, "machined"), rel=1e-9
        )

    def test_all_finish_keys_accepted(self):
        for finish in _KA_COEFFICIENTS:
            ka = ka_surface_finish(700.0, finish)
            assert 0 < ka <= 1.0

    def test_invalid_finish_raises(self):
        with pytest.raises(ValueError, match="Unknown finish"):
            ka_surface_finish(500.0, "polished")

    def test_output_strictly_positive(self):
        assert ka_surface_finish(1500.0, "as_forged") > 0


# ===========================================================================
# kb — size factor
# ===========================================================================

class TestKbSize:
    """
    Shigley Eq. 6-20 (SI, rotating round shaft in bending):
      d < 2.79 mm             : kb = 1.0   (correlation does not extend below)
      2.79 ≤ d ≤ 51 mm        : kb = 1.24 * d^(-0.107)
      51 < d ≤ 254 mm         : kb = 1.51 * d^(-0.157)
      d > 254 mm              : kb = 0.70  (conservative, Shigley note)

    Note: the formula 1.24*d^(-0.107) exceeds 1.0 for d < ~8 mm because
    Shigley's regression is calibrated for d ≥ 2.79 mm — the upper bound
    of 1.0 is NOT enforced by the standard for this regime.  Only the
    d < 2.79 mm shortcut returns exactly 1.0.
    """

    def test_small_diameter_below_2_79_returns_one(self):
        assert kb_size(2.0) == pytest.approx(1.0)
        assert kb_size(1.0) == pytest.approx(1.0)

    def test_at_boundary_2_79(self):
        # d=2.79 mm is the first point of the mid-range formula (2.79 ≤ d ≤ 51).
        # The implementation uses  d < 2.79 → 1.0, so d=2.79 uses the formula.
        expected = 1.24 * (2.79 ** -0.107)
        assert kb_size(2.79) == pytest.approx(expected, rel=TOL)

    def test_mid_range_below_51mm(self):
        # d=40mm: kb = 1.24 * 40^(-0.107)
        expected = 1.24 * (40.0 ** -0.107)
        assert kb_size(40.0) == pytest.approx(expected, rel=TOL)

    def test_mid_range_50mm(self):
        expected = 1.24 * (50.0 ** -0.107)
        assert kb_size(50.0) == pytest.approx(expected, rel=TOL)

    def test_upper_range_above_51mm(self):
        # d=100mm: kb = 1.51 * 100^(-0.157)
        expected = 1.51 * (100.0 ** -0.157)
        assert kb_size(100.0) == pytest.approx(expected, rel=TOL)

    def test_upper_range_at_254mm(self):
        expected = 1.51 * (254.0 ** -0.157)
        assert kb_size(254.0) == pytest.approx(expected, rel=TOL)

    def test_above_254mm_returns_conservative(self):
        assert kb_size(300.0) == pytest.approx(0.70)
        assert kb_size(1000.0) == pytest.approx(0.70)

    def test_result_bounded_practical_range(self):
        # For d ≥ 20 mm the formula reliably returns kb ≤ 1.0.
        # d < ~8 mm can exceed 1.0 by Shigley's regression — not a bug.
        # d=250mm is in the range 51<d≤254 → formula gives ~0.635; 0.70 floor
        # only applies for d > 254mm (Shigley conservative cap).
        for d in [20, 50, 100]:
            assert 0.70 <= kb_size(float(d)) <= 1.0
        # Transition region: upper range formula, no 0.70 floor guarantee
        assert kb_size(250.0) <= 1.0
        assert kb_size(254.0) <= 1.0
        # Large d always hits the conservative floor
        assert kb_size(300.0) == pytest.approx(0.70)

    def test_zero_diameter_raises(self):
        with pytest.raises(ValueError):
            kb_size(0.0)

    def test_negative_diameter_raises(self):
        with pytest.raises(ValueError):
            kb_size(-10.0)


# ===========================================================================
# kc — load type factor
# ===========================================================================

class TestKcLoad:

    def test_bending(self):
        assert kc_load("bending") == pytest.approx(1.0)

    def test_axial(self):
        assert kc_load("axial") == pytest.approx(0.85)

    def test_torsion(self):
        assert kc_load("torsion") == pytest.approx(0.59)

    def test_default_is_bending(self):
        assert kc_load() == pytest.approx(1.0)

    def test_invalid_load_type_raises(self):
        with pytest.raises(ValueError, match="load_type"):
            kc_load("shear")


# ===========================================================================
# kd — temperature factor (Phase 1 stub)
# ===========================================================================

class TestKdTemperature:

    def test_room_temperature(self):
        assert kd_temperature(20.0) == pytest.approx(1.0)

    def test_elevated_still_returns_one_phase1(self):
        # Phase 1: no degradation modelled
        assert kd_temperature(150.0) == pytest.approx(1.0)

    def test_default_is_room_temperature(self):
        assert kd_temperature() == pytest.approx(1.0)


# ===========================================================================
# ke — reliability factor
# ===========================================================================

class TestKeReliability:

    # Exact table values — Shigley Tab. 6-6
    @pytest.mark.parametrize("reliability, expected", [
        (50.0,    1.000),
        (90.0,    0.897),
        (95.0,    0.868),
        (99.0,    0.814),
        (99.9,    0.753),
        (99.99,   0.702),
        (99.999,  0.659),
        (99.9999, 0.620),
    ])
    def test_table_values(self, reliability, expected):
        assert ke_reliability(reliability) == pytest.approx(expected, abs=1e-9)

    def test_default_is_99_percent(self):
        assert ke_reliability() == pytest.approx(0.814, abs=1e-9)

    def test_invalid_reliability_raises(self):
        with pytest.raises(ValueError, match="reliability_percent"):
            ke_reliability(98.0)

    def test_all_values_lte_one(self):
        for val in _KE_TABLE.values():
            assert val <= 1.0

    def test_monotone_decreasing(self):
        # Higher reliability → lower ke
        reliabilities = sorted(_KE_TABLE.keys())
        ke_values = [_KE_TABLE[r] for r in reliabilities]
        for i in range(len(ke_values) - 1):
            assert ke_values[i] >= ke_values[i + 1]


class TestKeReliabilityFromZ:

    def test_z_1_645_gives_approx_95_percent(self):
        # z=1.645 → 95% → ke ≈ 0.868; formula: 1 - 0.08*1.645 = 0.8684
        expected = 1.0 - 0.08 * 1.645
        assert ke_reliability_from_z(1.645) == pytest.approx(expected, rel=TOL)

    def test_z_zero_gives_one(self):
        assert ke_reliability_from_z(0.0) == pytest.approx(1.0)

    def test_z_2_326_gives_approx_99_percent(self):
        # z=2.326 → 99% → formula: 1 - 0.08*2.326 = 0.8139
        expected = 1.0 - 0.08 * 2.326
        assert ke_reliability_from_z(2.326) == pytest.approx(expected, abs=0.002)


# ===========================================================================
# endurance_limit_corrected — full Marin chain
# ===========================================================================

class TestEnduranceLimitCorrected:
    """
    Reference: Shigley 10th ed. Example 6-14 / §6-6.

    S355: Sut=590 MPa → Se_base = 0.5*590 = 295 MPa
    d=40mm, machined, 99% reliability, bending, 20°C
    """

    def _se_prime_s355_40mm(self):
        Sut = 590.0
        Se_base = 0.5 * Sut
        result = endurance_limit_corrected(
            Se_base=Se_base,
            Sut_MPa=Sut,
            diameter_mm=40.0,
            finish="machined",
            reliability_percent=99.0,
            load_type="bending",
        )
        return result

    def test_keys_present(self):
        r = self._se_prime_s355_40mm()
        for key in ("ka", "kb", "kc", "kd", "ke", "Se_prime"):
            assert key in r

    def test_ka_value(self):
        r = self._se_prime_s355_40mm()
        expected_ka = 4.51 * (590.0 ** -0.265)
        assert r["ka"] == pytest.approx(expected_ka, rel=TOL)

    def test_kb_value(self):
        r = self._se_prime_s355_40mm()
        expected_kb = 1.24 * (40.0 ** -0.107)
        assert r["kb"] == pytest.approx(expected_kb, rel=TOL)

    def test_kc_bending_is_one(self):
        r = self._se_prime_s355_40mm()
        assert r["kc"] == pytest.approx(1.0)

    def test_ke_99_percent(self):
        r = self._se_prime_s355_40mm()
        assert r["ke"] == pytest.approx(0.814, abs=1e-9)

    def test_se_prime_product(self):
        # Se' = Se * ka * kb * kc * kd * ke — verify end-to-end
        r = self._se_prime_s355_40mm()
        manual = 295.0 * r["ka"] * r["kb"] * r["kc"] * r["kd"] * r["ke"]
        assert r["Se_prime"] == pytest.approx(manual, rel=1e-9)

    def test_se_prime_strictly_less_than_se_base(self):
        r = self._se_prime_s355_40mm()
        assert r["Se_prime"] < 295.0

    def test_axial_load_reduces_se(self):
        r_bending = endurance_limit_corrected(295.0, 590.0, 40.0, load_type="bending")
        r_axial   = endurance_limit_corrected(295.0, 590.0, 40.0, load_type="axial")
        assert r_axial["Se_prime"] < r_bending["Se_prime"]

    def test_ground_finish_gives_higher_se_than_hot_rolled(self):
        kw = {"Se_base": 295.0, "Sut_MPa": 590.0, "diameter_mm": 40.0}
        r_ground = endurance_limit_corrected(**kw, finish="ground")
        r_hot    = endurance_limit_corrected(**kw, finish="hot_rolled")
        assert r_ground["Se_prime"] > r_hot["Se_prime"]

    def test_high_reliability_reduces_se(self):
        kw = {"Se_base": 295.0, "Sut_MPa": 590.0, "diameter_mm": 40.0}
        r_99   = endurance_limit_corrected(**kw, reliability_percent=99.0)
        r_9999 = endurance_limit_corrected(**kw, reliability_percent=99.99)
        assert r_9999["Se_prime"] < r_99["Se_prime"]

    def test_se_prime_positive(self):
        r = self._se_prime_s355_40mm()
        assert r["Se_prime"] > 0


# ===========================================================================
# Peterson Kt — shoulder bending and torsion
# ===========================================================================

class TestKtShoulderBending:
    """
    Reference data: Peterson Fig. A-15-9, D/d=1.5 curve.
    Values at r/d knot points are exact (they ARE the table).
    """

    @pytest.mark.parametrize("r_over_d, expected_kt", [
        (0.005, 3.00),
        (0.01,  2.70),
        (0.02,  2.30),
        (0.03,  2.10),
        (0.05,  1.85),
        (0.10,  1.60),
        (0.20,  1.40),
    ])
    def test_knot_points_exact(self, r_over_d, expected_kt):
        assert kt_shoulder_bending(r_over_d) == pytest.approx(expected_kt, abs=0.01)

    def test_interpolation_between_knots(self):
        # At r/d=0.015, midpoint between 0.01 and 0.02 → Kt ≈ midpoint(2.70,2.30)=2.50
        kt = kt_shoulder_bending(0.015)
        assert kt == pytest.approx(2.50, abs=0.05)

    def test_clamp_below_range(self):
        # r/d < 0.005 → clamped to 3.00
        assert kt_shoulder_bending(0.001) == pytest.approx(3.00, abs=1e-9)

    def test_clamp_above_range(self):
        # r/d > 0.20 → clamped to 1.40
        assert kt_shoulder_bending(0.50) == pytest.approx(1.40, abs=1e-9)

    def test_kt_gte_one(self):
        for r_over_d in [0.001, 0.01, 0.05, 0.10, 0.30]:
            assert kt_shoulder_bending(r_over_d) >= 1.0

    def test_d_over_d_parameter_accepted_phase1(self):
        # Phase 1 ignores D_over_d — must not raise
        kt1 = kt_shoulder_bending(0.05, D_over_d=1.5)
        kt2 = kt_shoulder_bending(0.05, D_over_d=2.0)
        assert kt1 == kt2   # same result — D/d ignored in Phase 1

    def test_monotone_decreasing(self):
        r_values = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
        kt_values = [kt_shoulder_bending(r) for r in r_values]
        for i in range(len(kt_values) - 1):
            assert kt_values[i] >= kt_values[i + 1]


class TestKtShoulderTorsion:
    """
    Reference data: Peterson Fig. A-15-10, D/d=1.5 curve.
    """

    @pytest.mark.parametrize("r_over_d, expected_kts", [
        (0.005, 2.00),
        (0.01,  1.85),
        (0.02,  1.65),
        (0.03,  1.55),
        (0.05,  1.40),
        (0.10,  1.25),
        (0.20,  1.15),
    ])
    def test_knot_points_exact(self, r_over_d, expected_kts):
        assert kt_shoulder_torsion(r_over_d) == pytest.approx(expected_kts, abs=0.01)

    def test_clamp_below_range(self):
        assert kt_shoulder_torsion(0.001) == pytest.approx(2.00, abs=1e-9)

    def test_clamp_above_range(self):
        assert kt_shoulder_torsion(0.50) == pytest.approx(1.15, abs=1e-9)

    def test_kts_gte_one(self):
        for r_over_d in [0.001, 0.01, 0.05, 0.20, 0.50]:
            assert kt_shoulder_torsion(r_over_d) >= 1.0

    def test_torsion_lower_than_bending_at_same_r_over_d(self):
        # Bending Kt always ≥ torsion Kts for shoulders
        for r_over_d in [0.01, 0.05, 0.10, 0.20]:
            assert kt_shoulder_bending(r_over_d) >= kt_shoulder_torsion(r_over_d)


# ===========================================================================
# Neuber constant sqrt(a)
# ===========================================================================

class TestNeuberConstantSqrtA:
    """
    Reference: Shigley Eq. 6-35a/b.
    Hand-calculated at Sut = 690 MPa (≈ 100 kpsi) for bending.

    Sut_kpsi = 690 / 6.895 = 100.07 kpsi

    Eq. 6-35a evaluated:
      sqrt_a_in = 0.246 - 3.08e-3*(100.07)
                        + 1.51e-5*(100.07)²
                        - 2.67e-8*(100.07)³
                = 0.246 - 0.30822 + 0.15121 - 0.02677
                = 0.06222  sqrt(in)

    Converted to mm^0.5:
      sqrt_a_mm = 0.06222 * sqrt(25.4) = 0.06222 * 5.0398 ≈ 0.3136  mm^0.5

    Common error: the cubic term is 2.67e-8 * 100³ = 0.0267, not 0.00267.
    """

    def test_bending_at_100kpsi(self):
        # Sut = 690 MPa ≈ 100 kpsi → sqrt(a) ≈ 0.3136 mm^0.5
        sqrt_a = neuber_constant_sqrt_a(690.0, loading="bending")
        assert sqrt_a == pytest.approx(0.3136, abs=0.005)

    def test_torsion_lower_than_bending(self):
        # Torsion Neuber constant is smaller → higher notch sensitivity
        a_bend = neuber_constant_sqrt_a(700.0, loading="bending")
        a_tors = neuber_constant_sqrt_a(700.0, loading="torsion")
        assert a_tors < a_bend

    def test_higher_sut_smaller_sqrt_a(self):
        # Higher Sut → harder material → more notch-sensitive → smaller sqrt(a)
        a_low  = neuber_constant_sqrt_a(400.0, loading="bending")
        a_high = neuber_constant_sqrt_a(900.0, loading="bending")
        assert a_low > a_high

    def test_default_loading_is_bending(self):
        assert neuber_constant_sqrt_a(700.0) == neuber_constant_sqrt_a(700.0, "bending")

    def test_result_positive_in_practical_range(self):
        for Sut in [300, 500, 700, 900, 1200]:
            assert neuber_constant_sqrt_a(float(Sut)) > 0


# ===========================================================================
# Notch sensitivity q
# ===========================================================================

class TestNotchSensitivity:

    def test_large_radius_approaches_one(self):
        # Very large r → q → 1 (fully sensitive)
        q = notch_sensitivity(r_mm=50.0, Sut_MPa=700.0, loading="bending")
        assert q > 0.95

    def test_q_bounded(self):
        for r in [0.1, 0.5, 1.0, 2.0, 5.0]:
            q = notch_sensitivity(r_mm=r, Sut_MPa=700.0)
            assert 0.0 < q <= 1.0

    def test_higher_sut_higher_q_same_radius(self):
        # Harder material (higher Sut) → more notch-sensitive → higher q
        q_low  = notch_sensitivity(r_mm=1.0, Sut_MPa=400.0)
        q_high = notch_sensitivity(r_mm=1.0, Sut_MPa=900.0)
        assert q_high > q_low

    def test_torsion_q_differs_from_bending(self):
        # Different Neuber constants → different q
        q_bend = notch_sensitivity(r_mm=1.0, Sut_MPa=700.0, loading="bending")
        q_tors = notch_sensitivity(r_mm=1.0, Sut_MPa=700.0, loading="torsion")
        assert q_bend != q_tors

    def test_larger_radius_gives_higher_q(self):
        q1 = notch_sensitivity(r_mm=0.5, Sut_MPa=600.0)
        q2 = notch_sensitivity(r_mm=2.0, Sut_MPa=600.0)
        assert q2 > q1


# ===========================================================================
# Kf from Kt
# ===========================================================================

class TestKfFromKt:

    def test_q_zero_gives_kf_one(self):
        # q=0 (insensitive) → Kf = 1 + 0*(Kt-1) = 1.0
        assert kf_from_kt(Kt=2.5, q=0.0) == pytest.approx(1.0)

    def test_q_one_gives_kf_equals_kt(self):
        # q=1 (fully sensitive) → Kf = Kt
        assert kf_from_kt(Kt=2.5, q=1.0) == pytest.approx(2.5)

    def test_intermediate_q(self):
        # q=0.80, Kt=2.3 → Kf = 1 + 0.80*(2.3-1) = 1 + 1.04 = 2.04
        assert kf_from_kt(Kt=2.3, q=0.80) == pytest.approx(2.04, rel=TOL)

    def test_kf_between_one_and_kt(self):
        for q in [0.0, 0.25, 0.5, 0.75, 1.0]:
            kf = kf_from_kt(Kt=3.0, q=q)
            assert 1.0 <= kf <= 3.0

    def test_kt_one_always_gives_kf_one(self):
        # No stress concentration (Kt=1) → Kf=1 regardless of q
        for q in [0.0, 0.5, 1.0]:
            assert kf_from_kt(Kt=1.0, q=q) == pytest.approx(1.0)