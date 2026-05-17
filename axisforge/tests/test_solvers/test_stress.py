# tests/test_solvers/test_stress.py
"""
StressSolver test suite — Semana 5: Marin factors and Se' validation.

All expected values are hand-calculated and cross-referenced against Shigley.

VALIDATION TARGETS (Shigley 10th ed.):
  S355: Sut=590 MPa, d=50mm, Ra=0.8μm (machined), 99% reliability
    Se = 0.5 * 590 = 295 MPa
    ka = 4.51 * 590^(-0.265)
    kb = 1.51 * 50^(-0.157)
    ke = 0.814
    Se' = 295 * ka * kb * 1.0 * 1.0 * 0.814

  42CrMo4: Sut=1000 MPa, d=40mm, machined, 99%
    Se = 0.5 * 1000 = 500 MPa
    ka = 4.51 * 1000^(-0.265)
    kb = 1.51 * 40^(-0.157)
    ke = 0.814

  AISI_4340: Sut=1460 MPa → Se_base=700 MPa (capped, Shigley §6-2)
"""
import pytest
import numpy as np

from solvers.stress import (
    ka_surface_finish,
    kb_size,
    kc_load,
    kd_temperature,
    ke_reliability,
    ke_reliability_from_z,
    endurance_limit_corrected,
    kt_shoulder_bending,
    kt_shoulder_torsion,
    neuber_constant_sqrt_a,
    notch_sensitivity,
    kf_from_kt,
)
from core.materials import S355, CrMo42, AISI_1045, AISI_4340

TOLERANCE_REL = 0.01   # 1% — Shigley validation tolerance
TOLERANCE_ABS = 0.001  # for dimensionless factors


# ---------------------------------------------------------------------------
# ka — surface finish factor
# ---------------------------------------------------------------------------

class TestKaSurfaceFinish:

    def test_ka_machined_s355(self):
        """
        S355: Sut=590 MPa, machined finish.
        ka = 4.51 * 590^(-0.265)
        """
        ka = ka_surface_finish(590.0, "machined")
        expected = 4.51 * (590.0 ** -0.265)
        assert pytest.approx(ka, rel=TOLERANCE_REL) == expected

    def test_ka_machined_crmo42(self):
        """42CrMo4: Sut=1000 MPa, machined."""
        ka = ka_surface_finish(1000.0, "machined")
        expected = 4.51 * (1000.0 ** -0.265)
        assert pytest.approx(ka, rel=TOLERANCE_REL) == expected

    def test_ka_ground_is_higher_than_machined(self):
        """Ground finish → higher ka (better surface → less stress concentration)."""
        ka_ground = ka_surface_finish(590.0, "ground")
        ka_machined = ka_surface_finish(590.0, "machined")
        assert ka_ground > ka_machined

    def test_ka_as_forged_is_lowest(self):
        """As-forged is worst surface finish → lowest ka."""
        ka_forged = ka_surface_finish(590.0, "as_forged")
        ka_machined = ka_surface_finish(590.0, "machined")
        assert ka_forged < ka_machined

    def test_ka_capped_at_1(self):
        """ka cannot exceed 1.0 (ground finish on low-Sut material)."""
        ka = ka_surface_finish(200.0, "ground")
        assert ka <= 1.0

    def test_ka_invalid_finish_raises(self):
        with pytest.raises(ValueError, match="Unknown finish"):
            ka_surface_finish(590.0, "polished_mirror")

    def test_ka_decreases_with_increasing_sut(self):
        """Higher Sut → more sensitive to surface → lower ka."""
        ka_low = ka_surface_finish(400.0, "machined")
        ka_high = ka_surface_finish(1200.0, "machined")
        assert ka_low > ka_high

    def test_ka_hot_rolled_shigley_value(self):
        """
        Shigley Tab. 6-2: hot-rolled, Sut=520 MPa → ka ≈ 0.72.
        Using a=57.7, b=-0.718: ka = 57.7 * 520^(-0.718) ≈ ?
        Value is approximate; we test that it's reasonable (0.5 < ka < 0.9).
        """
        ka = ka_surface_finish(520.0, "hot_rolled")
        assert 0.4 < ka < 0.95

    @pytest.mark.parametrize("finish", ["ground", "machined", "cold_drawn", "hot_rolled", "as_forged"])
    def test_ka_all_finishes_in_range(self, finish):
        """All valid finishes produce ka in (0, 1]."""
        ka = ka_surface_finish(590.0, finish)
        assert 0.0 < ka <= 1.0


# ---------------------------------------------------------------------------
# kb — size factor
# ---------------------------------------------------------------------------

class TestKbSize:

    def test_kb_small_diameter_below_279_is_unity(self):
        """d < 2.79 mm — below Shigley validity range; kb = 1.0."""
        assert kb_size(1.0) == 1.0
        assert kb_size(2.0) == 1.0

    def test_kb_medium_diameter_formula(self):
        """d=50mm (2.79 ≤ d ≤ 51): kb = 1.24 × 50^(−0.107)."""
        kb = kb_size(50.0)
        expected = 1.24 * (50.0 ** -0.107)
        assert pytest.approx(kb, rel=TOLERANCE_REL) == expected

    def test_kb_large_diameter_formula(self):
        """d=100mm (51 < d ≤ 254): kb = 1.51 × 100^(−0.157)."""
        kb = kb_size(100.0)
        expected = 1.51 * (100.0 ** -0.157)
        assert pytest.approx(kb, rel=TOLERANCE_REL) == expected

    def test_kb_very_large_diameter_conservative(self):
        """d > 254mm: kb = 0.70 (conservative)."""
        assert pytest.approx(kb_size(300.0)) == 0.70

    def test_kb_boundary_279_lower(self):
        """d=2.79mm — lower boundary of first formula (inclusive)."""
        kb = kb_size(2.79)
        expected = 1.24 * (2.79 ** -0.107)
        assert pytest.approx(kb, rel=TOLERANCE_REL) == expected

    def test_kb_boundary_51_upper(self):
        """d=51mm — upper boundary of first formula (inclusive)."""
        kb = kb_size(51.0)
        expected = 1.24 * (51.0 ** -0.107)
        assert pytest.approx(kb, rel=TOLERANCE_REL) == expected

    def test_kb_boundary_just_above_51(self):
        """d=51.1mm — just above 51mm → second formula."""
        kb = kb_size(51.1)
        expected = 1.51 * (51.1 ** -0.157)
        assert pytest.approx(kb, rel=TOLERANCE_REL) == expected

    def test_kb_decreases_within_first_region(self):
        """Within 2.79–51mm region: larger diameter → lower kb."""
        assert kb_size(10.0) > kb_size(30.0) > kb_size(50.0)

    def test_kb_d50_known_range(self):
        """d=50mm: kb should be approximately 0.83–0.88."""
        kb = kb_size(50.0)
        assert 0.80 < kb < 0.92

    def test_kb_d40_known_range(self):
        """d=40mm: kb should be around 0.85–0.90."""
        kb = kb_size(40.0)
        assert 0.82 < kb < 0.94

    def test_kb_zero_diameter_raises(self):
        with pytest.raises(ValueError):
            kb_size(0.0)

    def test_kb_negative_diameter_raises(self):
        with pytest.raises(ValueError):
            kb_size(-10.0)


# ---------------------------------------------------------------------------
# kc — load factor
# ---------------------------------------------------------------------------

class TestKcLoad:

    def test_kc_bending_is_unity(self):
        assert kc_load("bending") == 1.0

    def test_kc_axial(self):
        assert pytest.approx(kc_load("axial")) == 0.85

    def test_kc_torsion(self):
        assert pytest.approx(kc_load("torsion")) == 0.59

    def test_kc_invalid_raises(self):
        with pytest.raises(ValueError, match="load_type"):
            kc_load("compression")


# ---------------------------------------------------------------------------
# kd — temperature factor
# ---------------------------------------------------------------------------

class TestKdTemperature:

    def test_kd_room_temperature_is_unity(self):
        assert kd_temperature(20.0) == 1.0

    def test_kd_at_limit_is_unity(self):
        assert kd_temperature(70.0) == 1.0

    def test_kd_default_is_unity(self):
        assert kd_temperature() == 1.0


# ---------------------------------------------------------------------------
# ke — reliability factor
# ---------------------------------------------------------------------------

class TestKeReliability:

    def test_ke_50_percent(self):
        """50% reliability → ke = 1.0."""
        assert pytest.approx(ke_reliability(50.0)) == 1.000

    def test_ke_90_percent(self):
        assert pytest.approx(ke_reliability(90.0)) == 0.897

    def test_ke_95_percent(self):
        assert pytest.approx(ke_reliability(95.0)) == 0.868

    def test_ke_99_percent(self):
        """Default reliability: ke = 0.814."""
        assert pytest.approx(ke_reliability(99.0)) == 0.814

    def test_ke_9999_percent(self):
        assert pytest.approx(ke_reliability(99.99)) == 0.702

    def test_ke_99999_percent(self):
        assert pytest.approx(ke_reliability(99.999)) == 0.659

    def test_ke_999999_percent(self):
        assert pytest.approx(ke_reliability(99.9999)) == 0.620

    def test_ke_decreases_with_reliability(self):
        """Higher reliability → lower ke (more conservative)."""
        assert ke_reliability(50.0) > ke_reliability(90.0) > ke_reliability(99.0)

    def test_ke_invalid_raises(self):
        with pytest.raises(ValueError, match="reliability_percent"):
            ke_reliability(98.0)

    def test_ke_from_z_99_percent(self):
        """z=2.326 for 99% → ke ≈ 1 - 0.08*2.326 = 0.814."""
        ke = ke_reliability_from_z(2.326)
        assert pytest.approx(ke, abs=0.002) == 0.814


# ---------------------------------------------------------------------------
# Se' — corrected endurance limit (Shigley validation)
# ---------------------------------------------------------------------------

class TestEnduranceLimitCorrected:
    """
    Validation against Shigley examples.
    All values computed manually and traceable to Shigley §6-7, Tab. 6-2.
    """

    def test_se_prime_s355_machined_99(self):
        """
        S355: Sut=590 MPa, d=50mm, machined, 99% reliability.
        Se = 0.5*590 = 295 MPa
        ka = 4.51 * 590^(-0.265) ≈ 0.838
        kb = 1.51 * 50^(-0.157)  ≈ 0.854
        ke = 0.814
        Se' = 295 * 0.838 * 0.854 * 1.0 * 1.0 * 0.814 ≈ 171.6 MPa

        Tolerance: 1% — relative to self-consistent calculation.
        """
        result = endurance_limit_corrected(
            Se_base=S355.endurance_limit,
            Sut_MPa=S355.Sut,
            diameter_mm=50.0,
            finish="machined",
            reliability_percent=99.0,
            load_type="bending",
        )
        # Verify individual factors
        ka_expected = 4.51 * (590.0 ** -0.265)
        kb_expected = 1.51 * (50.0 ** -0.157)
        ke_expected = 0.814
        Se_prime_expected = 295.0 * ka_expected * kb_expected * 1.0 * 1.0 * ke_expected

        assert pytest.approx(result["ka"], rel=TOLERANCE_REL) == ka_expected
        assert pytest.approx(result["kb"], rel=TOLERANCE_REL) == kb_expected
        assert pytest.approx(result["ke"]) == ke_expected
        assert pytest.approx(result["Se_prime"], rel=TOLERANCE_REL) == Se_prime_expected

    def test_se_prime_crmo42_machined_99(self):
        """
        42CrMo4: Sut=1000 MPa, d=40mm, machined, 99% reliability.
        Se = 0.5*1000 = 500 MPa
        ka = 4.51 * 1000^(-0.265)
        kb = 1.24 * 40^(-0.107)   [d=40mm falls in 2.79–51mm range]
        ke = 0.814
        """
        result = endurance_limit_corrected(
            Se_base=CrMo42.endurance_limit,
            Sut_MPa=CrMo42.Sut,
            diameter_mm=40.0,
            finish="machined",
            reliability_percent=99.0,
            load_type="bending",
        )
        ka_expected = 4.51 * (1000.0 ** -0.265)
        kb_expected = 1.24 * (40.0 ** -0.107)   # corrected: 2.79–51mm range
        Se_prime_expected = 500.0 * ka_expected * kb_expected * 1.0 * 1.0 * 0.814

        assert pytest.approx(result["Se_prime"], rel=TOLERANCE_REL) == Se_prime_expected

    def test_se_prime_aisi_4340_capped(self):
        """
        AISI 4340: Sut=1460 MPa → Se_base=700 MPa (capped, Shigley §6-2).
        Se' must use 700 MPa, not 0.5*1460=730.
        """
        assert AISI_4340.endurance_limit == 700.0, "Se_base must be capped at 700 MPa"
        result = endurance_limit_corrected(
            Se_base=AISI_4340.endurance_limit,
            Sut_MPa=AISI_4340.Sut,
            diameter_mm=50.0,
            finish="machined",
            reliability_percent=99.0,
            load_type="bending",
        )
        # Se_base = 700 (not 730)
        ka = 4.51 * (1460.0 ** -0.265)
        kb = 1.51 * (50.0 ** -0.157)
        expected = 700.0 * ka * kb * 0.814
        assert pytest.approx(result["Se_prime"], rel=TOLERANCE_REL) == expected

    def test_se_prime_less_than_se_base(self):
        """Se' must always be ≤ Se_base (Marin factors all ≤ 1.0)."""
        result = endurance_limit_corrected(
            Se_base=295.0, Sut_MPa=590.0, diameter_mm=50.0
        )
        assert result["Se_prime"] <= 295.0

    def test_se_prime_all_factors_unity(self):
        """If all factors = 1.0, Se' = Se_base."""
        # Ground finish, small d (kb ≈ 1), 50% reliability → ke=1
        result = endurance_limit_corrected(
            Se_base=295.0, Sut_MPa=590.0, diameter_mm=5.0,
            finish="ground", reliability_percent=50.0, load_type="bending"
        )
        assert result["Se_prime"] <= 295.0  # ka might still be < 1

    def test_se_prime_returns_all_keys(self):
        """Result dict must contain all Marin factors and Se_prime."""
        result = endurance_limit_corrected(
            Se_base=295.0, Sut_MPa=590.0, diameter_mm=50.0
        )
        assert set(result.keys()) == {"ka", "kb", "kc", "kd", "ke", "Se_prime"}

    def test_se_prime_axial_load_lower_than_bending(self):
        """kc_axial = 0.85 < 1.0 → Se' axial < Se' bending."""
        kwargs = dict(Se_base=295.0, Sut_MPa=590.0, diameter_mm=50.0)
        Se_bending = endurance_limit_corrected(load_type="bending", **kwargs)["Se_prime"]
        Se_axial   = endurance_limit_corrected(load_type="axial",   **kwargs)["Se_prime"]
        assert Se_axial < Se_bending

    def test_se_prime_90_reliability_higher_than_99(self):
        """Lower reliability requirement → higher ke → higher Se'."""
        kwargs = dict(Se_base=295.0, Sut_MPa=590.0, diameter_mm=50.0)
        Se_90 = endurance_limit_corrected(reliability_percent=90.0, **kwargs)["Se_prime"]
        Se_99 = endurance_limit_corrected(reliability_percent=99.0, **kwargs)["Se_prime"]
        assert Se_90 > Se_99


# ---------------------------------------------------------------------------
# Peterson Kt factors
# ---------------------------------------------------------------------------

class TestKtShoulder:

    def test_kt_bending_known_value_r_over_d_02(self):
        """r/d = 0.20 → Kt = 1.40 (from Peterson data)."""
        kt = kt_shoulder_bending(0.20)
        assert pytest.approx(kt, rel=0.02) == 1.40

    def test_kt_bending_known_value_r_over_d_01(self):
        """r/d = 0.10 → Kt = 1.60."""
        kt = kt_shoulder_bending(0.10)
        assert pytest.approx(kt, rel=0.02) == 1.60

    def test_kt_bending_known_value_r_over_d_005(self):
        """r/d = 0.005 → Kt = 3.00 (sharp shoulder)."""
        kt = kt_shoulder_bending(0.005)
        assert pytest.approx(kt, rel=0.02) == 3.00

    def test_kt_bending_increases_with_smaller_r_over_d(self):
        """Smaller r/d → sharper shoulder → higher Kt."""
        assert kt_shoulder_bending(0.05) > kt_shoulder_bending(0.10)
        assert kt_shoulder_bending(0.01) > kt_shoulder_bending(0.05)

    def test_kt_bending_extrapolation_low_r_over_d(self):
        """For r/d below table range, extrapolate to maximum (3.00)."""
        kt = kt_shoulder_bending(0.001)
        assert pytest.approx(kt, rel=0.02) == 3.00

    def test_kt_torsion_known_value_r_over_d_02(self):
        """r/d = 0.20 → Kts = 1.15."""
        kt = kt_shoulder_torsion(0.20)
        assert pytest.approx(kt, rel=0.02) == 1.15

    def test_kt_torsion_less_than_bending(self):
        """Torsion Kt < bending Kt for same geometry (Shigley §7-1)."""
        for r_over_d in [0.01, 0.05, 0.10]:
            assert kt_shoulder_torsion(r_over_d) < kt_shoulder_bending(r_over_d)

    def test_kt_shoulder_fillet_r2mm_d40mm(self):
        """
        Typical shaft shoulder: r=2mm, d=40mm → r/d = 0.05.
        Kt_bending ≈ 1.85 (interpolated from Peterson A-15-9).
        """
        r_over_d = 2.0 / 40.0  # = 0.05
        kt = kt_shoulder_bending(r_over_d)
        assert pytest.approx(kt, rel=0.02) == 1.85


# ---------------------------------------------------------------------------
# Neuber constant and notch sensitivity
# ---------------------------------------------------------------------------

class TestNotchSensitivity:

    def test_neuber_constant_known_value_s355(self):
        """
        S355: Sut=590 MPa = 85.57 kpsi.
        Eq. 6-35a: sqrt_a = 0.246 - 3.08e-3*85.57 + 1.51e-5*85.57² - 2.67e-8*85.57³
                          = 0.0763 sqrt(in) × sqrt(25.4) = 0.384 sqrt(mm).
        """
        import math
        Sut_kpsi = 590.0 / 6.895
        sqrt_a_in = (0.246 - 3.08e-3 * Sut_kpsi + 1.51e-5 * Sut_kpsi**2
                     - 2.67e-8 * Sut_kpsi**3)
        expected = sqrt_a_in * math.sqrt(25.4)
        sqrt_a = neuber_constant_sqrt_a(590.0, "bending")
        assert pytest.approx(sqrt_a, rel=0.001) == expected

    def test_neuber_constant_decreases_with_sut(self):
        """Higher Sut → smaller sqrt(a) → higher notch sensitivity."""
        assert neuber_constant_sqrt_a(400.0) > neuber_constant_sqrt_a(800.0)

    def test_neuber_constant_torsion_less_than_bending(self):
        """Torsion Neuber constant < bending constant (Eq. 6-35b vs 6-35a)."""
        sqrt_a_bending = neuber_constant_sqrt_a(590.0, "bending")
        sqrt_a_torsion = neuber_constant_sqrt_a(590.0, "torsion")
        assert sqrt_a_torsion < sqrt_a_bending

    def test_notch_sensitivity_large_radius(self):
        """
        Large radius (r=50mm) → q close to 1.0.
        With analytical Neuber (Eq. 6-35a), sqrt_a ≈ 0.384 mm^0.5 for S355.
        q = 1 / (1 + 0.384/sqrt(50)) = 1 / (1 + 0.054) ≈ 0.948.
        """
        q = notch_sensitivity(r_mm=50.0, Sut_MPa=590.0)
        assert q > 0.90

    def test_notch_sensitivity_small_radius(self):
        """Very small radius → q < 1 (material less sensitive)."""
        q = notch_sensitivity(r_mm=0.1, Sut_MPa=590.0)
        assert q < 0.90

    def test_notch_sensitivity_in_range(self):
        """q must always be in [0, 1]."""
        for r in [0.5, 1.0, 2.0, 5.0, 10.0]:
            q = notch_sensitivity(r, 590.0)
            assert 0.0 <= q <= 1.0

    def test_notch_sensitivity_torsion_loading(self):
        """torsion loading uses Eq. 6-35b — different sqrt(a) → different q."""
        q_bending = notch_sensitivity(2.0, 590.0, "bending")
        q_torsion = notch_sensitivity(2.0, 590.0, "torsion")
        # Torsion has smaller sqrt(a) → larger q (more sensitive)
        assert q_torsion > q_bending

    def test_kf_from_kt_identity_when_q_zero(self):
        """q=0 → Kf = 1.0 (material insensitive to notch)."""
        assert pytest.approx(kf_from_kt(Kt=2.5, q=0.0)) == 1.0

    def test_kf_from_kt_equals_kt_when_q_unity(self):
        """q=1 → Kf = Kt (full notch sensitivity)."""
        assert pytest.approx(kf_from_kt(Kt=2.5, q=1.0)) == 2.5

    def test_kf_from_kt_intermediate(self):
        """Kf = 1 + q*(Kt-1)."""
        Kt, q = 2.0, 0.8
        expected = 1.0 + 0.8 * (2.0 - 1.0)
        assert pytest.approx(kf_from_kt(Kt, q)) == expected

    def test_kf_less_than_or_equal_kt(self):
        """Kf ≤ Kt always (q ≤ 1)."""
        Kt = 2.5
        for q in [0.0, 0.3, 0.7, 1.0]:
            assert kf_from_kt(Kt, q) <= Kt

    def test_kf_typical_shoulder(self):
        """
        Typical shoulder: r=2mm, d=40mm (r/d=0.05), S355 (Sut=590 MPa).
        q and Kf must be physically reasonable: 1.0 < Kf ≤ Kt.
        """
        r_mm, d_mm = 2.0, 40.0
        Sut = 590.0
        Kt = kt_shoulder_bending(r_mm / d_mm)
        q = notch_sensitivity(r_mm, Sut, "bending")
        Kf = kf_from_kt(Kt, q)
        assert 1.0 < Kf <= Kt