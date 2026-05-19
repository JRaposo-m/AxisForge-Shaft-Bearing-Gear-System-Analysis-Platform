"""
tests/test_solvers/test_stress_solve.py
Validation tests for StressSolver.solve() — Phase 1 implementation.

Reference: Shigley's MED, 10th ed., §7-4.

Test strategy:
  1. Build a synthetic MechanicalSystem with known geometry.
  2. Inject a synthetic StaticsResult with exact M_res and T values.
  3. Compare StressSolver.solve() output against hand-calculated values.

Hand-calculation basis:
  All expected values computed independently in this file using the same
  formulas as the solver. Deviations from Shigley textbook are expected
  because Shigley uses graphical chart reading for Kt; our solver uses the
  tabulated interpolation from stress.py.

  Tolerance: 0.5% (rtol=0.005) for self-consistency tests.
  Note on Kt: textbook chart-read Kt values may differ up to ~5% from our
  interpolation table; the validation tests use our own expected values.

Fixtures defined here (not in conftest.py — solver-specific geometry):
  - shaft_35mm_shoulder: single section d=35mm, shoulder at midspan
  - AISI_1020_CD: material approximated from Shigley Tab. A-20

Validation cases:
  test_solve_shigley_ex7_2_style:
    Geometry analogous to Shigley §7-4 example.
    d=35mm, r=3mm shoulder, Ma=55000 N·mm, Tm=35000 N·mm.
    Material: Sut=470 MPa, Sy=390 MPa. Finish: machined. ke=50% (ke=1.0).
    Expected: nf_goodman, nf_asme, ny from hand calc.

  test_solve_pure_bending_no_torsion:
    Tm=0 → σ'_m=0 → Goodman reduces to σ_a/Se'.
    nf_goodman = Se' / σ_a.

  test_solve_returns_most_critical_first:
    Two shoulders; solver must return lower nf first.

  test_solve_no_bending_moment_gives_inf:
    M_res=0 at shoulder → nf = inf.

  test_solve_sections_sorted_ascending:
    All finite nf values are in ascending order.

  test_solve_yielding_factor_consistent:
    ny = Sy / σ'_max, verified independently.

  test_solve_raises_on_invalid_system:
    Missing bearing → validate_or_raise fires.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from core.materials import Material
from core.shaft import Shaft, ShaftSection, Shoulder
from core.components import Bearing
from core.system import MechanicalSystem
from models.statics_result import StaticsResult
from models.stress_result import StressRaiserType, StressResult
from solvers.stress import (
    StressSolver,
    ka_surface_finish,
    kb_size,
    ke_reliability,
    kt_shoulder_bending,
    kt_shoulder_torsion,
    neuber_constant_sqrt_a,
    notch_sensitivity,
    kf_from_kt,
    # helpers re-imported for hand-calc cross-check
    ka_surface_finish,
    kb_size,
)

RTOL = 0.005   # 0.5% — self-consistency tolerance


# ---------------------------------------------------------------------------
# Helpers — build synthetic StaticsResult for injection
# ---------------------------------------------------------------------------

def _make_statics_result(
    shaft_length: float,
    M_res_const: float,
    T_const: float,
    n_points: int = 1000,
) -> StaticsResult:
    """
    Synthetic StaticsResult with uniform M_res and T along entire shaft.

    For unit testing StressSolver in isolation from StaticsSolver.
    V_xz, V_xy, M_xz, M_xy, axial_force set to zero (not used by StressSolver).
    """
    x = np.linspace(0.0, shaft_length, n_points)
    zeros = np.zeros(n_points)
    M_res = np.full(n_points, M_res_const)
    T_arr = np.full(n_points, T_const)
    return StaticsResult(
        x=x,
        V_xz=zeros,
        V_xy=zeros,
        M_xz=zeros,
        M_xy=zeros,
        M_res=M_res,
        T=T_arr,
        axial_force=zeros,
        reactions={},
    )


def _make_system_35mm(r_fillet: float = 2.0) -> MechanicalSystem:
    """
    Shaft with a single shoulder at x=200mm.
    Section left: L=200, d=50mm (larger side, D=50)
    Shoulder: D=50, d_small=35, r=r_fillet — step_height=(50-35)/2=7.5mm, r≤7.5
    Section right: L=200, d=35mm (smaller side, critical)
    Total: 400mm. Bearings at x=0 and x=400.

    r/d = r/35. For r=2: r/d=0.0571 (between 0.05 and 0.10 in table).
    The shoulder is at x=200. Stress is evaluated at d=35mm (diameter_small).
    """
    shaft = Shaft(name="ex72_shaft")
    shaft.add_section(ShaftSection(
        length=200.0, diameter=50.0, label="left",
        shoulder_right=Shoulder(
            fillet_radius=r_fillet,
            diameter_large=50.0,
            diameter_small=35.0,
        ),
    ))
    shaft.add_section(ShaftSection(
        length=200.0, diameter=35.0, label="right",
    ))

    system = MechanicalSystem(shaft=shaft, name="ex72", speed_rpm=1450.0)
    system.add_bearing(Bearing(position=0.0, C=50_000.0, C0=30_000.0,
                                arrangement="fixed", label="A"))
    system.add_bearing(Bearing(position=400.0, C=50_000.0, C0=30_000.0,
                                arrangement="floating", label="B"))
    return system


# Material approximating AISI 1020 CD (Shigley Tab. A-20)
AISI_1020_CD = Material(
    material_id="AISI_1020_CD",
    Sut=470.0,
    Sy=390.0,
    E=207.0,
    density=7850.0,
    description="AISI 1020 CD — Shigley §7-4 example material",
)


# ---------------------------------------------------------------------------
# Hand-calculated expected values (self-consistent with solver equations)
# ---------------------------------------------------------------------------

def _hand_calc_ex72(reliability_percent: float = 50.0) -> dict:
    """
    Reproduce the solver equations by hand for d=35, r=2, Ma=55000, Tm=35000,
    Sut=470, Sy=390, machined, ke at reliability_percent.

    Shoulder geometry: D=50, d_small=35, r=2 → step_height=7.5mm ≥ r ✓
    r/d = 2/35 = 0.05714

    Returns dict of expected values.
    """
    d = 35.0
    r = 2.0
    Ma = 55_000.0
    Tm = 35_000.0
    Sut = 470.0
    Sy = 390.0

    r_over_d = r / d
    Kt  = float(kt_shoulder_bending(r_over_d))
    Kts = float(kt_shoulder_torsion(r_over_d))

    q  = notch_sensitivity(r, Sut, "bending")
    qs = notch_sensitivity(r, Sut, "torsion")
    Kf  = kf_from_kt(Kt, q)
    Kfs = kf_from_kt(Kts, qs)

    Se_base = 0.5 * Sut  # 235 MPa
    ka = ka_surface_finish(Sut, "machined")
    kb = kb_size(d)
    ke = ke_reliability(reliability_percent)
    Se_prime = Se_base * ka * kb * 1.0 * 1.0 * ke

    sigma_a = (32.0 * Kf * Ma) / (math.pi * d**3)
    sigma_m = math.sqrt(3.0) * (16.0 * Kfs * Tm) / (math.pi * d**3)

    inv_n_goodman = (16.0 / (math.pi * d**3)) * (
        2.0 * Kf * Ma / Se_prime
        + math.sqrt(3.0) * Kfs * Tm / Sut
    )
    nf_goodman = 1.0 / inv_n_goodman

    inv_n2_asme = (16.0 / (math.pi * d**3))**2 * (
        4.0 * (Kf * Ma / Se_prime)**2
        + 3.0 * (Kfs * Tm / Sy)**2
    )
    nf_asme = 1.0 / math.sqrt(inv_n2_asme)

    sigma_max = math.sqrt(
        (32.0 * Kf * Ma / (math.pi * d**3))**2
        + 3.0 * (16.0 * Kfs * Tm / (math.pi * d**3))**2
    )
    ny = Sy / sigma_max

    return {
        "Kt": Kt, "Kts": Kts, "q": q, "qs": qs,
        "Kf": Kf, "Kfs": Kfs,
        "Se_prime": Se_prime,
        "sigma_a": sigma_a, "sigma_m": sigma_m,
        "nf_goodman": nf_goodman,
        "nf_asme": nf_asme,
        "ny": ny,
    }


# ---------------------------------------------------------------------------
# Test: Shigley §7-4 style — primary validation case
# ---------------------------------------------------------------------------

class TestSolveShigleySec74:
    """
    Validation against Shigley §7-4 shaft stress example.
    d=35mm, r=3mm shoulder, Ma=55000 N·mm, Tm=35000 N·mm.
    Material: AISI 1020 CD (Sut=470, Sy=390 MPa).
    Finish: machined. Reliability: 50% (ke=1.0).

    Note: Shigley textbook reads Kt from charts; this test uses our
    interpolation tables. Expected values are self-consistent hand calcs.
    Tolerance: 0.5%.
    """

    @pytest.fixture
    def system(self):
        return _make_system_35mm(r_fillet=2.0)

    @pytest.fixture
    def statics(self):
        return _make_statics_result(
            shaft_length=400.0,
            M_res_const=55_000.0,
            T_const=35_000.0,
        )

    @pytest.fixture
    def expected(self):
        return _hand_calc_ex72(reliability_percent=50.0)

    @pytest.fixture
    def result(self, system, statics):
        solver = StressSolver()
        return solver.solve(
            system, statics, AISI_1020_CD,
            finish="machined", reliability_percent=50.0,
        )

    def test_one_critical_section(self, result):
        """One shoulder → one critical section."""
        assert len(result.sections) == 1

    def test_section_at_correct_position(self, result):
        """Shoulder is at x=200mm (right end of first section)."""
        assert pytest.approx(result.sections[0].x, rel=1e-6) == 200.0

    def test_section_diameter(self, result):
        """Critical diameter = d_small = 35mm."""
        assert pytest.approx(result.sections[0].diameter, rel=1e-6) == 35.0

    def test_raiser_type_is_shoulder(self, result):
        assert result.sections[0].raiser_type == StressRaiserType.SHOULDER

    def test_Kf_self_consistent(self, result, expected):
        """Kf matches hand calculation."""
        np.testing.assert_allclose(
            result.sections[0].Kf, expected["Kf"], rtol=RTOL
        )

    def test_Kfs_self_consistent(self, result, expected):
        np.testing.assert_allclose(
            result.sections[0].Kfs, expected["Kfs"], rtol=RTOL
        )

    def test_Se_prime_self_consistent(self, result, expected):
        """Se' matches Marin hand calculation."""
        np.testing.assert_allclose(
            result.sections[0].Se_prime, expected["Se_prime"], rtol=RTOL
        )

    def test_sigma_a_self_consistent(self, result, expected):
        np.testing.assert_allclose(
            result.sections[0].sigma_a, expected["sigma_a"], rtol=RTOL
        )

    def test_sigma_m_self_consistent(self, result, expected):
        np.testing.assert_allclose(
            result.sections[0].sigma_m, expected["sigma_m"], rtol=RTOL
        )

    def test_nf_goodman_self_consistent(self, result, expected):
        """
        DE-Goodman nf matches hand calc within 0.5%.
        Shigley textbook (chart Kt): nf ≈ 2.5 (different d, different example).
        Our geometry produces nf ≈ 7.57 (large section, low Sut).
        """
        np.testing.assert_allclose(
            result.sections[0].nf_goodman, expected["nf_goodman"], rtol=RTOL
        )

    def test_nf_asme_self_consistent(self, result, expected):
        """DE-ASME Elliptic nf matches hand calc within 0.5%."""
        np.testing.assert_allclose(
            result.sections[0].nf_asme, expected["nf_asme"], rtol=RTOL
        )

    def test_ny_self_consistent(self, result, expected):
        """Yielding factor ny matches hand calc within 0.5%."""
        np.testing.assert_allclose(
            result.sections[0].ny, expected["ny"], rtol=RTOL
        )

    def test_asme_greater_than_goodman(self, result):
        """
        DE-ASME Elliptic always ≥ DE-Goodman for the same loads.
        (Elliptic criterion is less conservative than Goodman line.)
        Shigley §6-11.
        """
        assert result.sections[0].nf_asme >= result.sections[0].nf_goodman

    def test_Ma_Mm_Ta_Tm_phase1_values(self, result):
        """Phase 1: Mm=0, Ta=0 (rotating shaft)."""
        s = result.sections[0]
        assert s.Mm == 0.0
        assert s.Ta == 0.0
        assert pytest.approx(s.Ma, rel=RTOL) == 55_000.0
        assert pytest.approx(s.Tm, rel=RTOL) == 35_000.0

    def test_result_material_id(self, result):
        assert result.material_id == "AISI_1020_CD"

    def test_result_finish(self, result):
        assert result.finish == "machined"

    def test_most_critical_property(self, result):
        """most_critical returns the (only) section."""
        assert result.most_critical is result.sections[0]

    def test_is_safe_property_truthy(self, result):
        """All nf > 1 → is_safe = True."""
        assert result.sections[0].is_safe is True


# ---------------------------------------------------------------------------
# Test: pure bending, Tm=0
# ---------------------------------------------------------------------------

class TestSolvePureBending:
    """
    Pure bending, no torsion. σ'_m = 0.
    Goodman reduces to: 1/n = σ_a / Se'.
    ASME reduces to:    1/n = σ_a / Se'  (same, since σ_m=0).
    """

    @pytest.fixture
    def result(self):
        system = _make_system_35mm(r_fillet=2.0)
        statics = _make_statics_result(400.0, M_res_const=50_000.0, T_const=0.0)
        return StressSolver().solve(system, statics, AISI_1020_CD,
                                    finish="machined", reliability_percent=50.0)

    def test_sigma_m_is_zero(self, result):
        assert pytest.approx(result.sections[0].sigma_m, abs=1e-10) == 0.0

    def test_Tm_is_zero(self, result):
        assert result.sections[0].Tm == 0.0

    def test_goodman_equals_asme_pure_bending(self, result):
        """With σ_m=0: Goodman = ASME Elliptic (both reduce to σ_a/Se')."""
        s = result.sections[0]
        np.testing.assert_allclose(s.nf_goodman, s.nf_asme, rtol=RTOL)

    def test_nf_goodman_is_Se_prime_over_sigma_a(self, result):
        """
        nf = Se' / σ_a when Tm=0.
        Se' / σ_a = Se' × W / (Kf × Ma)  — verify via ratio.
        """
        s = result.sections[0]
        expected_nf = s.Se_prime / s.sigma_a
        np.testing.assert_allclose(s.nf_goodman, expected_nf, rtol=RTOL)


# ---------------------------------------------------------------------------
# Test: M_res = 0 at shoulder → nf = inf
# ---------------------------------------------------------------------------

class TestSolveNoBendingMoment:
    """Section with no bending moment → nf = inf (no fatigue failure from bending)."""

    @pytest.fixture
    def result(self):
        system = _make_system_35mm(r_fillet=2.0)
        statics = _make_statics_result(400.0, M_res_const=0.0, T_const=50_000.0)
        return StressSolver().solve(system, statics, AISI_1020_CD,
                                    finish="machined", reliability_percent=50.0)

    def test_nf_goodman_is_inf(self, result):
        assert math.isinf(result.sections[0].nf_goodman)

    def test_nf_asme_is_inf(self, result):
        assert math.isinf(result.sections[0].nf_asme)

    def test_section_in_infinite_list(self, result):
        """Inf sections are appended after finite ones."""
        # Only one section total, and it's inf
        assert len(result.sections) == 1
        assert math.isinf(result.sections[0].nf_goodman)

    def test_most_critical_returns_inf_section(self, result):
        """most_critical falls back to sections[0] when all are inf."""
        assert result.most_critical is result.sections[0]


# ---------------------------------------------------------------------------
# Test: two shoulders — sorted by nf_goodman ascending
# ---------------------------------------------------------------------------

class TestSolveTwoShouldersSorted:
    """
    Shaft with two shoulders at different positions.
    Higher M_res at one shoulder → lower nf → listed first.
    """

    @pytest.fixture
    def system_two_shoulders(self):
        """
        Three-section shaft, two shoulders.
        §1: L=150, d=50mm → shoulder_right r=2mm, D=50, d_small=40
        §2: L=100, d=40mm → shoulder_right r=2mm, D=40, d_small=35
        §3: L=150, d=35mm
        Bearings at x=0 and x=400.
        """
        shaft = Shaft(name="two_shoulder")
        shaft.add_section(ShaftSection(
            length=150.0, diameter=50.0, label="§1",
            shoulder_right=Shoulder(fillet_radius=2.0, diameter_large=50.0, diameter_small=40.0),
        ))
        shaft.add_section(ShaftSection(
            length=100.0, diameter=40.0, label="§2",
            shoulder_right=Shoulder(fillet_radius=2.0, diameter_large=40.0, diameter_small=35.0),
        ))
        shaft.add_section(ShaftSection(length=150.0, diameter=35.0, label="§3"))

        system = MechanicalSystem(shaft=shaft, name="two_sh", speed_rpm=1000.0)
        system.add_bearing(Bearing(position=0.0, C=50_000.0, C0=30_000.0,
                                    arrangement="fixed", label="A"))
        system.add_bearing(Bearing(position=400.0, C=50_000.0, C0=30_000.0,
                                    arrangement="floating", label="B"))
        return system

    @pytest.fixture
    def statics_gradient(self, system_two_shoulders):
        """
        M_res linearly increases from 10000 to 60000 N·mm along shaft.
        Shoulder at x=150: M_res ≈ 10000 + (60000-10000)*(150/400) ≈ 28750 N·mm
        Shoulder at x=250: M_res ≈ 10000 + (60000-10000)*(250/400) ≈ 41250 N·mm
        → x=250 has higher M, lower nf → must appear first.
        """
        L = 400.0
        x = np.linspace(0.0, L, 1000)
        M_res = 10_000.0 + (60_000.0 - 10_000.0) * (x / L)
        T_arr = np.full(1000, 20_000.0)
        zeros = np.zeros(1000)
        return StaticsResult(
            x=x, V_xz=zeros, V_xy=zeros, M_xz=zeros, M_xy=zeros,
            M_res=M_res, T=T_arr, axial_force=zeros, reactions={},
        )

    @pytest.fixture
    def result(self, system_two_shoulders, statics_gradient):
        return StressSolver().solve(
            system_two_shoulders, statics_gradient, AISI_1020_CD,
            finish="machined", reliability_percent=50.0,
        )

    def test_two_sections(self, result):
        assert len(result.sections) == 2

    def test_sorted_ascending(self, result):
        """nf_goodman must be non-decreasing (most critical first)."""
        nfs = [s.nf_goodman for s in result.sections if math.isfinite(s.nf_goodman)]
        assert nfs == sorted(nfs)

    def test_first_section_lower_nf(self, result):
        """Section with higher moment (x=250) has lower nf → listed first."""
        assert result.sections[0].nf_goodman <= result.sections[1].nf_goodman

    def test_most_critical_is_first(self, result):
        assert result.most_critical is result.sections[0]

    def test_min_nf_goodman_property(self, result):
        expected = min(s.nf_goodman for s in result.sections)
        assert pytest.approx(result.min_nf_goodman) == expected

    def test_min_ny_property(self, result):
        expected = min(s.ny for s in result.sections)
        assert pytest.approx(result.min_ny) == expected


# ---------------------------------------------------------------------------
# Test: invalid system → validate_or_raise fires
# ---------------------------------------------------------------------------

class TestSolveValidation:

    def test_raises_on_missing_bearing(self):
        """System with only one bearing → ValueError."""
        shaft = Shaft(name="bad")
        shaft.add_section(ShaftSection(length=400.0, diameter=40.0))
        system = MechanicalSystem(shaft=shaft, speed_rpm=1000.0)
        system.add_bearing(Bearing(position=0.0, C=1.0, C0=1.0, arrangement="fixed"))
        statics = _make_statics_result(400.0, M_res_const=10_000.0, T_const=5_000.0)
        with pytest.raises(ValueError, match="≥ 2 bearing"):
            StressSolver().solve(system, statics, AISI_1020_CD)

    def test_no_shoulders_returns_empty_result(self):
        """Shaft with no shoulders → empty StressResult."""
        shaft = Shaft(name="no_sh")
        shaft.add_section(ShaftSection(length=400.0, diameter=40.0))
        system = MechanicalSystem(shaft=shaft, speed_rpm=1000.0)
        system.add_bearing(Bearing(position=0.0, C=50_000.0, C0=30_000.0, arrangement="fixed"))
        system.add_bearing(Bearing(position=400.0, C=50_000.0, C0=30_000.0, arrangement="floating"))
        statics = _make_statics_result(400.0, M_res_const=10_000.0, T_const=5_000.0)
        result = StressSolver().solve(system, statics, AISI_1020_CD)
        assert result.sections == []
        assert result.most_critical is None
        assert math.isinf(result.min_nf_goodman)


# ---------------------------------------------------------------------------
# Test: StressResult properties
# ---------------------------------------------------------------------------

class TestStressResultProperties:

    @pytest.fixture
    def result(self):
        system = _make_system_35mm(r_fillet=2.0)
        statics = _make_statics_result(400.0, M_res_const=55_000.0, T_const=35_000.0)
        return StressSolver().solve(system, statics, AISI_1020_CD,
                                    finish="machined", reliability_percent=50.0)

    def test_governing_nf_is_min_of_goodman_and_asme(self, result):
        s = result.sections[0]
        assert s.governing_nf == min(s.nf_goodman, s.nf_asme)

    def test_min_nf_goodman_matches_sections(self, result):
        expected = min(s.nf_goodman for s in result.sections)
        assert pytest.approx(result.min_nf_goodman) == expected

    def test_min_nf_asme_matches_sections(self, result):
        expected = min(s.nf_asme for s in result.sections)
        assert pytest.approx(result.min_nf_asme) == expected

    def test_langer_ok_true_when_ny_ge_nf_goodman(self, result):
        s = result.sections[0]
        # ny >> nf_goodman for these loads
        assert s.langer_ok is True

    def test_langer_warning_when_ny_lt_nf_goodman(self):
        """Very high Ma relative to Sy → ny < nf_goodman → Langer warning issued."""
        system = _make_system_35mm(r_fillet=2.0)
        # Extremely high moment → ny < 1 → ny < nf_goodman (which may be >1 for high Se')
        # Use material with very high Se and low Sy to force ny < nf_goodman
        low_sy_material = Material(
            material_id="LOW_SY",
            Sut=2000.0,   # very high Sut → high Se → high nf_goodman
            Sy=50.0,      # very low Sy → low ny
            E=210.0,
            density=7850.0,
            Se_base=700.0,
        )
        statics = _make_statics_result(400.0, M_res_const=10_000.0, T_const=5_000.0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            StressSolver().solve(system, statics, low_sy_material,
                                  finish="machined", reliability_percent=50.0)
            langer_warnings = [x for x in w if "Langer" in str(x.message)]
            assert len(langer_warnings) >= 1
