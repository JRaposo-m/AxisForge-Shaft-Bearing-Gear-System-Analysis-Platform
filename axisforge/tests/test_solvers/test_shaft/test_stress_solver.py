"""
tests/solvers/shaft/test_stress_solver.py
Integration tests for StressSolver (solvers/shaft/stress.py).

Validates:
  - StressSolver.solve() produces correct StressResult structure
  - Kf, Kfs, Se_prime, sigma_a, sigma_m at known sections
  - nf_goodman, nf_asme, ny computation
  - Sorting: most critical (lowest nf_goodman) first
  - Sections with M_res < 1 N·mm → nf = inf, appended last
  - Langer check fires correctly
  - kc = 1.0 always in StressSolver (combined loading)

Reference system (std_system fixture):
  Shaft 400mm: sections d=40/50/40mm.
  shoulder_right on §1 (x=100mm, r=2mm, 50→40mm transition).
  shoulder_right on §2 (x=300mm, r=2mm, 50→40mm transition).
  Bearings at x=50mm (fixed) and x=350mm (floating).
  Gear at x=200mm: Wt=3500N, Wr=1274N, T=175000 N·mm.
  Material: S355 (Sut=590MPa, Sy=355MPa, Se_base=295MPa).

Shoulder placement rule:
  shaft.shoulders() iterates section.shoulder_right for each section.
  A shoulder_right on section[i] marks the transition at axial_end(i),
  where diameter_small = section[i+1].diameter (the smaller section
  that carries the full stress concentration from the step).
  Therefore the shoulder_right must sit on the LARGER section,
  with diameter_small equal to the diameter of the following section.

Hand-calculated reference values used for critical assertions.
"""
from __future__ import annotations

import math
import pytest

from core.materials import AISI_1045, AISI_4340, S355, CrMo42
from core.shaft import Shaft, ShaftSection, Shoulder
from core.system import MechanicalSystem
from core.components import Bearing, GearElement
from models.stress_result import StressRaiserType
from solvers.shaft.statics import StaticsSolver
from solvers.shaft.stress import StressSolver
from solvers.shaft.utils import (
    endurance_limit_corrected,
    kf_from_kt,
    kt_shoulder_bending,
    kt_shoulder_torsion,
    notch_sensitivity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(system, material=S355, finish="machined", reliability=99.0):
    """Convenience: run statics + stress on a system."""
    statics = StaticsSolver().solve(system)
    return StressSolver().solve(system, statics, material, finish, reliability)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def std_system():
    """
    Standard 3-section shaft with two shoulders and symmetric loading.

    Geometry:
      §1: d=40mm, length=100mm, shoulder_right at x=100mm (50→40 transition).
      §2: d=50mm, length=200mm, shoulder_right at x=300mm (50→40 transition).
      §3: d=40mm, length=100mm.

    Shoulders use shoulder_right on the larger section (d=50mm side),
    with diameter_small=40mm pointing to the following smaller section.
    shaft.shoulders() therefore returns shoulders at x=100mm and x=300mm.

    Bearings: x=50mm (fixed, label="A"), x=350mm (floating, label="B").
    Gear: x=200mm, Wt=3500N, Wr=1274N, T=175000 N·mm, d_pitch=100mm.
    """
    shaft = Shaft(name="std")
    shaft.add_section(ShaftSection(
        length=100.0, diameter=50.0, label="§1",
        shoulder_right=Shoulder(
            fillet_radius=2.0, diameter_large=50.0, diameter_small=40.0
        ),
    ))
    shaft.add_section(ShaftSection(length=100.0, diameter=40.0, label="§2"))
    shaft.add_section(ShaftSection(
        length=100.0, diameter=50.0, label="§3",
        shoulder_right=Shoulder(
            fillet_radius=2.0, diameter_large=50.0, diameter_small=40.0
        ),
    ))
    shaft.add_section(ShaftSection(length=100.0, diameter=40.0, label="§4"))

    system = MechanicalSystem(shaft=shaft, speed_rpm=1450.0, design_life_hours=20_000.0)
    system.add_bearing(Bearing(
        position=50.0, C=35_000.0, C0=22_000.0, arrangement="fixed", label="A"
    ))
    system.add_bearing(Bearing(
        position=350.0, C=35_000.0, C0=22_000.0, arrangement="floating", label="B"
    ))
    system.add_gear(GearElement(
        position=200.0,
        tangential_force=3500.0,
        radial_force=1274.0,
        axial_force=0.0,
        pitch_diameter=100.0,
        torque=175_000.0,
        label="G1",
    ))
    return system


@pytest.fixture
def no_torque_system():
    """
    Same shaft geometry but no torque — pure bending.
    Uses a RadialLoad instead of a GearElement so T=0 everywhere.
    Useful for isolating Goodman bending-only path.
    """
    from core.loads import RadialLoad, LoadPlane
    shaft = Shaft(name="notorque")
    shaft.add_section(ShaftSection(
        length=100.0, diameter=50.0,
        shoulder_right=Shoulder(
            fillet_radius=2.0, diameter_large=50.0, diameter_small=40.0
        ),
    ))
    shaft.add_section(ShaftSection(length=100.0, diameter=40.0))
    shaft.add_section(ShaftSection(
        length=100.0, diameter=50.0,
        shoulder_right=Shoulder(
            fillet_radius=2.0, diameter_large=50.0, diameter_small=40.0
        ),
    ))
    shaft.add_section(ShaftSection(length=100.0, diameter=40.0))

    system = MechanicalSystem(shaft=shaft, speed_rpm=1450.0)
    system.add_bearing(Bearing(
        position=50.0, C=35_000.0, C0=22_000.0, arrangement="fixed", label="A"
    ))
    system.add_bearing(Bearing(
        position=350.0, C=35_000.0, C0=22_000.0, arrangement="floating", label="B"
    ))
    system.add_load(RadialLoad(position=200.0, magnitude=3500.0, plane=LoadPlane.XZ))
    return system


# ===========================================================================
# StressSolver — output structure
# ===========================================================================

class TestStressSolverStructure:

    def test_returns_stress_result(self, std_system):
        from models.stress_result import StressResult
        result = _run(std_system)
        assert isinstance(result, StressResult)

    def test_two_critical_sections(self, std_system):
        # Two shoulder_right entries → two critical sections
        result = _run(std_system)
        assert len(result.sections) == 2

    def test_section_positions(self, std_system):
        # shoulder_right on §1 → x=100mm; on §2 → x=300mm
        result = _run(std_system)
        positions = {s.x for s in result.sections}
        assert 100.0 in positions
        assert 300.0 in positions

    def test_all_sections_are_shoulders(self, std_system):
        result = _run(std_system)
        for s in result.sections:
            assert s.raiser_type == StressRaiserType.SHOULDER

    def test_diameter_is_small_side(self, std_system):
        # diameter_small = 40mm (the section experiencing full stress concentration)
        result = _run(std_system)
        for s in result.sections:
            assert s.diameter == pytest.approx(40.0)

    def test_material_id_propagated(self, std_system):
        result = _run(std_system, material=S355)
        assert result.material_id == "S355"

    def test_finish_propagated(self, std_system):
        result = _run(std_system, finish="ground")
        assert result.finish == "ground"

    def test_reliability_propagated(self, std_system):
        result = _run(std_system, reliability=95.0)
        assert result.reliability_percent == pytest.approx(95.0)


# ===========================================================================
# Sorting: most critical first
# ===========================================================================

class TestSorting:

    def test_sorted_ascending_nf_goodman(self, std_system):
        result = _run(std_system)
        nf_values = [s.nf_goodman for s in result.sections if math.isfinite(s.nf_goodman)]
        assert nf_values == sorted(nf_values)

    def test_infinite_sections_appended_last(self):
        """
        Shaft with one shoulder at a zero-moment location → nf = inf.

        Geometry: large section (d=50mm) first with shoulder_right pointing
        to smaller section (d=40mm). No transverse loads → M_res ≈ 0 everywhere
        → nf_goodman = inf at the shoulder.

        shoulder_right.diameter_small must match section[1].diameter (40mm).
        """
        shaft = Shaft(name="inf_test")
        shaft.add_section(ShaftSection(
            length=150.0, diameter=50.0, label="big",
            shoulder_right=Shoulder(
                fillet_radius=2.0, diameter_large=50.0, diameter_small=40.0
            ),
        ))
        shaft.add_section(ShaftSection(length=150.0, diameter=40.0, label="small"))

        system = MechanicalSystem(shaft=shaft, speed_rpm=1450.0)
        system.add_bearing(Bearing(
            position=0.0, C=1.0, C0=1.0, arrangement="fixed"
        ))
        system.add_bearing(Bearing(
            position=300.0, C=1.0, C0=1.0, arrangement="floating"
        ))
        # No transverse loads → M_res ≈ 0 everywhere → nf = inf

        statics = StaticsSolver().solve(system)
        result = StressSolver().solve(system, statics, S355)

        assert len(result.sections) == 1
        for s in result.sections:
            assert math.isinf(s.nf_goodman)


# ===========================================================================
# Kf, Kfs, q, qs — stress concentration at shoulders
# ===========================================================================

class TestStressConcentration:
    """
    Reference: r=2mm, d=40mm → r/d=0.05, D/d=50/40=1.25≈1.5 (Phase 1).
    Kt  = kt_shoulder_bending(0.05)  = 1.85  (Peterson knot point)
    Kts = kt_shoulder_torsion(0.05)  = 1.40
    """

    def test_kt_at_r_over_d_005(self, std_system):
        result = _run(std_system)
        for s in result.sections:
            assert s.Kt == pytest.approx(1.85, abs=0.05)

    def test_kts_at_r_over_d_005(self, std_system):
        result = _run(std_system)
        for s in result.sections:
            assert s.Kts == pytest.approx(1.40, abs=0.05)

    def test_kf_between_one_and_kt(self, std_system):
        result = _run(std_system)
        for s in result.sections:
            assert 1.0 <= s.Kf <= s.Kt

    def test_kfs_between_one_and_kts(self, std_system):
        result = _run(std_system)
        for s in result.sections:
            assert 1.0 <= s.Kfs <= s.Kts

    def test_kf_formula(self, std_system):
        # Kf = 1 + q*(Kt-1) — verify consistency
        result = _run(std_system)
        for s in result.sections:
            expected_kf = 1.0 + s.q * (s.Kt - 1.0)
            assert s.Kf == pytest.approx(expected_kf, rel=1e-9)

    def test_kfs_formula(self, std_system):
        result = _run(std_system)
        for s in result.sections:
            expected_kfs = 1.0 + s.qs * (s.Kts - 1.0)
            assert s.Kfs == pytest.approx(expected_kfs, rel=1e-9)


# ===========================================================================
# Se_prime and Marin factors at section
# ===========================================================================

class TestSeAndMarinFactors:

    def test_se_prime_positive(self, std_system):
        result = _run(std_system)
        for s in result.sections:
            assert s.Se_prime > 0

    def test_se_prime_less_than_se_base(self, std_system):
        Se_base = S355.endurance_limit  # 295 MPa
        result = _run(std_system)
        for s in result.sections:
            assert s.Se_prime < Se_base

    def test_ka_stored_in_section(self, std_system):
        result = _run(std_system)
        for s in result.sections:
            assert 0 < s.ka <= 1.0

    def test_kb_stored_in_section(self, std_system):
        # d=40mm: kb = 1.24*40^(-0.107) ≈ 0.836 — well within [0.70, 1.0]
        result = _run(std_system)
        for s in result.sections:
            assert 0.70 <= s.kb <= 1.0

    def test_ke_matches_99_percent(self, std_system):
        result = _run(std_system, reliability=99.0)
        for s in result.sections:
            assert s.ke == pytest.approx(0.814, abs=1e-9)

    def test_higher_reliability_gives_lower_se_prime(self, std_system):
        r95 = _run(std_system, reliability=95.0)
        r99 = _run(std_system, reliability=99.0)
        for s95, s99 in zip(r95.sections, r99.sections):
            assert s99.Se_prime < s95.Se_prime


# ===========================================================================
# Phase 1 rotating shaft assumptions
# ===========================================================================

class TestRotatingShaftAssumptions:

    def test_mm_zero(self, std_system):
        result = _run(std_system)
        for s in result.sections:
            assert s.Mm == pytest.approx(0.0)

    def test_ta_zero(self, std_system):
        result = _run(std_system)
        for s in result.sections:
            assert s.Ta == pytest.approx(0.0)

    def test_ma_equals_m_res(self, std_system):
        statics = StaticsSolver().solve(std_system)
        result = StressSolver().solve(std_system, statics, S355)
        import numpy as np
        for s in result.sections:
            m_res_at_x = float(np.interp(s.x, statics.x, statics.M_res))
            assert s.Ma == pytest.approx(m_res_at_x, rel=1e-6)

    def test_tm_equals_torque(self, std_system):
        statics = StaticsSolver().solve(std_system)
        result = StressSolver().solve(std_system, statics, S355)
        import numpy as np
        for s in result.sections:
            T_at_x = float(np.interp(s.x, statics.x, statics.T))
            assert s.Tm == pytest.approx(T_at_x, rel=1e-6)


# ===========================================================================
# Von Mises stresses
# ===========================================================================

class TestVonMisesStresses:

    def test_sigma_a_formula(self, std_system):
        # σ'_a = 32*Kf*Ma / (π*d³)
        result = _run(std_system)
        for s in result.sections:
            if s.Ma >= 1.0:
                expected = 32.0 * s.Kf * s.Ma / (math.pi * s.diameter ** 3)
                assert s.sigma_a == pytest.approx(expected, rel=1e-6)

    def test_sigma_m_formula(self, std_system):
        # σ'_m = √3 * 16*Kfs*Tm / (π*d³)
        result = _run(std_system)
        for s in result.sections:
            expected = math.sqrt(3.0) * 16.0 * s.Kfs * s.Tm / (math.pi * s.diameter ** 3)
            assert s.sigma_m == pytest.approx(expected, rel=1e-6)

    def test_sigma_a_positive(self, std_system):
        result = _run(std_system)
        for s in result.sections:
            assert s.sigma_a >= 0.0

    def test_sigma_m_positive(self, std_system):
        result = _run(std_system)
        for s in result.sections:
            assert s.sigma_m >= 0.0


# ===========================================================================
# Safety factors — Goodman, ASME, yielding
# ===========================================================================

class TestSafetyFactors:

    def test_nf_goodman_positive(self, std_system):
        result = _run(std_system)
        for s in result.sections:
            assert s.nf_goodman > 0

    def test_nf_asme_positive(self, std_system):
        result = _run(std_system)
        for s in result.sections:
            assert s.nf_asme > 0

    def test_ny_positive(self, std_system):
        result = _run(std_system)
        for s in result.sections:
            assert s.ny > 0

    def test_asme_gte_goodman(self, std_system):
        # DE-ASME Elliptic uses Sy instead of Sut in the torsion term →
        # less conservative for ductile materials (Sy < Sut) → nf_asme ≥ nf_goodman.
        result = _run(std_system)
        for s in result.sections:
            if math.isfinite(s.nf_goodman) and math.isfinite(s.nf_asme):
                assert s.nf_asme >= s.nf_goodman * 0.95  # 5% tolerance for near-equal

    def test_langer_ok_when_ny_exceeds_nf(self, std_system):
        result = _run(std_system)
        for s in result.sections:
            if math.isfinite(s.nf_goodman):
                if s.ny >= s.nf_goodman:
                    assert s.langer_ok is True

    def test_goodman_formula_consistency(self, std_system):
        """
        Verify: 1/nf_goodman = (16/π·d³) * { 2*Kf*Ma/Se' + √3*Kfs*Tm/Sut }
        """
        result = _run(std_system)
        Sut = S355.Sut
        for s in result.sections:
            if not math.isfinite(s.nf_goodman):
                continue
            d = s.diameter
            manual_inv_n = (16.0 / (math.pi * d ** 3)) * (
                2.0 * s.Kf * s.Ma / s.Se_prime
                + math.sqrt(3.0) * s.Kfs * s.Tm / Sut
            )
            assert s.nf_goodman == pytest.approx(1.0 / manual_inv_n, rel=1e-6)

    def test_asme_formula_consistency(self, std_system):
        """
        Verify: 1/nf_asme² = (16/π·d³)² * [ 4*(Kf*Ma/Se')² + 3*(Kfs*Tm/Sy)² ]
        """
        result = _run(std_system)
        Sy = S355.Sy
        for s in result.sections:
            if not math.isfinite(s.nf_asme):
                continue
            d = s.diameter
            manual_inv_n2 = (16.0 / (math.pi * d ** 3)) ** 2 * (
                4.0 * (s.Kf * s.Ma / s.Se_prime) ** 2
                + 3.0 * (s.Kfs * s.Tm / Sy) ** 2
            )
            assert s.nf_asme == pytest.approx(1.0 / math.sqrt(manual_inv_n2), rel=1e-6)

    def test_yielding_formula_consistency(self, std_system):
        """
        Verify: ny = Sy / sigma_max,
        sigma_max = sqrt[(32*Kf*Ma/π*d³)² + 3*(16*Kfs*Tm/π*d³)²]
        """
        result = _run(std_system)
        Sy = S355.Sy
        for s in result.sections:
            if not math.isfinite(s.ny):
                continue
            d = s.diameter
            sigma_max = math.sqrt(
                (32.0 * s.Kf * s.Ma / (math.pi * d ** 3)) ** 2
                + 3.0 * (16.0 * s.Kfs * s.Tm / (math.pi * d ** 3)) ** 2
            )
            assert s.ny == pytest.approx(Sy / sigma_max, rel=1e-6)


# ===========================================================================
# Parametric: material effect
# ===========================================================================

class TestMaterialEffect:

    def test_stronger_material_gives_higher_nf(self, std_system):
        r_s355  = _run(std_system, material=S355)
        r_crmo  = _run(std_system, material=CrMo42)
        nf_s355 = r_s355.sections[0].nf_goodman
        nf_crmo = r_crmo.sections[0].nf_goodman
        assert nf_crmo > nf_s355

    def test_high_sut_material_se_capped(self):
        """AISI_4340 has Sut=1460MPa → Se_base capped at 700MPa."""
        assert AISI_4340.endurance_limit == pytest.approx(700.0)


# ===========================================================================
# Edge: no shoulders → empty sections list
# ===========================================================================

class TestNoShoulders:

    def test_uniform_shaft_no_sections(self):
        shaft = Shaft(name="uniform")
        shaft.add_section(ShaftSection(length=400.0, diameter=50.0))
        system = MechanicalSystem(shaft=shaft, speed_rpm=1450.0)
        system.add_bearing(Bearing(
            position=0.0, C=1.0, C0=1.0, arrangement="fixed"
        ))
        system.add_bearing(Bearing(
            position=400.0, C=1.0, C0=1.0, arrangement="floating"
        ))

        statics = StaticsSolver().solve(system)
        result = StressSolver().solve(system, statics, S355)

        assert result.sections == []


# ===========================================================================
# Langer warning fires
# ===========================================================================

class TestLangerWarning:

    def test_langer_warns_when_ny_lt_nf(self, std_system):
        """
        Force a Langer violation by using a low-Sy material under heavy load.
        Material with very low Sy → static yield governs before fatigue.
        """
        fragile = __import__('core.materials', fromlist=['Material']).Material(
            material_id="FRAGILE",
            Sut=500.0,
            Sy=100.0,   # very low Sy → ny will be very small
            E=210.0,
            density=7850.0,
        )
        statics = StaticsSolver().solve(std_system)

        with pytest.warns(UserWarning, match="Langer"):
            StressSolver().solve(std_system, statics, fragile)