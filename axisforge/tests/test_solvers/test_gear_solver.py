"""
tests/test_solvers/test_gear_solver.py
TDD — GearSolver validation against GEARpie reference report.

Primary validation case (Brief Semana 9):
  m=2, z1=20, z2=60, α=20°, β=0°, al=80mm, T1=63.7 N·m
  Ft=3183.1 N | Fr=1158.6 N | Fa=0 N | εα=1.60
  Source: GEARpie report (Semana 9 context), tolerance ±0.5%

All tests use np.testing.assert_allclose(rtol=0.005) for textbook cases.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from solvers.gear_solver import GearSolver
from models.gear_result import GearGeometryResult, GearForceResult
from core.components import GearElement


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def solver() -> GearSolver:
    return GearSolver()


@pytest.fixture
def spur_geometry(solver) -> GearGeometryResult:
    """GEARpie reference: m=2, z1=20, z2=60, α=20°, β=0°, al=80mm."""
    return solver.compute_geometry(
        mn=2.0, z1=20, z2=60,
        alpha_n_deg=20.0, beta_deg=0.0,
        al=80.0,
    )


@pytest.fixture
def spur_forces(solver, spur_geometry) -> GearForceResult:
    """Forces from T1=63.7 N·m on the spur reference geometry."""
    return solver.compute_forces(T1_Nm=63.7, geometry=spur_geometry)


# ---------------------------------------------------------------------------
# 1. Geometry — spur reference case
# ---------------------------------------------------------------------------

class TestGearGeometrySpur:
    """Spur gear (β=0): αt=α, βb=0, εβ=0, standard centre."""

    def test_transverse_module_spur(self, spur_geometry):
        """mt = mn for β=0."""
        np.testing.assert_allclose(spur_geometry.mt, 2.0, rtol=1e-9)

    def test_transverse_pressure_angle_spur(self, spur_geometry):
        """αt = αn = 20° for β=0."""
        np.testing.assert_allclose(spur_geometry.alpha_t_deg, 20.0, rtol=1e-9)

    def test_base_helix_angle_spur(self, spur_geometry):
        """βb = 0 for β=0."""
        np.testing.assert_allclose(spur_geometry.beta_b_deg, 0.0, atol=1e-9)

    def test_gear_ratio(self, spur_geometry):
        """u = z2/z1 = 3."""
        np.testing.assert_allclose(spur_geometry.u, 3.0, rtol=1e-9)

    def test_reference_diameters(self, spur_geometry):
        """d1=40mm, d2=120mm."""
        np.testing.assert_allclose(spur_geometry.d1, 40.0, rtol=1e-9)
        np.testing.assert_allclose(spur_geometry.d2, 120.0, rtol=1e-9)

    def test_base_diameters(self, spur_geometry):
        """db = d * cos(αt)."""
        expected_db1 = 40.0 * math.cos(math.radians(20.0))
        expected_db2 = 120.0 * math.cos(math.radians(20.0))
        np.testing.assert_allclose(spur_geometry.db1, expected_db1, rtol=1e-6)
        np.testing.assert_allclose(spur_geometry.db2, expected_db2, rtol=1e-6)

    def test_standard_centre_distance(self, spur_geometry):
        """a = (d1+d2)/2 = 80mm."""
        np.testing.assert_allclose(spur_geometry.a, 80.0, rtol=1e-9)

    def test_working_centre_distance(self, spur_geometry):
        """al = 80mm (supplied)."""
        np.testing.assert_allclose(spur_geometry.al, 80.0, rtol=1e-9)

    def test_working_pressure_angle_standard(self, spur_geometry):
        """αtw = αt = 20° when al = a_standard."""
        np.testing.assert_allclose(spur_geometry.alpha_tw_deg, 20.0, rtol=1e-4)

    def test_working_pitch_diameters(self, spur_geometry):
        """dl1 = 2*al/(u+1) = 40mm; dl2 = 120mm."""
        np.testing.assert_allclose(spur_geometry.dl1, 40.0, rtol=1e-6)
        np.testing.assert_allclose(spur_geometry.dl2, 120.0, rtol=1e-6)

    def test_is_spur(self, spur_geometry):
        assert spur_geometry.is_spur is True

    def test_is_standard_centre(self, spur_geometry):
        assert spur_geometry.is_standard_centre is True

    def test_contact_ratio_eps_alpha(self, spur_geometry):
        """
        εα (ISO 21771) = 1.671 for this geometry.

        Note: the brief quotes 1.60 — this is an approximation.
        ISO 21771 formula (via tip pressure angles and αtw) gives 1.671,
        confirmed by independent hand-calculation and GEARpie line-of-action
        formula (AE/pbt). Both methods agree to 6 significant figures.
        Tolerance ±0.5% applied around the ISO result.
        """
        np.testing.assert_allclose(spur_geometry.eps_alpha, 1.671, rtol=0.005)

    def test_eps_beta_zero_for_spur(self, spur_geometry):
        """εβ = 0 for β=0."""
        np.testing.assert_allclose(spur_geometry.eps_beta, 0.0, atol=1e-9)

    def test_eps_gamma_equals_eps_alpha_for_spur(self, spur_geometry):
        """εγ = εα for β=0."""
        np.testing.assert_allclose(
            spur_geometry.eps_gamma, spur_geometry.eps_alpha, rtol=1e-9
        )

    def test_tip_diameters_no_shift(self, spur_geometry):
        """da = d + 2*mn (no profile shift, haP=1.0)."""
        np.testing.assert_allclose(spur_geometry.da1, 40.0 + 2 * 2.0, rtol=1e-6)
        np.testing.assert_allclose(spur_geometry.da2, 120.0 + 2 * 2.0, rtol=1e-6)

    def test_root_diameters_no_shift(self, spur_geometry):
        """df = d - 2*mn*hfP = d - 2*2*1.25 = d - 5."""
        np.testing.assert_allclose(spur_geometry.df1, 40.0 - 2 * 2.0 * 1.25, rtol=1e-6)
        np.testing.assert_allclose(spur_geometry.df2, 120.0 - 2 * 2.0 * 1.25, rtol=1e-6)

    def test_base_pitch(self, spur_geometry):
        """pbt = π * mt * cos(αt)."""
        expected = math.pi * 2.0 * math.cos(math.radians(20.0))
        np.testing.assert_allclose(spur_geometry.p_bt, expected, rtol=1e-6)

    def test_rb1_property(self, spur_geometry):
        np.testing.assert_allclose(spur_geometry.rb1, spur_geometry.db1 / 2.0, rtol=1e-9)

    def test_rl1_property(self, spur_geometry):
        np.testing.assert_allclose(spur_geometry.rl1, spur_geometry.dl1 / 2.0, rtol=1e-9)


# ---------------------------------------------------------------------------
# 2. Forces — primary validation (GEARpie reference)
# ---------------------------------------------------------------------------

class TestGearForcesSpur:
    """Primary validation: Ft=3183.1N, Fr=1158.6N, Fa=0N."""

    def test_tangential_force(self, spur_forces):
        """Ft = 1000*T1 / rl1 = 1000*63.7/20.0 = 3185 N ≈ 3183.1 N."""
        np.testing.assert_allclose(spur_forces.Ft, 3183.1, rtol=0.005)

    def test_radial_force(self, spur_forces):
        """Fr = Fbt * sin(αtw) ≈ 1158.6 N."""
        np.testing.assert_allclose(spur_forces.Fr, 1158.6, rtol=0.005)

    def test_axial_force_zero_spur(self, spur_forces):
        """Fa = 0 exactly for β=0."""
        assert spur_forces.Fa == 0.0

    def test_torque_T1_converted_to_Nmm(self, spur_forces):
        """T1 stored in N·mm = 63.7 * 1000."""
        np.testing.assert_allclose(spur_forces.T1_Nmm, 63_700.0, rtol=1e-9)

    def test_torque_T2(self, spur_forces):
        """T2 = T1 * u = 63700 * 3 = 191100 N·mm."""
        np.testing.assert_allclose(spur_forces.T2_Nmm, 63_700.0 * 3.0, rtol=1e-9)

    def test_gear_ratio(self, spur_forces):
        np.testing.assert_allclose(spur_forces.gear_ratio, 3.0, rtol=1e-9)

    def test_Fbt_greater_than_Ft(self, spur_forces):
        """Fbt = T1/rb1 > Ft = T1/rl1 since rb1 < rl1."""
        assert spur_forces.Fbt > spur_forces.Ft

    def test_normal_force_spur(self, spur_forces):
        """Fn = Fbt/cos(βb) = Fbt for β=0."""
        np.testing.assert_allclose(
            spur_forces.Fn, spur_forces.Fbt / math.cos(0.0), rtol=1e-9
        )

    def test_force_triangle(self, spur_forces):
        """Fn² = Ft² + Fr² + Fa²."""
        Fn_check = math.sqrt(
            spur_forces.Ft**2 + spur_forces.Fr**2 + spur_forces.Fa**2
        )
        np.testing.assert_allclose(spur_forces.Fn, Fn_check, rtol=0.005)


# ---------------------------------------------------------------------------
# 3. to_gear_element — interface com pipeline
# ---------------------------------------------------------------------------

class TestToGearElement:
    """GearElement produced must pass GearElement.__post_init__ consistency check."""

    def test_returns_gear_element(self, solver, spur_geometry, spur_forces):
        ge = solver.to_gear_element(
            position=200.0,
            forces=spur_forces,
            geometry=spur_geometry,
            label="G_test",
        )
        assert isinstance(ge, GearElement)

    def test_tangential_force_matches(self, solver, spur_geometry, spur_forces):
        ge = solver.to_gear_element(200.0, spur_forces, spur_geometry)
        np.testing.assert_allclose(ge.tangential_force, spur_forces.Ft, rtol=1e-9)

    def test_radial_force_matches(self, solver, spur_geometry, spur_forces):
        ge = solver.to_gear_element(200.0, spur_forces, spur_geometry)
        np.testing.assert_allclose(ge.radial_force, spur_forces.Fr, rtol=1e-9)

    def test_axial_force_matches(self, solver, spur_geometry, spur_forces):
        ge = solver.to_gear_element(200.0, spur_forces, spur_geometry)
        assert ge.axial_force == spur_forces.Fa

    def test_pitch_diameter_is_dl1(self, solver, spur_geometry, spur_forces):
        """pitch_diameter must be dl1 (working), not d1 (reference)."""
        ge = solver.to_gear_element(200.0, spur_forces, spur_geometry)
        np.testing.assert_allclose(ge.pitch_diameter, spur_geometry.dl1, rtol=1e-9)

    def test_torque_in_Nmm(self, solver, spur_geometry, spur_forces):
        """GearElement.torque must be in N·mm."""
        ge = solver.to_gear_element(200.0, spur_forces, spur_geometry)
        np.testing.assert_allclose(ge.torque, spur_forces.T1_Nmm, rtol=0.005)

    def test_gear_element_consistency_check_passes(self, solver, spur_geometry, spur_forces):
        """GearElement.__post_init__ must not raise (torque deviation < 2%)."""
        ge = solver.to_gear_element(200.0, spur_forces, spur_geometry)
        expected = ge.tangential_force * ge.pitch_diameter / 2.0
        deviation = abs(ge.torque - expected) / max(expected, 1.0)
        assert deviation < 0.02, f"Torque deviation {deviation*100:.2f}% > 2%"

    def test_pressure_angle_degrees(self, solver, spur_geometry, spur_forces):
        ge = solver.to_gear_element(200.0, spur_forces, spur_geometry)
        np.testing.assert_allclose(ge.pressure_angle, 20.0, rtol=1e-6)

    def test_helix_angle_degrees(self, solver, spur_geometry, spur_forces):
        ge = solver.to_gear_element(200.0, spur_forces, spur_geometry)
        np.testing.assert_allclose(ge.helix_angle, 0.0, atol=1e-9)

    def test_label_propagated(self, solver, spur_geometry, spur_forces):
        ge = solver.to_gear_element(200.0, spur_forces, spur_geometry, label="G1")
        assert ge.label == "G1"

    def test_position_propagated(self, solver, spur_geometry, spur_forces):
        ge = solver.to_gear_element(150.0, spur_forces, spur_geometry)
        assert ge.position == 150.0


# ---------------------------------------------------------------------------
# 4. Edge cases and guards
# ---------------------------------------------------------------------------

class TestGearSolverEdgeCases:

    def test_undercut_warning_z_less_than_17(self, solver):
        """z1=14 < 17 → warnings.warn issued."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            solver.compute_geometry(
                mn=2.0, z1=14, z2=40,
                alpha_n_deg=20.0, beta_deg=0.0,
                al=54.0,
            )
        messages = [str(warning.message) for warning in w]
        assert any("undercut" in m.lower() for m in messages), (
            f"Expected undercut warning, got: {messages}"
        )

    def test_no_undercut_warning_z_17(self, solver):
        """
        z1=17 is the Shigley minimum for α=20° — must NOT warn.

        zmin = int(2/sin²(20°)) = int(17.097) = 17.
        z1=17 is NOT < 17, so no warning (Shigley §13-10, Tab. 13-11).
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            solver.compute_geometry(
                mn=2.0, z1=17, z2=51,
                alpha_n_deg=20.0, beta_deg=0.0,
                al=68.0,
            )
        messages = [str(warning.message) for warning in w]
        assert not any("undercut" in m.lower() for m in messages)

    def test_zero_torque_gives_zero_forces(self, solver, spur_geometry):
        """T1=0 → all forces zero."""
        forces = solver.compute_forces(T1_Nm=0.0, geometry=spur_geometry)
        assert forces.Ft == 0.0
        assert forces.Fr == 0.0
        assert forces.Fa == 0.0

    def test_geometry_frozen(self, spur_geometry):
        """GearGeometryResult is frozen — mutation raises."""
        with pytest.raises((AttributeError, TypeError)):
            spur_geometry.mt = 99.0  # type: ignore

    def test_force_result_frozen(self, spur_forces):
        """GearForceResult is frozen — mutation raises."""
        with pytest.raises((AttributeError, TypeError)):
            spur_forces.Ft = 0.0  # type: ignore

    def test_negative_torque_raises(self, solver, spur_geometry):
        """Negative T1 is physically invalid — must raise."""
        with pytest.raises(ValueError, match="T1_Nm"):
            solver.compute_forces(T1_Nm=-10.0, geometry=spur_geometry)

    def test_z1_zero_raises(self, solver):
        with pytest.raises(ValueError):
            solver.compute_geometry(
                mn=2.0, z1=0, z2=60,
                alpha_n_deg=20.0, beta_deg=0.0, al=60.0,
            )

    def test_mn_zero_raises(self, solver):
        with pytest.raises(ValueError):
            solver.compute_geometry(
                mn=0.0, z1=20, z2=60,
                alpha_n_deg=20.0, beta_deg=0.0, al=80.0,
            )


# ---------------------------------------------------------------------------
# 5. Helical gear — qualitative checks (β > 0)
# ---------------------------------------------------------------------------

class TestHelicalGearQualitative:
    """
    Helical case: m=3, z1=18, z2=54, α=20°, β=15°, al=standard.
    No external reference — checks physical invariants only.
    """

    @pytest.fixture
    def helical_geometry(self, solver):
        mt = 3.0 / math.cos(math.radians(15.0))
        d1 = mt * 18
        d2 = mt * 54
        al = (d1 + d2) / 2.0
        return solver.compute_geometry(
            mn=3.0, z1=18, z2=54,
            alpha_n_deg=20.0, beta_deg=15.0,
            al=round(al, 6),
        )

    @pytest.fixture
    def helical_forces(self, solver, helical_geometry):
        return solver.compute_forces(T1_Nm=100.0, geometry=helical_geometry)

    def test_mt_greater_than_mn(self, helical_geometry):
        """mt = mn/cos(β) > mn for β > 0."""
        assert helical_geometry.mt > 3.0

    def test_alpha_t_greater_than_alpha_n(self, helical_geometry):
        """αt > αn for β > 0."""
        assert helical_geometry.alpha_t_deg > 20.0

    def test_beta_b_positive(self, helical_geometry):
        """βb > 0 for β > 0."""
        assert helical_geometry.beta_b_deg > 0.0

    def test_is_not_spur(self, helical_geometry):
        assert helical_geometry.is_spur is False

    def test_fa_nonzero_helical(self, helical_forces):
        """Fa > 0 for helical gear."""
        assert helical_forces.Fa > 0.0

    def test_force_triangle_helical(self, helical_forces):
        """Fn² = Ft² + Fr² + Fa²."""
        Fn_check = math.sqrt(
            helical_forces.Ft**2
            + helical_forces.Fr**2
            + helical_forces.Fa**2
        )
        np.testing.assert_allclose(helical_forces.Fn, Fn_check, rtol=0.005)

    def test_eps_beta_nonzero_without_face_width(self, helical_geometry):
        """εβ = 0 when b not supplied (default b=0)."""
        assert helical_geometry.eps_beta == 0.0


# ===========================================================================
# Profile Shift Validation — Real GEARpie Reports
# ===========================================================================

# ---------------------------------------------------------------------------
# C14 — Spur gear with profile shift
# m=4.5, z1=16, z2=24, α=20°, β=0°, x1=0.1817, x2=0.1715, al=91.5mm
# T1=200 N·m  →  Ft=5464.5N, Fr=2256.6N, Fa=0N, εα=1.46
# Source: GEARpie report C14 (real case)
# ---------------------------------------------------------------------------

@pytest.fixture
def c14_geometry(solver):
    """C14 spur gear with profile shift — al=91.5mm supplied directly."""
    return solver.compute_geometry(
        mn=4.5, z1=16, z2=24,
        alpha_n_deg=20.0, beta_deg=0.0,
        al=91.5,
        x1=0.1817, x2=0.1715,
    )

@pytest.fixture
def c14_forces(solver, c14_geometry):
    return solver.compute_forces(T1_Nm=200.0, geometry=c14_geometry)


class TestC14ProfileShiftSpur:
    """Validation against real GEARpie C14 report — spur gear with profile shift."""

    def test_tangential_force(self, c14_forces):
        """Ft=5464.5 N — GEARpie report C14."""
        np.testing.assert_allclose(c14_forces.Ft, 5464.5, rtol=0.005)

    def test_radial_force(self, c14_forces):
        """Fr=2256.6 N — GEARpie report C14."""
        np.testing.assert_allclose(c14_forces.Fr, 2256.6, rtol=0.005)

    def test_axial_force_zero_spur(self, c14_forces):
        """Fa=0 N — spur gear."""
        assert c14_forces.Fa == 0.0

    def test_contact_ratio_eps_alpha(self, c14_geometry):
        """εα=1.46 — GEARpie report C14."""
        np.testing.assert_allclose(c14_geometry.eps_alpha, 1.46, rtol=0.005)

    def test_working_centre_distance(self, c14_geometry):
        """al=91.5mm — supplied directly."""
        np.testing.assert_allclose(c14_geometry.al, 91.5, rtol=1e-9)

    def test_working_pressure_angle(self, c14_geometry):
        """αtw≈22.44° — derived from al and profile shift."""
        np.testing.assert_allclose(c14_geometry.alpha_tw_deg, 22.44, atol=0.05)

    def test_tip_diameter_pinion(self, c14_geometry):
        """da1 = 2 × 41.318 = 82.636 mm — GEARpie report C14."""
        np.testing.assert_allclose(c14_geometry.da1, 82.636, atol=0.05)

    def test_tip_diameter_wheel(self, c14_geometry):
        """da2 = 2 × 59.272 = 118.544 mm — GEARpie report C14."""
        np.testing.assert_allclose(c14_geometry.da2, 118.544, atol=0.05)

    def test_root_diameter_pinion(self, c14_geometry):
        """df1 = 2 × 31.193 = 62.386 mm — GEARpie report C14."""
        np.testing.assert_allclose(c14_geometry.df1, 62.386, atol=0.05)

    def test_profile_shift_stored(self, c14_geometry):
        """x1 and x2 stored correctly."""
        np.testing.assert_allclose(c14_geometry.x1, 0.1817, rtol=1e-6)
        np.testing.assert_allclose(c14_geometry.x2, 0.1715, rtol=1e-6)

    def test_reference_diameters(self, c14_geometry):
        """d1=72mm, d2=108mm (module × teeth, no shift effect on reference)."""
        np.testing.assert_allclose(c14_geometry.d1, 4.5 * 16, rtol=1e-9)
        np.testing.assert_allclose(c14_geometry.d2, 4.5 * 24, rtol=1e-9)

    def test_standard_centre_distance(self, c14_geometry):
        """a=90.0mm (reference, before profile shift)."""
        np.testing.assert_allclose(c14_geometry.a, 90.0, rtol=1e-9)

    def test_involute_consistency(self, c14_geometry):
        """
        inv(αtw) from al must equal inv(αt) + 2·tan(α)·(x1+x2)/(z1+z2).
        Verifies that al and x1+x2 are geometrically consistent.
        """
        alpha_t  = math.radians(c14_geometry.alpha_t_deg)
        alpha_tw = math.radians(c14_geometry.alpha_tw_deg)
        alpha_n  = math.radians(c14_geometry.alpha_n_deg)
        inv_from_al = math.tan(alpha_tw) - alpha_tw
        inv_from_x  = (math.tan(alpha_t) - alpha_t
                       + 2 * math.tan(alpha_n) * (c14_geometry.x1 + c14_geometry.x2)
                       / (c14_geometry.z1 + c14_geometry.z2))
        np.testing.assert_allclose(inv_from_al, inv_from_x, atol=1e-5)

    def test_gear_element_consistency(self, solver, c14_geometry, c14_forces):
        """GearElement torque check must pass (deviation < 2%)."""
        ge = solver.to_gear_element(100.0, c14_forces, c14_geometry, label="C14")
        expected = ge.tangential_force * ge.pitch_diameter / 2.0
        deviation = abs(ge.torque - expected) / max(expected, 1.0)
        assert deviation < 0.02


# ---------------------------------------------------------------------------
# H501 — Helical gear with profile shift
# m=3.5, z1=20, z2=30, α=20°, β=15°, x1=0.1809, x2=0.0891, al=91.5mm
# T1=100 N·m  →  Ft=2732.2N, Fr=1110.3N, Fa=739.5N, εα=1.46
# Source: GEARpie report H501 (real case)
# ---------------------------------------------------------------------------

@pytest.fixture
def h501_geometry(solver):
    """H501 helical gear with profile shift — al=91.5mm supplied directly."""
    return solver.compute_geometry(
        mn=3.5, z1=20, z2=30,
        alpha_n_deg=20.0, beta_deg=15.0,
        al=91.5,
        x1=0.1809, x2=0.0891,
    )

@pytest.fixture
def h501_forces(solver, h501_geometry):
    return solver.compute_forces(T1_Nm=100.0, geometry=h501_geometry)


class TestH501ProfileShiftHelical:
    """Validation against real GEARpie H501 report — helical gear with profile shift."""

    def test_tangential_force(self, h501_forces):
        """Ft=2732.2 N — GEARpie report H501."""
        np.testing.assert_allclose(h501_forces.Ft, 2732.2, rtol=0.005)

    def test_radial_force(self, h501_forces):
        """Fr=1110.3 N — GEARpie report H501."""
        np.testing.assert_allclose(h501_forces.Fr, 1110.3, rtol=0.005)

    def test_axial_force(self, h501_forces):
        """Fa=739.5 N — GEARpie report H501."""
        np.testing.assert_allclose(h501_forces.Fa, 739.5, rtol=0.005)

    def test_contact_ratio_eps_alpha(self, h501_geometry):
        """εα=1.46 — GEARpie report H501. rtol=0.01: εα sensitive to da rounding."""
        np.testing.assert_allclose(h501_geometry.eps_alpha, 1.46, rtol=0.01)

    def test_base_pitch_transverse(self, h501_geometry):
        """p_bt=10.652mm — GEARpie report H501."""
        np.testing.assert_allclose(h501_geometry.p_bt, 10.652, rtol=0.005)

    def test_working_pressure_angle(self, h501_geometry):
        """αtw≈22.11° — derived from al."""
        np.testing.assert_allclose(h501_geometry.alpha_tw_deg, 22.11, atol=0.05)

    def test_tip_diameter_pinion(self, h501_geometry):
        """da1 ≈ 80.67 mm — GEARpie report H501.
        Tolerance 0.1mm: GEARpie uses slightly different haP convention for helical gears.
        """
        np.testing.assert_allclose(h501_geometry.da1, 80.672, atol=0.1)

    def test_tip_diameter_wheel(self, h501_geometry):
        """da2 ≈ 116.26 mm — GEARpie report H501. Tolerance 0.1mm."""
        np.testing.assert_allclose(h501_geometry.da2, 116.264, atol=0.1)

    def test_root_diameter_pinion(self, h501_geometry):
        """df1 ≈ 64.99 mm — GEARpie report H501. Tolerance 0.1mm."""
        np.testing.assert_allclose(h501_geometry.df1, 64.986, atol=0.1)

    def test_profile_shift_stored(self, h501_geometry):
        np.testing.assert_allclose(h501_geometry.x1, 0.1809, rtol=1e-6)
        np.testing.assert_allclose(h501_geometry.x2, 0.0891, rtol=1e-6)

    def test_is_helical(self, h501_geometry):
        assert h501_geometry.is_spur is False

    def test_beta_b_positive(self, h501_geometry):
        assert h501_geometry.beta_b_deg > 0.0

    def test_involute_consistency(self, h501_geometry):
        """inv(αtw) from al matches involute equation with x1+x2."""
        alpha_t  = math.radians(h501_geometry.alpha_t_deg)
        alpha_tw = math.radians(h501_geometry.alpha_tw_deg)
        alpha_n  = math.radians(h501_geometry.alpha_n_deg)
        inv_from_al = math.tan(alpha_tw) - alpha_tw
        inv_from_x  = (math.tan(alpha_t) - alpha_t
                       + 2 * math.tan(alpha_n) * (h501_geometry.x1 + h501_geometry.x2)
                       / (h501_geometry.z1 + h501_geometry.z2))
        np.testing.assert_allclose(inv_from_al, inv_from_x, atol=1e-5)

    def test_force_triangle(self, h501_forces):
        """Fn_base = sqrt(Ft² + Fr² + Fa²)."""
        Fn_check = math.sqrt(
            h501_forces.Ft**2 + h501_forces.Fr**2 + h501_forces.Fa**2
        )
        np.testing.assert_allclose(h501_forces.Fn, Fn_check, rtol=0.005)

    def test_gear_element_consistency(self, solver, h501_geometry, h501_forces):
        """GearElement torque check must pass (deviation < 2%)."""
        ge = solver.to_gear_element(150.0, h501_forces, h501_geometry, label="H501")
        expected = ge.tangential_force * ge.pitch_diameter / 2.0
        deviation = abs(ge.torque - expected) / max(expected, 1.0)
        assert deviation < 0.02


# ---------------------------------------------------------------------------
# al=None — compute from involute equation (brentq)
# ---------------------------------------------------------------------------

class TestComputeAlFromInvolute:
    """When al is not supplied, GearSolver solves for al via brentq."""

    def test_al_none_spur_no_shift(self, solver):
        """x=0, al=None → al = a_standard."""
        geo = solver.compute_geometry(
            mn=2.0, z1=20, z2=60,
            alpha_n_deg=20.0, beta_deg=0.0,
            al=None, x1=0.0, x2=0.0,
        )
        np.testing.assert_allclose(geo.al, geo.a, rtol=1e-6)

    def test_al_none_with_profile_shift_greater_than_standard(self, solver):
        """x1+x2 > 0 → al > a_standard."""
        geo = solver.compute_geometry(
            mn=4.5, z1=16, z2=24,
            alpha_n_deg=20.0, beta_deg=0.0,
            al=None, x1=0.1817, x2=0.1715,
        )
        assert geo.al > geo.a

    def test_al_none_c14_matches_supplied_al(self, solver):
        """al=None with C14 x values must give al≈91.5mm."""
        geo = solver.compute_geometry(
            mn=4.5, z1=16, z2=24,
            alpha_n_deg=20.0, beta_deg=0.0,
            al=None, x1=0.1817, x2=0.1715,
        )
        np.testing.assert_allclose(geo.al, 91.5, rtol=0.001)

    def test_al_none_h501_matches_supplied_al(self, solver):
        """al=None with H501 x values must give al≈91.5mm."""
        geo = solver.compute_geometry(
            mn=3.5, z1=20, z2=30,
            alpha_n_deg=20.0, beta_deg=15.0,
            al=None, x1=0.1809, x2=0.0891,
        )
        np.testing.assert_allclose(geo.al, 91.5, rtol=0.001)

    def test_al_none_x_zero_helical(self, solver):
        """x=0, helical — al=None → al = a_standard."""
        geo_none = solver.compute_geometry(
            mn=3.5, z1=20, z2=30,
            alpha_n_deg=20.0, beta_deg=15.0,
            al=None, x1=0.0, x2=0.0,
        )
        geo_a = solver.compute_geometry(
            mn=3.5, z1=20, z2=30,
            alpha_n_deg=20.0, beta_deg=15.0,
            al=geo_none.a, x1=0.0, x2=0.0,
        )
        np.testing.assert_allclose(geo_none.al, geo_a.al, rtol=1e-6)
