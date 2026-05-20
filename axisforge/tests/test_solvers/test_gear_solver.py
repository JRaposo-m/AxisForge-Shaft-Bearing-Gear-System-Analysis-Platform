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
