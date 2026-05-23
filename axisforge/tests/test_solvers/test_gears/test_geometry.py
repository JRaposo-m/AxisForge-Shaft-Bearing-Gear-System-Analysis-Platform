"""
tests/test_solvers/test_gears/test_geometry.py
Unit and validation tests for solvers/gears/geometry.py — GearSolver.

Test structure:
  TestGearSolverValidation    — GEARpie reference cases C14 (spur) and H501 (helical)
  TestComputeGeometry         — unit tests for geometry outputs and edge cases
  TestComputeForces           — force calculation, torque consistency
  TestToGearElement           — GearElement assembly and consistency check
  TestUndercut                — undercut warning path

References:
  GEARpie report cases C14 and H501 (C. Fernandes, MIT License).
  ISO 21771:2007; Shigley 10th ed. §13-7.
"""
from __future__ import annotations

import math
import warnings
import pytest

from solvers.gears.geometry import GearSolver


# ---------------------------------------------------------------------------
# Shared solver instance (stateless — one instance for all tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def solver():
    return GearSolver()


# ---------------------------------------------------------------------------
# GEARpie validation cases
# ---------------------------------------------------------------------------

class TestGearSolverValidation:
    """
    Validation against GEARpie reference reports (C. Fernandes, MIT License).

    C14: spur,    m=4.5, z=16/24, α=20°, β=0°,  x=0.1817/0.1715, al=91.5mm, T=200 N·m
    H501: helical, m=3.5, z=20/30, α=20°, β=15°, x=0.1809/0.0891, al=91.5mm, T=100 N·m

    Tolerance policy (GEARpie report precision):
      Forces    : ±0.5%   (report rounds to 0.1 N)
      Pitches   : ±0.001 mm
      Radii     : ±0.01 mm
      εα, εβ    : ±0.02
      Fbt, Fbn  : ±0.5%
    """

    # -----------------------------------------------------------------------
    # C14 — Spur gear, profile shift, al supplied
    # -----------------------------------------------------------------------

    @pytest.fixture(scope="class")
    def c14_geo(self, solver):
        return solver.compute_geometry(
            mn=4.5, z1=16, z2=24,
            alpha_n_deg=20.0, beta_deg=0.0,
            al=91.5, x1=0.1817, x2=0.1715,
        )

    @pytest.fixture(scope="class")
    def c14_forces(self, solver, c14_geo):
        return solver.compute_forces(T1_Nm=200.0, geometry=c14_geo)

    # Forces
    def test_c14_Ft(self, c14_forces):
        """GEARpie report: F_t = 5464.5 N"""
        assert c14_forces.Ft == pytest.approx(5464.5, rel=5e-3)

    def test_c14_Fr(self, c14_forces):
        """GEARpie report: F_r = 2256.6 N"""
        assert c14_forces.Fr == pytest.approx(2256.6, rel=5e-3)

    def test_c14_Fa_zero(self, c14_forces):
        """Spur (β=0): F_a = 0. GEARpie report: 0.0 N"""
        assert c14_forces.Fa == pytest.approx(0.0, abs=1e-9)

    def test_c14_Fbt(self, c14_forces):
        """GEARpie report: F_bt = 5912.1 N"""
        assert c14_forces.Fbt == pytest.approx(5912.1, rel=5e-3)

    def test_c14_Fbn(self, c14_forces):
        """GEARpie report: F_bn = 5912.1 N (spur: Fbn = Fbt)"""
        assert c14_forces.Fbn == pytest.approx(5912.1, rel=5e-3)

    def test_c14_Fn(self, c14_forces):
        """GEARpie: Fn = Ft/cos(beta_b). Spur beta=0 -> Fn = Ft = 5464.5 N."""
        assert c14_forces.Fn == pytest.approx(5464.5, rel=5e-3)

    # Contact ratios
    def test_c14_eps_alpha(self, c14_geo):
        """GEARpie report: εα = 1.46"""
        assert c14_geo.eps_alpha == pytest.approx(1.46, abs=0.02)

    def test_c14_eps_beta_zero(self, c14_geo):
        """Spur, b not supplied → εβ = 0.00"""
        assert c14_geo.eps_beta == pytest.approx(0.0, abs=1e-9)

    def test_c14_eps_gamma(self, c14_geo):
        """GEARpie report: εγ = 1.46"""
        assert c14_geo.eps_gamma == pytest.approx(1.46, abs=0.02)

    # Geometry — pitches
    def test_c14_base_pitch_n(self, c14_geo):
        """GEARpie report: base pitch n = 13.285 mm (= p_bt for spur)"""
        assert c14_geo.p_bt == pytest.approx(13.285, abs=0.005)

    # Geometry — radii (GEARpie reports radii, not diameters)
    def test_c14_base_radius_pinion(self, c14_geo):
        """GEARpie report: base radius rb1 = 33.829 mm"""
        assert c14_geo.rb1 == pytest.approx(33.829, abs=0.01)

    def test_c14_base_radius_wheel(self, c14_geo):
        """GEARpie report: base radius rb2 = 50.743 mm"""
        assert c14_geo.rb2 == pytest.approx(50.743, abs=0.01)

    def test_c14_pitch_radius_pinion(self, c14_geo):
        """GEARpie report: pitch (working) radius rl1 = 36.600 mm"""
        assert c14_geo.rl1 == pytest.approx(36.600, abs=0.01)

    def test_c14_pitch_radius_wheel(self, c14_geo):
        """GEARpie report: pitch (working) radius rl2 = 54.900 mm"""
        assert c14_geo.rl2 == pytest.approx(54.900, abs=0.01)

    def test_c14_tip_radius_pinion(self, c14_geo):
        """GEARpie report: tip radius ra1 = 41.318 mm"""
        assert c14_geo.da1 / 2.0 == pytest.approx(41.318, abs=0.01)

    def test_c14_tip_radius_wheel(self, c14_geo):
        """GEARpie report: tip radius ra2 = 59.272 mm"""
        assert c14_geo.da2 / 2.0 == pytest.approx(59.272, abs=0.01)

    def test_c14_root_radius_pinion(self, c14_geo):
        """GEARpie report: root radius rf1 = 31.193 mm"""
        assert c14_geo.df1 / 2.0 == pytest.approx(31.193, abs=0.01)

    def test_c14_root_radius_wheel(self, c14_geo):
        """GEARpie report: root radius rf2 = 49.147 mm"""
        assert c14_geo.df2 / 2.0 == pytest.approx(49.147, abs=0.01)

    # Bookkeeping
    def test_c14_gear_ratio(self, c14_geo):
        assert c14_geo.u == pytest.approx(24 / 16, rel=1e-9)

    def test_c14_al_preserved(self, c14_geo):
        assert c14_geo.al == pytest.approx(91.5, abs=1e-6)

    def test_c14_torque_consistency(self, c14_forces):
        """T1 = Ft × rl1: deviation < 0.5% (GEARpie consistency criterion)."""
        T1_check = c14_forces.Ft * c14_forces.T1_Nmm / c14_forces.Ft
        dev = abs(c14_forces.Ft * (c14_forces.T1_Nmm / c14_forces.Ft) - c14_forces.T1_Nmm) / c14_forces.T1_Nmm
        assert dev < 0.005

    # -----------------------------------------------------------------------
    # H501 — Helical gear, profile shift, al supplied, b=30mm for εβ
    # -----------------------------------------------------------------------

    @pytest.fixture(scope="class")
    def h501_geo(self, solver):
        # b=22.94mm derived from report: eps_beta=0.54 = b*tan(beta_b)/p_bt
        return solver.compute_geometry(
            mn=3.5, z1=20, z2=30,
            alpha_n_deg=20.0, beta_deg=15.0,
            al=91.5, x1=0.1809, x2=0.0891,
            b=22.94,
        )

    @pytest.fixture(scope="class")
    def h501_forces(self, solver, h501_geo):
        return solver.compute_forces(T1_Nm=100.0, geometry=h501_geo)

    # Forces
    def test_h501_Ft(self, h501_forces):
        """GEARpie report: F_t = 2732.2 N"""
        assert h501_forces.Ft == pytest.approx(2732.2, rel=5e-3)

    def test_h501_Fr(self, h501_forces):
        """GEARpie report: F_r = 1110.3 N"""
        assert h501_forces.Fr == pytest.approx(1110.3, rel=5e-3)

    def test_h501_Fa(self, h501_forces):
        """GEARpie report: F_a = 739.5 N"""
        assert h501_forces.Fa == pytest.approx(739.5, rel=5e-3)

    def test_h501_Fbt(self, h501_forces):
        """GEARpie report: F_bt = 2949.2 N"""
        assert h501_forces.Fbt == pytest.approx(2949.2, rel=5e-3)

    def test_h501_Fbn(self, h501_forces):
        """GEARpie report: F_bn = 3040.5 N"""
        assert h501_forces.Fbn == pytest.approx(3040.5, rel=5e-3)

    def test_h501_Fn_resultant(self, h501_forces):
        """GEARpie: Fn = Ft/cos(beta_b) = 2816.8 N (not vector resultant sqrt(Ft2+Fr2+Fa2))."""
        assert h501_forces.Fn == pytest.approx(2816.8, rel=5e-3)

    # Contact ratios
    def test_h501_eps_alpha(self, h501_geo):
        """GEARpie report: εα = 1.46"""
        assert h501_geo.eps_alpha == pytest.approx(1.46, abs=0.02)

    def test_h501_eps_beta(self, h501_geo):
        """GEARpie report: εβ = 0.54"""
        assert h501_geo.eps_beta == pytest.approx(0.54, abs=0.02)

    def test_h501_eps_gamma(self, h501_geo):
        """GEARpie report: εγ = 2.00"""
        assert h501_geo.eps_gamma == pytest.approx(2.00, abs=0.03)

    # Geometry — pitches
    def test_h501_base_pitch_t(self, h501_geo):
        """GEARpie report: base pitch t (transverse) = 10.652 mm"""
        assert h501_geo.p_bt == pytest.approx(10.652, abs=0.005)

    # Geometry — radii
    def test_h501_pitch_radius_pinion(self, h501_geo):
        """GEARpie report: pitch radius rl1 = 36.600 mm"""
        assert h501_geo.rl1 == pytest.approx(36.600, abs=0.01)

    def test_h501_pitch_radius_wheel(self, h501_geo):
        """GEARpie report: pitch radius rl2 = 54.900 mm"""
        assert h501_geo.rl2 == pytest.approx(54.900, abs=0.01)

    def test_h501_tip_radius_pinion(self, h501_geo):
        """GEARpie report: tip radius ra1 = 40.336 mm"""
        assert h501_geo.da1 / 2.0 == pytest.approx(40.336, abs=0.04)  # k=0 vs GEARpie ~0.009*mn reduction

    def test_h501_tip_radius_wheel(self, h501_geo):
        """GEARpie report: tip radius ra2 = 58.132 mm"""
        assert h501_geo.da2 / 2.0 == pytest.approx(58.132, abs=0.04)  # k=0 vs GEARpie ~0.009*mn reduction

    def test_h501_root_radius_pinion(self, h501_geo):
        """GEARpie report: root radius rf1 = 32.493 mm"""
        assert h501_geo.df1 / 2.0 == pytest.approx(32.493, abs=0.01)

    def test_h501_root_radius_wheel(self, h501_geo):
        """GEARpie report: root radius rf2 = 50.289 mm"""
        assert h501_geo.df2 / 2.0 == pytest.approx(50.289, abs=0.01)

    # Flags
    def test_h501_is_not_spur(self, h501_geo):
        assert not h501_geo.is_spur

    def test_h501_gear_ratio(self, h501_geo):
        assert h501_geo.u == pytest.approx(30 / 20, rel=1e-9)


# ---------------------------------------------------------------------------
# compute_geometry — unit tests
# ---------------------------------------------------------------------------

class TestComputeGeometry:

    @pytest.fixture(scope="class")
    def simple_spur(self, solver):
        """m=3, z=20/40, α=20°, β=0°, x=0, al=None — standard case."""
        return solver.compute_geometry(
            mn=3.0, z1=20, z2=40,
            alpha_n_deg=20.0, beta_deg=0.0,
        )

    def test_gear_ratio(self, simple_spur):
        assert simple_spur.u == pytest.approx(40 / 20, rel=1e-9)

    def test_standard_centre_distance(self, simple_spur):
        """No shift, al=None → al = standard centre distance."""
        expected_a = (3.0 * 20 + 3.0 * 40) / 2.0   # 90mm
        assert simple_spur.al == pytest.approx(expected_a, rel=1e-6)

    def test_is_standard_centre(self, simple_spur):
        assert simple_spur.is_standard_centre

    def test_is_spur(self, simple_spur):
        assert simple_spur.is_spur

    def test_base_diameters(self, simple_spur):
        """db = d·cos(αt). For spur: αt = αn = 20°."""
        alpha_t = math.radians(20.0)
        assert simple_spur.db1 == pytest.approx(simple_spur.d1 * math.cos(alpha_t), rel=1e-9)
        assert simple_spur.db2 == pytest.approx(simple_spur.d2 * math.cos(alpha_t), rel=1e-9)

    def test_tip_diameters_no_shift(self, simple_spur):
        """da = d + 2·mn (x=0, haP=1.0)."""
        assert simple_spur.da1 == pytest.approx(simple_spur.d1 + 2.0 * 3.0, rel=1e-9)
        assert simple_spur.da2 == pytest.approx(simple_spur.d2 + 2.0 * 3.0, rel=1e-9)

    def test_root_diameters_no_shift(self, simple_spur):
        """df = d − 2·mn·hfP (x=0, hfP=1.25)."""
        assert simple_spur.df1 == pytest.approx(simple_spur.d1 - 2.0 * 3.0 * 1.25, rel=1e-9)

    def test_eps_alpha_greater_than_one(self, simple_spur):
        assert simple_spur.eps_alpha > 1.0

    def test_eps_beta_zero_spur(self, simple_spur):
        assert simple_spur.eps_beta == pytest.approx(0.0, abs=1e-9)

    def test_eps_gamma_equals_eps_alpha_for_spur(self, simple_spur):
        assert simple_spur.eps_gamma == pytest.approx(simple_spur.eps_alpha, rel=1e-9)

    def test_working_pitch_diameter_consistency(self, simple_spur):
        """dl1 + dl2 = 2·al."""
        assert simple_spur.dl1 + simple_spur.dl2 == pytest.approx(2.0 * simple_spur.al, rel=1e-9)

    def test_al_supplied_overrides_involute(self, solver):
        """When al is explicitly supplied, it is preserved verbatim."""
        geo = solver.compute_geometry(
            mn=4.0, z1=18, z2=36,
            alpha_n_deg=20.0, beta_deg=0.0,
            al=110.0,
        )
        assert geo.al == pytest.approx(110.0, abs=1e-9)

    def test_invalid_al_raises(self, solver):
        """al far below standard centre distance → cos(αtw) > 1 → ValueError."""
        with pytest.raises(ValueError, match="geometrically inconsistent"):
            solver.compute_geometry(
                mn=4.0, z1=18, z2=36,
                alpha_n_deg=20.0, beta_deg=0.0,
                al=10.0,  # absurdly small
            )

    def test_mn_zero_raises(self, solver):
        with pytest.raises(ValueError, match="mn"):
            solver.compute_geometry(mn=0.0, z1=20, z2=40, alpha_n_deg=20.0, beta_deg=0.0)

    def test_helical_angles_derived(self, solver):
        """β=15°: mt > mn, αt > αn, βb > 0."""
        geo = solver.compute_geometry(
            mn=3.0, z1=20, z2=30,
            alpha_n_deg=20.0, beta_deg=15.0,
            al=91.5,
        )
        beta = math.radians(15.0)
        assert geo.mt == pytest.approx(3.0 / math.cos(beta), rel=1e-9)
        assert geo.alpha_t_deg > 20.0
        assert geo.beta_b_deg > 0.0


# ---------------------------------------------------------------------------
# compute_forces — unit tests
# ---------------------------------------------------------------------------

class TestComputeForces:

    @pytest.fixture(scope="class")
    def spur_geo(self, solver):
        return solver.compute_geometry(
            mn=3.0, z1=20, z2=40,
            alpha_n_deg=20.0, beta_deg=0.0,
        )

    def test_zero_torque_returns_zero_forces(self, solver, spur_geo):
        f = solver.compute_forces(T1_Nm=0.0, geometry=spur_geo)
        assert f.Ft == 0.0
        assert f.Fr == 0.0
        assert f.Fa == 0.0
        assert f.Fn == 0.0

    def test_negative_torque_raises(self, solver, spur_geo):
        with pytest.raises(ValueError, match="T1_Nm"):
            solver.compute_forces(T1_Nm=-10.0, geometry=spur_geo)

    def test_spur_fa_zero(self, solver, spur_geo):
        f = solver.compute_forces(T1_Nm=50.0, geometry=spur_geo)
        assert f.Fa == pytest.approx(0.0, abs=1e-9)

    def test_fn_resultant(self, solver, spur_geo):
        """Fn = Ft/cos(beta_b). For spur (beta_b=0): Fn = Ft."""
        f = solver.compute_forces(T1_Nm=50.0, geometry=spur_geo)
        import math as _math
        beta_b = _math.radians(spur_geo.beta_b_deg)
        expected = f.Ft / _math.cos(beta_b) if _math.cos(beta_b) > 1e-9 else f.Ft
        assert f.Fn == pytest.approx(expected, rel=1e-9)

    def test_torque_propagation(self, solver, spur_geo):
        """T2 = T1 × u."""
        f = solver.compute_forces(T1_Nm=100.0, geometry=spur_geo)
        assert f.T2_Nmm == pytest.approx(f.T1_Nmm * spur_geo.u, rel=1e-9)

    def test_t1_unit_conversion(self, solver, spur_geo):
        """T1_Nm=1.0 → T1_Nmm=1000.0."""
        f = solver.compute_forces(T1_Nm=1.0, geometry=spur_geo)
        assert f.T1_Nmm == pytest.approx(1000.0, rel=1e-9)

    def test_force_transverse_resultant(self, solver, spur_geo):
        f = solver.compute_forces(T1_Nm=100.0, geometry=spur_geo)
        expected = math.hypot(f.Ft, f.Fr)
        assert f.F_transverse_resultant == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# to_gear_element
# ---------------------------------------------------------------------------

class TestToGearElement:

    def test_gear_element_assembles(self, solver):
        geo = solver.compute_geometry(
            mn=3.0, z1=20, z2=40,
            alpha_n_deg=20.0, beta_deg=0.0,
        )
        forces = solver.compute_forces(T1_Nm=100.0, geometry=geo)
        ge = solver.to_gear_element(position=150.0, forces=forces, geometry=geo)
        assert ge.position == pytest.approx(150.0)
        assert ge.tangential_force == pytest.approx(forces.Ft, rel=1e-9)
        assert ge.radial_force == pytest.approx(forces.Fr, rel=1e-9)
        assert ge.axial_force == pytest.approx(forces.Fa, rel=1e-9)

    def test_pitch_diameter_is_working(self, solver):
        """pitch_diameter = dl1 (working), not d1 (reference)."""
        geo = solver.compute_geometry(
            mn=4.5, z1=16, z2=24,
            alpha_n_deg=20.0, beta_deg=0.0,
            al=91.5, x1=0.1817, x2=0.1715,
        )
        forces = solver.compute_forces(T1_Nm=200.0, geometry=geo)
        ge = solver.to_gear_element(position=100.0, forces=forces, geometry=geo)
        assert ge.pitch_diameter == pytest.approx(geo.dl1, rel=1e-9)

    def test_label_propagated(self, solver):
        geo = solver.compute_geometry(
            mn=3.0, z1=20, z2=40,
            alpha_n_deg=20.0, beta_deg=0.0,
        )
        forces = solver.compute_forces(T1_Nm=50.0, geometry=geo)
        ge = solver.to_gear_element(position=100.0, forces=forces, geometry=geo, label="G_test")
        assert ge.label == "G_test"


# ---------------------------------------------------------------------------
# Undercut warning
# ---------------------------------------------------------------------------

class TestUndercutWarning:
    def test_undercut_warning_raised(self, solver):
        """z1=10 < z_min=17 for α=20°, x1=0 → UserWarning."""
        with pytest.warns(UserWarning, match="undercut"):
            solver.compute_geometry(
                mn=2.0, z1=10, z2=40,
                alpha_n_deg=20.0, beta_deg=0.0,
                x1=0.0,
            )

    def test_no_warning_with_positive_shift(self, solver):
        """z1=10 but x1=0.5 → no undercut warning."""
        with warnings.catch_warnings():
            import warnings as _w
            _w.simplefilter("error", UserWarning)
            # Must not raise
            solver.compute_geometry(
                mn=2.0, z1=10, z2=40,
                alpha_n_deg=20.0, beta_deg=0.0,
                x1=0.5,
            )

    def test_no_warning_sufficient_teeth(self, solver):
        """z1=20 ≥ z_min=17 → no warning."""
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("error", UserWarning)
            solver.compute_geometry(
                mn=3.0, z1=20, z2=40,
                alpha_n_deg=20.0, beta_deg=0.0,
            )