"""
tests/solvers/machine_elements/bearings/test_iso16281_ball_solver.py

Unit tests for ISO16281BallSolver -- the point-contact internal load
distribution solver -- and the postprocessing that consumes its output.

The load-bearing assertion in this file
---------------------------------------
TestEquilibrium recomputes Fr and Fa from the solver's own published
delta_j / alpha_j via ISO/TS 16281 Sec 4.2.2.1 and compares them against
the loads that went in. Everything else here -- seeds, shapes, monotonicity
-- is structural. That one check is the only thing that can catch the
solver converging to a self-consistent wrong answer, which is exactly the
failure mode a suite of derived-quantity comparisons is blind to.

solve_contact() is the public per-bearing seam: it takes raw scalars, so it
needs no ShaftSystem and no FEM library. The batch entry point solve() is
covered in test_rolling_bearing_pipeline.py.

References:
  ISO/TS 16281:2008 Sec 4.2, eq.(12)-(15)   -- kinematics and equilibrium
  ISO/TS 16281:2008 Sec 4.3.2, eq.(25)-(28) -- dynamic equivalent load

ASCII only.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from axisforge.solvers.machine_elements.bearings.ISO_16281.Ball_Bearing.single_row_solver import (
    ISO16281BallSolver,
    REQUIRED_ATTRS,
)
from axisforge.solvers.machine_elements.bearings.ISO_16281.Ball_Bearing import (
    postprocessing as pp,
)
from axisforge.solvers.machine_elements.bearings.ISO_16281.Ball_Bearing.results import (
    BallBearingResult,
)

FR = 2000.0     # radial load used throughout [N]
FA = 300.0      # axial load [N]


@pytest.fixture
def solver() -> ISO16281BallSolver:
    return ISO16281BallSolver()


@pytest.fixture
def solved(solver, brg_A):
    """One converged radial+axial solve on the 6204, phi_Fr = 0, psi = 0."""
    return solver.solve_contact(
        brg_A, Fr_xz=FR, Fr_xy=0.0, Fa=FA,
        delta_r_init=0.0, delta_a_init=0.0, psi=0.0, phi_Fr=0.0,
    )


# ===========================================================================
# Declared contract
# ===========================================================================

class TestRequiredAttrs:

    def test_required_attrs_are_all_present_on_an_assembled_bearing(self, brg_A):
        for attr in REQUIRED_ATTRS:
            assert getattr(brg_A, attr, None) is not None, attr

    def test_required_attrs_match_the_family_declaration(self, brg_A):
        """
        The solver's REQUIRED_ATTRS must be a subset of what the family
        promises to produce for point_contact -- otherwise a bearing can
        assemble cleanly and still be unsolvable.
        """
        promised = brg_A.family.REQUIRED_FOR["point_contact"]
        assert set(REQUIRED_ATTRS) <= set(promised)


# ===========================================================================
# Convergence and result shape
# ===========================================================================

class TestSolveContactShape:

    def test_converges(self, solved):
        assert solved.ok is True

    def test_residual_is_small(self, solved):
        assert solved.residual < 1.0e-6

    def test_per_element_arrays_have_one_entry_per_ball(self, solved, brg_A):
        assert len(solved.delta_j) == brg_A.Z
        assert len(solved.alpha_j) == brg_A.Z

    def test_deflections_are_never_negative(self, solved):
        """Unloaded elements are floored at zero, not left negative."""
        assert np.all(solved.delta_j >= 0.0)

    def test_inputs_are_echoed_back(self, solved):
        assert solved.psi == pytest.approx(0.0)
        assert solved.phi_Fr == pytest.approx(0.0)

    def test_radial_deflection_is_positive_under_radial_load(self, solved):
        assert solved.delta_r > 0.0

    def test_axial_deflection_increases_with_axial_load(self, solver, brg_A):
        """
        ORDERING, not sign -- see the test below for why the sign of delta_a
        does not track the sign of Fa.
        """
        def da(Fa):
            return solver.solve_contact(
                brg_A, Fr_xz=FR, Fr_xy=0.0, Fa=Fa,
                delta_r_init=0.0, delta_a_init=0.0,
                psi=0.0, phi_Fr=0.0).delta_a

        assert da(+FA) > da(0.0) > da(-FA)

    def test_pure_radial_load_seats_the_balls_at_zero_axial_offset(self, solver, brg_A):
        """
        delta_a is measured from the NOMINAL position, with the ball seated
        at alpha_0 -- not from a zero-axial-force state. With psi = 0 every
        element shares the same V_j = A*sin(alpha_0) + delta_a, and
        sin(alpha_j) = V_j / sqrt(U_j^2 + V_j^2). Axial equilibrium
        sum(d32*sin(alpha_j)) = 0 with d32 >= 0 therefore forces V_j = 0
        EXACTLY, so

            delta_a = -A*sin(alpha_0)     (about -0.056 mm for this 6204)

        An identity that falls straight out of the equation system, not an
        expectation about signs. An earlier version of this test asserted
        delta_a > 0 for Fa > 0 and was simply wrong about the convention.
        """
        row = solver.solve_contact(brg_A, Fr_xz=FR, Fr_xy=0.0, Fa=0.0,
                                    delta_r_init=0.0, delta_a_init=0.0,
                                    psi=0.0, phi_Fr=0.0)
        assert row.delta_a == pytest.approx(
            -brg_A.A * math.sin(brg_A.alpha_0), rel=1e-6)

    def test_moment_reaction_is_reported(self, solved):
        """Mz is a diagnostic byproduct, not a constraint -- but it exists."""
        assert math.isfinite(solved.Mz)


# ===========================================================================
# Equilibrium -- the independent check
# ===========================================================================

class TestEquilibrium:

    @pytest.mark.parametrize("Fr_xz,Fr_xy,Fa", [
        (2000.0,    0.0,    0.0),
        (2000.0,    0.0,  300.0),
        (   0.0, 1500.0,    0.0),
        (1200.0,  900.0,  250.0),
        ( 100.0,    0.0,    0.0),      # lightly loaded
        (20000.0,   0.0, 1000.0),      # heavily loaded
    ])
    def test_recovered_loads_match_the_applied_ones(self, solver, brg_A,
                                                    recovered_loads,
                                                    Fr_xz, Fr_xy, Fa):
        """
        Sec 4.2.2.1 equilibrium, recomputed from the published delta_j /
        alpha_j. If the solver converged to something self-consistent but
        wrong, this is where it shows.
        """
        row = solver.solve_contact(
            brg_A, Fr_xz=Fr_xz, Fr_xy=Fr_xy, Fa=Fa,
            delta_r_init=0.0, delta_a_init=0.0, psi=0.0,
            phi_Fr=math.atan2(Fr_xy, Fr_xz),
        )
        Fr_hat, Fa_hat = recovered_loads(brg_A, row)

        Fr_applied = math.hypot(Fr_xz, Fr_xy)
        assert Fr_hat == pytest.approx(Fr_applied, rel=1e-6, abs=1e-6)
        assert Fa_hat == pytest.approx(Fa, rel=1e-6, abs=1e-6)

    def test_equilibrium_holds_under_misalignment(self, solver, brg_A,
                                                  recovered_loads):
        """psi enters the kinematics; equilibrium must survive it."""
        row = solver.solve_contact(
            brg_A, Fr_xz=FR, Fr_xy=0.0, Fa=FA,
            delta_r_init=0.0, delta_a_init=0.0, psi=5.0e-4, phi_Fr=0.0)
        Fr_hat, Fa_hat = recovered_loads(brg_A, row)
        assert Fr_hat == pytest.approx(FR, rel=1e-6, abs=1e-6)
        assert Fa_hat == pytest.approx(FA, rel=1e-6, abs=1e-6)

    def test_the_sum_of_contact_forces_carries_the_radial_load(self, solved, brg_A):
        """
        Same statement one level up: the projected contact forces from
        Q_j() must add back up to Fr.
        """
        Q = pp.Q_j(brg_A, BallBearingResult.single(solved))[0]
        projected = float(np.sum(Q * np.cos(solved.alpha_j) * np.cos(brg_A.phi_j)))
        assert projected == pytest.approx(FR, rel=1e-6, abs=1e-6)


# ===========================================================================
# Physical behaviour
# ===========================================================================

class TestPhysicalBehaviour:

    def test_heavier_load_gives_larger_deflection(self, solver, brg_A):
        light = solver.solve_contact(brg_A, Fr_xz=500.0, Fr_xy=0.0, Fa=0.0,
                                      delta_r_init=0.0, delta_a_init=0.0,
                                      psi=0.0, phi_Fr=0.0)
        heavy = solver.solve_contact(brg_A, Fr_xz=5000.0, Fr_xy=0.0, Fa=0.0,
                                      delta_r_init=0.0, delta_a_init=0.0,
                                      psi=0.0, phi_Fr=0.0)
        assert heavy.delta_r > light.delta_r

    def test_deflection_grows_sublinearly_with_load(self, solver, brg_A):
        """
        Hertzian point contact: Q ~ delta^1.5, so delta ~ Q^(2/3). Ten times
        the load must give well under ten times the deflection.
        """
        light = solver.solve_contact(brg_A, Fr_xz=500.0, Fr_xy=0.0, Fa=0.0,
                                      delta_r_init=0.0, delta_a_init=0.0,
                                      psi=0.0, phi_Fr=0.0)
        heavy = solver.solve_contact(brg_A, Fr_xz=5000.0, Fr_xy=0.0, Fa=0.0,
                                      delta_r_init=0.0, delta_a_init=0.0,
                                      psi=0.0, phi_Fr=0.0)
        ratio = heavy.delta_r / light.delta_r
        assert 1.0 < ratio < 10.0

    def test_only_part_of_the_ball_set_is_loaded_under_pure_radial_load(
        self, solver, brg_A
    ):
        """
        The classic radial load zone: with clearance and no axial preload,
        the balls opposite the load carry nothing.
        """
        row = solver.solve_contact(brg_A, Fr_xz=FR, Fr_xy=0.0, Fa=0.0,
                                    delta_r_init=0.0, delta_a_init=0.0,
                                    psi=0.0, phi_Fr=0.0)
        loaded = np.count_nonzero(row.delta_j > 0.0)
        assert 0 < loaded < brg_A.Z

    def test_the_most_loaded_ball_sits_in_the_load_direction(self, solver, brg_A):
        """
        phi_j is local with 0 aligned to the resultant radial force, so the
        maximum deflection belongs to the element at phi_j = 0.
        """
        row = solver.solve_contact(brg_A, Fr_xz=FR, Fr_xy=0.0, Fa=0.0,
                                    delta_r_init=0.0, delta_a_init=0.0,
                                    psi=0.0, phi_Fr=0.0)
        assert int(np.argmax(row.delta_j)) == 0

    def test_axial_preload_closes_the_whole_ball_set(self, solver, brg_A):
        """Enough axial load and every element carries -- no load zone left."""
        row = solver.solve_contact(brg_A, Fr_xz=100.0, Fr_xy=0.0, Fa=5000.0,
                                    delta_r_init=0.0, delta_a_init=0.0,
                                    psi=0.0, phi_Fr=0.0)
        assert np.all(row.delta_j > 0.0)

    def test_pure_axial_load_shares_evenly_between_all_balls(self, solver, brg_A):
        """With Fr = 0 the problem is axisymmetric -- every ball alike."""
        row = solver.solve_contact(brg_A, Fr_xz=0.0, Fr_xy=0.0, Fa=2000.0,
                                    delta_r_init=0.0, delta_a_init=0.0,
                                    psi=0.0, phi_Fr=0.0)
        assert np.allclose(row.delta_j, row.delta_j[0])

    def test_result_is_independent_of_the_starting_guess(self, solver, brg_A):
        """
        A converged root should not remember its seed. If these disagree,
        the tolerance is too loose or there is more than one root nearby.
        """
        cold = solver.solve_contact(brg_A, Fr_xz=FR, Fr_xy=0.0, Fa=FA,
                                     delta_r_init=0.0, delta_a_init=0.0,
                                     psi=0.0, phi_Fr=0.0)
        warm = solver.solve_contact(brg_A, Fr_xz=FR, Fr_xy=0.0, Fa=FA,
                                     delta_r_init=0.05, delta_a_init=0.02,
                                     psi=0.0, phi_Fr=0.0)
        assert warm.delta_r == pytest.approx(cold.delta_r, rel=1e-6)
        assert warm.delta_a == pytest.approx(cold.delta_a, rel=1e-6)

    def test_load_direction_does_not_change_the_magnitudes(self, solver, brg_A):
        """
        The solve happens in the resultant-force plane, so rotating the load
        around the bearing must leave delta_r untouched.
        """
        along_xz = solver.solve_contact(brg_A, Fr_xz=FR, Fr_xy=0.0, Fa=0.0,
                                         delta_r_init=0.0, delta_a_init=0.0,
                                         psi=0.0, phi_Fr=0.0)
        along_xy = solver.solve_contact(brg_A, Fr_xz=0.0, Fr_xy=FR, Fa=0.0,
                                         delta_r_init=0.0, delta_a_init=0.0,
                                         psi=0.0, phi_Fr=math.pi / 2.0)
        assert along_xy.delta_r == pytest.approx(along_xz.delta_r, rel=1e-9)


# ===========================================================================
# Seeds
# ===========================================================================

class TestInitialGuesses:

    def test_a_meaningful_hint_is_trusted(self, solver, brg_A):
        assert solver._initial_delta_r(brg_A, FR, hint=0.05) == pytest.approx(0.05)

    def test_a_negligible_hint_falls_back_to_a_physical_estimate(self, solver, brg_A):
        seeded = solver._initial_delta_r(brg_A, FR, hint=0.0)
        gap = brg_A.A * (1.0 - math.cos(brg_A.alpha_0))
        assert seeded > gap        # gap plus a Hertz-scale deflection

    def test_axial_hint_is_trusted_including_a_negative_one(self, solver, brg_A):
        assert solver._initial_delta_a(brg_A, FA, hint=-0.02) == pytest.approx(-0.02)

    def test_zero_axial_load_seeds_zero(self, solver, brg_A):
        assert solver._initial_delta_a(brg_A, 0.0, hint=0.0) == 0.0

    def test_axial_seed_carries_the_load_sign(self, solver, brg_A):
        assert solver._initial_delta_a(brg_A, -FA, hint=0.0) < 0.0
        assert solver._initial_delta_a(brg_A, +FA, hint=0.0) > 0.0


# ===========================================================================
# minimum_axial_load
# ===========================================================================

class TestMinimumAxialLoad:

    def test_returns_a_positive_preload_and_its_solution(self, solver, brg_A):
        Fa_min, res = solver.minimum_axial_load(
            brg_A, Fr_xz=FR, Fr_xy=0.0, psi=0.0)
        assert Fa_min > 0.0
        assert res.ok is True

    def test_the_returned_preload_closes_the_contact(self, solver, brg_A):
        """delta_a >= 0 is the condition the root solve targets."""
        _, res = solver.minimum_axial_load(brg_A, Fr_xz=FR, Fr_xy=0.0, psi=0.0)
        assert res.delta_a == pytest.approx(0.0, abs=1e-6)

    def test_heavier_radial_load_needs_more_preload(self, solver, brg_A):
        light, _ = solver.minimum_axial_load(brg_A, Fr_xz=500.0, Fr_xy=0.0, psi=0.0)
        heavy, _ = solver.minimum_axial_load(brg_A, Fr_xz=5000.0, Fr_xy=0.0, psi=0.0)
        assert heavy > light

    def test_a_bracket_that_cannot_contain_the_root_is_rejected(self, solver, brg_A):
        with pytest.raises(ValueError):
            solver.minimum_axial_load(brg_A, Fr_xz=5.0e4, Fr_xy=0.0, psi=0.0,
                                       Fa_bracket=(0.0, 1.0e-3))


# ===========================================================================
# Postprocessing
# ===========================================================================

class TestPostprocessing:

    @pytest.fixture
    def result(self, solved) -> BallBearingResult:
        return BallBearingResult.single(solved)

    def test_single_row_result_holds_exactly_one_row(self, result, solved):
        assert len(result.rows) == 1
        assert result.rows[0] is solved

    def test_bearing_level_accessors_read_row_zero(self, result, solved):
        assert result.delta_r == pytest.approx(solved.delta_r)
        assert result.delta_a == pytest.approx(solved.delta_a)

    def test_multirow_wrapper_refuses_a_single_row(self, solved):
        with pytest.raises(ValueError, match=">=2 rows"):
            BallBearingResult.multirow(rows=[solved], f_r=np.array([1.0]),
                                        f_a=np.array([1.0]), n_iter=1,
                                        residual=0.0, ok=True)

    def test_q_j_follows_the_point_contact_law(self, brg_A, result, solved):
        """Q_j = cp * delta_j^1.5."""
        Q = pp.Q_j(brg_A, result)[0]
        assert np.allclose(Q, brg_A.cp * np.maximum(solved.delta_j, 0.0) ** 1.5)

    def test_q_j_returns_one_array_per_row(self, brg_A, result):
        Q = pp.Q_j(brg_A, result)
        assert isinstance(Q, list) and len(Q) == 1
        assert len(Q[0]) == brg_A.Z

    def test_unloaded_elements_carry_no_force(self, solver, brg_A):
        row = solver.solve_contact(brg_A, Fr_xz=FR, Fr_xy=0.0, Fa=0.0,
                                    delta_r_init=0.0, delta_a_init=0.0,
                                    psi=0.0, phi_Fr=0.0)
        Q = pp.Q_j(brg_A, BallBearingResult.single(row))[0]
        assert np.all(Q[row.delta_j == 0.0] == 0.0)

    def test_global_angles_are_wrapped_into_zero_two_pi(self, brg_A, result):
        phi = pp.phi_j_global(brg_A, result)[0]
        assert np.all(phi >= 0.0)
        assert np.all(phi < 2.0 * math.pi)

    def test_global_angles_equal_local_ones_when_phi_fr_is_zero(self, brg_A, result):
        phi = pp.phi_j_global(brg_A, result)[0]
        assert np.allclose(phi, brg_A.phi_j % (2.0 * math.pi))

    def test_global_angles_are_rotated_by_phi_fr(self, solver, brg_A):
        """A load along +XY puts phi_Fr at 90 deg and shifts every element."""
        row = solver.solve_contact(brg_A, Fr_xz=0.0, Fr_xy=FR, Fa=0.0,
                                    delta_r_init=0.0, delta_a_init=0.0,
                                    psi=0.0, phi_Fr=math.pi / 2.0)
        phi = pp.phi_j_global(brg_A, BallBearingResult.single(row))[0]
        assert np.allclose(phi, (brg_A.phi_j + math.pi / 2.0) % (2.0 * math.pi))

    def test_contact_distribution_pairs_angles_with_forces(self, brg_A, result):
        dist = pp.contact_distribution(brg_A, result)[0]
        assert dist.shape == (brg_A.Z, 2)
        assert np.allclose(dist[:, 0], pp.phi_j_global(brg_A, result)[0])
        assert np.allclose(dist[:, 1], pp.Q_j(brg_A, result)[0])

    def test_local_frame_uses_the_bearing_own_angles(self, brg_A, result):
        dist = pp.contact_distribution(brg_A, result, frame="local")[0]
        assert np.allclose(dist[:, 0], brg_A.phi_j)

    def test_unknown_frame_rejected(self, brg_A, result):
        with pytest.raises(ValueError, match="frame must be"):
            pp.contact_distribution(brg_A, result, frame="bogus")

    def test_stiffness_is_positive_in_the_loaded_direction(self, brg_A, result):
        k = pp.bearing_stiffness(brg_A, result, Fr_xz=FR, Fr_xy=0.0, Fa=FA)
        assert k.Kr_xz > 0.0

    def test_stiffness_is_infinite_where_nothing_moves(self, brg_A, result):
        """
        The load is pure XZ, so delta_r_xy is ~0 and the XY direction is
        reported rigid rather than dividing by zero.
        """
        k = pp.bearing_stiffness(brg_A, result, Fr_xz=FR, Fr_xy=0.0, Fa=FA)
        assert math.isinf(k.Kr_xy)

    def test_secant_stiffness_matches_load_over_deflection(self, brg_A, result, solved):
        k = pp.bearing_stiffness(brg_A, result, Fr_xz=FR, Fr_xy=0.0, Fa=FA)
        assert k.Kr_xz == pytest.approx(FR / (solved.delta_r * math.cos(solved.phi_Fr)))

    def test_stiffness_rises_with_load(self, solver, brg_A):
        """Hertzian contact hardens -- secant stiffness grows with load."""
        def k_of(Fr):
            row = solver.solve_contact(brg_A, Fr_xz=Fr, Fr_xy=0.0, Fa=0.0,
                                        delta_r_init=0.0, delta_a_init=0.0,
                                        psi=0.0, phi_Fr=0.0)
            return pp.bearing_stiffness(
                brg_A, BallBearingResult.single(row), Fr, 0.0, 0.0).Kr_xz

        assert k_of(5000.0) > k_of(500.0)

    def test_dynamic_equivalent_load_returns_one_entry_per_row(self, brg_A, result):
        derel = pp.DynamicEquivalentRollingElementLoad.from_bearing_result(
            brg_A, result, inner_rotating=True, outer_rotating=False)
        assert isinstance(derel, list) and len(derel) == 1

    def test_dynamic_equivalent_loads_are_positive(self, brg_A, result):
        derel = pp.DynamicEquivalentRollingElementLoad.from_bearing_result(
            brg_A, result, inner_rotating=True, outer_rotating=False)[0]
        assert derel.Q_ei > 0.0
        assert derel.Q_ee > 0.0

    def test_from_bearing_result_wraps_from_distribution(self, brg_A, result, solved):
        """The row-level primitive is what actually does the eq.(25)-(28) math."""
        wrapped = pp.DynamicEquivalentRollingElementLoad.from_bearing_result(
            brg_A, result, inner_rotating=True, outer_rotating=False)[0]
        direct = pp.DynamicEquivalentRollingElementLoad.from_distribution(
            brg_A, solved, inner_rotating=True, outer_rotating=False)
        assert wrapped.Q_ei == pytest.approx(direct.Q_ei)
        assert wrapped.Q_ee == pytest.approx(direct.Q_ee)
