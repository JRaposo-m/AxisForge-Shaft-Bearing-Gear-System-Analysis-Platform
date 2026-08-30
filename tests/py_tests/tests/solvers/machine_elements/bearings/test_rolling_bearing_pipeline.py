"""
tests/solvers/machine_elements/bearings/test_rolling_bearing_pipeline.py

Pipeline tests: FEM results library -> per-type solver -> orchestrator.

    SimpleFEMResultsLibrary          (prescribed, see conftest.py)
        |  .get(shaft.name).bearing_nodes
        v
    ISO16281BallSolver.solve()       per-type, batch over a bearing dict
        |
        v
    RollingBearingSolver.solve()     groups by BearingType, merges results

The nodal state is prescribed rather than produced by SimpleFEMSolver --
see conftest.py for why. This exercises the whole bearing-side chain with a
known input, which is what makes a failure attributable.

What the pipeline is responsible for, and what these tests check:
  - matching bearings to FEM nodes BY LABEL, not by order or position;
  - deriving phi_Fr from the nodal reaction components;
  - projecting psi_xz/psi_xy onto the resultant-force plane;
  - seeding the contact solve from the nodal displacements;
  - preserving the caller's bearing order on the way out.

ASCII only.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from axisforge.core.machine_elements.bearings.bearing_types import BearingType
from axisforge.solvers.machine_elements.bearings.ISO_16281.Ball_Bearing.single_row_solver import (
    ISO16281BallSolver,
)
from axisforge.solvers.machine_elements.bearings.ISO_16281.rolling_bearing_solver import (
    RollingBearingSolver,
)
from axisforge.solvers.machine_elements.bearings.ISO_16281.library import (
    BearingResultsLibrary,
)
from axisforge.solvers.machine_elements.shaft.oneD_analysis.static.static_analysis import (
    ShaftResults,
    SimpleFEMResultsLibrary,
)


# ===========================================================================
# ISO16281BallSolver.solve -- the per-type batch entry point
# ===========================================================================

class TestBallSolverBatch:

    @pytest.fixture
    def local_lib(self, brg_shaft_system, bearings_by_label, fem_library):
        return ISO16281BallSolver().solve(
            brg_shaft_system, bearings_by_label, fem_library)

    def test_every_bearing_is_solved(self, local_lib):
        assert set(local_lib.labels()) == {"A", "B"}

    def test_each_entry_is_a_single_row_container(self, local_lib):
        for label in ("A", "B"):
            assert len(local_lib.get(label).rows) == 1

    def test_both_converge(self, local_lib):
        for label in ("A", "B"):
            assert local_lib.get(label).rows[0].ok is True

    def test_equilibrium_holds_for_every_bearing(self, local_lib, fem_library,
                                                 bearings_by_label, brg_shaft_system,
                                                 recovered_loads):
        """
        The independent check, run through the full pipeline rather than on
        a bare solve_contact() call: the loads recovered from each solved
        row must equal the nodal reactions that went in.
        """
        nodes = {n.label: n for n in fem_library.get(brg_shaft_system.name).bearing_nodes}

        for label, bearing in bearings_by_label.items():
            row = local_lib.get(label).rows[0]
            Fr_hat, Fa_hat = recovered_loads(bearing, row)
            node = nodes[label]
            assert Fr_hat == pytest.approx(math.hypot(node.Fr_xz, node.Fr_xy),
                                           rel=1e-6, abs=1e-6)
            assert Fa_hat == pytest.approx(node.Fa, rel=1e-6, abs=1e-6)

    def test_phi_fr_comes_from_the_nodal_reaction_direction(self, local_lib):
        """A is loaded purely in XZ; B is off-axis at atan2(600, 800)."""
        assert local_lib.get("A").rows[0].phi_Fr == pytest.approx(0.0)
        assert local_lib.get("B").rows[0].phi_Fr == pytest.approx(
            math.atan2(600.0, 800.0))

    def test_psi_is_projected_onto_the_resultant_force_plane(self, local_lib,
                                                             fem_library,
                                                             brg_shaft_system):
        """psi = psi_xz*cos(phi_Fr) + psi_xy*sin(phi_Fr)."""
        node = {n.label: n for n in
                fem_library.get(brg_shaft_system.name).bearing_nodes}["B"]
        phi_Fr = math.atan2(node.Fr_xy, node.Fr_xz)
        expected = node.psi_xz * math.cos(phi_Fr) + node.psi_xy * math.sin(phi_Fr)
        assert local_lib.get("B").rows[0].psi == pytest.approx(expected)

    def test_bearings_are_matched_to_nodes_by_label_not_by_order(
        self, brg_shaft_system, bearings_by_label, fem_library
    ):
        """
        Feeding the bearings in reverse must give each one the same answer:
        the match is on label, so dict order is irrelevant.
        """
        forward = ISO16281BallSolver().solve(
            brg_shaft_system, bearings_by_label, fem_library)
        reversed_in = {k: bearings_by_label[k] for k in reversed(list(bearings_by_label))}
        backward = ISO16281BallSolver().solve(
            brg_shaft_system, reversed_in, fem_library)

        for label in ("A", "B"):
            assert (backward.get(label).rows[0].delta_r
                    == pytest.approx(forward.get(label).rows[0].delta_r))

    def test_a_subset_of_the_bearings_can_be_solved(self, brg_shaft_system,
                                                    bearings_by_label, fem_library):
        only_a = {"A": bearings_by_label["A"]}
        lib = ISO16281BallSolver().solve(brg_shaft_system, only_a, fem_library)
        assert set(lib.labels()) == {"A"}

    def test_an_unassembled_bearing_is_refused_before_solving(
        self, brg_shaft_system, fem_library
    ):
        class NotReady:
            label = "A"
            position = 50.0
            arrangement = "locating"
            A = None

        with pytest.raises(RuntimeError):
            ISO16281BallSolver().solve(
                brg_shaft_system, {"A": NotReady()}, fem_library)

    def test_a_label_with_no_fem_node_is_a_key_error(self, brg_shaft_system,
                                                     bearings_by_label, fem_library,
                                                     dgbb_factory):
        """A bearing the FEM never saw cannot be solved -- fail, don't guess."""
        extra = dict(bearings_by_label)
        extra["C"] = dgbb_factory("C", 150.0)
        with pytest.raises(KeyError):
            ISO16281BallSolver().solve(brg_shaft_system, extra, fem_library)

    def test_a_loaded_floating_bearing_warns_through_the_pipeline(
        self, brg_shaft_system, bearings_by_label, bearing_node_factory
    ):
        """
        B is floating, so it has no shoulder to react an axial load. Give it
        one in the nodal state and the guard in library.py must fire from
        inside solve(), not only when called directly.
        """
        library = SimpleFEMResultsLibrary()
        library.store(ShaftResults(
            name=brg_shaft_system.name,
            bearing_nodes=[
                bearing_node_factory("A", 50.0, 2000.0, 0.0, Fa=300.0),
                bearing_node_factory("B", 260.0, 800.0, 600.0, Fa=500.0),
            ],
        ))
        with pytest.warns(UserWarning, match="B"):
            ISO16281BallSolver().solve(
                brg_shaft_system, bearings_by_label, library)


# ===========================================================================
# RollingBearingSolver -- the orchestrator
# ===========================================================================

class TestRollingBearingSolverDispatch:

    @pytest.fixture
    def merged(self, brg_shaft_system, bearings_by_label, fem_library):
        return RollingBearingSolver().solve(
            brg_shaft_system, bearings_by_label, fem_library)

    def test_every_bearing_appears_in_the_merged_result(self, merged):
        assert set(merged) == {"A", "B"}

    def test_every_value_is_a_list_of_rows(self, merged):
        """
        Option A, documented: always a list -- length 1 for a single-row
        bearing -- so callers index [0] explicitly instead of the shape
        changing per bearing.
        """
        for label in ("A", "B"):
            assert isinstance(merged[label], list)
            assert len(merged[label]) == 1

    def test_the_callers_bearing_order_is_preserved(self, brg_shaft_system,
                                                    fem_library, bearings_by_label):
        reversed_in = {k: bearings_by_label[k] for k in reversed(list(bearings_by_label))}
        merged = RollingBearingSolver().solve(
            brg_shaft_system, reversed_in, fem_library)
        assert list(merged) == list(reversed_in)

    def test_the_orchestrator_agrees_with_the_per_type_solver(
        self, merged, brg_shaft_system, bearings_by_label, fem_library
    ):
        """Dispatch must not change the answer, only route it."""
        direct = ISO16281BallSolver().solve(
            brg_shaft_system, bearings_by_label, fem_library)
        for label in ("A", "B"):
            assert (merged[label][0].delta_r
                    == pytest.approx(direct.get(label).rows[0].delta_r))

    def test_results_are_recorded_into_a_bearing_results_library(
        self, brg_shaft_system, bearings_by_label, fem_library
    ):
        results = BearingResultsLibrary()
        RollingBearingSolver().solve(
            brg_shaft_system, bearings_by_label, fem_library, results=results)

        assert set(results.labels()) == {"A", "B"}
        for label in ("A", "B"):
            bundle = results.get(label)
            assert bundle.bearing_type is BearingType.DEEP_GROOVE_BALL
            assert bundle.load_distribution is not None
            assert len(bundle.load_distribution) == 1

    def test_the_per_type_sub_library_is_reachable_by_type(
        self, brg_shaft_system, bearings_by_label, fem_library
    ):
        results = BearingResultsLibrary()
        RollingBearingSolver().solve(
            brg_shaft_system, bearings_by_label, fem_library, results=results)
        sub = results.load_distribution_library(BearingType.DEEP_GROOVE_BALL)
        assert set(sub.labels()) == {"A", "B"}

    def test_an_unregistered_bearing_type_names_the_offending_labels(
        self, brg_shaft_system, bearings_by_label, fem_library
    ):
        """
        Only DEEP_GROOVE_BALL, ANGULAR_CONTACT and CYLINDRICAL_ROLLER are
        wired. Anything else must fail loudly and say which bearings it
        could not handle -- not skip them.
        """
        class WrongType:
            label = "X"
            bearing_type = BearingType.SPHERICAL_ROLLER
            arrangement = "locating"

        with pytest.raises(NotImplementedError) as excinfo:
            RollingBearingSolver().solve(
                brg_shaft_system, {"X": WrongType()}, fem_library)
        assert "X" in str(excinfo.value)
        assert "SPHERICAL_ROLLER" in str(excinfo.value)

    def test_tolerance_is_forwarded_to_the_per_type_solver(
        self, brg_shaft_system, bearings_by_label, fem_library
    ):
        """
        Both tolerances must reach the same root -- a converged solve does
        not depend on how tightly it was asked to converge, only on how
        precisely. Checked loosely on purpose: this is about the tol
        reaching the per-type solver at all, not about its exact effect.
        """
        loose = RollingBearingSolver(tol=1e-4).solve(
            brg_shaft_system, bearings_by_label, fem_library)
        tight = RollingBearingSolver(tol=1e-12).solve(
            brg_shaft_system, bearings_by_label, fem_library)
        assert tight["A"][0].delta_r == pytest.approx(loose["A"][0].delta_r, rel=1e-3)


class TestPsiOverride:

    def test_override_replaces_the_fem_projection(self, brg_shaft_system,
                                                  bearings_by_label, fem_library):
        solver = ISO16281BallSolver(psi_input=True)
        lib = solver.solve(brg_shaft_system, bearings_by_label, fem_library,
                           psi_override={"A": 1.0e-3})
        assert lib.get("A").rows[0].psi == pytest.approx(1.0e-3)

    def test_bearings_without_an_override_keep_the_fem_value(
        self, brg_shaft_system, bearings_by_label, fem_library
    ):
        solver = ISO16281BallSolver(psi_input=True)
        lib = solver.solve(brg_shaft_system, bearings_by_label, fem_library,
                           psi_override={"A": 1.0e-3})
        node = {n.label: n for n in
                fem_library.get(brg_shaft_system.name).bearing_nodes}["B"]
        phi_Fr = math.atan2(node.Fr_xy, node.Fr_xz)
        expected = node.psi_xz * math.cos(phi_Fr) + node.psi_xy * math.sin(phi_Fr)
        assert lib.get("B").rows[0].psi == pytest.approx(expected)

    def test_an_unknown_override_label_warns_rather_than_failing_silently(
        self, brg_shaft_system, bearings_by_label, fem_library
    ):
        """A typo in an override key would otherwise be a no-op."""
        solver = ISO16281BallSolver(psi_input=True)
        with pytest.warns(UserWarning, match="typo"):
            solver.solve(brg_shaft_system, bearings_by_label, fem_library,
                         psi_override={"NOT_A_BEARING": 1.0e-3})

    def test_override_is_ignored_when_psi_input_is_off(self, brg_shaft_system,
                                                       bearings_by_label,
                                                       fem_library):
        solver = ISO16281BallSolver(psi_input=False)
        lib = solver.solve(brg_shaft_system, bearings_by_label, fem_library,
                           psi_override={"A": 1.0e-3})
        assert lib.get("A").rows[0].psi != pytest.approx(1.0e-3)


# ===========================================================================
# The FEM boundary itself
# ===========================================================================

class TestFemLibraryBoundary:

    def test_the_library_is_keyed_by_shaft_name(self, fem_library, brg_shaft_system):
        assert fem_library.get(brg_shaft_system.name) is not None

    def test_an_unnamed_result_cannot_be_stored(self):
        """The name is the key -- an empty one would be silently unreachable."""
        with pytest.raises(ValueError, match="non-empty"):
            SimpleFEMResultsLibrary().store(ShaftResults(name=""))

    def test_restoring_the_same_name_overwrites(self, fem_library, brg_shaft_system):
        fem_library.store(ShaftResults(name=brg_shaft_system.name))
        assert fem_library.get(brg_shaft_system.name).bearing_nodes == []

    def test_a_missing_shaft_fails_rather_than_returning_empty(self, fem_library):
        with pytest.raises(KeyError):
            fem_library.get("no_such_shaft")

    def test_zero_load_solves_without_blowing_up(self, brg_shaft_system,
                                                 bearings_by_label,
                                                 bearing_node_factory):
        """
        An unloaded bearing is a legitimate load case (a shaft at rest).
        It must not divide by zero or fail to converge.
        """
        library = SimpleFEMResultsLibrary()
        library.store(ShaftResults(
            name=brg_shaft_system.name,
            bearing_nodes=[
                bearing_node_factory("A", 50.0, 0.0, 0.0),
                bearing_node_factory("B", 260.0, 0.0, 0.0),
            ],
        ))
        lib = ISO16281BallSolver().solve(
            brg_shaft_system, bearings_by_label, library)

        for label in ("A", "B"):
            row = lib.get(label).rows[0]
            assert np.all(np.isfinite(row.delta_j))
            # Floating-point noise, not contact: 1e-14 mm of deflection is
            # Q_j ~ 1e-16 N. An exact == 0.0 here was too tight.
            assert np.allclose(row.delta_j, 0.0, atol=1e-9)
