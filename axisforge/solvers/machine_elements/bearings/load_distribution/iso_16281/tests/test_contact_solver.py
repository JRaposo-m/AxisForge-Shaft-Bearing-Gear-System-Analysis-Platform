# test_contact_solver.py
"""Testes de contact_solver.py -- SolverBase (batch loop, contrato,
testado com um solver-stub para não depender das classes *Result
reais, cujo corpo não vimos), LineContactSolverBase, MultiRowSolverBase,
e os solvers concretos ISO16281BallSolver / ISO16281RollerSolver / os
dois multi-row.

Física: testes de equilíbrio/contrato (residual ~0, sinais plausíveis,
relações entre grandezas), não valores certificados contra a norma --
tal como os testes de capacity.py em bearings/families/tests."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from axisforge.solvers.machine_elements.bearings.load_distribution.iso_16281.contact_solver import (
    SolverBase, LineContactSolverBase, MultiRowSolverBase,
    ISO16281BallSolver, ISO16281RollerSolver,
    ISO16281MultiRowBallSolverSharedDisplacement,
    ISO16281MultiRowRollerSolverSharedDisplacement,
)
from axisforge.solvers.machine_elements.bearings.load_distribution.iso_16281.tests.conftest import (
    BALL_BEARING_KWARGS, ROLLER_BEARING_KWARGS, make_node, make_shaft_results,
)


# =====================================================================
# SolverBase -- batch loop, testado com um stub para isolar da física
# concreta e das classes *Result reais.
# =====================================================================

class _StubResultCls:
    @staticmethod
    def single(row):
        return ("wrapped", row)


class _StubSolver(SolverBase):
    _result_cls = _StubResultCls
    REQUIRED_ATTRS = ()

    def _solve_one_bearing(self, bearing, node, phi_Fr, psi):
        return ("row", bearing.label, phi_Fr, psi)


class _StubSolverNoResultCls(SolverBase):
    def _solve_one_bearing(self, bearing, node, phi_Fr, psi):
        return None


class TestSolverBaseSolve:
    def test_requires_result_cls(self):
        solver = _StubSolverNoResultCls()
        with pytest.raises(NotImplementedError, match="_result_cls"):
            solver.solve(SimpleNamespace(), {"b1": SimpleNamespace()}, make_shaft_results())

    def test_wraps_each_bearing_via_result_cls_single(self):
        solver = _StubSolver()
        bearing = SimpleNamespace(label="b1")
        node = make_node(Fr_xz=1.0, Fr_xy=0.0, Fa=0.0, psi_xz=0.0, psi_xy=0.0, label="b1")
        results = solver.solve(SimpleNamespace(), {"b1": bearing}, make_shaft_results(node))

        assert set(results) == {"b1"}
        kind, row = results["b1"]
        assert kind == "wrapped"
        assert row[0] == "row"
        assert row[1] == "b1"

    def test_computes_psi_from_projection_when_not_overridden(self):
        solver = _StubSolver()
        bearing = SimpleNamespace(label="b1")
        node = make_node(Fr_xz=3.0, Fr_xy=4.0, psi_xz=0.1, psi_xy=0.2, label="b1")
        phi_Fr_expected = math.atan2(4.0, 3.0)
        psi_expected = 0.1 * math.cos(phi_Fr_expected) + 0.2 * math.sin(phi_Fr_expected)

        results = solver.solve(SimpleNamespace(), {"b1": bearing}, make_shaft_results(node))
        _, (_, _, phi_Fr, psi) = results["b1"]

        assert phi_Fr == pytest.approx(phi_Fr_expected)
        assert psi == pytest.approx(psi_expected)

    def test_psi_override_ignored_when_psi_input_false(self):
        solver = _StubSolver(psi_input=False)
        bearing = SimpleNamespace(label="b1")
        node = make_node(Fr_xz=1.0, Fr_xy=0.0, psi_xz=0.5, psi_xy=0.0, label="b1")

        results = solver.solve(SimpleNamespace(), {"b1": bearing}, make_shaft_results(node),
                                psi_override={"b1": 999.0})
        _, (_, _, _, psi) = results["b1"]
        assert psi == pytest.approx(0.5)  # projeção normal, não o override

    def test_psi_override_honoured_when_psi_input_true(self):
        solver = _StubSolver(psi_input=True)
        bearing = SimpleNamespace(label="b1")
        node = make_node(Fr_xz=1.0, Fr_xy=0.0, psi_xz=0.5, psi_xy=0.0, label="b1")

        results = solver.solve(SimpleNamespace(), {"b1": bearing}, make_shaft_results(node),
                                psi_override={"b1": 999.0})
        _, (_, _, _, psi) = results["b1"]
        assert psi == pytest.approx(999.0)

    def test_warns_on_psi_override_with_unknown_label(self):
        solver = _StubSolver(psi_input=True)
        bearing = SimpleNamespace(label="b1")
        node = make_node(label="b1")

        with pytest.warns(UserWarning, match="has labels not in"):
            solver.solve(SimpleNamespace(), {"b1": bearing}, make_shaft_results(node),
                         psi_override={"typo_label": 1.0})

    def test_check_bearing_ready_enforced(self):
        class _NeedsFoo(_StubSolver):
            REQUIRED_ATTRS = ("foo",)

        solver = _NeedsFoo()
        bearing = SimpleNamespace(label="b1")  # sem .foo
        node = make_node(label="b1")
        with pytest.raises(RuntimeError, match="foo"):
            solver.solve(SimpleNamespace(), {"b1": bearing}, make_shaft_results(node))

    def test_warns_when_floating_bearing_is_axially_loaded(self):
        solver = _StubSolver()
        bearing = SimpleNamespace(label="b1", arrangement="floating")
        node = make_node(Fa=100.0, label="b1")
        with pytest.warns(UserWarning, match="floating but Fa"):
            solver.solve(SimpleNamespace(), {"b1": bearing}, make_shaft_results(node))


# =====================================================================
# LineContactSolverBase -- check extra de laminae (Sec 5.2.2)
# =====================================================================

class TestLineContactSolverExtraReadyChecks:
    def test_rejects_fewer_than_min_laminae(self, roller_bearing):
        solver = ISO16281RollerSolver()
        roller_bearing.n_s = 20
        roller_bearing.x_k = np.linspace(-2.0, 2.0, 20)
        with pytest.raises(ValueError, match="Sec 5.2.2"):
            solver._extra_ready_checks(roller_bearing, "b2")

    def test_rejects_x_k_length_mismatch(self, roller_bearing):
        solver = ISO16281RollerSolver()
        roller_bearing.x_k = np.linspace(-2.0, 2.0, roller_bearing.n_s - 1)
        with pytest.raises(ValueError, match="x_k must have exactly one lamina"):
            solver._extra_ready_checks(roller_bearing, "b2")

    def test_passes_with_valid_laminae(self, roller_bearing):
        solver = ISO16281RollerSolver()
        solver._extra_ready_checks(roller_bearing, "b2")  # não deve levantar


# =====================================================================
# ISO16281BallSolver -- point contact
# =====================================================================

class TestISO16281BallSolverRegistration:
    def test_capability_and_required_attrs(self):
        assert ISO16281BallSolver.CAPABILITY == "point_contact"
        assert set(ISO16281BallSolver.REQUIRED_ATTRS) == {
            "A", "alpha_0", "phi_j", "Ri", "cp", "Dpw", "Z",
        }

    def test_linked_to_its_multirow_sibling(self):
        assert ISO16281BallSolver.MULTIROW_SOLVER is ISO16281MultiRowBallSolverSharedDisplacement


class TestISO16281BallSolverElements:
    def test_delta_j_never_negative(self, ball_bearing):
        delta_j, alpha_j, ca, sa, cp_j, d32 = ISO16281BallSolver.elements(
            ball_bearing, delta_r=-10.0, delta_a=-10.0, Vpsi=np.zeros(ball_bearing.Z),
        )
        assert (delta_j >= 0.0).all()

    def test_shapes_match_Z(self, ball_bearing):
        out = ISO16281BallSolver.elements(ball_bearing, delta_r=0.1, delta_a=0.0,
                                          Vpsi=np.zeros(ball_bearing.Z))
        for arr in out:
            assert np.shape(arr) == (ball_bearing.Z,)


class TestISO16281BallSolverSolveContact:
    def test_pure_radial_load_converges_and_balances(self, ball_bearing):
        solver = ISO16281BallSolver()
        Fr_xz, Fr_xy, Fa = 500.0, 0.0, 0.0
        result = solver.solve_contact(ball_bearing, Fr_xz=Fr_xz, Fr_xy=Fr_xy, Fa=Fa,
                                      delta_r_init=0.0, delta_a_init=0.0, psi=0.0, phi_Fr=0.0)
        assert result.ok
        assert result.residual < 1e-3
        assert result.delta_r > 0.0
        assert (result.delta_j >= 0.0).all()

    def test_pure_axial_load_gives_positive_delta_a(self, ball_bearing):
        solver = ISO16281BallSolver()
        result = solver.solve_contact(ball_bearing, Fr_xz=0.0, Fr_xy=0.0, Fa=800.0,
                                      delta_r_init=0.0, delta_a_init=0.0, psi=0.0, phi_Fr=0.0)
        assert result.ok
        assert result.residual < 1e-3
        assert result.delta_a > 0.0

    def test_negative_axial_load_gives_negative_delta_a(self, ball_bearing):
        solver = ISO16281BallSolver()
        result = solver.solve_contact(ball_bearing, Fr_xz=0.0, Fr_xy=0.0, Fa=-800.0,
                                      delta_r_init=0.0, delta_a_init=0.0, psi=0.0, phi_Fr=0.0)
        assert result.ok
        assert result.delta_a < 0.0

    def test_zero_load_gives_near_zero_residual(self, ball_bearing):
        """A carga nula não tem raiz única em (delta_r, delta_a) --
        qualquer folga sem contacto (delta_j=0 em todo o lado) dá
        resíduo zero, por isso não se fixa um valor exato, só que o
        solver converge com resíduo desprezável."""
        solver = ISO16281BallSolver()
        result = solver.solve_contact(ball_bearing, Fr_xz=0.0, Fr_xy=0.0, Fa=0.0,
                                      delta_r_init=0.0, delta_a_init=0.0, psi=0.0, phi_Fr=0.0)
        assert result.ok
        assert result.residual < 1e-3

    def test_larger_radial_load_gives_larger_delta_r(self, ball_bearing):
        solver = ISO16281BallSolver()
        small = solver.solve_contact(ball_bearing, Fr_xz=200.0, Fr_xy=0.0, Fa=0.0,
                                     delta_r_init=0.0, delta_a_init=0.0, psi=0.0, phi_Fr=0.0)
        large = solver.solve_contact(ball_bearing, Fr_xz=800.0, Fr_xy=0.0, Fa=0.0,
                                     delta_r_init=0.0, delta_a_init=0.0, psi=0.0, phi_Fr=0.0)
        assert large.delta_r > small.delta_r


class TestISO16281BallSolverMinimumAxialLoad:
    def test_trivial_branch_when_already_engaged(self, ball_bearing):
        """Com alpha_0=0 e psi=0, delta_a=0 já satisfaz Fa=0 (V_j=0 em
        todo o lado, logo sin(alpha_j)=0) -- já está 'engaged' sem
        pré-carga nenhuma, então minimum_axial_load devolve 0.0
        diretamente, sem chamar brentq."""
        solver = ISO16281BallSolver()
        ball_bearing.alpha_0 = 0.0
        Fa_min, result = solver.minimum_axial_load(ball_bearing, Fr_xz=100.0, Fr_xy=0.0, psi=0.0)
        assert Fa_min == 0.0
        assert result.delta_a == pytest.approx(0.0, abs=1e-9)

    def test_finds_minimum_axial_load_that_closes_the_clearance(self, ball_bearing):
        """Com alpha_0=15deg a folga axial não está fechada a Fa=0 --
        minimum_axial_load tem de encontrar, via brentq, o Fa que traz
        delta_a para a fronteira de fecho (delta_a ~ 0)."""
        solver = ISO16281BallSolver()
        Fa_min, result = solver.minimum_axial_load(ball_bearing, Fr_xz=100.0, Fr_xy=0.0, psi=0.0)
        assert Fa_min > 0.0
        assert result.delta_a == pytest.approx(0.0, abs=1e-4)

    def test_raises_when_bracket_too_narrow(self, ball_bearing):
        """Forçar um alpha_0 muito raso para exigir mais pré-carga do
        que o bracket por omissão cobre."""
        solver = ISO16281BallSolver()
        ball_bearing.A = 5.0  # gap gigante -- precisa de muito mais Fa do que o bracket default
        with pytest.raises(ValueError, match="widen Fa_bracket"):
            solver.minimum_axial_load(ball_bearing, Fr_xz=100.0, Fr_xy=0.0, psi=0.0,
                                      Fa_bracket=(0.0, 1.0))


# =====================================================================
# ISO16281RollerSolver -- line contact
# =====================================================================

class TestISO16281RollerSolverRegistration:
    def test_capability_and_required_attrs(self):
        assert ISO16281RollerSolver.CAPABILITY == "line_contact"
        assert set(ISO16281RollerSolver.REQUIRED_ATTRS) == {
            "Z", "Dwe", "Lwe", "Dpw", "phi_j", "s", "n_s",
            "x_k", "cL", "cs", "alpha_0", "P_xk",
        }

    def test_linked_to_its_multirow_sibling(self):
        assert ISO16281RollerSolver.MULTIROW_SOLVER is ISO16281MultiRowRollerSolverSharedDisplacement


class TestISO16281RollerSolverElements:
    def test_delta_jk_never_negative(self, roller_bearing):
        _, _, delta_jk, q_jk, _ = ISO16281RollerSolver.elements(roller_bearing, delta_r=0.0, psi=0.0)
        assert (delta_jk >= 0.0).all()
        assert (q_jk >= 0.0).all()

    def test_shapes(self, roller_bearing):
        delta_j, psi_j, delta_jk, q_jk, cp_j = ISO16281RollerSolver.elements(
            roller_bearing, delta_r=0.05, psi=0.0)
        Z, n_s = roller_bearing.Z, roller_bearing.n_s
        assert delta_j.shape == (Z,)
        assert psi_j.shape == (Z,)
        assert delta_jk.shape == (Z, n_s)
        assert q_jk.shape == (Z, n_s)
        assert cp_j.shape == (Z,)


class TestISO16281RollerSolverSolveContact:
    def test_pure_radial_load_converges_and_balances(self, roller_bearing):
        solver = ISO16281RollerSolver()
        result = solver.solve_contact(roller_bearing, Fr_xz=1000.0, Fr_xy=0.0,
                                      delta_r_init=0.0, psi=0.0, phi_Fr=0.0)
        assert result.ok
        assert result.residual < 1e-2
        assert result.delta_r > 0.0
        assert result.delta_a == 0.0  # roller radial não tem capacidade axial

    def test_zero_load_gives_near_zero_residual(self, roller_bearing):
        """Tal como no ball solver, carga nula não tem raiz única em
        delta_r -- qualquer delta_r que deixe todos os rolos fora de
        contacto (delta_jk=0 em todo o lado) dá resíduo zero. Só se
        verifica que o solver converge com resíduo desprezável."""
        solver = ISO16281RollerSolver()
        result = solver.solve_contact(roller_bearing, Fr_xz=0.0, Fr_xy=0.0,
                                      delta_r_init=0.0, psi=0.0, phi_Fr=0.0)
        assert result.ok
        assert result.residual < 1e-3

    def test_larger_radial_load_gives_larger_delta_r(self, roller_bearing):
        solver = ISO16281RollerSolver()
        small = solver.solve_contact(roller_bearing, Fr_xz=500.0, Fr_xy=0.0,
                                     delta_r_init=0.0, psi=0.0, phi_Fr=0.0)
        large = solver.solve_contact(roller_bearing, Fr_xz=2000.0, Fr_xy=0.0,
                                     delta_r_init=0.0, psi=0.0, phi_Fr=0.0)
        assert large.delta_r > small.delta_r

    def test_alpha_j_is_constant_equal_to_bearing_alpha_0(self, roller_bearing):
        solver = ISO16281RollerSolver()
        result = solver.solve_contact(roller_bearing, Fr_xz=500.0, Fr_xy=0.0,
                                      delta_r_init=0.0, psi=0.0, phi_Fr=0.0)
        assert np.all(result.alpha_j == roller_bearing.alpha_0)


# =====================================================================
# MultiRowSolverBase -- shared-displacement multi-row (não revisto para
# acoplamento fila-a-fila -- ver docstring do próprio módulo)
# =====================================================================

class TestMultiRowSolverBaseRowViews:
    def test_rejects_fewer_than_two_rows(self):
        bearing = SimpleNamespace(label="b1", rows=[{"cp": 1.0}])
        with pytest.raises(ValueError, match=">= 2 rows"):
            MultiRowSolverBase._row_views(bearing)

    def test_rejects_missing_rows_attribute(self):
        bearing = SimpleNamespace(label="b1")
        with pytest.raises(ValueError, match=">= 2 rows"):
            MultiRowSolverBase._row_views(bearing)

    def test_wraps_dict_rows_as_simplenamespace(self):
        bearing = SimpleNamespace(label="b1", rows=[{"cp": 1.0}, {"cp": 2.0}])
        views = MultiRowSolverBase._row_views(bearing)
        assert all(isinstance(v, SimpleNamespace) for v in views)
        assert [v.cp for v in views] == [1.0, 2.0]

    def test_leaves_object_rows_untouched(self):
        row_obj = SimpleNamespace(cp=3.0)
        bearing = SimpleNamespace(label="b1", rows=[row_obj, SimpleNamespace(cp=4.0)])
        views = MultiRowSolverBase._row_views(bearing)
        assert views[0] is row_obj


class TestMultiRowSolverBaseFractions:
    def test_proportional_split(self):
        assert MultiRowSolverBase._fractions([1.0, 3.0], 4.0) == pytest.approx([0.25, 0.75])

    def test_equal_split_when_total_near_zero(self):
        fractions = MultiRowSolverBase._fractions([5.0, -5.0, 2.0], 0.0)
        assert fractions == pytest.approx([1.0 / 3.0] * 3)


class TestISO16281MultiRowBallSolver:
    def test_check_row_ready_rejects_non_positive_cp(self):
        solver = ISO16281MultiRowBallSolverSharedDisplacement()
        row = SimpleNamespace(cp=0.0)
        with pytest.raises(ValueError, match="cp must be positive"):
            solver._check_row_ready(row, "b1[row0]")

    def test_check_row_ready_accepts_positive_cp(self):
        solver = ISO16281MultiRowBallSolverSharedDisplacement()
        solver._check_row_ready(SimpleNamespace(cp=1.0), "b1[row0]")  # não deve levantar

    def test_two_identical_rows_each_take_half_the_load(self, ball_bearing):
        solver = ISO16281MultiRowBallSolverSharedDisplacement()
        row_views = [SimpleNamespace(**BALL_BEARING_KWARGS), SimpleNamespace(**BALL_BEARING_KWARGS)]
        row_results, row_reactions, nfev, res, ok = solver._solve_rows(
            row_views, Fr_xz=1000.0, Fr_xy=0.0, Fa=0.0, psi=0.0, phi_Fr=0.0,
        )
        assert ok
        assert len(row_results) == len(row_reactions) == 2
        Fr_row0, _ = row_reactions[0]
        Fr_row1, _ = row_reactions[1]
        assert Fr_row0 == pytest.approx(Fr_row1, rel=1e-6)


class TestISO16281MultiRowRollerSolver:
    def test_check_row_ready_rejects_too_few_laminae(self):
        solver = ISO16281MultiRowRollerSolverSharedDisplacement()
        row = SimpleNamespace(n_s=10, x_k=np.zeros(10))
        with pytest.raises(ValueError, match="Sec 5.2.2"):
            solver._check_row_ready(row, "b1[row0]")

    def test_check_row_ready_rejects_x_k_length_mismatch(self):
        solver = ISO16281MultiRowRollerSolverSharedDisplacement()
        row = SimpleNamespace(n_s=30, x_k=np.zeros(29))
        with pytest.raises(ValueError, match="len\\(x_k\\)"):
            solver._check_row_ready(row, "b1[row0]")

    def test_two_identical_rows_each_take_half_the_load(self):
        solver = ISO16281MultiRowRollerSolverSharedDisplacement()
        row_views = [SimpleNamespace(**ROLLER_BEARING_KWARGS), SimpleNamespace(**ROLLER_BEARING_KWARGS)]
        row_results, row_reactions, nfev, res, ok = solver._solve_rows(
            row_views, Fr_xz=1000.0, Fr_xy=0.0, Fa=0.0, psi=0.0, phi_Fr=0.0,
        )
        assert ok
        assert len(row_results) == len(row_reactions) == 2
        Fr_row0, Fa_row0 = row_reactions[0]
        Fr_row1, Fa_row1 = row_reactions[1]
        assert Fr_row0 == pytest.approx(Fr_row1, rel=1e-6)
        assert Fa_row0 == 0.0 and Fa_row1 == 0.0
