# test_submodel_postprocessing.py
"""Testes de submodel_solver/submodel_postprocessing.py. Ao contrário do
lado global (que precisa de TorsionSolver -> shaft.J_at, ainda por
implementar), este caminho não depende de ShaftSystem nenhum -- dá para
montar um SubmodelSolution real (a dataclass verdadeira, importada do
módulo real) só com FakeElem, e correr build_submodel_result() de
ponta a ponta tal como SubmodelSolver.solve() o chama."""
import math
import numpy as np
import pytest

from axisforge.solvers.machine_elements.shaft.fem_solvers.submodel_solver.lagrange_multipliers import (
    SubmodelSolution,
)
from axisforge.solvers.machine_elements.shaft.fem_solvers.submodel_solver.submodel_postprocessing import (
    SubmodelEulerBernoulliPostProcessing, SubmodelTimoshenkoPostProcessing,
    postprocessor_for_submodel, build_submodel_result, _SubmodelSolverView,
)
from axisforge.results.fem_results.submodel_results import SubmodelResult


def _make_solution(elements, x_nodes, *, kGA_override=None):
    n = len(x_nodes)
    rng = np.random.default_rng(2)
    return SubmodelSolution(
        x_lo=x_nodes[0], x_hi=x_nodes[-1], grade="grade_0",
        x_nodes=x_nodes,
        d_xz=rng.normal(scale=0.01, size=3 * n),
        d_xy=rng.normal(scale=0.01, size=3 * n),
        lam_xz=rng.normal(size=6),
        lam_xy=rng.normal(size=6),
        elements=elements,
        kGA_override=kGA_override,
    )


class TestSubmodelSolverView:
    def test_translates_field_names(self, euler_elements, x_nodes_2elem):
        solution = _make_solution(euler_elements, x_nodes_2elem, kGA_override=123.0)
        view = _SubmodelSolverView(solution)
        assert view.elements is solution.elements
        assert np.array_equal(view.d_total_xz, solution.d_xz)
        assert np.array_equal(view.d_total_xy, solution.d_xy)
        assert view._kGA_override == 123.0


class TestPostprocessorForSubmodel:
    def test_dispatches_euler(self, euler_elements, x_nodes_2elem):
        solution = _make_solution(euler_elements, x_nodes_2elem)
        postproc = postprocessor_for_submodel(solution)
        assert isinstance(postproc, SubmodelEulerBernoulliPostProcessing)

    def test_dispatches_timoshenko(self, timoshenko_elements, x_nodes_2elem):
        solution = _make_solution(timoshenko_elements, x_nodes_2elem)
        postproc = postprocessor_for_submodel(solution)
        assert isinstance(postproc, SubmodelTimoshenkoPostProcessing)

    def test_unknown_theory_raises(self, euler_elements, x_nodes_2elem):
        euler_elements[0].beam_theory = "something_else"
        solution = _make_solution(euler_elements, x_nodes_2elem)
        with pytest.raises(ValueError, match="unknown beam_theory"):
            postprocessor_for_submodel(solution)


class TestBuildSubmodelResultEndToEnd:
    @pytest.mark.parametrize("elements_fixture", ["euler_elements", "timoshenko_elements"])
    def test_full_sequence_returns_correctly_shaped_result(self, elements_fixture, request, x_nodes_2elem):
        elements = request.getfixturevalue(elements_fixture)
        solution = _make_solution(elements, x_nodes_2elem)

        result = build_submodel_result(solution)

        assert isinstance(result, SubmodelResult)
        n = len(x_nodes_2elem)
        for field_name in ("u", "v_xz", "v_xy", "theta_xz", "theta_xy", "M_xz", "M_xy", "V_xz", "V_xy"):
            arr = getattr(result, field_name)
            assert arr.shape == (n,), f"{field_name} shape mismatch"
            assert np.all(np.isfinite(arr))

        assert result.x_lo == solution.x_lo
        assert result.x_hi == solution.x_hi
        assert result.grade == solution.grade
        assert np.array_equal(result.lam_xz, solution.lam_xz)
        assert np.array_equal(result.lam_xy, solution.lam_xy)

    def test_u_is_read_from_d_xz_never_d_xy(self, euler_elements, x_nodes_2elem):
        """Documentado em build_submodel_result: u vem só de d_xz. Aqui
        d_xy tem valores muito diferentes de d_xz para o teste falhar
        claramente se essa convenção alguma vez for trocada por engano."""
        n = len(x_nodes_2elem)
        solution = SubmodelSolution(
            x_lo=x_nodes_2elem[0], x_hi=x_nodes_2elem[-1], grade="grade_0",
            x_nodes=x_nodes_2elem,
            d_xz=np.arange(3 * n, dtype=float),
            d_xy=np.arange(3 * n, dtype=float) + 1000.0,
            lam_xz=np.zeros(6), lam_xy=np.zeros(6),
            elements=euler_elements,
        )
        result = build_submodel_result(solution)
        expected_u = np.array([solution.d_xz[3 * i] for i in range(n)])
        assert np.array_equal(result.u, expected_u)
        assert not np.any(np.isin(result.u, solution.d_xy))

    def test_kGA_override_flows_through_to_timoshenko_shear(self, timoshenko_elements, x_nodes_2elem):
        solution_a = _make_solution(timoshenko_elements, x_nodes_2elem, kGA_override=None)
        solution_b = _make_solution(timoshenko_elements, x_nodes_2elem, kGA_override=1e10)
        solution_b.d_xz = solution_a.d_xz  # isola o efeito ao kGA_override
        solution_b.d_xy = solution_a.d_xy

        result_a = build_submodel_result(solution_a)
        result_b = build_submodel_result(solution_b)
        assert not np.allclose(result_a.V_xz, result_b.V_xz)
