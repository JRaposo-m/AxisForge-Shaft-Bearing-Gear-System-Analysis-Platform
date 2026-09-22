# test_global_postprocessing.py
"""Testes de global_solver/global_postprocessing.py.

build_shaft_result() chama TorsionSolver().solve() como primeiro passo
-- e isso está bloqueado hoje pelo mesmo gap documentado em
test_torsion.py (Shaft.J_at() em falta). Para poder testar o RESTO da
sequência (dispatch de teoria, recuperação de M/V, deflexões,
propriedades de secção, reações e dados de chumaceira -- tudo o que já
está implementado e não depende de J_at), a classe TorsionSolver é
substituída aqui por um duplo que devolve zeros, via monkeypatch do
nome importado dentro do próprio módulo global_postprocessing. Isto é
isolado do gap de J_at de propósito -- ver test_torsion.py para o teste
que cobre esse gap diretamente, sem o contornar."""
import math
import numpy as np
import pytest

from axisforge.solvers.machine_elements.shaft.fem_solvers.global_solver import (
    global_postprocessing as gp_module,
)
from axisforge.solvers.machine_elements.shaft.fem_solvers.global_solver.global_postprocessing import (
    GlobalEulerBernoulliPostProcessing, GlobalTimoshenkoPostProcessing,
    postprocessor_for_global, build_shaft_result,
)
from axisforge.solvers.machine_elements.shaft.fem_solvers.global_solver.rigid_support import (
    RigidSupportSolution,
)
from axisforge.results.fem_results.shaft_results import ShaftResults


class _ZeroTorsionSolver:
    """Duplo de TorsionSolver -- devolve zeros, só para destravar o
    resto de build_shaft_result() enquanto Shaft.J_at() não existir
    (ver docstring do módulo)."""
    def solve(self, shaft_system, x_nodes):
        n = len(x_nodes)
        return np.zeros(n), np.zeros(n), np.zeros(n), [[] for _ in range(n)]


@pytest.fixture(autouse=True)
def _bypass_torsion(monkeypatch):
    monkeypatch.setattr(gp_module, "TorsionSolver", _ZeroTorsionSolver)


def _make_rigid_solution(elements, x_nodes, *, kGA_override=None):
    n = len(x_nodes)
    n_dofs = 3 * n
    rng = np.random.default_rng(3)
    K = np.eye(n_dofs) * 1000.0
    free_dofs = list(range(2, n_dofs))
    constrained_dofs = [0, 1]
    d_xz = rng.normal(scale=0.01, size=n_dofs)
    d_xy = rng.normal(scale=0.01, size=n_dofs)
    return RigidSupportSolution(
        x_nodes=x_nodes, elements=elements, K=K,
        free_dofs=free_dofs, constrained_dofs=constrained_dofs,
        d_total_xz=d_xz, d_total_xy=d_xy,
        f_xz_ext=np.zeros(n_dofs), f_xy_ext=np.zeros(n_dofs),
        f_xz_total=rng.normal(size=n_dofs), f_xy_total=rng.normal(size=n_dofs),
        f_xz_reaction=rng.normal(size=n_dofs), f_xy_reaction=rng.normal(size=n_dofs),
        _kGA_override=kGA_override,
    )


class TestPostprocessorForGlobal:
    def test_dispatches_euler(self, euler_elements, x_nodes_2elem):
        solution = _make_rigid_solution(euler_elements, x_nodes_2elem)
        postproc = postprocessor_for_global(solution)
        assert isinstance(postproc, GlobalEulerBernoulliPostProcessing)

    def test_dispatches_timoshenko(self, timoshenko_elements, x_nodes_2elem):
        solution = _make_rigid_solution(timoshenko_elements, x_nodes_2elem)
        postproc = postprocessor_for_global(solution)
        assert isinstance(postproc, GlobalTimoshenkoPostProcessing)

    def test_unknown_theory_raises(self, euler_elements, x_nodes_2elem):
        euler_elements[0].beam_theory = "something_else"
        solution = _make_rigid_solution(euler_elements, x_nodes_2elem)
        with pytest.raises(ValueError, match="unknown beam_theory"):
            postprocessor_for_global(solution)

    def test_no_view_adapter_needed(self, euler_elements, x_nodes_2elem):
        """Ao contrário do lado submodel, RigidSupportSolution já tem
        os nomes certos -- recover_internal_forces deve aceitar a
        solution diretamente, sem wrapper."""
        solution = _make_rigid_solution(euler_elements, x_nodes_2elem)
        postproc = postprocessor_for_global(solution)
        M_xz, M_xy, V_xz, V_xy = postproc.recover_internal_forces(solution, x_nodes_2elem, len(x_nodes_2elem))
        assert M_xz.shape == (len(x_nodes_2elem),)


class TestBuildShaftResultEndToEnd:
    @pytest.mark.parametrize("elements_fixture", ["euler_elements", "timoshenko_elements"])
    def test_full_sequence_returns_correctly_shaped_result(
        self, elements_fixture, request, x_nodes_2elem, fake_shaft_system_2brg,
    ):
        elements = request.getfixturevalue(elements_fixture)
        solution = _make_rigid_solution(elements, x_nodes_2elem)

        result = build_shaft_result(solution, fake_shaft_system_2brg)

        assert isinstance(result, ShaftResults)
        n = len(x_nodes_2elem)
        for field_name in ("M_xz", "M_xy", "M", "V_xz", "V_xy", "V",
                           "v_xz", "v_xy", "v", "u", "theta_xz", "theta_xy",
                           "d", "W", "Wt", "sigma_b", "tau"):
            arr = getattr(result, field_name)
            assert len(arr) == n, f"{field_name} length mismatch"

        assert len(result.bearing_nodes) == len(fake_shaft_system_2brg.bearings)
        assert len(result.R) == len(fake_shaft_system_2brg.bearings)

    def test_bearing_reactions_only_locating_bearing_gets_axial(
        self, euler_elements, x_nodes_2elem, fake_shaft_system_2brg,
    ):
        solution = _make_rigid_solution(euler_elements, x_nodes_2elem)
        result = build_shaft_result(solution, fake_shaft_system_2brg)

        # fake_shaft_system_2brg: brg-A é "locating", brg-B é "floating"
        labels = [n.label for n in result.bearing_nodes]
        idx_a = labels.index("brg-A")
        idx_b = labels.index("brg-B")

        node_a = result.bearing_nodes[idx_a]
        node_b = result.bearing_nodes[idx_b]
        i_a = x_nodes_2elem.index(fake_shaft_system_2brg.bearings[0].position)

        # floating (brg-B) nunca reporta axial -- estrutural, não depende
        # dos valores aleatórios de d_xz/f_xz_total
        assert node_b.Fa == 0.0
        # locating (brg-A) lê Fa diretamente de f_xz_total no seu nó --
        # verificado contra a mesma fonte, não um valor à parte
        assert math.isclose(node_a.Fa, float(solution.f_xz_total[3 * i_a]))

    def test_M_and_V_equivalent_are_hypot_of_planes(self, euler_elements, x_nodes_2elem, fake_shaft_system_2brg):
        solution = _make_rigid_solution(euler_elements, x_nodes_2elem)
        result = build_shaft_result(solution, fake_shaft_system_2brg)
        assert np.allclose(result.M, np.hypot(result.M_xz, result.M_xy))
        assert np.allclose(result.V, np.hypot(result.V_xz, result.V_xy))

    def test_max_indices_point_to_actual_max_values(self, euler_elements, x_nodes_2elem, fake_shaft_system_2brg):
        solution = _make_rigid_solution(euler_elements, x_nodes_2elem)
        result = build_shaft_result(solution, fake_shaft_system_2brg)
        assert math.isclose(result.M_max, float(np.max(result.M)))
        assert math.isclose(result.v_max, float(np.max(np.hypot(result.v_xz, result.v_xy))))
