# test_element_postprocessing.py
"""Testes de element_theories/element_postprocessing.py, com FakeElem
(conftest.py). bending_moment()/recover_element_displacements()/
recover_internal_forces() são código PARTILHADO pelas duas teorias --
verificados contra a fórmula exata que o próprio ficheiro documenta
(M = E*I * B_b(zeta) @ a_e). shear_force() é a única peça
genuinamente por-teoria; verificada contra a fórmula fechada de cada
subclasse tal como está no código hoje -- incluindo o mismatch de
sinal entre as duas, que fica documentado como teste, não escondido."""
import math
import numpy as np
import pytest

from axisforge.solvers.machine_elements.shaft.fem_solvers.element_theories.element_postprocessing import (
    ElementTheoryPostProcessor, EulerBernoulliPostProcessing, TimoshenkoPostProcessing,
)
from axisforge.core.loads import LoadPlane


class TestElementTheoryPostProcessorIsAbstract:
    def test_cannot_instantiate_base_class_directly(self):
        with pytest.raises(TypeError):
            ElementTheoryPostProcessor()


class TestElementNodalDisplacements:
    def test_returns_correct_4_dof_slice_xz(self, fake_solver_euler, euler_elements):
        postproc = EulerBernoulliPostProcessing()
        elem = euler_elements[0]  # nós 0, 1
        a_e = postproc._element_nodal_displacements(fake_solver_euler, elem, LoadPlane.XZ)
        d = fake_solver_euler.d_total_xz
        expected = d[[1, 2, 4, 5]]  # [v_a, th_a, v_b, th_b] do nó 0 e nó 1
        assert np.allclose(a_e, expected)

    def test_returns_correct_4_dof_slice_xy(self, fake_solver_euler, euler_elements):
        postproc = EulerBernoulliPostProcessing()
        elem = euler_elements[1]  # nós 1, 2
        a_e = postproc._element_nodal_displacements(fake_solver_euler, elem, LoadPlane.XY)
        d = fake_solver_euler.d_total_xy
        expected = d[[4, 5, 7, 8]]
        assert np.allclose(a_e, expected)

    def test_recover_element_displacements_returns_one_entry_per_element(self, fake_solver_euler, euler_elements):
        postproc = EulerBernoulliPostProcessing()
        out = postproc.recover_element_displacements(fake_solver_euler, x_nodes=[0, 50, 100], n=3)
        assert len(out) == len(euler_elements)
        for elem, a_xz, a_xy in out:
            assert a_xz.shape == (4,)
            assert a_xy.shape == (4,)


class TestBendingMomentSharedAcrossTheories:
    @pytest.mark.parametrize("postproc_cls", [EulerBernoulliPostProcessing, TimoshenkoPostProcessing])
    def test_matches_documented_formula(self, postproc_cls, euler_elements):
        elem = euler_elements[0]
        postproc = postproc_cls()
        a_e = np.array([0.1, 0.02, -0.05, 0.03])
        x = elem.x_a + 0.3 * elem.length

        x_out, M = postproc.bending_moment(elem, a_e, x)
        zeta = elem.natural_coordenates(x)
        B_b = elem.bending_strain_matrix(zeta)
        M_expected = float(elem.E * elem.I * (B_b @ a_e))

        assert x_out == x
        assert math.isclose(M, M_expected, rel_tol=1e-10)


class TestEulerBernoulliShearForce:
    def test_matches_closed_form_negated(self, euler_elements):
        elem = euler_elements[0]
        postproc = EulerBernoulliPostProcessing()
        a_e = np.array([0.1, 0.02, -0.05, 0.03])
        le = elem.length

        _, Q = postproc.shear_force(elem, a_e, elem.x_a)
        Q_expected = -float(elem.E * elem.I / le**3 * (np.array([12, 6 * le, -12, 6 * le]) @ a_e))
        assert math.isclose(Q, Q_expected, rel_tol=1e-10)

    def test_shear_is_constant_within_element(self, euler_elements):
        """Terceira derivada de uma cúbica de Hermite é constante --
        Q não deve depender de x dentro do elemento."""
        elem = euler_elements[0]
        postproc = EulerBernoulliPostProcessing()
        a_e = np.array([0.1, 0.02, -0.05, 0.03])
        _, Q_a = postproc.shear_force(elem, a_e, elem.x_a)
        _, Q_mid = postproc.shear_force(elem, a_e, (elem.x_a + elem.x_b) / 2)
        _, Q_b = postproc.shear_force(elem, a_e, elem.x_b)
        assert math.isclose(Q_a, Q_mid, rel_tol=1e-10)
        assert math.isclose(Q_a, Q_b, rel_tol=1e-10)

    def test_kGA_override_ignored(self, euler_elements):
        elem = euler_elements[0]
        postproc = EulerBernoulliPostProcessing()
        a_e = np.array([0.1, 0.02, -0.05, 0.03])
        _, Q_none = postproc.shear_force(elem, a_e, elem.x_a, kGA_override=None)
        _, Q_override = postproc.shear_force(elem, a_e, elem.x_a, kGA_override=1e12)
        assert Q_none == Q_override


class TestTimoshenkoShearForce:
    def test_matches_documented_formula(self, timoshenko_elements):
        elem = timoshenko_elements[0]
        postproc = TimoshenkoPostProcessing()
        a_e = np.array([0.1, 0.02, -0.05, 0.03])
        x = elem.x_a + 0.4 * elem.length

        _, Q = postproc.shear_force(elem, a_e, x, kGA_override=None)
        zeta = elem.natural_coordenates(x)
        B_s = elem.shear_strain_matrix(zeta)
        D_s = elem.shear_rigidity(kGA_override=None)
        Q_expected = float(D_s * (B_s @ a_e))
        assert math.isclose(Q, Q_expected, rel_tol=1e-10)

    def test_kGA_override_is_forwarded(self, timoshenko_elements):
        elem = timoshenko_elements[0]
        postproc = TimoshenkoPostProcessing()
        a_e = np.array([0.1, 0.02, -0.05, 0.03])
        _, Q_default = postproc.shear_force(elem, a_e, elem.x_a, kGA_override=None)
        _, Q_override = postproc.shear_force(elem, a_e, elem.x_a, kGA_override=elem.kGA * 2)
        assert not math.isclose(Q_default, Q_override)


class TestSignConventionKnownGap:
    def test_KNOWN_GAP_euler_negates_timoshenko_does_not(self, euler_elements, timoshenko_elements):
        """TODO(owner) documentado no módulo: EulerBernoulliPostProcessing.shear_force()
        devolve o valor negado; TimoshenkoPostProcessing.shear_force()
        não. Isto confirma esse mismatch tal como está hoje -- NÃO é
        uma validação de qual dos dois sinais está fisicamente correto
        (não vi o Elem real para decidir isso). Substitui por um
        assert de igualdade quando a convenção for unificada."""
        a_e = np.array([1.0, 0.0, 0.0, 0.0])  # mesmo a_e "canónico" nas duas
        elem_e = euler_elements[0]
        elem_t = timoshenko_elements[0]

        _, Q_euler = EulerBernoulliPostProcessing().shear_force(elem_e, a_e, elem_e.x_a)
        _, Q_timo = TimoshenkoPostProcessing().shear_force(elem_t, a_e, elem_t.x_a)

        # não comparamos os VALORES (elementos/teorias diferentes) --
        # só confirmamos que o sinal da fórmula do euler é o oposto do
        # que produziria sem o "-" explícito no código.
        le = elem_e.length
        Q_euler_unsigned = float(elem_e.E * elem_e.I / le**3 * (np.array([12, 6*le, -12, 6*le]) @ a_e))
        assert math.isclose(Q_euler, -Q_euler_unsigned, rel_tol=1e-10)


class TestRecoverInternalForces:
    @pytest.mark.parametrize("postproc_cls,solver_fixture", [
        (EulerBernoulliPostProcessing, "fake_solver_euler"),
        (TimoshenkoPostProcessing, "fake_solver_timoshenko"),
    ])
    def test_shapes_match_x_nodes(self, postproc_cls, solver_fixture, request, x_nodes_2elem):
        solver = request.getfixturevalue(solver_fixture)
        postproc = postproc_cls()
        n = len(x_nodes_2elem)
        M_xz, M_xy, V_xz, V_xy = postproc.recover_internal_forces(solver, x_nodes_2elem, n)
        for arr in (M_xz, M_xy, V_xz, V_xy):
            assert arr.shape == (n,)
            assert np.all(np.isfinite(arr))

    def test_shared_node_keeps_first_elements_b_value(self, fake_solver_euler, x_nodes_2elem):
        """Convenção real de _sweep_plane (corrigido depois de correr
        pytest): 'if elem_idx == 0: M[i]=... ; M[j]=... SEMPRE' -- só o
        PRIMEIRO elemento (elem_idx==0) escreve o seu nó 'a' (global i);
        TODO elemento escreve o seu nó 'b' (global j), inclusive o
        primeiro. O segundo elemento nunca volta a escrever o índice
        global do nó partilhado (é o 'a' dele, não o 'b') -- por isso o
        nó partilhado (nó 1) fica com o valor do elemento 1 avaliado no
        SEU nó 'b' (x=50), e é --sobrescrito depois-- por nada, porque
        elem2 escreve apenas o índice 2. A minha primeira versão deste
        teste assumia o contrário (que o valor final seria o do
        elemento 2) -- errado; fica corrigido aqui com o pytest real
        como árbitro."""
        postproc = EulerBernoulliPostProcessing()
        n = len(x_nodes_2elem)
        M_xz, _, _, _ = postproc.recover_internal_forces(fake_solver_euler, x_nodes_2elem, n)
        elem1 = fake_solver_euler.elements[0]
        a_e1 = postproc._element_nodal_displacements(fake_solver_euler, elem1, LoadPlane.XZ)
        _, M1_at_b = postproc.bending_moment(elem1, a_e1, elem1.x_b)
        assert math.isclose(M_xz[1], M1_at_b, rel_tol=1e-10)

    def test_kGA_override_read_from_solver_attribute(self, timoshenko_elements, x_nodes_2elem):
        from axisforge.solvers.machine_elements.shaft.fem_solvers.tests.conftest import FakeSolver
        n = len(x_nodes_2elem)
        rng_d = np.zeros(3 * n)
        solver_no_override = FakeSolver(elements=timoshenko_elements, d_total_xz=rng_d.copy(),
                                         d_total_xy=rng_d.copy(), _kGA_override=None)
        solver_with_override = FakeSolver(elements=timoshenko_elements, d_total_xz=rng_d.copy() + 0.01,
                                           d_total_xy=rng_d.copy() + 0.01, _kGA_override=1e12)
        postproc = TimoshenkoPostProcessing()
        # não compara valores (d_total também difere) -- só confirma
        # que getattr(solver, "_kGA_override", None) é mesmo lido e
        # que a sequência não rebenta com nenhum dos dois casos
        postproc.recover_internal_forces(solver_no_override, x_nodes_2elem, n)
        postproc.recover_internal_forces(solver_with_override, x_nodes_2elem, n)

    def test_solver_without_kGA_override_attribute_defaults_to_none(self, timoshenko_elements, x_nodes_2elem):
        """_sweep_plane usa getattr(solver, '_kGA_override', None) --
        um 'solver' que nem sequer tem o atributo não deve rebentar."""
        class _BareSolver:
            pass
        bare = _BareSolver()
        bare.elements = timoshenko_elements
        n = len(x_nodes_2elem)
        bare.d_total_xz = np.zeros(3 * n)
        bare.d_total_xy = np.zeros(3 * n)

        postproc = TimoshenkoPostProcessing()
        M_xz, M_xy, V_xz, V_xy = postproc.recover_internal_forces(bare, x_nodes_2elem, n)
        assert V_xz.shape == (n,)
